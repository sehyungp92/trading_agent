from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backtests.shared.parity import generate_calibration_report as generator
from backtests.shared.parity.calibration_report import (
    CalibrationReportError,
    REQUIRED_CALIBRATION_FAMILIES,
    REQUIRED_CALIBRATION_STRATEGY_IDS,
    load_calibration_report,
    validate_calibration_report,
)
from libs.config.loader import load_strategy_registry


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
VALID_TOLERANCES = {
    "backtest_source_present": True,
    "slippage_mean_abs_diff": True,
    "commission_abs_diff": True,
    "partial_fill_rate_abs_diff": True,
    "ioc_reject_rate_abs_diff": True,
}


def test_valid_broker_backed_calibration_report_passes(tmp_path) -> None:
    path = tmp_path / "parity_calibration_2026-05-20.json"
    path.write_text(json.dumps(_valid_report()), encoding="utf-8")

    assert load_calibration_report(path)["within_tolerance"] is True


def test_required_calibration_roster_matches_enabled_non_scalp_runtime_config() -> None:
    registry = load_strategy_registry(CONFIG_DIR)
    enabled = [
        manifest
        for manifest in registry.enabled_strategies(live=True)
        if manifest.family != "scalp"
    ]

    by_family = {
        family: {
            manifest.strategy_id
            for manifest in enabled
            if manifest.family == family
        }
        for family in REQUIRED_CALIBRATION_FAMILIES
    }

    assert set(REQUIRED_CALIBRATION_STRATEGY_IDS) == {
        manifest.strategy_id for manifest in enabled
    }
    assert by_family == {
        family: set(strategy_ids)
        for family, strategy_ids in REQUIRED_CALIBRATION_FAMILIES.items()
    }


def test_markdown_string_report_is_rejected(tmp_path) -> None:
    path = tmp_path / "parity_calibration_2026-05-20.json"
    path.write_text("Calibration status: PASS\nwithin_tolerance: true\n", encoding="utf-8")

    with pytest.raises(CalibrationReportError, match="not valid JSON"):
        load_calibration_report(path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"report_type": "fixture_bootstrap"}, "broker_backed"),
        ({"source_tables": ["fills", "order_events", "trades"]}, "orders"),
        ({"broker_fill_sample_count": 0}, "broker_fill_sample_count"),
        ({"within_tolerance": False}, "within_tolerance"),
        ({"strategy_results": []}, "strategy_results"),
    ],
)
def test_invalid_calibration_report_is_rejected(updates: dict, message: str) -> None:
    payload = _valid_report()
    payload.update(updates)

    with pytest.raises(CalibrationReportError, match=message):
        validate_calibration_report(payload)


def test_calibration_report_rejects_missing_expected_strategy_coverage() -> None:
    payload = _valid_report()
    payload["expected_strategy_ids"] = ["TPC"]

    with pytest.raises(CalibrationReportError, match="expected_strategy_ids missing required ids"):
        validate_calibration_report(payload)


def test_calibration_report_rejects_unknown_expected_strategy_id() -> None:
    payload = _valid_report()
    payload["expected_strategy_ids"] = [*REQUIRED_CALIBRATION_STRATEGY_IDS, "EXPERIMENTAL_STRATEGY"]

    with pytest.raises(CalibrationReportError, match="expected_strategy_ids has unknown ids"):
        validate_calibration_report(payload)


def test_calibration_report_rejects_missing_expected_family_coverage() -> None:
    payload = _valid_report()
    payload["expected_family_ids"] = ["swing"]

    with pytest.raises(CalibrationReportError, match="expected_family_ids missing required ids"):
        validate_calibration_report(payload)


def test_calibration_report_rejects_missing_strategy_result() -> None:
    payload = _valid_report()
    payload["strategy_results"] = payload["strategy_results"][:-1]

    with pytest.raises(CalibrationReportError, match="missing strategy results"):
        validate_calibration_report(payload)


def test_calibration_report_rejects_unknown_strategy_result() -> None:
    payload = _valid_report()
    payload["strategy_results"].append(_valid_strategy_result("EXPERIMENTAL_STRATEGY"))

    with pytest.raises(CalibrationReportError, match="unknown strategy results"):
        validate_calibration_report(payload)


def test_calibration_report_rejects_duplicate_strategy_result() -> None:
    payload = _valid_report()
    payload["strategy_results"].append(dict(payload["strategy_results"][0]))

    with pytest.raises(CalibrationReportError, match="duplicate strategy results"):
        validate_calibration_report(payload)


@pytest.mark.parametrize(
    ("result_updates", "message"),
    [
        ({"backtest_fill_sample_count": 0}, "backtest_fill_sample_count"),
        ({"tolerance_results": {"placeholder_ok": True}}, "missing tolerance keys"),
        (
            {"tolerance_results": {**VALID_TOLERANCES, "placeholder_ok": True}},
            "unknown tolerance keys",
        ),
        (
            {"tolerance_results": {**VALID_TOLERANCES, "backtest_source_present": False}},
            "backtest_source_present",
        ),
        ({"broker_metrics": {"broker_fill_sample_count": 12}}, "broker_metrics missing metric keys"),
        ({"backtest_metrics": {"broker_fill_sample_count": 12}}, "backtest_metrics missing metric keys"),
    ],
)
def test_invalid_strategy_calibration_result_is_rejected(result_updates: dict, message: str) -> None:
    payload = _valid_report()
    payload["strategy_results"][0].update(result_updates)

    with pytest.raises(CalibrationReportError, match=message):
        validate_calibration_report(payload)


def test_stale_calibration_report_is_rejected_by_default() -> None:
    payload = _valid_report()
    payload["window"]["end"] = "2026-01-01T00:00:00+00:00"

    with pytest.raises(CalibrationReportError, match="stale"):
        validate_calibration_report(payload, now=datetime(2026, 5, 20, tzinfo=timezone.utc))


def test_stale_calibration_report_can_skip_freshness_for_archival_checks() -> None:
    payload = _valid_report()
    payload["window"]["end"] = "2026-01-01T00:00:00+00:00"

    assert validate_calibration_report(payload, max_age_days=None)["within_tolerance"] is True


def test_future_dated_calibration_report_is_rejected_even_without_stale_check() -> None:
    payload = _valid_report()
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    payload["window"]["end"] = (now + timedelta(hours=1)).isoformat()

    with pytest.raises(CalibrationReportError, match="future"):
        validate_calibration_report(payload, max_age_days=None, now=now)


def test_generator_report_with_matching_replay_fills_validates() -> None:
    now = datetime.now(timezone.utc)
    rows = _metric_rows_for_required_strategies(now)
    trade_counts = {strategy_id: 1 for strategy_id in REQUIRED_CALIBRATION_STRATEGY_IDS}

    report = generator._build_report(
        broker_rows=rows,
        backtest_rows=rows,
        trade_counts=trade_counts,
        start=now - timedelta(days=1),
        end=now,
        days=1,
        has_backtest_source=True,
    )

    assert report["within_tolerance"] is True
    assert set(report["source_tables"]) == {"orders", "trades", "fills", "order_events"}
    assert validate_calibration_report(report)["within_tolerance"] is True


def test_generator_report_without_replay_fills_is_incomplete_and_rejected() -> None:
    now = datetime.now(timezone.utc)
    rows = _metric_rows_for_required_strategies(now)
    trade_counts = {strategy_id: 1 for strategy_id in REQUIRED_CALIBRATION_STRATEGY_IDS}

    report = generator._build_report(
        broker_rows=rows,
        backtest_rows=[],
        trade_counts=trade_counts,
        start=now - timedelta(days=1),
        end=now,
        days=1,
        has_backtest_source=False,
    )

    assert report["within_tolerance"] is False
    with pytest.raises(CalibrationReportError):
        validate_calibration_report(report)


@pytest.mark.asyncio
async def test_generator_requires_replay_fills_unless_marked_incomplete(tmp_path) -> None:
    with pytest.raises(ValueError, match="backtest_fills_json is required"):
        await generator.generate_report(dsn="postgres://example", output_dir=tmp_path)


@pytest.mark.asyncio
async def test_generator_validates_before_writing_production_report(monkeypatch, tmp_path) -> None:
    now = datetime.now(timezone.utc)
    row = _metric_row("TPC", now)

    async def fake_load_broker_rows(*_args, **_kwargs):
        return [row], {"TPC": 1}

    monkeypatch.setattr(generator, "_load_broker_rows", fake_load_broker_rows)
    monkeypatch.setattr(generator, "_load_backtest_rows", lambda _path: [row])

    with pytest.raises(CalibrationReportError, match="within_tolerance"):
        await generator.generate_report(
            dsn="postgres://example",
            output_dir=tmp_path,
            as_of=now,
            backtest_fills_json=tmp_path / "replay_fills.json",
        )

    assert list(tmp_path.glob("parity_calibration_*.json")) == []
    assert list(tmp_path.glob("parity_calibration_*.md")) == []


@pytest.mark.asyncio
async def test_generator_allow_incomplete_can_write_diagnostic_report(monkeypatch, tmp_path) -> None:
    now = datetime.now(timezone.utc)

    async def fake_load_broker_rows(*_args, **_kwargs):
        return [], {}

    monkeypatch.setattr(generator, "_load_broker_rows", fake_load_broker_rows)

    json_path, md_path = await generator.generate_report(
        dsn="postgres://example",
        output_dir=tmp_path,
        as_of=now,
        allow_incomplete=True,
    )

    assert json_path.exists()
    assert md_path.exists()
    with pytest.raises(CalibrationReportError):
        load_calibration_report(json_path, now=now)


def _valid_report() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "report_type": "broker_backed",
        "generated_at": now.isoformat(),
        "window": {
            "days": 30,
            "start": (now - timedelta(days=30)).isoformat(),
            "end": now.isoformat(),
        },
        "source_tables": ["orders", "trades", "fills", "order_events"],
        "expected_family_ids": list(REQUIRED_CALIBRATION_FAMILIES),
        "expected_strategy_ids": list(REQUIRED_CALIBRATION_STRATEGY_IDS),
        "broker_fill_sample_count": 12 * len(REQUIRED_CALIBRATION_STRATEGY_IDS),
        "broker_completed_trade_count": 6 * len(REQUIRED_CALIBRATION_STRATEGY_IDS),
        "within_tolerance": True,
        "strategy_results": [
            _valid_strategy_result(strategy_id)
            for strategy_id in REQUIRED_CALIBRATION_STRATEGY_IDS
        ],
    }


def _valid_strategy_result(strategy_id: str) -> dict:
    return {
        "strategy_id": strategy_id,
        "broker_fill_sample_count": 12,
        "broker_completed_trade_count": 6,
        "backtest_fill_sample_count": 12,
        "within_tolerance": True,
        "broker_metrics": _valid_metrics(12),
        "backtest_metrics": _valid_metrics(12),
        "tolerance_results": _valid_tolerances(),
    }


def _metric_rows_for_required_strategies(now: datetime) -> list[generator.FillMetricRow]:
    return [_metric_row(strategy_id, now) for strategy_id in REQUIRED_CALIBRATION_STRATEGY_IDS]


def _metric_row(strategy_id: str, now: datetime) -> generator.FillMetricRow:
    return generator.FillMetricRow(
        strategy_id=strategy_id,
        symbol="MNQ" if strategy_id in REQUIRED_CALIBRATION_FAMILIES["momentum"] else "QQQ",
        side="BUY",
        order_type="LIMIT",
        tif="DAY",
        order_qty=10,
        fill_qty=10,
        reference_price=101.0,
        fill_price=101.0,
        commission=0.35,
        submit_ts=now,
        fill_ts=now,
    )


def _valid_metrics(sample_count: int) -> dict:
    return {
        "broker_fill_sample_count": sample_count,
        "mean_slippage_ticks": 0.0,
        "mean_commission": 0.35,
        "partial_fill_rate": 0.0,
        "ioc_reject_rate": 0.0,
        "mean_time_to_fill_seconds": 0.0,
    }


def _valid_tolerances() -> dict:
    return dict(VALID_TOLERANCES)
