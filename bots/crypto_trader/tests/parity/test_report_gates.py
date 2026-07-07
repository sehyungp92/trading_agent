"""Parity report and promotion gate tests."""

import json
from datetime import datetime, timedelta, timezone

from crypto_trader.core.models import Side
from crypto_trader.live.oms_store import OmsStore
from crypto_trader.parity.report import build_parity_report, evaluate_promotion_gate


def _write_event(path, stream: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stream": stream,
            "payload": payload,
        }) + "\n")


def test_parity_report_passes_clean_state(tmp_path) -> None:
    event_path = tmp_path / "parity_events.jsonl"
    _write_event(event_path, "decision", {
        "decision_id": "d1",
        "strategy_id": "momentum",
        "symbol": "BTC",
        "timeframe": "15m",
        "action": "no_order",
    })
    oms = OmsStore(tmp_path)
    oms.set_watermark("fills_since", datetime.now(timezone.utc).isoformat())
    oms.close()

    report = build_parity_report(tmp_path)
    gate = evaluate_promotion_gate(report)

    assert report.stream_counts["decision"] == 1
    assert gate.passed is True


def test_parity_gate_fails_discrepancy_stale_watermark_and_unprotected_entry(tmp_path) -> None:
    event_path = tmp_path / "parity_events.jsonl"
    _write_event(event_path, "execution", {
        "symbol": "BTC",
        "side": Side.LONG.value,
        "metadata": {"tag": "entry", "strategy_id": "momentum"},
    })
    oms = OmsStore(tmp_path)
    oms.set_watermark(
        "fills_since",
        (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
    )
    oms.record_discrepancy(kind="missing_position", description="missing BTC")
    oms.close()

    report = build_parity_report(tmp_path, max_watermark_age_sec=60)
    gate = evaluate_promotion_gate(report)

    assert gate.passed is False
    assert set(gate.failures) == {
        "unresolved_oms_discrepancies",
        "stale_fill_watermark",
        "unprotected_entry_fills",
    }


def test_parity_report_counts_decision_drift(tmp_path) -> None:
    event_path = tmp_path / "parity_events.jsonl"
    _write_event(event_path, "decision", {
        "decision_id": "actual",
        "strategy_id": "momentum",
        "symbol": "BTC",
        "timeframe": "15m",
        "action": "order",
    })
    expected = [{
        "stream": "decision",
        "payload": {
            "decision_id": "expected",
            "strategy_id": "momentum",
            "symbol": "BTC",
            "timeframe": "15m",
            "action": "no_order",
        },
    }]

    report = build_parity_report(tmp_path, expected_events=expected)

    assert report.decision_drift_count == 2


def test_parity_report_without_oms_does_not_create_database(tmp_path) -> None:
    report = build_parity_report(tmp_path)

    assert report.unresolved_oms_discrepancies == []
    assert report.fill_watermark_age_sec is None
    assert not (tmp_path / "live_oms.sqlite3").exists()


def test_parity_report_includes_allocation_drift_metrics(tmp_path) -> None:
    event_path = tmp_path / "parity_events.jsonl"
    _write_event(event_path, "position_allocation_snapshot", {
        "position_instance_id": "pos_1",
        "symbol": "BTC",
        "unallocated_qty": 0.0,
        "unknown_allocation": False,
    })
    _write_event(event_path, "position_allocation_snapshot", {
        "symbol": "BTC",
        "unallocated_qty": 0.2,
        "unknown_allocation": True,
    })

    report = build_parity_report(tmp_path)
    gate = evaluate_promotion_gate(report)

    assert report.allocation_count == 1
    assert report.unallocated_exposure_count == 1
    assert report.max_allocation_net_residual == 0.2
    assert report.position_ownership_drift is True
    assert "position_ownership_drift" in gate.failures
