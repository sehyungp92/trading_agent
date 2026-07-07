from __future__ import annotations

import json
import importlib.util
from datetime import date
from pathlib import Path

from trading_assistant.orchestrator import learning_sufficiency_audit as sufficiency_audit
from trading_assistant.orchestrator.action_handlers.daily_data import (
    DAILY_CURATED_EVENT_FILES,
    DAILY_RAW_EVENT_TAXONOMY,
)
from trading_assistant.orchestrator.learning_sufficiency_audit import LearningSufficiencyAuditor
from trading_assistant.orchestrator.lineage_audit import LineageAuditor
from trading_assistant.schemas.learning_sufficiency import (
    CoverageStatus,
    LearningCapabilityAuthority,
    LearningEligibility,
    RuntimeEvidenceSupportState,
)
from trading_assistant.schemas.telemetry_manifest import TelemetryEligibility, TelemetryManifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_trade(**overrides) -> dict:
    trade = {
        "event_id": "trade-1",
        "event_type": "trade",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "trade_id": "t1",
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "decision_id": "dec1",
        "risk_decision_id": "risk1",
        "order_id": "ord1",
        "fill_id": "fill1",
        "net_pnl": 12.5,
        "net_pnl_source": "observed_broker_statement",
        "after_cost_status": "observed",
        "fees_paid": 0.25,
        "entry_slippage_bps": 1.0,
        "exit_slippage_bps": 1.5,
        "experiment_id": "exp1",
    }
    trade.update(overrides)
    return trade


def _runtime_support_payload(*, source_path: str = "sidecar.py") -> dict:
    event_classes = (
        "trade",
        "missed_opportunity",
        "filter_decision",
        "orderbook_context",
        "portfolio_rule",
        "order",
        "fill",
        "pipeline_funnel",
        "deployment_metadata",
    )
    return {
        "schema_version": "runtime_evidence_support_v1",
        "support_source_paths": [source_path],
        "event_value_classifications": {event_class: "learning_authority" for event_class in event_classes},
        "evidence_classes": {
            event_class: {
                "supported": True,
                "configured_event_types": [event_class],
            }
            for event_class in event_classes
        },
    }


def _complete_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_curated = curated / "2026-05-02" / "bot1"
    bot_raw = raw / "2026-05-02" / "bot1"
    _write_jsonl(bot_curated / "trades.jsonl", [_base_trade()])
    _write_jsonl(bot_curated / "missed.jsonl", [{
        "event_id": "miss-1",
        "event_type": "missed_opportunity",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "risk_decision_id": "risk1",
        "would_have_pnl": 3.2,
        "experiment_id": "exp1",
    }])
    _write_jsonl(bot_raw / "filter_decision.jsonl", [{
        "event_type": "filter_decision",
        "strategy_id": "strat1",
        "decision_id": "dec1",
        "filter_name": "rsi",
        "threshold": 55,
        "actual_value": 60,
        "passed": True,
    }])
    _write_jsonl(bot_raw / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "strat1",
        "decision_id": "dec1",
        "order_id": "ord1",
    }])
    _write_jsonl(bot_raw / "fill.jsonl", [{
        "event_type": "fill",
        "strategy_id": "strat1",
        "fill_id": "fill1",
        "order_id": "ord1",
    }])
    _write_jsonl(bot_raw / "orderbook_context.jsonl", [{
        "event_type": "orderbook_context",
        "strategy_id": "strat1",
        "spread_bps": 2.0,
    }])
    _write_jsonl(bot_raw / "post_exit.jsonl", [{
        "event_type": "post_exit",
        "strategy_id": "strat1",
        "trade_id": "t1",
        "post_exit_backfill_status": "complete",
    }])
    _write_jsonl(bot_raw / "pipeline_funnel.jsonl", [{
        "event_type": "pipeline_funnel",
        "strategy_id": "strat1",
        "setups_seen": 10,
        "entries_attempted": 3,
        "fills": 2,
        "trades_closed": 1,
    }])
    _write_jsonl(bot_raw / "portfolio_rule_check.jsonl", [{
        "event_type": "portfolio_rule_check",
        "strategy_id": "strat1",
        "risk_decision_id": "risk1",
        "rule_id": "concentration",
        "result": "pass",
    }])
    metadata_path = tmp_path / "deployment_metadata.json"
    _write_json(metadata_path, {
        "deployment_id": "dep1",
        "proposal_ids": ["prop1"],
        "experiment_id": "exp1",
    })
    _write_json(raw / "expected_active_sessions.json", {
        "expected_session_days": ["2026-05-02"],
    })
    _write_json(raw / "runtime_evidence_support.json", _runtime_support_payload())
    return curated, raw, metadata_path


def test_learning_sufficiency_auditor_composes_with_telemetry_manifest(tmp_path: Path) -> None:
    curated, raw, metadata_path = _complete_fixture(tmp_path)
    findings = tmp_path / "memory" / "findings"
    telemetry_path = tmp_path / "telemetry_manifest.json"
    output_path = tmp_path / "learning_sufficiency_manifest.json"

    telemetry = LineageAuditor(curated, findings).build_telemetry_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        output_path=telemetry_path,
    )
    manifest = LearningSufficiencyAuditor(curated, findings, raw_data_dir=raw).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=telemetry_path,
        deployment_metadata_paths=[metadata_path],
        output_path=output_path,
    )

    assert output_path.exists()
    assert manifest.lineage_coverage.coverage_ratio == telemetry.lineage_coverage_ratio
    assert manifest.telemetry_authoritative_eligibility == telemetry.authoritative_eligibility.value
    assert manifest.event_counts_by_type["trade"] == 1
    assert manifest.event_counts_by_type["filter_decision"] == 1
    assert manifest.event_counts_by_type["pipeline_funnel"] == 1
    assert manifest.required_event_coverage["filter_decision_coverage"].status == CoverageStatus.PASS
    assert manifest.join_coverage["decision_to_trade_join"].status == CoverageStatus.PASS
    assert manifest.join_coverage["decision_to_order_join"].status == CoverageStatus.PASS
    assert manifest.join_coverage["order_to_fill_join"].status == CoverageStatus.PASS
    assert manifest.join_coverage["risk_portfolio_join"].status == CoverageStatus.PASS
    assert manifest.denominator_coverage["denominator_coverage"].status == CoverageStatus.PASS
    assert manifest.runtime_evidence_support["trade"].support_state == RuntimeEvidenceSupportState.OBSERVED
    assert manifest.deployment_metadata_coverage.evidence_paths == [str(metadata_path).replace("\\", "/")]
    assert manifest.eligibility == LearningEligibility.LEARNING_AUTHORITATIVE


def test_learning_sufficiency_auditor_does_not_overwrite_supplied_telemetry_manifest(tmp_path: Path) -> None:
    curated, raw, metadata_path = _complete_fixture(tmp_path)
    telemetry_path = tmp_path / "prebuilt_telemetry_manifest.json"
    telemetry = TelemetryManifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        event_counts_by_type={"trade": 7},
        lineage_coverage_ratio=0.5,
        missing_field_counts={"deployment_id": 7},
        authoritative_eligibility=TelemetryEligibility.INSUFFICIENT_LINEAGE,
    )
    telemetry_path.write_text(telemetry.model_dump_json(indent=2), encoding="utf-8")
    before = telemetry_path.read_text(encoding="utf-8")

    manifest = LearningSufficiencyAuditor(curated, tmp_path / "memory" / "findings", raw_data_dir=raw).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=telemetry_path,
        deployment_metadata_paths=[metadata_path],
    )

    assert telemetry_path.read_text(encoding="utf-8") == before
    assert manifest.lineage_coverage.coverage_ratio == 0.5
    assert manifest.telemetry_authoritative_eligibility == TelemetryEligibility.INSUFFICIENT_LINEAGE.value


def test_order_to_fill_join_accepts_partial_fill(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-02" / "bot1"
    _write_jsonl(bot_raw / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "strat1",
        "order_id": "ord-partial",
        "status": "partially_filled",
    }])
    _write_jsonl(bot_raw / "fill.jsonl", [{
        "event_type": "fill",
        "strategy_id": "strat1",
        "fill_id": "fill-partial",
        "order_id": "ord-partial",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["order_to_fill_join"]
    assert check.status == CoverageStatus.PASS
    assert check.details["matched_fill_order_count"] == 1
    assert check.details["terminal_no_fill_order_count"] == 0


def test_order_to_fill_join_accepts_terminal_no_fill_orders(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-02" / "bot1"
    _write_jsonl(bot_raw / "order.jsonl", [
        {
            "event_type": "order",
            "strategy_id": "strat1",
            "order_id": "ord-cancel",
            "status": "canceled",
            "cancel_reason": "limit_moved",
        },
        {
            "event_type": "order",
            "strategy_id": "strat1",
            "order_id": "ord-reject",
            "status": "rejected",
            "reject_reason": "risk_limit",
        },
        {
            "event_type": "order",
            "strategy_id": "strat1",
            "order_id": "ord-none",
            "status": "no_order",
            "no_order_reason": "spread_too_wide",
        },
    ])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["order_to_fill_join"]
    assert check.status == CoverageStatus.PASS
    assert check.observed_count == 3
    assert check.details["terminal_no_fill_order_count"] == 3
    assert check.missing_fields == []


def test_order_to_fill_join_groups_repeated_lifecycle_rows(tmp_path: Path) -> None:
    cases = [
        {
            "name": "filled",
            "orders": [
                {"event_type": "order", "strategy_id": "strat1", "order_id": "ord-repeat", "status": "submitted"},
                {"event_type": "order", "strategy_id": "strat1", "order_id": "ord-repeat", "status": "filled"},
            ],
            "fills": [{"event_type": "fill", "strategy_id": "strat1", "order_id": "ord-repeat", "fill_id": "fill-repeat"}],
            "details": {"matched_fill_order_count": 1, "fill_required_order_count": 1, "terminal_no_fill_order_count": 0},
        },
        {
            "name": "canceled",
            "orders": [
                {"event_type": "order", "strategy_id": "strat1", "order_id": "ord-cancel", "status": "submitted"},
                {
                    "event_type": "order",
                    "strategy_id": "strat1",
                    "order_id": "ord-cancel",
                    "status": "canceled",
                    "cancel_reason": "limit_moved",
                },
            ],
            "fills": [],
            "details": {"matched_fill_order_count": 0, "fill_required_order_count": 0, "terminal_no_fill_order_count": 1},
        },
    ]

    for case in cases:
        root = tmp_path / case["name"]
        curated = root / "curated"
        raw = root / "raw"
        bot_raw = raw / "2026-05-02" / "bot1"
        _write_jsonl(bot_raw / "order.jsonl", case["orders"])
        if case["fills"]:
            _write_jsonl(bot_raw / "fill.jsonl", case["fills"])

        manifest = LearningSufficiencyAuditor(
            curated,
            root / "memory" / "findings",
            raw_data_dir=raw,
        ).build_manifest(
            bot_id="bot1",
            strategy_id="strat1",
            run_month="2026-05",
            window_start=date(2026, 5, 1),
            window_end=date(2026, 5, 31),
            telemetry_manifest_path=root / "telemetry_manifest.json",
        )

        check = manifest.join_coverage["order_to_fill_join"]
        assert check.status == CoverageStatus.PASS, case["name"]
        assert check.observed_count == 1, case["name"]
        assert check.required_count == 1, case["name"]
        assert check.details["order_lifecycle_row_count"] == 2, case["name"]
        assert check.details["order_identity_group_count"] == 1, case["name"]
        for key, value in case["details"].items():
            assert check.details[key] == value, case["name"]


def test_decision_to_order_join_accepts_terminal_order_without_completed_trade(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-02" / "bot1"
    _write_jsonl(bot_raw / "filter_decision.jsonl", [{
        "event_type": "filter_decision",
        "strategy_id": "strat1",
        "decision_id": "decision-cancel",
        "filter_name": "rsi",
        "threshold": 55,
        "actual_value": 60,
        "passed": True,
    }])
    _write_jsonl(bot_raw / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "strat1",
        "decision_id": "decision-cancel",
        "order_id": "ord-cancel",
        "status": "canceled",
        "cancel_reason": "limit_moved",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.join_coverage["decision_to_order_join"].status == CoverageStatus.PASS
    assert manifest.join_coverage["order_to_fill_join"].status == CoverageStatus.PASS


def test_decision_to_order_join_accepts_decision_record_no_order_reason(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(raw / "2026-05-02" / "bot1" / "filter_decision.jsonl", [{
        "event_type": "filter_decision",
        "strategy_id": "strat1",
        "decision_id": "decision-no-order",
        "filter_name": "spread",
        "actual_value": 8.5,
        "passed": False,
        "no_order_reason": "spread_too_wide",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["decision_to_order_join"]
    assert check.status == CoverageStatus.PASS
    assert check.details["target_record_count"] == 1


def test_order_to_fill_join_accepts_inferred_fill_events(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-02" / "bot1"
    _write_jsonl(bot_raw / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "strat1",
        "order_id": "ord-inferred",
        "status": "submitted",
    }])
    _write_jsonl(bot_raw / "inferred_fill.jsonl", [{
        "event_type": "inferred_fill",
        "strategy_id": "strat1",
        "fill_id": "fill-inferred",
        "order_id": "ord-inferred",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["order_to_fill_join"]
    assert check.status == CoverageStatus.PASS
    assert check.details["matched_fill_order_count"] == 1
    assert manifest.event_counts_by_type["fill"] == 1


def test_order_to_fill_join_accepts_kis_order_keys(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-06-04" / "k_stock_trader"
    _write_jsonl(bot_raw / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "KALCB",
        "kis_order_id": "kis-ord-1",
        "status": "submitted",
    }])
    _write_jsonl(bot_raw / "fill.jsonl", [{
        "event_type": "fill",
        "strategy_id": "KALCB",
        "kis_order_id": "kis-ord-1",
        "kis_exec_id": "kis-fill-1",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="k_stock_trader",
        strategy_id="KALCB",
        run_month="2026-06",
        window_start=date(2026, 6, 1),
        window_end=date(2026, 6, 30),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["order_to_fill_join"]
    assert check.status == CoverageStatus.PASS
    assert check.details["matched_fill_order_count"] == 1
    assert check.details["fill_id_count"] == 1


def test_order_to_fill_join_accepts_singular_anchored_plural_aliases(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-02" / "bot1"
    _write_jsonl(bot_raw / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "strat1",
        "order_id": "ord-canonical",
        "client_order_ids": ["client-a", "client-b"],
        "order_ids_are_aliases": True,
        "status": "submitted",
    }])
    _write_jsonl(bot_raw / "fill.jsonl", [{
        "event_type": "fill",
        "strategy_id": "strat1",
        "order_id": "ord-canonical",
        "fill_id": "fill-canonical",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["order_to_fill_join"]
    assert check.status == CoverageStatus.PASS
    assert check.details["order_identity_group_count"] == 1
    assert check.details["fill_required_order_count"] == 1
    assert check.details["matched_fill_order_count"] == 1
    assert check.details["orphan_fill_count"] == 0


def test_join_coverage_rejects_mismatched_decision_ids(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [
        _base_trade(decision_id="trade-decision")
    ])
    _write_jsonl(raw / "2026-05-02" / "bot1" / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "strat1",
        "decision_id": "different-decision",
        "order_id": "ord1",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    trade_join = manifest.join_coverage["decision_to_trade_join"]
    order_join = manifest.join_coverage["decision_to_order_join"]
    assert trade_join.status == CoverageStatus.MISSING
    assert order_join.status == CoverageStatus.MISSING
    assert "different-decision" in order_join.details["orphan_join_refs"]


def test_cross_record_join_rejects_orphan_refs_above_ratio_threshold(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    decisions = [
        {"event_type": "filter_decision", "strategy_id": "strat1", "decision_id": f"dec-{index}"}
        for index in range(19)
    ]
    orders = [
        {"event_type": "order", "strategy_id": "strat1", "decision_id": f"dec-{index}", "order_id": f"ord-{index}"}
        for index in range(19)
    ]
    orders.append({"event_type": "order", "strategy_id": "strat1", "decision_id": "dec-orphan", "order_id": "ord-orphan"})
    _write_jsonl(raw / "2026-05-02" / "bot1" / "filter_decision.jsonl", decisions)
    _write_jsonl(raw / "2026-05-02" / "bot1" / "order.jsonl", orders)

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["decision_to_order_join"]
    assert check.status == CoverageStatus.MISSING
    assert check.observed_count == 0
    assert check.details["joined_target_count"] == 19
    assert check.details["target_record_count"] == 20
    assert check.details["orphan_join_refs"] == ["dec-orphan"]


def test_decision_to_order_join_rejects_trade_only_source(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [
        _base_trade(decision_id="trade-decision")
    ])
    _write_jsonl(raw / "2026-05-02" / "bot1" / "order.jsonl", [{
        "event_type": "order",
        "strategy_id": "strat1",
        "decision_id": "trade-decision",
        "order_id": "ord1",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.join_coverage["decision_to_trade_join"].status == CoverageStatus.PASS
    assert manifest.join_coverage["decision_to_order_join"].status == CoverageStatus.MISSING


def test_order_to_fill_join_rejects_invalid_order_fill_rows(tmp_path: Path) -> None:
    cases = [
        {
            "name": "fill_missing_id",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "order_id": "ord-valid", "status": "submitted"}],
            "fills": [{"event_type": "fill", "strategy_id": "strat1", "order_id": "ord-valid"}],
            "details": {"missing_fill_id_count": 1, "unfilled_required_order_count": 1},
        },
        {
            "name": "orphan_fill_only",
            "orders": [],
            "fills": [{"event_type": "fill", "strategy_id": "strat1", "fill_id": "fill-orphan", "order_id": "ord-missing"}],
            "details": {"orphan_fill_count": 1},
        },
        {
            "name": "mixed_valid_and_orphan",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "order_id": "ord-valid", "status": "submitted"}],
            "fills": [
                {"event_type": "fill", "strategy_id": "strat1", "fill_id": "fill-valid", "order_id": "ord-valid"},
                {"event_type": "fill", "strategy_id": "strat1", "fill_id": "fill-orphan", "order_id": "ord-missing"},
            ],
            "details": {"matched_fill_order_count": 1, "orphan_fill_count": 1},
        },
        {
            "name": "fill_missing_order_ref",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "order_id": "ord-valid", "status": "submitted"}],
            "fills": [{"event_type": "fill", "strategy_id": "strat1", "fill_id": "fill-no-order"}],
            "details": {"missing_fill_order_ref_count": 1, "unfilled_required_order_count": 1},
        },
        {
            "name": "fill_required_order_missing_id",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "status": "submitted"}],
            "fills": [],
            "details": {"missing_fill_required_order_id_count": 1},
        },
        {
            "name": "fill_required_order_unfilled",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "order_id": "ord-unfilled", "status": "submitted"}],
            "fills": [],
            "details": {"unfilled_required_order_count": 1},
        },
        {
            "name": "terminal_order_missing_id",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "status": "rejected", "reject_reason": "risk_limit"}],
            "fills": [],
            "details": {"terminal_order_missing_id_count": 1},
        },
        {
            "name": "placeholder_order_id",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "order_id": "unknown", "status": "submitted"}],
            "fills": [{"event_type": "fill", "strategy_id": "strat1", "fill_id": "fill-placeholder", "order_id": "unknown"}],
            "details": {"missing_fill_required_order_id_count": 1, "missing_fill_order_ref_count": 1},
        },
        {
            "name": "plural_order_ids_one_fill",
            "orders": [{"event_type": "order", "strategy_id": "strat1", "order_ids": ["ord-a", "ord-b"], "status": "submitted"}],
            "fills": [{"event_type": "fill", "strategy_id": "strat1", "fill_id": "fill-a", "order_id": "ord-a"}],
            "details": {"matched_fill_order_count": 1, "fill_required_order_count": 2, "unfilled_required_order_count": 1},
        },
        {
            "name": "parent_basket_order_ids_one_fill",
            "orders": [{
                "event_type": "order",
                "strategy_id": "strat1",
                "order_id": "parent-basket",
                "order_ids": ["ord-a", "ord-b"],
                "status": "submitted",
            }],
            "fills": [{"event_type": "fill", "strategy_id": "strat1", "fill_id": "fill-a", "order_id": "ord-a"}],
            "details": {"matched_fill_order_count": 1, "fill_required_order_count": 2, "unfilled_required_order_count": 1},
        },
    ]

    for case in cases:
        root = tmp_path / case["name"]
        curated = root / "curated"
        raw = root / "raw"
        bot_raw = raw / "2026-05-02" / "bot1"
        if case["orders"]:
            _write_jsonl(bot_raw / "order.jsonl", case["orders"])
        if case["fills"]:
            _write_jsonl(bot_raw / "fill.jsonl", case["fills"])

        manifest = LearningSufficiencyAuditor(
            curated,
            root / "memory" / "findings",
            raw_data_dir=raw,
        ).build_manifest(
            bot_id="bot1",
            strategy_id="strat1",
            run_month="2026-05",
            window_start=date(2026, 5, 1),
            window_end=date(2026, 5, 31),
            telemetry_manifest_path=root / "telemetry_manifest.json",
        )

        check = manifest.join_coverage["order_to_fill_join"]
        assert check.status == CoverageStatus.MISSING, case["name"]
        assert check.observed_count == 0, case["name"]
        for key, value in case["details"].items():
            assert check.details[key] == value, case["name"]


def test_portfolio_rule_denial_counts_as_risk_portfolio_join(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-02" / "bot1" / "missed.jsonl", [{
        "event_id": "miss-portfolio-1",
        "event_type": "missed_opportunity",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "portfolio_rule_id": "concentration",
        "would_have_pnl": 1.2,
    }])
    _write_jsonl(raw / "2026-05-02" / "bot1" / "portfolio_rule_check.jsonl", [{
        "event_type": "portfolio_rule_check",
        "strategy_id": "strat1",
        "rule_id": "concentration",
        "result": "deny",
        "deny_reason": "sector_limit",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.join_coverage["risk_portfolio_join"].status == CoverageStatus.PASS
    assert manifest.required_event_coverage["portfolio_rule_coverage"].status == CoverageStatus.PASS


def test_risk_decision_events_count_as_portfolio_risk_evidence(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [
        _base_trade(risk_decision_id="risk-1")
    ])
    _write_jsonl(raw / "2026-05-02" / "bot1" / "risk_decision.jsonl", [{
        "event_type": "risk_decision",
        "strategy_id": "strat1",
        "risk_decision_id": "risk-1",
        "decision": "allow",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.event_counts_by_type["portfolio_rule"] == 1
    assert manifest.required_event_coverage["portfolio_rule_coverage"].status == CoverageStatus.PASS
    assert manifest.join_coverage["risk_portfolio_join"].status == CoverageStatus.PASS


def test_standalone_risk_decision_denial_counts_as_risk_portfolio_join(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(raw / "2026-05-02" / "bot1" / "risk_decision.jsonl", [{
        "event_type": "risk_decision",
        "strategy_id": "strat1",
        "risk_decision_id": "risk-deny-1",
        "decision": "deny",
        "deny_reason": "gross_exposure_limit",
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.join_coverage["risk_portfolio_join"]
    assert check.status == CoverageStatus.PASS
    assert check.details["target_record_count"] == 1


def test_curated_risk_decision_summary_counts_as_portfolio_risk_evidence(tmp_path: Path) -> None:
    from trading_assistant.skills.build_daily_metrics import build_portfolio_rules_summary

    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    summary = build_portfolio_rules_summary([{
        "event_type": "risk_decision",
        "strategy_id": "strat1",
        "risk_decision_id": "risk-curated-1",
        "decision": "deny",
        "deny_reason": "gross_exposure_limit",
    }])
    _write_json(curated / "2026-05-02" / "portfolio" / "rule_blocks_summary.json", summary)

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.event_counts_by_type["portfolio_rule"] == 1
    assert manifest.required_event_coverage["portfolio_rule_coverage"].status == CoverageStatus.PASS
    assert manifest.join_coverage["risk_portfolio_join"].status == CoverageStatus.PASS


def test_curated_risk_decision_summary_without_id_cannot_join_on_unknown(tmp_path: Path) -> None:
    from trading_assistant.skills.build_daily_metrics import build_portfolio_rules_summary

    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    summary = build_portfolio_rules_summary([{
        "event_type": "risk_decision",
        "strategy_id": "strat1",
        "decision": "deny",
        "deny_reason": "gross_exposure_limit",
    }])
    _write_json(curated / "2026-05-02" / "portfolio" / "rule_blocks_summary.json", summary)

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert summary["records"][0]["rule_id"] == ""
    assert summary["records"][0]["portfolio_rule_id"] == ""
    assert manifest.event_counts_by_type["portfolio_rule"] == 1
    assert manifest.join_coverage["risk_portfolio_join"].status == CoverageStatus.MISSING


def test_k_stock_pipeline_funnel_reports_denominator_coverage_not_unknown(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_json(raw / "expected_active_sessions.json", {
        "expected_session_days": ["2026-06-04"],
    })
    _write_jsonl(raw / "2026-06-04" / "k_stock_trader" / "pipeline_funnels.jsonl", [{
        "event_type": "pipeline_funnels",
        "bot_id": "k_stock_trader",
        "strategy_id": "KALCB",
        "date": "2026-06-04",
        "funnel": {
            "setups_detected": {"005930": 12},
            "confirmations": {"005930": 8},
            "entries_attempted": {"005930": 4},
            "fills": {"005930": 3},
            "trades_closed": {"005930": 2},
        },
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="k_stock_trader",
        strategy_id="KALCB",
        run_month="2026-06",
        window_start=date(2026, 6, 1),
        window_end=date(2026, 6, 30),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    check = manifest.denominator_coverage["denominator_coverage"]
    assert check.status == CoverageStatus.PASS
    assert check.observed_count == 1
    assert manifest.event_counts_by_type["pipeline_funnel"] == 1


def test_denominator_coverage_requires_ninety_percent_session_coverage(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_json(raw / "expected_active_sessions.json", {
        "expected_session_days": [f"2026-05-{day:02d}" for day in range(1, 11)],
    })
    for day in range(1, 11):
        date_str = f"2026-05-{day:02d}"
        if day <= 9:
            _write_jsonl(curated / date_str / "bot1" / "trades.jsonl", [
                _base_trade(trade_id=f"t{day}", event_id=f"trade-{day}")
            ])
        if day <= 8:
            _write_jsonl(raw / date_str / "bot1" / "pipeline_funnel.jsonl", [{
                "event_type": "pipeline_funnel",
                "strategy_id": "strat1",
                "setups_seen": 5,
                "entries_attempted": 1,
            }])

    auditor = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    )
    partial = auditor.build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_partial.json",
    )
    _write_jsonl(raw / "2026-05-09" / "bot1" / "pipeline_funnel.jsonl", [{
        "event_type": "pipeline_funnel",
        "strategy_id": "strat1",
        "setups_seen": 5,
        "entries_attempted": 1,
    }])
    passing = auditor.build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_passing.json",
    )

    partial_check = partial.denominator_coverage["denominator_coverage"]
    passing_check = passing.denominator_coverage["denominator_coverage"]
    assert partial_check.status == CoverageStatus.PARTIAL
    assert partial_check.observed_count == 8
    assert partial_check.required_count == 10
    assert passing_check.status == CoverageStatus.PASS
    assert passing_check.coverage_ratio == 0.9
    assert "2026-05-10" in passing_check.details["missing_expected_session_days"]


def test_denominator_coverage_counts_expected_session_with_no_records(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_json(raw / "expected_active_sessions.json", {
        "expected_session_days": ["2026-05-01", "2026-05-02"],
    })
    _write_jsonl(raw / "2026-05-01" / "bot1" / "pipeline_funnel.jsonl", [{
        "event_type": "pipeline_funnel",
        "strategy_id": "strat1",
        "setups_seen": 5,
        "entries_attempted": 1,
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry.json",
    )

    check = manifest.denominator_coverage["denominator_coverage"]
    assert check.status == CoverageStatus.PARTIAL
    assert check.observed_count == 1
    assert check.required_count == 2
    assert check.details["missing_expected_session_days"] == ["2026-05-02"]


def test_runtime_evidence_support_distinguishes_support_states(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-01" / "bot1" / "trades.jsonl", [_base_trade()])
    _write_json(raw / "expected_active_sessions.json", {"expected_session_days": ["2026-05-01"]})
    _write_json(raw / "runtime_evidence_support.json", {
        "support_source_paths": ["sidecar.py"],
        "event_value_classifications": {
            "trade": "learning_authority",
            "missed_opportunity": "learning_authority",
        },
        "evidence_classes": {
            "trade": {"supported": True, "configured_event_types": ["trade"]},
            "missed_opportunity": {"supported": True, "configured_event_types": ["missed_opportunity"]},
            "order": {"support_state": "unsupported", "reason": "sidecar does not export orders"},
        },
    })

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    support = manifest.runtime_evidence_support
    assert support["trade"].support_state == RuntimeEvidenceSupportState.OBSERVED
    assert support["missed_opportunity"].support_state == RuntimeEvidenceSupportState.SUPPORTED_BUT_UNOBSERVED
    assert support["order"].support_state == RuntimeEvidenceSupportState.UNSUPPORTED
    assert "sidecar.py" in support["order"].support_source_paths


def test_runtime_evidence_observation_without_support_does_not_authorize_learning(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-01" / "bot1" / "trades.jsonl", [_base_trade()])
    _write_json(raw / "expected_active_sessions.json", {"expected_session_days": ["2026-05-01"]})

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.runtime_evidence_support["trade"].support_state == RuntimeEvidenceSupportState.UNSUPPORTED
    assert any(
        "runtime_evidence_support:" in reason and ":trade:unsupported" in reason
        for status in manifest.capability_status.values()
        for reason in status.blocking_reasons
    )


def test_ops_only_events_do_not_satisfy_learning_runtime_evidence(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-01" / "bot1" / "trades.jsonl", [_base_trade()])
    bot_raw = raw / "2026-05-01" / "bot1"
    _write_jsonl(bot_raw / "health_report.jsonl", [{"event_type": "health_report", "strategy_id": "strat1"}])
    _write_jsonl(bot_raw / "daily_snapshot.jsonl", [{"event_type": "daily_snapshot", "strategy_id": "strat1"}])
    _write_json(raw / "expected_active_sessions.json", {"expected_session_days": ["2026-05-01"]})
    _write_json(raw / "runtime_evidence_support.json", {
        "support_source_paths": ["ops_sidecar.py"],
        "event_value_classifications": {
            "trade": "operational_health",
            "health_report": "operational_health",
            "daily_snapshot": "operational_health",
        },
        "evidence_classes": {"trade": {"supported": True, "configured_event_types": ["trade"]}},
    })

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert "health_report" not in manifest.event_counts_by_type
    assert "daily_snapshot" not in manifest.event_counts_by_type
    assert manifest.runtime_evidence_support["trade"].support_state == RuntimeEvidenceSupportState.SUPPORTED_BUT_UNOBSERVED
    assert manifest.runtime_evidence_support["trade"].observed_event_count == 1
    assert manifest.runtime_evidence_support["trade"].reason == "configured_runtime_support_not_learning_authority"
    assert manifest.eligibility != LearningEligibility.LEARNING_AUTHORITATIVE


def test_missing_event_value_classification_does_not_authorize_observed_runtime_evidence(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    _write_jsonl(curated / "2026-05-01" / "bot1" / "trades.jsonl", [_base_trade()])
    _write_json(raw / "expected_active_sessions.json", {"expected_session_days": ["2026-05-01"]})
    _write_json(raw / "runtime_evidence_support.json", {
        "support_source_paths": ["sidecar.py"],
        "evidence_classes": {"trade": {"supported": True, "configured_event_types": ["trade"]}},
    })

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    support = manifest.runtime_evidence_support["trade"]
    assert support.support_state == RuntimeEvidenceSupportState.SUPPORTED_BUT_UNOBSERVED
    assert support.reason == "configured_runtime_support_not_learning_authority"


def test_runtime_support_credits_learning_authority_aliases_for_canonical_classes(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-01" / "bot1"
    _write_jsonl(bot_raw / "inferred_fill.jsonl", [{
        "event_type": "inferred_fill",
        "strategy_id": "strat1",
        "order_id": "ord1",
        "fill_id": "fill1",
    }])
    _write_jsonl(bot_raw / "portfolio_rule_check.jsonl", [{
        "event_type": "portfolio_rule_check",
        "strategy_id": "strat1",
        "risk_decision_id": "risk1",
        "rule_id": "concentration",
    }])
    _write_json(raw / "runtime_evidence_support.json", {
        "support_source_paths": ["ibkr_sidecar.py"],
        "event_value_classifications": {
            "inferred_fill": "learning_authority",
            "portfolio_rule_check": "learning_authority",
        },
        "capabilities": {
            "order_to_fill_join": {
                "required_event_types": ["fill"],
                "missing_configured_event_types": ["fill"],
            },
            "risk_portfolio_join": {
                "required_event_types": ["portfolio_rule"],
                "missing_configured_event_types": ["portfolio_rule"],
            },
        },
    })

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.event_counts_by_type["fill"] == 1
    assert manifest.event_counts_by_type["portfolio_rule"] == 1
    assert manifest.runtime_evidence_support["fill"].support_state == RuntimeEvidenceSupportState.OBSERVED
    assert manifest.runtime_evidence_support["fill"].configured_event_types == ["inferred_fill"]
    assert manifest.runtime_evidence_support["portfolio_rule"].support_state == RuntimeEvidenceSupportState.OBSERVED
    assert manifest.runtime_evidence_support["portfolio_rule"].configured_event_types == ["portfolio_rule_check"]


def test_direct_runtime_support_payloads_credit_learning_authority_alias_keys(tmp_path: Path) -> None:
    cases = {
        "runtime_evidence_support": {"runtime_evidence_support": {"inferred_fill": {"supported": True}}},
        "evidence_classes": {"evidence_classes": {"inferred_fill": {"supported": True}}},
        "events": {"events": {"inferred_fill": {"supported": True}}},
        "support": {"support": [{"event_type": "inferred_fill", "supported": True}]},
    }
    for name, support_payload in cases.items():
        root = tmp_path / name
        curated = root / "curated"
        raw = root / "raw"
        bot_raw = raw / "2026-05-01" / "bot1"
        _write_jsonl(bot_raw / "inferred_fill.jsonl", [{
            "event_type": "inferred_fill",
            "strategy_id": "strat1",
            "order_id": "ord1",
            "fill_id": f"fill-{name}",
        }])
        _write_json(raw / "runtime_evidence_support.json", {
            "support_source_paths": ["sidecar.py"],
            "event_value_classifications": {"inferred_fill": "learning_authority"},
            **support_payload,
        })

        manifest = LearningSufficiencyAuditor(
            curated,
            root / "memory" / "findings",
            raw_data_dir=raw,
        ).build_manifest(
            bot_id="bot1",
            strategy_id="strat1",
            run_month="2026-05",
            window_start=date(2026, 5, 1),
            window_end=date(2026, 5, 31),
            telemetry_manifest_path=root / "telemetry_manifest.json",
        )

        support = manifest.runtime_evidence_support["fill"]
        assert support.support_state == RuntimeEvidenceSupportState.OBSERVED
        assert support.configured_event_types == ["inferred_fill"]


def test_direct_runtime_support_payload_rejects_non_authority_alias_key(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-01" / "bot1"
    _write_jsonl(bot_raw / "inferred_fill.jsonl", [{
        "event_type": "inferred_fill",
        "strategy_id": "strat1",
        "order_id": "ord1",
        "fill_id": "fill1",
    }])
    _write_json(raw / "runtime_evidence_support.json", {
        "support_source_paths": ["sidecar.py"],
        "event_value_classifications": {"inferred_fill": "operational_health"},
        "evidence_classes": {"inferred_fill": {"supported": True}},
    })

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    support = manifest.runtime_evidence_support["fill"]
    assert support.support_state == RuntimeEvidenceSupportState.SUPPORTED_BUT_UNOBSERVED
    assert support.reason == "configured_runtime_support_not_learning_authority"


def test_runtime_support_rejects_non_authority_aliases_for_canonical_classes(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_raw = raw / "2026-05-01" / "bot1"
    _write_jsonl(bot_raw / "inferred_fill.jsonl", [{
        "event_type": "inferred_fill",
        "strategy_id": "strat1",
        "order_id": "ord1",
        "fill_id": "fill1",
    }])
    _write_json(raw / "runtime_evidence_support.json", {
        "support_source_paths": ["ibkr_sidecar.py"],
        "event_value_classifications": {"inferred_fill": "operational_health"},
        "capabilities": {
            "order_to_fill_join": {
                "required_event_types": ["fill"],
                "missing_configured_event_types": ["fill"],
            },
        },
    })

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    support = manifest.runtime_evidence_support["fill"]
    assert support.support_state == RuntimeEvidenceSupportState.UNSUPPORTED
    assert support.reason == "not_configured_by_runtime_support_source"


def test_multisource_join_requires_learning_authority_for_each_consumed_class(tmp_path: Path) -> None:
    curated, raw, metadata_path = _complete_fixture(tmp_path)
    payload = _runtime_support_payload()
    payload["event_value_classifications"]["filter_decision"] = "operational_health"
    _write_json(raw / "runtime_evidence_support.json", payload)

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
        deployment_metadata_paths=[metadata_path],
    )

    assert manifest.join_coverage["decision_to_order_join"].status == CoverageStatus.PASS
    execution = manifest.capability_status["execution_learning"]
    assert execution.status == LearningCapabilityAuthority.BLOCKED
    assert (
        "runtime_evidence_support:decision_to_order_join:filter_decision:"
        "runtime_support_not_learning_authority"
    ) in execution.blocking_checks


def test_phase_runtime_support_payload_carries_source_event_value_classes() -> None:
    tool_path = Path(__file__).resolve().parents[3] / "tools" / "build_learning_sufficiency_manifests.py"
    spec = importlib.util.spec_from_file_location("build_learning_sufficiency_manifests", tool_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases = [
        (
            {"contract_id": "crypto_momentum_v1", "bot_id": "crypto", "strategy_id": "crypto_momentum_v1"},
            {"trade": "learning_authority", "heartbeat": "operational_health"},
        ),
        (
            {"contract_id": "k_stock_olr_kalcb", "bot_id": "k_stock", "strategy_id": "KALCB"},
            {"order": "learning_authority", "daily_snapshot": "operational_health"},
        ),
        (
            {"contract_id": "trading_momentum_family", "bot_id": "ibkr", "strategy_id": "trading_momentum_family"},
            {"risk_decision": "learning_authority", "heartbeat": "operational_health"},
        ),
    ]
    for scope, expected in cases:
        classes = module._runtime_support_payload(scope, {"capabilities": {}})["event_value_classifications"]
        for event_type, value_class in expected.items():
            assert classes[event_type] == value_class
    ibkr_scope = {"contract_id": "trading_momentum_family", "bot_id": "ibkr", "strategy_id": "trading_momentum_family"}
    payload = module._runtime_support_payload(ibkr_scope, {
        "capabilities": {
            "order_to_fill_join": {
                "configured": False,
                "status": "unsupported",
                "observed": False,
                "required_event_types": ["fill"],
                "missing_configured_event_types": ["fill"],
            },
            "risk_portfolio_join": {
                "configured": True,
                "status": "configured_unobserved",
                "observed": False,
                "required_event_types": ["portfolio_rule"],
                "missing_configured_event_types": ["portfolio_rule"],
            },
            "decision_to_order_join": {
                "configured": False,
                "status": "unsupported",
                "observed": False,
                "required_event_types": ["decision_event", "order"],
                "missing_configured_event_types": ["decision_event"],
            },
            "counterfactual_coverage": {
                "configured": False,
                "status": "unsupported",
                "observed": False,
                "required_event_types": ["missed_opportunity", "post_exit"],
                "missing_configured_event_types": ["post_exit"],
            },
        }
    })
    assert payload["capabilities"]["order_to_fill_join"]["configured"] is True
    assert payload["capabilities"]["order_to_fill_join"]["missing_configured_event_types"] == []
    assert payload["capabilities"]["risk_portfolio_join"]["missing_configured_event_types"] == []
    assert payload["capabilities"]["decision_to_order_join"]["required_event_types"] == [
        "filter_decision",
        "order",
    ]
    assert "decision_event" not in payload["capabilities"]["decision_to_order_join"]["required_event_types"]
    assert payload["capabilities"]["counterfactual_coverage"]["required_event_types"] == ["missed_opportunity"]
    assert "post_exit" not in payload["capabilities"]["counterfactual_coverage"]["required_event_types"]


def test_learning_sufficiency_auditor_emits_deterministic_ranked_gaps(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    findings = tmp_path / "memory" / "findings"
    bot_dir = curated / "2026-05-02" / "bot1"
    _write_jsonl(bot_dir / "trades.jsonl", [_base_trade(config_version="")])

    auditor = LearningSufficiencyAuditor(curated, findings, raw_data_dir=raw)
    first = auditor.build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
        output_path=tmp_path / "manifest_1.json",
    )
    second = auditor.build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
        output_path=tmp_path / "manifest_2.json",
    )

    first_gaps = [
        (gap.expected_learning_value.value, gap.frequency, gap.blocked_learning_capability, gap.missing_field)
        for gap in first.known_gaps
    ]
    second_gaps = [
        (gap.expected_learning_value.value, gap.frequency, gap.blocked_learning_capability, gap.missing_field)
        for gap in second.known_gaps
    ]

    assert first.eligibility == LearningEligibility.INSUFFICIENT_LINEAGE
    assert "approval_grade_strategy_change" in first.blocked_learning_capabilities
    assert any("config_version" in gap.missing_field for gap in first.known_gaps)
    assert first_gaps == second_gaps


def test_learning_sufficiency_auditor_reuses_daily_raw_taxonomy(tmp_path: Path) -> None:
    assert sufficiency_audit.DAILY_RAW_EVENT_TAXONOMY == DAILY_RAW_EVENT_TAXONOMY
    assert sufficiency_audit.DAILY_CURATED_EVENT_FILES == DAILY_CURATED_EVENT_FILES

    curated, raw, metadata_path = _complete_fixture(tmp_path)
    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
        deployment_metadata_paths=[metadata_path],
    )

    for event_type in [
        "trade",
        "missed_opportunity",
        "filter_decision",
        "order",
        "fill",
        "orderbook_context",
        "post_exit",
        "pipeline_funnel",
        "portfolio_rule",
        "deployment_metadata",
    ]:
        assert manifest.event_counts_by_type[event_type] >= 1


def test_learning_sufficiency_manifest_omits_raw_secret_payloads_and_stays_bounded(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_dir = curated / "2026-05-02" / "bot1"
    secret_trade = _base_trade(
        api_key="sk-live-secret-value",
        password="do-not-leak",
        raw_payload="x" * 200_000,
    )
    _write_jsonl(bot_dir / "trades.jsonl", [secret_trade])
    output_path = tmp_path / "learning_sufficiency_manifest.json"

    LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
        output_path=output_path,
    )

    payload = output_path.read_text(encoding="utf-8")

    assert "sk-live-secret-value" not in payload
    assert "do-not-leak" not in payload
    assert "x" * 1000 not in payload
    assert len(payload.encode("utf-8")) < 64_000


def test_declared_not_applicable_join_requires_strategy_contract(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    bot_dir = curated / "2026-05-02" / "bot1"
    _write_jsonl(bot_dir / "trades.jsonl", [_base_trade()])
    contract_path = tmp_path / "strategy_plugin_contract.json"
    _write_json(contract_path, {"not_applicable_learning_checks": ["order_to_fill_join"]})

    without_contract = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_1.json",
    )
    with_contract = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_2.json",
        strategy_contract_path=contract_path,
    )

    assert without_contract.join_coverage["order_to_fill_join"].status == CoverageStatus.MISSING
    assert with_contract.join_coverage["order_to_fill_join"].status == CoverageStatus.NOT_APPLICABLE


def test_learning_sufficiency_does_not_borrow_other_strategy_evidence(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    other_trade = _base_trade(strategy_id="other-strategy")
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [other_trade])
    _write_jsonl(raw / "2026-05-02" / "bot1" / "pipeline_funnel.jsonl", [{
        "event_type": "pipeline_funnel",
        "strategy_id": "other-strategy",
        "setups_seen": 10,
        "entries_attempted": 3,
    }])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.event_counts_by_type.get("trade", 0) == 0
    assert manifest.event_counts_by_type.get("pipeline_funnel", 0) == 0
    assert manifest.denominator_coverage["denominator_coverage"].status == CoverageStatus.MISSING


def test_gross_as_net_fallback_is_not_after_cost_authoritative(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    trade = _base_trade()
    trade.pop("net_pnl", None)
    trade.pop("net_pnl_source", None)
    trade.pop("after_cost_status", None)
    trade["fees_paid"] = 0.0
    trade["total_fees"] = 0.0
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [trade])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.after_cost_coverage.status == CoverageStatus.MISSING
    assert manifest.after_cost_coverage.reason == "gross_as_net_or_missing_cost_fields"


def test_observed_after_cost_status_without_net_pnl_is_not_authoritative(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    trade = _base_trade()
    trade.pop("net_pnl", None)
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [trade])

    manifest = LearningSufficiencyAuditor(
        curated,
        tmp_path / "memory" / "findings",
        raw_data_dir=raw,
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
    )

    assert manifest.after_cost_coverage.status == CoverageStatus.MISSING
    assert manifest.after_cost_coverage.reason == "observed_status_without_numeric_net_pnl"
