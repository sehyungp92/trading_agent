from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from trading_assistant.orchestrator.learning_sufficiency_audit import LearningSufficiencyAuditor
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.skills.monthly_candidate_pipeline import MonthlyCandidatePipeline
from tests.test_learning_sufficiency_auditor import _complete_fixture, _write_jsonl


def _monthly_result(manifest_path: Path) -> MonthlyValidationResult:
    return MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.EXPERIMENT,
        learning_sufficiency_manifest_path=str(manifest_path),
    )


def test_execution_candidate_fails_when_learning_joins_are_insufficient(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    output_path = tmp_path / "learning_sufficiency_manifest.json"
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [{
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "net_pnl": 1.0,
        "net_pnl_source": "observed_broker_statement",
        "after_cost_status": "observed",
    }])
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

    gate = MonthlyCandidatePipeline()._learning_sufficiency_gate(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="execution_change"),
        _monthly_result(output_path),
    )

    assert gate.passed is False
    assert "execution_learning" in gate.reason

    gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="execution_change"),
        _monthly_result(output_path),
    )
    by_name = {gate.name: gate for gate in gates}
    assert set(by_name) == {
        "learning_sufficiency_manifest_present",
        "learning_capability_authority",
        "causal_join_completeness",
        "denominator_coverage",
        "after_cost_outcome_coverage",
        "proposal_trace_coverage",
        "counterfactual_backfill_coverage",
        "runtime_evidence_coverage",
        "instrumentation_gap_impact",
    }
    assert by_name["learning_sufficiency_manifest_present"].passed is True
    assert by_name["learning_capability_authority"].passed is False
    assert by_name["causal_join_completeness"].passed is False
    assert by_name["runtime_evidence_coverage"].passed is False
    assert by_name["instrumentation_gap_impact"].passed is False
    assert "execution_learning" in by_name["instrumentation_gap_impact"].reason
    assert by_name["after_cost_outcome_coverage"].passed is True


def test_rollback_candidate_does_not_require_capability_specific_learning(tmp_path: Path) -> None:
    gate = MonthlyCandidatePipeline()._learning_sufficiency_gate(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="rollback"),
        _monthly_result(tmp_path / "missing.json"),
    )

    assert gate.passed is True
    gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="rollback"),
        _monthly_result(tmp_path / "missing.json"),
    )
    assert all(gate.passed for gate in gates)


def test_filter_threshold_candidate_requires_counterfactuals_and_trade_joins(tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    raw = tmp_path / "raw"
    output_path = tmp_path / "learning_sufficiency_manifest.json"
    _write_jsonl(curated / "2026-05-02" / "bot1" / "trades.jsonl", [{
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "decision_id": "decision-1",
        "net_pnl": 1.0,
        "net_pnl_source": "observed_broker_statement",
        "after_cost_status": "observed",
    }])
    _write_jsonl(raw / "2026-05-02" / "bot1" / "filter_decision.jsonl", [{
        "strategy_id": "strat1",
        "decision_id": "decision-1",
        "filter_name": "rsi",
        "threshold": 55,
        "actual_value": 60,
        "passed": True,
    }])
    _write_jsonl(raw / "2026-05-02" / "bot1" / "pipeline_funnel.jsonl", [{
        "strategy_id": "strat1",
        "setups_seen": 10,
        "entries_attempted": 3,
    }])
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

    gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="filter_threshold_change"),
        _monthly_result(output_path),
    )
    by_name = {gate.name: gate for gate in gates}

    assert by_name["learning_capability_authority"].passed is False
    assert "counterfactual_coverage" in by_name["learning_capability_authority"].reason
    assert by_name["counterfactual_backfill_coverage"].passed is False
    assert by_name["causal_join_completeness"].passed is True


def test_execution_candidate_passes_with_authoritative_execution_learning(tmp_path: Path) -> None:
    curated, raw, metadata_path = _complete_fixture(tmp_path)
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
        deployment_metadata_paths=[metadata_path],
        output_path=output_path,
    )

    gate = MonthlyCandidatePipeline()._learning_sufficiency_gate(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="execution_change"),
        _monthly_result(output_path),
    )

    assert gate.passed is True

    gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="execution_change"),
        _monthly_result(output_path),
    )

    assert all(gate.passed for gate in gates)


def test_runtime_gate_rejects_observed_non_authority_source_labels(tmp_path: Path) -> None:
    curated, raw, metadata_path = _complete_fixture(tmp_path)
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
        deployment_metadata_paths=[metadata_path],
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    for support in payload["runtime_evidence_support"].values():
        support["event_value_classifications"] = {
            event_type: "operational_health"
            for event_type in support.get("configured_event_types", [])
        }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="execution_change"),
        _monthly_result(output_path),
    )
    runtime_gate = {gate.name: gate for gate in gates}["runtime_evidence_coverage"]

    assert runtime_gate.passed is False
    assert "runtime_support_not_learning_authority" in runtime_gate.reason


def test_runtime_gate_rejects_non_authority_multisource_join_side(tmp_path: Path) -> None:
    curated, raw, metadata_path = _complete_fixture(tmp_path)
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
        deployment_metadata_paths=[metadata_path],
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["runtime_evidence_support"]["filter_decision"]["event_value_classifications"] = {
        "filter_decision": "operational_health",
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="execution_change"),
        _monthly_result(output_path),
    )
    runtime_gate = {gate.name: gate for gate in gates}["runtime_evidence_coverage"]

    assert runtime_gate.passed is False
    assert (
        "runtime_evidence_support:decision_to_order_join:filter_decision:"
        "runtime_support_not_learning_authority"
    ) in runtime_gate.reason


def test_runtime_gate_accepts_learning_authority_alias_source_labels(tmp_path: Path) -> None:
    curated, raw, metadata_path = _complete_fixture(tmp_path)
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
        deployment_metadata_paths=[metadata_path],
        output_path=output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["runtime_evidence_support"]["fill"]["configured_event_types"] = ["inferred_fill"]
    payload["runtime_evidence_support"]["fill"]["event_value_classifications"] = {
        "inferred_fill": "learning_authority",
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
        MonthlyImprovementCandidate(candidate_id="cand1", change_kind="execution_change"),
        _monthly_result(output_path),
    )
    runtime_gate = {gate.name: gate for gate in gates}["runtime_evidence_coverage"]

    assert runtime_gate.passed is True
