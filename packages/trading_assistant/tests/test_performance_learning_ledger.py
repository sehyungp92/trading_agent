from __future__ import annotations

import json
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.schemas.performance_learning_ledger import (
    AuthorityLevel,
    DecisionStage,
    LearningLayer,
    PerformanceLearningRecord,
    PerformanceMetricDeltas,
    PerformanceRecordType,
    SourceCadence,
)
from trading_assistant.schemas.proposal_ledger import (
    ProposalCandidate,
    ProposalKind,
    ProposalSource,
)
from trading_assistant.skills.performance_learning_ledger import (
    PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME,
    PerformanceLearningRefreshMarkerError,
    PerformanceLearningLedgerStore,
    PerformanceLearningProjector,
    _context_from_learning_sufficiency_manifest,
    refresh_performance_learning_projection,
    validate_performance_learning_records,
)
from trading_assistant.skills.proposal_ledger import ProposalLedger
from trading_assistant.testing.performance_learning_fixtures import (
    write_performance_learning_sources as _write_sources,
)


def test_performance_learning_projection_preserves_authority_and_context(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    before = _source_snapshots(findings)

    records = PerformanceLearningProjector(findings).project_to_ledger()
    ledger_path = findings / "performance_learning_ledger.jsonl"
    stored = PerformanceLearningLedgerStore(ledger_path).read()

    assert records
    assert len(stored) == len(records)
    assert _source_snapshots(findings) == before
    assert validate_performance_learning_records(stored) == []
    ledger_text = ledger_path.read_text(encoding="utf-8")
    assert '"return_"' not in ledger_text
    assert '"return"' in ledger_text

    proposal_eval = next(
        record for record in stored
        if record.decision_stage == DecisionStage.APPROVED
        and record.proposal_ids == ["proposal-strat-1"]
    )
    assert proposal_eval.source_cadence == SourceCadence.MONTHLY
    assert proposal_eval.authority_level == AuthorityLevel.MONTHLY_REPLAY_AUTHORITY
    assert proposal_eval.material_approval_evidence is True
    assert proposal_eval.expected_deltas.objective == pytest.approx(0.143)
    assert proposal_eval.realized_after_cost_deltas.has_any() is False
    assert proposal_eval.loop_run_id == "loop-1"
    assert proposal_eval.task_id == "task-1"
    assert proposal_eval.source_weekly_signal_ids == ["weekly-signal-breakout"]
    assert proposal_eval.data_bundle_id == "bundle-source-1"
    assert proposal_eval.scoring_profile == "monthly_optimizer_v1"
    assert proposal_eval.verifier_version == "monthly_evidence_verifier_v1"
    assert proposal_eval.artifact_authority_version == "artifact_authority_registry_v1"

    strategy_outcome = next(
        record for record in stored
        if record.decision_stage == DecisionStage.MEASURED
        and record.strategy_change_record_ids == ["change-1"]
    )
    assert strategy_outcome.material_approval_evidence is True
    assert strategy_outcome.realized_after_cost_deltas.objective == pytest.approx(0.091)
    assert strategy_outcome.expected_deltas.has_any() is False
    assert strategy_outcome.learning_sufficiency_manifest_id == "learning-sufficiency-1"
    assert strategy_outcome.learning_sufficiency_status == "learning_authoritative"
    assert strategy_outcome.supported_learning_capabilities == ["approval_grade_strategy_change"]
    assert strategy_outcome.learning_capability_status == {
        "approval_grade_strategy_change": "learning_authoritative",
    }
    assert strategy_outcome.strategy_slice.regime == "risk_on"
    assert strategy_outcome.strategy_slice.trade_count == 132

    portfolio_record = next(
        record for record in stored
        if record.record_type == PerformanceRecordType.PORTFOLIO
        and record.decision_stage == DecisionStage.FOLLOW_UP
    )
    assert portfolio_record.source_cadence == SourceCadence.FOLLOW_UP
    assert portfolio_record.material_approval_evidence is False
    assert portfolio_record.realized_after_cost_deltas.objective == pytest.approx(0.034)
    assert portfolio_record.portfolio_context.allocation_weights == {
        "strat1": 0.55,
        "mean_reversion": 0.45,
    }
    assert portfolio_record.portfolio_context.correlation == {"strat1:mean_reversion": 0.21}
    assert portfolio_record.portfolio_context.marginal_contribution == {"strat1": 0.018}


def test_performance_learning_context_rejects_non_authority_runtime_sources(tmp_path: Path) -> None:
    path = tmp_path / "learning_sufficiency_manifest.json"
    context = _context_from_learning_sufficiency_manifest({
        "manifest_id": "manifest-ops-runtime",
        "eligibility": "learning_authoritative",
        "supported_learning_capabilities": ["execution_learning"],
        "blocked_learning_capabilities": [],
        "capability_status": {"execution_learning": {"status": "learning_authoritative"}},
        "runtime_evidence_support": {
            "trade": {
                "support_state": "observed",
                "configured_event_types": ["trade"],
                "event_value_classifications": {"trade": "operational_health"},
            }
        },
    }, path)

    assert context["learning_sufficiency_status"] == "diagnostics_only"
    assert context["supported_learning_capabilities"] == []
    assert context["blocked_learning_capability_ids"] == []
    assert context["learning_capability_status"] == {}


def test_performance_learning_context_rejects_missing_runtime_support(tmp_path: Path) -> None:
    path = tmp_path / "learning_sufficiency_manifest.json"
    context = _context_from_learning_sufficiency_manifest({
        "manifest_id": "manifest-legacy",
        "eligibility": "learning_authoritative",
        "supported_learning_capabilities": ["execution_learning"],
        "blocked_learning_capabilities": [],
        "capability_status": {"execution_learning": {"status": "learning_authoritative"}},
    }, path)

    assert context["learning_sufficiency_status"] == "diagnostics_only"
    assert context["supported_learning_capabilities"] == []
    assert context["blocked_learning_capability_ids"] == []
    assert context["learning_capability_status"] == {}


def test_performance_learning_context_loader_is_bounded(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    PerformanceLearningProjector(findings).project_to_ledger()

    summaries = ContextBuilder(tmp_path / "memory").load_recent_performance_learning_entries(
        bot_id="bot1",
        strategy_id="strat1",
        limit=2,
    )

    assert len(summaries) == 2
    assert all(summary["strategy_id"] == "strat1" for summary in summaries)
    assert {summary["source_cadence"] for summary in summaries} <= {"monthly"}
    assert summaries[0]["evidence_paths"]


def test_base_package_includes_performance_learning_context(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    PerformanceLearningProjector(findings).project_to_ledger()

    package = ContextBuilder(tmp_path / "memory").base_package(
        agent_type="weekly_analysis",
        bot_id="bot1",
    )

    assert "performance_learning" in package.data
    assert package.data["performance_learning"]


def test_base_package_includes_portfolio_learning_for_bot_scope(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    PerformanceLearningProjector(findings).project_to_ledger()

    package = ContextBuilder(tmp_path / "memory").base_package(
        agent_type="weekly_analysis",
        bot_id="bot1",
    )

    assert any(
        summary["record_type"] == "portfolio"
        for summary in package.data["performance_learning"]
    )


def test_bot_scoped_performance_learning_keeps_strategy_rows_when_non_bot_rows_are_newer(
    tmp_path: Path,
) -> None:
    ledger = PerformanceLearningLedgerStore(
        tmp_path / "memory" / "findings" / "performance_learning_ledger.jsonl"
    )
    base_time = datetime(2026, 6, 20, tzinfo=timezone.utc)
    strategy_record = PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        bot_id="bot1",
        strategy_id="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.04),
        summary="Bot strategy lesson must remain visible",
        event_time=base_time,
    )
    relevant_portfolio = PerformanceLearningRecord(
        record_id="portfolio-relevant-newer",
        record_type=PerformanceRecordType.PORTFOLIO,
        scope="core_portfolio",
        bot_id="PORTFOLIO",
        portfolio_id="core_portfolio",
        source_cadence=SourceCadence.FOLLOW_UP,
        learning_layer=LearningLayer.PERSISTENCE_CONFIRMATION,
        authority_level=AuthorityLevel.PERSISTENCE_CONFIRMATION,
        decision_stage=DecisionStage.FOLLOW_UP,
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.03),
        summary="Relevant portfolio lesson for strat1",
        event_time=base_time + timedelta(minutes=20),
        portfolio_context={
            "allocation_weights": {"strat1": 0.38, "other_strategy": 0.62},
            "marginal_contribution": {"strat1": 0.021},
        },
    )
    unrelated_portfolio_records = [
        PerformanceLearningRecord(
            record_id=f"portfolio-unrelated-newer-{index}",
            record_type=PerformanceRecordType.PORTFOLIO,
            scope=f"portfolio-{index}",
            bot_id="PORTFOLIO",
            portfolio_id=f"portfolio-{index}",
            source_cadence=SourceCadence.FOLLOW_UP,
            learning_layer=LearningLayer.PERSISTENCE_CONFIRMATION,
            authority_level=AuthorityLevel.PERSISTENCE_CONFIRMATION,
            decision_stage=DecisionStage.FOLLOW_UP,
            realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01 * index),
            summary=f"Unrelated portfolio lesson {index}",
            event_time=base_time + timedelta(minutes=index),
            portfolio_context={
                "allocation_weights": {f"other_strategy_{index}": 0.4},
                "marginal_contribution": {f"other_strategy_{index}": 0.01},
            },
        )
        for index in range(1, 13)
    ]
    unrelated_global_records = [
        PerformanceLearningRecord(
            record_id=f"global-unrelated-newer-{index}",
            record_type=PerformanceRecordType.STRATEGY,
            scope=f"global-strategy-{index}",
            strategy_id=f"global-strategy-{index}",
            source_cadence=SourceCadence.MONTHLY,
            learning_layer=LearningLayer.TRADING_AUTHORITY,
            authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
            decision_stage=DecisionStage.MEASURED,
            realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
            summary=f"Generic strategy lesson {index}",
            event_time=base_time + timedelta(hours=1, minutes=index),
        )
        for index in range(1, 13)
    ]
    ledger.append_records([
        strategy_record,
        relevant_portfolio,
        *unrelated_portfolio_records,
        *unrelated_global_records,
    ])

    summaries = ledger.recent_summaries(bot_id="bot1", limit=10)

    assert any(summary["record_id"] == strategy_record.record_id for summary in summaries)
    assert any(summary["record_id"] == relevant_portfolio.record_id for summary in summaries)
    assert not any(str(summary["record_id"]).startswith("portfolio-unrelated") for summary in summaries)
    assert 1 <= sum(summary["record_type"] == "portfolio" for summary in summaries) <= 3
    assert summaries[0]["record_type"] == "strategy"


def test_performance_learning_schema_rejects_non_monthly_material_authority() -> None:
    with pytest.raises(ValidationError):
        PerformanceLearningRecord(
            record_type=PerformanceRecordType.STRATEGY,
            scope="strat1",
            source_cadence=SourceCadence.WEEKLY,
            learning_layer=LearningLayer.BOUNDED_SEARCH_PRIOR,
            authority_level=AuthorityLevel.ADVISORY_PRIOR,
            decision_stage=DecisionStage.EVALUATED,
            material_approval_evidence=True,
        )


def test_cadence_authority_boundaries_include_daily_weekly_monthly_followup_and_harness() -> None:
    records = [
        PerformanceLearningRecord(
            record_type=PerformanceRecordType.STRATEGY,
            scope="diagnostic",
            source_cadence=SourceCadence.DAILY,
            learning_layer=LearningLayer.SENSOR_CONTEXT,
            authority_level=AuthorityLevel.DIAGNOSTIC,
            decision_stage=DecisionStage.PROPOSED,
        ),
        PerformanceLearningRecord(
            record_type=PerformanceRecordType.STRATEGY,
            scope="weekly-prior",
            source_cadence=SourceCadence.WEEKLY,
            learning_layer=LearningLayer.BOUNDED_SEARCH_PRIOR,
            authority_level=AuthorityLevel.ADVISORY_PRIOR,
            decision_stage=DecisionStage.EVALUATED,
            source_weekly_signal_ids=["weekly-signal-1"],
        ),
        PerformanceLearningRecord(
            record_type=PerformanceRecordType.STRATEGY,
            scope="monthly-authority",
            source_cadence=SourceCadence.MONTHLY,
            learning_layer=LearningLayer.TRADING_AUTHORITY,
            authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
            decision_stage=DecisionStage.MEASURED,
            material_approval_evidence=True,
            realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.03),
        ),
        PerformanceLearningRecord(
            record_type=PerformanceRecordType.PORTFOLIO,
            scope="portfolio-followup",
            source_cadence=SourceCadence.FOLLOW_UP,
            learning_layer=LearningLayer.PERSISTENCE_CONFIRMATION,
            authority_level=AuthorityLevel.PERSISTENCE_CONFIRMATION,
            decision_stage=DecisionStage.FOLLOW_UP,
        ),
        PerformanceLearningRecord(
            record_type=PerformanceRecordType.STRATEGY,
            scope="prompt-benchmark",
            source_cadence=SourceCadence.HARNESS,
            learning_layer=LearningLayer.HARNESS_META_LEARNING,
            authority_level=AuthorityLevel.BENCHMARK_ONLY,
            decision_stage=DecisionStage.EVALUATED,
            material_approval_evidence=False,
        ),
    ]

    assert records[-1].authority_level == AuthorityLevel.BENCHMARK_ONLY
    assert records[-1].material_approval_evidence is False


def test_performance_learning_check_catches_stage_collapsing(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    memory = tmp_path / "memory"
    ledger = memory / "findings" / "performance_learning_ledger.jsonl"
    malformed = PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        expected_deltas=PerformanceMetricDeltas(objective=0.1),
        source_records=[],
    )
    _write_jsonl(ledger, [malformed.model_dump(mode="json")])

    messages = checker.run_checks(memory_dir=memory)

    assert any("outcome-stage record" in message for message in messages)


def test_performance_learning_check_fails_on_invalid_raw_line(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    ledger = tmp_path / "memory" / "findings" / "performance_learning_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    valid = PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.EVALUATED,
    )
    ledger.write_text(valid.model_dump_json() + "\n{bad-json}\n", encoding="utf-8")

    messages = checker.run_checks(memory_dir=tmp_path / "memory")

    assert any("malformed or invalid JSONL row" in message for message in messages)


def test_performance_learning_check_fails_on_invalid_source_ledger_line(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    with (findings / "proposal_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{bad-json}\n")

    messages = checker.run_checks(memory_dir=tmp_path / "memory")

    assert any("malformed or invalid source JSONL row" in message for message in messages)
    with pytest.raises(ValueError):
        PerformanceLearningProjector(findings).build_records()


@pytest.mark.parametrize(
    "source_file",
    [
        "proposal_ledger.jsonl",
        "strategy_change_ledger.jsonl",
        "portfolio_outcomes.jsonl",
        "loop_run_ledger.jsonl",
    ],
)
def test_runtime_refresh_raises_on_malformed_source_rows(tmp_path: Path, source_file: str) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    with (findings / source_file).open("a", encoding="utf-8") as handle:
        handle.write("{bad-json}\n")

    with pytest.raises(ValueError):
        refresh_performance_learning_projection(findings)
    marker = findings / PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME
    assert marker.exists()
    assert source_file in marker.read_text(encoding="utf-8")


def test_source_ledger_caller_quarantines_refresh_failure(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    (findings / "portfolio_outcomes.jsonl").write_text("{bad-json}\n", encoding="utf-8")
    candidate = ProposalCandidate(
        proposal_id="proposal-runtime-failure",
        source=ProposalSource.MONTHLY_MODEL_REVIEW,
        kind=ProposalKind.PARAMETER_CHANGE,
        bot_id="bot1",
        strategy_id="strat1",
        title="Runtime refresh failure marker",
    )

    assert ProposalLedger(findings).record_candidate(candidate) is True

    marker = findings / PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME
    assert marker.exists()
    assert "portfolio_outcomes.jsonl" in marker.read_text(encoding="utf-8")
    assert not (findings / "performance_learning_ledger.jsonl").exists()
    (findings / "portfolio_outcomes.jsonl").write_text("", encoding="utf-8")
    assert refresh_performance_learning_projection(findings) == 1
    assert not marker.exists()


def test_source_ledger_caller_fails_when_refresh_marker_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    (findings / "portfolio_outcomes.jsonl").write_text("{bad-json}\n", encoding="utf-8")
    candidate = ProposalCandidate(
        proposal_id="proposal-marker-failure",
        source=ProposalSource.MONTHLY_MODEL_REVIEW,
        kind=ProposalKind.PARAMETER_CHANGE,
        bot_id="bot1",
        strategy_id="strat1",
        title="Runtime marker failure",
    )
    original_write_text = Path.write_text

    def fail_marker_write(path: Path, *args, **kwargs):
        if path.name == PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME:
            raise OSError("marker path is not writable")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_marker_write)

    with pytest.raises(PerformanceLearningRefreshMarkerError):
        ProposalLedger(findings).record_candidate(candidate)


def test_portfolio_outcome_schema_rejects_empty_rows(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    (findings / "portfolio_outcomes.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid portfolio outcome row"):
        PerformanceLearningProjector(findings).build_records()
    messages = checker.run_checks(memory_dir=tmp_path / "memory")

    assert any("malformed or invalid source JSONL row" in message for message in messages)


@pytest.mark.parametrize("bad_delta", [True, "nan", "inf", "-inf"])
def test_portfolio_outcome_schema_rejects_bool_or_nonfinite_deltas(
    tmp_path: Path,
    bad_delta,
) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    row = json.loads((findings / "portfolio_outcomes.jsonl").read_text(encoding="utf-8"))
    row["composite_delta"] = bad_delta
    _write_jsonl(findings / "portfolio_outcomes.jsonl", [row])

    with pytest.raises(ValueError, match="composite_delta must be finite numeric"):
        PerformanceLearningProjector(findings).build_records()


def test_performance_learning_check_fails_on_refresh_error_marker(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    PerformanceLearningProjector(findings).project_to_ledger()
    (findings / PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME).write_text(
        json.dumps({"status": "failed", "error": "source row invalid"}),
        encoding="utf-8",
    )

    messages = checker.run_checks(memory_dir=tmp_path / "memory")

    assert any("runtime refresh failure marker exists" in message for message in messages)


def test_performance_learning_check_fails_without_backing_source_ledgers(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    source_findings = tmp_path / "source" / "memory" / "findings"
    static_findings = tmp_path / "static" / "memory" / "findings"
    source_findings.mkdir(parents=True)
    static_findings.mkdir(parents=True)
    _write_sources(source_findings)
    PerformanceLearningProjector(source_findings).project_to_ledger()
    static_ledger = static_findings / "performance_learning_ledger.jsonl"
    static_ledger.write_text(
        (source_findings / "performance_learning_ledger.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    messages = checker.run_checks(memory_dir=tmp_path / "static" / "memory")

    assert any("not backed by current source ledgers" in message for message in messages)
    assert any("projector produced no current records" in message for message in messages)


def test_strategy_change_update_keeps_stable_projection_identity(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    first_records = PerformanceLearningProjector(findings).project_to_ledger()
    original = next(
        record for record in first_records
        if record.strategy_change_record_ids == ["change-1"]
        and record.decision_stage == DecisionStage.MEASURED
    )
    _append_jsonl(findings / "strategy_change_ledger.jsonl", {
        "type": "update",
        "payload": {
            "record_id": "change-1",
            "decision_reason": "Approval packet passed verifier after revised costs.",
            "monthly_verdict": {
                "verdict": "improved",
                "summary": "Revised after-cost outcome improved.",
                "objective_delta": 0.123,
                "return_delta": 0.035,
                "drawdown_delta": -0.014,
                "cost_delta": -0.003,
                "evidence_paths": ["artifacts/monthly_outcome.json"],
                "strategy_slice": {
                    "regime": "risk_on",
                    "symbol": "BTC",
                    "session": "us",
                    "side": "long",
                    "liquidity": "high",
                    "sample_size": 240,
                    "trade_count": 132,
                    "cost_bps": 6.4,
                    "failure_mode": "late_breakout_noise",
                },
            },
            "updated_at": "2026-07-02T10:00:00+00:00",
        },
    })

    updated_records = PerformanceLearningProjector(findings).project_to_ledger()
    updated = next(
        record for record in updated_records
        if record.strategy_change_record_ids == ["change-1"]
        and record.decision_stage == DecisionStage.MEASURED
    )
    history = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read_history(strict=True)
    latest = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read(strict=True)

    assert updated.record_id == original.record_id
    assert [
        record.record_id for record in history
        if record.record_id == original.record_id
    ] == [original.record_id, original.record_id]
    assert next(record for record in latest if record.record_id == original.record_id).realized_after_cost_deltas.objective == pytest.approx(0.123)
    assert checker.run_checks(memory_dir=tmp_path / "memory") == []


def test_proposal_evaluation_correction_keeps_stable_projection_identity(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    first_records = PerformanceLearningProjector(findings).project_to_ledger()
    original = next(
        record for record in first_records
        if record.source_records[0].kind == "proposal_evaluation"
    )
    rows = [
        json.loads(line)
        for line in (findings / "proposal_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row["type"] == "evaluation":
            row["payload"].update({
                "decision": "approved",
                "evaluated_at": "2026-06-21T10:30:00+00:00",
                "summary": "Replay gates passed after corrected timing",
                "objective_score": 0.155,
            })
    _write_jsonl(findings / "proposal_ledger.jsonl", rows)

    updated_records = PerformanceLearningProjector(findings).project_to_ledger()
    updated = next(
        record for record in updated_records
        if record.source_records[0].kind == "proposal_evaluation"
    )
    history = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read_history(strict=True)
    latest = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read(strict=True)

    assert updated.record_id == original.record_id
    assert [
        record.record_id for record in history
        if record.record_id == original.record_id
    ] == [original.record_id, original.record_id]
    assert next(
        record for record in latest
        if record.record_id == original.record_id
    ).expected_deltas.objective == pytest.approx(0.155)
    assert checker.run_checks(memory_dir=tmp_path / "memory") == []


def test_proposal_outcome_correction_keeps_stable_fallback_identity(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    outcome = {
        "type": "outcome",
        "payload": {
            "proposal_id": "proposal-strat-1",
            "deployment_id": "deploy-1",
            "objective_delta": 0.071,
            "objective_version": "objective_weights_v1",
            "verdict": "improved",
            "outcome_source": "monthly_replay",
            "strategy_change_record_id": "change-1",
            "measured_at": "2026-06-22T10:00:00+00:00",
        },
    }
    _append_jsonl(findings / "proposal_ledger.jsonl", outcome)
    first_records = PerformanceLearningProjector(findings).project_to_ledger()
    original = next(
        record for record in first_records
        if record.source_records[0].kind == "proposal_outcome"
    )
    rows = [
        json.loads(line)
        for line in (findings / "proposal_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["payload"].update({
        "verdict": "regressed",
        "objective_delta": -0.022,
        "measured_at": "2026-06-23T10:00:00+00:00",
    })
    _write_jsonl(findings / "proposal_ledger.jsonl", rows)

    updated_records = PerformanceLearningProjector(findings).project_to_ledger()
    updated = next(
        record for record in updated_records
        if record.source_records[0].kind == "proposal_outcome"
    )
    history = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read_history(strict=True)
    latest = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read(strict=True)

    assert updated.record_id == original.record_id
    assert [
        record.record_id for record in history
        if record.record_id == original.record_id
    ] == [original.record_id, original.record_id]
    latest_record = next(record for record in latest if record.record_id == original.record_id)
    assert latest_record.verdict == "regressed"
    assert latest_record.realized_after_cost_deltas.objective == pytest.approx(-0.022)
    assert checker.run_checks(memory_dir=tmp_path / "memory") == []


def test_proposal_outcome_and_strategy_verdict_keep_distinct_current_ids(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    _append_jsonl(findings / "proposal_ledger.jsonl", {
        "type": "outcome",
        "payload": {
            "proposal_id": "proposal-strat-1",
            "deployment_id": "deploy-1",
            "objective_delta": 0.071,
            "objective_version": "objective_weights_v1",
            "verdict": "improved",
            "measurement_path": "artifacts/monthly_outcome.json",
            "outcome_source": "monthly_replay",
            "strategy_change_record_id": "change-1",
            "measured_at": "2026-06-22T10:00:00+00:00",
        },
    })

    records = PerformanceLearningProjector(findings).project_to_ledger()
    latest = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read(strict=True)
    measured_change_records = [
        record for record in records
        if record.decision_stage == DecisionStage.MEASURED
        and record.strategy_change_record_ids == ["change-1"]
    ]

    assert len(measured_change_records) == 2
    assert len({record.record_id for record in measured_change_records}) == 2
    assert len(latest) == len(records)
    assert checker.run_checks(memory_dir=tmp_path / "memory") == []


def test_same_proposal_multiple_evaluations_and_outcomes_keep_distinct_ids(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    _append_jsonl(findings / "proposal_ledger.jsonl", {
        "type": "evaluation",
        "payload": {
            "proposal_id": "proposal-strat-1",
            "method": "monthly_replay",
            "summary": "Second replay also passed with tighter costs",
            "objective_score": 0.151,
            "confidence": 0.81,
            "decision": "approve",
            "decision_reason": "Second append-only evaluation event.",
            "objective_version": "objective_weights_v1",
            "evidence_paths": ["artifacts/second_eval.json"],
            "evaluated_at": "2026-06-21T09:00:00+00:00",
        },
    })
    for index, objective_delta in enumerate((0.071, 0.083), start=1):
        _append_jsonl(findings / "proposal_ledger.jsonl", {
            "type": "outcome",
            "payload": {
                "proposal_id": "proposal-strat-1",
                "deployment_id": "deploy-1",
                "objective_delta": objective_delta,
                "objective_version": "objective_weights_v1",
                "verdict": "improved",
                "measurement_path": "artifacts/monthly_outcome.json",
                "outcome_source": "monthly_replay",
                "monthly_outcome_id": f"monthly-outcome-{index}",
                "strategy_change_record_id": "change-1",
                "measured_at": "2026-06-22T10:00:00+00:00",
            },
        })

    records = PerformanceLearningProjector(findings).project_to_ledger()
    latest = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read(strict=True)
    evaluation_records = [
        record for record in records
        if record.source_records[0].kind == "proposal_evaluation"
    ]
    outcome_records = [
        record for record in records
        if record.source_records[0].kind == "proposal_outcome"
    ]

    assert len(evaluation_records) == 2
    assert len({record.record_id for record in evaluation_records}) == 2
    assert len(outcome_records) == 2
    assert len({record.record_id for record in outcome_records}) == 2
    assert len(latest) == len(records)
    assert checker.run_checks(memory_dir=tmp_path / "memory") == []


def test_decision_stage_records_do_not_absorb_hindsight_context(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)

    records = PerformanceLearningProjector(findings).project_to_ledger()
    decision_records = [
        record for record in records
        if record.decision_stage
        in {
            DecisionStage.PROPOSED,
            DecisionStage.EVALUATED,
            DecisionStage.APPROVED,
            DecisionStage.DEPLOYED,
            DecisionStage.REJECTED,
        }
    ]

    assert decision_records
    assert all(record.intended_learning_effects.outcome_prior_update == "" for record in decision_records)
    assert all(not record.portfolio_context.has_any() for record in decision_records)
    assert all(
        "outcome_priors" not in path and "portfolio_rolling_metrics" not in path
        for record in decision_records
        for path in record.evidence_paths
    )
    assert checker.run_checks(memory_dir=tmp_path / "memory") == []


def test_performance_learning_check_catches_decision_stage_hindsight_context(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    _write_sources(findings)
    records = PerformanceLearningProjector(findings).build_records()
    contaminated = records[0].model_copy(update={
        "intended_learning_effects": records[0].intended_learning_effects.model_copy(update={
            "outcome_prior_update": "leaked prior",
        }),
        "portfolio_context": records[0].portfolio_context.model_copy(update={
            "allocation_weights": {"strat1": 1.0},
        }),
        "evidence_paths": [
            *records[0].evidence_paths,
            "artifacts/outcome_priors_snapshot.json",
        ],
    })
    _write_jsonl(
        findings / "performance_learning_ledger.jsonl",
        [contaminated.model_dump(mode="json"), *[
            record.model_dump(mode="json") for record in records[1:]
        ]],
    )

    messages = checker.run_checks(memory_dir=tmp_path / "memory")

    assert any("outcome-prior learning effects" in message for message in messages)
    assert any("realized portfolio interaction context" in message for message in messages)
    assert any("outcome-only context" in message for message in messages)


def test_performance_learning_check_catches_duplicate_current_ids(tmp_path: Path) -> None:
    checker = _load_performance_learning_check()
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    first = PerformanceLearningRecord(
        record_id="duplicate-current",
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
    )
    second = first.model_copy(update={
        "summary": "distinct current row",
        "realized_after_cost_deltas": PerformanceMetricDeltas(objective=0.02),
    })

    messages = checker._check_projection_freshness([second], [first, second], findings)

    assert any("duplicate current record_id" in message for message in messages)


def test_performance_learning_validator_rejects_measured_record_without_sufficiency_context() -> None:
    record = PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        material_approval_evidence=False,
        proposal_ids=["proposal-1"],
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
    )

    messages = validate_performance_learning_records([record])

    assert any("missing learning-sufficiency context" in message for message in messages)


def test_performance_learning_validator_does_not_treat_deployment_id_alone_as_assistant_driven() -> None:
    record = PerformanceLearningRecord(
        record_id="legacy-deployment-outcome",
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        deployment_id="legacy-human-deploy",
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
    )

    messages = validate_performance_learning_records([record])

    assert not any("missing learning-sufficiency context" in message for message in messages)
    assert not any("missing proposal or strategy-change trace" in message for message in messages)


def test_projection_appends_changed_record_instead_of_dropping_stale_duplicate(tmp_path: Path) -> None:
    ledger = PerformanceLearningLedgerStore(tmp_path / "performance_learning_ledger.jsonl")
    original = PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        strategy_change_record_ids=["change-1"],
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
    )
    changed = original.model_copy(update={
        "realized_after_cost_deltas": PerformanceMetricDeltas(objective=0.05),
    })

    assert ledger.append_records([original]) == 1
    assert ledger.append_records([changed]) == 1
    assert len(ledger.read_history(strict=True)) == 2
    assert ledger.read(strict=True)[0].realized_after_cost_deltas.objective == pytest.approx(0.05)


def test_proposal_ledger_write_refreshes_performance_projection(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    ledger = ProposalLedger(findings)
    candidate = ProposalCandidate(
        proposal_id="proposal-runtime-1",
        source=ProposalSource.MONTHLY_MODEL_REVIEW,
        kind=ProposalKind.PARAMETER_CHANGE,
        bot_id="bot1",
        strategy_id="strat1",
        title="Runtime projection refresh",
        evaluation_method="monthly_replay",
    )

    assert ledger.record_candidate(candidate) is True

    projected = PerformanceLearningLedgerStore(
        findings / "performance_learning_ledger.jsonl"
    ).read(strict=True)
    assert any(record.proposal_ids == ["proposal-runtime-1"] for record in projected)


def test_performance_learning_validator_catches_missing_am14_coverage() -> None:
    messages = validate_performance_learning_records([])

    assert messages
    assert messages[0].startswith("AM-14")


def test_projector_tolerates_missing_optional_links(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    candidate = ProposalCandidate(
        proposal_id="proposal-minimal",
        source=ProposalSource.LLM_DAILY,
        kind=ProposalKind.BUG_FIX,
        bot_id="bot1",
        title="Record diagnostic only",
    )
    _write_jsonl(findings / "proposal_ledger.jsonl", [
        {"type": "candidate", "payload": candidate.model_dump(mode="json")},
    ])

    records = PerformanceLearningProjector(findings).build_records()

    assert len(records) == 1
    assert records[0].source_cadence == SourceCadence.DAILY
    assert records[0].authority_level == AuthorityLevel.DIAGNOSTIC
    assert records[0].material_approval_evidence is False


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def _source_snapshots(findings: Path) -> dict[str, str]:
    return {
        name: (findings / name).read_text(encoding="utf-8")
        for name in [
            "proposal_ledger.jsonl",
            "strategy_change_ledger.jsonl",
            "portfolio_outcomes.jsonl",
            "loop_run_ledger.jsonl",
        ]
    }


def _load_performance_learning_check():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "check_performance_learning_ledger.py"
    spec = importlib.util.spec_from_file_location("check_performance_learning_ledger", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
