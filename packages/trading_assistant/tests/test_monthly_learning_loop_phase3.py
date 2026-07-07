from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate
from trading_assistant.schemas.monthly_outcome import (
    MonthlyOutcomeRecord,
    MonthlyOutcomeVerdict,
    OutcomeDataSufficiency,
    OutcomeSource,
)
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.schemas.proposal_ledger import ProposalCandidate, ProposalKind, ProposalSource
from trading_assistant.schemas.strategy_change_ledger import StrategyChangeRecord, StrategyChangeRecordType
from trading_assistant.schemas.suggestion_tracking import SuggestionRecord, SuggestionStatus
from trading_assistant.skills.monthly_candidate_pipeline import MonthlyCandidatePipeline
from trading_assistant.skills.monthly_outcome_measurer import MonthlyOutcomeMeasurer
from trading_assistant.skills.monthly_validation_orchestrator import MonthlyValidationOrchestrator
from trading_assistant.skills.outcome_prior_store import OutcomePriorStore
from trading_assistant.skills.proposal_ledger import ProposalLedger
from trading_assistant.skills.rollback_advisor import RollbackAdvisor
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger
from trading_assistant.skills.suggestion_scorer import SuggestionScorer
from trading_assistant.skills.suggestion_tracker import SuggestionTracker


def test_source_aware_mark_measured_does_not_finalize_early_warning(tmp_path: Path) -> None:
    tracker = SuggestionTracker(tmp_path)
    tracker.record(SuggestionRecord(
        suggestion_id="s1",
        bot_id="bot1",
        strategy_id="strat1",
        title="Adjust filter",
        tier="parameter",
        category="filter_threshold",
        source_report_id="weekly-1",
    ))
    tracker.mark_deployed("s1", deployment_id="dep1")

    tracker.mark_measured("s1", source="early_warning", final=False)
    row = tracker.load_all()[0]
    assert row["status"] == SuggestionStatus.DEPLOYED.value
    assert row["outcome_source"] == "early_warning"

    tracker.mark_measured("s1", source="monthly", outcome_id="mo1", final=True)
    row = tracker.load_all()[0]
    assert row["status"] == SuggestionStatus.MEASURED.value
    assert row["outcome_source"] == "monthly"
    assert row["monthly_outcome_id"] == "mo1"
    assert len(row["outcome_source_history"]) == 2


def test_monthly_verdict_writes_ledgers_followup_and_priors(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    suggestion_tracker = SuggestionTracker(findings)
    suggestion_tracker.record(SuggestionRecord(
        suggestion_id="s1",
        bot_id="bot1",
        strategy_id="strat1",
        title="Rollback harmful filter",
        tier="parameter",
        category="filter_threshold",
        source_report_id="monthly-2026-04",
        proposal_id="p1",
    ))
    suggestion_tracker.mark_deployed("s1", deployment_id="dep1")

    proposal_ledger = ProposalLedger(findings)
    proposal_ledger.record_candidate(ProposalCandidate(
        proposal_id="p1",
        source=ProposalSource.MONTHLY_SMOKE_REPAIR,
        kind=ProposalKind.ROLLBACK,
        bot_id="bot1",
        strategy_id="strat1",
        title="Rollback harmful filter",
        suggestion_id="s1",
    ))

    strategy_ledger = StrategyChangeLedger(findings)
    change = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.DEPLOYED_CHANGE,
        source_proposal_ids=["p1"],
        source_suggestion_ids=["s1"],
        deployment_id="dep1",
    )
    strategy_ledger.record(change)

    outcome = MonthlyOutcomeRecord(
        source=OutcomeSource.MONTHLY,
        bot_id="bot1",
        strategy_id="strat1",
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        strategy_change_record_id=change.record_id,
        deployment_id="dep1",
        proposal_ids=["p1"],
        suggestion_ids=["s1"],
        mutation_family="accepted_prefix_rollback",
        category="rollback",
        verdict=MonthlyOutcomeVerdict.ROLLBACK,
        live_vs_expected_objective_delta=-0.2,
        drawdown_delta=0.18,
        confidence=0.9,
        data_sufficiency=OutcomeDataSufficiency.SUFFICIENT,
        evidence_paths=["monthly_validation_result.json"],
    )

    measurer = MonthlyOutcomeMeasurer(
        findings,
        suggestion_tracker=suggestion_tracker,
        proposal_ledger=proposal_ledger,
        strategy_change_ledger=strategy_ledger,
    )
    measurer.record_monthly_verdict(outcome)
    measurer.record_monthly_verdict(outcome)

    assert (findings / "monthly_outcomes.jsonl").exists()
    assert (findings / "monthly_outcome_followups.jsonl").exists()
    assert len(measurer.load_outcomes()) == 1
    suggestion = suggestion_tracker.load_all()[0]
    assert suggestion["status"] == "measured"
    assert suggestion["outcome_source"] == "monthly"
    assert len(suggestion["outcome_source_history"]) == 1
    assert len(suggestion_tracker.load_outcomes()) == 1
    assert suggestion_tracker.load_outcomes()[0]["outcome_source"] == "monthly"

    proposal = proposal_ledger.get_by_id("p1")
    assert proposal is not None
    assert len(proposal.outcomes) == 1
    assert proposal.outcomes[-1].outcome_source == "monthly"
    assert proposal.outcomes[-1].verdict == "negative"

    updated = strategy_ledger.get_by_id(change.record_id)
    assert updated is not None
    assert updated.monthly_verdict is not None
    assert updated.monthly_verdict["verdict"] == "rollback"

    prior = OutcomePriorStore(findings).get_prior(
        bot_id="bot1",
        strategy_id="strat1",
        mutation_family="accepted_prefix_rollback",
        category="rollback",
    )
    assert prior is not None
    assert prior.negative_count == 1
    assert prior.gate_strictness.value == "stricter"
    assert prior.rollback_priority.value == "critical"

    rollback_rows = [
        json.loads(line)
        for line in (findings / "rollback_recommendations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rollback_rows) == 1


def test_context_builder_separates_monthly_outcomes_and_early_warnings(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    findings = memory / "findings"
    findings.mkdir(parents=True)
    (findings / "outcomes.jsonl").write_text(
        json.dumps({"suggestion_id": "legacy", "verdict": "positive", "measurement_quality": "high"}) + "\n",
        encoding="utf-8",
    )
    (findings / "monthly_outcomes.jsonl").write_text(
        MonthlyOutcomeRecord(
            bot_id="bot1",
            strategy_id="strat1",
            run_month="2026-04",
            verdict=MonthlyOutcomeVerdict.KEEP,
            confidence=0.9,
            data_sufficiency=OutcomeDataSufficiency.SUFFICIENT,
        ).model_dump_json() + "\n",
        encoding="utf-8",
    )
    OutcomePriorStore(findings).record_outcome(MonthlyOutcomeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        mutation_family="filter",
        category="filter_threshold",
        verdict=MonthlyOutcomeVerdict.KEEP,
        confidence=0.9,
        data_sufficiency=OutcomeDataSufficiency.SUFFICIENT,
        minimum_trade_count_met=True,
    ))

    ctx = ContextBuilder(memory)
    reliable, _ = ctx.load_outcome_measurements()
    assert reliable[0]["outcome_source"] == "early_warning"

    package = ctx.base_package(bot_id="bot1")
    assert "monthly_outcomes" in package.data
    assert "outcome_priors" in package.data


def test_monthly_keep_counts_as_positive_authoritative_scorecard(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    suggestion_tracker = SuggestionTracker(findings)
    suggestion_tracker.record(SuggestionRecord(
        suggestion_id="s-keep",
        bot_id="bot1",
        strategy_id="strat1",
        title="Keep useful filter change",
        tier="parameter",
        category="filter_threshold",
        source_report_id="monthly-2026-04",
    ))

    MonthlyOutcomeMeasurer(
        findings,
        suggestion_tracker=suggestion_tracker,
    ).record_monthly_verdict(MonthlyOutcomeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        suggestion_ids=["s-keep"],
        mutation_family="filter",
        category="filter_threshold",
        verdict=MonthlyOutcomeVerdict.KEEP,
        live_vs_expected_objective_delta=0.12,
        confidence=0.9,
        data_sufficiency=OutcomeDataSufficiency.SUFFICIENT,
        minimum_trade_count_met=True,
    ))

    scorecard = SuggestionScorer(findings).compute_authoritative_scorecard()
    row = next(score for score in scorecard.scores if score.category == "filter_threshold")
    assert row.win_rate == 1.0
    assert row.avg_pnl_delta == 0.12


def test_negative_monthly_prior_requires_stronger_candidate_evidence(tmp_path: Path) -> None:
    store = OutcomePriorStore(tmp_path)
    store.record_outcome(MonthlyOutcomeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        mutation_family="accepted_prefix_rollback",
        category="rollback",
        verdict=MonthlyOutcomeVerdict.ROLLBACK,
        live_vs_expected_objective_delta=-0.2,
        confidence=0.9,
        data_sufficiency=OutcomeDataSufficiency.SUFFICIENT,
    ))
    pipeline = MonthlyCandidatePipeline(outcome_prior_store=store)
    candidate = MonthlyImprovementCandidate(
        bot_id="bot1",
        strategy_id="strat1",
        family="accepted_prefix_rollback",
        change_kind="rollback",
    )

    gate = pipeline._outcome_prior_gate(candidate)
    assert gate.passed is False
    assert "negative monthly priors" in gate.reason

    flexible_prior = store.get_prior(
        bot_id="bot1",
        strategy_id="strat1",
        mutation_family="",
        category="accepted_prefix_rollback",
    )
    assert flexible_prior is not None
    assert flexible_prior.negative_count == 1

    candidate.deterministic_gate_inputs["stronger_evidence_passed"] = True
    assert pipeline._outcome_prior_gate(candidate).passed is True


def test_non_shadow_monthly_run_closes_deployed_change(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=tmp_path / "curated",
        findings_dir=findings,
        market_data_root=tmp_path / "market",
        backtest_repo_path=tmp_path / "backtest_repo",
        backtest_artifact_root=tmp_path / "artifacts",
    )
    change = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.DEPLOYED_CHANGE,
        deployed_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        deployment_id="dep1",
        mutation_diff={"family": "filter", "change_kind": "filter_threshold"},
        source_proposal_ids=["p1"],
        source_suggestion_ids=["s1"],
    )
    assert orchestrator.ledger.record(change) is True

    result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.ROLLBACK,
        evidence_paths=["monthly_validation_result.json"],
        shadow=False,
    )
    artifact_root = tmp_path / "artifacts" / "bot1" / "2026-04" / "strat1"
    artifact_root.mkdir(parents=True)
    (artifact_root / "objective_breakdown.json").write_text(
        json.dumps({
            "live_vs_expected_objective_delta": -0.2,
            "drawdown_delta": 0.18,
            "minimum_trade_count_met": True,
        }),
        encoding="utf-8",
    )
    artifact_index = BacktestArtifactIndex(
        run_id=result.run_id,
        artifact_root=str(artifact_root),
    )
    orchestrator._record_deployed_change_outcomes(
        result=result,
        window_start=date(2026, 4, 1),
        artifact_index=artifact_index,
    )
    orchestrator._record_deployed_change_outcomes(
        result=result,
        window_start=date(2026, 4, 1),
        artifact_index=artifact_index,
    )

    updated = orchestrator.ledger.get_by_id(change.record_id)
    assert updated is not None
    assert updated.monthly_verdict is not None
    assert updated.monthly_verdict["verdict"] == "rollback"
    assert updated.monthly_verdict["live_vs_expected_objective_delta"] == -0.2
    assert updated.monthly_verdict["drawdown_delta"] == 0.18
    assert updated.monthly_verdict["proposal_ids"] == ["p1"]
    assert updated.monthly_verdict["suggestion_ids"] == ["s1"]

    prior = OutcomePriorStore(findings).get_prior(
        bot_id="bot1",
        strategy_id="strat1",
        mutation_family="filter",
        category="filter_threshold",
    )
    assert prior is not None
    assert prior.negative_count == 1


def test_due_follow_up_verdict_updates_change_and_priors(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=tmp_path / "curated",
        findings_dir=findings,
        market_data_root=tmp_path / "market",
        backtest_repo_path=tmp_path / "backtest_repo",
        backtest_artifact_root=tmp_path / "artifacts",
    )
    change = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.DEPLOYED_CHANGE,
        deployed_at=datetime(2025, 12, 15, tzinfo=timezone.utc),
        deployment_id="dep1",
        mutation_diff={"family": "filter", "change_kind": "filter_threshold"},
        monthly_verdict={"run_month": "2026-01", "verdict": "keep", "outcome_id": "monthly-1"},
    )
    assert orchestrator.ledger.record(change) is True

    result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.KEEP,
        evidence_paths=["monthly_validation_result.json"],
        shadow=False,
    )
    artifact_root = tmp_path / "artifacts" / "bot1" / "2026-04" / "strat1"
    artifact_root.mkdir(parents=True)
    (artifact_root / "objective_breakdown.json").write_text(
        json.dumps({
            "live_vs_expected_objective_delta": 0.11,
            "minimum_trade_count_met": True,
        }),
        encoding="utf-8",
    )
    orchestrator._record_deployed_change_outcomes(
        result=result,
        window_start=date(2026, 4, 1),
        artifact_index=BacktestArtifactIndex(run_id=result.run_id, artifact_root=str(artifact_root)),
    )

    updated = orchestrator.ledger.get_by_id(change.record_id)
    assert updated is not None
    assert updated.follow_up_verdict is not None
    assert updated.follow_up_verdict["source"] == OutcomeSource.FOLLOW_UP.value
    assert updated.follow_up_verdict["persistence_confirmed"] is True

    prior = OutcomePriorStore(findings).get_prior(
        bot_id="bot1",
        strategy_id="strat1",
        mutation_family="filter",
        category="filter_threshold",
    )
    assert prior is not None
    assert prior.confirmed_positive_count == 1


def test_rollback_advisor_is_approval_gated(tmp_path: Path) -> None:
    policy = tmp_path / "rollback_thresholds.yaml"
    policy.write_text(
        "policy_version: rollback_thresholds_v1\n"
        "min_confidence: 0.6\n"
        "rollback_objective_delta: -0.10\n",
        encoding="utf-8",
    )
    advisor = RollbackAdvisor.from_policy_file(policy)
    rec = advisor.recommend(MonthlyOutcomeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        strategy_change_record_id="chg1",
        deployment_id="dep1",
        verdict=MonthlyOutcomeVerdict.REPAIR,
        live_vs_expected_objective_delta=-0.2,
        confidence=0.8,
        data_sufficiency=OutcomeDataSufficiency.SUFFICIENT,
        evidence_paths=["monthly_validation_result.json"],
    ))

    assert rec is not None
    assert rec.action.value == "rollback"
    assert rec.requires_approval is True
    assert rec.source_outcome_id
