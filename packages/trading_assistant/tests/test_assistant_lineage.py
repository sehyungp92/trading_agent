from __future__ import annotations

import json
from datetime import datetime, timezone, date
from pathlib import Path

from trading_instrumentation.approval_metadata import live_deployment_metadata_errors

from trading_assistant.orchestrator.learning_sufficiency_audit import LearningSufficiencyAuditor
from trading_assistant.schemas.events import MissedOpportunityEvent, TradeEvent
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate
from trading_assistant.schemas.monthly_outcome import MonthlyOutcomeRecord
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.schemas.performance_learning_ledger import (
    AuthorityLevel,
    DecisionStage,
    LearningLayer,
    PerformanceLearningRecord,
    PerformanceMetricDeltas,
    PerformanceRecordType,
    SourceCadence,
)
from trading_assistant.schemas.learning_sufficiency import CoverageStatus
from trading_assistant.skills.performance_learning_ledger import validate_performance_learning_records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_monthly_artifacts_derive_assistant_lineage(tmp_path: Path) -> None:
    manifest = MonthlyRunManifest(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        deployment_id="dep1",
        parameter_set_id="ps1",
        proposal_ids=["proposal-1"],
        suggestion_ids=["suggestion-1"],
        latest_month_start="2026-05-01",
        latest_month_end="2026-05-31",
        market_data_manifest_path=str(tmp_path / "market.json"),
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        artifact_root=str(tmp_path),
        monthly_search_brief_id="brief-1",
        source_weekly_signal_ids=["weekly-1"],
    )
    candidate = MonthlyImprovementCandidate.from_raw({
        "candidate_id": "candidate-1",
        "proposal_id": "proposal-2",
        "suggestion_id": "suggestion-2",
        "hypothesis_id": "hypothesis-1",
        "deployment_id": "dep2",
        "parameter_set_id": "ps2",
        "source_weekly_signal_ids": ["weekly-2"],
    })
    outcome = MonthlyOutcomeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        proposal_ids=["proposal-3"],
        suggestion_ids=["suggestion-3"],
        deployment_id="dep3",
        strategy_change_record_id="change-1",
    )
    result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.EXPERIMENT,
        proposal_ids=["proposal-4"],
        strategy_change_record_id="change-2",
    )

    assert manifest.assistant_lineage.weekly_signal_ids == ["weekly-1"]
    assert manifest.assistant_lineage.monthly_search_brief_id == "brief-1"
    assert manifest.assistant_lineage.proposal_ids == ["proposal-1"]
    assert manifest.assistant_lineage.deployment_id == "dep1"
    assert candidate.assistant_lineage.proposal_ids == ["proposal-2"]
    assert candidate.assistant_lineage.hypothesis_ids == ["hypothesis-1"]
    assert outcome.assistant_lineage.monthly_outcome_id == outcome.outcome_id
    assert outcome.assistant_lineage.strategy_change_record_ids == ["change-1"]
    assert result.assistant_lineage.proposal_ids == ["proposal-4"]
    assert result.assistant_lineage.strategy_change_record_ids == ["change-2"]


def test_runtime_events_carry_assistant_lineage_block() -> None:
    trade = TradeEvent(
        trade_id="t1",
        bot_id="bot1",
        strategy_id="strat1",
        pair="BTC/USDT",
        side="LONG",
        entry_time=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 5, 1, 11, 0, tzinfo=timezone.utc),
        entry_price=100.0,
        exit_price=105.0,
        position_size=1.0,
        pnl=5.0,
        pnl_pct=5.0,
        proposal_id="proposal-trade",
        assistant_lineage={"proposal_ids": ["proposal-nested"], "weekly_signal_ids": ["weekly-1"]},
    )
    missed = MissedOpportunityEvent(
        bot_id="bot1",
        strategy_id="strat1",
        pair="BTC/USDT",
        signal="RSI_OVERSOLD",
        proposal_id="proposal-missed",
        deployment_id="dep1",
    )

    assert trade.assistant_lineage.proposal_ids == ["proposal-trade", "proposal-nested"]
    assert trade.assistant_lineage.weekly_signal_ids == ["weekly-1"]
    assert missed.assistant_lineage.proposal_ids == ["proposal-missed"]
    assert missed.assistant_lineage.deployment_id == "dep1"


def test_learning_sufficiency_counts_nested_assistant_lineage(tmp_path: Path) -> None:
    deployment_path = tmp_path / "deployment_metadata.json"
    _write_json(deployment_path, {
        "deployment_id": "dep1",
        "assistant_lineage": {
            "deployment_id": "dep1",
            "proposal_ids": ["proposal-1"],
            "weekly_signal_ids": ["weekly-1"],
        },
    })

    manifest = LearningSufficiencyAuditor(
        tmp_path / "curated",
        tmp_path / "memory" / "findings",
        raw_data_dir=tmp_path / "raw",
    ).build_manifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        telemetry_manifest_path=tmp_path / "telemetry_manifest.json",
        deployment_metadata_paths=[deployment_path],
    )

    assert manifest.proposal_trace_coverage.status == CoverageStatus.PASS


def test_assistant_driven_deployment_metadata_requires_proposal_ids() -> None:
    base = {
        "metadata_source": "vps_live_bot_runtime_deployment_metadata_v1",
        "emission_environment": "paper_vps",
        "emission_context": "paper_vps_startup",
        "emitted_at_utc": "2026-06-01T00:00:00Z",
        "live_runtime_started_at_utc": "2026-06-01T00:00:00Z",
        "runtime_entrypoint": "bot.main",
        "runtime_instance_id": "bot:strategy:sha",
        "runtime_host_fingerprint": "host",
        "source_control_origin": "https://github.com/example/repo",
        "repo_url": "https://github.com/example/repo",
        "source_control_commit_sha": "abc123",
        "deployed_commit_sha": "abc123",
        "source_control_worktree_clean": True,
        "dry_run": False,
        "deployment_id": "dep1",
    }

    missing = live_deployment_metadata_errors({**base, "assistant_driven": True})
    complete = live_deployment_metadata_errors({
        **base,
        "assistant_lineage": {"deployment_id": "dep1", "proposal_ids": ["proposal-1"]},
    })

    assert any("assistant_lineage" in message for message in missing)
    assert not complete


def test_performance_ledger_rejects_measured_assistant_record_without_trace() -> None:
    record = PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope="strat1",
        source_cadence=SourceCadence.MONTHLY,
        learning_layer=LearningLayer.TRADING_AUTHORITY,
        authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
        decision_stage=DecisionStage.MEASURED,
        material_approval_evidence=True,
        realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
    )

    messages = validate_performance_learning_records([record])

    assert any(message.startswith("AM-20") for message in messages)
