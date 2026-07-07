"""Tests for the persisted scheduled run store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading_assistant.orchestrator.scheduled_runs import ScheduledRunStore
from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.orchestrator.loops.monthly_services import ScheduledMonthlyProjection
from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.skills.loop_run_ledger import LoopRunLedgerStore
from trading_assistant.skills.performance_learning_ledger import PerformanceLearningLedgerStore


@pytest.mark.asyncio
async def test_mark_started_and_completed_round_trip(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc)
        await store.mark_started("weekly_summary", "global", scheduled_for)
        await store.mark_completed("weekly_summary", "global", scheduled_for)

        records = await store.get_records("weekly_summary", "global")
        assert len(records) == 1
        assert records[0].status == "completed"
        assert records[0].scheduled_for == scheduled_for
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mark_enqueued_is_not_completed(tmp_path: Path):
    """P1-3: trigger-only cron specs mark_enqueued — handler must signal mark_completed."""
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 6, 0, tzinfo=timezone.utc)
        await store.mark_started("daily_analysis", "global", scheduled_for)
        await store.mark_enqueued("daily_analysis", "global", scheduled_for)

        # Until the handler signals completion, is_completed must return False
        # so catch-up will re-fire the trigger.
        assert await store.is_completed("daily_analysis", "global", scheduled_for) is False
        assert await store.get_status("daily_analysis", "global", scheduled_for) == "enqueued"

        # Handler signals completion → status flips to completed.
        await store.mark_completed("daily_analysis", "global", scheduled_for)
        assert await store.is_completed("daily_analysis", "global", scheduled_for) is True
        assert await store.get_status("daily_analysis", "global", scheduled_for) == "completed"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_handler_scheduled_failure_marks_failed_not_completed(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 6, 0, tzinfo=timezone.utc)
        await store.mark_enqueued("daily_analysis", "global", scheduled_for)

        handler = Handlers.__new__(Handlers)
        handler._scheduled_run_store = store
        action = Action(
            type=ActionType.SPAWN_DAILY_ANALYSIS,
            event_id="daily-1",
            bot_id="scheduler",
            details={
                "__scheduled_run__": {
                    "job_key": "daily_analysis",
                    "scope_key": "global",
                    "scheduled_for": scheduled_for.isoformat(),
                },
            },
        )

        await handler._signal_scheduled_result(
            action,
            success=False,
            error="agent failed",
        )

        assert await store.get_status("daily_analysis", "global", scheduled_for) == "failed"
        assert await store.is_completed("daily_analysis", "global", scheduled_for) is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_handler_scheduled_success_marks_completed(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc)
        await store.mark_enqueued("weekly_summary", "global", scheduled_for)

        handler = Handlers.__new__(Handlers)
        handler._scheduled_run_store = store
        action = Action(
            type=ActionType.SPAWN_WEEKLY_SUMMARY,
            event_id="weekly-1",
            bot_id="scheduler",
            details={
                "__scheduled_run__": {
                    "job_key": "weekly_summary",
                    "scope_key": "global",
                    "scheduled_for": scheduled_for.isoformat(),
                },
            },
        )

        await handler._signal_scheduled_result(action, success=True)

        assert await store.get_status("weekly_summary", "global", scheduled_for) == "completed"
        assert await store.is_completed("weekly_summary", "global", scheduled_for) is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_monthly_handler_projects_runtime_artifacts_after_scheduled_completion(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc)
        await store.mark_enqueued("monthly_validation", "bot:bot1", scheduled_for)

        artifact_root = tmp_path / "monthly" / "bot1" / "strat1"
        artifact_root.mkdir(parents=True)
        paths = {
            "artifact_index": artifact_root / "artifact_index.json",
            "runner_observability": artifact_root / "runner_observability.json",
            "gate": artifact_root / "candidate_gate_report.json",
            "model_validation": artifact_root / "model_review_validation.json",
            "verifier": artifact_root / "monthly_evidence_verification_cand1.json",
            "packet": artifact_root / "approval_packet_cand1.json",
            "summary": artifact_root / "candidate_generation_summary.json",
            "report": artifact_root / "monthly_report.md",
            "manifest": artifact_root / "run_manifest.json",
        }
        for path in paths.values():
            path.write_text("{}" if path.suffix == ".json" else "", encoding="utf-8")
        findings = tmp_path / "memory" / "findings"
        findings.mkdir(parents=True)
        (findings / "proposal_ledger.jsonl").write_text(
            json.dumps({
                "type": "candidate",
                "payload": {
                    "proposal_id": "proposal-cand1",
                    "source": "monthly_model_review",
                    "kind": "parameter_change",
                    "bot_id": "bot1",
                    "strategy_id": "strat1",
                    "title": "scheduled monthly performance projection",
                    "evaluation_method": "monthly_replay",
                    "linked_diagnostics": [str(paths["artifact_index"])],
                    "linked_run_id": "monthly-bot1-strat1-2026-04",
                    "proposed_at": scheduled_for.isoformat(),
                },
            }) + "\n",
            encoding="utf-8",
        )
        result = MonthlyValidationResult(
            run_id="monthly-bot1-strat1-2026-04",
            run_month="2026-04",
            bot_id="bot1",
            strategy_id="strat1",
            status=MonthlyValidationStatus.REPAIR,
            artifact_index_path=str(paths["artifact_index"]),
            monthly_report_path=str(paths["report"]),
            run_manifest_path=str(paths["manifest"]),
            candidate_summary_path=str(paths["summary"]),
            candidate_gate_report_path=str(paths["gate"]),
            model_review_validation_path=str(paths["model_validation"]),
            monthly_evidence_verification_paths=[str(paths["verifier"])],
            approval_packet_paths=[str(paths["packet"])],
            proposal_ids=["proposal-cand1"],
            blocking_reasons=["deployment metadata is shadow"],
            model_review_provider="fixture-provider",
            model_review_model="fixture-model",
            model_review_cost_usd=0.42,
        )

        projection = ScheduledMonthlyProjection(
            scheduled_run_store=store,
            memory_dir=tmp_path / "memory",
        )
        action = Action(
            type=ActionType.SPAWN_MONTHLY_VALIDATION,
            event_id="monthly-1",
            bot_id="bot1",
            details={
                "task_id": "monthly-task-1",
                "__scheduled_run__": {
                    "job_key": "monthly_validation",
                    "scope_key": "bot:bot1",
                    "scheduled_for": scheduled_for.isoformat(),
                },
            },
        )

        await projection.signal_result(action, success=True)
        await projection.project_results(
            action,
            results=[result],
            run_id="monthly-handler-run",
            run_dir=tmp_path / "runs" / "monthly-handler-run",
            duration_ms=1234,
        )

        ledger_path = tmp_path / "memory" / "findings" / "loop_run_ledger.jsonl"
        entries = LoopRunLedgerStore(ledger_path).read()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.status == "completed"
        assert entry.agent_run_id == "monthly-handler-run"
        assert entry.task_id == "monthly-task-1"
        assert entry.provider == "fixture-provider"
        assert entry.model == "fixture-model"
        assert entry.cost_usd == 0.42
        assert entry.duration_ms == 1234
        assert str(paths["gate"]) in entry.output_artifacts
        assert str(paths["packet"]) in entry.approval_packet_paths
        assert entry.proposal_ids == ["proposal-cand1"]
        assert "deployment metadata is shadow" in entry.blocking_reasons
        work_log = (tmp_path / "memory" / "work_log.md").read_text(encoding="utf-8")
        assert "monthly_validation repair for bot1/strat1 2026-04" in work_log
        assert str(paths["gate"]) in work_log
        performance_entries = PerformanceLearningLedgerStore(
            tmp_path / "memory" / "findings" / "performance_learning_ledger.jsonl"
        ).read(strict=True)
        assert any(entry.proposal_ids == ["proposal-cand1"] for entry in performance_entries)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_monthly_handler_projects_multi_strategy_metadata_and_artifacts(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc)
        await store.mark_enqueued("monthly_validation", "bot:bot1", scheduled_for)

        def make_result(strategy_id: str, provider: str, model: str, cost: float) -> MonthlyValidationResult:
            artifact_root = tmp_path / "monthly" / "bot1" / strategy_id
            artifact_root.mkdir(parents=True)
            paths = {
                "artifact_index": artifact_root / "artifact_index.json",
                "runner_observability": artifact_root / "runner_observability.json",
                "gate": artifact_root / "candidate_gate_report.json",
                "model_validation": artifact_root / "model_review_validation.json",
                "verifier": artifact_root / f"monthly_evidence_verification_{strategy_id}.json",
                "packet": artifact_root / f"approval_packet_{strategy_id}.json",
                "summary": artifact_root / "candidate_generation_summary.json",
                "report": artifact_root / "monthly_report.md",
                "manifest": artifact_root / "run_manifest.json",
            }
            for path in paths.values():
                path.write_text("{}" if path.suffix == ".json" else "monthly report", encoding="utf-8")
            return MonthlyValidationResult(
                run_id=f"monthly-bot1-{strategy_id}-2026-04",
                run_month="2026-04",
                bot_id="bot1",
                strategy_id=strategy_id,
                status=MonthlyValidationStatus.REPAIR,
                artifact_index_path=str(paths["artifact_index"]),
                monthly_report_path=str(paths["report"]),
                run_manifest_path=str(paths["manifest"]),
                candidate_summary_path=str(paths["summary"]),
                candidate_gate_report_path=str(paths["gate"]),
                model_review_validation_path=str(paths["model_validation"]),
                monthly_evidence_verification_paths=[str(paths["verifier"])],
                approval_packet_paths=[str(paths["packet"])],
                proposal_ids=[f"proposal-{strategy_id}"],
                blocking_reasons=[f"{strategy_id} shadow metadata"],
                model_review_provider=provider,
                model_review_model=model,
                model_review_cost_usd=cost,
            )

        result1 = make_result("strat1", "provider-a", "model-a", 0.31)
        result2 = make_result("strat2", "provider-b", "model-b", 0.42)

        projection = ScheduledMonthlyProjection(
            scheduled_run_store=store,
            memory_dir=tmp_path / "memory",
        )
        action = Action(
            type=ActionType.SPAWN_MONTHLY_VALIDATION,
            event_id="monthly-1",
            bot_id="bot1",
            details={
                "task_id": "monthly-task-multi",
                "__scheduled_run__": {
                    "job_key": "monthly_validation",
                    "scope_key": "bot:bot1",
                    "scheduled_for": scheduled_for.isoformat(),
                },
            },
        )

        await projection.signal_result(action, success=True)
        await projection.project_results(
            action,
            results=[result1, result2],
            run_id="monthly-handler-run",
            run_dir=tmp_path / "runs" / "monthly-handler-run",
            duration_ms=2345,
        )

        ledger_path = tmp_path / "memory" / "findings" / "loop_run_ledger.jsonl"
        entries = LoopRunLedgerStore(ledger_path).read()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.bot_id == "bot1"
        assert entry.strategy_id == "strat1, strat2"
        assert entry.provider == "provider-a, provider-b"
        assert entry.model == "model-a, model-b"
        assert entry.cost_usd == pytest.approx(0.73)
        assert entry.duration_ms == 2345
        assert entry.proposal_ids == ["proposal-strat1", "proposal-strat2"]
        assert str(Path(result1.candidate_gate_report_path)) in entry.output_artifacts
        assert str(Path(result2.candidate_gate_report_path)) in entry.output_artifacts
        assert str(Path(result1.approval_packet_paths[0])) in entry.approval_packet_paths
        assert str(Path(result2.approval_packet_paths[0])) in entry.approval_packet_paths
        assert str(Path(result1.approval_packet_paths[0])) not in entry.evidence_paths
        assert "strat1 shadow metadata" in entry.blocking_reasons
        assert "strat2 shadow metadata" in entry.blocking_reasons
        work_log = (tmp_path / "memory" / "work_log.md").read_text(encoding="utf-8")
        assert "monthly_validation completed for 2 strategy result(s)" in work_log
        assert str(Path(result1.approval_packet_paths[0])) in work_log
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_baseline_round_trip(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        baseline = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        await store.set_baseline(baseline)
        assert await store.get_baseline() == baseline
    finally:
        await store.close()
