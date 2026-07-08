"""Tests for orchestrator/handlers.py — handler implementations."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_assistant.orchestrator.agent_runner import AgentResult
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.schemas.agent_response import AgentSuggestion, ParsedAnalysis, StructuralProposal
from trading_assistant.schemas.approval import ApprovalRequest
from trading_assistant.schemas.bot_profile import BotConfigProfile
from trading_assistant.schemas.notifications import NotificationPreferences, NotificationPriority
from trading_assistant.schemas.repo_changes import ChangeKind, GitHubIssueResult
from trading_assistant.skills.approval_tracker import ApprovalTracker


@pytest.fixture
def event_stream() -> EventStream:
    return EventStream()


@pytest.fixture
def mock_agent_runner():
    # P2-6: use MagicMock as the base so sync methods (e.g. refresh_run_index,
    # update_preferences) don't return coroutines. Mark only async methods
    # with AsyncMock.
    runner = MagicMock()
    runner.invoke = AsyncMock(return_value=AgentResult(
        response="Analysis complete.",
        run_dir=Path("/tmp/test-run"),
        cost_usd=0.05,
        duration_ms=3000,
        session_id="test-session-123",
        success=True,
    ))
    runner.invoke_with_selection = AsyncMock(return_value=AgentResult(
        response="Analysis complete.",
        run_dir=Path("/tmp/test-run"),
        cost_usd=0.05,
        duration_ms=3000,
        session_id="test-session-123",
        success=True,
    ))
    return runner


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock()
    dispatcher.dispatch = AsyncMock(return_value=[])
    return dispatcher


@pytest.fixture
def handlers(
    tmp_path: Path,
    mock_agent_runner,
    event_stream: EventStream,
    mock_dispatcher,
) -> Handlers:
    return Handlers(
        agent_runner=mock_agent_runner,
        event_stream=event_stream,
        dispatcher=mock_dispatcher,
        notification_prefs=NotificationPreferences(),
        curated_dir=tmp_path / "data" / "curated",
        memory_dir=tmp_path / "memory",
        runs_dir=tmp_path / "runs",
        source_root=tmp_path / "src",
        bots=["bot1", "bot2"],
        heartbeat_dir=tmp_path / "heartbeats",
        failure_log_path=tmp_path / "failure_log.jsonl",
    )


def _make_action(
    action_type: ActionType,
    bot_id: str = "bot1",
    details: dict | None = None,
) -> Action:
    return Action(
        type=action_type,
        event_id="evt-test-001",
        bot_id=bot_id,
        details=details,
    )


class TestDailyAnalysis:
    @pytest.mark.asyncio
    async def test_full_flow_with_passing_quality_gate(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """Quality gate PASS -> assemble -> invoke -> notify."""
        date = "2026-03-02"
        # Create curated data so quality gate passes
        for bot in ["bot1", "bot2"]:
            bot_dir = tmp_path / "data" / "curated" / date / bot
            bot_dir.mkdir(parents=True)
            for f in [
                "summary.json", "winners.json", "losers.json",
                "process_failures.json", "notable_missed.json",
                "regime_analysis.json", "filter_analysis.json",
                "root_cause_summary.json",
                "hourly_performance.json", "slippage_stats.json",
                "factor_attribution.json", "exit_efficiency.json",
            ]:
                (bot_dir / f).write_text("{}")
        # Portfolio risk card
        (tmp_path / "data" / "curated" / date / "portfolio_risk_card.json").write_text("{}")
        # Trade data so minimum-data threshold passes
        for bot in ["bot1", "bot2"]:
            bot_dir = tmp_path / "data" / "curated" / date / bot
            (bot_dir / "trades.jsonl").write_text(
                '{"trade_id":"t1"}\n{"trade_id":"t2"}\n'
            )
        # Memory dir needs to exist for context builder
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)

        action = _make_action(ActionType.SPAWN_DAILY_ANALYSIS, details={"date": date})
        await handlers.handle_daily_analysis(action)

        mock_agent_runner.invoke.assert_called_once()


class TestMonthlyValidation:
    @pytest.mark.asyncio
    async def test_handler_passes_optimizer_sequence_contract_to_monthly_request(
        self,
        tmp_path: Path,
        mock_agent_runner,
        event_stream: EventStream,
        mock_dispatcher,
        monkeypatch,
    ):
        captured = []

        class FakeMonthlyValidationOrchestrator:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self, request):
                captured.append(request)

                class Result:
                    run_month = request.run_month
                    status = SimpleNamespace(value="watch")
                    approval_ready_candidate_count = 0
                    approval_request_ids: list[str] = []

                    def model_dump(self, mode="json"):
                        return {
                            "run_month": self.run_month,
                            "status": self.status.value,
                        }

                return Result()

        import trading_assistant.skills.monthly_validation_orchestrator as monthly_module

        monkeypatch.setattr(
            monthly_module,
            "MonthlyValidationOrchestrator",
            FakeMonthlyValidationOrchestrator,
        )
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            monthly_validation_mode="shadow",
            monthly_optimizer_sequence_enabled=True,
            monthly_backtest_command=["python", "runner.py", "{manifest}"],
            monthly_workflow_contract_path="MONTHLY_OPTIMIZER_WORKFLOW.md",
            monthly_workflow_contract_version="monthly_optimizer_workflow_contract_v1",
        )

        await handlers.handle_monthly_validation(_make_action(
            ActionType.SPAWN_MONTHLY_VALIDATION,
            details={"bot_id": "bot1", "strategy_id": "strat1", "run_month": "2026-04"},
        ))

        assert captured
        request = captured[0]
        assert request.optimizer_sequence_enabled is True
        assert request.backtest_command == ["python", "runner.py", "{manifest}"]
        assert request.workflow_contract_path == "MONTHLY_OPTIMIZER_WORKFLOW.md"
        assert request.workflow_contract_version == "monthly_optimizer_workflow_contract_v1"
        assert request.shadow is True
        progress_events = [
            event for event in event_stream.get_recent()
            if event.event_type == "monthly_validation_progress"
        ]
        assert [event.data["stage"] for event in progress_events] == ["started", "completed"]

    @pytest.mark.asyncio
    async def test_approval_gated_monthly_validation_is_scope_allowlisted(
        self,
        tmp_path: Path,
        mock_agent_runner,
        event_stream: EventStream,
        mock_dispatcher,
        monkeypatch,
    ):
        captured = []

        class FakeMonthlyValidationOrchestrator:
            def __init__(self, **_kwargs):
                pass

            def run(self, request):
                captured.append(request)

                class Result:
                    run_month = request.run_month
                    status = SimpleNamespace(value="watch")
                    approval_ready_candidate_count = 0
                    approval_request_ids: list[str] = []

                    def model_dump(self, mode="json"):
                        return {"run_month": self.run_month, "status": self.status.value}

                return Result()

        import trading_assistant.skills.monthly_validation_orchestrator as monthly_module

        monkeypatch.setattr(
            monthly_module,
            "MonthlyValidationOrchestrator",
            FakeMonthlyValidationOrchestrator,
        )
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            monthly_validation_mode="approval_gated",
            monthly_approval_scope_allowlist=["trading_stock_family"],
        )

        for scope_id in ("trading_stock_family", "trading_momentum_family"):
            await handlers.handle_monthly_validation(_make_action(
                ActionType.SPAWN_MONTHLY_VALIDATION,
                details={
                    "bot_id": "bot1",
                    "strategy_id": scope_id,
                    "approval_scope": scope_id,
                    "run_month": "2026-04",
                    "shadow": False,
                    "approval_evidence_mode": True,
                },
            ))

        assert [request.strategy_id for request in captured] == [
            "trading_stock_family",
            "trading_momentum_family",
        ]
        assert captured[0].shadow is False
        assert captured[0].approval_evidence_mode is True
        assert captured[1].shadow is True
        assert captured[1].approval_evidence_mode is False

    @pytest.mark.asyncio
    async def test_handler_indexes_completed_monthly_artifacts_when_later_strategy_fails(
        self,
        tmp_path: Path,
        mock_agent_runner,
        event_stream: EventStream,
        mock_dispatcher,
        monkeypatch,
    ):
        class FakeRegistry:
            def strategies_for_bot(self, bot_id: str):
                return {"strat1": object(), "strat2": object()}

        class FakeMonthlyValidationOrchestrator:
            def __init__(self, **_kwargs):
                pass

            def run(self, request):
                if request.strategy_id == "strat2":
                    raise RuntimeError("strat2 failed")
                artifact_root = tmp_path / "artifacts" / request.strategy_id
                artifact_root.mkdir(parents=True, exist_ok=True)
                report = artifact_root / "monthly_report.md"
                manifest = artifact_root / "run_manifest.json"
                index = artifact_root / "artifact_index.json"
                for path in (report, manifest, index):
                    path.write_text("{}", encoding="utf-8")

                class Result:
                    run_month = request.run_month
                    run_id = f"monthly-{request.bot_id}-{request.strategy_id}-{request.run_month}"
                    bot_id = request.bot_id
                    strategy_id = request.strategy_id
                    status = SimpleNamespace(value="watch")
                    approval_ready_candidate_count = 0
                    approval_request_ids: list[str] = []
                    monthly_report_path = str(report)
                    run_manifest_path = str(manifest)
                    artifact_index_path = str(index)

                    def model_dump(self, mode="json"):
                        return {
                            "run_id": self.run_id,
                            "run_month": self.run_month,
                            "bot_id": self.bot_id,
                            "strategy_id": self.strategy_id,
                            "status": self.status.value,
                            "monthly_report_path": self.monthly_report_path,
                            "run_manifest_path": self.run_manifest_path,
                            "artifact_index_path": self.artifact_index_path,
                        }

                return Result()

        import trading_assistant.skills.monthly_validation_orchestrator as monthly_module

        monkeypatch.setattr(
            monthly_module,
            "MonthlyValidationOrchestrator",
            FakeMonthlyValidationOrchestrator,
        )
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            strategy_registry=FakeRegistry(),
            run_history_path=tmp_path / "run_history.jsonl",
            monthly_validation_mode="shadow",
        )

        with pytest.raises(RuntimeError, match="strat2 failed"):
            await handlers.handle_monthly_validation(_make_action(
                ActionType.SPAWN_MONTHLY_VALIDATION,
                details={"bot_id": "bot1", "run_month": "2026-04"},
            ))

        run_dir = tmp_path / "runs" / "monthly-bot1-all-2026-04"
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["monthly_artifact_roots"] == [str(tmp_path / "artifacts" / "strat1")]
        history_rows = [
            json.loads(line)
            for line in (tmp_path / "run_history.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        failed = history_rows[-1]
        assert failed["status"] == "failed"
        assert failed["metadata"]["partial_results"] is True

    @pytest.mark.parametrize(
        ("config_limit", "requested_limit", "expected_limit"),
        [(2, 4, 2), (3, 1, 1)],
    )
    @pytest.mark.asyncio
    async def test_handler_caps_or_narrows_parallel_strategy_limit(
        self,
        tmp_path: Path,
        mock_agent_runner,
        event_stream: EventStream,
        mock_dispatcher,
        monkeypatch,
        config_limit: int,
        requested_limit: int,
        expected_limit: int,
    ):
        class FakeRegistry:
            def strategies_for_bot(self, bot_id: str):
                return {"strat1": object(), "strat2": object(), "strat3": object()}

        active = 0
        max_active = 0
        lock = threading.Lock()

        class FakeMonthlyValidationOrchestrator:
            def __init__(self, **_kwargs):
                pass

            def run(self, request):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.05)
                finally:
                    with lock:
                        active -= 1

                class Result:
                    run_month = request.run_month
                    bot_id = request.bot_id
                    strategy_id = request.strategy_id
                    status = SimpleNamespace(value="watch")
                    approval_ready_candidate_count = 0
                    approval_request_ids: list[str] = []

                    def model_dump(self, mode="json"):
                        return {
                            "run_month": self.run_month,
                            "bot_id": self.bot_id,
                            "strategy_id": self.strategy_id,
                            "status": self.status.value,
                        }

                return Result()

        import trading_assistant.skills.monthly_validation_orchestrator as monthly_module

        monkeypatch.setattr(
            monthly_module,
            "MonthlyValidationOrchestrator",
            FakeMonthlyValidationOrchestrator,
        )
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot1"],
            strategy_registry=FakeRegistry(),
            monthly_validation_mode="shadow",
            backtest_max_parallel_strategies=config_limit,
        )

        await handlers.handle_monthly_validation(_make_action(
            ActionType.SPAWN_MONTHLY_VALIDATION,
            details={
                "bot_id": "bot1",
                "run_month": "2026-04",
                "max_parallel_strategies": requested_limit,
            },
        ))

        assert max_active == expected_limit

    @pytest.mark.asyncio
    async def test_quality_gate_degraded_still_invokes_claude(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, event_stream: EventStream,
        tmp_path: Path,
    ):
        """Quality gate FAIL (degraded) -> still invoke the agent runtime with available data."""
        date = "2026-03-02"
        # Create trade data to pass minimum-data threshold
        for bot in ["bot1", "bot2"]:
            bot_dir = tmp_path / "data" / "curated" / date / bot
            bot_dir.mkdir(parents=True, exist_ok=True)
            (bot_dir / "trades.jsonl").write_text(
                '{"trade_id":"t1"}\n{"trade_id":"t2"}\n'
            )
        action = _make_action(ActionType.SPAWN_DAILY_ANALYSIS, details={"date": date})

        # No curated JSON files -> quality gate will report degraded but can_proceed=True
        await handlers.handle_daily_analysis(action)

        # Agent SHOULD be invoked (graceful degradation)
        mock_agent_runner.invoke.assert_called_once()
        assert mock_agent_runner.invoke.call_args.kwargs["allowed_tools"] == ["Read", "Grep", "Glob"]
        # Dispatcher called for the success notification
        assert mock_dispatcher.dispatch.call_count >= 1


class TestWeeklyAnalysis:
    @pytest.mark.asyncio
    async def test_full_flow(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """Metrics -> strategy -> assemble -> invoke."""
        week_start = "2026-02-23"
        week_end = "2026-03-01"
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_WEEKLY_SUMMARY,
            details={"week_start": week_start, "week_end": week_end},
        )
        await handlers.handle_weekly_analysis(action)

        # Agent should be invoked even with empty data (assembler handles gracefully)
        mock_agent_runner.invoke.assert_called_once()
        call_args = mock_agent_runner.invoke.call_args
        assert "weekly" in str(call_args)
        assert call_args.kwargs["allowed_tools"] == ["Read", "Grep", "Glob"]


class TestHandlerProposalLedger:
    def test_transfer_candidate_records_target_strategy(self, handlers: Handlers, tmp_path: Path):
        from trading_assistant.skills.proposal_ledger import ProposalLedger

        ledger = ProposalLedger(tmp_path / "memory" / "findings")
        handlers._proposal_ledger = ledger

        proposal_id = handlers._ledger_write_candidate(
            source="transfer",
            kind_hint="structural_change",
            bot_id="target_bot",
            strategy_id="target_strategy",
            title="Transfer pattern: tighten exits",
            description="Pattern worked on source bot",
            run_id="weekly-1",
            lifecycle_stage="exit",
            evaluation_method="approval",
            stable_link_key="transfer:pattern-1:source_bot:target_bot:target_strategy",
        )

        record = ledger.get_by_id(proposal_id)
        assert record is not None
        assert record.candidate.source.value == "transfer"
        assert record.candidate.bot_id == "target_bot"
        assert record.candidate.strategy_id == "target_strategy"

    def test_validator_rejections_are_recorded_as_rejected_candidates(
        self, handlers: Handlers, tmp_path: Path
    ):
        from trading_assistant.analysis.response_validator import (
            BlockedPortfolioProposal,
            BlockedStructuralProposal,
            BlockedSuggestion,
            ValidationResult,
        )
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposal, PortfolioProposalType
        from trading_assistant.skills.proposal_ledger import ProposalLedger

        ledger = ProposalLedger(tmp_path / "memory" / "findings")
        handlers._proposal_ledger = ledger
        validation = ValidationResult(
            blocked_suggestions=[
                BlockedSuggestion(
                    suggestion=AgentSuggestion(
                        suggestion_id="blocked-agent-suggestion",
                        bot_id="bot1",
                        strategy_id="alpha",
                        category="exit_timing",
                        title="Rejected exit tweak",
                        confidence=0.8,
                        evidence_summary="weak sample",
                        target_param="exit_threshold",
                    ),
                    reason="insufficient evidence",
                ),
            ],
            blocked_structural_proposals=[
                BlockedStructuralProposal(
                    proposal=StructuralProposal(
                        bot_id="bot1",
                        title="Rejected structural change",
                        description="needs code changes",
                        confidence=0.6,
                        acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
                    ),
                    reason="malformed acceptance criteria",
                ),
            ],
            blocked_portfolio_proposals=[
                BlockedPortfolioProposal(
                    proposal=PortfolioProposal(
                        proposal_type=PortfolioProposalType.RISK_CAP_CHANGE,
                        evidence_summary="too broad",
                        confidence=0.7,
                    ),
                    reason="risk cap loosening blocked",
                ),
            ],
        )

        handlers._record_rejected_validation_proposals(
            validation,
            run_id="daily-2026-03-01",
            source="llm_daily",
        )

        records = ledger.list_all()
        assert len(records) == 3
        assert {r.evaluations[0].decision for r in records} == {"reject"}
        assert any(r.candidate.strategy_id == "alpha" for r in records)
        assert any(
            r.candidate.suggestion_id == "blocked-agent-suggestion"
            for r in records
        )
        assert any(r.candidate.bot_id == "PORTFOLIO" for r in records)

    def test_blocked_structural_proposals_do_not_create_experiments(
        self, handlers: Handlers, tmp_path: Path
    ):
        from trading_assistant.analysis.response_validator import (
            BlockedStructuralProposal,
            ValidationResult,
        )
        from trading_assistant.skills.proposal_ledger import ProposalLedger

        blocked = StructuralProposal(
            bot_id="bot1",
            title="Blocked structural change",
            description="Has criteria but validator rejected it",
            confidence=0.6,
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        handlers._proposal_ledger = ProposalLedger(tmp_path / "memory" / "findings")
        handlers._structural_experiment_tracker = MagicMock()

        handlers._record_agent_suggestions(
            ValidationResult(
                blocked_structural_proposals=[
                    BlockedStructuralProposal(
                        proposal=blocked,
                        reason="validator rejected",
                    ),
                ],
            ),
            run_id="weekly-2026-03-01",
            parsed=ParsedAnalysis(structural_proposals=[blocked]),
        )

        assert handlers._proposal_ledger.list_all() == []
        handlers._structural_experiment_tracker.record_experiment.assert_not_called()

    def test_blocked_structural_context_not_attached_to_approved_suggestion(
        self, handlers: Handlers, tmp_path: Path
    ):
        from trading_assistant.analysis.response_validator import (
            BlockedStructuralProposal,
            ValidationResult,
        )
        from trading_assistant.skills.proposal_ledger import ProposalLedger
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker

        blocked = StructuralProposal(
            linked_suggestion_id="approved-linked",
            bot_id="bot1",
            title="Blocked linked implementation",
            description="This implementation was rejected by the validator",
            confidence=0.6,
            acceptance_criteria=[{"metric": "pnl", "direction": "improve"}],
        )
        handlers._proposal_ledger = ProposalLedger(tmp_path / "memory" / "findings")
        handlers._suggestion_tracker = SuggestionTracker(tmp_path / "memory" / "findings")
        handlers._structural_experiment_tracker = MagicMock()

        handlers._record_agent_suggestions(
            ValidationResult(
                approved_suggestions=[
                    AgentSuggestion(
                        suggestion_id="approved-linked",
                        bot_id="bot1",
                        category="structural",
                        title="Approved structural hypothesis",
                        confidence=0.8,
                        evidence_summary="The hypothesis is valid",
                    ),
                ],
                blocked_structural_proposals=[
                    BlockedStructuralProposal(
                        proposal=blocked,
                        reason="validator rejected implementation",
                    ),
                ],
            ),
            run_id="weekly-2026-03-01",
            parsed=ParsedAnalysis(structural_proposals=[blocked]),
        )

        suggestions = handlers._suggestion_tracker.load_all()
        assert len(suggestions) == 1
        assert suggestions[0].get("implementation_context") in (None, {})
        records = handlers._proposal_ledger.list_all()
        assert len(records) == 1
        assert records[0].candidate.title == "Approved structural hypothesis"
        handlers._structural_experiment_tracker.record_experiment.assert_not_called()


class TestTriage:
    @pytest.mark.asyncio
    async def test_known_fix_invokes_claude(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """KNOWN_FIX outcome -> context builder -> assembler -> invoke."""
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_TRIAGE,
            bot_id="bot1",
            details={
                "bot_id": "bot1",
                "error_type": "ImportError",
                "message": "No module named 'foo'",
                "stack_trace": "Traceback...\nImportError: No module named 'foo'",
                "source_file": "",
                "source_line": 0,
                "severity": "high",
            },
        )

        from trading_assistant.schemas.bug_triage import (
            BugComplexity, BugSeverity, ErrorEvent, TriageOutcome, TriageResult,
        )

        mock_triage_result = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1",
                error_type="ImportError",
                message="No module named 'foo'",
                stack_trace="Traceback...",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
        )

        with (
            patch("trading_assistant.skills.run_bug_triage.TriageRunner") as MockRunner,
            patch("trading_assistant.skills.failure_log.FailureLog") as MockLog,
        ):
            MockRunner.return_value.triage.return_value = mock_triage_result
            MockLog.return_value.get_past_rejections.return_value = []
            await handlers.handle_triage(action)

        mock_agent_runner.invoke.assert_called_once()
        call_args = mock_agent_runner.invoke.call_args
        assert "triage" in str(call_args)
        assert call_args.kwargs["allowed_tools"] == ["Read", "Bash", "Grep", "Glob"]

    @pytest.mark.asyncio
    async def test_needs_human_skips_claude_but_notifies(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """NEEDS_HUMAN -> no agent-runtime invocation, but notify."""
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_TRIAGE,
            bot_id="bot1",
            details={
                "bot_id": "bot1",
                "error_type": "RuntimeError",
                "message": "Complex state issue",
                "stack_trace": "Traceback...",
            },
        )

        from trading_assistant.schemas.bug_triage import (
            BugComplexity, BugSeverity, ErrorEvent, TriageOutcome, TriageResult,
        )

        mock_result = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1",
                error_type="RuntimeError",
                message="Complex state issue",
                stack_trace="Traceback...",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.STATE_DEPENDENT,
            outcome=TriageOutcome.NEEDS_HUMAN,
        )

        with (
            patch("trading_assistant.skills.run_bug_triage.TriageRunner") as MockRunner,
            patch("trading_assistant.skills.failure_log.FailureLog") as MockLog,
        ):
            MockRunner.return_value.triage.return_value = mock_result
            MockLog.return_value.get_past_rejections.return_value = []
            await handlers.handle_triage(action)

        # Agent should NOT be invoked
        mock_agent_runner.invoke.assert_not_called()
        # But notification should be sent
        mock_dispatcher.dispatch.assert_called_once()
        payload = mock_dispatcher.dispatch.call_args[0][0]
        assert "needs_human" in payload.notification_type

    @pytest.mark.asyncio
    async def test_known_fix_creates_bug_fix_request(self, tmp_path: Path, mock_agent_runner, mock_dispatcher, event_stream: EventStream):
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        approval_handler = MagicMock()
        approval_handler.handle_approve = AsyncMock(return_value="PR created: https://github.com/x/y/pull/1")
        config_registry = MagicMock()
        config_registry.get_profile.return_value = BotConfigProfile(
            bot_id="bot1",
            repo_dir=str(tmp_path),
            verification_commands=[],
        )
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
            heartbeat_dir=tmp_path / "heartbeats",
            failure_log_path=tmp_path / "failure_log.jsonl",
            approval_handler=approval_handler,
            approval_tracker=approval_tracker,
            config_registry=config_registry,
        )
        mock_agent_runner.invoke.return_value = AgentResult(
            response="""<!-- TRIAGE_RESULT
{"proposal_type":"bug_fix","confidence":0.9,"candidate_files":["tests/test_bot.py"],"issue_title":"Fix missing import","fix_plan":"Add a regression test.","file_changes":[{"file_path":"tests/test_bot.py","new_content":"def test_ok():\\n    assert True\\n"}]}
-->""",
            run_dir=tmp_path / "runs" / "triage",
            success=True,
        )

        from trading_assistant.schemas.bug_triage import BugComplexity, BugSeverity, ErrorEvent, TriageOutcome, TriageResult

        action = _make_action(
            ActionType.SPAWN_TRIAGE,
            bot_id="bot1",
            details={"bot_id": "bot1", "error_type": "ImportError", "message": "No module", "stack_trace": "Traceback"},
        )
        triage_result = TriageResult(
            error_event=ErrorEvent(bot_id="bot1", error_type="ImportError", message="No module", stack_trace="Traceback"),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
        )

        with patch("trading_assistant.skills.run_bug_triage.TriageRunner") as MockRunner:
            MockRunner.return_value.triage.return_value = triage_result
            await handlers.handle_triage(action)

        approval_handler.handle_approve.assert_called_once()
        pending = approval_tracker.get_pending()
        assert len(pending) == 1
        assert pending[0].change_kind == ChangeKind.BUG_FIX

    @pytest.mark.asyncio
    async def test_investigation_creates_issue_and_links_failure_log(self, tmp_path: Path, mock_agent_runner, mock_dispatcher, event_stream: EventStream):
        pr_builder = MagicMock()
        pr_builder.create_issue = AsyncMock(return_value=GitHubIssueResult(
            success=True,
            issue_url="https://github.com/x/y/issues/7",
        ))
        config_registry = MagicMock()
        config_registry.get_profile.return_value = BotConfigProfile(
            bot_id="bot1",
            repo_dir=str(tmp_path),
            verification_commands=[],
        )
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
            heartbeat_dir=tmp_path / "heartbeats",
            failure_log_path=tmp_path / "failure_log.jsonl",
            pr_builder=pr_builder,
            config_registry=config_registry,
        )
        mock_agent_runner.invoke.return_value = AgentResult(
            response="""<!-- TRIAGE_RESULT
{"proposal_type":"investigation","confidence":0.7,"candidate_files":["connector.py"],"issue_title":"Investigate retry loop","issue_body":"Stack trace and context.","fix_plan":"Inspect reconnect path."}
-->""",
            run_dir=tmp_path / "runs" / "triage",
            success=True,
        )

        from trading_assistant.schemas.bug_triage import BugComplexity, BugSeverity, ErrorEvent, TriageOutcome, TriageResult

        action = _make_action(
            ActionType.SPAWN_TRIAGE,
            bot_id="bot1",
            details={"bot_id": "bot1", "error_type": "RuntimeError", "message": "retry loop", "stack_trace": "Traceback"},
        )
        triage_result = TriageResult(
            error_event=ErrorEvent(bot_id="bot1", error_type="RuntimeError", message="retry loop", stack_trace="Traceback"),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.MULTI_FILE,
            outcome=TriageOutcome.NEEDS_INVESTIGATION,
        )

        with patch("trading_assistant.skills.run_bug_triage.TriageRunner") as MockRunner:
            MockRunner.return_value.triage.return_value = triage_result
            await handlers.handle_triage(action)

        pr_builder.create_issue.assert_called_once()
        assert "https://github.com/x/y/issues/7" in (tmp_path / "failure_log.jsonl").read_text(encoding="utf-8")


class TestFeedbackRouting:
    @pytest.mark.asyncio
    async def test_feedback_accept_routes_to_pending_approval(self, tmp_path: Path, mock_agent_runner, mock_dispatcher, event_stream: EventStream):
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        approval_tracker.create_request(
            ApprovalRequest(
                request_id="req-1",
                suggestion_id="abc123",
                bot_id="bot1",
            ),
        )
        approval_handler = MagicMock()
        approval_handler.handle_approve = AsyncMock(return_value="approved")
        suggestion_tracker = MagicMock()
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
            heartbeat_dir=tmp_path / "heartbeats",
            failure_log_path=tmp_path / "failure_log.jsonl",
            suggestion_tracker=suggestion_tracker,
            approval_handler=approval_handler,
            approval_tracker=approval_tracker,
        )

        action = _make_action(
            ActionType.PROCESS_FEEDBACK,
            bot_id="bot1",
            details={"text": "approve suggestion #abc123", "report_id": "r1"},
        )
        await handlers.handle_feedback(action)

        approval_handler.handle_approve.assert_called_once_with("req-1")
        suggestion_tracker.implement.assert_not_called()

    @pytest.mark.asyncio
    async def test_feedback_accept_does_not_mark_implemented_when_approval_stays_pending(self, tmp_path: Path, mock_agent_runner, mock_dispatcher, event_stream: EventStream):
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        approval_tracker.create_request(
            ApprovalRequest(
                request_id="req-1",
                suggestion_id="abc123",
                bot_id="bot1",
            ),
        )
        approval_handler = MagicMock()
        approval_handler.handle_approve = AsyncMock(return_value="PR creation failed")
        suggestion_tracker = MagicMock()
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
            heartbeat_dir=tmp_path / "heartbeats",
            failure_log_path=tmp_path / "failure_log.jsonl",
            suggestion_tracker=suggestion_tracker,
            approval_handler=approval_handler,
            approval_tracker=approval_tracker,
        )

        action = _make_action(
            ActionType.PROCESS_FEEDBACK,
            bot_id="bot1",
            details={"text": "approve suggestion #abc123", "report_id": "r1"},
        )
        await handlers.handle_feedback(action)

        approval_handler.handle_approve.assert_called_once_with("req-1")
        suggestion_tracker.implement.assert_not_called()
        recent = event_stream.get_recent()
        assert not any(e.event_type == "suggestion_accepted" for e in recent)


class TestStructuralLinking:
    def test_structural_proposals_link_by_explicit_id(self, tmp_path: Path, mock_agent_runner, mock_dispatcher, event_stream: EventStream):
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1", "bot2"],
            heartbeat_dir=tmp_path / "heartbeats",
            failure_log_path=tmp_path / "failure_log.jsonl",
            suggestion_tracker=MagicMock(),
        )
        handlers._suggestion_tracker.load_all.return_value = []
        handlers._suggestion_tracker.record.return_value = True

        validation = MagicMock()
        validation.approved_suggestions = [
            AgentSuggestion(
                suggestion_id="sg-struct-1",
                bot_id="bot1",
                category="structural",
                title="Rename engine module",
                evidence_summary="code duplication",
                confidence=0.8,
            ),
        ]
        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(
                    hypothesis_id="hyp-1",
                    linked_suggestion_id="sg-struct-1",
                    bot_id="bot1",
                    title="Completely different pattern title",
                    description="Extract common execution helper",
                ),
            ],
        )

        suggestion_ids = handlers._record_agent_suggestions(validation, "run-1", parsed)
        handlers._extract_and_record_patterns(parsed, ["bot1", "bot2"], suggestion_ids)

        from trading_assistant.skills.pattern_library import PatternLibrary

        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 1
        assert entries[0].linked_suggestion_id == "sg-struct-1"

    def test_structural_patterns_preserve_explicit_link_without_same_run_mapping(self, tmp_path: Path, mock_agent_runner, mock_dispatcher, event_stream: EventStream):
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1", "bot2"],
            heartbeat_dir=tmp_path / "heartbeats",
            failure_log_path=tmp_path / "failure_log.jsonl",
            suggestion_tracker=MagicMock(),
        )

        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(
                    hypothesis_id="hyp-1",
                    linked_suggestion_id="sg-existing",
                    bot_id="bot1",
                    title="Extract execution helper",
                    description="Reduce duplication",
                ),
            ],
        )

        handlers._extract_and_record_patterns(parsed, ["bot1", "bot2"], suggestion_ids={})

        from trading_assistant.skills.pattern_library import PatternLibrary

        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 1
        assert entries[0].linked_suggestion_id == "sg-existing"

    def test_agent_suggestions_only_link_hypotheses_by_explicit_suggestion_id(self, tmp_path: Path, mock_agent_runner, mock_dispatcher, event_stream: EventStream):
        tracker = MagicMock()
        tracker.record.return_value = True
        handlers = Handlers(
            agent_runner=mock_agent_runner,
            event_stream=event_stream,
            dispatcher=mock_dispatcher,
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
            heartbeat_dir=tmp_path / "heartbeats",
            failure_log_path=tmp_path / "failure_log.jsonl",
            suggestion_tracker=tracker,
        )

        validation = MagicMock()
        validation.approved_suggestions = [
            AgentSuggestion(
                suggestion_id="sg-param",
                bot_id="bot1",
                category="entry_signal",
                title="Tighten filter",
                evidence_summary="good evidence",
                confidence=0.8,
            ),
        ]
        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(
                    hypothesis_id="hyp-1",
                    linked_suggestion_id="sg-other",
                    bot_id="bot1",
                    title="Different structural idea",
                    description="Not linked to sg-param",
                ),
            ],
        )

        handlers._record_agent_suggestions(validation, "run-1", parsed)
        recorded = tracker.record.call_args[0][0]
        assert recorded.hypothesis_id is None


class TestAlert:
    @pytest.mark.asyncio
    async def test_dispatches_critical(
        self, handlers: Handlers, mock_dispatcher
    ):
        """Alert -> immediate CRITICAL notification."""
        action = _make_action(
            ActionType.ALERT_IMMEDIATE,
            bot_id="bot1",
            details={"message": "Exchange API down", "severity": "CRITICAL"},
        )
        await handlers.handle_alert(action)

        mock_dispatcher.dispatch.assert_called_once()
        payload = mock_dispatcher.dispatch.call_args[0][0]
        assert payload.priority == NotificationPriority.CRITICAL
        assert "ALERT" in payload.title
        assert "bot1" in payload.title


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_writes_heartbeat_file(
        self, handlers: Handlers, tmp_path: Path
    ):
        """Heartbeat -> file write."""
        action = _make_action(ActionType.UPDATE_HEARTBEAT, bot_id="bot1")
        await handlers.handle_heartbeat(action)

        hb_path = tmp_path / "heartbeats" / "bot1.heartbeat"
        assert hb_path.exists()
        content = hb_path.read_text()
        # Should be a valid ISO timestamp
        datetime.fromisoformat(content)


class TestNotification:
    @pytest.mark.asyncio
    async def test_dispatches_payload(
        self, handlers: Handlers, mock_dispatcher
    ):
        """Notification action -> dispatcher."""
        action = _make_action(
            ActionType.SEND_NOTIFICATION,
            details={
                "notification_type": "custom_alert",
                "priority": "high",
                "title": "Test",
                "body": "Test body",
            },
        )
        await handlers.handle_notification(action)

        mock_dispatcher.dispatch.assert_called_once()
        payload = mock_dispatcher.dispatch.call_args[0][0]
        assert payload.notification_type == "custom_alert"
        assert payload.priority == NotificationPriority.HIGH


class TestLoadCoordinatorEvents:
    def test_loads_events_from_coordinator_impact_json(self, handlers: Handlers, tmp_path: Path):
        """Write coordinator_impact.json with events list, verify parsed CoordinatorAction objects."""
        date = "2026-03-01"
        coord_dir = tmp_path / "data" / "curated" / date / "swing_multi_01"
        coord_dir.mkdir(parents=True)
        events_data = {
            "total_events": 2,
            "by_action": {"tighten_stop_be": 1, "size_boost": 1},
            "by_rule": {"rule_1": 2},
            "symbols_affected": ["BTCUSDT"],
            "events": [
                {
                    "action": "tighten_stop_be",
                    "trigger_strategy": "strat_a",
                    "target_strategy": "strat_b",
                    "symbol": "BTCUSDT",
                    "rule": "rule_1",
                    "timestamp": "2026-03-01T12:00:00Z",
                },
                {
                    "action": "size_boost",
                    "trigger_strategy": "strat_c",
                    "target_strategy": "strat_d",
                    "symbol": "BTCUSDT",
                    "rule": "rule_1",
                    "timestamp": "2026-03-01T13:00:00Z",
                },
            ],
        }
        (coord_dir / "coordinator_impact.json").write_text(json.dumps(events_data))

        result = handlers._load_coordinator_events(date, date)
        assert len(result) == 2
        assert result[0].action == "tighten_stop_be"
        assert result[1].action == "size_boost"

    def test_returns_empty_when_no_files(self, handlers: Handlers):
        """No files exist -> empty list."""
        result = handlers._load_coordinator_events("2026-01-01", "2026-01-07")
        assert result == []

    def test_skips_malformed_events(self, handlers: Handlers, tmp_path: Path):
        """Some valid, some malformed entries -> only valid returned."""
        date = "2026-03-01"
        coord_dir = tmp_path / "data" / "curated" / date / "swing_multi_01"
        coord_dir.mkdir(parents=True)
        events_data = {
            "events": [
                {"action": "tighten_stop_be", "rule": "rule_1"},
                {"not_a_valid_field_only": 123},  # missing required 'action'
                {"action": "size_boost", "rule": "rule_2"},
            ],
        }
        (coord_dir / "coordinator_impact.json").write_text(json.dumps(events_data))

        result = handlers._load_coordinator_events(date, date)
        # CoordinatorAction requires 'action' field; the malformed one without it should be skipped
        # Actually, checking the schema: action is required (no default), so the middle one should fail
        # But wait - it has no 'action' key but all others have defaults. Let me check...
        # The second entry {"not_a_valid_field_only": 123} has no 'action', which is required.
        # So it should be skipped. The first and third are valid.
        assert len(result) == 2
        assert result[0].action == "tighten_stop_be"
        assert result[1].action == "size_boost"


class TestErrorEmission:
    @pytest.mark.asyncio
    async def test_handler_error_emits_sse(
        self, handlers: Handlers, mock_agent_runner, event_stream: EventStream
    ):
        """Exception in handler -> SSE error event."""
        # Make agent runner raise
        mock_agent_runner.invoke.side_effect = RuntimeError("Unexpected error")

        # Create minimal curated data to pass quality gate
        curated_dir = handlers._curated_dir
        date = "2026-03-02"
        for bot in ["bot1", "bot2"]:
            bot_dir = curated_dir / date / bot
            bot_dir.mkdir(parents=True)
            for f in [
                "summary.json", "winners.json", "losers.json",
                "process_failures.json", "notable_missed.json",
                "regime_analysis.json", "filter_analysis.json",
                "root_cause_summary.json",
                "hourly_performance.json", "slippage_stats.json",
                "factor_attribution.json", "exit_efficiency.json",
            ]:
                (bot_dir / f).write_text("{}")
        (curated_dir / date / "portfolio_risk_card.json").write_text("{}")
        # Trade data so minimum-data threshold passes
        for bot in ["bot1", "bot2"]:
            (curated_dir / date / bot / "trades.jsonl").write_text(
                '{"trade_id":"t1"}\n{"trade_id":"t2"}\n'
            )
        handlers._memory_dir.mkdir(parents=True, exist_ok=True)

        action = _make_action(ActionType.SPAWN_DAILY_ANALYSIS, details={"date": date})
        await handlers.handle_daily_analysis(action)

        # Verify SSE error event was broadcast
        recent = event_stream.get_recent()
        error_events = [e for e in recent if "error" in e.event_type]
        assert len(error_events) >= 1
