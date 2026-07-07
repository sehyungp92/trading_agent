"""Tests for per-workflow tuning (Phase 4)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trading_assistant.orchestrator.agent_preferences import (
    AgentPreferencesManager,
    DEFAULT_WORKFLOW_TUNING,
)
from trading_assistant.orchestrator.agent_runner import AgentResult, AgentRunner, _DEFAULT_TIMEOUT_SECONDS
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
    WorkflowTuning,
)


def _ready(provider: AgentProvider) -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(provider=provider, available=True, runtime=runtime)


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------


class TestWorkflowTuningSchema:
    def test_all_fields_optional(self):
        tuning = WorkflowTuning()
        assert tuning.timeout_seconds is None
        assert tuning.max_turns is None
        assert tuning.allowed_tools is None

    def test_fields_set(self):
        tuning = WorkflowTuning(timeout_seconds=900, max_turns=8, allowed_tools=["Read"])
        assert tuning.timeout_seconds == 900
        assert tuning.max_turns == 8
        assert tuning.allowed_tools == ["Read"]

    def test_preferences_with_workflow_tuning(self):
        prefs = AgentPreferences(
            workflow_tuning={
                AgentWorkflow.MONTHLY_MODEL_REVIEW: WorkflowTuning(timeout_seconds=1200, max_turns=15),
            },
        )
        assert prefs.workflow_tuning[AgentWorkflow.MONTHLY_MODEL_REVIEW].timeout_seconds == 1200

    def test_serialization_round_trip(self):
        prefs = AgentPreferences(
            workflow_tuning={
                AgentWorkflow.TRIAGE: WorkflowTuning(timeout_seconds=300, max_turns=4),
            },
        )
        data = prefs.model_dump()
        restored = AgentPreferences(**data)
        assert restored.workflow_tuning[AgentWorkflow.TRIAGE].timeout_seconds == 300


# ---------------------------------------------------------------------------
# Default Tunings
# ---------------------------------------------------------------------------


class TestDefaultTunings:
    def test_daily_defaults(self):
        d = DEFAULT_WORKFLOW_TUNING[AgentWorkflow.DAILY_ANALYSIS]
        assert d.timeout_seconds == 600
        assert d.max_turns == 5

    def test_weekly_defaults(self):
        d = DEFAULT_WORKFLOW_TUNING[AgentWorkflow.WEEKLY_ANALYSIS]
        assert d.timeout_seconds == 900
        assert d.max_turns == 8

    def test_monthly_defaults(self):
        validation = DEFAULT_WORKFLOW_TUNING[AgentWorkflow.MONTHLY_VALIDATION]
        review = DEFAULT_WORKFLOW_TUNING[AgentWorkflow.MONTHLY_MODEL_REVIEW]
        assert validation.timeout_seconds == 1200
        assert review.timeout_seconds == 900

    def test_triage_defaults(self):
        d = DEFAULT_WORKFLOW_TUNING[AgentWorkflow.TRIAGE]
        assert d.timeout_seconds == 300
        assert d.max_turns == 4


# ---------------------------------------------------------------------------
# Resolve Tuning
# ---------------------------------------------------------------------------


class TestResolveTuning:
    def test_returns_built_in_defaults_when_no_user_prefs(self):
        mgr = AgentPreferencesManager(AgentPreferences())
        tuning = mgr.resolve_tuning(AgentWorkflow.DAILY_ANALYSIS)
        assert tuning.timeout_seconds == 600
        assert tuning.max_turns == 5

    def test_user_prefs_override_defaults(self):
        prefs = AgentPreferences(
            workflow_tuning={
                AgentWorkflow.DAILY_ANALYSIS: WorkflowTuning(timeout_seconds=800, max_turns=10),
            },
        )
        mgr = AgentPreferencesManager(prefs)
        tuning = mgr.resolve_tuning(AgentWorkflow.DAILY_ANALYSIS)
        assert tuning.timeout_seconds == 800
        assert tuning.max_turns == 10

    def test_caller_override_beats_user_prefs(self):
        prefs = AgentPreferences(
            workflow_tuning={
                AgentWorkflow.DAILY_ANALYSIS: WorkflowTuning(max_turns=10),
            },
        )
        mgr = AgentPreferencesManager(prefs)
        tuning = mgr.resolve_tuning(
            AgentWorkflow.DAILY_ANALYSIS,
            max_turns_override=3,
        )
        assert tuning.max_turns == 3

    def test_allowed_tools_override(self):
        mgr = AgentPreferencesManager(AgentPreferences())
        tuning = mgr.resolve_tuning(
            AgentWorkflow.TRIAGE,
            allowed_tools_override=["Read", "Grep"],
        )
        assert tuning.allowed_tools == ["Read", "Grep"]

    def test_none_workflow_returns_empty_tuning(self):
        mgr = AgentPreferencesManager(AgentPreferences())
        tuning = mgr.resolve_tuning(None)
        assert tuning.timeout_seconds is None
        assert tuning.max_turns is None
        assert tuning.allowed_tools is None

    def test_partial_user_prefs_fall_through_to_defaults(self):
        prefs = AgentPreferences(
            workflow_tuning={
                AgentWorkflow.TRIAGE: WorkflowTuning(timeout_seconds=450),
            },
        )
        mgr = AgentPreferencesManager(prefs)
        tuning = mgr.resolve_tuning(AgentWorkflow.TRIAGE)
        assert tuning.timeout_seconds == 450
        # max_turns should fall through to default
        assert tuning.max_turns == 4

    def test_legacy_wfo_user_prefs_do_not_get_builtin_defaults(self):
        prefs = AgentPreferences.model_validate({
            "workflow_tuning": {
                "wfo": {"timeout_seconds": 1500},
            },
        })
        mgr = AgentPreferencesManager(prefs)
        tuning = mgr.resolve_tuning(None)
        assert prefs.workflow_tuning == {}
        assert tuning.timeout_seconds is None
        assert tuning.max_turns is None


# ---------------------------------------------------------------------------
# AgentRunner Integration
# ---------------------------------------------------------------------------


class TestAgentRunnerTuningIntegration:
    @pytest.fixture
    def runner(self, tmp_path: Path) -> AgentRunner:
        return AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            event_stream=EventStream(),
            timeout_seconds=600,
        )

    @pytest.mark.asyncio
    async def test_invoke_passes_tuned_max_turns(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(
            AgentProvider.CLAUDE_MAX,
        )

        captured_kwargs: dict = {}

        async def _capture(**kwargs):
            captured_kwargs.update(kwargs)
            return AgentResult(
                response="ok", run_dir=Path("/tmp"),
                success=True, provider="claude_max", runtime="claude_cli",
            )

        with patch.object(runner, "invoke_with_selection", side_effect=_capture):
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-tuned",
            )

        # Default daily max_turns is 5
        assert captured_kwargs["max_turns"] == 5

    @pytest.mark.asyncio
    async def test_invoke_passes_tuned_timeout(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(
            AgentProvider.CLAUDE_MAX,
        )

        captured_kwargs: dict = {}

        async def _capture(**kwargs):
            captured_kwargs.update(kwargs)
            return AgentResult(
                response="ok", run_dir=Path("/tmp"),
                success=True, provider="claude_max", runtime="claude_cli",
            )

        with patch.object(runner, "invoke_with_selection", side_effect=_capture):
            await runner.invoke(
                agent_type="weekly_analysis",
                prompt_package=sample_package,
                run_id="run-weekly",
            )

        # Weekly default timeout is 900
        assert captured_kwargs["timeout_seconds"] == 900

    @pytest.mark.asyncio
    async def test_caller_max_turns_overrides_tuning(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(
            AgentProvider.CLAUDE_MAX,
        )

        captured_kwargs: dict = {}

        async def _capture(**kwargs):
            captured_kwargs.update(kwargs)
            return AgentResult(
                response="ok", run_dir=Path("/tmp"),
                success=True, provider="claude_max", runtime="claude_cli",
            )

        with patch.object(runner, "invoke_with_selection", side_effect=_capture):
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-override",
                max_turns=20,
            )

        assert captured_kwargs["max_turns"] == 20

    @pytest.mark.asyncio
    async def test_timeout_override_applied_during_invocation(self, runner: AgentRunner, sample_package):
        """Test that timeout_seconds param on invoke_with_selection is passed through."""
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(
            AgentProvider.CLAUDE_MAX,
        )

        observed_timeout: int | None = None

        async def _fake_inner(**kwargs):
            nonlocal observed_timeout
            observed_timeout = kwargs.get("timeout_seconds")
            return AgentResult(
                response="ok", run_dir=Path("/tmp"),
                success=True, provider="claude_max", runtime="claude_cli",
            )

        with patch.object(runner, "_invoke_with_selection_inner", side_effect=_fake_inner):
            await runner.invoke_with_selection(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-timeout",
                selection=AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet"),
                timeout_seconds=999,
            )

        assert observed_timeout == 999
        # Instance state is never mutated
        assert runner._timeout_seconds == _DEFAULT_TIMEOUT_SECONDS
