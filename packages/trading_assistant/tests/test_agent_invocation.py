"""Tests for AgentRunner.invoke() and invoke_with_selection() flow."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from trading_assistant.orchestrator.agent_runner import AgentRunner, AgentResult
from trading_assistant.schemas.agent_capabilities import AgentCapability, AgentType, CapabilityCheckResult
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import (
    AgentProvider,
    AgentSelection,
    ProviderReadiness,
)
from trading_assistant.schemas.prompt_package import PromptPackage
from tests.factories import make_sample_package


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def runner(tmp_path: Path, session_store: SessionStore, event_stream) -> AgentRunner:
    return AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        event_stream=event_stream,
    )


@pytest.fixture
def sample_package():
    return make_sample_package(
        task_prompt="Analyse daily trades.",
        data={"trades": [{"id": 1}]},
    )


def _ready(provider: AgentProvider) -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(provider=provider, available=True, runtime=runtime)


def _unavailable(provider: AgentProvider, reason: str = "down") -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(
        provider=provider, available=False, runtime=runtime, reason=reason,
    )


class TestInvokeWorkflowResolution:
    @pytest.mark.asyncio
    async def test_invoke_resolves_daily_analysis_workflow(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        """invoke() should resolve agent_type to workflow and delegate."""
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        stream_result = AgentResult(
            response="analysis done", run_dir=Path("/tmp/run"),
            cost_usd=0.01, duration_ms=100, session_id="s1", success=True,
        )
        with patch.object(
            runner, "invoke_with_selection", new_callable=AsyncMock, return_value=stream_result,
        ) as mock_invoke:
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-001",
            )

        assert mock_invoke.called
        call_kwargs = mock_invoke.call_args
        # selection should have been passed
        assert call_kwargs.kwargs.get("selection") is not None or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] is not None
        )

    @pytest.mark.asyncio
    async def test_invoke_unknown_agent_type_uses_default_selection(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        """Unknown agent types should still work using default preferences."""
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        with patch.object(
            runner, "invoke_with_selection", new_callable=AsyncMock,
            return_value=AgentResult(
                response="done", run_dir=Path("/tmp"), cost_usd=0, duration_ms=0,
                session_id="s", success=True,
            ),
        ) as mock_invoke:
            await runner.invoke(
                agent_type="heartbeat",
                prompt_package=sample_package,
                run_id="run-002",
            )

        assert mock_invoke.called


class TestInvokeWithSelectionErrors:
    @pytest.mark.asyncio
    async def test_unavailable_provider_returns_failure(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _unavailable(
            AgentProvider.CLAUDE_MAX, "auth expired",
        )

        selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
        result = await runner.invoke_with_selection(
            agent_type="daily_analysis",
            prompt_package=sample_package,
            run_id="run-fail",
            selection=selection,
        )

        assert result.success is False
        assert "auth expired" in result.error

    @pytest.mark.asyncio
    async def test_unavailable_provider_broadcasts_failure_event(
        self, runner: AgentRunner, sample_package: PromptPackage, event_stream: EventStream,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CODEX_PRO] = _unavailable(
            AgentProvider.CODEX_PRO, "no tokens",
        )

        # Subscribe via queue
        queue = event_stream.subscribe()

        selection = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        await runner.invoke_with_selection(
            agent_type="triage",
            prompt_package=sample_package,
            run_id="run-evt",
            selection=selection,
        )

        # Check the broadcast event
        assert not queue.empty()
        event = queue.get_nowait()
        assert event.event_type == "agent_invocation_failed"
        assert event.data["run_id"] == "run-evt"
        assert "no tokens" in event.data.get("error", "")


def _mock_subprocess(stdout: str = "", returncode: int = 0):
    """Create a mock asyncio subprocess."""
    mock_proc = AsyncMock()
    mock_proc.stdout = AsyncMock()
    mock_proc.stderr = AsyncMock()
    mock_proc.returncode = returncode

    # Make readline return lines then empty bytes
    lines = [line.encode() + b"\n" for line in stdout.split("\n") if line]
    readline_iter = iter(lines + [b""])
    mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(readline_iter, b""))

    # stderr read returns empty
    mock_proc.stderr.read = AsyncMock(return_value=b"")

    mock_proc.wait = AsyncMock(return_value=returncode)
    return mock_proc


class TestInvokeWithSelectionSuccess:
    @pytest.mark.asyncio
    async def test_successful_claude_invocation(
        self, runner: AgentRunner, sample_package: PromptPackage, event_stream: EventStream,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        queue = event_stream.subscribe()

        import json as _json
        stream_output = "\n".join([
            _json.dumps({"type": "result", "result": "Great analysis."}),
            _json.dumps({"cost_usd": 0.05}),
        ])
        mock_proc = _mock_subprocess(stream_output)

        with (
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
        ):
            selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
            result = await runner.invoke_with_selection(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-ok",
                selection=selection,
            )

        assert result.success is True
        assert "Great analysis." in result.response

        # Should have at least started + completed events
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        event_types = [e.event_type for e in events]
        assert "agent_invocation_started" in event_types
        assert "agent_invocation_completed" in event_types

    @pytest.mark.asyncio
    async def test_run_directory_created(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        mock_proc = _mock_subprocess("")

        with (
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
        ):
            selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
            await runner.invoke_with_selection(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-dir-test",
                selection=selection,
            )

        assert (runner._runs_dir / "run-dir-test").exists()

    @pytest.mark.asyncio
    async def test_run_files_written(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        mock_proc = _mock_subprocess("")

        with (
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
        ):
            selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
            await runner.invoke_with_selection(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-files",
                selection=selection,
            )

        run_dir = runner._runs_dir / "run-files"
        # _write_run_files writes system_prompt.md and data/*.json files
        assert (run_dir / "system_prompt.md").exists()
        # Data files from the package
        assert (run_dir / "trades.json").exists()


class TestInvokeModelOverride:
    @pytest.mark.asyncio
    async def test_model_override_flows_through(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        with patch.object(
            runner, "invoke_with_selection", new_callable=AsyncMock,
            return_value=AgentResult(
                response="ok", run_dir=Path("/tmp"), cost_usd=0, duration_ms=0,
                session_id="s", success=True,
            ),
        ) as mock:
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-model",
                model="opus",
            )

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["requested_model"] == "opus"


class TestInvokeEventBroadcasting:
    @pytest.mark.asyncio
    async def test_started_event_includes_provider_details(
        self, runner: AgentRunner, sample_package: PromptPackage, event_stream: EventStream,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        queue = event_stream.subscribe()

        mock_proc = _mock_subprocess("")

        with (
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
        ):
            selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
            await runner.invoke_with_selection(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-bcast",
                selection=selection,
            )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        started = [e for e in events if e.event_type == "agent_invocation_started"]
        assert len(started) == 1
        assert started[0].data["provider"] == "claude_max"
        assert started[0].data["runtime"] == "claude_cli"


# --- Merged from test_agent_capabilities.py ---


class TestAgentTypeEnum:
    def test_has_8_members(self):
        assert len(AgentType) == 8

    def test_values_are_lowercase(self):
        for member in AgentType:
            assert member.value == member.value.lower()


class TestAgentCapabilitySchema:
    def test_construction_with_defaults(self):
        cap = AgentCapability(agent_type=AgentType.DAILY_ANALYSIS)
        assert cap.can_execute_shell is False
        assert cap.max_concurrent_tasks == 1
        assert cap.allowed_actions == []
        assert cap.forbidden_actions == []

    def test_construction_with_custom_values(self):
        cap = AgentCapability(
            agent_type=AgentType.BUG_TRIAGE,
            allowed_actions=["read_source", "open_github_issue"],
            forbidden_actions=["merge_pr"],
            can_execute_shell=False,
            max_concurrent_tasks=2,
        )
        assert cap.agent_type == AgentType.BUG_TRIAGE
        assert "read_source" in cap.allowed_actions
        assert cap.max_concurrent_tasks == 2


class TestCapabilityCheckResult:
    def test_result_construction(self):
        result = CapabilityCheckResult(
            allowed=True,
            agent_type="daily_analysis",
            action="generate_report",
            reason="Allowed",
        )
        assert result.allowed is True
        assert result.agent_type == "daily_analysis"
