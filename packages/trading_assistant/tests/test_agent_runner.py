"""Tests for orchestrator/agent_runner.py provider-backed CLI orchestration."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_assistant.orchestrator.agent_runner import AgentRunner
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
from trading_assistant.schemas.prompt_package import PromptPackage
from tests.factories import make_sample_package


class _FakeStreamReader:
    def __init__(
        self,
        *,
        lines: list[str] | None = None,
        text: str = "",
        line_delay_seconds: float = 0.0,
    ) -> None:
        self._lines = [line.encode("utf-8") + b"\n" for line in (lines or [])]
        self._text = text.encode("utf-8")
        self._line_delay_seconds = line_delay_seconds

    async def readline(self) -> bytes:
        if self._line_delay_seconds:
            await asyncio.sleep(self._line_delay_seconds)
        if not self._lines:
            return b""
        return self._lines.pop(0)

    async def read(self) -> bytes:
        data = self._text
        self._text = b""
        return data


class _StreamingProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str],
        stderr: str = "",
        returncode: int = 0,
        wait_delay_seconds: float = 0.0,
        line_delay_seconds: float = 0.0,
    ) -> None:
        self.stdout = _FakeStreamReader(
            lines=stdout_lines,
            line_delay_seconds=line_delay_seconds,
        )
        self.stderr = _FakeStreamReader(text=stderr)
        self.returncode = returncode
        self.killed = False
        self._wait_delay_seconds = wait_delay_seconds

    async def wait(self) -> int:
        if self.killed:
            return self.returncode
        if self._wait_delay_seconds:
            await asyncio.sleep(self._wait_delay_seconds)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _ready(provider: AgentProvider, runtime: str) -> ProviderReadiness:
    return ProviderReadiness(provider=provider, available=True, runtime=runtime)


def _write_codex_home(
    tmp_path: Path,
    *,
    auth_mode: str = "chatgpt",
    include_api_key: bool = False,
    include_tokens: bool = True,
    config_text: str = "",
    log_text: str = "",
) -> tuple[Path, Path]:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    auth_payload: dict[str, object] = {"auth_mode": auth_mode}
    if include_tokens:
        auth_payload["tokens"] = {
            "access_token": "token",
            "refresh_token": "refresh",
        }
    if include_api_key:
        auth_payload["OPENAI_API_KEY"] = "sk-test"
    (codex_home / "auth.json").write_text(
        json.dumps(auth_payload),
        encoding="utf-8",
    )
    if config_text:
        (codex_home / "config.toml").write_text(config_text, encoding="utf-8")
    log_path = tmp_path / "Codex.log"
    if log_text:
        log_path.write_text(log_text, encoding="utf-8")
    return codex_home, log_path


@pytest.fixture
def tmp_runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def event_stream() -> EventStream:
    return EventStream()


@pytest.fixture
def runner(tmp_runs: Path, session_store: SessionStore, event_stream: EventStream) -> AgentRunner:
    return AgentRunner(
        runs_dir=tmp_runs,
        session_store=session_store,
        claude_command="claude",
        codex_command="codex",
        default_model="sonnet",
        default_max_turns=5,
        timeout_seconds=30,
        event_stream=event_stream,
    )


@pytest.fixture
def sample_package() -> PromptPackage:
    return make_sample_package(
        data={"summary": {"pnl": 100}, "risk": {"drawdown": 0.02}},
        instructions="Step 1: Look at PnL.\nStep 2: Check risk.",
    )


class TestPreferenceResolution:
    def test_resolves_default_provider_model(self, tmp_runs: Path, session_store: SessionStore):
        runner = AgentRunner(
            runs_dir=tmp_runs,
            session_store=session_store,
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CODEX_PRO)
            ),
        )

        selection, requested_model = runner._preferences.resolve_selection(
            AgentWorkflow.DAILY_ANALYSIS
        )

        assert selection.provider == AgentProvider.CODEX_PRO
        assert selection.model == "gpt-5.4"
        assert requested_model is None

    def test_resolves_workflow_override_before_global_default(
        self,
        tmp_runs: Path,
        session_store: SessionStore,
    ):
        runner = AgentRunner(
            runs_dir=tmp_runs,
            session_store=session_store,
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="opus"),
                overrides={
                    AgentWorkflow.MONTHLY_MODEL_REVIEW: AgentSelection(
                        provider=AgentProvider.OPENROUTER,
                    ),
                },
            ),
        )

        selection, requested_model = runner._preferences.resolve_selection(
            AgentWorkflow.MONTHLY_MODEL_REVIEW
        )

        assert selection.provider == AgentProvider.OPENROUTER
        assert selection.model == "minimax/minimax-m2.5"
        assert requested_model is None


class TestInvoke:
    @pytest.mark.asyncio
    async def test_streamed_claude_run_persists_artifacts_and_events(
        self,
        runner: AgentRunner,
        event_stream: EventStream,
        session_store: SessionStore,
        sample_package: PromptPackage,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(
            AgentProvider.CLAUDE_MAX,
            "claude_cli",
        )
        stdout_lines = [
            json.dumps({"session_id": "claude-session-123"}),
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Inspecting summary."},
                        {"type": "tool_use", "name": "Read", "input": {"file": "summary.json"}},
                    ]
                },
            }),
            json.dumps({
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": "Loaded summary"}]},
            }),
            "not-json",
            json.dumps({"type": "result", "result": "Final analysis", "cost_usd": 0.05}),
        ]
        mock_proc = _StreamingProcess(
            stdout_lines=stdout_lines,
            stderr="claude stderr",
            line_delay_seconds=0.01,
        )

        with (
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-2026-03-02",
                allowed_tools=["Read", "Grep", "Glob"],
            )

        run_dir = runner._runs_dir / "daily-2026-03-02"
        assert result.success is True
        assert result.provider == "claude_max"
        assert result.runtime == "claude_cli"
        assert result.session_id == "claude-session-123"
        assert result.response == "Final analysis"
        assert result.cost_usd == 0.05
        assert result.tool_call_count == 1
        assert result.stream_event_count == 5
        assert result.auth_mode == "claude.ai:max"
        assert result.first_output_ms > 0
        assert (run_dir / "response.md").read_text(encoding="utf-8") == "Final analysis"
        assert (run_dir / "claude-session.jsonl").read_text(encoding="utf-8").strip().splitlines() == stdout_lines
        assert (run_dir / "claude-stderr.log").read_text(encoding="utf-8") == "claude stderr"

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        records = session_store.get_session(result.session_id, "daily_analysis", date_str)
        assert len(records) == 1
        metadata = records[0].metadata
        assert metadata["provider"] == "claude_max"
        assert metadata["runtime"] == "claude_cli"
        assert metadata["requested_model"] == "sonnet"
        assert metadata["effective_model"] == "sonnet"
        assert metadata["cost_usd"] == 0.05
        assert metadata["tool_call_count"] == 1
        assert metadata["stream_event_count"] == 5
        assert metadata["auth_mode"] == "claude.ai:max"
        assert metadata["first_output_ms"] > 0

        events = event_stream.get_recent()
        event_types = [event.event_type for event in events]
        assert "agent_invocation_started" in event_types
        assert "agent_invocation_progress" in event_types
        assert "agent_invocation_completed" in event_types
        progress_kinds = {
            event.data.get("kind")
            for event in events
            if event.event_type == "agent_invocation_progress"
        }
        assert {"assistant_text", "tool_call", "tool_result", "raw", "result"} <= progress_kinds

    @pytest.mark.asyncio
    async def test_streamed_claude_falls_back_without_result_block(
        self,
        runner: AgentRunner,
        sample_package: PromptPackage,
    ):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(
            AgentProvider.CLAUDE_MAX,
            "claude_cli",
        )
        stdout_lines = [
            json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "First chunk"}]},
            }),
            json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Second chunk"}]},
            }),
        ]
        mock_proc = _StreamingProcess(stdout_lines=stdout_lines)

        with (
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-fallback",
            )

        assert result.success is True
        assert result.response == "First chunk\nSecond chunk"
        assert (runner._runs_dir / "daily-fallback" / "response.md").read_text(
            encoding="utf-8"
        ) == "First chunk\nSecond chunk"

    @pytest.mark.asyncio
    async def test_streamed_codex_run_persists_artifacts_and_events(
        self,
        tmp_runs: Path,
        session_store: SessionStore,
        sample_package: PromptPackage,
        event_stream: EventStream,
        tmp_path: Path,
    ):
        runner = AgentRunner(
            runs_dir=tmp_runs,
            session_store=session_store,
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CODEX_PRO)
            ),
            event_stream=event_stream,
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CODEX_PRO] = _ready(
            AgentProvider.CODEX_PRO,
            "codex_cli",
        )
        codex_home, log_path = _write_codex_home(tmp_path)
        stdout_lines = [
            json.dumps({"thread_id": "thread-123"}),
            json.dumps({"item": {"type": "message", "text": "First answer"}}),
            json.dumps({"item": {"type": "message_delta", "text": "Second answer"}}),
            json.dumps({"item": {"type": "tool_call", "name": "Read", "arguments": {"path": "summary.json"}}}),
            json.dumps({"item": {"type": "tool_result", "output": "loaded summary"}}),
            "not-json",
            json.dumps({"usage": {"total_cost_usd": 1.25}}),
        ]
        mock_proc = _StreamingProcess(
            stdout_lines=stdout_lines,
            stderr="codex stderr",
            line_delay_seconds=0.01,
        )

        with (
            patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False),
            patch.object(runner._auth_checker, "_find_latest_codex_log", return_value=log_path),
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"),
            patch("trading_assistant.orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="codex-run",
            )

        run_dir = runner._runs_dir / "codex-run"
        assert result.success is True
        assert result.provider == "codex_pro"
        assert result.runtime == "codex_cli"
        assert result.session_id == "thread-123"
        assert result.cost_usd == 1.25
        assert result.response == "First answer\nSecond answer"
        assert result.stream_event_count == 7
        assert result.tool_call_count == 1
        assert result.auth_mode == "chatgpt"
        assert result.first_output_ms > 0
        assert (run_dir / "response.md").read_text(encoding="utf-8") == "First answer\nSecond answer"
        assert (run_dir / "codex-session.jsonl").read_text(encoding="utf-8").strip().splitlines() == stdout_lines
        assert (run_dir / "codex-stderr.log").read_text(encoding="utf-8") == "codex stderr"

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        records = session_store.get_session(result.session_id, "daily_analysis", date_str)
        assert len(records) == 1
        metadata = records[0].metadata
        assert metadata["provider"] == "codex_pro"
        assert metadata["runtime"] == "codex_cli"
        assert metadata["effective_model"] == "gpt-5.4"
        assert metadata["cost_usd"] == 1.25
        assert metadata["tool_call_count"] == 1
        assert metadata["stream_event_count"] == 7
        assert metadata["auth_mode"] == "chatgpt"

        events = event_stream.get_recent()
        progress_kinds = {
            event.data.get("kind")
            for event in events
            if event.event_type == "agent_invocation_progress"
        }
        assert {"assistant_text", "tool_call", "tool_result", "raw"} <= progress_kinds

    @pytest.mark.asyncio
    async def test_provider_unavailable_returns_failed_result(
        self,
        tmp_runs: Path,
        session_store: SessionStore,
        sample_package: PromptPackage,
    ):
        runner = AgentRunner(
            runs_dir=tmp_runs,
            session_store=session_store,
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CODEX_PRO)
            ),
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CODEX_PRO] = ProviderReadiness(
            provider=AgentProvider.CODEX_PRO,
            available=False,
            runtime="codex_cli",
            reason="Command preflight failed for codex: missing executable",
        )

        result = await runner.invoke(
            agent_type="daily_analysis",
            prompt_package=sample_package,
            run_id="codex-unavailable",
        )

        assert result.success is False
        assert result.provider == "codex_pro"
        assert result.runtime == "codex_cli"
        assert "missing executable" in result.error

    @pytest.mark.asyncio
    async def test_buffered_timeout_kills_process(
        self,
        tmp_runs: Path,
        session_store: SessionStore,
        sample_package: PromptPackage,
        tmp_path: Path,
    ):
        runner = AgentRunner(
            runs_dir=tmp_runs,
            session_store=session_store,
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CODEX_PRO),
                workflow_tuning={
                    AgentWorkflow.DAILY_ANALYSIS: WorkflowTuning(timeout_seconds=0),
                },
            ),
            timeout_seconds=0,
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CODEX_PRO] = _ready(
            AgentProvider.CODEX_PRO,
            "codex_cli",
        )
        codex_home, log_path = _write_codex_home(tmp_path)
        mock_proc = _StreamingProcess(
            stdout_lines=[],
            wait_delay_seconds=10,
        )

        with (
            patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=False),
            patch.object(runner._auth_checker, "_find_latest_codex_log", return_value=log_path),
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"),
            patch("trading_assistant.orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="codex-timeout",
            )

        assert result.success is False
        assert "Timeout" in result.error
        assert mock_proc.killed is True
