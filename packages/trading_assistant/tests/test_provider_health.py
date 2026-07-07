"""Tests for provider health checking in AgentRunner."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_assistant.orchestrator.agent_runner import AgentRunner
from trading_assistant.orchestrator.provider_auth import (
    _CODEX_AUTH_MODE,
    _PROVIDER_STATUS_CACHE_TTL_SECONDS,
)
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import AgentProvider, ProviderReadiness


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def event_stream() -> EventStream:
    return EventStream()


@pytest.fixture
def runner(tmp_path: Path, session_store: SessionStore, event_stream: EventStream) -> AgentRunner:
    return AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        event_stream=event_stream,
    )


# ---------------------------------------------------------------------------
# Claude Max Auth
# ---------------------------------------------------------------------------


class TestClaudeMaxAuth:
    def test_available_with_valid_max_subscription(self, runner: AgentRunner):
        completed = subprocess.CompletedProcess(
            args=["claude", "auth", "status"],
            returncode=0,
            stdout=json.dumps({
                "loggedIn": True,
                "authMethod": "claude.ai",
                "subscriptionType": "max",
            }),
            stderr="",
        )
        with (
            patch.object(runner._auth_checker, "resolve_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.provider_auth.subprocess.run", return_value=completed),
        ):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is True
        assert status.auth_method == "claude.ai"
        assert status.subscription_type == "max"

    def test_unavailable_when_not_logged_in(self, runner: AgentRunner):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({"loggedIn": False, "authMethod": "", "subscriptionType": ""}),
            stderr="",
        )
        with (
            patch.object(runner._auth_checker, "resolve_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.provider_auth.subprocess.run", return_value=completed),
        ):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is False
        assert "login required" in status.reason

    def test_unavailable_when_non_claude_ai_auth(self, runner: AgentRunner):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({
                "loggedIn": True,
                "authMethod": "api-key",
                "subscriptionType": "max",
            }),
            stderr="",
        )
        with (
            patch.object(runner._auth_checker, "resolve_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.provider_auth.subprocess.run", return_value=completed),
        ):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is False
        assert "claude.ai auth" in status.reason

    def test_unavailable_when_pro_subscription(self, runner: AgentRunner):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps({
                "loggedIn": True,
                "authMethod": "claude.ai",
                "subscriptionType": "pro",
            }),
            stderr="",
        )
        with (
            patch.object(runner._auth_checker, "resolve_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.provider_auth.subprocess.run", return_value=completed),
        ):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is False
        assert "subscription required" in status.reason

    def test_unavailable_when_command_not_found(self, runner: AgentRunner):
        with patch.object(runner._auth_checker, "resolve_command", return_value=None):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is False
        assert "Command not found" in status.reason

    def test_unavailable_on_timeout(self, runner: AgentRunner):
        with (
            patch.object(runner._auth_checker, "resolve_command", return_value="/usr/bin/claude"),
            patch(
                "trading_assistant.orchestrator.provider_auth.subprocess.run",
                side_effect=subprocess.TimeoutExpired("claude", 5),
            ),
        ):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is False
        assert "timed out" in status.reason

    def test_unavailable_on_invalid_json(self, runner: AgentRunner):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-json", stderr="",
        )
        with (
            patch.object(runner._auth_checker, "resolve_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.provider_auth.subprocess.run", return_value=completed),
        ):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is False
        assert "invalid JSON" in status.reason

    def test_unavailable_on_nonzero_exit(self, runner: AgentRunner):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="auth error",
        )
        with (
            patch.object(runner._auth_checker, "resolve_command", return_value="/usr/bin/claude"),
            patch("trading_assistant.orchestrator.provider_auth.subprocess.run", return_value=completed),
        ):
            status = runner._auth_checker.claude_max_auth_status()

        assert status.available is False
        assert "auth error" in status.reason or "exit code" in status.reason


# ---------------------------------------------------------------------------
# Codex Auth
# ---------------------------------------------------------------------------


class TestCodexAuth:
    def test_available_with_valid_chatgpt_auth(self, runner: AgentRunner, tmp_path: Path):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        auth = {
            "auth_mode": _CODEX_AUTH_MODE,
            "tokens": {"access_token": "tok", "refresh_token": "ref"},
        }
        (codex_home / "auth.json").write_text(json.dumps(auth))

        with patch.object(runner._auth_checker, "codex_home", return_value=codex_home):
            status = runner._auth_checker.codex_auth_status()

        assert status.available is True
        assert status.auth_mode == _CODEX_AUTH_MODE

    def test_unavailable_when_auth_file_missing(self, runner: AgentRunner, tmp_path: Path):
        codex_home = tmp_path / ".codex-missing"
        codex_home.mkdir()

        with patch.object(runner._auth_checker, "codex_home", return_value=codex_home):
            status = runner._auth_checker.codex_auth_status()

        assert status.available is False
        assert "not found" in status.reason

    def test_unavailable_when_wrong_auth_mode(self, runner: AgentRunner, tmp_path: Path):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        auth = {"auth_mode": "api-key", "tokens": {"access_token": "tok"}}
        (codex_home / "auth.json").write_text(json.dumps(auth))

        with patch.object(runner._auth_checker, "codex_home", return_value=codex_home):
            status = runner._auth_checker.codex_auth_status()

        assert status.available is False
        assert "chatgpt auth mode" in status.reason

    def test_unavailable_when_openai_api_key_present(self, runner: AgentRunner, tmp_path: Path):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        auth = {
            "auth_mode": _CODEX_AUTH_MODE,
            "OPENAI_API_KEY": "sk-xxx",
            "tokens": {"access_token": "tok"},
        }
        (codex_home / "auth.json").write_text(json.dumps(auth))

        with patch.object(runner._auth_checker, "codex_home", return_value=codex_home):
            status = runner._auth_checker.codex_auth_status()

        assert status.available is False
        assert "OPENAI_API_KEY" in status.reason

    def test_unavailable_when_no_tokens(self, runner: AgentRunner, tmp_path: Path):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        auth = {"auth_mode": _CODEX_AUTH_MODE, "tokens": {}}
        (codex_home / "auth.json").write_text(json.dumps(auth))

        with patch.object(runner._auth_checker, "codex_home", return_value=codex_home):
            status = runner._auth_checker.codex_auth_status()

        assert status.available is False
        assert "tokens" in status.reason.lower()

    def test_unavailable_on_malformed_json(self, runner: AgentRunner, tmp_path: Path):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        (codex_home / "auth.json").write_text("{bad")

        with patch.object(runner._auth_checker, "codex_home", return_value=codex_home):
            status = runner._auth_checker.codex_auth_status()

        assert status.available is False
        assert "failed" in status.reason.lower()

    def test_codex_has_tokens_requires_access_or_refresh(self, runner: AgentRunner):
        assert runner._auth_checker.codex_has_tokens({"tokens": {"access_token": "a"}}) is True
        assert runner._auth_checker.codex_has_tokens({"tokens": {"refresh_token": "r"}}) is True
        assert runner._auth_checker.codex_has_tokens({"tokens": {}}) is False
        assert runner._auth_checker.codex_has_tokens({}) is False

    def test_runtime_diagnostics_includes_log_info(self, runner: AgentRunner, tmp_path: Path):
        codex_home = tmp_path / ".codex"
        codex_home.mkdir()
        logs_dir = codex_home / "logs"
        logs_dir.mkdir()
        (logs_dir / "codex-2026-03-14.log").write_text("debug info\n" * 100)

        with patch.object(runner._auth_checker, "codex_home", return_value=codex_home):
            diagnostics = runner._auth_checker.codex_runtime_diagnostics(codex_home)

        assert isinstance(diagnostics, list)


# ---------------------------------------------------------------------------
# Provider Status Caching
# ---------------------------------------------------------------------------


class TestProviderStatusCaching:
    def test_cached_status_returned_within_ttl(self, runner: AgentRunner):
        cached = ProviderReadiness(
            provider=AgentProvider.CLAUDE_MAX,
            available=True,
            runtime="claude_cli",
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = cached
        runner._auth_checker._provider_status_checked_at[AgentProvider.CLAUDE_MAX] = datetime.now(timezone.utc)

        result = runner._get_provider_status(AgentProvider.CLAUDE_MAX)
        assert result is cached

    def test_cache_expires_after_ttl(self, runner: AgentRunner):
        stale = ProviderReadiness(
            provider=AgentProvider.CLAUDE_MAX,
            available=False,
            runtime="claude_cli",
            reason="stale",
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = stale
        runner._auth_checker._provider_status_checked_at[AgentProvider.CLAUDE_MAX] = (
            datetime.now(timezone.utc)
            - timedelta(seconds=_PROVIDER_STATUS_CACHE_TTL_SECONDS + 10)
        )

        with patch.object(
            runner._auth_checker,
            "claude_max_auth_status",
            return_value=type(
                "_S", (), {"available": True, "reason": "", "auth_method": "claude.ai", "subscription_type": "max"},
            )(),
        ):
            result = runner._get_provider_status(AgentProvider.CLAUDE_MAX)

        assert result.available is True

    def test_invalidate_single_provider(self, runner: AgentRunner):
        cached = ProviderReadiness(
            provider=AgentProvider.CLAUDE_MAX, available=True, runtime="claude_cli",
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = cached
        runner._auth_checker._provider_status_checked_at[AgentProvider.CLAUDE_MAX] = datetime.now(timezone.utc)

        runner.invalidate_provider_status_cache(AgentProvider.CLAUDE_MAX)
        assert AgentProvider.CLAUDE_MAX not in runner._auth_checker._provider_status_cache

    def test_invalidate_all_providers(self, runner: AgentRunner):
        for prov in AgentProvider:
            runner._auth_checker._provider_status_cache[prov] = ProviderReadiness(
                provider=prov, available=True, runtime="cli",
            )
        runner.invalidate_provider_status_cache(None)
        assert len(runner._auth_checker._provider_status_cache) == 0


# ---------------------------------------------------------------------------
# Z.ai / OpenRouter Key Checks
# ---------------------------------------------------------------------------


class TestZaiOpenrouterStatus:
    def test_zai_available_when_key_and_cli_present(self, runner: AgentRunner):
        runner._auth_checker.zai_api_key = "zai-key-123"
        with patch.object(runner._auth_checker, "command_ready", return_value=(True, "")):
            status = runner._get_provider_status(AgentProvider.ZAI_CODING_PLAN)
        assert status.available is True

    def test_zai_unavailable_when_no_key(self, runner: AgentRunner):
        runner._auth_checker.zai_api_key = ""
        with patch.object(runner._auth_checker, "command_ready", return_value=(True, "")):
            status = runner._get_provider_status(AgentProvider.ZAI_CODING_PLAN)
        assert status.available is False
        assert "ZAI_API_KEY" in status.reason

    def test_openrouter_available_when_key_and_cli_present(self, runner: AgentRunner):
        runner._auth_checker.openrouter_api_key = "or-key-123"
        with patch.object(runner._auth_checker, "command_ready", return_value=(True, "")):
            status = runner._get_provider_status(AgentProvider.OPENROUTER)
        assert status.available is True

    def test_openrouter_unavailable_when_no_key(self, runner: AgentRunner):
        runner._auth_checker.openrouter_api_key = ""
        with patch.object(runner._auth_checker, "command_ready", return_value=(True, "")):
            status = runner._get_provider_status(AgentProvider.OPENROUTER)
        assert status.available is False
        assert "OPENROUTER_API_KEY" in status.reason

    def test_zai_unavailable_when_cli_missing(self, runner: AgentRunner):
        runner._auth_checker.zai_api_key = "key"
        with patch.object(runner._auth_checker, "command_ready", return_value=(False, "Command not found: claude")):
            status = runner._get_provider_status(AgentProvider.ZAI_CODING_PLAN)
        assert status.available is False
