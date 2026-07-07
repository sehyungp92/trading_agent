"""Tests for CLI invocation building and env isolation in AgentRunner."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_assistant.orchestrator.agent_runner import AgentRunner
from trading_assistant.orchestrator.invocation_builder import (
    InvocationBuilder,
    _ANTHROPIC_ENV_KEYS_TO_CLEAR,
    _CODEX_RUNTIME,
    _CODEX_SANDBOX,
    _CLAUDE_RUNTIME,
    _OPENAI_ENV_KEYS_TO_CLEAR,
)
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import AgentProvider, AgentSelection
from trading_assistant.schemas.prompt_package import PromptPackage


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


# ---------------------------------------------------------------------------
# Claude CLI Arg Building
# ---------------------------------------------------------------------------


class TestBuildClaudeInvocation:
    def test_basic_claude_args(self, runner: AgentRunner, sample_package: PromptPackage):
        selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
            spec = runner._invocation_builder.build_claude(
                prompt_package=sample_package,
                selection=selection,
                requested_model=None,
                max_turns=5,
                allowed_tools=None,
            )
        assert spec.runtime == _CLAUDE_RUNTIME
        assert "-p" in spec.args
        assert "stream-json" in spec.args
        assert "--model" in spec.args
        model_idx = spec.args.index("--model")
        assert spec.args[model_idx + 1] == "sonnet"

    def test_system_prompt_appended(self, runner: AgentRunner, sample_package: PromptPackage):
        selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
            spec = runner._invocation_builder.build_claude(
                prompt_package=sample_package,
                selection=selection,
                requested_model=None,
                max_turns=5,
                allowed_tools=None,
            )
        assert "--append-system-prompt" in spec.args
        idx = spec.args.index("--append-system-prompt")
        assert spec.args[idx + 1] == sample_package.system_prompt

    def test_no_system_prompt_when_empty(self, runner: AgentRunner):
        pkg = PromptPackage(task_prompt="test", system_prompt="", data={})
        selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
            spec = runner._invocation_builder.build_claude(
                prompt_package=pkg,
                selection=selection,
                requested_model=None,
                max_turns=5,
                allowed_tools=None,
            )
        assert "--append-system-prompt" not in spec.args

    def test_allowed_tools_passed(self, runner: AgentRunner, sample_package: PromptPackage):
        selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
            spec = runner._invocation_builder.build_claude(
                prompt_package=sample_package,
                selection=selection,
                requested_model=None,
                max_turns=5,
                allowed_tools=["Read", "Write"],
            )
        assert "--allowed-tools" in spec.args
        idx = spec.args.index("--allowed-tools")
        assert spec.args[idx + 1] == "Read,Write"

    def test_max_turns_forwarded(self, runner: AgentRunner, sample_package: PromptPackage):
        selection = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
            spec = runner._invocation_builder.build_claude(
                prompt_package=sample_package,
                selection=selection,
                requested_model=None,
                max_turns=10,
                allowed_tools=None,
            )
        idx = spec.args.index("--max-turns")
        assert spec.args[idx + 1] == "10"


# ---------------------------------------------------------------------------
# Codex CLI Arg Building
# ---------------------------------------------------------------------------


class TestBuildCodexInvocation:
    def test_basic_codex_args(self, runner: AgentRunner, sample_package: PromptPackage):
        selection = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"):
            spec = runner._invocation_builder.build_codex(sample_package, selection, None)
        assert spec.runtime == _CODEX_RUNTIME
        assert "exec" in spec.args
        assert "--json" in spec.args
        assert "--sandbox" in spec.args
        assert _CODEX_SANDBOX in spec.args
        assert "--model" in spec.args

    def test_codex_full_prompt_includes_task(self, runner: AgentRunner, sample_package: PromptPackage):
        result = InvocationBuilder.build_full_prompt(sample_package)
        assert sample_package.task_prompt in result

    def test_codex_full_prompt_task_only_when_no_extras(self, runner: AgentRunner):
        pkg = PromptPackage(task_prompt="test task", system_prompt="", data={})
        result = InvocationBuilder.build_full_prompt(pkg)
        assert result == "test task"

    def test_codex_model_forwarded(self, runner: AgentRunner, sample_package: PromptPackage):
        selection = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"):
            spec = runner._invocation_builder.build_codex(sample_package, selection, None)
        idx = spec.args.index("--model")
        assert spec.args[idx + 1] == "gpt-5.4"


# ---------------------------------------------------------------------------
# Env Isolation
# ---------------------------------------------------------------------------


class TestEnvIsolation:
    def test_anthropic_keys_cleared_for_claude(self, runner: AgentRunner):
        env_with_keys = {k: "test-val" for k in _ANTHROPIC_ENV_KEYS_TO_CLEAR}
        with patch.dict(os.environ, env_with_keys, clear=False):
            result = runner._invocation_builder.build_env(clear_keys=_ANTHROPIC_ENV_KEYS_TO_CLEAR)
        for key in _ANTHROPIC_ENV_KEYS_TO_CLEAR:
            assert key not in result

    def test_openai_keys_cleared_for_codex(self, runner: AgentRunner):
        env_with_keys = {k: "test-val" for k in _OPENAI_ENV_KEYS_TO_CLEAR}
        with patch.dict(os.environ, env_with_keys, clear=False):
            result = runner._invocation_builder.build_env(clear_keys=_OPENAI_ENV_KEYS_TO_CLEAR)
        for key in _OPENAI_ENV_KEYS_TO_CLEAR:
            assert key not in result

    def test_overrides_applied_after_clear(self, runner: AgentRunner):
        overrides = {"ANTHROPIC_AUTH_TOKEN": "new-token"}
        with patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "old"}, clear=False):
            result = runner._invocation_builder.build_env(overrides, clear_keys=_ANTHROPIC_ENV_KEYS_TO_CLEAR)
        assert result["ANTHROPIC_AUTH_TOKEN"] == "new-token"

    def test_existing_env_preserved(self, runner: AgentRunner):
        with patch.dict(os.environ, {"MY_CUSTOM_VAR": "keep"}, clear=False):
            result = runner._invocation_builder.build_env(clear_keys=_ANTHROPIC_ENV_KEYS_TO_CLEAR)
        assert result["MY_CUSTOM_VAR"] == "keep"


# ---------------------------------------------------------------------------
# Model Resolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_claude_max_passes_model_directly(self, runner: AgentRunner):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="opus")
        assert runner._invocation_builder.resolve_claude_cli_model(sel) == "opus"

    def test_zai_maps_opus_to_opus(self, runner: AgentRunner):
        sel = AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model="glm-5-opus")
        assert runner._invocation_builder.resolve_claude_cli_model(sel) == "opus"

    def test_zai_maps_haiku_to_haiku(self, runner: AgentRunner):
        sel = AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model="glm-4.5-air")
        assert runner._invocation_builder.resolve_claude_cli_model(sel) == "haiku"

    def test_zai_defaults_to_sonnet(self, runner: AgentRunner):
        sel = AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model="glm-5")
        assert runner._invocation_builder.resolve_claude_cli_model(sel) == "sonnet"

    def test_openrouter_maps_to_sonnet(self, runner: AgentRunner):
        sel = AgentSelection(
            provider=AgentProvider.OPENROUTER,
            model="minimax/minimax-m2.5",
        )
        assert runner._invocation_builder.resolve_claude_cli_model(sel) == "sonnet"

    def test_default_model_used_when_selection_has_none(self, runner: AgentRunner):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model=None)
        result = runner._invocation_builder.resolve_claude_cli_model(sel)
        assert result == runner._default_model


# ---------------------------------------------------------------------------
# Claude Env Overrides
# ---------------------------------------------------------------------------


class TestClaudeEnvOverrides:
    def test_zai_overrides_include_auth_and_base_url(self, runner: AgentRunner):
        runner._auth_checker.zai_api_key = "zai-key"
        sel = AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model="glm-5")
        overrides = runner._invocation_builder.claude_env_overrides(sel)
        assert overrides["ANTHROPIC_AUTH_TOKEN"] == "zai-key"
        assert "z.ai" in overrides["ANTHROPIC_BASE_URL"]
        assert overrides["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5"
        assert "API_TIMEOUT_MS" in overrides

    def test_openrouter_overrides_include_auth_and_base_url(self, runner: AgentRunner):
        runner._auth_checker.openrouter_api_key = "or-key"
        sel = AgentSelection(
            provider=AgentProvider.OPENROUTER, model="minimax/minimax-m2.5",
        )
        overrides = runner._invocation_builder.claude_env_overrides(sel)
        assert overrides["ANTHROPIC_AUTH_TOKEN"] == "or-key"
        assert "openrouter.ai" in overrides["ANTHROPIC_BASE_URL"]

    def test_claude_max_returns_empty_overrides(self, runner: AgentRunner):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet")
        assert runner._invocation_builder.claude_env_overrides(sel) == {}

    def test_zai_uses_default_model_when_none(self, runner: AgentRunner):
        runner._auth_checker.zai_api_key = "key"
        sel = AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model=None)
        overrides = runner._invocation_builder.claude_env_overrides(sel)
        assert overrides["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-5"

    def test_openrouter_uses_default_model_when_none(self, runner: AgentRunner):
        runner._auth_checker.openrouter_api_key = "key"
        sel = AgentSelection(provider=AgentProvider.OPENROUTER, model=None)
        overrides = runner._invocation_builder.claude_env_overrides(sel)
        assert overrides["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "minimax/minimax-m2.5"


# ---------------------------------------------------------------------------
# Workflow Mapping
# ---------------------------------------------------------------------------


class TestWorkflowMapping:
    def test_known_agent_types_resolve(self, runner: AgentRunner):
        from trading_assistant.schemas.agent_preferences import AgentWorkflow

        assert runner._resolve_workflow("daily_analysis") == AgentWorkflow.DAILY_ANALYSIS
        assert runner._resolve_workflow("weekly_analysis") == AgentWorkflow.WEEKLY_ANALYSIS
        assert runner._resolve_workflow("monthly_validation") == AgentWorkflow.MONTHLY_VALIDATION
        assert runner._resolve_workflow("monthly_model_review") == AgentWorkflow.MONTHLY_MODEL_REVIEW
        assert runner._resolve_workflow("triage") == AgentWorkflow.TRIAGE
        assert runner._resolve_workflow("wfo") is None

    def test_unknown_agent_type_returns_none(self, runner: AgentRunner):
        assert runner._resolve_workflow("bug_report") is None
        assert runner._resolve_workflow("heartbeat") is None

    def test_build_invocation_routes_codex_to_codex_builder(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        sel = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"):
            spec = runner._invocation_builder.build(
                prompt_package=sample_package,
                selection=sel,
                requested_model=None,
                max_turns=5,
                allowed_tools=None,
            )
        assert spec.runtime == _CODEX_RUNTIME

    def test_build_invocation_routes_claude_providers_to_claude_builder(
        self, runner: AgentRunner, sample_package: PromptPackage,
    ):
        for provider in (
            AgentProvider.CLAUDE_MAX,
            AgentProvider.ZAI_CODING_PLAN,
            AgentProvider.OPENROUTER,
        ):
            sel = AgentSelection(provider=provider, model="sonnet")
            with patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
                runner._auth_checker.zai_api_key = "k"
                runner._auth_checker.openrouter_api_key = "k"
                spec = runner._invocation_builder.build(
                    prompt_package=sample_package,
                    selection=sel,
                    requested_model=None,
                    max_turns=5,
                    allowed_tools=None,
                )
            assert spec.runtime == _CLAUDE_RUNTIME
