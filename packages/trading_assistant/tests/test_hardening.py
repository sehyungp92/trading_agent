"""Tests for Phase 6 hardening improvements."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trading_assistant.orchestrator.agent_runner import AgentRunner
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.invocation_builder import InvocationBuilder
from trading_assistant.orchestrator.provider_auth import ProviderAuthChecker
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import (
    AgentProvider,
    AgentSelection,
)
from trading_assistant.schemas.prompt_package import PromptPackage


@pytest.fixture
def builder() -> InvocationBuilder:
    auth = ProviderAuthChecker(
        claude_command="claude",
        codex_command="codex",
    )
    return InvocationBuilder(
        auth_checker=auth,
        claude_command="claude",
        codex_command="codex",
    )


# ---------------------------------------------------------------------------
# Codex --instructions flag
# ---------------------------------------------------------------------------


class TestCodexInstructionsFlag:
    def test_system_prompt_passed_via_instructions(self, builder: InvocationBuilder, sample_package: PromptPackage):
        selection = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        with patch.object(builder._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"):
            spec = builder.build_codex(sample_package, selection, None)
        assert "--instructions" in spec.args
        idx = spec.args.index("--instructions")
        assert spec.args[idx + 1] == "You are a trading analyst."
        # build_full_prompt merges task_prompt + data file manifest
        assert spec.args[-1].startswith("Analyze today's performance.")

    def test_no_instructions_flag_when_system_prompt_empty(self, builder: InvocationBuilder):
        pkg = PromptPackage(task_prompt="Just do the task.", system_prompt="", data={})
        selection = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        with patch.object(builder._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"):
            spec = builder.build_codex(pkg, selection, None)
        assert "--instructions" not in spec.args
        assert spec.args[-1] == "Just do the task."

    def test_no_instructions_flag_when_system_prompt_whitespace(self, builder: InvocationBuilder):
        pkg = PromptPackage(task_prompt="Task.", system_prompt="   ", data={})
        selection = AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")
        with patch.object(builder._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"):
            spec = builder.build_codex(pkg, selection, None)
        assert "--instructions" not in spec.args


# ---------------------------------------------------------------------------
# DEFAULT_PROVIDER_MODELS single source of truth
# ---------------------------------------------------------------------------


class TestDefaultProviderModels:
    def test_invocation_builder_imports_from_preferences(self):
        """Verify invocation_builder uses the canonical DEFAULT_PROVIDER_MODELS."""
        from trading_assistant.orchestrator.agent_preferences import DEFAULT_PROVIDER_MODELS
        from trading_assistant.orchestrator.invocation_builder import DEFAULT_PROVIDER_MODELS as IB_MODELS
        assert IB_MODELS is DEFAULT_PROVIDER_MODELS

    def test_zai_default_model_matches(self, builder: InvocationBuilder, sample_package: PromptPackage):
        from trading_assistant.orchestrator.agent_preferences import DEFAULT_PROVIDER_MODELS
        selection = AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN)
        with patch.object(builder._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
            spec = builder.build_claude(sample_package, selection, None, max_turns=None, allowed_tools=None)
        assert spec.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == DEFAULT_PROVIDER_MODELS[AgentProvider.ZAI_CODING_PLAN]

    def test_openrouter_default_model_matches(self, builder: InvocationBuilder, sample_package: PromptPackage):
        from trading_assistant.orchestrator.agent_preferences import DEFAULT_PROVIDER_MODELS
        builder._auth_checker.openrouter_api_key = "sk-test"
        selection = AgentSelection(provider=AgentProvider.OPENROUTER)
        with patch.object(builder._auth_checker, "require_resolved_command", return_value="/usr/bin/claude"):
            spec = builder.build_claude(sample_package, selection, None, max_turns=None, allowed_tools=None)
        assert spec.env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == DEFAULT_PROVIDER_MODELS[AgentProvider.OPENROUTER]


# ---------------------------------------------------------------------------
# Cost tracker optional wiring
# ---------------------------------------------------------------------------


class TestCostTrackerOptional:
    def test_runner_works_without_cost_tracker(self, tmp_path: Path):
        """AgentRunner initializes fine without cost_tracker."""
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            event_stream=EventStream(),
        )
        assert runner._cost_tracker is None

    def test_runner_accepts_cost_tracker(self, tmp_path: Path):
        from trading_assistant.orchestrator.cost_tracker import CostTracker
        ct = CostTracker(tmp_path / "costs.jsonl")
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            event_stream=EventStream(),
            cost_tracker=ct,
        )
        assert runner._cost_tracker is ct
