"""Tests for SkillsRegistry enforcement in AgentRunner (A2)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from trading_assistant.orchestrator.agent_runner import AgentRunner
from trading_assistant.orchestrator.app import create_app
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.orchestrator.skills_registry import SkillsRegistry
from trading_assistant.schemas.agent_capabilities import AgentType
from trading_assistant.schemas.agent_preferences import AgentPreferences, AgentProvider, AgentSelection, ProviderReadiness


def _streaming_codex_process():
    process = AsyncMock()
    process.stdout = AsyncMock()
    process.stdout.readline = AsyncMock(side_effect=[b'{"item":{"type":"message","text":"ok"}}\n', b""])
    process.stderr = AsyncMock()
    process.stderr.read = AsyncMock(return_value=b"")
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    process.kill = AsyncMock()
    return process


@pytest.fixture
def app_with_tmp(tmp_path):
    config = AppConfig(
        data_dir=str(tmp_path),
        bot_ids=["bot1"],
        allow_unauthenticated_local=True,
    )
    return create_app(db_dir=str(tmp_path), config=config)


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(base_dir=str(tmp_path / "sessions"))


def test_registry_instantiated_in_create_app(app_with_tmp):
    assert hasattr(app_with_tmp.state, "skills_registry")
    assert isinstance(app_with_tmp.state.skills_registry, SkillsRegistry)


def test_agent_runner_accepts_skills_registry(tmp_path, session_store):
    registry = SkillsRegistry()
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
    )
    assert runner._skills_registry is registry


@pytest.mark.asyncio
async def test_invocation_with_allowed_agent_type_succeeds(tmp_path, session_store, sample_package):
    registry = SkillsRegistry()
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
        enforce_skills=False,
        preferences=AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO)
        ),
        codex_command="codex",
    )
    runner._auth_checker._provider_status_cache[AgentProvider.CODEX_PRO] = ProviderReadiness(
        provider=AgentProvider.CODEX_PRO,
        available=True,
        runtime="codex_cli",
    )
    mock_proc = _streaming_codex_process()
    # daily_analysis has generate_report in allowed_actions, so no denial
    with (
        patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"),
        patch("trading_assistant.orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        result = await runner.invoke(
            agent_type="daily_analysis",
            prompt_package=sample_package,
            run_id="test-run",
        )
    # Should reach invocation (may fail on CLI but not on permission)
    assert result.session_id.startswith("test-run-")


@pytest.mark.asyncio
async def test_invocation_with_forbidden_action_logs_warning(tmp_path, session_store, sample_package, caplog):
    """In soft mode (enforce=False), forbidden agent types log a warning but proceed."""
    from trading_assistant.orchestrator.skills_registry import SkillsRegistry
    from trading_assistant.schemas.agent_capabilities import AgentCapability, AgentType

    # Create a registry where 'daily_analysis' does NOT have 'generate_report'
    restricted = AgentCapability(
        agent_type=AgentType.DAILY_ANALYSIS,
        allowed_actions=["read_data"],
        forbidden_actions=["generate_report"],
    )
    registry = SkillsRegistry(overrides={AgentType.DAILY_ANALYSIS: restricted})
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
        enforce_skills=False,
        preferences=AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO)
        ),
        codex_command="codex",
    )
    runner._auth_checker._provider_status_cache[AgentProvider.CODEX_PRO] = ProviderReadiness(
        provider=AgentProvider.CODEX_PRO,
        available=True,
        runtime="codex_cli",
    )
    mock_proc = _streaming_codex_process()

    import logging
    with caplog.at_level(logging.WARNING):
        with (
            patch.object(runner._auth_checker, "require_resolved_command", return_value="/usr/bin/codex"),
            patch("trading_assistant.orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="test-soft",
            )
    assert "SkillsRegistry denied" in caplog.text
    # But invocation still proceeds
    assert result.session_id.startswith("test-soft-")


@pytest.mark.asyncio
async def test_invocation_with_forbidden_action_raises_in_enforce_mode(tmp_path, session_store, sample_package):
    from trading_assistant.schemas.agent_capabilities import AgentCapability, AgentType

    restricted = AgentCapability(
        agent_type=AgentType.DAILY_ANALYSIS,
        allowed_actions=["read_data"],
        forbidden_actions=["generate_report"],
    )
    registry = SkillsRegistry(overrides={AgentType.DAILY_ANALYSIS: restricted})
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
        enforce_skills=True,
    )

    with pytest.raises(PermissionError, match="SkillsRegistry denied"):
        await runner.invoke(
            agent_type="daily_analysis",
            prompt_package=sample_package,
            run_id="test-enforce",
        )


def test_missing_registry_skips_enforcement(tmp_path, session_store):
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
    )
    assert runner._skills_registry is None


def test_all_seven_agent_types_have_profiles():
    registry = SkillsRegistry()
    for agent_type in AgentType:
        result = registry.check_action(agent_type, "generate_report")
        # Should have a profile (either allowed or denied, not "unknown agent type")
        assert "Unknown agent type" not in result.reason
