from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from trading_assistant.orchestrator.agent_runner import AgentResult, AgentRunner
from trading_assistant.orchestrator.repo_task_runner import RepoTaskRunner
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
)
from trading_assistant.schemas.prompt_package import PromptPackage


def _ready(provider: AgentProvider) -> ProviderReadiness:
    return ProviderReadiness(provider=provider, available=True, runtime="claude_cli")


def _unready(provider: AgentProvider, reason: str) -> ProviderReadiness:
    return ProviderReadiness(provider=provider, available=False, runtime="claude_cli", reason=reason)


@pytest.fixture
def agent_runner(tmp_path: Path) -> AgentRunner:
    return AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
        preferences=AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet"),
            overrides={
                AgentWorkflow.WEEKLY_ANALYSIS: AgentSelection(
                    provider=AgentProvider.CODEX_PRO,
                    model="gpt-5.4",
                ),
            },
        ),
    )


@pytest.mark.asyncio
async def test_repo_task_runner_falls_back_from_codex_override(agent_runner: AgentRunner):
    agent_runner._auth_checker._provider_status_cache.update({
        AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
        AgentProvider.CODEX_PRO: ProviderReadiness(
            provider=AgentProvider.CODEX_PRO,
            available=True,
            runtime="codex_cli",
        ),
    })
    agent_runner.invoke_with_selection = AsyncMock(return_value=AgentResult(
        response="ok",
        run_dir=agent_runner._runs_dir / "repo-task",
    ))
    runner = RepoTaskRunner(agent_runner)

    result = await runner.invoke(
        agent_type="weekly_analysis",
        prompt_package=PromptPackage(task_prompt="Apply approved change."),
        run_id="repo-task",
        allowed_tools=["Read", "Edit"],
    )

    assert result.success is True
    call = agent_runner.invoke_with_selection.await_args
    assert call.kwargs["selection"].provider == AgentProvider.CLAUDE_MAX
    assert call.kwargs["requested_model"] is None


@pytest.mark.asyncio
async def test_repo_task_runner_falls_back_when_requested_provider_unavailable(agent_runner: AgentRunner):
    agent_runner._preferences.set_preferences(AgentPreferences(
        default=AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="sonnet"),
        overrides={
            AgentWorkflow.TRIAGE: AgentSelection(
                provider=AgentProvider.OPENROUTER,
                model="minimax/minimax-m2.5",
            ),
        },
    ))
    agent_runner._auth_checker._provider_status_cache.update({
        AgentProvider.OPENROUTER: _unready(AgentProvider.OPENROUTER, "missing API key"),
        AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
    })
    agent_runner.invoke_with_selection = AsyncMock(return_value=AgentResult(
        response="ok",
        run_dir=agent_runner._runs_dir / "repo-task",
    ))
    runner = RepoTaskRunner(agent_runner)

    await runner.invoke(
        agent_type="triage",
        prompt_package=PromptPackage(task_prompt="Apply approved fix."),
        run_id="repo-task",
    )

    call = agent_runner.invoke_with_selection.await_args
    assert call.kwargs["selection"].provider == AgentProvider.CLAUDE_MAX


@pytest.mark.asyncio
async def test_repo_task_runner_raises_when_no_write_provider_is_ready(agent_runner: AgentRunner):
    agent_runner._auth_checker._provider_status_cache.update({
        AgentProvider.CLAUDE_MAX: _unready(AgentProvider.CLAUDE_MAX, "not authenticated"),
        AgentProvider.OPENROUTER: _unready(AgentProvider.OPENROUTER, "missing API key"),
        AgentProvider.ZAI_CODING_PLAN: _unready(AgentProvider.ZAI_CODING_PLAN, "missing API key"),
        AgentProvider.CODEX_PRO: ProviderReadiness(
            provider=AgentProvider.CODEX_PRO,
            available=True,
            runtime="codex_cli",
        ),
    })
    runner = RepoTaskRunner(agent_runner)

    with pytest.raises(ValueError) as exc:
        await runner.invoke(
            agent_type="weekly_analysis",
            prompt_package=PromptPackage(task_prompt="Apply approved change."),
            run_id="repo-task",
        )

    assert "write-capable providers" in str(exc.value)
