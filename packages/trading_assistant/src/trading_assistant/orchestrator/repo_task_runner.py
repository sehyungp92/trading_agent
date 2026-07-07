"""Write-capable runner for repo mutation tasks."""
from __future__ import annotations

import logging

from trading_assistant.orchestrator.agent_preferences import DEFAULT_PROVIDER_MODELS
from trading_assistant.orchestrator.agent_runner import AgentRunner, AgentResult
from trading_assistant.schemas.agent_preferences import AgentProvider, AgentSelection, AgentWorkflow
from trading_assistant.schemas.prompt_package import PromptPackage

logger = logging.getLogger(__name__)

_WRITE_CAPABLE_PROVIDERS: tuple[AgentProvider, ...] = (
    AgentProvider.CLAUDE_MAX,
    AgentProvider.OPENROUTER,
    AgentProvider.ZAI_CODING_PLAN,
)


class RepoTaskRunner:
    """Restricts repo write tasks to write-capable providers in v1."""

    def __init__(self, agent_runner: AgentRunner) -> None:
        self._agent_runner = agent_runner

    async def invoke(
        self,
        agent_type: str,
        prompt_package: PromptPackage,
        run_id: str,
        model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        workflow = self._agent_runner._resolve_workflow(agent_type)
        selection, requested_model = self._resolve_selection(agent_type, model, workflow=workflow)
        result = await self._agent_runner.invoke_with_selection(
            agent_type=agent_type,
            prompt_package=prompt_package,
            run_id=run_id,
            selection=selection,
            requested_model=requested_model,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
        )
        self._agent_runner._preferences.record_committed_route_change(
            workflow=workflow,
            selected=selection,
            model_override=model,
            agent_type=agent_type,
            run_id=run_id,
            success=result.success,
        )
        return result

    def _resolve_selection(
        self,
        agent_type: str,
        model: str | None,
        *,
        workflow: AgentWorkflow | None = None,
    ) -> tuple[AgentSelection, str | None]:
        workflow = workflow if workflow is not None else self._agent_runner._resolve_workflow(agent_type)
        requested, requested_model = self._agent_runner._preferences.resolve_selection(workflow, model)
        candidates = [(requested, requested_model)]

        default_selection, _ = self._agent_runner._preferences.resolve_selection(None)
        if default_selection.provider != requested.provider:
            candidates.append((default_selection, None))

        for provider in _WRITE_CAPABLE_PROVIDERS:
            if any(existing.provider == provider for existing, _ in candidates):
                continue
            candidates.append((
                AgentSelection(provider=provider, model=DEFAULT_PROVIDER_MODELS[provider]),
                None,
            ))

        reasons: list[str] = []
        for selection, candidate_requested_model in candidates:
            if selection.provider not in _WRITE_CAPABLE_PROVIDERS:
                reasons.append(f"{selection.provider.value}: read-only for repo tasks")
                continue

            status = self._agent_runner._get_provider_status(selection.provider)
            if status.available:
                if selection.provider != requested.provider:
                    logger.warning(
                        "RepoTaskRunner falling back from %s to %s for %s",
                        requested.provider.value,
                        selection.provider.value,
                        agent_type,
                    )
                return selection, candidate_requested_model

            reasons.append(
                f"{selection.provider.value}: {status.reason or 'unavailable'}",
            )

        raise ValueError(
            "RepoTaskRunner only supports write-capable providers in v1 and none are ready. "
            + "; ".join(reasons),
        )
