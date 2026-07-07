"""Preference manager for agent runtime provider selection."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.orchestrator.provider_cooldown import ProviderCooldownTracker
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentPreferencesView,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
    WorkflowTuning,
)

DEFAULT_PROVIDER_MODELS: dict[AgentProvider, str] = {
    AgentProvider.CLAUDE_MAX: "sonnet",
    AgentProvider.CODEX_PRO: "gpt-5.4",
    AgentProvider.ZAI_CODING_PLAN: "glm-5",
    AgentProvider.OPENROUTER: "minimax/minimax-m2.5",
}

WORKFLOW_ORDER: tuple[AgentWorkflow, ...] = (
    AgentWorkflow.DAILY_ANALYSIS,
    AgentWorkflow.WEEKLY_ANALYSIS,
    AgentWorkflow.MONTHLY_VALIDATION,
    AgentWorkflow.MONTHLY_MODEL_REVIEW,
    AgentWorkflow.TRIAGE,
)

DEFAULT_WORKFLOW_TUNING: dict[AgentWorkflow, WorkflowTuning] = {
    AgentWorkflow.DAILY_ANALYSIS: WorkflowTuning(timeout_seconds=600, max_turns=5),
    AgentWorkflow.WEEKLY_ANALYSIS: WorkflowTuning(timeout_seconds=900, max_turns=8),
    AgentWorkflow.MONTHLY_VALIDATION: WorkflowTuning(timeout_seconds=1200, max_turns=6),
    AgentWorkflow.MONTHLY_MODEL_REVIEW: WorkflowTuning(timeout_seconds=900, max_turns=6),
    AgentWorkflow.TRIAGE: WorkflowTuning(timeout_seconds=300, max_turns=4),
}


class AgentPreferencesManager:
    """Manages persisted preferences and resolves effective selections."""

    def __init__(
        self,
        preferences: AgentPreferences | None = None,
        provider_status_resolver: Callable[[], list[ProviderReadiness]] | None = None,
        findings_dir: Path | None = None,
    ) -> None:
        self._preferences = preferences or AgentPreferences()
        self._provider_status_resolver = provider_status_resolver
        self._findings_dir = Path(findings_dir) if findings_dir is not None else None

    def set_provider_status_resolver(
        self, resolver: Callable[[], list[ProviderReadiness]] | None
    ) -> None:
        self._provider_status_resolver = resolver

    def get_preferences(self) -> AgentPreferences:
        return self._preferences.model_copy(deep=True)

    def set_preferences(self, preferences: AgentPreferences) -> None:
        self._preferences = preferences.model_copy(deep=True)

    def resolve_selection(
        self,
        workflow: AgentWorkflow | None,
        model_override: str | None = None,
        *,
        cooldown_tracker: ProviderCooldownTracker | None = None,
    ) -> tuple[AgentSelection, str | None]:
        requested, workflow_override = self._requested_selection(workflow)
        explicit_override = model_override.strip() if model_override and model_override.strip() else None
        requested_model = explicit_override or requested.model
        selection = AgentSelection(
            provider=requested.provider,
            model=requested_model or DEFAULT_PROVIDER_MODELS[requested.provider],
        )
        if workflow is not None and explicit_override is None and not workflow_override:
            learned = self._resolve_learned_selection(
                workflow,
                selection,
                cooldown_tracker=cooldown_tracker,
            )
            if learned is not None:
                selection = learned
                requested_model = selection.model
        return selection, requested_model

    def build_view(self) -> AgentPreferencesView:
        effective = {
            workflow: self.resolve_selection(workflow)[0]
            for workflow in WORKFLOW_ORDER
        }
        providers = self._provider_status_resolver() if self._provider_status_resolver else []
        return AgentPreferencesView(
            default=self._preferences.default.model_copy(deep=True),
            overrides={
                workflow: selection.model_copy(deep=True) if selection is not None else None
                for workflow, selection in self._preferences.overrides.items()
                if workflow in WORKFLOW_ORDER
            },
            effective=effective,
            providers=providers,
        )

    def unavailable_reasons(self, preferences: AgentPreferences) -> list[str]:
        if self._provider_status_resolver is None:
            return []
        provider_status = {
            status.provider: status
            for status in self._provider_status_resolver()
        }
        errors: list[str] = []

        default_status = provider_status.get(preferences.default.provider)
        if default_status and not default_status.available:
            errors.append(
                f"default provider '{preferences.default.provider.value}' is unavailable: "
                f"{default_status.reason or 'unknown reason'}"
            )

        for workflow, selection in preferences.overrides.items():
            if selection is None:
                continue
            status = provider_status.get(selection.provider)
            if status and not status.available:
                errors.append(
                    f"override for '{workflow.value}' uses unavailable provider "
                    f"'{selection.provider.value}': {status.reason or 'unknown reason'}"
                )

        return errors

    def resolve_tuning(
        self,
        workflow: AgentWorkflow | None,
        *,
        max_turns_override: int | None = None,
        allowed_tools_override: list[str] | None = None,
    ) -> WorkflowTuning:
        """Resolve effective tuning for a workflow.

        Merges: caller overrides > user preferences > built-in defaults.
        """
        default = DEFAULT_WORKFLOW_TUNING.get(workflow) if workflow else None
        user = self._preferences.workflow_tuning.get(workflow) if workflow else None

        def _attr(obj: WorkflowTuning | None, name: str):
            return getattr(obj, name, None) if obj is not None else None

        def _first_set(*values):
            for v in values:
                if v is not None:
                    return v
            return None

        timeout = _first_set(_attr(user, "timeout_seconds"), _attr(default, "timeout_seconds"))
        max_turns = _first_set(max_turns_override, _attr(user, "max_turns"), _attr(default, "max_turns"))
        allowed_tools = _first_set(allowed_tools_override, _attr(user, "allowed_tools"), _attr(default, "allowed_tools"))

        return WorkflowTuning(
            timeout_seconds=timeout,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
        )

    def resolve_with_fallbacks(
        self,
        workflow: AgentWorkflow | None,
        model_override: str | None = None,
        cooldown_tracker: ProviderCooldownTracker | None = None,
    ) -> list[tuple[AgentSelection, str | None]]:
        """Return ordered list of (selection, requested_model) candidates.

        The primary selection is first, followed by fallback_chain entries,
        skipping providers that are in cooldown or unavailable.
        """
        primary_selection, requested_model = self.resolve_selection(
            workflow,
            model_override,
            cooldown_tracker=cooldown_tracker,
        )

        # Gather provider availability
        provider_status: dict[AgentProvider, bool] = {}
        if self._provider_status_resolver:
            for status in self._provider_status_resolver():
                provider_status[status.provider] = status.available

        candidates: list[tuple[AgentSelection, str | None]] = []
        seen_providers: set[AgentProvider] = set()

        def _add_candidate(provider: AgentProvider, model: str | None, req_model: str | None) -> None:
            if provider in seen_providers:
                return
            seen_providers.add(provider)
            if cooldown_tracker and cooldown_tracker.is_cooled_down(provider):
                return
            if provider_status.get(provider) is False:
                return
            effective_model = model or DEFAULT_PROVIDER_MODELS.get(provider, "")
            candidates.append((
                AgentSelection(provider=provider, model=effective_model),
                req_model,
            ))

        # Primary first
        _add_candidate(primary_selection.provider, primary_selection.model, requested_model)

        configured, _workflow_override = self._requested_selection(workflow)
        explicit_model = model_override.strip() if model_override and model_override.strip() else None
        configured_model = explicit_model or configured.model or DEFAULT_PROVIDER_MODELS[configured.provider]
        if configured.provider != primary_selection.provider:
            _add_candidate(configured.provider, configured_model, configured_model)

        # Fallback chain
        for entry in self._preferences.fallback_chain:
            _add_candidate(entry.provider, entry.model, None)

        return candidates

    def record_committed_route_change(
        self,
        *,
        workflow: AgentWorkflow | None,
        selected: AgentSelection,
        model_override: str | None = None,
        agent_type: str = "",
        run_id: str = "",
        success: bool | None = None,
    ) -> None:
        """Record a learned route only after the recommended provider was tried."""

        if workflow is None:
            return
        explicit_model = model_override.strip() if model_override and model_override.strip() else None
        if explicit_model is not None:
            return
        requested, workflow_override = self._requested_selection(workflow)
        if workflow_override or selected.provider == requested.provider:
            return
        try:
            recommendation = self._provider_route_recommendation(workflow, requested)
            if recommendation is None or recommendation.get("provider") != selected.provider.value:
                return
            self._record_provider_route_change(
                workflow=workflow,
                requested=AgentSelection(
                    provider=requested.provider,
                    model=requested.model or DEFAULT_PROVIDER_MODELS[requested.provider],
                ),
                selected=selected,
                recommendation=recommendation,
                usage={
                    "agent_type": agent_type or workflow.value,
                    "run_id": run_id,
                    "success": success,
                },
            )
        except Exception:
            return

    def _requested_selection(self, workflow: AgentWorkflow | None) -> tuple[AgentSelection, bool]:
        if workflow is not None:
            override = self._preferences.overrides.get(workflow)
            if override is not None:
                return override, True
        return self._preferences.default, False

    def _resolve_learned_selection(
        self,
        workflow: AgentWorkflow,
        requested: AgentSelection,
        *,
        cooldown_tracker: ProviderCooldownTracker | None = None,
    ) -> AgentSelection | None:
        recommendation = self._provider_route_recommendation(workflow, requested)
        if recommendation is None:
            return None
        try:
            provider_value = recommendation.get("provider", "")
            provider = AgentProvider(provider_value)
            if cooldown_tracker and cooldown_tracker.is_cooled_down(provider):
                return None
            if self._provider_available(provider) is False:
                return None
            return AgentSelection(
                provider=provider,
                model=recommendation.get("model") or DEFAULT_PROVIDER_MODELS.get(provider, requested.model),
            )
        except Exception:
            return None

    def _provider_route_recommendation(
        self,
        workflow: AgentWorkflow,
        requested: AgentSelection,
    ) -> dict | None:
        if self._findings_dir is None:
            return None
        try:
            from trading_assistant.skills.provider_route_scorer import ProviderRouteScorer

            recommendation = ProviderRouteScorer(self._findings_dir).recommend_provider(
                workflow=workflow.value,
                requested_provider=requested.provider.value,
            )
            if not recommendation:
                return None
            provider_value = recommendation.get("provider", "")
            if not provider_value or provider_value == requested.provider.value:
                return None
            AgentProvider(provider_value)
            return recommendation
        except Exception:
            return None

    def _provider_available(self, provider: AgentProvider) -> bool | None:
        if self._provider_status_resolver is None:
            return None
        try:
            for status in self._provider_status_resolver():
                if status.provider == provider:
                    return status.available
        except Exception:
            return None
        return None

    def _record_provider_route_change(
        self,
        *,
        workflow: AgentWorkflow,
        requested: AgentSelection,
        selected: AgentSelection,
        recommendation: dict,
        usage: dict | None = None,
    ) -> None:
        if self._findings_dir is None:
            return
        try:
            previous_model = requested.model or DEFAULT_PROVIDER_MODELS.get(requested.provider, "")
            recommended_model = selected.model or DEFAULT_PROVIDER_MODELS.get(selected.provider, "")
            record = {
                "workflow": workflow.value,
                "previous_provider": requested.provider.value,
                "previous_model": previous_model,
                "recommended_provider": selected.provider.value,
                "recommended_model": recommended_model,
                "composite_score": recommendation.get("composite_score"),
                "requested_composite_score": recommendation.get("requested_composite_score"),
                "score_gap": recommendation.get("score_gap"),
                "sample_count": int(recommendation.get("sample_count") or 0),
                "benchmark_quality": recommendation.get("benchmark_quality"),
                "validation_pass_rate": recommendation.get("validation_pass_rate"),
                "outcome_quality": recommendation.get("outcome_quality"),
                "calibration_accuracy": recommendation.get("calibration_accuracy"),
                "rollback_condition": recommendation.get("rollback_condition")
                or (
                    f"revert to {requested.provider.value} if {workflow.value} learned-route evidence "
                    "falls below the override threshold or validation failures recur"
                ),
                "source": "provider_route_scorer",
            }
            if usage:
                record.update({
                    "first_used_agent_type": usage.get("agent_type") or workflow.value,
                    "first_used_run_id": usage.get("run_id") or "",
                    "first_attempt_success": usage.get("success"),
                    "committed_at": datetime.now(timezone.utc).isoformat(),
                })
            route_key_payload = {
                key: record[key]
                for key in [
                    "workflow",
                    "previous_provider",
                    "previous_model",
                    "recommended_provider",
                    "recommended_model",
                ]
            }
            route_change_id = hashlib.sha256(
                json.dumps(route_key_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            record["route_change_id"] = route_change_id
            record["recorded_at"] = datetime.now(timezone.utc).isoformat()

            path = self._findings_dir / "provider_route_changes.jsonl"
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                rewritten: list[str] = []
                updated_existing = False
                for line in lines:
                    if not line.strip():
                        rewritten.append(line)
                        continue
                    try:
                        existing = json.loads(line)
                        if existing.get("route_change_id") == route_change_id:
                            if existing.get("committed_at"):
                                return
                            existing.update(record)
                            rewritten.append(json.dumps(existing, sort_keys=True))
                            updated_existing = True
                            continue
                        rewritten.append(line)
                    except json.JSONDecodeError:
                        rewritten.append(line)
                if updated_existing:
                    self._findings_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text("\n".join(rewritten).rstrip() + "\n", encoding="utf-8")
                    return
            self._findings_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            return
