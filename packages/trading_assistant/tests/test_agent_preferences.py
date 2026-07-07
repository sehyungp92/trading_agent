"""Tests for agent preferences: manager, integration (seeding/loading/saving), and API."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from trading_assistant.orchestrator.agent_preferences import (
    DEFAULT_PROVIDER_MODELS,
    AgentPreferencesManager,
    WORKFLOW_ORDER,
)
from trading_assistant.orchestrator.provider_cooldown import ProviderCooldownTracker
from trading_assistant.orchestrator.app import (
    _load_agent_preferences,
    _requires_monthly_outcome,
    _save_agent_preferences,
    _seed_agent_preferences,
    _selection_from_env,
    create_app,
)
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentPreferencesView,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
)


def _readiness(
    provider: AgentProvider,
    available: bool = True,
    reason: str = "",
) -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(
        provider=provider, available=available, runtime=runtime, reason=reason,
    )


class TestResolveSelection:
    def test_default_provider_when_no_workflow(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(workflow=None)
        assert sel.provider == AgentProvider.CODEX_PRO
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.CODEX_PRO]
        assert req_model is None

    def test_default_provider_when_workflow_has_no_override(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)
        assert sel.provider == AgentProvider.CODEX_PRO
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.CODEX_PRO]

    def test_override_provider_for_workflow(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            overrides={
                AgentWorkflow.MONTHLY_MODEL_REVIEW: AgentSelection(
                    provider=AgentProvider.CODEX_PRO, model="gpt-5.4",
                ),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.MONTHLY_MODEL_REVIEW)
        assert sel.provider == AgentProvider.CODEX_PRO
        assert sel.model == "gpt-5.4"

    def test_non_overridden_workflow_falls_back_to_default(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
            overrides={
                AgentWorkflow.MONTHLY_MODEL_REVIEW: AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)
        assert sel.provider == AgentProvider.CODEX_PRO

    def test_model_override_takes_precedence(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(
            AgentWorkflow.DAILY_ANALYSIS, model_override="opus",
        )
        assert sel.model == "opus"
        assert req_model == "opus"

    def test_blank_model_override_ignored(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(
            AgentWorkflow.DAILY_ANALYSIS, model_override="   ",
        )
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.CODEX_PRO]
        assert req_model is None

    def test_none_model_override_ignored(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(
            AgentWorkflow.DAILY_ANALYSIS, model_override=None,
        )
        assert req_model is None

    def test_model_from_selection_used_when_no_override(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.OPENROUTER, model="custom/model"),
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.TRIAGE)
        assert sel.model == "custom/model"

    def test_default_model_filled_when_selection_has_none(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model=None),
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.WEEKLY_ANALYSIS)
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.ZAI_CODING_PLAN]


class TestBuildView:
    def test_view_includes_effective_for_all_workflows(self):
        mgr = AgentPreferencesManager()
        view = mgr.build_view()
        assert isinstance(view, AgentPreferencesView)
        assert set(view.effective.keys()) == set(WORKFLOW_ORDER)
        assert AgentWorkflow.MONTHLY_VALIDATION in view.effective
        assert AgentWorkflow.MONTHLY_MODEL_REVIEW in view.effective
        for workflow in WORKFLOW_ORDER:
            assert view.effective[workflow].provider == AgentProvider.CODEX_PRO

    def test_view_reflects_overrides(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(provider=AgentProvider.CODEX_PRO),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        view = mgr.build_view()
        assert view.effective[AgentWorkflow.TRIAGE].provider == AgentProvider.CODEX_PRO
        assert view.effective[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.CLAUDE_MAX

    def test_view_includes_provider_statuses_when_resolver_set(self):
        statuses = [_readiness(AgentProvider.CLAUDE_MAX, True)]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        view = mgr.build_view()
        assert len(view.providers) == 1
        assert view.providers[0].provider == AgentProvider.CLAUDE_MAX
        assert view.providers[0].available is True

    def test_view_empty_providers_when_no_resolver(self):
        mgr = AgentPreferencesManager()
        view = mgr.build_view()
        assert view.providers == []

    def test_view_overrides_deep_copied(self):
        prefs = AgentPreferences(
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(
                    provider=AgentProvider.CODEX_PRO, model="gpt-5.4",
                ),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        view = mgr.build_view()
        assert AgentWorkflow.TRIAGE in view.overrides
        # Mutating the view should not affect the manager
        view.overrides[AgentWorkflow.TRIAGE].model = "mutated"
        sel, _ = mgr.resolve_selection(AgentWorkflow.TRIAGE)
        assert sel.model == "gpt-5.4"

    def test_view_hides_legacy_wfo_overrides(self):
        prefs = AgentPreferences.model_validate({
            "overrides": {
                "wfo": {"provider": "codex_pro"},
            },
        })
        view = AgentPreferencesManager(preferences=prefs).build_view()
        assert "wfo" not in {workflow.value for workflow in view.effective}
        assert "wfo" not in {workflow.value for workflow in view.overrides}


class TestUnavailableReasons:
    def test_no_resolver_returns_empty(self):
        mgr = AgentPreferencesManager()
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
        )
        assert mgr.unavailable_reasons(prefs) == []

    def test_available_default_no_errors(self):
        statuses = [_readiness(AgentProvider.CLAUDE_MAX, True)]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences()
        assert mgr.unavailable_reasons(prefs) == []

    def test_unavailable_default_returns_error(self):
        statuses = [_readiness(AgentProvider.CODEX_PRO, False, "auth missing")]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
        )
        errors = mgr.unavailable_reasons(prefs)
        assert len(errors) == 1
        assert "codex_pro" in errors[0]
        assert "auth missing" in errors[0]

    def test_unavailable_override_returns_error(self):
        statuses = [
            _readiness(AgentProvider.CLAUDE_MAX, True),
            _readiness(AgentProvider.ZAI_CODING_PLAN, False, "no key"),
        ]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences(
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(
                    provider=AgentProvider.ZAI_CODING_PLAN,
                ),
            },
        )
        errors = mgr.unavailable_reasons(prefs)
        assert len(errors) == 1
        assert "triage" in errors[0]
        assert "zai_coding_plan" in errors[0]

    def test_none_override_skipped(self):
        statuses = [_readiness(AgentProvider.CLAUDE_MAX, True)]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences.model_validate({"overrides": {"wfo": None}})
        assert mgr.unavailable_reasons(prefs) == []


class TestGetSetPreferences:
    def test_get_returns_deep_copy(self):
        mgr = AgentPreferencesManager()
        prefs = mgr.get_preferences()
        prefs.default.model = "mutated"
        assert mgr.get_preferences().default.model is None

    def test_set_replaces_preferences(self):
        mgr = AgentPreferencesManager()
        new_prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4"),
        )
        mgr.set_preferences(new_prefs)
        assert mgr.get_preferences().default.provider == AgentProvider.CODEX_PRO

    def test_set_preferences_deep_copies(self):
        mgr = AgentPreferencesManager()
        new_prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
        )
        mgr.set_preferences(new_prefs)
        new_prefs.default.model = "mutated-after-set"
        assert mgr.get_preferences().default.model is None

    def test_set_provider_status_resolver(self):
        mgr = AgentPreferencesManager()
        assert mgr.build_view().providers == []
        mgr.set_provider_status_resolver(
            lambda: [_readiness(AgentProvider.CLAUDE_MAX)],
        )
        assert len(mgr.build_view().providers) == 1


class TestModelNormalization:
    def test_whitespace_model_normalized_to_none(self):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="   ")
        assert sel.model is None


class TestLearnedRoutingOverride:
    def _write_scores(self, findings: Path, entries: list[dict]) -> None:
        findings.mkdir(parents=True, exist_ok=True)
        with (findings / "provider_route_scores.jsonl").open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")

    def test_overrides_when_evidence_is_strong(self, tmp_path):
        findings = tmp_path / "memory" / "findings"
        self._write_scores(findings, [
            {
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            },
            {
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.72,
                "sample_count": 7,
            },
        ])
        mgr = AgentPreferencesManager(
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX)
            ),
            findings_dir=findings,
        )

        selection, _ = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)

        assert selection.provider == AgentProvider.CODEX_PRO
        assert selection.model == "gpt-5.4"
        assert not (findings / "provider_route_changes.jsonl").exists()

        mgr.record_committed_route_change(
            workflow=AgentWorkflow.DAILY_ANALYSIS,
            selected=selection,
            agent_type="daily_analysis",
            run_id="daily-2026-05-29",
            success=True,
        )
        changes = [
            json.loads(line)
            for line in (findings / "provider_route_changes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(changes) == 1
        assert changes[0]["workflow"] == "daily_analysis"
        assert changes[0]["previous_provider"] == "claude_max"
        assert changes[0]["recommended_provider"] == "codex_pro"
        assert changes[0]["score_gap"] == 0.2
        assert changes[0]["sample_count"] == 7
        assert changes[0]["rollback_condition"]
        assert changes[0]["first_used_run_id"] == "daily-2026-05-29"
        assert changes[0]["first_attempt_success"] is True

        mgr.record_committed_route_change(
            workflow=AgentWorkflow.DAILY_ANALYSIS,
            selected=selection,
            agent_type="daily_analysis",
            run_id="daily-2026-05-30",
            success=True,
        )
        changes_after_repeat = [
            json.loads(line)
            for line in (findings / "provider_route_changes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(changes_after_repeat) == 1

    def test_build_view_previews_learned_routing_without_audit(self, tmp_path):
        findings = tmp_path / "memory" / "findings"
        self._write_scores(findings, [
            {
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            },
            {
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.90,
                "sample_count": 7,
            },
        ])
        mgr = AgentPreferencesManager(
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX)
            ),
            findings_dir=findings,
        )

        view = mgr.build_view()

        assert view.effective[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.CODEX_PRO
        assert not (findings / "provider_route_changes.jsonl").exists()

    def test_committed_route_change_is_stable_when_score_samples_change(self, tmp_path):
        findings = tmp_path / "memory" / "findings"
        self._write_scores(findings, [
            {
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            },
            {
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.90,
                "sample_count": 7,
            },
        ])
        mgr = AgentPreferencesManager(
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX)
            ),
            findings_dir=findings,
        )
        selection, _ = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)
        mgr.record_committed_route_change(
            workflow=AgentWorkflow.DAILY_ANALYSIS,
            selected=selection,
            agent_type="daily_analysis",
            run_id="daily-2026-05-29",
            success=True,
        )

        self._write_scores(findings, [
            {
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            },
            {
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.93,
                "benchmark_quality": 0.98,
                "sample_count": 12,
            },
        ])
        selection, _ = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)
        mgr.record_committed_route_change(
            workflow=AgentWorkflow.DAILY_ANALYSIS,
            selected=selection,
            agent_type="daily_analysis",
            run_id="daily-2026-05-30",
            success=True,
        )

        changes = [
            json.loads(line)
            for line in (findings / "provider_route_changes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(changes) == 1
        assert changes[0]["first_used_run_id"] == "daily-2026-05-29"

    def test_does_not_override_without_requested_baseline_evidence(self, tmp_path):
        findings = tmp_path / "memory" / "findings"
        self._write_scores(findings, [{
            "workflow": "daily_analysis",
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "composite_score": 0.72,
            "sample_count": 7,
        }])
        mgr = AgentPreferencesManager(
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX)
            ),
            findings_dir=findings,
        )

        selection, _ = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)

        assert selection.provider == AgentProvider.CLAUDE_MAX

    def test_learned_routing_does_not_override_explicit_workflow_choice(self, tmp_path):
        findings = tmp_path / "memory" / "findings"
        self._write_scores(findings, [
            {
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            },
            {
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.90,
                "sample_count": 7,
            },
        ])
        prefs = AgentPreferences(overrides={
            AgentWorkflow.DAILY_ANALYSIS: AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="opus"),
        })
        mgr = AgentPreferencesManager(preferences=prefs, findings_dir=findings)

        selection, _ = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)

        assert selection.provider == AgentProvider.CLAUDE_MAX
        assert selection.model == "opus"
        assert not (findings / "provider_route_changes.jsonl").exists()

    def test_learned_routing_skips_cooled_down_recommendation(self, tmp_path):
        findings = tmp_path / "memory" / "findings"
        self._write_scores(findings, [
            {
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            },
            {
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.90,
                "sample_count": 7,
            },
        ])
        cooldown = ProviderCooldownTracker()
        cooldown.record_failure(AgentProvider.CODEX_PRO)
        mgr = AgentPreferencesManager(
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX)
            ),
            findings_dir=findings,
        )

        candidates = mgr.resolve_with_fallbacks(
            AgentWorkflow.DAILY_ANALYSIS,
            cooldown_tracker=cooldown,
        )

        assert [selection.provider for selection, _ in candidates] == [AgentProvider.CLAUDE_MAX]
        assert not (findings / "provider_route_changes.jsonl").exists()

    def test_learned_routing_keeps_configured_provider_as_fallback(self, tmp_path):
        findings = tmp_path / "memory" / "findings"
        self._write_scores(findings, [
            {
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            },
            {
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.90,
                "sample_count": 7,
            },
        ])
        mgr = AgentPreferencesManager(
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX)
            ),
            findings_dir=findings,
        )

        candidates = mgr.resolve_with_fallbacks(AgentWorkflow.DAILY_ANALYSIS)

        assert [selection.provider for selection, _ in candidates] == [
            AgentProvider.CODEX_PRO,
            AgentProvider.CLAUDE_MAX,
        ]

    def test_stripped_model_preserved(self):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="  opus  ")
        assert sel.model == "opus"

    def test_none_model_stays_none(self):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model=None)
        assert sel.model is None


# ---------------------------------------------------------------------------
# Integration tests: seeding, loading, saving, env config
# (merged from test_agent_preferences_integration.py)
# ---------------------------------------------------------------------------


class TestSelectionFromEnv:
    def test_valid_provider(self):
        sel = _selection_from_env("claude_max")
        assert sel is not None
        assert sel.provider == AgentProvider.CLAUDE_MAX
        assert sel.model is None

    def test_valid_provider_with_model(self):
        sel = _selection_from_env("codex_pro", "gpt-5.4")
        assert sel is not None
        assert sel.provider == AgentProvider.CODEX_PRO
        assert sel.model == "gpt-5.4"

    def test_empty_string_returns_none(self):
        assert _selection_from_env("") is None
        assert _selection_from_env("  ") is None

    def test_invalid_provider_returns_none(self):
        assert _selection_from_env("invalid_provider") is None

    def test_whitespace_model_normalised_to_none(self):
        sel = _selection_from_env("claude_max", "   ")
        assert sel is not None
        assert sel.model is None

    def test_case_insensitive(self):
        sel = _selection_from_env("CLAUDE_MAX")
        assert sel is not None
        assert sel.provider == AgentProvider.CLAUDE_MAX

    def test_zai_provider(self):
        sel = _selection_from_env("zai_coding_plan", "glm-5")
        assert sel is not None
        assert sel.provider == AgentProvider.ZAI_CODING_PLAN

    def test_openrouter_provider(self):
        sel = _selection_from_env("openrouter", "minimax/minimax-m2.5")
        assert sel is not None
        assert sel.provider == AgentProvider.OPENROUTER


class TestSeedAgentPreferences:
    def test_defaults_to_codex_pro(self):
        config = AppConfig()
        prefs = _seed_agent_preferences(config)
        assert prefs.default.provider == AgentProvider.CODEX_PRO
        assert len(prefs.overrides) == 0

    def test_custom_default_provider(self):
        config = AppConfig(agent_default_provider="codex_pro", agent_default_model="gpt-5.4")
        prefs = _seed_agent_preferences(config)
        assert prefs.default.provider == AgentProvider.CODEX_PRO
        assert prefs.default.model == "gpt-5.4"

    def test_per_workflow_overrides(self):
        config = AppConfig(
            daily_agent_provider="zai_coding_plan",
            daily_agent_model="glm-5",
            weekly_agent_provider="codex_pro",
            weekly_agent_model="gpt-5.4",
            monthly_model_review_agent_provider="openrouter",
            monthly_model_review_agent_model="custom/reviewer",
        )
        prefs = _seed_agent_preferences(config)
        assert AgentWorkflow.DAILY_ANALYSIS in prefs.overrides
        assert prefs.overrides[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.ZAI_CODING_PLAN
        assert AgentWorkflow.WEEKLY_ANALYSIS in prefs.overrides
        assert prefs.overrides[AgentWorkflow.WEEKLY_ANALYSIS].provider == AgentProvider.CODEX_PRO
        assert AgentWorkflow.MONTHLY_MODEL_REVIEW in prefs.overrides
        assert prefs.overrides[AgentWorkflow.MONTHLY_MODEL_REVIEW].provider == AgentProvider.OPENROUTER

    def test_empty_workflow_overrides_not_added(self):
        config = AppConfig(daily_agent_provider="", triage_agent_provider="")
        prefs = _seed_agent_preferences(config)
        assert len(prefs.overrides) == 0

    def test_invalid_default_falls_back_to_codex_pro(self):
        config = AppConfig(agent_default_provider="nonsense")
        prefs = _seed_agent_preferences(config)
        assert prefs.default.provider == AgentProvider.CODEX_PRO


class TestLoadSavePreferences:
    def test_load_from_disk(self, tmp_path: Path):
        prefs_path = tmp_path / "agent_preferences.json"
        prefs_path.write_text(json.dumps({
            "default": {"provider": "codex_pro", "model": "gpt-5.4"},
            "overrides": {
                "wfo": {"provider": "openrouter", "model": "minimax/minimax-m2.5"},
            },
        }), encoding="utf-8")

        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == AgentProvider.CODEX_PRO
        assert "wfo" not in {workflow.value for workflow in loaded.overrides}

    def test_load_falls_back_to_seed_on_missing(self, tmp_path: Path):
        prefs_path = tmp_path / "missing.json"
        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == AgentProvider.CODEX_PRO

    def test_load_falls_back_on_corrupt_file(self, tmp_path: Path):
        prefs_path = tmp_path / "corrupt.json"
        prefs_path.write_text("{bad json", encoding="utf-8")
        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == AgentProvider.CODEX_PRO

    def test_save_creates_file(self, tmp_path: Path):
        prefs_path = tmp_path / "data" / "agent_preferences.json"
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model="glm-5"),
        )
        _save_agent_preferences(prefs, prefs_path)
        assert prefs_path.exists()
        stored = json.loads(prefs_path.read_text(encoding="utf-8"))
        assert stored["default"]["provider"] == "zai_coding_plan"

    def test_round_trip(self, tmp_path: Path):
        prefs_path = tmp_path / "prefs.json"
        original = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.OPENROUTER, model="custom/model"),
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            },
        )
        _save_agent_preferences(original, prefs_path)
        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == original.default.provider
        assert loaded.default.model == original.default.model
        assert loaded.overrides[AgentWorkflow.TRIAGE].provider == AgentProvider.CLAUDE_MAX


class TestAppConfigEnvVars:
    def test_from_env_reads_agent_settings(self, monkeypatch):
        monkeypatch.setenv("AGENT_PROVIDER", "codex_pro")
        monkeypatch.setenv("AGENT_MODEL", "gpt-5.4")
        monkeypatch.setenv("DAILY_AGENT_PROVIDER", "zai_coding_plan")
        monkeypatch.setenv("MONTHLY_MODEL_REVIEW_AGENT_PROVIDER", "openrouter")
        monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        config = AppConfig.from_env()
        assert config.agent_default_provider == "codex_pro"
        assert config.agent_default_model == "gpt-5.4"
        assert config.daily_agent_provider == "zai_coding_plan"
        assert config.monthly_model_review_agent_provider == "openrouter"
        assert config.zai_api_key == "zai-test-key"
        assert config.openrouter_api_key == "or-test-key"

    def test_from_env_defaults(self):
        config = AppConfig()
        assert config.claude_command == "claude"
        assert config.codex_command == "codex"
        assert config.zai_api_key == ""
        assert config.openrouter_api_key == ""
        assert config.agent_default_provider == ""


class TestMaterialOutcomeRouting:
    def test_parameter_without_strategy_id_still_waits_for_monthly_outcome(self):
        assert _requires_monthly_outcome({
            "suggestion_id": "s1",
            "category": "parameter",
            "param_name": "quality_min",
        }) is True

    def test_non_material_observation_can_finish_from_early_warning(self):
        assert _requires_monthly_outcome({
            "suggestion_id": "s2",
            "category": "reporting",
            "tier": "observation",
        }) is False


# ---------------------------------------------------------------------------
# API tests: agent provider preference persistence and validation
# (merged from test_agent_preferences_api.py)
# ---------------------------------------------------------------------------


def _set_provider_statuses(app, statuses: dict[AgentProvider, tuple[bool, str, str]]) -> None:
    cache = app.state.agent_runner._auth_checker._provider_status_cache
    cache.clear()
    cache.update({
        provider: ProviderReadiness(
            provider=provider,
            available=available,
            runtime=runtime,
            reason=reason,
        )
        for provider, (available, runtime, reason) in statuses.items()
    })


@pytest.fixture
def app_factory(tmp_path):
    def _create():
        return create_app(
            db_dir=str(tmp_path),
            config=AppConfig(allow_unauthenticated_local=True),
        )

    return _create


class TestAgentPreferencesApi:
    @pytest.mark.asyncio
    async def test_get_returns_default_effective_and_provider_readiness(self, app_factory):
        app = app_factory()
        await app.state.queue.initialize()
        await app.state.registry.initialize()
        _set_provider_statuses(app, {
            AgentProvider.CLAUDE_MAX: (
                False,
                "claude_cli",
                "Claude Max login required: run 'claude auth login'",
            ),
            AgentProvider.CODEX_PRO: (True, "codex_cli", ""),
            AgentProvider.ZAI_CODING_PLAN: (False, "claude_cli", "ZAI_API_KEY is not configured"),
            AgentProvider.OPENROUTER: (False, "claude_cli", "OPENROUTER_API_KEY is not configured"),
        })

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/agent/preferences")

        assert resp.status_code == 200
        data = resp.json()
        assert data["default"]["provider"] == "codex_pro"
        assert "daily_analysis" in data["effective"]
        assert "providers" in data
        readiness = {item["provider"]: item for item in data["providers"]}
        assert readiness["claude_max"]["available"] is False
        assert readiness["claude_max"]["reason"] == "Claude Max login required: run 'claude auth login'"
        assert readiness["codex_pro"]["available"] is True
        assert readiness["zai_coding_plan"]["reason"] == "ZAI_API_KEY is not configured"

    @pytest.mark.asyncio
    async def test_put_updates_state_persists_and_applies_override_precedence(self, app_factory, tmp_path):
        app = app_factory()
        await app.state.queue.initialize()
        await app.state.registry.initialize()
        _set_provider_statuses(app, {
            AgentProvider.CLAUDE_MAX: (True, "claude_cli", ""),
            AgentProvider.CODEX_PRO: (True, "codex_cli", ""),
            AgentProvider.ZAI_CODING_PLAN: (True, "claude_cli", ""),
            AgentProvider.OPENROUTER: (True, "claude_cli", ""),
        })

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put("/agent/preferences", json={
                "default": {"provider": "codex_pro", "model": None},
                "overrides": {
                    "daily_analysis": {"provider": "claude_max", "model": "opus"},
                },
            })

        assert resp.status_code == 200
        data = resp.json()
        assert app.state.agent_preferences.default.provider == AgentProvider.CODEX_PRO
        assert data["effective"]["daily_analysis"]["provider"] == "claude_max"
        assert data["effective"]["daily_analysis"]["model"] == "opus"
        assert data["effective"]["weekly_analysis"]["provider"] == "codex_pro"
        assert data["effective"]["weekly_analysis"]["model"] == "gpt-5.4"

        stored = json.loads((tmp_path / "data" / "agent_preferences.json").read_text(encoding="utf-8"))
        assert stored["default"]["provider"] == "codex_pro"
        assert stored["overrides"]["daily_analysis"]["provider"] == "claude_max"

    @pytest.mark.asyncio
    async def test_put_rejects_unavailable_provider(self, app_factory):
        app = app_factory()
        await app.state.queue.initialize()
        await app.state.registry.initialize()
        _set_provider_statuses(app, {
            AgentProvider.CLAUDE_MAX: (True, "claude_cli", ""),
            AgentProvider.CODEX_PRO: (
                False,
                "codex_cli",
                "Command preflight failed for codex: missing executable",
            ),
            AgentProvider.ZAI_CODING_PLAN: (False, "claude_cli", "ZAI_API_KEY is not configured"),
            AgentProvider.OPENROUTER: (False, "claude_cli", "OPENROUTER_API_KEY is not configured"),
        })

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put("/agent/preferences", json={
                "default": {"provider": "codex_pro", "model": None},
                "overrides": {},
            })

        assert resp.status_code == 400
        assert "unavailable" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_put_rejects_legacy_wfo_override(self, app_factory):
        app = app_factory()
        await app.state.queue.initialize()
        await app.state.registry.initialize()
        _set_provider_statuses(app, {
            AgentProvider.CLAUDE_MAX: (True, "claude_cli", ""),
            AgentProvider.CODEX_PRO: (True, "codex_cli", ""),
            AgentProvider.ZAI_CODING_PLAN: (False, "claude_cli", "ZAI_API_KEY is not configured"),
            AgentProvider.OPENROUTER: (False, "claude_cli", "OPENROUTER_API_KEY is not configured"),
        })

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put("/agent/preferences", json={
                "default": {"provider": "claude_max", "model": None},
                "overrides": {
                    "wfo": {"provider": "codex_pro", "model": None},
                },
            })

        assert resp.status_code == 400
        assert "wfo" in resp.json()["detail"]
