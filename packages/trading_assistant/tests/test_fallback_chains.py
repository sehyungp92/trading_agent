"""Tests for automatic provider fallback chains."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from trading_assistant.orchestrator.agent_preferences import AgentPreferencesManager, DEFAULT_PROVIDER_MODELS
from trading_assistant.orchestrator.agent_runner import AgentResult, AgentRunner
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.provider_cooldown import ProviderCooldownTracker
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    FallbackEntry,
    ProviderReadiness,
)


def _ready(provider: AgentProvider) -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(provider=provider, available=True, runtime=runtime)


def _unavailable(provider: AgentProvider, reason: str = "down") -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(provider=provider, available=False, runtime=runtime, reason=reason)


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------


class TestFallbackSchema:
    def test_fallback_entry_round_trips(self):
        entry = FallbackEntry(provider=AgentProvider.OPENROUTER, model="custom-model")
        assert entry.provider == AgentProvider.OPENROUTER
        assert entry.model == "custom-model"

    def test_preferences_with_fallback_chain(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            fallback_chain=[
                FallbackEntry(provider=AgentProvider.ZAI_CODING_PLAN),
                FallbackEntry(provider=AgentProvider.OPENROUTER),
            ],
        )
        assert len(prefs.fallback_chain) == 2
        assert prefs.fallback_chain[0].provider == AgentProvider.ZAI_CODING_PLAN

    def test_preferences_serialization_includes_fallback(self):
        prefs = AgentPreferences(
            fallback_chain=[FallbackEntry(provider=AgentProvider.CODEX_PRO, model="gpt-5.4")],
        )
        data = prefs.model_dump()
        assert len(data["fallback_chain"]) == 1
        assert data["fallback_chain"][0]["provider"] == "codex_pro"

    def test_empty_fallback_chain_by_default(self):
        prefs = AgentPreferences()
        assert prefs.fallback_chain == []


# ---------------------------------------------------------------------------
# Resolve With Fallbacks
# ---------------------------------------------------------------------------


class TestResolveWithFallbacks:
    def _make_manager(
        self,
        prefs: AgentPreferences,
        statuses: dict[AgentProvider, bool] | None = None,
    ) -> AgentPreferencesManager:
        mgr = AgentPreferencesManager(prefs)
        if statuses is not None:
            def resolver():
                return [
                    ProviderReadiness(
                        provider=p,
                        available=a,
                        runtime="codex_cli" if p == AgentProvider.CODEX_PRO else "claude_cli",
                    )
                    for p, a in statuses.items()
                ]
            mgr.set_provider_status_resolver(resolver)
        return mgr

    def test_returns_primary_when_available_and_no_fallback(self):
        mgr = self._make_manager(
            AgentPreferences(default=AgentSelection(provider=AgentProvider.CLAUDE_MAX)),
            {AgentProvider.CLAUDE_MAX: True},
        )
        candidates = mgr.resolve_with_fallbacks(AgentWorkflow.DAILY_ANALYSIS)
        assert len(candidates) == 1
        assert candidates[0][0].provider == AgentProvider.CLAUDE_MAX

    def test_returns_primary_plus_fallbacks(self):
        mgr = self._make_manager(
            AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[
                    FallbackEntry(provider=AgentProvider.ZAI_CODING_PLAN),
                    FallbackEntry(provider=AgentProvider.OPENROUTER),
                ],
            ),
            {
                AgentProvider.CLAUDE_MAX: True,
                AgentProvider.ZAI_CODING_PLAN: True,
                AgentProvider.OPENROUTER: True,
            },
        )
        candidates = mgr.resolve_with_fallbacks(AgentWorkflow.DAILY_ANALYSIS)
        assert len(candidates) == 3
        providers = [c[0].provider for c in candidates]
        assert providers == [
            AgentProvider.CLAUDE_MAX,
            AgentProvider.ZAI_CODING_PLAN,
            AgentProvider.OPENROUTER,
        ]

    def test_skips_unavailable_providers(self):
        mgr = self._make_manager(
            AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[
                    FallbackEntry(provider=AgentProvider.ZAI_CODING_PLAN),
                    FallbackEntry(provider=AgentProvider.OPENROUTER),
                ],
            ),
            {
                AgentProvider.CLAUDE_MAX: False,
                AgentProvider.ZAI_CODING_PLAN: False,
                AgentProvider.OPENROUTER: True,
            },
        )
        candidates = mgr.resolve_with_fallbacks(AgentWorkflow.DAILY_ANALYSIS)
        assert len(candidates) == 1
        assert candidates[0][0].provider == AgentProvider.OPENROUTER

    def test_skips_cooled_down_providers(self):
        cooldown = ProviderCooldownTracker(cooldown_seconds=300)
        cooldown.record_failure(AgentProvider.CLAUDE_MAX)

        mgr = self._make_manager(
            AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[
                    FallbackEntry(provider=AgentProvider.OPENROUTER),
                ],
            ),
            {AgentProvider.CLAUDE_MAX: True, AgentProvider.OPENROUTER: True},
        )
        candidates = mgr.resolve_with_fallbacks(
            AgentWorkflow.DAILY_ANALYSIS, cooldown_tracker=cooldown,
        )
        assert len(candidates) == 1
        assert candidates[0][0].provider == AgentProvider.OPENROUTER

    def test_deduplicates_primary_in_fallback_chain(self):
        mgr = self._make_manager(
            AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[
                    FallbackEntry(provider=AgentProvider.CLAUDE_MAX),  # duplicate
                    FallbackEntry(provider=AgentProvider.OPENROUTER),
                ],
            ),
            {AgentProvider.CLAUDE_MAX: True, AgentProvider.OPENROUTER: True},
        )
        candidates = mgr.resolve_with_fallbacks(AgentWorkflow.DAILY_ANALYSIS)
        assert len(candidates) == 2
        providers = [c[0].provider for c in candidates]
        assert providers == [AgentProvider.CLAUDE_MAX, AgentProvider.OPENROUTER]

    def test_fills_default_model_for_fallback_entries(self):
        mgr = self._make_manager(
            AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[
                    FallbackEntry(provider=AgentProvider.ZAI_CODING_PLAN),
                ],
            ),
            {AgentProvider.CLAUDE_MAX: True, AgentProvider.ZAI_CODING_PLAN: True},
        )
        candidates = mgr.resolve_with_fallbacks(AgentWorkflow.DAILY_ANALYSIS)
        zai = candidates[1][0]
        assert zai.model == DEFAULT_PROVIDER_MODELS[AgentProvider.ZAI_CODING_PLAN]

    def test_returns_empty_when_all_unavailable(self):
        mgr = self._make_manager(
            AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[FallbackEntry(provider=AgentProvider.OPENROUTER)],
            ),
            {AgentProvider.CLAUDE_MAX: False, AgentProvider.OPENROUTER: False},
        )
        candidates = mgr.resolve_with_fallbacks(AgentWorkflow.DAILY_ANALYSIS)
        assert candidates == []

    def test_model_override_applies_to_primary_only(self):
        mgr = self._make_manager(
            AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[FallbackEntry(provider=AgentProvider.OPENROUTER)],
            ),
            {AgentProvider.CLAUDE_MAX: True, AgentProvider.OPENROUTER: True},
        )
        candidates = mgr.resolve_with_fallbacks(
            AgentWorkflow.DAILY_ANALYSIS, model_override="opus",
        )
        assert candidates[0][0].model == "opus"
        assert candidates[0][1] == "opus"
        # Fallback should use its default model, not the override
        assert candidates[1][1] is None


# ---------------------------------------------------------------------------
# AgentRunner Fallback Integration
# ---------------------------------------------------------------------------


class TestAgentRunnerFallback:
    @pytest.fixture
    def runner(self, tmp_path: Path) -> AgentRunner:
        return AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            event_stream=EventStream(),
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
                fallback_chain=[
                    FallbackEntry(provider=AgentProvider.ZAI_CODING_PLAN),
                    FallbackEntry(provider=AgentProvider.OPENROUTER),
                ],
            ),
            zai_api_key="zai-key",
            openrouter_api_key="or-key",
        )

    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)
        success = AgentResult(
            response="done", run_dir=Path("/tmp/run"),
            success=True, provider="claude_max", runtime="claude_cli",
        )
        with patch.object(runner, "invoke_with_selection", new_callable=AsyncMock, return_value=success):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-ok",
            )
        assert result.success is True
        assert result.provider == "claude_max"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache.update({
            AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
            AgentProvider.ZAI_CODING_PLAN: _ready(AgentProvider.ZAI_CODING_PLAN),
            AgentProvider.OPENROUTER: _ready(AgentProvider.OPENROUTER),
        })

        failure = AgentResult(
            response="", run_dir=Path("/tmp/run"),
            success=False, error="auth expired", provider="claude_max", runtime="claude_cli",
        )
        zai_success = AgentResult(
            response="fallback ok", run_dir=Path("/tmp/run"),
            success=True, provider="zai_coding_plan", runtime="claude_cli",
        )

        call_count = 0
        async def _mock_invoke(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["selection"].provider == AgentProvider.CLAUDE_MAX:
                return failure
            return zai_success

        with patch.object(runner, "invoke_with_selection", side_effect=_mock_invoke):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-fallback",
            )

        assert result.success is True
        assert result.provider == "zai_coding_plan"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_fallbacks_exhausted(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache.update({
            AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
            AgentProvider.ZAI_CODING_PLAN: _ready(AgentProvider.ZAI_CODING_PLAN),
            AgentProvider.OPENROUTER: _ready(AgentProvider.OPENROUTER),
        })

        failure = AgentResult(
            response="", run_dir=Path("/tmp/run"),
            success=False, error="fail", provider="x", runtime="y",
        )

        with patch.object(runner, "invoke_with_selection", new_callable=AsyncMock, return_value=failure):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-all-fail",
            )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_cooldown_recorded_on_failure(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache.update({
            AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
            AgentProvider.ZAI_CODING_PLAN: _ready(AgentProvider.ZAI_CODING_PLAN),
            AgentProvider.OPENROUTER: _ready(AgentProvider.OPENROUTER),
        })

        failure = AgentResult(
            response="", run_dir=Path("/tmp/run"),
            success=False, error="fail", provider="x", runtime="y",
        )
        success = AgentResult(
            response="ok", run_dir=Path("/tmp/run"),
            success=True, provider="zai_coding_plan", runtime="claude_cli",
        )

        async def _mock_invoke(**kwargs):
            if kwargs["selection"].provider == AgentProvider.CLAUDE_MAX:
                return failure
            return success

        with patch.object(runner, "invoke_with_selection", side_effect=_mock_invoke):
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-cd",
            )

        assert runner._cooldown_tracker.is_cooled_down(AgentProvider.CLAUDE_MAX) is True
        assert runner._cooldown_tracker.is_cooled_down(AgentProvider.ZAI_CODING_PLAN) is False

    @pytest.mark.asyncio
    async def test_fallback_event_broadcast(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache.update({
            AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
            AgentProvider.ZAI_CODING_PLAN: _ready(AgentProvider.ZAI_CODING_PLAN),
            AgentProvider.OPENROUTER: _ready(AgentProvider.OPENROUTER),
        })

        queue = runner._event_stream.subscribe()

        failure = AgentResult(
            response="", run_dir=Path("/tmp/run"),
            success=False, error="auth expired", provider="claude_max", runtime="claude_cli",
        )
        success = AgentResult(
            response="ok", run_dir=Path("/tmp/run"),
            success=True, provider="zai_coding_plan", runtime="claude_cli",
        )

        async def _mock_invoke(**kwargs):
            if kwargs["selection"].provider == AgentProvider.CLAUDE_MAX:
                return failure
            return success

        with patch.object(runner, "invoke_with_selection", side_effect=_mock_invoke):
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-evt",
            )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        fallback_events = [e for e in events if e.event_type == "provider_fallback"]
        assert len(fallback_events) == 1
        assert fallback_events[0].data["failed_provider"] == "claude_max"
        assert fallback_events[0].data["next_provider"] == "zai_coding_plan"

    @pytest.mark.asyncio
    async def test_no_fallback_chain_uses_simple_path(self, tmp_path: Path, sample_package):
        """Without fallback_chain, invoke() behaves as before."""
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            ),
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)

        success = AgentResult(
            response="ok", run_dir=Path("/tmp/run"),
            success=True, provider="claude_max", runtime="claude_cli",
        )
        with patch.object(runner, "invoke_with_selection", new_callable=AsyncMock, return_value=success):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-simple",
            )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_candidates_respects_cooldown_without_bypass(self, tmp_path: Path, sample_package):
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            ),
        )
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(AgentProvider.CLAUDE_MAX)
        runner._cooldown_tracker.record_failure(AgentProvider.CLAUDE_MAX)

        with patch.object(runner, "invoke_with_selection", new_callable=AsyncMock) as mock_invoke:
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-cooled-down",
            )

        assert result.success is False
        assert "cooldown filtering" in result.error
        assert mock_invoke.await_count == 0

    @pytest.mark.asyncio
    async def test_learned_route_audit_written_after_attempt(self, tmp_path: Path, sample_package):
        findings = tmp_path / "memory" / "findings"
        findings.mkdir(parents=True)
        with (findings / "provider_route_scores.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "workflow": "daily_analysis",
                "provider": "claude_max",
                "model": "sonnet",
                "composite_score": 0.52,
                "sample_count": 6,
            }) + "\n")
            handle.write(json.dumps({
                "workflow": "daily_analysis",
                "provider": "codex_pro",
                "model": "gpt-5.4",
                "composite_score": 0.90,
                "benchmark_quality": 0.85,
                "sample_count": 7,
            }) + "\n")
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            preferences=AgentPreferences(
                default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            ),
        )
        runner._auth_checker._provider_status_cache.update({
            AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
            AgentProvider.CODEX_PRO: _ready(AgentProvider.CODEX_PRO),
        })
        success = AgentResult(
            response="ok",
            run_dir=tmp_path / "runs" / "daily-learned",
            success=True,
            provider="codex_pro",
            runtime="codex_cli",
        )

        with patch.object(runner, "invoke_with_selection", new_callable=AsyncMock, return_value=success) as mock_invoke:
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-learned",
            )

        assert result.success is True
        assert mock_invoke.await_args.kwargs["selection"].provider == AgentProvider.CODEX_PRO
        changes = [
            json.loads(line)
            for line in (findings / "provider_route_changes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(changes) == 1
        assert changes[0]["previous_provider"] == "claude_max"
        assert changes[0]["recommended_provider"] == "codex_pro"
        assert changes[0]["first_used_run_id"] == "daily-learned"
        assert changes[0]["first_attempt_success"] is True

    @pytest.mark.asyncio
    async def test_fallback_run_id_suffixed(self, runner: AgentRunner, sample_package):
        runner._auth_checker._provider_status_cache.update({
            AgentProvider.CLAUDE_MAX: _ready(AgentProvider.CLAUDE_MAX),
            AgentProvider.ZAI_CODING_PLAN: _ready(AgentProvider.ZAI_CODING_PLAN),
            AgentProvider.OPENROUTER: _ready(AgentProvider.OPENROUTER),
        })

        call_run_ids: list[str] = []

        async def _mock_invoke(**kwargs):
            call_run_ids.append(kwargs["run_id"])
            if kwargs["selection"].provider == AgentProvider.CLAUDE_MAX:
                return AgentResult(
                    response="", run_dir=Path("/tmp"), success=False,
                    error="fail", provider="x", runtime="y",
                )
            return AgentResult(
                response="ok", run_dir=Path("/tmp"), success=True,
                provider="zai", runtime="cli",
            )

        with patch.object(runner, "invoke_with_selection", side_effect=_mock_invoke):
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-001",
            )

        assert call_run_ids[0] == "run-001"
        assert call_run_ids[1] == "run-001-fallback-1"
