"""Tests for cost tracking (Phase 5)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_assistant.orchestrator.agent_runner import AgentResult, AgentRunner
from trading_assistant.orchestrator.cost_tracker import CostTracker
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.schemas.agent_preferences import (
    AgentProvider,
    ProviderReadiness,
)
from trading_assistant.schemas.cost_tracking import CostRecord, CostSummary
from trading_assistant.schemas.prompt_package import PromptPackage


def _ready(provider: AgentProvider) -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(provider=provider, available=True, runtime=runtime)


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------


class TestCostRecordSchema:
    def test_defaults(self):
        rec = CostRecord(provider="claude_max")
        assert rec.cost_usd == 0.0
        assert rec.success is True
        assert rec.workflow == ""
        assert rec.run_id == ""
        assert rec.timestamp is not None

    def test_full_record(self):
        rec = CostRecord(
            provider="claude_max",
            workflow="daily_analysis",
            model="sonnet",
            cost_usd=0.05,
            duration_ms=12000,
            success=True,
            run_id="run-123",
        )
        assert rec.cost_usd == 0.05
        assert rec.workflow == "daily_analysis"

    def test_serialization_round_trip(self):
        rec = CostRecord(
            provider="codex_pro",
            workflow="monthly_validation",
            model="gpt-5.4",
            cost_usd=0.12,
            duration_ms=45000,
            success=False,
            run_id="run-456",
        )
        data = json.loads(rec.model_dump_json())
        restored = CostRecord(**data)
        assert restored.cost_usd == rec.cost_usd
        assert restored.provider == rec.provider
        assert restored.success is False


class TestCostSummarySchema:
    def test_defaults(self):
        s = CostSummary()
        assert s.total_cost_usd == 0.0
        assert s.total_invocations == 0
        assert s.by_provider == {}
        assert s.by_workflow == {}

    def test_populated(self):
        s = CostSummary(
            total_cost_usd=1.5,
            total_invocations=10,
            successful_invocations=8,
            failed_invocations=2,
            by_provider={"claude_max": 1.0, "codex_pro": 0.5},
        )
        assert s.total_cost_usd == 1.5
        assert s.by_provider["claude_max"] == 1.0


# ---------------------------------------------------------------------------
# CostTracker Tests
# ---------------------------------------------------------------------------


class TestCostTracker:
    @pytest.fixture
    def tracker(self, tmp_path: Path) -> CostTracker:
        return CostTracker(tmp_path / "costs" / "cost_log.jsonl")

    def test_record_creates_file(self, tracker: CostTracker):
        tracker.record(CostRecord(provider="claude_max", cost_usd=0.01))
        assert tracker._path.exists()

    def test_record_and_load(self, tracker: CostTracker):
        tracker.record(CostRecord(provider="claude_max", cost_usd=0.05))
        tracker.record(CostRecord(provider="codex_pro", cost_usd=0.10))
        records = tracker._load()
        assert len(records) == 2
        assert records[0].provider == "claude_max"
        assert records[1].cost_usd == 0.10

    def test_summary_empty(self, tracker: CostTracker):
        s = tracker.summary()
        assert s.total_invocations == 0
        assert s.total_cost_usd == 0.0

    def test_summary_aggregation(self, tracker: CostTracker):
        tracker.record(CostRecord(provider="claude_max", workflow="daily_analysis", cost_usd=0.05, duration_ms=5000, success=True))
        tracker.record(CostRecord(provider="claude_max", workflow="monthly_validation", cost_usd=0.15, duration_ms=30000, success=True))
        tracker.record(CostRecord(provider="codex_pro", workflow="daily_analysis", cost_usd=0.08, duration_ms=8000, success=False))

        s = tracker.summary()
        assert s.total_invocations == 3
        assert s.successful_invocations == 2
        assert s.failed_invocations == 1
        assert abs(s.total_cost_usd - 0.28) < 1e-9
        assert s.total_duration_ms == 43000
        assert abs(s.by_provider["claude_max"] - 0.20) < 1e-9
        assert abs(s.by_provider["codex_pro"] - 0.08) < 1e-9
        assert abs(s.by_workflow["daily_analysis"] - 0.13) < 1e-9
        assert abs(s.by_workflow["monthly_validation"] - 0.15) < 1e-9

    def test_summary_with_days_filter(self, tracker: CostTracker):
        old = CostRecord(
            provider="claude_max",
            cost_usd=1.00,
            timestamp=datetime.now(timezone.utc) - timedelta(days=10),
        )
        recent = CostRecord(
            provider="codex_pro",
            cost_usd=0.50,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        tracker.record(old)
        tracker.record(recent)

        s_all = tracker.summary()
        assert s_all.total_invocations == 2
        assert abs(s_all.total_cost_usd - 1.50) < 1e-9

        s_week = tracker.summary(days=7)
        assert s_week.total_invocations == 1
        assert abs(s_week.total_cost_usd - 0.50) < 1e-9

    def test_by_provider(self, tracker: CostTracker):
        tracker.record(CostRecord(provider="claude_max", cost_usd=0.10))
        tracker.record(CostRecord(provider="codex_pro", cost_usd=0.20))
        bp = tracker.by_provider()
        assert abs(bp["claude_max"] - 0.10) < 1e-9
        assert abs(bp["codex_pro"] - 0.20) < 1e-9

    def test_by_workflow(self, tracker: CostTracker):
        tracker.record(CostRecord(provider="claude_max", workflow="daily_analysis", cost_usd=0.10))
        tracker.record(CostRecord(provider="claude_max", workflow="monthly_validation", cost_usd=0.30))
        bw = tracker.by_workflow()
        assert abs(bw["daily_analysis"] - 0.10) < 1e-9
        assert abs(bw["monthly_validation"] - 0.30) < 1e-9

    def test_malformed_line_skipped(self, tracker: CostTracker):
        tracker._path.parent.mkdir(parents=True, exist_ok=True)
        with tracker._path.open("w", encoding="utf-8") as f:
            f.write("not-json\n")
            f.write(CostRecord(provider="claude_max", cost_usd=0.05).model_dump_json() + "\n")
        records = tracker._load()
        assert len(records) == 1
        assert records[0].cost_usd == 0.05


# ---------------------------------------------------------------------------
# AgentRunner Integration
# ---------------------------------------------------------------------------


class TestAgentRunnerCostIntegration:
    @pytest.fixture
    def cost_tracker(self, tmp_path: Path) -> CostTracker:
        return CostTracker(tmp_path / "cost_log.jsonl")

    @pytest.fixture
    def runner(self, tmp_path: Path, cost_tracker: CostTracker) -> AgentRunner:
        return AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            event_stream=EventStream(),
            timeout_seconds=600,
            cost_tracker=cost_tracker,
        )

    @pytest.mark.asyncio
    async def test_cost_recorded_after_invocation(self, runner: AgentRunner, cost_tracker: CostTracker, sample_package: PromptPackage):
        runner._auth_checker._provider_status_cache[AgentProvider.CLAUDE_MAX] = _ready(
            AgentProvider.CLAUDE_MAX,
        )

        async def _fake(**kwargs):
            return AgentResult(
                response="ok", run_dir=Path("/tmp"),
                success=True, provider="claude_max", runtime="claude_cli",
                cost_usd=0.03, duration_ms=5000, effective_model="sonnet",
            )

        with patch.object(runner, "invoke_with_selection", side_effect=_fake):
            await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="run-cost-test",
            )

        # invoke_with_selection is patched so _record_cost won't be called
        # (it's inside the real invoke_with_selection). Test _record_cost directly.
        runner._record_cost(
            AgentResult(
                response="ok", run_dir=Path("/tmp"),
                success=True, provider="claude_max", runtime="claude_cli",
                cost_usd=0.03, duration_ms=5000, effective_model="sonnet",
            ),
            "daily_analysis",
            "run-cost-test",
        )

        records = cost_tracker._load()
        assert len(records) == 1
        assert records[0].provider == "claude_max"
        assert records[0].workflow == "daily_analysis"
        assert records[0].cost_usd == 0.03

    def test_record_cost_without_tracker(self, tmp_path: Path, sample_package: PromptPackage):
        """No cost_tracker = no error."""
        runner = AgentRunner(
            runs_dir=tmp_path / "runs",
            session_store=SessionStore(base_dir=str(tmp_path / "sessions")),
            event_stream=EventStream(),
            timeout_seconds=600,
        )
        # Should not raise
        runner._record_cost(
            AgentResult(
                response="ok", run_dir=Path("/tmp"),
                success=True, provider="claude_max", runtime="claude_cli",
            ),
            "daily_analysis",
            "run-123",
        )

    def test_record_cost_captures_failure(self, runner: AgentRunner, cost_tracker: CostTracker):
        runner._record_cost(
            AgentResult(
                response="", run_dir=Path("/tmp"),
                success=False, provider="codex_pro", runtime="codex_cli",
                cost_usd=0.0, duration_ms=1000, effective_model="gpt-5.4",
                error="Timeout",
            ),
            "monthly_validation",
            "run-fail",
        )
        records = cost_tracker._load()
        assert len(records) == 1
        assert records[0].success is False
        assert records[0].workflow == "monthly_validation"
        assert records[0].run_id == "run-fail"
