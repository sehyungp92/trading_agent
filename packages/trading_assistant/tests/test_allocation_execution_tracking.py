"""Tests for allocation execution tracking — schemas, tracker, drift analyzer,
handler wiring, and weekly assembler integration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_assistant.schemas.allocation_history import (
    AllocationRecord,
    AllocationSnapshot,
    BotAllocationSnapshot,
)
from trading_assistant.schemas.portfolio_allocation import (
    BotAllocationRecommendation,
    PortfolioAllocationReport,
)
from trading_assistant.skills.allocation_tracker import AllocationTracker


# ---------------------------------------------------------------------------
# 1. Schema tests
# ---------------------------------------------------------------------------

class TestAllocationSnapshotSchema:
    def test_bot_snapshot_defaults(self):
        snap = BotAllocationSnapshot(bot_id="bot1", recommended_pct=40.0, actual_pct=35.0)
        assert snap.drift_pct == 0.0
        assert snap.abs_drift_pct == 0.0

    def test_bot_snapshot_populated(self):
        snap = BotAllocationSnapshot(
            bot_id="bot1",
            recommended_pct=40.0,
            actual_pct=35.0,
            drift_pct=5.0,
            abs_drift_pct=5.0,
        )
        assert snap.drift_pct == 5.0
        assert snap.abs_drift_pct == 5.0

    def test_allocation_snapshot_defaults(self):
        snap = AllocationSnapshot(date="2026-03-01", week_start="2026-03-01", week_end="2026-03-07")
        assert snap.bot_allocations == []
        assert snap.total_drift_pct == 0.0
        assert snap.max_single_drift_pct == 0.0
        assert snap.source == "weekly_handler"


# ---------------------------------------------------------------------------
# 2. Tracker snapshot methods
# ---------------------------------------------------------------------------

class TestAllocationTrackerSnapshots:
    def test_record_and_load_snapshot(self, tmp_path: Path):
        tracker = AllocationTracker(tmp_path)
        snap = AllocationSnapshot(
            date="2026-03-01", week_start="2026-03-01", week_end="2026-03-07",
            bot_allocations=[
                BotAllocationSnapshot(bot_id="b1", recommended_pct=50, actual_pct=50),
            ],
            total_drift_pct=0.0,
        )
        tracker.record_snapshot(snap)
        loaded = tracker.load_snapshots()
        assert len(loaded) == 1
        assert loaded[0]["date"] == "2026-03-01"

    def test_load_snapshots_empty(self, tmp_path: Path):
        tracker = AllocationTracker(tmp_path)
        assert tracker.load_snapshots() == []

    def test_multiple_snapshots(self, tmp_path: Path):
        tracker = AllocationTracker(tmp_path)
        for i in range(3):
            snap = AllocationSnapshot(
                date=f"2026-03-0{i+1}",
                week_start=f"2026-03-0{i+1}",
                week_end=f"2026-03-0{i+7}",
            )
            tracker.record_snapshot(snap)
        assert len(tracker.load_snapshots()) == 3

    def test_get_latest_actuals_empty(self, tmp_path: Path):
        tracker = AllocationTracker(tmp_path)
        assert tracker.get_latest_actuals() == {}

    def test_get_latest_actuals_single_record(self, tmp_path: Path):
        tracker = AllocationTracker(tmp_path)
        rec = AllocationRecord(date="2026-03-01", bot_id="b1", allocation_pct=60.0)
        tracker.record(rec)
        actuals = tracker.get_latest_actuals()
        assert actuals == {"b1": 60.0}

    def test_get_latest_actuals_multi_date(self, tmp_path: Path):
        tracker = AllocationTracker(tmp_path)
        # Older record
        tracker.record(AllocationRecord(date="2026-02-01", bot_id="b1", allocation_pct=40.0))
        tracker.record(AllocationRecord(date="2026-02-01", bot_id="b2", allocation_pct=60.0))
        # Newer record — only b1 updated
        tracker.record(AllocationRecord(date="2026-03-01", bot_id="b1", allocation_pct=55.0))
        actuals = tracker.get_latest_actuals()
        assert actuals["b1"] == 55.0  # latest
        assert actuals["b2"] == 60.0  # only one record


# ---------------------------------------------------------------------------
# 3. Handler drift wiring
# ---------------------------------------------------------------------------

class TestHandlerDriftWiring:
    @pytest.fixture
    def handlers_setup(self, tmp_path: Path):
        from trading_assistant.orchestrator.event_stream import EventStream
        from trading_assistant.orchestrator.handlers import Handlers
        from trading_assistant.schemas.notifications import NotificationPreferences

        return Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1", "bot2"],
        )

    def _make_portfolio_summary(self):
        """Create a minimal portfolio summary with bot_summaries."""
        summary = MagicMock()
        bot1 = MagicMock()
        bot1.per_strategy_summary = {}
        bot2 = MagicMock()
        bot2.per_strategy_summary = {}
        summary.bot_summaries = {"bot1": bot1, "bot2": bot2}
        return summary

    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute")
    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute_intra_bot")
    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute_bot_correlation_matrix")
    @patch("trading_assistant.skills.portfolio_allocator.PortfolioAllocator.compute")
    @patch("trading_assistant.skills.strategy_proportion_optimizer.StrategyProportionOptimizer.compute")
    @patch("trading_assistant.skills.structural_analyzer.StructuralAnalyzer.compute")
    @patch("trading_assistant.analysis.strategy_engine.StrategyEngine.compute_regime_conditional_metrics")
    def test_uses_latest_actuals(
        self, mock_regime, mock_structural, mock_proportion, mock_alloc,
        mock_corr, mock_intra, mock_synergy, handlers_setup, tmp_path,
    ):
        """When allocation_history.jsonl has records, handler uses them."""
        # Set up allocation history
        findings_dir = tmp_path / "memory" / "findings"
        findings_dir.mkdir(parents=True, exist_ok=True)
        tracker = AllocationTracker(findings_dir)
        tracker.record(AllocationRecord(date="2026-02-28", bot_id="bot1", allocation_pct=70.0))
        tracker.record(AllocationRecord(date="2026-02-28", bot_id="bot2", allocation_pct=30.0))

        # Mock returns
        synergy_report = MagicMock()
        synergy_report.model_dump = MagicMock(return_value={})
        mock_synergy.return_value = synergy_report
        mock_corr.return_value = {}

        alloc_report = PortfolioAllocationReport(
            week_start="2026-03-01", week_end="2026-03-07",
            recommendations=[
                BotAllocationRecommendation(bot_id="bot1", suggested_allocation_pct=60.0),
                BotAllocationRecommendation(bot_id="bot2", suggested_allocation_pct=40.0),
            ],
        )
        mock_alloc.return_value = alloc_report

        prop_report = MagicMock()
        prop_report.model_dump = MagicMock(return_value={})
        mock_proportion.return_value = prop_report

        struct_report = MagicMock()
        struct_report.model_dump = MagicMock(return_value={})
        mock_structural.return_value = struct_report

        regime_report = MagicMock()
        regime_report.model_dump = MagicMock(return_value={})
        mock_regime.return_value = regime_report

        summary = self._make_portfolio_summary()
        handlers_setup._run_allocation_analyses(summary, "2026-03-01", "2026-03-07")

        # Verify allocator was called with latest actuals (70/30), not equal-weight
        call_args = mock_alloc.call_args
        current_arg = call_args[0][1]  # second positional arg
        assert current_arg["bot1"] == 70.0
        assert current_arg["bot2"] == 30.0

    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute")
    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute_intra_bot")
    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute_bot_correlation_matrix")
    @patch("trading_assistant.skills.portfolio_allocator.PortfolioAllocator.compute")
    @patch("trading_assistant.skills.strategy_proportion_optimizer.StrategyProportionOptimizer.compute")
    @patch("trading_assistant.skills.structural_analyzer.StructuralAnalyzer.compute")
    @patch("trading_assistant.analysis.strategy_engine.StrategyEngine.compute_regime_conditional_metrics")
    def test_falls_back_to_equal_weight(
        self, mock_regime, mock_structural, mock_proportion, mock_alloc,
        mock_corr, mock_intra, mock_synergy, handlers_setup, tmp_path,
    ):
        """When no allocation history, handler uses equal-weight."""
        synergy_report = MagicMock()
        synergy_report.model_dump = MagicMock(return_value={})
        mock_synergy.return_value = synergy_report
        mock_corr.return_value = {}

        alloc_report = PortfolioAllocationReport(
            week_start="2026-03-01", week_end="2026-03-07",
            recommendations=[
                BotAllocationRecommendation(bot_id="bot1", suggested_allocation_pct=50.0),
                BotAllocationRecommendation(bot_id="bot2", suggested_allocation_pct=50.0),
            ],
        )
        mock_alloc.return_value = alloc_report

        prop_report = MagicMock()
        prop_report.model_dump = MagicMock(return_value={})
        mock_proportion.return_value = prop_report

        struct_report = MagicMock()
        struct_report.model_dump = MagicMock(return_value={})
        mock_structural.return_value = struct_report

        regime_report = MagicMock()
        regime_report.model_dump = MagicMock(return_value={})
        mock_regime.return_value = regime_report

        summary = self._make_portfolio_summary()
        handlers_setup._run_allocation_analyses(summary, "2026-03-01", "2026-03-07")

        call_args = mock_alloc.call_args
        current_arg = call_args[0][1]
        assert current_arg["bot1"] == 50.0
        assert current_arg["bot2"] == 50.0

    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute")
    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute_intra_bot")
    @patch("trading_assistant.skills.synergy_analyzer.SynergyAnalyzer.compute_bot_correlation_matrix")
    @patch("trading_assistant.skills.portfolio_allocator.PortfolioAllocator.compute")
    @patch("trading_assistant.skills.strategy_proportion_optimizer.StrategyProportionOptimizer.compute")
    @patch("trading_assistant.skills.structural_analyzer.StructuralAnalyzer.compute")
    @patch("trading_assistant.analysis.strategy_engine.StrategyEngine.compute_regime_conditional_metrics")
    def test_records_snapshot_and_drift(
        self, mock_regime, mock_structural, mock_proportion, mock_alloc,
        mock_corr, mock_intra, mock_synergy, handlers_setup, tmp_path,
    ):
        """Handler records snapshot and includes drift trend in results."""
        synergy_report = MagicMock()
        synergy_report.model_dump = MagicMock(return_value={})
        mock_synergy.return_value = synergy_report
        mock_corr.return_value = {}

        alloc_report = PortfolioAllocationReport(
            week_start="2026-03-01", week_end="2026-03-07",
            recommendations=[
                BotAllocationRecommendation(bot_id="bot1", suggested_allocation_pct=60.0),
                BotAllocationRecommendation(bot_id="bot2", suggested_allocation_pct=40.0),
            ],
        )
        mock_alloc.return_value = alloc_report

        prop_report = MagicMock()
        prop_report.model_dump = MagicMock(return_value={})
        mock_proportion.return_value = prop_report

        struct_report = MagicMock()
        struct_report.model_dump = MagicMock(return_value={})
        mock_structural.return_value = struct_report

        regime_report = MagicMock()
        regime_report.model_dump = MagicMock(return_value={})
        mock_regime.return_value = regime_report

        summary = self._make_portfolio_summary()
        results = handlers_setup._run_allocation_analyses(summary, "2026-03-01", "2026-03-07")

        assert "allocation_drift" in results
        assert "current_snapshot" in results["allocation_drift"]
        assert "trend" in results["allocation_drift"]

        # Verify snapshot was persisted
        tracker = AllocationTracker(tmp_path / "memory" / "findings")
        snaps = tracker.load_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["date"] == "2026-03-01"

        # Verify curated file was written
        drift_file = tmp_path / "data" / "curated" / "weekly" / "2026-03-01" / "allocation_drift.json"
        assert drift_file.exists()


# ---------------------------------------------------------------------------
# 4. Weekly assembler drift integration
# ---------------------------------------------------------------------------

class TestWeeklyAssemblerDrift:
    def test_loads_drift_file(self, tmp_path: Path):
        from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

        curated = tmp_path / "data" / "curated"
        weekly_dir = curated / "weekly" / "2026-03-01"
        weekly_dir.mkdir(parents=True)
        drift_data = {"current_snapshot": {}, "trend": {"trend_direction": "stable"}}
        (weekly_dir / "allocation_drift.json").write_text(json.dumps(drift_data))

        assembler = WeeklyPromptAssembler(
            week_start="2026-03-01",
            week_end="2026-03-07",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
        )
        pkg = assembler.assemble()
        assert "allocation_drift" in pkg.data
        assert pkg.data["allocation_drift"]["trend"]["trend_direction"] == "stable"

    def test_includes_drift_in_context_files(self, tmp_path: Path):
        from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

        curated = tmp_path / "data" / "curated"
        weekly_dir = curated / "weekly" / "2026-03-01"
        weekly_dir.mkdir(parents=True)
        (weekly_dir / "allocation_drift.json").write_text("{}")

        assembler = WeeklyPromptAssembler(
            week_start="2026-03-01",
            week_end="2026-03-07",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
        )
        pkg = assembler.assemble()
        drift_files = [f for f in pkg.context_files if "allocation_drift" in f]
        assert len(drift_files) == 1

    def test_instruction_19_present(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "PORTFOLIO IMPROVEMENT ASSESSMENT" in _WEEKLY_INSTRUCTIONS
        assert "Allocation analysis" in _WEEKLY_INSTRUCTIONS
