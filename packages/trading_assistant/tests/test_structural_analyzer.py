"""Tests for StructuralAnalyzer — lifecycle, mismatches, filter ROI."""
from __future__ import annotations


from trading_assistant.schemas.structural_analysis import (
    ArchitectureMismatch,
    FilterROI,
    StrategyLifecycleStatus,
    StructuralReport,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary
from trading_assistant.skills.structural_analyzer import StructuralAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(
    bot_id: str = "bot1",
    strategy_id: str = "s1",
    daily_pnl: dict[str, float] | None = None,
    total_trades: int = 50,
) -> StrategyWeeklySummary:
    return StrategyWeeklySummary(
        strategy_id=strategy_id,
        bot_id=bot_id,
        total_trades=total_trades,
        daily_pnl=daily_pnl or {},
    )


def _daily_series(n: int, value: float) -> dict[str, float]:
    """Generate n days of constant PnL."""
    return {f"2026-01-{i+1:02d}": value for i in range(n)}


def _growing_series() -> dict[str, float]:
    """90d series where recent 30d Sharpe is much better than early 60d."""
    from datetime import datetime, timedelta
    series = {}
    base = datetime(2025, 11, 1)
    # First 60 days: low, noisy returns (low Sharpe)
    for i in range(60):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        series[d] = 5.0 + (i % 3 - 1) * 20.0  # -15, 5, 25, -15, 5, 25...
    # Last 30 days: high, consistent returns (high Sharpe)
    for i in range(30):
        d = (base + timedelta(days=60 + i)).strftime("%Y-%m-%d")
        series[d] = 80.0 + (i % 2) * 5.0  # 80, 85, 80, 85...
    return series


def _decaying_series() -> dict[str, float]:
    """90d series where early 60d Sharpe is high, recent 30d poor."""
    from datetime import datetime, timedelta
    series = {}
    base = datetime(2025, 11, 1)
    # First 60 days: high, consistent returns (high Sharpe)
    for i in range(60):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        series[d] = 80.0 + (i % 2) * 5.0
    # Last 30 days: low, noisy returns (low Sharpe)
    for i in range(30):
        d = (base + timedelta(days=60 + i)).strftime("%Y-%m-%d")
        series[d] = 5.0 + (i % 3 - 1) * 20.0
    return series


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------

class TestSchemaDefaults:
    def test_lifecycle_status_defaults(self):
        s = StrategyLifecycleStatus(bot_id="b", strategy_id="s", phase="mature")
        assert s.sharpe_30d == 0.0
        assert s.edge_half_life_days is None
        assert s.trade_count_90d == 0

    def test_architecture_mismatch_defaults(self):
        m = ArchitectureMismatch(
            bot_id="b", strategy_id="s", mismatch_type="test",
            current_setup="c", recommended_setup="r", evidence="e",
        )
        assert m.estimated_impact_pnl == 0.0
        assert m.confidence == 0.0

    def test_filter_roi_defaults(self):
        f = FilterROI(bot_id="b", strategy_id="s", filter_name="f1")
        assert f.blocks_saved_count == 0
        assert f.roi == 0.0

    def test_structural_report_defaults(self):
        r = StructuralReport(week_start="2026-01-01", week_end="2026-01-07")
        assert r.lifecycle_statuses == []
        assert r.growing_strategies == []
        assert r.proposed_changes == []


# ---------------------------------------------------------------------------
# Lifecycle classification
# ---------------------------------------------------------------------------

class TestLifecycleClassification:
    def test_empty_input_returns_empty_report(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        report = sa.compute({})
        assert report.lifecycle_statuses == []
        assert report.growing_strategies == []
        assert report.decaying_strategies == []

    def test_single_strategy_mature(self):
        """Constant PnL → mature classification."""
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_daily_series(90, 50.0))}}
        report = sa.compute(summaries)
        assert len(report.lifecycle_statuses) == 1
        assert report.lifecycle_statuses[0].phase == "mature"

    def test_growing_strategy(self):
        """Recent Sharpe much higher than long-term → growing."""
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07", growth_sharpe_threshold=0.1)
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_growing_series())}}
        report = sa.compute(summaries)
        assert "bot1:s1" in report.growing_strategies

    def test_decaying_strategy(self):
        """Old Sharpe much higher than recent → decaying."""
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07", decay_sharpe_threshold=0.1)
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_decaying_series())}}
        report = sa.compute(summaries)
        assert "bot1:s1" in report.decaying_strategies

    def test_sharpe_trend_positive_for_growing(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07", growth_sharpe_threshold=0.1)
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_growing_series())}}
        report = sa.compute(summaries)
        status = report.lifecycle_statuses[0]
        assert status.sharpe_trend > 0

    def test_sharpe_trend_negative_for_decaying(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07", decay_sharpe_threshold=0.1)
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_decaying_series())}}
        report = sa.compute(summaries)
        status = report.lifecycle_statuses[0]
        assert status.sharpe_trend < 0

    def test_edge_half_life_only_when_decaying(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_daily_series(90, 50.0))}}
        report = sa.compute(summaries)
        assert report.lifecycle_statuses[0].edge_half_life_days is None

    def test_too_few_data_points(self):
        """With only 1 day, should still produce a status (mature, zero Sharpe)."""
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        summaries = {"bot1": {"s1": _make_summary(daily_pnl={"2026-01-01": 100.0})}}
        report = sa.compute(summaries)
        assert report.lifecycle_statuses[0].phase == "mature"

    def test_multiple_bots_multiple_strategies(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        summaries = {
            "bot1": {
                "s1": _make_summary("bot1", "s1", _daily_series(90, 50.0)),
                "s2": _make_summary("bot1", "s2", _daily_series(90, 50.0)),
            },
            "bot2": {
                "helix": _make_summary("bot2", "helix", _daily_series(90, 50.0)),
            },
        }
        report = sa.compute(summaries)
        assert len(report.lifecycle_statuses) == 3


# ---------------------------------------------------------------------------
# Architecture mismatch detection
# ---------------------------------------------------------------------------

class TestArchitectureMismatch:
    def test_momentum_fixed_tp_mismatch(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        meta = {"bot1:s1": {"signal_type": "momentum", "exit_type": "fixed_tp"}}
        report = sa.compute({}, strategy_metadata=meta)
        assert len(report.architecture_mismatches) == 1
        assert report.architecture_mismatches[0].mismatch_type == "momentum_fixed_tp"

    def test_mean_reversion_trailing_mismatch(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        meta = {"bot1:s1": {"signal_type": "mean_reversion", "exit_type": "trailing"}}
        report = sa.compute({}, strategy_metadata=meta)
        assert len(report.architecture_mismatches) == 1
        assert report.architecture_mismatches[0].mismatch_type == "mean_reversion_trailing_stop"

    def test_breakout_time_based_mismatch(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        meta = {"bot1:s1": {"signal_type": "breakout", "exit_type": "time_based"}}
        report = sa.compute({}, strategy_metadata=meta)
        assert len(report.architecture_mismatches) == 1

    def test_no_mismatch_when_compatible(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        meta = {"bot1:s1": {"signal_type": "momentum", "exit_type": "trailing"}}
        report = sa.compute({}, strategy_metadata=meta)
        assert len(report.architecture_mismatches) == 0

    def test_no_mismatch_with_empty_metadata(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        report = sa.compute({}, strategy_metadata={})
        assert len(report.architecture_mismatches) == 0

    def test_missing_signal_or_exit_type_skipped(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        meta = {"bot1:s1": {"signal_type": "momentum"}}  # no exit_type
        report = sa.compute({}, strategy_metadata=meta)
        assert len(report.architecture_mismatches) == 0


# ---------------------------------------------------------------------------
# Filter ROI
# ---------------------------------------------------------------------------

class TestFilterROI:
    def test_positive_roi_filter(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        filters = {
            "bot1:s1": {
                "regime_filter": {"saved": 5, "cost": 2, "net_pnl": 300.0, "missed_value": 200.0},
            },
        }
        report = sa.compute({}, filter_data=filters)
        assert len(report.filter_roi) == 1
        assert report.filter_roi[0].roi > 0

    def test_negative_roi_generates_proposal(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        filters = {
            "bot1:s1": {
                "regime_filter": {"saved": 1, "cost": 5, "net_pnl": -500.0, "missed_value": 1000.0},
            },
        }
        report = sa.compute({}, filter_data=filters)
        assert len(report.filter_roi) == 1
        assert report.filter_roi[0].net_pnl_impact < 0
        # Should generate a proposal
        filter_proposals = [p for p in report.proposed_changes if p["category"] == "filter"]
        assert len(filter_proposals) == 1

    def test_zero_missed_value_roi(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        filters = {
            "bot1:s1": {
                "f1": {"saved": 0, "cost": 0, "net_pnl": 0.0, "missed_value": 0.0},
            },
        }
        report = sa.compute({}, filter_data=filters)
        assert report.filter_roi[0].roi == 0.0


# ---------------------------------------------------------------------------
# Proposed changes
# ---------------------------------------------------------------------------

class TestProposedChanges:
    def test_decaying_strategy_generates_proposal(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07", decay_sharpe_threshold=0.1)
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_decaying_series())}}
        report = sa.compute(summaries)
        lifecycle_proposals = [p for p in report.proposed_changes if p["category"] == "lifecycle"]
        assert len(lifecycle_proposals) >= 1

    def test_mismatch_generates_proposal(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        meta = {"bot1:s1": {"signal_type": "momentum", "exit_type": "fixed_tp"}}
        report = sa.compute({}, strategy_metadata=meta)
        arch_proposals = [p for p in report.proposed_changes if p["category"] == "architecture"]
        assert len(arch_proposals) == 1
        assert arch_proposals[0]["reversibility"] == "high"

    def test_no_proposals_for_healthy_strategies(self):
        sa = StructuralAnalyzer("2026-01-01", "2026-01-07")
        summaries = {"bot1": {"s1": _make_summary(daily_pnl=_daily_series(90, 50.0))}}
        report = sa.compute(summaries)
        assert len(report.proposed_changes) == 0
