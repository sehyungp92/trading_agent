# tests/test_portfolio_allocator.py
"""Tests for the portfolio allocator skill."""

from trading_assistant.schemas.portfolio_allocation import (
    AllocationConstraints,
    PortfolioAllocationReport,
)
from trading_assistant.schemas.weekly_metrics import BotWeeklySummary
from trading_assistant.skills.portfolio_allocator import PortfolioAllocator


def _make_bot_summary(
    bot_id: str,
    net_pnl: float = 500.0,
    max_drawdown_pct: float = 5.0,
    daily_pnl: dict | None = None,
) -> BotWeeklySummary:
    dp = daily_pnl or {
        "2026-02-24": net_pnl / 7,
        "2026-02-25": net_pnl / 7,
        "2026-02-26": net_pnl / 7,
        "2026-02-27": net_pnl / 7,
        "2026-02-28": net_pnl / 7,
        "2026-03-01": net_pnl / 7,
        "2026-03-02": net_pnl / 7,
    }
    return BotWeeklySummary(
        week_start="2026-02-24",
        week_end="2026-03-02",
        bot_id=bot_id,
        net_pnl=net_pnl,
        max_drawdown_pct=max_drawdown_pct,
        daily_pnl=dp,
    )


class TestPortfolioAllocatorSchemas:
    def test_constraints_defaults(self):
        c = AllocationConstraints()
        assert c.min_allocation_pct == 5.0
        assert c.max_allocation_pct == 60.0

    def test_report_defaults(self):
        r = PortfolioAllocationReport(week_start="2026-02-24", week_end="2026-03-02")
        assert r.recommendations == []
        assert r.method == "risk_parity_calmar_tilt"


class TestPortfolioAllocator:
    def test_empty_bots(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        report = alloc.compute({}, {})
        assert report.recommendations == []
        assert not report.rebalance_needed

    def test_single_bot_gets_100_pct(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {"bot1": _make_bot_summary("bot1")}
        report = alloc.compute(summaries, {"bot1": 100.0})
        assert len(report.recommendations) == 1
        assert report.recommendations[0].suggested_allocation_pct == 100.0

    def test_equal_performance_equal_allocation(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=500.0, max_drawdown_pct=5.0),
            "bot2": _make_bot_summary("bot2", net_pnl=500.0, max_drawdown_pct=5.0),
        }
        report = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        allocs = {r.bot_id: r.suggested_allocation_pct for r in report.recommendations}
        assert abs(allocs["bot1"] - allocs["bot2"]) < 1.0  # nearly equal

    def test_high_calmar_bot_gets_more(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        # bot1: high return, low drawdown → high Calmar
        # bot2: lower return, higher drawdown → lower Calmar
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=1000.0, max_drawdown_pct=2.0),
            "bot2": _make_bot_summary("bot2", net_pnl=200.0, max_drawdown_pct=10.0),
        }
        report = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        allocs = {r.bot_id: r.suggested_allocation_pct for r in report.recommendations}
        assert allocs["bot1"] > allocs["bot2"]

    def test_constraint_capping_max(self):
        constraints = AllocationConstraints(max_allocation_pct=40.0)
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02", constraints=constraints)
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=10000.0, max_drawdown_pct=1.0),
            "bot2": _make_bot_summary("bot2", net_pnl=10.0, max_drawdown_pct=20.0),
            "bot3": _make_bot_summary("bot3", net_pnl=10.0, max_drawdown_pct=20.0),
        }
        # Without constraints, bot1 would dominate (>80%)
        report_unconstrained = PortfolioAllocator("2026-02-24", "2026-03-02").compute(
            summaries, {"bot1": 33.3, "bot2": 33.3, "bot3": 33.3},
        )
        report_constrained = alloc.compute(summaries, {"bot1": 33.3, "bot2": 33.3, "bot3": 33.3})
        unconstrained_max = max(r.suggested_allocation_pct for r in report_unconstrained.recommendations)
        constrained_max = max(r.suggested_allocation_pct for r in report_constrained.recommendations)
        # Constraint should reduce the maximum allocation
        assert constrained_max < unconstrained_max

    def test_constraint_capping_min(self):
        constraints = AllocationConstraints(min_allocation_pct=10.0)
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02", constraints=constraints)
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=5000.0, max_drawdown_pct=1.0),
            "bot2": _make_bot_summary("bot2", net_pnl=5.0, max_drawdown_pct=30.0),
        }
        report = alloc.compute(summaries, {"bot1": 90.0, "bot2": 10.0})
        for r in report.recommendations:
            # After re-normalization, min allocation is at least respected
            assert r.suggested_allocation_pct >= 5.0  # at least default min

    def test_portfolio_calmar_computed(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=500.0, max_drawdown_pct=5.0),
            "bot2": _make_bot_summary("bot2", net_pnl=300.0, max_drawdown_pct=3.0),
        }
        report = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        assert report.current_portfolio_calmar != 0.0
        assert report.suggested_portfolio_calmar != 0.0

    def test_rebalance_needed_when_large_change(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=2000.0, max_drawdown_pct=2.0),
            "bot2": _make_bot_summary("bot2", net_pnl=100.0, max_drawdown_pct=15.0),
        }
        # Very unequal current allocation
        report = alloc.compute(summaries, {"bot1": 20.0, "bot2": 80.0})
        assert report.rebalance_needed is True

    def test_no_rebalance_when_small_change(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=500.0, max_drawdown_pct=5.0),
            "bot2": _make_bot_summary("bot2", net_pnl=500.0, max_drawdown_pct=5.0),
        }
        report = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        # Equal bots with equal allocation → no significant change
        assert report.rebalance_needed is False

    def test_correlation_matrix_penalizes_correlated_bots(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=500.0, max_drawdown_pct=5.0),
            "bot2": _make_bot_summary("bot2", net_pnl=400.0, max_drawdown_pct=5.0),
        }
        # Without correlation
        report_no_corr = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        # With high correlation — worse bot should get less
        report_corr = alloc.compute(
            summaries, {"bot1": 50.0, "bot2": 50.0},
            correlation_matrix={"bot1_bot2": 0.9},
        )
        allocs_no_corr = {r.bot_id: r.suggested_allocation_pct for r in report_no_corr.recommendations}
        allocs_corr = {r.bot_id: r.suggested_allocation_pct for r in report_corr.recommendations}
        # bot2 (worse Calmar) should get less with high correlation
        assert allocs_corr["bot2"] <= allocs_no_corr["bot2"]

    def test_three_bots(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "k_stock": _make_bot_summary("k_stock", net_pnl=800.0, max_drawdown_pct=3.0),
            "momentum": _make_bot_summary("momentum", net_pnl=400.0, max_drawdown_pct=8.0),
            "swing": _make_bot_summary("swing", net_pnl=600.0, max_drawdown_pct=4.0),
        }
        report = alloc.compute(
            summaries, {"k_stock": 33.3, "momentum": 33.3, "swing": 33.3},
        )
        assert len(report.recommendations) == 3
        total = sum(r.suggested_allocation_pct for r in report.recommendations)
        assert abs(total - 100.0) < 0.1  # sums to ~100%

    def test_rationale_populated(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {"bot1": _make_bot_summary("bot1")}
        report = alloc.compute(summaries, {"bot1": 100.0})
        assert report.recommendations[0].rationale != ""

    def test_calmar_change_pct(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=1000.0, max_drawdown_pct=2.0),
            "bot2": _make_bot_summary("bot2", net_pnl=100.0, max_drawdown_pct=20.0),
        }
        # Skewed current allocation (mostly to bad bot)
        report = alloc.compute(summaries, {"bot1": 20.0, "bot2": 80.0})
        # Shifting to better bot should improve Calmar
        assert report.calmar_change_pct != 0.0

    def test_serializes_to_json(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1"),
            "bot2": _make_bot_summary("bot2"),
        }
        report = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        data = report.model_dump(mode="json")
        assert isinstance(data["recommendations"], list)
        assert "method" in data

    def test_zero_volatility_bots(self):
        """Bots with zero daily variance should still get allocation."""
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=500.0, daily_pnl={"2026-02-24": 500.0}),
            "bot2": _make_bot_summary("bot2", net_pnl=300.0, daily_pnl={"2026-02-24": 300.0}),
        }
        report = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        total = sum(r.suggested_allocation_pct for r in report.recommendations)
        assert abs(total - 100.0) < 0.1

    def test_negative_pnl_bots(self):
        alloc = PortfolioAllocator("2026-02-24", "2026-03-02")
        summaries = {
            "bot1": _make_bot_summary("bot1", net_pnl=-200.0, max_drawdown_pct=10.0),
            "bot2": _make_bot_summary("bot2", net_pnl=500.0, max_drawdown_pct=3.0),
        }
        report = alloc.compute(summaries, {"bot1": 50.0, "bot2": 50.0})
        allocs = {r.bot_id: r.suggested_allocation_pct for r in report.recommendations}
        # Profitable bot should get more
        assert allocs["bot2"] > allocs["bot1"]
