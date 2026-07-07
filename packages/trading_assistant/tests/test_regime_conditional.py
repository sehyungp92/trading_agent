"""Tests for regime-conditional metrics in StrategyEngine."""
from __future__ import annotations

import pytest

from trading_assistant.analysis.strategy_engine import StrategyEngine
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.regime_conditional import (
    RegimeAllocation,
    RegimeConditionalReport,
    RegimeDistribution,
    RegimeStrategyMetrics,
)
from trading_assistant.schemas.weekly_metrics import StrategyWeeklySummary
from tests.factories import make_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    bot_id: str = "bot1",
    pair: str = "NQ",
    pnl: float = 100.0,
    market_regime: str = "trending",
    entry_signal: str = "momentum",
    strategy_id: str = "",
    entry_time: str = "2026-01-05T10:00:00",
    exit_time: str = "2026-01-05T14:00:00",
) -> TradeEvent:
    return make_trade(
        trade_id=f"t-{bot_id}-{pnl}-{market_regime}",
        bot_id=bot_id,
        pair=pair,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=100.0 + pnl / 10,
        position_size=10.0,
        pnl=pnl,
        pnl_pct=pnl / 1000 * 100,
        market_regime=market_regime,
        entry_signal=entry_signal,
        strategy_id=strategy_id,
    )


def _make_summary(
    bot_id: str = "bot1",
    strategy_id: str = "s1",
) -> StrategyWeeklySummary:
    return StrategyWeeklySummary(
        strategy_id=strategy_id,
        bot_id=bot_id,
        total_trades=10,
    )


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------

class TestSchemaDefaults:
    def test_regime_strategy_metrics_defaults(self):
        m = RegimeStrategyMetrics(bot_id="b", strategy_id="s", regime="trending")
        assert m.trade_count == 0
        assert m.win_rate == 0.0
        assert m.sharpe == 0.0

    def test_regime_allocation_defaults(self):
        a = RegimeAllocation(regime="trending")
        assert a.allocations == {}
        assert a.rationale == ""

    def test_regime_distribution_defaults(self):
        d = RegimeDistribution(regime="trending")
        assert d.pct_of_time == 0.0
        assert d.trade_count == 0

    def test_regime_conditional_report_defaults(self):
        r = RegimeConditionalReport(week_start="2026-01-01", week_end="2026-01-07")
        assert r.metrics == []
        assert r.optimal_allocations == []
        assert r.suggestions == []


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_empty_trades(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        report = engine.compute_regime_conditional_metrics({}, {})
        assert report.metrics == []
        assert report.regime_distribution == []
        assert report.optimal_allocations == []

    def test_single_bot_no_trades(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        summaries = {"bot1": {"s1": _make_summary()}}
        report = engine.compute_regime_conditional_metrics(summaries, {"bot1": []})
        assert report.metrics == []


# ---------------------------------------------------------------------------
# Single regime
# ---------------------------------------------------------------------------

class TestSingleRegime:
    def test_single_regime_single_strategy(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(pnl=100.0), _make_trade(pnl=-50.0)]
        summaries = {"bot1": {"momentum": _make_summary()}}
        report = engine.compute_regime_conditional_metrics(summaries, {"bot1": trades})
        assert len(report.metrics) == 1
        m = report.metrics[0]
        assert m.regime == "trending"
        assert m.trade_count == 2
        assert m.win_rate == pytest.approx(0.5)

    def test_expectancy_calculation(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(pnl=100.0), _make_trade(pnl=-50.0)]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        m = report.metrics[0]
        assert m.expectancy == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Multiple regimes
# ---------------------------------------------------------------------------

class TestMultipleRegimes:
    def test_multiple_regimes_split(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [
            _make_trade(pnl=100.0, market_regime="trending"),
            _make_trade(pnl=200.0, market_regime="trending"),
            _make_trade(pnl=-50.0, market_regime="ranging"),
        ]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        assert len(report.metrics) == 2

    def test_regime_distribution(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [
            _make_trade(pnl=100.0, market_regime="trending"),
            _make_trade(pnl=200.0, market_regime="trending"),
            _make_trade(pnl=-50.0, market_regime="ranging"),
        ]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        dist = {d.regime: d for d in report.regime_distribution}
        assert dist["trending"].pct_of_time == pytest.approx(66.7, abs=0.1)
        assert dist["ranging"].pct_of_time == pytest.approx(33.3, abs=0.1)
        assert dist["trending"].trade_count == 2
        assert dist["ranging"].trade_count == 1


# ---------------------------------------------------------------------------
# Optimal allocations
# ---------------------------------------------------------------------------

class TestOptimalAllocations:
    def test_allocation_per_regime(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [
            _make_trade(pnl=100.0, market_regime="trending", strategy_id="s1"),
            _make_trade(pnl=100.0, market_regime="trending", strategy_id="s1"),
            _make_trade(pnl=50.0, market_regime="trending", strategy_id="s2"),
            _make_trade(pnl=50.0, market_regime="trending", strategy_id="s2"),
        ]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        allocs = {a.regime: a for a in report.optimal_allocations}
        assert "trending" in allocs
        # Both strategies should get some allocation
        assert len(allocs["trending"].allocations) == 2

    def test_zero_trade_regime_still_appears(self):
        """All regimes with trades should produce metrics."""
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(pnl=100.0, market_regime="volatile")]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        assert len(report.regime_distribution) == 1


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

class TestSuggestions:
    def test_suggestion_for_poor_regime(self):
        """Strategy with low win rate and negative expectancy in a regime gets suggestion."""
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(pnl=-20.0, market_regime="ranging") for _ in range(12)]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        assert len(report.suggestions) >= 1
        assert report.suggestions[0]["regime"] == "ranging"
        assert "reduce" in report.suggestions[0]["suggested_alloc"]

    def test_no_suggestion_for_good_regime(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(pnl=100.0, market_regime="trending") for _ in range(12)]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        assert len(report.suggestions) == 0

    def test_no_suggestion_below_trade_threshold(self):
        """Fewer than 10 trades in a regime → no suggestion."""
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(pnl=-20.0, market_regime="ranging") for _ in range(5)]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        assert len(report.suggestions) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_strategy_single_trade(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(pnl=100.0)]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        assert len(report.metrics) == 1
        assert report.metrics[0].trade_count == 1

    def test_multiple_bots(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades_bot1 = [_make_trade(bot_id="bot1", pnl=100.0)]
        trades_bot2 = [_make_trade(bot_id="bot2", pnl=-50.0, strategy_id="helix")]
        report = engine.compute_regime_conditional_metrics(
            {},
            {"bot1": trades_bot1, "bot2": trades_bot2},
        )
        assert len(report.metrics) == 2

    def test_unknown_regime(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        trades = [_make_trade(market_regime="")]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        assert report.metrics[0].regime == "unknown"

    def test_max_drawdown_calculation(self):
        engine = StrategyEngine("2026-01-01", "2026-01-07")
        # Create trades that produce a drawdown
        trades = [
            _make_trade(pnl=100.0, entry_time="2026-01-05T10:00:00", exit_time="2026-01-05T11:00:00"),
            _make_trade(pnl=-80.0, entry_time="2026-01-05T12:00:00", exit_time="2026-01-05T13:00:00"),
        ]
        report = engine.compute_regime_conditional_metrics({}, {"bot1": trades})
        m = report.metrics[0]
        assert m.max_drawdown_pct > 0
