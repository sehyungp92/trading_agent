"""Tests for regime exclusion P&L computation in strategy engine."""
from trading_assistant.schemas.weekly_metrics import RegimePerformanceTrend
from trading_assistant.analysis.strategy_engine import StrategyEngine
from tests.factories import make_trade


def _make_trade(trade_id, pnl, regime="trending"):
    return make_trade(trade_id=trade_id, pnl=pnl, market_regime=regime)


class TestRegimeExclusionPnL:
    def test_compute_exclusion_impact(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        trades = [
            _make_trade("t1", 10.0, "trending"),
            _make_trade("t2", 8.0, "trending"),
            _make_trade("t3", -15.0, "ranging"),
            _make_trade("t4", -12.0, "ranging"),
            _make_trade("t5", 5.0, "volatile"),
        ]
        impact = engine.compute_regime_exclusion_impact("bot1", trades, "ranging")
        assert impact["baseline_pnl"] == -4.0
        assert impact["excluded_pnl"] == 23.0
        assert impact["delta_pnl"] == 27.0
        assert impact["excluded_trade_count"] == 2

    def test_exclusion_impact_empty_regime(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        trades = [_make_trade("t1", 10.0, "trending")]
        impact = engine.compute_regime_exclusion_impact("bot1", trades, "nonexistent")
        assert impact["delta_pnl"] == 0.0

    def test_regime_gate_suggestions_include_quantified_impact(self):
        engine = StrategyEngine(
            week_start="2026-02-24", week_end="2026-03-02",
            regime_min_weeks=2,
        )
        trends = [
            RegimePerformanceTrend(
                bot_id="bot1", regime="ranging", weekly_pnl=[-100.0, -80.0, -90.0],
            ),
        ]
        trades = [
            _make_trade("t1", 10.0, "trending"),
            _make_trade("t2", -50.0, "ranging"),
        ]
        suggestions = engine.analyze_regime_fit_quantified("bot1", trends, trades)
        assert len(suggestions) == 1
        assert "$" in suggestions[0].description

    def test_quantified_without_trades_still_works(self):
        engine = StrategyEngine(
            week_start="2026-02-24", week_end="2026-03-02",
            regime_min_weeks=2,
        )
        trends = [
            RegimePerformanceTrend(
                bot_id="bot1", regime="ranging", weekly_pnl=[-100.0, -80.0],
            ),
        ]
        suggestions = engine.analyze_regime_fit_quantified("bot1", trends)
        assert len(suggestions) == 1
        assert "regime gate" in suggestions[0].description.lower()
