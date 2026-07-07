# tests/test_backtest_simulator.py
"""Tests for the backtest simulator."""
from datetime import datetime

from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.schemas.cost_model import CostModelConfig
from trading_assistant.skills.backtest_simulator import BacktestSimulator
from trading_assistant.skills.cost_model import CostModel


def _trade(
    trade_id: str,
    pnl: float,
    entry_price: float = 40000.0,
    position_size: float = 0.1,
    entry_signal_strength: float = 0.8,
    market_regime: str = "trending_up",
    entry_signal: str = "ema_cross",
    **kwargs,
) -> TradeEvent:
    return TradeEvent(
        trade_id=trade_id,
        bot_id="bot1",
        pair="BTCUSDT",
        side="LONG",
        entry_time=datetime(2026, 1, 15, 12, 0),
        exit_time=datetime(2026, 1, 15, 14, 0),
        entry_price=entry_price,
        exit_price=entry_price + pnl / position_size,
        position_size=position_size,
        pnl=pnl,
        pnl_pct=pnl / (entry_price * position_size) * 100,
        entry_signal=entry_signal,
        entry_signal_strength=entry_signal_strength,
        market_regime=market_regime,
        **kwargs,
    )


def _missed(
    bot_id: str = "bot1",
    signal_strength: float = 0.6,
    blocked_by: str = "volume_filter",
    outcome_24h: float = 100.0,
    hypothetical_entry: float = 40000.0,
) -> MissedOpportunityEvent:
    return MissedOpportunityEvent(
        bot_id=bot_id,
        pair="BTCUSDT",
        signal="ema_cross",
        signal_strength=signal_strength,
        blocked_by=blocked_by,
        hypothetical_entry=hypothetical_entry,
        outcome_24h=outcome_24h,
        confidence=0.7,
        assumption_tags=["next_trade_fill", "5bps_slippage"],
    )


def _zero_cost_model() -> CostModel:
    return CostModel(CostModelConfig(fees_per_trade_bps=0.0, fixed_slippage_bps=0.0))


def _real_cost_model() -> CostModel:
    return CostModel(CostModelConfig(fees_per_trade_bps=7.0, fixed_slippage_bps=5.0))


class TestBasicSimulation:
    def test_all_trades_pass_with_no_params(self):
        trades = [_trade("t1", 100.0), _trade("t2", -50.0), _trade("t3", 200.0)]
        sim = BacktestSimulator(_zero_cost_model())
        result = sim.simulate(trades, [], params={})
        assert result.total_trades == 3
        assert result.win_count == 2
        assert result.loss_count == 1
        assert result.gross_pnl == 250.0

    def test_filters_by_signal_strength_threshold(self):
        trades = [
            _trade("t1", 100.0, entry_signal_strength=0.9),
            _trade("t2", -50.0, entry_signal_strength=0.3),
            _trade("t3", 200.0, entry_signal_strength=0.7),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        # Only trades with signal_strength >= 0.5 pass
        result = sim.simulate(trades, [], params={"signal_strength_min": 0.5})
        assert result.total_trades == 2
        assert result.gross_pnl == 300.0

    def test_empty_trades(self):
        sim = BacktestSimulator(_zero_cost_model())
        result = sim.simulate([], [], params={})
        assert result.total_trades == 0
        assert result.gross_pnl == 0.0


class TestCostIntegration:
    def test_costs_reduce_net_pnl(self):
        trades = [_trade("t1", 100.0, entry_price=40000.0, position_size=0.1)]
        sim = BacktestSimulator(_real_cost_model())
        result = sim.simulate(trades, [], params={})
        assert result.gross_pnl == 100.0
        assert result.net_pnl < result.gross_pnl
        assert result.total_fees > 0
        assert result.total_slippage > 0

    def test_cost_multiplier(self):
        trades = [_trade("t1", 100.0, entry_price=40000.0, position_size=0.1)]
        sim = BacktestSimulator(_real_cost_model())
        base = sim.simulate(trades, [], params={}, cost_multiplier=1.0)
        doubled = sim.simulate(trades, [], params={}, cost_multiplier=2.0)
        assert doubled.total_fees > base.total_fees
        assert doubled.net_pnl < base.net_pnl


class TestMissedOpportunityInclusion:
    def test_includes_missed_when_filter_relaxed(self):
        trades = [_trade("t1", 100.0)]
        missed = [_missed(outcome_24h=150.0, blocked_by="volume_filter")]
        sim = BacktestSimulator(_zero_cost_model())
        # include_blocked_by=["volume_filter"] means we now take those missed opps
        result = sim.simulate(trades, missed, params={"include_blocked_by": ["volume_filter"]})
        assert result.total_trades == 2
        assert result.gross_pnl == 250.0  # 100 + 150

    def test_does_not_include_missed_without_param(self):
        trades = [_trade("t1", 100.0)]
        missed = [_missed(outcome_24h=150.0)]
        sim = BacktestSimulator(_zero_cost_model())
        result = sim.simulate(trades, missed, params={})
        assert result.total_trades == 1


class TestMetricsComputation:
    def test_regime_breakdown(self):
        trades = [
            _trade("t1", 100.0, market_regime="trending_up"),
            _trade("t2", -50.0, market_regime="ranging"),
            _trade("t3", 200.0, market_regime="trending_up"),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        result = sim.simulate(trades, [], params={})
        assert result.trades_by_regime["trending_up"] == 2
        assert result.trades_by_regime["ranging"] == 1
        assert result.pnl_by_regime["trending_up"] == 300.0
        assert result.pnl_by_regime["ranging"] == -50.0

    def test_sharpe_ratio_positive(self):
        trades = [
            _trade("t1", 100.0),
            _trade("t2", 80.0),
            _trade("t3", 120.0),
            _trade("t4", 90.0),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        result = sim.simulate(trades, [], params={})
        assert result.sharpe_ratio > 0

    def test_max_drawdown(self):
        trades = [
            _trade("t1", 100.0),
            _trade("t2", -200.0),
            _trade("t3", 50.0),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        result = sim.simulate(trades, [], params={})
        assert result.max_drawdown_pct > 0
