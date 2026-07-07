# tests/test_robustness_tester.py
"""Tests for the robustness tester."""
from datetime import datetime

from trading_assistant.schemas.cost_model import CostModelConfig
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.parameter_space import ParameterDef, ParameterSpace, RobustnessConfig
from trading_assistant.schemas.validation_results import RobustnessResult
from trading_assistant.skills.backtest_simulator import BacktestSimulator
from trading_assistant.skills.cost_model import CostModel
from trading_assistant.skills.robustness_tester import RobustnessTester


def _trade(trade_id: str, pnl: float, regime: str = "trending_up", signal_strength: float = 0.8) -> TradeEvent:
    return TradeEvent(
        trade_id=trade_id,
        bot_id="bot1",
        pair="BTCUSDT",
        side="LONG",
        entry_time=datetime(2026, 1, 15, 12, 0),
        exit_time=datetime(2026, 1, 15, 14, 0),
        entry_price=40000.0,
        exit_price=40000.0 + pnl / 0.1,
        position_size=0.1,
        pnl=pnl,
        pnl_pct=pnl / 4000 * 100,
        market_regime=regime,
        entry_signal_strength=signal_strength,
    )


def _zero_cost_model() -> CostModel:
    return CostModel(CostModelConfig(fees_per_trade_bps=0.0, fixed_slippage_bps=0.0))


class TestNeighborhoodStability:
    def test_stable_neighborhood(self):
        cfg = RobustnessConfig(neighborhood_pct=0.1)
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="signal_strength_min", min_value=0.1, max_value=0.9, step=0.1, current_value=0.5),
            ],
        )
        trades = [
            _trade("t1", 100.0, signal_strength=0.9),
            _trade("t2", 80.0, signal_strength=0.8),
            _trade("t3", 60.0, signal_strength=0.7),
            _trade("t4", 40.0, signal_strength=0.6),
            _trade("t5", 20.0, signal_strength=0.5),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        tester = RobustnessTester(cfg, space, sim)
        best_params = {"signal_strength_min": 0.5}
        result = tester.test_neighborhood(trades, [], best_params)
        assert isinstance(result, dict)
        # Neighborhood params should all be tested
        assert len(result) >= 1

    def test_detects_unstable_neighborhood(self):
        cfg = RobustnessConfig(neighborhood_pct=0.5)  # large neighborhood
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="signal_strength_min", min_value=0.1, max_value=0.9, step=0.1, current_value=0.5),
            ],
        )
        trades = [
            _trade("t1", 100.0, signal_strength=0.51),  # only passes at exactly 0.5
            _trade("t2", -1000.0, signal_strength=0.3),  # included when threshold drops to 0.25
        ]
        sim = BacktestSimulator(_zero_cost_model())
        tester = RobustnessTester(cfg, space, sim)
        best_params = {"signal_strength_min": 0.5}
        scores = tester.test_neighborhood(trades, [], best_params)
        # Wide neighborhood should show degradation at 0.25 (includes the -1000 trade)
        assert any(v < 0 for v in scores.values())


class TestRegimeStability:
    def test_stable_across_regimes(self):
        cfg = RobustnessConfig(min_profitable_regimes=3, total_regime_types=4)
        trades = [
            _trade("t1", 100.0, regime="trending_up"),
            _trade("t2", 80.0, regime="trending_down"),
            _trade("t3", 60.0, regime="ranging"),
            _trade("t4", 40.0, regime="volatile"),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        space = ParameterSpace(bot_id="bot1", parameters=[])
        tester = RobustnessTester(cfg, space, sim)
        regime_pnl, count = tester.test_regime_stability(trades, [], {})
        assert count >= 3
        assert all(v > 0 for v in regime_pnl.values())

    def test_unstable_in_some_regimes(self):
        cfg = RobustnessConfig(min_profitable_regimes=3, total_regime_types=4)
        trades = [
            _trade("t1", 100.0, regime="trending_up"),
            _trade("t2", -200.0, regime="ranging"),
            _trade("t3", -150.0, regime="volatile"),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        space = ParameterSpace(bot_id="bot1", parameters=[])
        tester = RobustnessTester(cfg, space, sim)
        regime_pnl, count = tester.test_regime_stability(trades, [], {})
        assert count < 3


class TestFullRobustnessEvaluation:
    def test_produces_result(self):
        cfg = RobustnessConfig()
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="signal_strength_min", min_value=0.3, max_value=0.7, step=0.1, current_value=0.5),
            ],
        )
        trades = [
            _trade("t1", 100.0, regime="trending_up", signal_strength=0.9),
            _trade("t2", 80.0, regime="trending_down", signal_strength=0.8),
            _trade("t3", 60.0, regime="ranging", signal_strength=0.7),
            _trade("t4", 40.0, regime="volatile", signal_strength=0.6),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        tester = RobustnessTester(cfg, space, sim)
        result = tester.evaluate(trades, [], {"signal_strength_min": 0.5})
        assert isinstance(result, RobustnessResult)
        assert 0 <= result.robustness_score <= 100


class TestSafetyFlags:
    def test_flat_surface_flag(self):
        cfg = RobustnessConfig()
        space = ParameterSpace(bot_id="bot1", parameters=[])
        sim = BacktestSimulator(_zero_cost_model())
        tester = RobustnessTester(cfg, space, sim)
        # All neighborhood scores identical → flat surface
        scores = {"a=1": 1.0, "a=2": 1.0, "a=3": 1.0}
        flags = tester.detect_safety_flags(scores, regime_stable=True, best_sharpe=1.0)
        flag_types = [f.flag_type for f in flags]
        assert "low_conviction" in flag_types

    def test_spiky_surface_flag(self):
        cfg = RobustnessConfig()
        space = ParameterSpace(bot_id="bot1", parameters=[])
        sim = BacktestSimulator(_zero_cost_model())
        tester = RobustnessTester(cfg, space, sim)
        # Best is 3.0 but neighbors drop by >50%
        scores = {"a=1": 0.5, "a=2": 3.0, "a=3": 0.8}
        flags = tester.detect_safety_flags(scores, regime_stable=True, best_sharpe=3.0)
        flag_types = [f.flag_type for f in flags]
        assert "likely_overfit" in flag_types

    def test_no_flags_for_healthy_surface(self):
        cfg = RobustnessConfig()
        space = ParameterSpace(bot_id="bot1", parameters=[])
        sim = BacktestSimulator(_zero_cost_model())
        tester = RobustnessTester(cfg, space, sim)
        scores = {"a=1": 1.6, "a=2": 2.0, "a=3": 1.8}
        flags = tester.detect_safety_flags(scores, regime_stable=True, best_sharpe=2.0)
        flag_types = [f.flag_type for f in flags]
        assert "likely_overfit" not in flag_types
        assert "low_conviction" not in flag_types
