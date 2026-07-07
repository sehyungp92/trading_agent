# tests/test_exit_strategy_simulator.py
"""Tests for exit strategy comparison simulator."""

from trading_assistant.schemas.exit_simulation import ExitStrategyType, ExitStrategyConfig, ExitSimulationResult
from tests.factories import make_trade


def _make_trade(trade_id, entry_price, exit_price, pnl, exit_reason="SIGNAL",
                post_1h=None, post_4h=None, atr=2.0):
    return make_trade(trade_id=trade_id, entry_price=entry_price, exit_price=exit_price,
                      pnl=pnl, exit_reason=exit_reason, atr_at_entry=atr,
                      post_exit_1h_price=post_1h, post_exit_4h_price=post_4h)


class TestExitSimulationSchema:
    def test_strategy_config(self):
        cfg = ExitStrategyConfig(
            strategy_type=ExitStrategyType.TRAILING_STOP,
            params={"trail_pct": 2.0},
        )
        assert cfg.strategy_type == ExitStrategyType.TRAILING_STOP

    def test_result_model(self):
        r = ExitSimulationResult(
            strategy=ExitStrategyConfig(
                strategy_type=ExitStrategyType.FIXED_STOP,
                params={"stop_pct": 1.0},
            ),
            total_trades=10, trades_with_data=8,
            baseline_pnl=500.0, simulated_pnl=650.0,
        )
        assert r.improvement == 150.0


class TestExitStrategySimulator:
    def test_wider_stop_captures_more_on_favorable_continuation(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        trades = [
            # Stopped out at 98, but price recovered to 105 at 1h
            _make_trade("t1", 100, 98, -2.0, exit_reason="STOP_LOSS", post_1h=105.0),
        ]
        sim = ExitStrategySimulator()
        config = ExitStrategyConfig(
            strategy_type=ExitStrategyType.FIXED_STOP,
            params={"stop_pct": 5.0},  # wider stop (5% vs whatever original was)
        )
        result = sim.simulate(trades, config)
        assert result.trades_with_data == 1
        # With wider stop, would have captured the recovery
        assert result.simulated_pnl > result.baseline_pnl

    def test_skips_trades_without_post_exit_data(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        trades = [
            _make_trade("t1", 100, 105, 5.0),  # no post-exit data
            _make_trade("t2", 100, 103, 3.0, post_1h=107.0),
        ]
        sim = ExitStrategySimulator()
        config = ExitStrategyConfig(
            strategy_type=ExitStrategyType.FIXED_STOP,
            params={"stop_pct": 2.0},
        )
        result = sim.simulate(trades, config)
        assert result.total_trades == 2
        assert result.trades_with_data == 1

    def test_time_based_exit_comparison(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        trades = [
            # Exited by signal at 103, but at 4h price was 110
            _make_trade("t1", 100, 103, 3.0, exit_reason="SIGNAL", post_4h=110.0),
        ]
        sim = ExitStrategySimulator()
        config = ExitStrategyConfig(
            strategy_type=ExitStrategyType.TIME_BASED,
            params={"hold_hours": 4},
        )
        result = sim.simulate(trades, config)
        assert result.simulated_pnl > result.baseline_pnl

    def test_empty_trades(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        sim = ExitStrategySimulator()
        config = ExitStrategyConfig(
            strategy_type=ExitStrategyType.FIXED_STOP,
            params={"stop_pct": 2.0},
        )
        result = sim.simulate([], config)
        assert result.total_trades == 0
        assert result.simulated_pnl == 0.0


class TestExitSweep:
    def test_sweep_returns_all_12_configs(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        sim = ExitStrategySimulator()
        trades = [
            _make_trade("t1", 100, 103, 3.0, post_1h=105.0, post_4h=107.0),
        ]
        results = sim.sweep(trades)
        assert len(results) == 12
        # Verify all 4 strategy types appear
        types = {r.strategy.strategy_type for r in results}
        assert types == {
            ExitStrategyType.FIXED_STOP,
            ExitStrategyType.TRAILING_STOP,
            ExitStrategyType.ATR_STOP,
            ExitStrategyType.TIME_BASED,
        }

    def test_sweep_custom_configs(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        sim = ExitStrategySimulator()
        custom = [
            ExitStrategyConfig(strategy_type=ExitStrategyType.FIXED_STOP, params={"stop_pct": 5.0}),
            ExitStrategyConfig(strategy_type=ExitStrategyType.TIME_BASED, params={"hold_hours": 1}),
        ]
        results = sim.sweep([], configs=custom)
        assert len(results) == 2

    def test_sweep_empty_trades(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        sim = ExitStrategySimulator()
        results = sim.sweep([])
        assert len(results) == 12
        for r in results:
            assert r.baseline_pnl == 0.0
            assert r.simulated_pnl == 0.0

    def test_sweep_identifies_best_improvement(self):
        from trading_assistant.skills.exit_strategy_simulator import ExitStrategySimulator

        sim = ExitStrategySimulator()
        trades = [
            # Stopped out early, but price recovered significantly
            _make_trade("t1", 100, 98, -2.0, exit_reason="STOP_LOSS",
                        post_1h=105.0, post_4h=110.0),
        ]
        results = sim.sweep(trades)
        best = max(results, key=lambda r: r.improvement)
        # Best improvement should be positive (some strategy beats actual)
        assert best.improvement > 0
        assert best.strategy is not None


class TestExitSweepResult:
    def test_schema(self):
        from trading_assistant.schemas.exit_simulation import ExitSweepResult

        sweep = ExitSweepResult(
            bot_id="bot1",
            configs_tested=12,
            baseline_pnl=100.0,
            results=[],
            best_strategy=ExitStrategyConfig(
                strategy_type=ExitStrategyType.TRAILING_STOP,
                params={"trail_pct": 2.0},
            ),
            best_improvement=50.0,
        )
        assert sweep.bot_id == "bot1"
        assert sweep.configs_tested == 12
        d = sweep.model_dump(mode="json")
        assert d["best_strategy"]["strategy_type"] == "trailing_stop"
