"""Exit strategy comparison simulator.

Compares actual exit results against alternative strategies using
post-exit price data (1h and 4h snapshots). Limited precision without
tick-level data, but sufficient for directional comparison.
"""
from __future__ import annotations

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.exit_simulation import (
    ExitSimulationResult,
    ExitStrategyConfig,
    ExitStrategyType,
    TradeExitComparison,
)


class ExitStrategySimulator:
    """Simulate alternative exit strategies using post-exit price snapshots."""

    def sweep(
        self, trades: list[TradeEvent], configs: list[ExitStrategyConfig] | None = None,
    ) -> list[ExitSimulationResult]:
        """Run simulate() across multiple configs and return all results."""
        if configs is None:
            configs = self.default_sweep_configs()
        return [self.simulate(trades, c) for c in configs]

    @staticmethod
    def default_sweep_configs() -> list[ExitStrategyConfig]:
        """12 configs: 4 types x 3 parameter values."""
        configs: list[ExitStrategyConfig] = []
        for stop_pct in [1.0, 2.0, 3.0]:
            configs.append(ExitStrategyConfig(
                strategy_type=ExitStrategyType.FIXED_STOP,
                params={"stop_pct": stop_pct},
            ))
        for trail_pct in [1.0, 2.0, 3.0]:
            configs.append(ExitStrategyConfig(
                strategy_type=ExitStrategyType.TRAILING_STOP,
                params={"trail_pct": trail_pct},
            ))
        for atr_mult in [1.0, 2.0, 3.0]:
            configs.append(ExitStrategyConfig(
                strategy_type=ExitStrategyType.ATR_STOP,
                params={"atr_multiplier": atr_mult},
            ))
        for hold_hours in [1, 4, 24]:
            configs.append(ExitStrategyConfig(
                strategy_type=ExitStrategyType.TIME_BASED,
                params={"hold_hours": hold_hours},
            ))
        return configs

    def simulate(
        self, trades: list[TradeEvent], strategy: ExitStrategyConfig
    ) -> ExitSimulationResult:
        comparisons: list[TradeExitComparison] = []
        baseline_pnl = sum(t.pnl for t in trades)

        for t in trades:
            if t.post_exit_1h_price is None and t.post_exit_4h_price is None:
                continue

            sim_pnl = self._simulate_trade(t, strategy)
            comparisons.append(TradeExitComparison(
                trade_id=t.trade_id,
                actual_pnl=t.pnl,
                simulated_pnl=sim_pnl,
            ))

        simulated_total = sum(c.simulated_pnl for c in comparisons)
        # For trades without data, keep actual PnL
        no_data_pnl = sum(
            t.pnl for t in trades
            if t.post_exit_1h_price is None and t.post_exit_4h_price is None
        )

        return ExitSimulationResult(
            strategy=strategy,
            total_trades=len(trades),
            trades_with_data=len(comparisons),
            baseline_pnl=baseline_pnl,
            simulated_pnl=simulated_total + no_data_pnl,
            comparisons=comparisons,
        )

    def _simulate_trade(
        self, trade: TradeEvent, strategy: ExitStrategyConfig
    ) -> float:
        """Simulate a single trade exit under the alternative strategy."""
        is_long = trade.side.upper() == "LONG"
        entry = trade.entry_price
        post_1h = trade.post_exit_1h_price
        post_4h = trade.post_exit_4h_price

        if strategy.strategy_type == ExitStrategyType.FIXED_STOP:
            stop_pct = strategy.params.get("stop_pct", 2.0) / 100
            stop_price = entry * (1 - stop_pct) if is_long else entry * (1 + stop_pct)
            return self._best_exit(trade, stop_price, is_long)

        elif strategy.strategy_type == ExitStrategyType.TIME_BASED:
            hold_hours = strategy.params.get("hold_hours", 4)
            price = post_4h if hold_hours >= 4 and post_4h else post_1h
            if price is None:
                return trade.pnl
            return (price - entry) * trade.position_size if is_long else (entry - price) * trade.position_size

        elif strategy.strategy_type == ExitStrategyType.ATR_STOP:
            atr_mult = strategy.params.get("atr_multiplier", 2.0)
            atr = trade.atr_at_entry or 0
            stop_dist = atr * atr_mult
            stop_price = (entry - stop_dist) if is_long else (entry + stop_dist)
            return self._best_exit(trade, stop_price, is_long)

        elif strategy.strategy_type == ExitStrategyType.TRAILING_STOP:
            trail_pct = strategy.params.get("trail_pct", 2.0) / 100
            # Sequential trailing stop: process snapshots in time order
            snapshots = [p for p in [post_1h, post_4h] if p is not None]
            if is_long:
                peak = entry
                trail_stop = entry * (1 - trail_pct)
                for p in snapshots:
                    if p <= trail_stop:
                        return (trail_stop - entry) * trade.position_size
                    if p > peak:
                        peak = p
                        trail_stop = peak * (1 - trail_pct)
                # Check MAE against final trail level
                if trade.mae_price is not None and trade.mae_price <= trail_stop:
                    return (trail_stop - entry) * trade.position_size
                last = snapshots[-1] if snapshots else trade.exit_price
                return (last - entry) * trade.position_size
            else:
                trough = entry
                trail_stop = entry * (1 + trail_pct)
                for p in snapshots:
                    if p >= trail_stop:
                        return (entry - trail_stop) * trade.position_size
                    if p < trough:
                        trough = p
                        trail_stop = trough * (1 + trail_pct)
                if trade.mae_price is not None and trade.mae_price >= trail_stop:
                    return (entry - trail_stop) * trade.position_size
                last = snapshots[-1] if snapshots else trade.exit_price
                return (entry - last) * trade.position_size

        return trade.pnl  # fallback

    def _best_exit(self, trade: TradeEvent, stop_price: float, is_long: bool) -> float:
        """Determine best exit given stop price and post-exit data."""
        post_1h = trade.post_exit_1h_price
        post_4h = trade.post_exit_4h_price

        # Use the best available post-exit price as the potential exit
        best_price = trade.exit_price  # start with actual exit
        if post_1h is not None:
            if is_long:
                best_price = max(best_price, post_1h)
            else:
                best_price = min(best_price, post_1h)
        if post_4h is not None:
            if is_long:
                best_price = max(best_price, post_4h)
            else:
                best_price = min(best_price, post_4h)

        # If stop would have been triggered
        stop_hit = False
        for p in [post_1h, post_4h]:
            if p is not None:
                if is_long and p < stop_price:
                    stop_hit = True
                elif not is_long and p > stop_price:
                    stop_hit = True

        if stop_hit:
            exit_at = stop_price
        else:
            exit_at = best_price

        if is_long:
            return (exit_at - trade.entry_price) * trade.position_size
        else:
            return (trade.entry_price - exit_at) * trade.position_size
