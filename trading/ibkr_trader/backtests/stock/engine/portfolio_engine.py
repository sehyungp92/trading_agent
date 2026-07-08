"""Post-hoc portfolio merge engine for stock family backtests.

Replicates the stock family portfolio rules (from libs/oms/risk/portfolio_rules.py):
- Family-scoped 8R max aggregate same-direction exposure
- Symbol collision: half_size if sibling strategy holds same ticker
- Combined heat cap
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_trades,
    trade_outcomes_from_records,
)
from backtests.stock.config_portfolio import PortfolioBacktestConfig
from backtests.stock.models import Direction, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    """Result from running the portfolio backtest."""

    trades: list[TradeRecord]
    equity_curve: np.ndarray
    timestamps: np.ndarray
    alcb_trades: list[TradeRecord]
    iaric_trades: list[TradeRecord]
    blocked_trades: list[TradeRecord]
    decision_stream: list[dict] = field(default_factory=list)
    trade_outcomes: list[dict] = field(default_factory=list)


class StockPortfolioEngine:
    """Post-hoc merge of ALCB and IARIC trade lists.

    Takes independently-generated trade lists from both strategies
    and replays them chronologically, applying family-level portfolio rules.
    """

    def __init__(self, config: PortfolioBacktestConfig):
        self._config = config

    def _base_risk_dollars(self, equity: float) -> float:
        """1R in dollar terms at current equity."""
        return equity * self._config.base_risk_fraction

    def _to_r_units(self, dollar_risk: float, equity: float) -> float:
        """Convert dollar risk to R-units."""
        base = self._base_risk_dollars(equity)
        return dollar_risk / base if base > 0 else 0.0

    def run(
        self,
        alcb_trades: list[TradeRecord],
        iaric_trades: list[TradeRecord],
    ) -> PortfolioResult:
        """Merge and replay trades with family portfolio rules."""
        cfg = self._config

        # Combine and sort all trades by entry time
        all_trades: list[tuple[str, TradeRecord]] = []
        for t in alcb_trades:
            all_trades.append(("ALCB", t))
        for t in iaric_trades:
            all_trades.append(("IARIC", t))
        all_trades.sort(key=lambda x: x[1].entry_time)

        # Replay state
        equity = cfg.initial_equity
        active: dict[str, list[TradeRecord]] = {}  # symbol -> list of active trades
        strategy_active: dict[str, set[str]] = {"ALCB": set(), "IARIC": set()}
        accepted_trades: list[TradeRecord] = []
        blocked_trades: list[TradeRecord] = []
        equity_history: list[float] = [equity]
        ts_history: list[datetime] = []

        # Track directional exposure in R-units
        direction_r: dict[int, float] = {1: 0.0, -1: 0.0}  # LONG, SHORT

        for strategy, trade in all_trades:
            ts = trade.entry_time

            # Close any trades that have exited before this entry
            self._close_exited(active, strategy_active, direction_r, equity, ts)

            dir_val = 1 if trade.direction == Direction.LONG else -1
            trade_dollar_risk = trade.risk_per_share * trade.quantity
            trade_r_units = self._to_r_units(trade_dollar_risk, equity)

            # Rule 1: Family directional cap (max 8R same direction)
            current_dir_r = direction_r.get(dir_val, 0.0)
            if current_dir_r + trade_r_units > cfg.family_directional_cap_r:
                blocked_trades.append(trade)
                continue

            # Rule 2: Symbol collision (sibling strategy holds same ticker)
            sibling = "IARIC" if strategy == "ALCB" else "ALCB"
            adjusted_trade = trade
            if cfg.symbol_collision_half_size and trade.symbol in strategy_active.get(sibling, set()):
                new_qty = max(1, trade.quantity // 2)
                ratio = new_qty / trade.quantity if trade.quantity > 0 else 0.5
                adjusted_trade = TradeRecord(
                    strategy=trade.strategy,
                    symbol=trade.symbol,
                    direction=trade.direction,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    quantity=new_qty,
                    pnl=trade.pnl * ratio,
                    r_multiple=trade.r_multiple,
                    risk_per_share=trade.risk_per_share,
                    commission=trade.commission * ratio,
                    slippage=trade.slippage * ratio,
                    entry_type=trade.entry_type,
                    exit_reason=trade.exit_reason,
                    sector=trade.sector,
                    regime_tier=trade.regime_tier,
                    hold_bars=trade.hold_bars,
                    max_favorable=trade.max_favorable,
                    max_adverse=trade.max_adverse,
                    metadata={**trade.metadata, "collision_halved": True},
                )
                # Recompute R-units for adjusted trade
                trade_dollar_risk = adjusted_trade.risk_per_share * adjusted_trade.quantity
                trade_r_units = self._to_r_units(trade_dollar_risk, equity)
                logger.debug(
                    "Symbol collision: %s held by %s, halving %s entry",
                    trade.symbol, sibling, strategy,
                )

            # Rule 3: Combined heat cap
            total_active_risk = sum(
                t.risk_per_share * t.quantity
                for trades_list in active.values()
                for t in trades_list
            )
            proposed_risk = adjusted_trade.risk_per_share * adjusted_trade.quantity
            max_heat = self._base_risk_dollars(equity) * cfg.combined_heat_cap_r
            if total_active_risk + proposed_risk > max_heat:
                blocked_trades.append(trade)
                continue

            # Accept trade
            accepted_trades.append(adjusted_trade)
            active.setdefault(adjusted_trade.symbol, []).append(adjusted_trade)
            strategy_active[strategy].add(adjusted_trade.symbol)
            direction_r[dir_val] = current_dir_r + trade_r_units

            # Update equity from P&L
            equity += adjusted_trade.pnl_net
            equity_history.append(equity)
            ts_history.append(ts)

        # Final cleanup
        self._close_exited(active, strategy_active, direction_r, equity, datetime.max.replace(tzinfo=timezone.utc))

        logger.info(
            "Portfolio merge: %d accepted, %d blocked, final equity $%.2f",
            len(accepted_trades), len(blocked_trades), equity,
        )

        return PortfolioResult(
            trades=accepted_trades,
            equity_curve=np.array(equity_history),
            timestamps=np.array([
                np.datetime64(ts.replace(tzinfo=None)) if ts.year < 9999 else np.datetime64("NaT")
                for ts in ts_history
            ]),
            alcb_trades=[t for t in accepted_trades if t.strategy == "ALCB"],
            iaric_trades=[t for t in accepted_trades if t.strategy == "IARIC"],
            blocked_trades=blocked_trades,
            decision_stream=decision_stream_from_trades(accepted_trades, timeframe="portfolio"),
            trade_outcomes=trade_outcomes_from_records(accepted_trades),
        )

    def _close_exited(
        self,
        active: dict[str, list[TradeRecord]],
        strategy_active: dict[str, set[str]],
        direction_r: dict[int, float],
        equity: float,
        before: datetime,
    ) -> None:
        """Remove trades that have exited before *before*."""
        for sym in list(active):
            remaining = []
            for t in active[sym]:
                if t.exit_time <= before:
                    dir_val = 1 if t.direction == Direction.LONG else -1
                    r_units = self._to_r_units(t.risk_per_share * t.quantity, equity)
                    direction_r[dir_val] = max(0.0, direction_r.get(dir_val, 0.0) - r_units)
                    for strat in strategy_active:
                        strategy_active[strat].discard(sym)
                else:
                    remaining.append(t)
            if remaining:
                active[sym] = remaining
            else:
                del active[sym]
