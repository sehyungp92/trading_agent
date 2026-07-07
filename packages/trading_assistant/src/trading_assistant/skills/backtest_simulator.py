# skills/backtest_simulator.py
"""Backtest simulator — simplified trade replay with parameter filtering.

Replays historical trades with parameter adjustments:
- Filters trades by signal_strength_min threshold
- Optionally includes missed opportunities (relaxed filters)
- Applies cost model
- Computes performance metrics (Sharpe, Sortino, Calmar, max DD, profit factor)
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.schemas.simulation_metrics import SimulationMetrics
from trading_assistant.skills.cost_model import CostModel


class BacktestSimulator:
    """Replays trades with parameter variations and computes metrics."""

    def __init__(self, cost_model: CostModel) -> None:
        self._cost_model = cost_model

    def simulate(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        params: dict,
        cost_multiplier: float = 1.0,
    ) -> SimulationMetrics:
        """Simulate a parameter set over historical trades and missed opportunities."""
        # 1. Filter trades by params
        accepted = self._filter_trades(trades, params)

        # 2. Include missed opportunities if filters are relaxed
        synthetic = self._include_missed(missed, params)

        # 3. Build PnL series with costs
        pnl_series: list[float] = []
        total_fees = 0.0
        total_slippage = 0.0
        regime_trades: dict[str, int] = defaultdict(int)
        regime_pnl: dict[str, float] = defaultdict(float)

        for t in accepted:
            costs = self._cost_model.compute_costs(
                entry_price=t.entry_price,
                position_size=t.position_size,
                regime=t.market_regime,
                cost_multiplier=cost_multiplier,
            )
            net = t.pnl - costs.total
            pnl_series.append(net)
            total_fees += costs.fees
            total_slippage += costs.slippage
            regime = t.market_regime or "unknown"
            regime_trades[regime] += 1
            regime_pnl[regime] += net

        for m in synthetic:
            outcome = m.outcome_24h or 0.0
            costs = self._cost_model.compute_costs(
                entry_price=m.hypothetical_entry,
                position_size=0.1,  # default position for missed opps
                cost_multiplier=cost_multiplier,
            )
            net = outcome - costs.total
            pnl_series.append(net)
            total_fees += costs.fees
            total_slippage += costs.slippage

        if not pnl_series:
            return SimulationMetrics()

        # 4. Compute metrics
        wins = [p for p in pnl_series if p > 0]
        losses = [p for p in pnl_series if p <= 0]
        gross_pnl = sum(t.pnl for t in accepted) + sum((m.outcome_24h or 0.0) for m in synthetic)
        net_pnl = sum(pnl_series)
        avg_notional = self._avg_notional(accepted)
        max_dd = self._max_drawdown(pnl_series, avg_notional)
        sharpe = self._sharpe(pnl_series)
        sortino = self._sortino(pnl_series)
        calmar = abs(net_pnl / max_dd) if max_dd > 0 else (float("inf") if net_pnl > 0 else 0.0)
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        pf = total_wins / total_losses if total_losses > 0 else (float("inf") if total_wins > 0 else 0.0)
        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = abs(statistics.mean(losses)) if losses else 0.0

        return SimulationMetrics(
            total_trades=len(pnl_series),
            win_count=len(wins),
            loss_count=len(losses),
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            profit_factor=pf,
            total_fees=total_fees,
            total_slippage=total_slippage,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades_by_regime=dict(regime_trades),
            pnl_by_regime=dict(regime_pnl),
        )

    def _filter_trades(self, trades: list[TradeEvent], params: dict) -> list[TradeEvent]:
        """Apply parameter-based filters to trades."""
        result = list(trades)
        min_strength = params.get("signal_strength_min")
        if min_strength is not None:
            result = [t for t in result if t.entry_signal_strength >= min_strength]
        return result

    def _include_missed(
        self, missed: list[MissedOpportunityEvent], params: dict
    ) -> list[MissedOpportunityEvent]:
        """Include missed opportunities whose blocking filter is being relaxed."""
        include_filters: list[str] = params.get("include_blocked_by", [])
        if not include_filters:
            return []
        return [m for m in missed if m.blocked_by in include_filters]

    @staticmethod
    def _avg_notional(trades: list[TradeEvent]) -> float:
        """Compute average notional exposure across trades."""
        if not trades:
            return 0.0
        return sum(t.entry_price * t.position_size for t in trades) / len(trades)

    @staticmethod
    def _max_drawdown(pnl_series: list[float], notional_base: float = 0.0) -> float:
        """Compute max drawdown as a fraction of notional capital (or peak equity as fallback).

        When notional_base > 0, drawdown is measured as peak-to-trough / notional_base,
        which is standard for PnL-based backtests without explicit initial capital.
        When notional_base == 0, falls back to peak-relative drawdown.
        """
        if not pnl_series:
            return 0.0
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_series:
            equity += pnl
            if equity > peak:
                peak = equity
            trough_depth = peak - equity
            if notional_base > 0:
                dd = trough_depth / notional_base
            elif peak > 0:
                dd = trough_depth / peak
            else:
                dd = 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _sharpe(pnl_series: list[float]) -> float:
        """Annualized Sharpe ratio (daily PnL, ~252 trading days)."""
        if len(pnl_series) < 2:
            return 0.0
        mean = statistics.mean(pnl_series)
        stdev = statistics.stdev(pnl_series)
        if stdev == 0:
            return 0.0
        return (mean / stdev) * math.sqrt(252)

    @staticmethod
    def _sortino(pnl_series: list[float]) -> float:
        """Annualized Sortino ratio (only downside deviation)."""
        if len(pnl_series) < 2:
            return 0.0
        mean = statistics.mean(pnl_series)
        downside = [p for p in pnl_series if p < 0]
        if not downside:
            return float("inf") if mean > 0 else 0.0
        downside_dev = math.sqrt(statistics.mean([d ** 2 for d in downside]))
        if downside_dev == 0:
            return 0.0
        return (mean / downside_dev) * math.sqrt(252)
