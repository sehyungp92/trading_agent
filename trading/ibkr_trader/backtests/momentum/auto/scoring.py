"""Composite scoring for automated momentum backtesting.

Weights (identical to swing):
  - Net profit (35%): log(1+return)/log(4), equity curve total
  - Profit factor (30%): (PF-1)/2, win quality
  - Calmar ratio (20%): calmar/10, risk-adjusted return
  - Inverse drawdown (15%): 1-DD/0.30, low DD as reward

Hard rejects: <30 trades (20 vdubus, 25 nqdtc), >35% max DD, PF < 0.8
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backtests.momentum.analysis.metrics import PerformanceMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositeScore:
    calmar_component: float       # 0.20 weight, norm: calmar / 10.0
    pf_component: float           # 0.30 weight, norm: (PF - 1) / 2.0
    inv_dd_component: float       # 0.15 weight, norm: 1.0 - max_dd / 0.30
    net_profit_component: float   # 0.35 weight, norm: log(1+return) / log(4)
    total: float
    rejected: bool = False
    reject_reason: str = ""


# Weights
_W_CALMAR = 0.20
_W_PF = 0.30
_W_INV_DD = 0.15
_W_NET_PROFIT = 0.35

# Trade-count thresholds by strategy (momentum-tuned)
_LOW_TRADE_STRATEGIES = ("vdubus",)       # 15-min bars, lowest frequency
_MEDIUM_TRADE_STRATEGIES = ("nqdtc",)     # breakout-driven
_MIN_TRADES_DEFAULT = 30
_MIN_TRADES_MEDIUM = 25                   # NQDTC
_MIN_TRADES_LOW = 20                      # Vdubus


def composite_score(
    metrics: PerformanceMetrics,
    initial_equity: float = 100_000.0,
    strategy: str | None = None,
    equity_curve: np.ndarray | None = None,
) -> CompositeScore:
    """Compute the composite score.

    Args:
        metrics: Performance metrics from trade records.
        initial_equity: Starting equity for normalization.
        strategy: Strategy name for trade-count thresholds.
        equity_curve: If provided, net profit is derived from the equity curve
            (final - initial). Falls back to metrics.net_profit if not provided.
    """
    if strategy in _LOW_TRADE_STRATEGIES:
        min_trades = _MIN_TRADES_LOW
    elif strategy in _MEDIUM_TRADE_STRATEGIES:
        min_trades = _MIN_TRADES_MEDIUM
    else:
        min_trades = _MIN_TRADES_DEFAULT

    # Hard rejects
    if metrics.total_trades < min_trades:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Too few trades: {metrics.total_trades} < {min_trades}")
    if metrics.max_drawdown_pct > 0.35:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Max DD too high: {metrics.max_drawdown_pct:.1%} > 35%")
    if metrics.profit_factor < 0.8:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"PF too low: {metrics.profit_factor:.2f} < 0.80")

    # Net profit: prefer equity curve over trade-only PnL
    if equity_curve is not None and len(equity_curve) >= 2:
        net_profit = float(equity_curve[-1]) - initial_equity
    else:
        net_profit = metrics.net_profit

    # Component scores (each clipped to [0, 1])
    calmar_raw = min(max(metrics.calmar / 10.0, 0.0), 1.0)
    pf_raw = min(max((metrics.profit_factor - 1.0) / 2.0, 0.0), 1.0)
    inv_dd_raw = min(max(1.0 - metrics.max_drawdown_pct / 0.30, 0.0), 1.0)
    # Net profit: log scale so 300% (4x money) -> 1.0
    np_return = max(net_profit / initial_equity, 0.0)
    np_raw = min(math.log(1.0 + np_return) / math.log(4.0), 1.0)

    total = (
        _W_CALMAR * calmar_raw
        + _W_PF * pf_raw
        + _W_INV_DD * inv_dd_raw
        + _W_NET_PROFIT * np_raw
    )

    return CompositeScore(
        calmar_component=calmar_raw,
        pf_component=pf_raw,
        inv_dd_component=inv_dd_raw,
        net_profit_component=np_raw,
        total=total,
    )


# ---------------------------------------------------------------------------
# Metrics extraction helper
# ---------------------------------------------------------------------------

def extract_metrics(
    trades,
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
) -> PerformanceMetrics:
    """Standard metrics extraction from engine result.

    Accepts trade records from any active momentum engine (NQDTC, Vdubus,
    PortfolioTrade). Each record must have pnl_dollars, initial_stop,
    entry_price, qty (or equivalent), bars_held, commission, symbol.
    """
    if not trades:
        return PerformanceMetrics()

    pnls = np.array([_get_pnl(t) for t in trades])
    risks = np.array([_get_risk(t) for t in trades])
    hold_hours = np.array([_get_hold_hours(t) for t in trades])
    commissions = np.array([getattr(t, 'commission', 0.0) for t in trades])
    symbols = [getattr(t, 'symbol', '') for t in trades]

    return compute_metrics(
        pnls, risks, hold_hours, commissions,
        equity_curve, timestamps, initial_equity,
        trade_symbols=symbols,
    )


def _get_pnl(trade) -> float:
    """Extract PnL from various trade record types."""
    # PortfolioTrade uses adjusted_pnl (portfolio-sized)
    if hasattr(trade, 'adjusted_pnl'):
        return trade.adjusted_pnl
    if hasattr(trade, 'pnl_dollars'):
        return trade.pnl_dollars
    if hasattr(trade, 'pnl'):
        return trade.pnl
    return 0.0


def _get_risk(trade) -> float:
    """Extract risk (dollar value of 1R) from various trade record types."""
    if hasattr(trade, 'initial_stop') and hasattr(trade, 'entry_price'):
        r = abs(trade.entry_price - trade.initial_stop)
        # PortfolioTrade uses portfolio_qty; strategy trades use qty
        qty = getattr(trade, 'portfolio_qty', None) or getattr(trade, 'qty', 1)
        # For futures, risk = price distance * point_value * qty
        # MNQ=2.0, NQ=20.0 — try trade attr first, then default to MNQ
        point_value = getattr(trade, 'point_value', None)
        if point_value is None:
            point_value = 2.0 if abs(qty) <= 10 else 20.0
        return r * point_value * abs(qty) if r > 0 else 1.0
    return 1.0


def _get_hold_hours(trade) -> float:
    """Extract hold duration in hours from various trade record types."""
    # Prefer actual time delta over bar counts
    if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
        if trade.entry_time and trade.exit_time:
            delta = trade.exit_time - trade.entry_time
            return delta.total_seconds() / 3600.0
    if hasattr(trade, 'bars_held'):
        # bars_held is a bar count; approximate using bar interval
        bar_minutes = getattr(trade, 'bar_interval_minutes', 5)
        return float(trade.bars_held) * bar_minutes / 60.0
    return 0.0
