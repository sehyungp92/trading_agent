"""Composite scoring for automated swing backtesting.

Weights:
  - Net profit (35%): log(1+return)/log(4), equity curve total by default
  - Profit factor (30%): (PF-1)/2, win quality
  - Calmar ratio (20%): calmar/10, risk-adjusted return
  - Inverse drawdown (15%): 1-DD/0.30, low DD as reward

Net profit is derived from the equity curve (final - initial) by default,
so overlay PnL is properly credited. Callers can provide an explicit
net_profit_override when a different return basis is required. Callers can
also tighten the drawdown score and add an explicit drawdown penalty for
portfolio-level risk profiles.

Hard rejects: <30 trades, >35% max DD, PF < 0.8
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backtests.swing.analysis.metrics import PerformanceMetrics, compute_metrics


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

_MIN_TRADES_DEFAULT = 30


def composite_score(
    metrics: PerformanceMetrics,
    initial_equity: float = 100_000.0,
    strategy: str | None = None,
    equity_curve: np.ndarray | None = None,
    net_profit_override: float | None = None,
    max_drawdown_hard_pct: float = 0.35,
    drawdown_score_scale_pct: float = 0.30,
    drawdown_penalty_start_pct: float | None = None,
    drawdown_penalty_full_pct: float | None = None,
    drawdown_penalty_weight: float = 0.0,
) -> CompositeScore:
    """Compute the composite score.

    Args:
        metrics: Performance metrics from trade records.
        initial_equity: Starting equity for normalization.
        strategy: Strategy name for trade-count thresholds.
        equity_curve: If provided, net profit is derived from the equity curve
            (final - initial) so overlay PnL is included. Falls back to
            metrics.net_profit if not provided.
        net_profit_override: Optional caller-supplied PnL for the net-profit
            component, used when the optimizer should score a comparable
            non-compounded return basis.
        max_drawdown_hard_pct: Hard reject threshold for max drawdown.
        drawdown_score_scale_pct: Drawdown where inverse-DD component reaches 0.
        drawdown_penalty_start_pct: Optional drawdown where extra penalty starts.
        drawdown_penalty_full_pct: Optional drawdown where extra penalty reaches full weight.
        drawdown_penalty_weight: Amount subtracted from total score at full penalty.
    """
    min_trades = _MIN_TRADES_DEFAULT

    # Hard rejects
    if metrics.total_trades < min_trades:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"Too few trades: {metrics.total_trades} < {min_trades}")
    if metrics.max_drawdown_pct > max_drawdown_hard_pct:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=(
                                  f"Max DD too high: {metrics.max_drawdown_pct:.1%} "
                                  f"> {max_drawdown_hard_pct:.1%}"
                              ))
    if metrics.profit_factor < 0.8:
        return CompositeScore(0, 0, 0, 0, 0, rejected=True,
                              reject_reason=f"PF too low: {metrics.profit_factor:.2f} < 0.80")

    # Net profit: prefer explicit override, then equity curve, then trade-only PnL.
    if net_profit_override is not None:
        net_profit = float(net_profit_override)
    elif equity_curve is not None and len(equity_curve) >= 2:
        net_profit = float(equity_curve[-1]) - initial_equity
    else:
        net_profit = metrics.net_profit

    # Component scores (each clipped to [0, 1])
    # Calmar: /10.0 so Calmar=10 → 1.0 (avoids saturation at Calmar ~5+)
    calmar_raw = min(max(metrics.calmar / 10.0, 0.0), 1.0)
    pf_raw = min(max((metrics.profit_factor - 1.0) / 2.0, 0.0), 1.0)
    dd_scale = max(float(drawdown_score_scale_pct), 1e-9)
    inv_dd_raw = min(max(1.0 - metrics.max_drawdown_pct / dd_scale, 0.0), 1.0)
    # Net profit: log scale so 300% (4× money) → 1.0, diminishing marginal value
    np_return = max(net_profit / initial_equity, 0.0)
    np_raw = min(math.log(1.0 + np_return) / math.log(4.0), 1.0)

    total = (
        _W_CALMAR * calmar_raw
        + _W_PF * pf_raw
        + _W_INV_DD * inv_dd_raw
        + _W_NET_PROFIT * np_raw
    )
    if (
        drawdown_penalty_start_pct is not None
        and drawdown_penalty_weight > 0.0
        and metrics.max_drawdown_pct > drawdown_penalty_start_pct
    ):
        penalty_full = (
            drawdown_penalty_full_pct
            if drawdown_penalty_full_pct is not None
            else max_drawdown_hard_pct
        )
        penalty_range = max(float(penalty_full) - float(drawdown_penalty_start_pct), 1e-9)
        penalty_ratio = min(
            max((metrics.max_drawdown_pct - float(drawdown_penalty_start_pct)) / penalty_range, 0.0),
            1.0,
        )
        total = max(total - float(drawdown_penalty_weight) * penalty_ratio, 0.0)

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

    Accepts trade records from any active swing engine. Each record must have pnl_dollars,
    initial_stop, entry_price, qty (or equivalent), bars_held, commission, symbol.
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
    if hasattr(trade, 'pnl_dollars'):
        return trade.pnl_dollars
    if hasattr(trade, 'pnl'):
        return trade.pnl
    return 0.0


def _get_risk(trade) -> float:
    """Extract risk (dollar value of 1R) from various trade record types."""
    if hasattr(trade, 'initial_stop') and hasattr(trade, 'entry_price'):
        r = abs(trade.entry_price - trade.initial_stop)
        qty = getattr(trade, 'qty', 1)
        return r * abs(qty) if r > 0 else 1.0
    return 1.0


def _get_hold_hours(trade) -> float:
    """Extract hold duration in hours from various trade record types."""
    if hasattr(trade, 'bars_held'):
        return float(trade.bars_held)
    if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
        if trade.entry_time and trade.exit_time:
            delta = trade.exit_time - trade.entry_time
            return delta.total_seconds() / 3600.0
    return 0.0
