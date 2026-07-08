"""Composite objective function for optimization."""
from __future__ import annotations

from backtests.momentum.analysis.metrics import PerformanceMetrics


def composite_objective(
    metrics: PerformanceMetrics,
    min_trades_per_month: float = 2.0,
    min_total_trades: int = 200,
    max_drawdown_cap: float = 0.35,
) -> float:
    """Compute a composite score for optimization.

    Objective (maximize):
        + 0.45 * norm(CAGR)
        + 0.25 * norm(Sharpe)
        + 0.20 * norm(PF)
        - 0.30 * norm(MaxDD)
        - 0.10 * norm(TailLoss)
        - frequency_penalty

    Returns -1.0 if total_trades < min_total_trades or
    max_drawdown_pct exceeds *max_drawdown_cap* (hard reject).
    """
    if metrics.total_trades < min_total_trades:
        return -1.0

    if metrics.max_drawdown_pct > max_drawdown_cap:
        return -1.0

    # Normalize to roughly [0, 1] ranges
    cagr_norm = _clip(metrics.cagr / 0.50, 0, 1)       # 50% CAGR = perfect
    sharpe_norm = _clip(metrics.sharpe / 3.0, 0, 1)     # Sharpe 3 = perfect
    pf_norm = _clip((metrics.profit_factor - 1) / 2, 0, 1)  # PF 3 = perfect
    dd_norm = _clip(metrics.max_drawdown_pct / 0.30, 0, 1)  # 30% DD = worst
    # Use R-based tail loss for scale-independence (fallback to dollar-based)
    tail_r = metrics.tail_loss_r if metrics.tail_loss_r != 0 else 0.0
    tail_norm = _clip(abs(tail_r) / 3.0, 0, 1)  # -3R avg tail = worst

    score = (
        0.45 * cagr_norm
        + 0.25 * sharpe_norm
        + 0.20 * pf_norm
        - 0.30 * dd_norm
        - 0.10 * tail_norm
    )

    # Frequency penalty: steep below target trades/month
    if metrics.trades_per_month < min_trades_per_month:
        freq_ratio = metrics.trades_per_month / min_trades_per_month
        score *= freq_ratio ** 2  # Quadratic penalty

    return score


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
