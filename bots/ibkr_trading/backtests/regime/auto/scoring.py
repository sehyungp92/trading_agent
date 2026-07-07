"""Immutable composite scoring for regime portfolio backtesting.

IMMUTABLE — do not change weights or normalizations after first optimization run.

Weights:
  - Sharpe (25%): risk-adjusted return quality, norm: sharpe / 1.5
  - Calmar (25%): return per unit max drawdown, norm: calmar / 5.0
  - Inverse DD (20%): direct drawdown penalty, norm: 1.0 - max_dd / 0.25
  - CAGR (15%): absolute wealth creation, norm: log(1+cagr) / log(1.15)
  - Sortino (15%): downside-specific risk management, norm: sortino / 2.5

Hard rejects: max_dd > 30%, sharpe < 0.15, cagr < 2%
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backtests.regime.analysis.metrics import PortfolioMetrics


@dataclass(frozen=True)
class CompositeScore:
    sharpe_component: float
    calmar_component: float
    inv_dd_component: float
    cagr_component: float
    sortino_component: float
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


# Weights — IMMUTABLE
W_SHARPE = 0.25
W_CALMAR = 0.25
W_INV_DD = 0.20
W_CAGR = 0.15
W_SORTINO = 0.15


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def composite_score(metrics: PortfolioMetrics) -> CompositeScore:
    """Compute the immutable composite score for a portfolio backtest.

    Args:
        metrics: Portfolio metrics from simulation.

    Returns:
        CompositeScore with component values and total.
    """
    # Hard rejects
    if metrics.max_drawdown_pct > 0.30:
        return CompositeScore(
            0, 0, 0, 0, 0, rejected=True,
            reject_reason=f"Max DD too high: {metrics.max_drawdown_pct:.1%} > 30%",
        )
    if metrics.sharpe < 0.15:
        return CompositeScore(
            0, 0, 0, 0, 0, rejected=True,
            reject_reason=f"Sharpe too low: {metrics.sharpe:.3f} < 0.15",
        )
    if metrics.cagr < 0.02:
        return CompositeScore(
            0, 0, 0, 0, 0, rejected=True,
            reject_reason=f"CAGR too low: {metrics.cagr:.1%} < 2%",
        )

    # Component scores (each clipped to [0, 1])
    sharpe_c = _clip01(metrics.sharpe / 1.5)
    calmar_c = _clip01(metrics.calmar / 5.0)
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.25)
    cagr_c = _clip01(math.log(1.0 + metrics.cagr) / math.log(1.15)) if metrics.cagr > 0 else 0.0
    sortino_c = _clip01(metrics.sortino / 2.5)

    total = (
        W_SHARPE * sharpe_c
        + W_CALMAR * calmar_c
        + W_INV_DD * inv_dd_c
        + W_CAGR * cagr_c
        + W_SORTINO * sortino_c
    )

    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=sortino_c,
        total=total,
    )
