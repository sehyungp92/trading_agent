"""Portfolio-level metrics for regime backtesting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioMetrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    avg_annual_turnover: float
    n_rebalances: int


def compute_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    turnover_series: pd.Series,
    n_rebalances: int,
    ann_factor: float = 252.0,
) -> PortfolioMetrics:
    """Compute portfolio metrics from equity curve and daily returns.

    Args:
        equity_curve: Daily equity values (initial_equity * cumprod(1+r)).
        daily_returns: Daily portfolio returns series.
        turnover_series: Per-rebalance turnover values.
        n_rebalances: Number of rebalance dates used.
        ann_factor: Trading days per year.
    """
    if len(equity_curve) < 2:
        return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Total return
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

    # CAGR
    n_days = len(equity_curve)
    n_years = n_days / ann_factor
    if n_years > 0 and equity_curve.iloc[-1] > 0 and equity_curve.iloc[0] > 0:
        cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / n_years) - 1.0)
    else:
        cagr = 0.0

    # Sharpe
    mean_ret = float(daily_returns.mean())
    std_ret = float(daily_returns.std(ddof=1))
    if std_ret > 1e-12:
        sharpe = float(mean_ret / std_ret * np.sqrt(ann_factor))
    else:
        sharpe = 0.0

    # Sortino (downside deviation)
    downside = daily_returns.clip(upper=0.0)
    downside_std = float(np.sqrt((downside ** 2).mean()))
    if downside_std > 1e-12:
        sortino = float(mean_ret / downside_std * np.sqrt(ann_factor))
    else:
        sortino = 0.0

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = float(-drawdown.min())

    # Max drawdown duration (trading days)
    is_in_dd = drawdown < 0
    if is_in_dd.any():
        dd_groups = (~is_in_dd).cumsum()
        dd_lengths = is_in_dd.groupby(dd_groups).sum()
        max_drawdown_duration = int(dd_lengths.max())
    else:
        max_drawdown_duration = 0

    # Calmar
    if max_drawdown_pct > 1e-12:
        calmar = float(cagr / max_drawdown_pct)
    else:
        calmar = float(cagr * 100.0) if cagr > 0 else 0.0

    # Average annual turnover
    if n_years > 0:
        avg_annual_turnover = float(turnover_series.sum() / n_years)
    else:
        avg_annual_turnover = 0.0

    return PortfolioMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration=max_drawdown_duration,
        avg_annual_turnover=avg_annual_turnover,
        n_rebalances=n_rebalances,
    )
