"""Portfolio simulation: apply weekly regime allocations to daily returns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtests.regime.analysis.metrics import PortfolioMetrics, compute_metrics
from backtests.regime.config import RegimeBacktestConfig


@dataclass
class PortfolioResult:
    metrics: PortfolioMetrics
    equity_curve: pd.Series
    daily_returns: pd.Series
    weight_history: pd.DataFrame
    turnover_series: pd.Series


def simulate_portfolio(
    signals_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    sim_cfg: RegimeBacktestConfig,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> PortfolioResult:
    """Simulate portfolio performance from weekly regime signals.

    Args:
        signals_df: Output of run_signal_engine() — weekly index, pi_* columns.
        daily_returns: strat_ret_df — daily log returns for all 6 sleeves.
        sim_cfg: Backtest config with cost and equity params.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        PortfolioResult with metrics, equity curve, and weight history.
    """
    sleeves = [c for c in daily_returns.columns if f"pi_{c}" in signals_df.columns]
    pi_cols = [f"pi_{c}" for c in sleeves]

    # Extract levered allocations and forward-fill to daily
    pi_weekly = signals_df[pi_cols].copy()
    pi_weekly.columns = sleeves

    # Create daily allocation DataFrame by forward-filling weekly signals
    daily_idx = daily_returns.index
    if start_date is not None:
        daily_idx = daily_idx[daily_idx >= start_date]
    if end_date is not None:
        daily_idx = daily_idx[daily_idx <= end_date]

    # Reindex to daily and forward-fill
    pi_daily = pi_weekly.reindex(daily_idx, method="ffill")

    # Pre-signal period: 100% CASH before first signal
    first_signal_date = pi_weekly.index.min()
    pre_signal = pi_daily.index < first_signal_date
    pi_daily.loc[pre_signal] = 0.0
    if "CASH" in sleeves:
        pi_daily.loc[pre_signal, "CASH"] = 1.0

    # Fill any remaining NaN (between start_date and first signal)
    pi_daily = pi_daily.fillna(0.0)

    # Compute turnover on rebalance days
    rebalance_dates = pi_weekly.index.intersection(daily_idx)
    turnover_values = []
    prev_weights = None
    for dt in rebalance_dates:
        curr_weights = pi_daily.loc[dt].values
        if prev_weights is not None:
            turnover = float(np.sum(np.abs(curr_weights - prev_weights)))
        else:
            turnover = float(np.sum(np.abs(curr_weights)))
        turnover_values.append(turnover)
        prev_weights = curr_weights.copy()
    turnover_series = pd.Series(turnover_values, index=rebalance_dates, name="turnover")

    # Daily portfolio returns — convert log returns to arithmetic before weighting
    ret_aligned = daily_returns[sleeves].reindex(daily_idx).fillna(0.0)
    arith_ret = np.expm1(ret_aligned)  # exp(log_ret) - 1
    cost_bps = sim_cfg.rebalance_cost_bps / 10_000.0

    daily_port_ret = (pi_daily.values * arith_ret.values).sum(axis=1)
    daily_port_ret = pd.Series(daily_port_ret, index=daily_idx, name="portfolio_return")

    # Apply turnover cost on rebalance days only
    for dt in rebalance_dates:
        if dt in daily_port_ret.index and dt in turnover_series.index:
            daily_port_ret.loc[dt] -= turnover_series.loc[dt] * cost_bps

    # Equity curve
    equity = sim_cfg.initial_equity * (1.0 + daily_port_ret).cumprod()
    equity.name = "equity"

    # Compute metrics
    metrics = compute_metrics(
        equity_curve=equity,
        daily_returns=daily_port_ret,
        turnover_series=turnover_series,
        n_rebalances=len(rebalance_dates),
    )

    return PortfolioResult(
        metrics=metrics,
        equity_curve=equity,
        daily_returns=daily_port_ret,
        weight_history=pi_daily,
        turnover_series=turnover_series,
    )


def simulate_benchmark_60_40(
    daily_returns: pd.DataFrame,
    sim_cfg: RegimeBacktestConfig,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> PortfolioResult:
    """Simulate a 60/40 SPY/TLT benchmark, monthly rebalanced."""
    daily_idx = daily_returns.index
    if start_date is not None:
        daily_idx = daily_idx[daily_idx >= start_date]
    if end_date is not None:
        daily_idx = daily_idx[daily_idx <= end_date]

    sleeves = daily_returns.columns.tolist()
    weights = pd.DataFrame(0.0, index=daily_idx, columns=sleeves)
    weights["SPY"] = 0.60
    weights["TLT"] = 0.40

    # Monthly rebalance dates
    monthly = pd.date_range(daily_idx.min(), daily_idx.max(), freq="MS")
    rebalance_dates = monthly.intersection(daily_idx)

    turnover_values = [0.0] * len(rebalance_dates)
    turnover_series = pd.Series(turnover_values, index=rebalance_dates, name="turnover")

    ret_aligned = daily_returns.reindex(daily_idx).fillna(0.0)
    arith_ret = np.expm1(ret_aligned[sleeves])  # exp(log_ret) - 1
    daily_port_ret = (weights.values * arith_ret.values).sum(axis=1)
    daily_port_ret = pd.Series(daily_port_ret, index=daily_idx, name="benchmark_return")

    equity = sim_cfg.initial_equity * (1.0 + daily_port_ret).cumprod()
    equity.name = "equity"

    metrics = compute_metrics(
        equity_curve=equity,
        daily_returns=daily_port_ret,
        turnover_series=turnover_series,
        n_rebalances=len(rebalance_dates),
    )

    return PortfolioResult(
        metrics=metrics,
        equity_curve=equity,
        daily_returns=daily_port_ret,
        weight_history=weights,
        turnover_series=turnover_series,
    )
