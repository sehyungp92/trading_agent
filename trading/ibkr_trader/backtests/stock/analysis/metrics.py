"""Performance metrics computation.

Copied from momentum backtest (strategy-agnostic). Computes Sharpe, Sortino,
CAGR, drawdown, profit factor, expectancy, tail loss, and per-instrument stats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_dollar: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollar: float = 0.0
    avg_hold_hours: float = 0.0
    trades_per_month: float = 0.0
    total_commissions: float = 0.0
    tail_loss_pct: float = 0.0
    tail_loss_r: float = 0.0
    per_instrument_trades_per_month: dict[str, float] = field(default_factory=dict)


def compute_cagr(initial_equity: float, final_equity: float, years: float) -> float:
    if years <= 0 or initial_equity <= 0:
        return 0.0
    return (final_equity / initial_equity) ** (1.0 / years) - 1.0


def compute_sharpe(
    equity_curve: np.ndarray,
    periods_per_year: float = 252,
    risk_free_rate: float = 0.0,
) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def compute_sortino(
    equity_curve: np.ndarray,
    periods_per_year: float = 252,
    risk_free_rate: float = 0.0,
) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 1:
        return 0.0
    down_std = np.sqrt(np.mean(downside ** 2))
    if down_std == 0:
        return 0.0
    return float(np.mean(excess) / down_std * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: np.ndarray) -> tuple[float, float]:
    if len(equity_curve) < 2:
        return 0.0, 0.0
    peak = equity_curve[0]
    max_dd_dollar = 0.0
    max_dd_pct = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd_dollar = peak - val
        dd_pct = dd_dollar / peak if peak > 0 else 0.0
        if dd_dollar > max_dd_dollar:
            max_dd_dollar = dd_dollar
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd_pct, max_dd_dollar


def compute_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def compute_expectancy(trade_pnls: np.ndarray, trade_risks: np.ndarray) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    r_multiples = []
    for pnl, risk in zip(trade_pnls, trade_risks):
        if risk > 0:
            r_multiples.append(pnl / risk)
    if not r_multiples:
        return 0.0
    return float(np.mean(r_multiples))


def compute_tail_loss(trade_pnls: np.ndarray, percentile: float = 5.0) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    sorted_pnls = np.sort(trade_pnls)
    n_tail = max(1, int(len(sorted_pnls) * percentile / 100.0))
    return float(np.mean(sorted_pnls[:n_tail]))


def compute_tail_loss_r(
    trade_pnls: np.ndarray,
    trade_risks: np.ndarray,
    percentile: float = 5.0,
) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    r_multiples = np.where(trade_risks > 0, trade_pnls / trade_risks, 0.0)
    sorted_r = np.sort(r_multiples)
    n_tail = max(1, int(len(sorted_r) * percentile / 100.0))
    return float(np.mean(sorted_r[:n_tail]))


def compute_metrics(
    trade_pnls: np.ndarray,
    trade_risks: np.ndarray,
    trade_hold_hours: np.ndarray,
    trade_commissions: np.ndarray,
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
    trade_symbols: list[str] | None = None,
) -> PerformanceMetrics:
    n_trades = len(trade_pnls)
    if n_trades == 0:
        return PerformanceMetrics()

    wins = trade_pnls > 0
    losses = trade_pnls < 0
    gross_profit = float(np.sum(trade_pnls[wins]))
    gross_loss = float(np.sum(trade_pnls[losses]))
    net_profit = float(np.sum(trade_pnls))
    total_commissions = float(np.sum(trade_commissions))

    if len(timestamps) >= 2:
        delta = timestamps[-1] - timestamps[0]
        if hasattr(delta, 'astype'):
            span_s = float(delta / np.timedelta64(1, 's'))
        else:
            span_s = delta.total_seconds()
        years = span_s / (365.25 * 24 * 3600)
    else:
        years = 1.0

    final_equity = equity_curve[-1] if len(equity_curve) > 0 else initial_equity
    max_dd_pct, max_dd_dollar = compute_max_drawdown(equity_curve)
    cagr = compute_cagr(initial_equity, final_equity, years)
    calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

    months = years * 12
    trades_per_month = n_trades / months if months > 0 else 0.0

    per_inst_tpm: dict[str, float] = {}
    if trade_symbols and months > 0:
        from collections import Counter
        sym_counts = Counter(trade_symbols)
        per_inst_tpm = {sym: count / months for sym, count in sym_counts.items()}

    return PerformanceMetrics(
        total_trades=n_trades,
        winning_trades=int(np.sum(wins)),
        losing_trades=int(np.sum(losses)),
        win_rate=float(np.mean(wins)),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        profit_factor=compute_profit_factor(gross_profit, gross_loss),
        expectancy=compute_expectancy(trade_pnls, trade_risks),
        expectancy_dollar=net_profit / n_trades,
        cagr=cagr,
        sharpe=compute_sharpe(equity_curve),
        sortino=compute_sortino(equity_curve),
        calmar=calmar,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_dollar=max_dd_dollar,
        avg_hold_hours=float(np.mean(trade_hold_hours)),
        trades_per_month=trades_per_month,
        total_commissions=total_commissions,
        tail_loss_pct=compute_tail_loss(trade_pnls),
        tail_loss_r=compute_tail_loss_r(trade_pnls, trade_risks),
        per_instrument_trades_per_month=per_inst_tpm,
    )
