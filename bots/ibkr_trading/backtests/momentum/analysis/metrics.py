"""Performance metrics computation."""
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
    expectancy: float = 0.0         # avg R per trade
    expectancy_dollar: float = 0.0  # avg $ per trade
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0             # CAGR / MaxDD
    max_drawdown_pct: float = 0.0
    max_drawdown_dollar: float = 0.0
    avg_hold_hours: float = 0.0
    trades_per_month: float = 0.0
    total_commissions: float = 0.0
    tail_loss_pct: float = 0.0      # worst 5% of trades avg loss (dollars)
    tail_loss_r: float = 0.0        # worst 5% of trades avg loss (R-multiples)
    per_instrument_trades_per_month: dict[str, float] = field(default_factory=dict)


@dataclass
class BuyAndHoldMetrics:
    """Buy-and-hold benchmark statistics for 1 share."""

    symbol: str = ""
    start_price: float = 0.0
    end_price: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    max_drawdown_pct: float = 0.0


def compute_buy_and_hold(
    symbol: str,
    daily_closes: np.ndarray,
    years: float,
) -> BuyAndHoldMetrics:
    """Compute buy-and-hold metrics from daily close prices.

    Assumes buying 1 share at the first close and holding to the last close.
    """
    closes = daily_closes[np.isfinite(daily_closes)]
    if len(closes) < 2:
        return BuyAndHoldMetrics(symbol=symbol)

    start_price = float(closes[0])
    end_price = float(closes[-1])
    total_return_pct = (end_price / start_price - 1.0) * 100.0

    cagr = compute_cagr(start_price, end_price, years) if years > 0 else 0.0

    # Max drawdown from the close series
    peak = closes[0]
    max_dd = 0.0
    for c in closes:
        if c > peak:
            peak = c
        dd = (peak - c) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return BuyAndHoldMetrics(
        symbol=symbol,
        start_price=start_price,
        end_price=end_price,
        total_return_pct=total_return_pct,
        cagr=cagr,
        max_drawdown_pct=max_dd,
    )


def compute_cagr(
    initial_equity: float,
    final_equity: float,
    years: float,
) -> float:
    """Compound annual growth rate."""
    if years <= 0 or initial_equity <= 0:
        return 0.0
    return (final_equity / initial_equity) ** (1.0 / years) - 1.0


def compute_sharpe(
    equity_curve: np.ndarray,
    periods_per_year: float = 252 * 7,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio from an hourly equity curve."""
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
    periods_per_year: float = 252 * 7,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
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
    """Max drawdown as (pct, dollar).

    Returns (max_dd_pct, max_dd_dollar).
    """
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
    """Profit factor = gross_profit / |gross_loss|."""
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def compute_expectancy(
    trade_pnls: np.ndarray,
    trade_risks: np.ndarray,
) -> float:
    """Average R-multiple across all trades.

    trade_risks is the per-trade initial risk in dollars.
    """
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
    """Average loss of the worst percentile of trades (in dollars)."""
    if len(trade_pnls) == 0:
        return 0.0
    sorted_pnls = np.sort(trade_pnls)
    n_tail = max(1, int(len(sorted_pnls) * percentile / 100.0))
    tail = sorted_pnls[:n_tail]
    return float(np.mean(tail))


def compute_tail_loss_r(
    trade_pnls: np.ndarray,
    trade_risks: np.ndarray,
    percentile: float = 5.0,
) -> float:
    """Average R-multiple of the worst percentile of trades."""
    if len(trade_pnls) == 0:
        return 0.0
    r_multiples = np.where(trade_risks > 0, trade_pnls / trade_risks, 0.0)
    sorted_r = np.sort(r_multiples)
    n_tail = max(1, int(len(sorted_r) * percentile / 100.0))
    tail = sorted_r[:n_tail]
    return float(np.mean(tail))


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
    """Compute all performance metrics from trade and equity data.

    Args:
        trade_symbols: Optional per-trade symbol list for per-instrument stats.
    """
    n_trades = len(trade_pnls)
    if n_trades == 0:
        return PerformanceMetrics()

    # Use fee-net PnL for PF, expectancy, and tail metrics
    fee_net_pnls = trade_pnls - trade_commissions

    wins = fee_net_pnls > 0
    losses = fee_net_pnls < 0

    gross_profit = float(np.sum(fee_net_pnls[wins]))
    gross_loss = float(np.sum(fee_net_pnls[losses]))
    net_profit = float(np.sum(fee_net_pnls))
    total_commissions = float(np.sum(trade_commissions))

    # Time span in years
    if len(timestamps) >= 2:
        delta = timestamps[-1] - timestamps[0]
        # Handle numpy timedelta64, Python timedelta, or float (unix seconds)
        if isinstance(delta, (int, float)) or (hasattr(delta, 'dtype') and np.issubdtype(type(delta), np.floating)):
            span_s = float(delta)
        elif hasattr(delta, 'astype'):
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

    # Trades per month
    months = years * 12
    trades_per_month = n_trades / months if months > 0 else 0.0

    # Per-instrument trades/month
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
        expectancy=compute_expectancy(fee_net_pnls, trade_risks),
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
        tail_loss_pct=compute_tail_loss(fee_net_pnls),
        tail_loss_r=compute_tail_loss_r(fee_net_pnls, trade_risks),
        per_instrument_trades_per_month=per_inst_tpm,
    )
