"""Performance metrics computation from broker state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crypto_trader.core.runtime_types import TradeOutcome


@dataclass
class PerformanceMetrics:
    # Core
    net_profit: float = 0.0
    net_return_pct: float = 0.0
    realized_pnl_net: float = 0.0
    terminal_mark_pnl_net: float = 0.0
    terminal_mark_count: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    # R-multiple
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0
    expectancy_r: float = 0.0
    # Risk-adjusted
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    # Execution
    avg_bars_held: float = 0.0
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    exit_efficiency: float = 0.0
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    # Edge & concentration
    edge_ratio: float = 0.0
    recovery_factor: float = 0.0
    profit_concentration: float = 0.0
    payoff_ratio: float = 0.0
    # Breakdown
    a_setup_win_rate: float = 0.0
    b_setup_win_rate: float = 0.0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    total_fees: float = 0.0
    funding_cost_total: float = 0.0
    per_asset: dict[str, dict] = field(default_factory=dict)
    per_session: dict[str, dict] = field(default_factory=dict)
    # Detailed breakdowns
    per_confirmation: dict[str, dict] = field(default_factory=dict)
    per_confluence_count: dict[int, dict] = field(default_factory=dict)
    per_exit_reason: dict[str, dict] = field(default_factory=dict)
    r_distribution: dict[str, int] = field(default_factory=dict)
    weekly_returns: list[dict] = field(default_factory=list)


def _preferred_equity_history(broker: object) -> list[tuple[object, float]]:
    liquidation_history = getattr(broker, "_liquidation_equity_history", [])
    if liquidation_history:
        return liquidation_history
    return getattr(broker, "_equity_history", [])


def _trade_reporting_r(trade: object) -> float | None:
    economic_r = getattr(trade, "economic_r_multiple", None)
    if economic_r is not None:
        return float(economic_r)
    realized_r = getattr(trade, "realized_r_multiple", None)
    if realized_r is not None:
        return float(realized_r)
    geometric_r = getattr(trade, "r_multiple", None)
    if geometric_r is None:
        return None
    return float(geometric_r)


def _trade_outcome(trade: object) -> TradeOutcome:
    return TradeOutcome.from_trade(trade)


def _trade_net_pnl(trade: object) -> float:
    try:
        return _trade_outcome(trade).realized_pnl_net
    except AttributeError:
        return float(getattr(trade, "net_pnl", 0.0))


def _trade_funding_paid(trade: object) -> float:
    try:
        return _trade_outcome(trade).funding_paid
    except AttributeError:
        return float(getattr(trade, "funding_paid", 0.0) or 0.0)


def _trade_total_fees(trade: object) -> float:
    try:
        return _trade_outcome(trade).total_fees
    except AttributeError:
        return float(getattr(trade, "commission", 0.0) or 0.0)


def _trade_reporting_rs(trades: list) -> list[float]:
    rs: list[float] = []
    for trade in trades:
        reporting_r = _trade_reporting_r(trade)
        if reporting_r is not None:
            rs.append(reporting_r)
    return rs


def _compute_exit_efficiency(trades: list) -> float:
    efficiencies = []
    for trade in trades:
        reporting_r = _trade_reporting_r(trade)
        if trade.mfe_r and trade.mfe_r > 0 and reporting_r is not None and reporting_r > 0:
            efficiencies.append(reporting_r / trade.mfe_r)
    return float(np.mean(efficiencies)) if efficiencies else 0.0


def compute_metrics(broker: object) -> PerformanceMetrics:
    """Compute performance metrics from a broker-like object."""
    from crypto_trader.core.models import SetupGrade, Side

    trades = getattr(broker, "_closed_trades", [])
    terminal_marks = getattr(broker, "_terminal_marks", [])
    equity_history = _preferred_equity_history(broker)
    initial_equity = (
        getattr(broker, "initial_equity", None)
        or getattr(broker, "_initial_equity", 10_000.0)
    )

    m = PerformanceMetrics()
    m.total_trades = len(trades)
    m.realized_pnl_net = float(sum(_trade_net_pnl(t) for t in trades))
    m.terminal_mark_count = len(terminal_marks)
    m.terminal_mark_pnl_net = float(sum(mark.unrealized_pnl_net for mark in terminal_marks))
    m.total_fees = float(sum(_trade_total_fees(t) for t in trades))

    if equity_history:
        final_equity = equity_history[-1][1]
        m.net_profit = float(final_equity - initial_equity)
        m.net_return_pct = m.net_profit / initial_equity * 100 if initial_equity > 0 else 0.0
    else:
        m.net_profit = m.realized_pnl_net + m.terminal_mark_pnl_net
        m.net_return_pct = m.net_profit / initial_equity * 100 if initial_equity > 0 else 0.0

    if equity_history:
        equities = [eq for _, eq in equity_history]
        peak = equities[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0
        for i, eq in enumerate(equities):
            if eq > peak:
                peak = eq
                duration = i - current_dd_start
                max_dd_duration = max(max_dd_duration, duration)
                current_dd_start = i
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak)
        m.max_drawdown_pct = max_dd * 100
        m.max_drawdown_duration = max_dd_duration

    if equity_history and len(equity_history) > 1:
        daily_equity: dict = {}
        for ts, eq in equity_history:
            daily_equity[ts.date()] = eq
        sorted_days = sorted(daily_equity.keys())
        if len(sorted_days) >= 2:
            daily_rets = np.array(
                [
                    (daily_equity[sorted_days[i]] - daily_equity[sorted_days[i - 1]])
                    / daily_equity[sorted_days[i - 1]]
                    for i in range(1, len(sorted_days))
                ]
            )
            mean_r = np.mean(daily_rets)
            std_r = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else 0.0
            m.sharpe_ratio = float(mean_r / std_r * np.sqrt(365)) if std_r > 0 else 0.0

            downside = daily_rets[daily_rets < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
            m.sortino_ratio = float(mean_r / downside_std * np.sqrt(365)) if downside_std > 0 else 0.0

    if m.max_drawdown_pct > 0:
        m.calmar_ratio = m.net_return_pct / m.max_drawdown_pct
        m.recovery_factor = abs(m.net_return_pct) / m.max_drawdown_pct

    if m.total_trades == 0:
        return m

    winners = [t for t in trades if _trade_net_pnl(t) > 0]
    losers = [t for t in trades if _trade_net_pnl(t) <= 0]
    m.win_rate = len(winners) / m.total_trades * 100

    winner_rs = _trade_reporting_rs(winners)
    loser_rs = _trade_reporting_rs(losers)
    m.avg_winner_r = float(np.mean(winner_rs)) if winner_rs else 0.0
    m.avg_loser_r = float(np.mean(loser_rs)) if loser_rs else 0.0

    all_rs = _trade_reporting_rs(trades)
    m.expectancy_r = float(np.mean(all_rs)) if all_rs else 0.0

    gross_profit = sum(_trade_net_pnl(t) for t in winners) if winners else 0.0
    gross_loss = abs(sum(_trade_net_pnl(t) for t in losers)) if losers else 0.0
    m.profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf") if gross_profit > 0
        else 0.0
    )

    bars_held = [t.bars_held for t in trades]
    m.avg_bars_held = float(np.mean(bars_held)) if bars_held else 0.0

    maes = [t.mae_r for t in trades if t.mae_r is not None]
    mfes = [t.mfe_r for t in trades if t.mfe_r is not None]
    m.avg_mae_r = float(np.mean(maes)) if maes else 0.0
    m.avg_mfe_r = float(np.mean(mfes)) if mfes else 0.0

    m.exit_efficiency = _compute_exit_efficiency(trades)

    if m.avg_mae_r != 0:
        m.edge_ratio = float(m.avg_mfe_r / abs(m.avg_mae_r))

    avg_winner_pnl = float(np.mean([_trade_net_pnl(t) for t in winners])) if winners else 0.0
    avg_loser_pnl = abs(float(np.mean([_trade_net_pnl(t) for t in losers]))) if losers else 0.0
    m.payoff_ratio = float(avg_winner_pnl / avg_loser_pnl) if avg_loser_pnl > 0 else 0.0

    if winners:
        sorted_winners = sorted(winners, key=_trade_net_pnl, reverse=True)
        top_n = max(1, len(sorted_winners) // 5)
        top_profit = sum(_trade_net_pnl(trade) for trade in sorted_winners[:top_n])
        total_profit = sum(_trade_net_pnl(trade) for trade in sorted_winners)
        m.profit_concentration = float(top_profit / total_profit * 100) if total_profit > 0 else 0.0

    m.max_consecutive_wins, m.max_consecutive_losses = _compute_streaks(trades)
    m.r_distribution = _compute_r_distribution(all_rs)

    m.per_confirmation = _compute_group_breakdown(
        trades, key_fn=lambda trade: trade.confirmation_type or "unknown"
    )
    m.per_confluence_count = _compute_group_breakdown(
        trades,
        key_fn=lambda trade: len(trade.confluences_used) if trade.confluences_used else 0,
    )
    m.per_exit_reason = _compute_group_breakdown(
        trades, key_fn=lambda trade: trade.exit_reason or "unknown"
    )
    m.weekly_returns = _compute_weekly_returns(trades)

    a_trades = [t for t in trades if t.setup_grade == SetupGrade.A]
    b_trades = [t for t in trades if t.setup_grade == SetupGrade.B]
    m.a_setup_win_rate = (
        sum(1 for trade in a_trades if _trade_net_pnl(trade) > 0) / len(a_trades) * 100
    ) if a_trades else 0.0
    m.b_setup_win_rate = (
        sum(1 for trade in b_trades if _trade_net_pnl(trade) > 0) / len(b_trades) * 100
    ) if b_trades else 0.0

    longs = [t for t in trades if t.direction == Side.LONG]
    shorts = [t for t in trades if t.direction == Side.SHORT]
    m.long_win_rate = (
        sum(1 for trade in longs if _trade_net_pnl(trade) > 0) / len(longs) * 100
    ) if longs else 0.0
    m.short_win_rate = (
        sum(1 for trade in shorts if _trade_net_pnl(trade) > 0) / len(shorts) * 100
    ) if shorts else 0.0

    m.funding_cost_total = sum(_trade_funding_paid(t) for t in trades)

    symbols = set(t.symbol for t in trades)
    for symbol in symbols:
        sym_trades = [t for t in trades if t.symbol == symbol]
        sym_winners = [t for t in sym_trades if _trade_net_pnl(t) > 0]
        m.per_asset[symbol] = {
            "trades": len(sym_trades),
            "win_rate": len(sym_winners) / len(sym_trades) * 100 if sym_trades else 0.0,
            "net_profit": sum(_trade_net_pnl(t) for t in sym_trades),
        }

    def _session_label(hour: int) -> str:
        if 0 <= hour < 8:
            return "Asia"
        if 8 <= hour < 13:
            return "London"
        if 13 <= hour < 16:
            return "Overlap"
        if 16 <= hour < 21:
            return "NY"
        return "Off-hours"

    session_buckets: dict[str, list] = {}
    for trade in trades:
        hour = trade.entry_time.hour if trade.entry_time else 0
        session_buckets.setdefault(_session_label(hour), []).append(trade)

    for label, session_trades in session_buckets.items():
        session_winners = [t for t in session_trades if _trade_net_pnl(t) > 0]
        m.per_session[label] = {
            "trades": len(session_trades),
            "win_rate": len(session_winners) / len(session_trades) * 100 if session_trades else 0.0,
            "net_profit": sum(_trade_net_pnl(t) for t in session_trades),
        }

    return m


def _compute_streaks(trades: list) -> tuple[int, int]:
    """Compute max consecutive wins and losses (net of commissions)."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for trade in trades:
        if _trade_net_pnl(trade) > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


def _compute_r_distribution(r_multiples: list[float]) -> dict[str, int]:
    """Bucket R-multiples into ranges."""
    buckets = {
        "< -1.0": 0,
        "-1.0 to -0.5": 0,
        "-0.5 to 0": 0,
        "0 to 0.5": 0,
        "0.5 to 1.0": 0,
        "1.0 to 2.0": 0,
        "> 2.0": 0,
    }
    for r_multiple in r_multiples:
        if r_multiple < -1.0:
            buckets["< -1.0"] += 1
        elif r_multiple < -0.5:
            buckets["-1.0 to -0.5"] += 1
        elif r_multiple < 0:
            buckets["-0.5 to 0"] += 1
        elif r_multiple < 0.5:
            buckets["0 to 0.5"] += 1
        elif r_multiple < 1.0:
            buckets["0.5 to 1.0"] += 1
        elif r_multiple < 2.0:
            buckets["1.0 to 2.0"] += 1
        else:
            buckets["> 2.0"] += 1
    return buckets


def _compute_group_breakdown(trades: list, *, key_fn: object) -> dict:
    """Group trades by a key function and compute stats per group."""
    groups: dict = {}
    for trade in trades:
        key = key_fn(trade)
        groups.setdefault(key, []).append(trade)

    result = {}
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        group_winners = [trade for trade in group if _trade_net_pnl(trade) > 0]
        group_rs = _trade_reporting_rs(group)
        result[key] = {
            "trades": len(group),
            "win_rate": len(group_winners) / len(group) * 100 if group else 0.0,
            "net_profit": sum(_trade_net_pnl(trade) for trade in group),
            "avg_r": float(np.mean(group_rs)) if group_rs else 0.0,
        }
    return result


def metrics_to_dict(metrics: PerformanceMetrics) -> dict[str, float]:
    """Convert PerformanceMetrics to a flat dict for scoring."""
    return {
        "net_profit": metrics.net_profit,
        "net_return_pct": metrics.net_return_pct,
        "realized_pnl_net": metrics.realized_pnl_net,
        "terminal_mark_pnl_net": metrics.terminal_mark_pnl_net,
        "terminal_mark_count": float(metrics.terminal_mark_count),
        "total_trades": float(metrics.total_trades),
        "win_rate": metrics.win_rate,
        "avg_winner_r": metrics.avg_winner_r,
        "avg_loser_r": metrics.avg_loser_r,
        "expectancy_r": metrics.expectancy_r,
        "profit_factor": metrics.profit_factor,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "max_drawdown_duration": float(metrics.max_drawdown_duration),
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "avg_bars_held": metrics.avg_bars_held,
        "avg_mae_r": metrics.avg_mae_r,
        "avg_mfe_r": metrics.avg_mfe_r,
        "exit_efficiency": metrics.exit_efficiency,
        "a_setup_win_rate": metrics.a_setup_win_rate,
        "b_setup_win_rate": metrics.b_setup_win_rate,
        "long_win_rate": metrics.long_win_rate,
        "short_win_rate": metrics.short_win_rate,
        "total_fees": metrics.total_fees,
        "funding_cost_total": metrics.funding_cost_total,
        "edge_ratio": metrics.edge_ratio,
        "payoff_ratio": metrics.payoff_ratio,
        "recovery_factor": metrics.recovery_factor,
        "max_consecutive_losses": float(metrics.max_consecutive_losses),
    }


def filter_metrics_for_scoring(
    metrics: dict[str, float],
    trades: list,
    exclude_exit_reasons: set[str] | None,
) -> dict[str, float]:
    """Return a metrics dict with trade-based fields recomputed after excluding trades."""
    if not exclude_exit_reasons:
        return metrics

    from crypto_trader.core.models import SetupGrade, Side

    filtered = [trade for trade in trades if (trade.exit_reason or "") not in exclude_exit_reasons]
    out = dict(metrics)

    n = len(filtered)
    out["total_trades"] = float(n)

    if n == 0:
        for key in (
            "win_rate",
            "avg_winner_r",
            "avg_loser_r",
            "expectancy_r",
            "profit_factor",
            "avg_bars_held",
            "avg_mae_r",
            "avg_mfe_r",
            "exit_efficiency",
            "edge_ratio",
            "payoff_ratio",
            "max_consecutive_losses",
            "a_setup_win_rate",
            "b_setup_win_rate",
            "long_win_rate",
            "short_win_rate",
            "total_fees",
            "funding_cost_total",
        ):
            out[key] = 0.0
        return out

    winners = [trade for trade in filtered if _trade_net_pnl(trade) > 0]
    losers = [trade for trade in filtered if _trade_net_pnl(trade) <= 0]

    out["win_rate"] = len(winners) / n * 100

    winner_rs = _trade_reporting_rs(winners)
    loser_rs = _trade_reporting_rs(losers)
    out["avg_winner_r"] = float(np.mean(winner_rs)) if winner_rs else 0.0
    out["avg_loser_r"] = float(np.mean(loser_rs)) if loser_rs else 0.0

    all_rs = _trade_reporting_rs(filtered)
    out["expectancy_r"] = float(np.mean(all_rs)) if all_rs else 0.0

    gross_profit = sum(_trade_net_pnl(trade) for trade in winners) if winners else 0.0
    gross_loss = abs(sum(_trade_net_pnl(trade) for trade in losers)) if losers else 0.0
    out["profit_factor"] = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf") if gross_profit > 0
        else 0.0
    )

    bars_held = [trade.bars_held for trade in filtered]
    out["avg_bars_held"] = float(np.mean(bars_held)) if bars_held else 0.0

    maes = [trade.mae_r for trade in filtered if trade.mae_r is not None]
    mfes = [trade.mfe_r for trade in filtered if trade.mfe_r is not None]
    out["avg_mae_r"] = float(np.mean(maes)) if maes else 0.0
    out["avg_mfe_r"] = float(np.mean(mfes)) if mfes else 0.0

    out["exit_efficiency"] = _compute_exit_efficiency(filtered)

    if out["avg_mae_r"] != 0:
        out["edge_ratio"] = float(out["avg_mfe_r"] / abs(out["avg_mae_r"]))
    else:
        out["edge_ratio"] = 0.0

    avg_winner_pnl = float(np.mean([_trade_net_pnl(trade) for trade in winners])) if winners else 0.0
    avg_loser_pnl = abs(float(np.mean([_trade_net_pnl(trade) for trade in losers]))) if losers else 0.0
    out["payoff_ratio"] = float(avg_winner_pnl / avg_loser_pnl) if avg_loser_pnl > 0 else 0.0

    _, max_consecutive_losses = _compute_streaks(filtered)
    out["max_consecutive_losses"] = float(max_consecutive_losses)

    a_trades = [trade for trade in filtered if trade.setup_grade == SetupGrade.A]
    b_trades = [trade for trade in filtered if trade.setup_grade == SetupGrade.B]
    out["a_setup_win_rate"] = (
        sum(1 for trade in a_trades if _trade_net_pnl(trade) > 0) / len(a_trades) * 100
    ) if a_trades else 0.0
    out["b_setup_win_rate"] = (
        sum(1 for trade in b_trades if _trade_net_pnl(trade) > 0) / len(b_trades) * 100
    ) if b_trades else 0.0

    longs = [trade for trade in filtered if trade.direction == Side.LONG]
    shorts = [trade for trade in filtered if trade.direction == Side.SHORT]
    out["long_win_rate"] = (
        sum(1 for trade in longs if _trade_net_pnl(trade) > 0) / len(longs) * 100
    ) if longs else 0.0
    out["short_win_rate"] = (
        sum(1 for trade in shorts if _trade_net_pnl(trade) > 0) / len(shorts) * 100
    ) if shorts else 0.0

    out["funding_cost_total"] = sum(_trade_funding_paid(trade) for trade in filtered)
    out["total_fees"] = sum(_trade_total_fees(trade) for trade in filtered)
    return out


def _compute_weekly_returns(trades: list) -> list[dict]:
    """Compute weekly PnL from realized closed trades."""
    if not trades:
        return []

    weekly: dict[str, float] = {}
    for trade in trades:
        if not trade.entry_time:
            continue
        iso = trade.entry_time.isocalendar()
        week_key = f"{iso[0]}-W{iso[1]:02d}"
        weekly[week_key] = weekly.get(week_key, 0.0) + _trade_net_pnl(trade)

    return [{"week": key, "pnl": pnl} for key, pnl in sorted(weekly.items())]
