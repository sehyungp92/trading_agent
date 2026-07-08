"""Text and selection-quality reports for stock backtest results."""
from __future__ import annotations

from datetime import date

import numpy as np

from backtests.stock.analysis.metrics import PerformanceMetrics, compute_metrics
from backtests.stock.models import TradeRecord


def format_summary(metrics: PerformanceMetrics, title: str = "Performance Summary") -> str:
    """Format PerformanceMetrics into a readable text report."""
    lines = [
        f"\n{'='*60}",
        f"  {title}",
        f"{'='*60}",
        "",
        f"  Total Trades:      {metrics.total_trades}",
        f"  Winning:           {metrics.winning_trades} ({metrics.win_rate:.1%})",
        f"  Losing:            {metrics.losing_trades}",
        "",
        f"  Net Profit:        ${metrics.net_profit:>12,.2f}",
        f"  Gross Profit:      ${metrics.gross_profit:>12,.2f}",
        f"  Gross Loss:        ${metrics.gross_loss:>12,.2f}",
        f"  Profit Factor:     {metrics.profit_factor:>8.2f}",
        f"  Expectancy (R):    {metrics.expectancy:>8.2f}",
        f"  Expectancy ($):    ${metrics.expectancy_dollar:>12,.2f}",
        "",
        f"  CAGR:              {metrics.cagr:.2%}",
        f"  Sharpe:            {metrics.sharpe:>8.2f}",
        f"  Sortino:           {metrics.sortino:>8.2f}",
        f"  Calmar:            {metrics.calmar:>8.2f}",
        "",
        f"  Max Drawdown:      {metrics.max_drawdown_pct:.2%}  (${metrics.max_drawdown_dollar:,.2f})",
        f"  Tail Loss (5%):    ${metrics.tail_loss_pct:>12,.2f}",
        f"  Tail Loss (R):     {metrics.tail_loss_r:>8.2f}",
        "",
        f"  Avg Hold Hours:    {metrics.avg_hold_hours:>8.1f}",
        f"  Trades/Month:      {metrics.trades_per_month:>8.1f}",
        f"  Commissions:       ${metrics.total_commissions:>12,.2f}",
        f"{'='*60}",
    ]
    return "\n".join(lines)


def compute_and_format(
    trades: list[TradeRecord],
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
    title: str = "Performance Summary",
) -> tuple[PerformanceMetrics, str]:
    """Compute metrics from trades and format a summary report."""
    if not trades:
        m = PerformanceMetrics()
        return m, format_summary(m, title)

    pnls = np.array([t.pnl_net for t in trades])
    risks = np.array([t.risk_per_share * t.quantity for t in trades])
    hold_hours = np.array([t.hold_hours for t in trades])
    commissions = np.array([t.commission for t in trades])
    symbols = [t.symbol for t in trades]

    m = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=hold_hours,
        trade_commissions=commissions,
        equity_curve=equity_curve,
        timestamps=timestamps,
        initial_equity=initial_equity,
        trade_symbols=symbols,
    )
    return m, format_summary(m, title)


def regime_breakdown(trades: list[TradeRecord]) -> str:
    """Break down performance by regime tier."""
    tiers: dict[str, list[TradeRecord]] = {}
    for t in trades:
        tier = t.regime_tier or "UNKNOWN"
        tiers.setdefault(tier, []).append(t)

    lines = ["\n  Regime Breakdown:", "  " + "-" * 50]
    for tier in sorted(tiers):
        group = tiers[tier]
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        total_pnl = sum(t.pnl_net for t in group)
        avg_r = np.mean([t.r_multiple for t in group]) if group else 0
        wr = wins / n if n > 0 else 0.0
        lines.append(
            f"  Tier {tier}: {n:>4} trades, WR={wr:.1%}, "
            f"PnL=${total_pnl:>10,.2f}, Avg R={avg_r:.2f}"
        )
    return "\n".join(lines)


def sector_breakdown(trades: list[TradeRecord]) -> str:
    """Break down performance by sector."""
    sectors: dict[str, list[TradeRecord]] = {}
    for t in trades:
        sec = t.sector or "UNKNOWN"
        sectors.setdefault(sec, []).append(t)

    lines = ["\n  Sector Breakdown:", "  " + "-" * 60]
    sorted_sectors = sorted(sectors.items(), key=lambda x: sum(t.pnl_net for t in x[1]), reverse=True)
    for sec, group in sorted_sectors:
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        total_pnl = sum(t.pnl_net for t in group)
        wr = wins / n if n > 0 else 0.0
        lines.append(
            f"  {sec:<28s} {n:>3} trades, WR={wr:.1%}, PnL=${total_pnl:>10,.2f}"
        )
    return "\n".join(lines)


def entry_type_breakdown(trades: list[TradeRecord]) -> str:
    """Break down performance by entry type."""
    types: dict[str, list[TradeRecord]] = {}
    for t in trades:
        et = t.entry_type or "UNKNOWN"
        types.setdefault(et, []).append(t)

    lines = ["\n  Entry Type Breakdown:", "  " + "-" * 50]
    for et, group in sorted(types.items()):
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        total_pnl = sum(t.pnl_net for t in group)
        avg_r = np.mean([t.r_multiple for t in group]) if group else 0
        wr = wins / n if n > 0 else 0.0
        lines.append(
            f"  {et:<25s} {n:>4} trades, WR={wr:.1%}, "
            f"PnL=${total_pnl:>10,.2f}, Avg R={avg_r:.2f}"
        )
    return "\n".join(lines)


def exit_reason_breakdown(trades: list[TradeRecord]) -> str:
    """Break down performance by exit reason."""
    reasons: dict[str, list[TradeRecord]] = {}
    for t in trades:
        reason = t.exit_reason or "UNKNOWN"
        reasons.setdefault(reason, []).append(t)

    lines = ["\n  Exit Reason Breakdown:", "  " + "-" * 50]
    for reason, group in sorted(reasons.items()):
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        total_pnl = sum(t.pnl_net for t in group)
        wr = wins / n if n > 0 else 0.0
        lines.append(
            f"  {reason:<25s} {n:>4} trades, WR={wr:.1%}, PnL=${total_pnl:>10,.2f}"
        )
    return "\n".join(lines)


def selection_quality_report(
    trades: list[TradeRecord],
    daily_selections: dict | None = None,
) -> str:
    """Selection-quality report for Tier 1 backtests.

    Analyzes hit rate by score tier, sector attribution, and direction bias.
    """
    if not trades:
        return "\n  No trades to analyze."

    lines = [
        "\n" + "=" * 60,
        "  Selection Quality Analysis",
        "=" * 60,
    ]

    # Direction breakdown
    longs = [t for t in trades if t.direction.value > 0]
    shorts = [t for t in trades if t.direction.value < 0]
    if longs:
        l_wr = sum(1 for t in longs if t.is_winner) / len(longs)
        l_pnl = sum(t.pnl_net for t in longs)
        lines.append(f"\n  LONG:  {len(longs)} trades, WR={l_wr:.1%}, PnL=${l_pnl:,.2f}")
    if shorts:
        s_wr = sum(1 for t in shorts if t.is_winner) / len(shorts)
        s_pnl = sum(t.pnl_net for t in shorts)
        lines.append(f"  SHORT: {len(shorts)} trades, WR={s_wr:.1%}, PnL=${s_pnl:,.2f}")

    # Monthly performance
    monthly: dict[str, list[TradeRecord]] = {}
    for t in trades:
        key = t.entry_time.strftime("%Y-%m")
        monthly.setdefault(key, []).append(t)

    lines.append("\n  Monthly Performance:")
    lines.append("  " + "-" * 50)
    for month in sorted(monthly):
        group = monthly[month]
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        total_pnl = sum(t.pnl_net for t in group)
        wr = wins / n if n > 0 else 0
        lines.append(f"  {month}: {n:>3} trades, WR={wr:.1%}, PnL=${total_pnl:>10,.2f}")

    # Append sector and regime breakdowns
    lines.append(regime_breakdown(trades))
    lines.append(sector_breakdown(trades))
    lines.append(entry_type_breakdown(trades))
    lines.append(exit_reason_breakdown(trades))

    return "\n".join(lines)


def full_report(
    trades: list[TradeRecord],
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
    strategy: str = "Stock",
    daily_selections: dict | None = None,
) -> str:
    """Generate a complete backtest report."""
    metrics, summary = compute_and_format(
        trades, equity_curve, timestamps, initial_equity,
        title=f"{strategy} Backtest Results",
    )
    parts = [summary]
    if trades:
        parts.append(selection_quality_report(trades, daily_selections))
    return "\n".join(parts)
