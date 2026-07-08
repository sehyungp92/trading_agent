"""Portfolio-level backtest reports.

Each function takes a PortfolioResult and returns a formatted string section.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np

from backtests.momentum.analysis.metrics import (
    PerformanceMetrics,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
)
from backtests.momentum.engine.portfolio_engine import PortfolioResult, PortfolioTrade


# ---------------------------------------------------------------------------
# 1. Overall summary
# ---------------------------------------------------------------------------

def portfolio_summary_report(
    result: PortfolioResult,
    independent_pnl: dict[str, float] | None = None,
) -> str:
    """Overall portfolio performance summary.

    Args:
        independent_pnl: Mapping strategy_id -> total R-multiples from isolated run.
            Used to compute the "sum-of-independent" baseline R-capture.
    """
    trades = result.trades
    blocked = result.blocked_trades
    n_total = len(trades) + len(blocked)

    lines = ["=" * 60, "PORTFOLIO BACKTEST SUMMARY", "=" * 60, ""]

    if not trades:
        lines.append("No approved trades.")
        return "\n".join(lines)

    pnls = np.array([t.adjusted_pnl for t in trades])
    wins = pnls > 0
    net_pnl = float(np.sum(pnls))
    win_rate = float(np.mean(wins)) if len(pnls) > 0 else 0.0
    gross_profit = float(np.sum(pnls[wins]))
    gross_loss = float(np.sum(pnls[~wins]))
    pf = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    # R-multiples (adjusted by size mult)
    r_mults = np.array([t.r_multiple * t.size_multiplier for t in trades])
    exp_r = float(np.mean(r_mults)) if len(r_mults) > 0 else 0.0

    # Equity curve stats (daily-sampled)
    ec = result.equity_curve
    max_dd_pct, max_dd_dollar = compute_max_drawdown(ec)
    sharpe = compute_sharpe(ec, periods_per_year=252)
    sortino = compute_sortino(ec, periods_per_year=252)

    # CAGR
    if len(ec) >= 2 and len(result.equity_timestamps) >= 2:
        span = result.equity_timestamps[-1] - result.equity_timestamps[0]
        years = span.total_seconds() / (365.25 * 24 * 3600)
    else:
        years = 1.0
    final_equity = float(ec[-1]) if len(ec) > 0 else result.initial_equity
    cagr = compute_cagr(result.initial_equity, final_equity, years)
    calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

    lines.extend([
        f"Initial equity:     ${result.initial_equity:,.0f}",
        f"Final equity:       ${final_equity:,.0f}",
        f"Net P&L:            ${net_pnl:+,.0f}",
        f"",
        f"Trades approved:    {len(trades)} / {n_total} "
        f"({len(blocked)} blocked)",
        f"Win rate:           {win_rate:.1%}",
        f"Profit factor:      {pf:.2f}",
        f"Expectancy (R):     {exp_r:+.3f}",
        f"Expectancy ($):     ${net_pnl / len(trades):+,.0f}",
        f"",
        f"CAGR:               {cagr:.1%}",
        f"Sharpe:             {sharpe:.2f}",
        f"Sortino:            {sortino:.2f}",
        f"Calmar:             {calmar:.2f}",
        f"Max drawdown:       {max_dd_pct:.1%} (${max_dd_dollar:,.0f})",
        f"Period:             {years:.1f} years",
    ])

    # Sum-of-independent baseline (R-multiples)
    if independent_pnl:
        iso_total_R = sum(independent_pnl.values())
        portfolio_total_R = float(np.sum(r_mults))
        capture_pct = portfolio_total_R / iso_total_R * 100 if iso_total_R != 0 else 0
        lines.extend([
            f"",
            f"--- Sum-of-Independent Baseline (R) ---",
            f"Isolated total R:   {iso_total_R:+.1f}R",
            f"Portfolio total R:  {portfolio_total_R:+.1f}R",
            f"R-capture ratio:    {capture_pct:.1f}%",
        ])
        for sid, total_r in sorted(independent_pnl.items()):
            lines.append(f"  {sid}: {total_r:+.1f}R")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Per-strategy breakdown
# ---------------------------------------------------------------------------

def portfolio_strategy_breakdown(result: PortfolioResult) -> str:
    """Per-strategy trade counts, PnL contribution, win rates."""
    lines = ["", "=" * 60, "STRATEGY BREAKDOWN", "=" * 60, ""]

    total_pnl = sum(t.adjusted_pnl for t in result.trades)

    strategies = sorted(set(
        [t.strategy_id for t in result.trades]
        + [t.strategy_id for t in result.blocked_trades]
    ))

    header = f"{'Strategy':<12} {'Approved':>8} {'Blocked':>8} {'WR':>6} {'E[R]':>8} {'PnL':>12} {'% Total':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for sid in strategies:
        approved = [t for t in result.trades if t.strategy_id == sid]
        blocked = [t for t in result.blocked_trades if t.strategy_id == sid]
        n_approved = len(approved)
        n_blocked = len(blocked)

        if n_approved > 0:
            pnls = [t.adjusted_pnl for t in approved]
            wr = sum(1 for p in pnls if p > 0) / n_approved
            r_mults = [t.r_multiple * t.size_multiplier for t in approved]
            exp_r = sum(r_mults) / len(r_mults)
            pnl = sum(pnls)
            pct = pnl / total_pnl * 100 if total_pnl != 0 else 0
        else:
            wr = 0.0
            exp_r = 0.0
            pnl = 0.0
            pct = 0.0

        lines.append(
            f"{sid:<12} {n_approved:>8} {n_blocked:>8} "
            f"{wr:>5.0%} {exp_r:>+7.3f} ${pnl:>+11,.0f} {pct:>7.1f}%"
        )

    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<12} {len(result.trades):>8} {len(result.blocked_trades):>8} "
        f"{'':>6} {'':>8} ${total_pnl:>+11,.0f} {'100.0%':>8}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Rule impact report
# ---------------------------------------------------------------------------

def portfolio_rule_impact_report(result: PortfolioResult) -> str:
    """For each rule: block count, raw PnL of blocked trades."""
    lines = ["", "=" * 60, "RULE IMPACT ANALYSIS", "=" * 60, ""]

    if not result.rule_blocks:
        lines.append("No trades blocked by portfolio rules.")
        return "\n".join(lines)

    lines.append(
        "Blocked R = R-multiples of trades that were rejected."
    )
    lines.append(
        "Negative = rule saved us from losses.  Positive = opportunity cost."
    )
    lines.append("")

    # Aggregate blocked R-multiples per rule
    rule_blocked_R: dict[str, float] = defaultdict(float)
    for t in result.blocked_trades:
        rule_blocked_R[t.denial_reason] += t.r_multiple

    header = f"{'Rule':<28} {'Blocks':>7} {'Blocked R':>12} {'Avg R':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for rule in sorted(result.rule_blocks.keys(), key=lambda r: -result.rule_blocks[r]):
        count = result.rule_blocks[rule]
        total_r = rule_blocked_R.get(rule, 0.0)
        avg_r = total_r / count if count > 0 else 0.0
        lines.append(f"{rule:<28} {count:>7} {total_r:>+11.1f}R {avg_r:>+9.3f}R")

    total_blocks = sum(result.rule_blocks.values())
    total_r = sum(rule_blocked_R.values())
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<28} {total_blocks:>7} {total_r:>+11.1f}R"
    )

    # Interpretation
    lines.append("")
    if total_r > 0:
        lines.append(
            f"Rules collectively cost {total_r:+.1f}R in missed "
            f"profitable trades (opportunity cost of portfolio constraints)."
        )
    elif total_r < 0:
        lines.append(
            f"Rules collectively saved {abs(total_r):.1f}R by blocking "
            f"net-losing trades."
        )
    else:
        lines.append("Rules had zero net R impact.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Concurrent position analysis
# ---------------------------------------------------------------------------

def portfolio_concurrent_analysis(result: PortfolioResult) -> str:
    """Distribution of simultaneous open positions, max heat reached."""
    lines = ["", "=" * 60, "CONCURRENT POSITIONS", "=" * 60, ""]

    lines.append(f"Max simultaneous positions: {result.max_concurrent}")
    lines.append("")

    if result.concurrent_distribution:
        header = f"{'Positions':>10} {'Events':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for n in sorted(result.concurrent_distribution.keys()):
            count = result.concurrent_distribution[n]
            lines.append(f"{n:>10} {count:>8}")

    # Max heat reached
    max_heat = max(
        (t.portfolio_heat_at_entry for t in result.trades),
        default=0.0,
    )
    lines.append(f"\nMax heat at entry: {max_heat:.2f}R")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Daily return correlation matrix
# ---------------------------------------------------------------------------

def portfolio_correlation_report(result: PortfolioResult) -> str:
    """Daily return correlation matrix across strategies."""
    lines = ["", "=" * 60, "STRATEGY CORRELATION (DAILY RETURNS)", "=" * 60, ""]

    # Build daily PnL per strategy
    daily_pnl: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))

    for t in result.trades:
        if t.exit_time:
            from backtests.momentum.engine.portfolio_engine import _trading_day
            day = _trading_day(t.exit_time)
            daily_pnl[t.strategy_id][day] += t.adjusted_pnl

    strategies = sorted(daily_pnl.keys())
    if len(strategies) < 2:
        lines.append("Need at least 2 strategies for correlation.")
        return "\n".join(lines)

    # Get all days
    all_days = sorted(set().union(*(daily_pnl[s].keys() for s in strategies)))

    # Build arrays
    arrays = {}
    for sid in strategies:
        arrays[sid] = np.array([daily_pnl[sid].get(d, 0.0) for d in all_days])

    # Correlation matrix
    lines.append(f"{'':>10}" + "".join(f"{s:>10}" for s in strategies))
    for s1 in strategies:
        row = f"{s1:>10}"
        for s2 in strategies:
            if np.std(arrays[s1]) > 0 and np.std(arrays[s2]) > 0:
                corr = float(np.corrcoef(arrays[s1], arrays[s2])[0, 1])
            else:
                corr = 0.0
            row += f"{corr:>10.3f}"
        lines.append(row)

    lines.append(f"\n({len(all_days)} trading days)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Monthly P&L table
# ---------------------------------------------------------------------------

def portfolio_monthly_table(result: PortfolioResult) -> str:
    """Monthly PnL by strategy and total."""
    lines = ["", "=" * 60, "MONTHLY P&L TABLE", "=" * 60, ""]

    if not result.trades:
        lines.append("No trades.")
        return "\n".join(lines)

    # Collect monthly PnL
    monthly: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    strategies = set()

    for t in result.trades:
        if t.exit_time:
            key = (t.exit_time.year, t.exit_time.month)
            monthly[key][t.strategy_id] += t.adjusted_pnl
            monthly[key]["_total"] += t.adjusted_pnl
            strategies.add(t.strategy_id)

    strategies = sorted(strategies)
    months = sorted(monthly.keys())

    # Header
    header = f"{'Month':>10}" + "".join(f"{s:>12}" for s in strategies) + f"{'TOTAL':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for yr, mo in months:
        row = f"{yr}-{mo:02d}    "
        for sid in strategies:
            pnl = monthly[(yr, mo)].get(sid, 0.0)
            row += f"${pnl:>+10,.0f} "
        total = monthly[(yr, mo)]["_total"]
        row += f"${total:>+10,.0f}"
        lines.append(row)

    # Totals
    lines.append("-" * len(header))
    total_row = f"{'TOTAL':>10}"
    for sid in strategies:
        total = sum(monthly[m].get(sid, 0.0) for m in months)
        total_row += f"${total:>+10,.0f} "
    grand_total = sum(monthly[m]["_total"] for m in months)
    total_row += f"${grand_total:>+10,.0f}"
    lines.append(total_row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Drawdown report
# ---------------------------------------------------------------------------

def portfolio_drawdown_report(result: PortfolioResult) -> str:
    """Max DD depth, duration, recovery analysis."""
    lines = ["", "=" * 60, "DRAWDOWN ANALYSIS", "=" * 60, ""]

    ec = result.equity_curve
    ts = result.equity_timestamps

    if len(ec) < 2:
        lines.append("Insufficient data for drawdown analysis.")
        return "\n".join(lines)

    max_dd_pct, max_dd_dollar = compute_max_drawdown(ec)
    lines.append(f"Max drawdown:       {max_dd_pct:.1%} (${max_dd_dollar:,.0f})")

    # Find max DD period
    peak = ec[0]
    peak_idx = 0
    max_dd_start = 0
    max_dd_end = 0
    max_dd_val = 0.0

    for i, val in enumerate(ec):
        if val > peak:
            peak = val
            peak_idx = i
        dd = peak - val
        if dd > max_dd_val:
            max_dd_val = dd
            max_dd_start = peak_idx
            max_dd_end = i

    if max_dd_start < len(ts) and max_dd_end < len(ts):
        dd_start_dt = ts[max_dd_start]
        dd_end_dt = ts[max_dd_end]
        dd_duration = dd_end_dt - dd_start_dt
        lines.append(f"DD start:           {dd_start_dt.strftime('%Y-%m-%d')}")
        lines.append(f"DD trough:          {dd_end_dt.strftime('%Y-%m-%d')}")
        lines.append(f"DD duration:        {dd_duration.days} days")

        # Recovery: find first point after trough where equity >= peak
        recovery_dt = None
        for i in range(max_dd_end, len(ec)):
            if ec[i] >= ec[max_dd_start]:
                recovery_dt = ts[i]
                break
        if recovery_dt:
            recovery_days = (recovery_dt - dd_end_dt).days
            lines.append(f"Recovery:           {recovery_dt.strftime('%Y-%m-%d')} ({recovery_days} days)")
        else:
            lines.append("Recovery:           Not recovered")

    # DD tier usage
    lines.append("")
    dd_tier_usage: dict[float, int] = defaultdict(int)
    for t in result.trades:
        dd_tier_usage[t.dd_tier_mult] += 1

    if dd_tier_usage:
        lines.append("Drawdown tier usage:")
        for mult in sorted(dd_tier_usage.keys(), reverse=True):
            count = dd_tier_usage[mult]
            lines.append(f"  {mult:.0%} size: {count} trades")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report generator
# ---------------------------------------------------------------------------

def portfolio_full_report(
    result: PortfolioResult,
    independent_pnl: dict[str, float] | None = None,
) -> str:
    """Generate all portfolio report sections as a single string."""
    sections = [
        portfolio_summary_report(result, independent_pnl),
        portfolio_strategy_breakdown(result),
        portfolio_rule_impact_report(result),
        portfolio_concurrent_analysis(result),
        portfolio_correlation_report(result),
        portfolio_monthly_table(result),
        portfolio_drawdown_report(result),
    ]
    return "\n\n".join(sections) + "\n"
