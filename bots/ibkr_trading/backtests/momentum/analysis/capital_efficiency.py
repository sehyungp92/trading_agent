"""Capital efficiency analysis — how effectively is risk budget deployed.

Measures heat utilization over time, capital-deployed vs idle by time-of-day,
R-per-hour-deployed efficiency, and priority blocking cost analysis.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime

from backtests.momentum.analysis._utils import utc_to_et as _utc_to_et


def _hours_held(trade) -> float:
    """Compute hours a trade was held."""
    entry = getattr(trade, "entry_time", None)
    exit_ = getattr(trade, "exit_time", None)
    if entry is None or exit_ is None:
        return 0.0
    if not isinstance(entry, datetime):
        try:
            import pandas as pd
            entry = pd.Timestamp(entry).to_pydatetime()
            exit_ = pd.Timestamp(exit_).to_pydatetime()
        except Exception:
            return 0.0
    return max(0.0, (exit_ - entry).total_seconds() / 3600)


def generate_capital_efficiency_report(
    trades: list,
    equity_curve: list | np.ndarray | None = None,
    timestamps: list | None = None,
    initial_equity: float = 10_000.0,
    blocked_trades: list | None = None,
) -> str:
    """Generate capital efficiency and heat utilization report.

    Args:
        trades: Trade records with standard attributes.
        equity_curve: Optional equity values over time.
        timestamps: Optional timestamp list aligned with equity_curve.
        initial_equity: Starting account equity.
        blocked_trades: Trades rejected by portfolio rules (from result.blocked_trades).

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  CAPITAL EFFICIENCY REPORT")
    lines.append("=" * 72)
    lines.append("")

    if not trades:
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    pnl_arr = np.array([getattr(t, "pnl_dollars", 0.0) for t in trades])
    total_pnl = float(np.sum(pnl_arr))
    n_trades = len(trades)

    # ── A. Heat utilization timeline ──
    lines.append("  A. HEAT UTILIZATION TIMELINE")
    lines.append("  " + "-" * 55)

    # Build minute-level occupancy grid
    all_intervals = []
    for t in trades:
        entry = getattr(t, "entry_time", None)
        exit_ = getattr(t, "exit_time", None)
        if entry is None or exit_ is None:
            continue
        if not isinstance(entry, datetime):
            try:
                import pandas as pd
                entry = pd.Timestamp(entry).to_pydatetime()
                exit_ = pd.Timestamp(exit_).to_pydatetime()
            except Exception:
                continue
        all_intervals.append((entry, exit_))

    if all_intervals:
        min_t = min(e for e, _ in all_intervals)
        max_t = max(x for _, x in all_intervals)
        total_hours = (max_t - min_t).total_seconds() / 3600

        # Count concurrent positions at each trade boundary
        events = []
        for entry, exit_ in all_intervals:
            events.append((entry, 1))
            events.append((exit_, -1))
        events.sort(key=lambda x: x[0])

        concurrent = 0
        max_concurrent = 0
        time_deployed = 0.0
        last_time = events[0][0]

        for evt_time, delta in events:
            if concurrent > 0:
                time_deployed += (evt_time - last_time).total_seconds() / 3600
            last_time = evt_time
            concurrent += delta
            max_concurrent = max(max_concurrent, concurrent)

        pct_deployed = time_deployed / total_hours * 100 if total_hours > 0 else 0
        lines.append(f"    Backtest span:       {total_hours:,.0f} hours")
        lines.append(f"    Hours deployed:      {time_deployed:,.0f} ({pct_deployed:.1f}%)")
        lines.append(f"    Hours idle:          {total_hours - time_deployed:,.0f} ({100 - pct_deployed:.1f}%)")
        lines.append(f"    Max concurrent:      {max_concurrent}")
    else:
        total_hours = 0
        time_deployed = 0
        lines.append("    [Insufficient timestamp data for timeline]")

    # ── B. Capital deployed vs idle by time-of-day ──
    lines.append("")
    lines.append("  B. CAPITAL DEPLOYED BY HOUR (ET)")
    lines.append("  " + "-" * 55)

    hour_deployed: dict[int, float] = defaultdict(float)
    hour_count: dict[int, int] = defaultdict(int)

    for t in trades:
        et_entry = _utc_to_et(getattr(t, "entry_time", None))
        if et_entry is None:
            continue
        h = et_entry.hour
        dur = _hours_held(t)
        hour_deployed[h] += dur
        hour_count[h] += 1

    if hour_deployed:
        lines.append(f"    {'Hour':>4s} {'Trades':>6s} {'TotalHrs':>9s} {'AvgHrs':>7s} {'Bar':>20s}")
        max_hrs = max(hour_deployed.values()) if hour_deployed else 1
        for h in range(24):
            cnt = hour_count.get(h, 0)
            hrs = hour_deployed.get(h, 0.0)
            avg = hrs / cnt if cnt > 0 else 0.0
            bar_len = int(hrs / max_hrs * 15) if max_hrs > 0 else 0
            bar = "#" * bar_len
            lines.append(f"    {h:>4d} {cnt:>6d} {hrs:>9.1f} {avg:>7.2f} {bar}")

    # ── C. R-per-hour-deployed efficiency ──
    lines.append("")
    lines.append("  C. R-PER-HOUR-DEPLOYED EFFICIENCY")
    lines.append("  " + "-" * 55)

    total_hold_hours = sum(_hours_held(t) for t in trades)
    r_per_hour = total_pnl / total_hold_hours if total_hold_hours > 0 else 0
    lines.append(f"    Total P&L:            ${total_pnl:+,.0f}")
    lines.append(f"    Total hold hours:     {total_hold_hours:,.1f}")
    lines.append(f"    $/hour deployed:      ${r_per_hour:+,.2f}")

    # Per-strategy breakdown
    strat_groups: dict[str, list] = defaultdict(list)
    for t in trades:
        strat_groups[getattr(t, "entry_class", "unknown")].append(t)

    if len(strat_groups) > 1:
        lines.append(f"\n    {'Strategy':<14s} {'PnL':>10s} {'Hours':>8s} {'$/hr':>8s}")
        lines.append("    " + "-" * 44)
        for name, strades in sorted(strat_groups.items()):
            s_pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in strades)
            s_hrs = sum(_hours_held(t) for t in strades)
            s_eff = s_pnl / s_hrs if s_hrs > 0 else 0
            lines.append(f"    {name:<14s} ${s_pnl:>+9,.0f} {s_hrs:>7.1f} ${s_eff:>+7.2f}")

    # ── D. Priority blocking cost ──
    lines.append("")
    lines.append("  D. PRIORITY BLOCKING COST ANALYSIS")
    lines.append("  " + "-" * 55)

    if blocked_trades:
        blocked_pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in blocked_trades)
        lines.append(f"    Blocked trades:    {len(blocked_trades)}")
        lines.append(f"    Foregone P&L:      ${blocked_pnl:+,.0f}")
    else:
        lines.append("    No blocked trades provided.")

    # ── E. Equity utilization ratio ──
    lines.append("")
    lines.append("  E. EQUITY UTILIZATION")
    lines.append("  " + "-" * 55)

    if equity_curve is not None and len(equity_curve) > 0:
        eq = np.array(equity_curve)
        avg_eq = float(np.mean(eq))
        lines.append(f"    Initial equity:     ${initial_equity:,.0f}")
        lines.append(f"    Final equity:       ${float(eq[-1]):,.0f}")
        lines.append(f"    Avg equity:         ${avg_eq:,.0f}")
        lines.append(f"    Return on avg eq:   {total_pnl / avg_eq * 100:.1f}%")
    else:
        roi = total_pnl / initial_equity * 100 if initial_equity > 0 else 0
        lines.append(f"    Initial equity:     ${initial_equity:,.0f}")
        lines.append(f"    Net P&L:            ${total_pnl:+,.0f}")
        lines.append(f"    Simple ROI:         {roi:.1f}%")

    return "\n".join(lines)
