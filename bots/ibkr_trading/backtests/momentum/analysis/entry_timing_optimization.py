"""Entry timing optimization — identify highest-expectancy entry windows.

Analyzes time-of-day histograms per entry class, finds optimal entry windows,
evaluates "should-have-waited" scenarios, and breaks down RTH vs ETH distribution.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime, time

from backtests.momentum.analysis._utils import utc_to_et as _utc_to_et


def _is_rth(et_time: time) -> bool:
    """Check if time falls within RTH (09:30-16:00 ET)."""
    return time(9, 30) <= et_time < time(16, 0)


def generate_entry_timing_report(trades: list) -> str:
    """Generate entry timing optimization report.

    Args:
        trades: Trade records with entry_time, pnl_dollars, entry_class, etc.

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  ENTRY TIMING OPTIMIZATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    if not trades:
        lines.append("  No trades to analyze.")
        return "\n".join(lines)

    # ── A. Time-of-day histogram per entry class ──
    lines.append("  A. ENTRY TIME HISTOGRAM BY HOUR (ET)")
    lines.append("  " + "-" * 55)

    hour_class: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    hour_all: dict[int, list[float]] = defaultdict(list)

    for t in trades:
        et_dt = _utc_to_et(getattr(t, "entry_time", None))
        if et_dt is None:
            continue
        h = et_dt.hour
        pnl = getattr(t, "pnl_dollars", 0.0)
        ec = getattr(t, "entry_class", "unknown")
        hour_class[ec][h].append(pnl)
        hour_all[h].append(pnl)

    classes = sorted(hour_class.keys())

    # All-strategies combined histogram
    lines.append("\n    Combined (all entry classes):")
    lines.append(f"    {'Hour':>4s} {'Count':>5s} {'WR%':>6s} {'AvgPnL':>10s} {'TotalPnL':>10s} {'Hist':>15s}")
    lines.append("    " + "-" * 55)

    max_count = max((len(v) for v in hour_all.values()), default=1)
    for h in range(24):
        if h not in hour_all:
            continue
        arr = np.array(hour_all[h])
        wr = float(np.mean(arr > 0)) * 100
        bar_len = int(len(arr) / max_count * 12)
        bar = "#" * bar_len
        lines.append(
            f"    {h:>4d} {len(arr):>5d} {wr:>5.1f}% ${np.mean(arr):>+9,.0f} "
            f"${np.sum(arr):>+9,.0f} {bar}"
        )

    # Per entry-class breakdown
    for ec in classes:
        lines.append(f"\n    Entry class: {ec}")
        lines.append(f"    {'Hour':>4s} {'Count':>5s} {'WR%':>6s} {'AvgPnL':>10s}")
        lines.append("    " + "-" * 30)
        for h in range(24):
            if h not in hour_class[ec]:
                continue
            arr = np.array(hour_class[ec][h])
            if len(arr) == 0:
                continue
            wr = float(np.mean(arr > 0)) * 100
            lines.append(f"    {h:>4d} {len(arr):>5d} {wr:>5.1f}% ${np.mean(arr):>+9,.0f}")

    # ── B. Optimal entry windows ──
    lines.append("")
    lines.append("  B. OPTIMAL ENTRY WINDOWS (highest expectancy hours)")
    lines.append("  " + "-" * 55)

    hour_stats = []
    for h in range(24):
        if h not in hour_all or len(hour_all[h]) < 3:
            continue
        arr = np.array(hour_all[h])
        wr = float(np.mean(arr > 0))
        avg = float(np.mean(arr))
        hour_stats.append({"hour": h, "count": len(arr), "wr": wr, "avg_pnl": avg, "total": float(np.sum(arr))})

    hour_stats.sort(key=lambda x: x["avg_pnl"], reverse=True)

    lines.append(f"    {'Rank':>4s} {'Hour':>4s} {'Count':>5s} {'WR%':>6s} {'AvgPnL':>10s} {'TotalPnL':>10s}")
    lines.append("    " + "-" * 43)
    for rank, hs in enumerate(hour_stats[:5], 1):
        lines.append(
            f"    {rank:>4d} {hs['hour']:>4d} {hs['count']:>5d} {hs['wr']*100:>5.1f}% "
            f"${hs['avg_pnl']:>+9,.0f} ${hs['total']:>+9,.0f}"
        )

    if hour_stats:
        worst = hour_stats[-1]
        lines.append(f"\n    Worst hour: {worst['hour']:02d}:00 ET "
                     f"({worst['count']} trades, avg=${worst['avg_pnl']:+,.0f})")

    # ── C. "Should-have-waited" analysis ──
    lines.append("")
    lines.append("  C. SHOULD-HAVE-WAITED ANALYSIS")
    lines.append("  " + "-" * 55)
    lines.append("    Trades entered in poor windows (bottom-3 hours by expectancy):")

    if len(hour_stats) >= 3:
        poor_hours = {hs["hour"] for hs in hour_stats[-3:] if hs["avg_pnl"] < 0}
        if poor_hours:
            poor_trades = []
            for t in trades:
                et_dt = _utc_to_et(getattr(t, "entry_time", None))
                if et_dt and et_dt.hour in poor_hours:
                    poor_trades.append(t)

            poor_pnl = np.array([getattr(t, "pnl_dollars", 0.0) for t in poor_trades])
            total_pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in trades)

            lines.append(f"    Poor-window hours: {sorted(poor_hours)}")
            lines.append(f"    Trades in poor windows:  {len(poor_trades)}")
            if len(poor_pnl) > 0:
                lines.append(f"    P&L from poor windows:   ${np.sum(poor_pnl):+,.0f}")
                lines.append(f"    Overall P&L:             ${total_pnl:+,.0f}")
                lines.append(f"    If skipped poor windows: ${total_pnl - np.sum(poor_pnl):+,.0f}")
        else:
            lines.append("    No hours with negative expectancy found.")
    else:
        lines.append("    Insufficient data for window ranking.")

    # ── D. RTH vs ETH distribution ──
    lines.append("")
    lines.append("  D. RTH vs ETH DISTRIBUTION")
    lines.append("  " + "-" * 55)

    rth_trades = []
    eth_trades = []
    for t in trades:
        et_dt = _utc_to_et(getattr(t, "entry_time", None))
        if et_dt is None:
            continue
        if _is_rth(et_dt.time()):
            rth_trades.append(t)
        else:
            eth_trades.append(t)

    for label, group in [("RTH (09:30-16:00)", rth_trades), ("ETH (other)", eth_trades)]:
        if group:
            arr = np.array([getattr(t, "pnl_dollars", 0.0) for t in group])
            wr = float(np.mean(arr > 0)) * 100
            lines.append(
                f"    {label:<20s}: {len(arr):>4d} trades ({len(arr)/len(trades)*100:.0f}%), "
                f"WR={wr:.0f}%, net=${np.sum(arr):+,.0f}, avg=${np.mean(arr):+,.0f}"
            )

    if rth_trades and eth_trades:
        rth_avg = np.mean([getattr(t, "pnl_dollars", 0.0) for t in rth_trades])
        eth_avg = np.mean([getattr(t, "pnl_dollars", 0.0) for t in eth_trades])
        lines.append(f"\n    RTH vs ETH avg P&L delta: ${rth_avg - eth_avg:+,.0f}")

    return "\n".join(lines)
