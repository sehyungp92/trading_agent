"""Signal conflict analysis — cross-strategy opposing-signal detection.

Identifies instances where momentum strategies took opposing positions
within a proximity window, determines which was correct, and quantifies
the cost/benefit of veto mechanisms.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

from backtests.momentum.analysis._utils import parse_dt as _parse_dt


def _direction_str(d: int) -> str:
    return "LONG" if d >= 1 else "SHORT"


def generate_signal_conflict_report(
    trades_by_strategy: dict[str, list],
    proximity_minutes: int = 60,
) -> str:
    """Generate signal conflict and veto analysis report.

    Args:
        trades_by_strategy: Dict mapping strategy name -> trade list.
        proximity_minutes: Window in minutes to detect overlapping signals.

    Returns:
        Formatted text report.
    """
    lines = ["=" * 72]
    lines.append("  SIGNAL CONFLICT REPORT")
    lines.append("=" * 72)
    lines.append("")

    names = sorted(trades_by_strategy.keys())
    if len(names) < 2:
        lines.append("  Need at least 2 strategies for conflict analysis.")
        return "\n".join(lines)

    proximity_sec = proximity_minutes * 60

    # Build pairs
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))

    # ── A. Opposing-signal instances ──
    lines.append("  A. OPPOSING-SIGNAL INSTANCES")
    lines.append("  " + "-" * 55)

    all_conflicts: list[dict] = []

    for n1, n2 in pairs:
        conflicts = []
        t1_list = trades_by_strategy[n1]
        t2_list = trades_by_strategy[n2]

        for t1 in t1_list:
            dt1 = _parse_dt(getattr(t1, "entry_time", None))
            if dt1 is None:
                continue
            d1 = getattr(t1, "direction", 1)

            for t2 in t2_list:
                dt2 = _parse_dt(getattr(t2, "entry_time", None))
                if dt2 is None:
                    continue
                d2 = getattr(t2, "direction", 1)

                if abs((dt1 - dt2).total_seconds()) <= proximity_sec and d1 != d2:
                    pnl1 = getattr(t1, "pnl_dollars", 0.0)
                    pnl2 = getattr(t2, "pnl_dollars", 0.0)
                    winner = n1 if pnl1 > pnl2 else n2
                    conflicts.append({
                        "time": min(dt1, dt2),
                        "strat1": n1, "dir1": d1, "pnl1": pnl1,
                        "strat2": n2, "dir2": d2, "pnl2": pnl2,
                        "winner": winner,
                        "net": pnl1 + pnl2,
                    })

        lines.append(f"\n    {n1} vs {n2}: {len(conflicts)} opposing-signal events")
        if conflicts:
            n1_wins = sum(1 for c in conflicts if c["winner"] == n1)
            n2_wins = len(conflicts) - n1_wins
            net = sum(c["net"] for c in conflicts)
            lines.append(f"      {n1} won: {n1_wins}  |  {n2} won: {n2_wins}")
            lines.append(f"      Combined net during conflicts: ${net:+,.0f}")

        all_conflicts.extend(conflicts)

    total_conflicts = len(all_conflicts)
    lines.append(f"\n    Total opposing-signal events: {total_conflicts}")

    # ── B. Which strategy was right ──
    lines.append("")
    lines.append("  B. CONFLICT WINNER SCORECARD")
    lines.append("  " + "-" * 55)

    win_count: dict[str, int] = defaultdict(int)
    win_pnl: dict[str, float] = defaultdict(float)

    for c in all_conflicts:
        win_count[c["winner"]] += 1
        win_pnl[c["winner"]] += abs(c["pnl1"]) if c["winner"] == c["strat1"] else abs(c["pnl2"])

    if all_conflicts:
        lines.append(f"    {'Strategy':<14s} {'Wins':>5s} {'Win%':>6s} {'WinPnL':>10s}")
        lines.append("    " + "-" * 38)
        for n in names:
            pct = win_count[n] / total_conflicts * 100 if total_conflicts > 0 else 0
            lines.append(f"    {n:<14s} {win_count[n]:>5d} {pct:>5.1f}% ${win_pnl[n]:>+9,.0f}")
    else:
        lines.append("    No conflicts found.")

    # ── C. Veto analysis ──
    lines.append("")
    lines.append("  C. VETO ANALYSIS (if one strategy vetoed)")
    lines.append("  " + "-" * 55)
    lines.append("    Simulates removing the losing side from each conflict:")
    lines.append("")

    if all_conflicts:
        for n in names:
            # Sum of PnL if this strategy's trades were vetoed when in conflict
            vetoed_pnl = 0.0
            vetoed_count = 0
            for c in all_conflicts:
                if c["strat1"] == n:
                    vetoed_pnl += c["pnl1"]
                    vetoed_count += 1
                elif c["strat2"] == n:
                    vetoed_pnl += c["pnl2"]
                    vetoed_count += 1

            total_strat_pnl = sum(getattr(t, "pnl_dollars", 0.0) for t in trades_by_strategy.get(n, []))
            adj_pnl = total_strat_pnl - vetoed_pnl
            lines.append(
                f"    Veto {n:<10s}: remove {vetoed_count} trades, "
                f"save ${-vetoed_pnl:+,.0f} -> adjusted P&L ${adj_pnl:+,.0f} "
                f"(was ${total_strat_pnl:+,.0f})"
            )

        # Best veto: remove the losing side from each conflict
        best_veto_savings = sum(
            min(c["pnl1"], c["pnl2"]) for c in all_conflicts
        )
        lines.append(f"\n    Perfect hindsight veto savings: ${-best_veto_savings:+,.0f}")

    # ── D. Proximity window analysis ──
    lines.append("")
    lines.append("  D. PROXIMITY WINDOW SENSITIVITY")
    lines.append("  " + "-" * 55)
    lines.append(f"    Testing conflict counts at various windows:")
    lines.append(f"    {'Window':>8s} {'Conflicts':>9s} {'Same-Dir':>9s} {'Opposing':>9s}")

    for window in [5, 15, 30, 60, 120]:
        wsec = window * 60
        same_dir = 0
        opposing = 0

        for n1, n2 in pairs:
            for t1 in trades_by_strategy[n1]:
                dt1 = _parse_dt(getattr(t1, "entry_time", None))
                if dt1 is None:
                    continue
                for t2 in trades_by_strategy[n2]:
                    dt2 = _parse_dt(getattr(t2, "entry_time", None))
                    if dt2 is None:
                        continue
                    if abs((dt1 - dt2).total_seconds()) <= wsec:
                        d1 = getattr(t1, "direction", 1)
                        d2 = getattr(t2, "direction", 1)
                        if d1 == d2:
                            same_dir += 1
                        else:
                            opposing += 1

        total = same_dir + opposing
        lines.append(f"    {window:>5d}min {total:>9d} {same_dir:>9d} {opposing:>9d}")

    return "\n".join(lines)
