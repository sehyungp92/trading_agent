"""Priority 5: Day-of-Week Conditional Analysis.

Investigates Tuesday underperformance and day-of-week patterns
across exit types, sectors, and regime tiers.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


_DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def iaric_dow_conditional(trades: list[TradeRecord]) -> str:
    """Day-of-week conditional analysis with exit type attribution."""
    lines = [_hdr("DOW-1  Day-of-Week Conditional Analysis")]

    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)

    # Group by day of week (0=Mon, 4=Fri)
    by_dow: dict[int, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        dow = t.entry_time.weekday()
        if 0 <= dow <= 4:
            by_dow[dow].append(t)

    # --- Overview by day ---
    lines.append(f"\n  Day-of-Week Overview:")
    lines.append(f"    {'Day':<12s} {'n':>5s} {'WR':>6s} {'Mean R':>8s} {'Med R':>8s} {'PF':>6s} {'Total R':>9s} {'PnL':>10s}")
    lines.append(f"    {'-'*68}")
    for dow in range(5):
        group = by_dow.get(dow, [])
        if not group:
            continue
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        rs = [t.r_multiple for t in group]
        gross_p = sum(r for r in rs if r > 0)
        gross_l = abs(sum(r for r in rs if r < 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        pnl = sum(t.pnl_net for t in group)
        lines.append(
            f"    {_DOW_NAMES[dow]:<12s} {n:>5d} {wins/n:>5.1%} "
            f"{np.mean(rs):>+7.3f} {np.median(rs):>+7.3f} "
            f"{pf:>5.2f} {sum(rs):>+8.2f} {pnl:>+9,.0f}"
        )

    # --- Exit type breakdown by day ---
    lines.append(f"\n  Exit Type Distribution by Day:")
    exit_types = sorted({t.exit_reason for t in trades})
    header = f"    {'Day':<12s}" + "".join(f" {et[:12]:>12s}" for et in exit_types)
    lines.append(header)
    lines.append(f"    {'-'*(12 + 13 * len(exit_types))}")
    for dow in range(5):
        group = by_dow.get(dow, [])
        if not group:
            continue
        n = len(group)
        counts = defaultdict(int)
        for t in group:
            counts[t.exit_reason] += 1
        row = f"    {_DOW_NAMES[dow]:<12s}"
        for et in exit_types:
            c = counts.get(et, 0)
            row += f" {c:>5d}({c/n:>4.0%})"
            # Pad to 12 chars
        lines.append(row)

    # --- CLOSE_STOP rate by day ---
    lines.append(f"\n  CLOSE_STOP Rate by Day (is Tuesday losing from more stops?):")
    for dow in range(5):
        group = by_dow.get(dow, [])
        if not group:
            continue
        n = len(group)
        stops = sum(1 for t in group if t.exit_reason == "CLOSE_STOP")
        stop_pnl = sum(t.pnl_net for t in group if t.exit_reason == "CLOSE_STOP")
        lines.append(f"    {_DOW_NAMES[dow]:<12s}: {stops:>3d}/{n:<3d} ({stops/n:.1%}) stop loss PnL: ${stop_pnl:+,.0f}")

    # --- FLOW_REVERSAL rate by day ---
    lines.append(f"\n  FLOW_REVERSAL Rate by Day (is Tuesday missing flow reversals?):")
    for dow in range(5):
        group = by_dow.get(dow, [])
        if not group:
            continue
        n = len(group)
        frs = [t for t in group if t.exit_reason == "FLOW_REVERSAL"]
        fr_pnl = sum(t.pnl_net for t in frs)
        lines.append(f"    {_DOW_NAMES[dow]:<12s}: {len(frs):>3d}/{n:<3d} ({len(frs)/n:.1%}) FR PnL: ${fr_pnl:+,.0f}")

    # --- Sector mix by day (is Tuesday overweight weak sectors?) ---
    lines.append(f"\n  Sector Mix by Day:")
    for dow in range(5):
        group = by_dow.get(dow, [])
        if not group:
            continue
        sector_counts: dict[str, int] = defaultdict(int)
        for t in group:
            sector_counts[t.sector] += 1
        top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        mix_str = ", ".join(f"{s}({c})" for s, c in top_sectors)
        lines.append(f"    {_DOW_NAMES[dow]:<12s}: {mix_str}")

    # --- Regime tier by day ---
    lines.append(f"\n  Regime Tier Distribution by Day:")
    for dow in range(5):
        group = by_dow.get(dow, [])
        if not group:
            continue
        n = len(group)
        tier_a = sum(1 for t in group if t.regime_tier == "A")
        tier_b = sum(1 for t in group if t.regime_tier == "B")
        lines.append(f"    {_DOW_NAMES[dow]:<12s}: A={tier_a} ({tier_a/n:.0%}), B={tier_b} ({tier_b/n:.0%})")

    # --- Mean R by day × exit type ---
    lines.append(f"\n  Mean R by Day × Exit Type:")
    lines.append(f"    {'Day':<12s}" + "".join(f" {et[:12]:>12s}" for et in exit_types))
    lines.append(f"    {'-'*(12 + 13 * len(exit_types))}")
    for dow in range(5):
        group = by_dow.get(dow, [])
        if not group:
            continue
        by_exit: dict[str, list[float]] = defaultdict(list)
        for t in group:
            by_exit[t.exit_reason].append(t.r_multiple)
        row = f"    {_DOW_NAMES[dow]:<12s}"
        for et in exit_types:
            rs = by_exit.get(et, [])
            if rs:
                row += f" {np.mean(rs):>+11.3f} "
            else:
                row += f" {'---':>12s}"
        lines.append(row)

    return "\n".join(lines)
