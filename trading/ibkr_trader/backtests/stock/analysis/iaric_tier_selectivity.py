"""Priority 7: Tier A Selectivity Test.

Tests whether tightening Tier A selection would improve alpha,
motivated by the Tier B paradox (0.717R vs 0.480R).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def iaric_tier_selectivity(trades: list[TradeRecord]) -> str:
    """Analyze whether tighter Tier A selection improves alpha."""
    lines = [_hdr("SEL-1  Tier A Selectivity Test")]

    tier_a = [t for t in trades if t.regime_tier == "A"]
    tier_b = [t for t in trades if t.regime_tier == "B"]

    if not tier_a:
        lines.append("  No Tier A trades found.")
        return "\n".join(lines)

    # --- Current tier comparison ---
    lines.append(f"  Current Tier Performance:")
    for label, group in [("Tier A", tier_a), ("Tier B", tier_b)]:
        if not group:
            continue
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        rs = [t.r_multiple for t in group]
        pnl = sum(t.pnl_net for t in group)
        gross_p = sum(r for r in rs if r > 0)
        gross_l = abs(sum(r for r in rs if r < 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        lines.append(
            f"    {label}: n={n}, WR={wins/n:.1%}, Mean R={np.mean(rs):+.3f}, "
            f"PF={pf:.2f}, PnL=${pnl:+,.0f}"
        )

    lines.append(f"\n  Tier B paradox: higher avg R despite 'worse' market conditions.")
    lines.append(f"  Hypothesis: Tier B's position cap forces stronger selectivity.")

    # --- Conviction multiplier distribution in Tier A ---
    lines.append(f"\n  Tier A Conviction Multiplier Distribution:")
    conv_mults = [t.metadata.get("conviction_multiplier", 1.0) if t.metadata else 1.0 for t in tier_a]
    if conv_mults:
        arr = np.array(conv_mults)
        for pct, label in [(10, "P10"), (25, "P25"), (50, "P50"), (75, "P75"), (90, "P90")]:
            lines.append(f"    {label}: {np.percentile(arr, pct):.3f}")

    # --- Top-N simulation within Tier A ---
    lines.append(f"\n  Top-N Selectivity Simulation (Tier A only):")
    lines.append(f"  Simulates keeping only top-N trades per day by conviction multiplier.")
    lines.append(f"    {'Max N':>6s} {'Trades':>7s} {'WR':>6s} {'Mean R':>8s} {'PF':>6s} {'PnL':>10s} {'vs All':>8s}")
    lines.append(f"    {'-'*56}")

    # Group tier A trades by date
    by_date: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in tier_a:
        date_key = t.entry_time.strftime("%Y-%m-%d")
        by_date[date_key].append(t)

    all_tier_a_pnl = sum(t.pnl_net for t in tier_a)

    for max_n in [2, 3, 4, 5, 6, 8]:
        selected: list[TradeRecord] = []
        for date_key, day_trades in by_date.items():
            # Sort by conviction multiplier descending, take top N
            sorted_trades = sorted(
                day_trades,
                key=lambda t: t.metadata.get("conviction_multiplier", 0) if t.metadata else 0,
                reverse=True,
            )
            selected.extend(sorted_trades[:max_n])

        if not selected:
            continue

        n = len(selected)
        wins = sum(1 for t in selected if t.is_winner)
        rs = [t.r_multiple for t in selected]
        pnl = sum(t.pnl_net for t in selected)
        gross_p = sum(r for r in rs if r > 0)
        gross_l = abs(sum(r for r in rs if r < 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        delta = pnl - all_tier_a_pnl
        lines.append(
            f"    {max_n:>6d} {n:>7d} {wins/n:>5.1%} "
            f"{np.mean(rs):>+7.3f} {pf:>5.2f} {pnl:>+9,.0f} {delta:>+7,.0f}"
        )

    # --- Conviction multiplier threshold test ---
    lines.append(f"\n  Conviction Multiplier Threshold Test (Tier A):")
    lines.append(f"  Simulates requiring minimum conviction_multiplier to trade.")
    lines.append(f"    {'Min Conv':>9s} {'Trades':>7s} {'WR':>6s} {'Mean R':>8s} {'PF':>6s} {'PnL':>10s}")
    lines.append(f"    {'-'*48}")

    for min_conv in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        selected = [
            t for t in tier_a
            if (t.metadata.get("conviction_multiplier", 0) if t.metadata else 0) >= min_conv
        ]
        if not selected:
            continue
        n = len(selected)
        wins = sum(1 for t in selected if t.is_winner)
        rs = [t.r_multiple for t in selected]
        pnl = sum(t.pnl_net for t in selected)
        gross_p = sum(r for r in rs if r > 0)
        gross_l = abs(sum(r for r in rs if r < 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
        lines.append(
            f"    {min_conv:>9.2f} {n:>7d} {wins/n:>5.1%} "
            f"{np.mean(rs):>+7.3f} {pf:>5.2f} {pnl:>+9,.0f}"
        )

    # --- Sponsorship state selectivity ---
    lines.append(f"\n  Tier A by Sponsorship State:")
    by_spons: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in tier_a:
        state = t.metadata.get("sponsorship_state", "?") if t.metadata else "?"
        by_spons[state].append(t)
    for state in sorted(by_spons):
        group = by_spons[state]
        n = len(group)
        wins = sum(1 for t in group if t.is_winner)
        rs = [t.r_multiple for t in group]
        pnl = sum(t.pnl_net for t in group)
        lines.append(f"    {state:<15s}: n={n}, WR={wins/n:.1%}, Mean R={np.mean(rs):+.3f}, PnL=${pnl:+,.0f}")

    # --- What if Tier A had same position cap as Tier B? ---
    lines.append(f"\n  Position Count Distribution (Tier A per-day):")
    day_counts = [len(day_trades) for day_trades in by_date.values()]
    if day_counts:
        arr = np.array(day_counts)
        lines.append(f"    Mean positions/day: {np.mean(arr):.1f}")
        lines.append(f"    Median positions/day: {np.median(arr):.1f}")
        lines.append(f"    Max positions/day: {int(np.max(arr))}")
        for n_cap in [2, 3, 4, 5, 6]:
            over_cap = sum(1 for c in day_counts if c > n_cap)
            lines.append(f"    Days exceeding {n_cap} positions: {over_cap}/{len(day_counts)} ({over_cap/len(day_counts):.0%})")

    return "\n".join(lines)
