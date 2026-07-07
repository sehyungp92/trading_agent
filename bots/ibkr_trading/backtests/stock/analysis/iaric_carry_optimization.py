"""Priority 6: Carry Optimization Analysis.

Determines if more aggressive overnight carry would improve results
by analyzing EOD_FLATTEN trades that were carry-eligible.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def iaric_carry_optimization(trades: list[TradeRecord]) -> str:
    """Analyze carry utilization and optimization opportunities."""
    lines = [_hdr("CARRY-1  Carry Optimization Analysis")]

    carry_trades = [t for t in trades if t.exit_reason == "CARRY_EXIT"]
    eod_trades = [t for t in trades if t.exit_reason == "EOD_FLATTEN"]
    fr_trades = [t for t in trades if t.exit_reason == "FLOW_REVERSAL"]

    n_total = len(trades)
    if n_total == 0:
        lines.append("  No trades.")
        return "\n".join(lines)

    # --- Current carry stats ---
    lines.append(f"  Current Carry Usage:")
    lines.append(f"    Carry exits: {len(carry_trades)} ({len(carry_trades)/n_total:.1%} of trades)")
    lines.append(f"    EOD flattens: {len(eod_trades)} ({len(eod_trades)/n_total:.1%})")
    lines.append(f"    Flow reversals: {len(fr_trades)} ({len(fr_trades)/n_total:.1%})")

    if carry_trades:
        carry_rs = [t.r_multiple for t in carry_trades]
        carry_pnl = sum(t.pnl_net for t in carry_trades)
        lines.append(f"    Carry WR: {sum(1 for t in carry_trades if t.is_winner)/len(carry_trades):.1%}")
        lines.append(f"    Carry Mean R: {np.mean(carry_rs):+.3f}")
        lines.append(f"    Carry Total PnL: ${carry_pnl:+,.0f}")

    # --- EOD_FLATTEN trades by unrealized R ---
    lines.append(f"\n  EOD_FLATTEN Trades by Unrealized R at Close:")
    if eod_trades:
        eod_rs = [t.r_multiple for t in eod_trades]
        arr = np.array(eod_rs)

        # Categorize by R at exit
        deep_winners = [t for t in eod_trades if t.r_multiple >= 1.0]
        moderate_winners = [t for t in eod_trades if 0.5 <= t.r_multiple < 1.0]
        small_winners = [t for t in eod_trades if 0.0 < t.r_multiple < 0.5]
        losers = [t for t in eod_trades if t.r_multiple <= 0]

        lines.append(f"    Deep winners (>=1.0R): {len(deep_winners)} — carry-prime candidates")
        lines.append(f"    Moderate winners (0.5-1.0R): {len(moderate_winners)} — carry candidates")
        lines.append(f"    Small winners (0-0.5R): {len(small_winners)} — marginal")
        lines.append(f"    Losers (<=0R): {len(losers)} — not carry-eligible")

        carry_eligible = [t for t in eod_trades if t.r_multiple >= 0.5]
        if carry_eligible:
            ce_pnl = sum(t.pnl_net for t in carry_eligible)
            lines.append(f"\n    Carry-eligible (>= 0.5R) EOD flattens: {len(carry_eligible)}")
            lines.append(f"    Their total PnL: ${ce_pnl:+,.0f}")
            if carry_trades:
                lines.append(f"    If these carried with avg carry R ({np.mean([t.r_multiple for t in carry_trades]):+.3f}R): potential additional alpha")

        # R distribution
        lines.append(f"\n    EOD_FLATTEN R Distribution:")
        for pct, label in [(10, "P10"), (25, "P25"), (50, "P50"), (75, "P75"), (90, "P90")]:
            lines.append(f"      {label}: {np.percentile(arr, pct):+.3f}R")

    # --- Carry-eligible by sponsorship state ---
    lines.append(f"\n  Carry Eligibility by Sponsorship State:")
    lines.append(f"    (Carry requires STRONG sponsorship + Tier A regime)")
    for exit_type, group_label in [("EOD_FLATTEN", "EOD"), ("CARRY_EXIT", "Carry"), ("FLOW_REVERSAL", "FR")]:
        group = [t for t in trades if t.exit_reason == exit_type]
        if not group:
            continue
        by_spons: dict[str, int] = defaultdict(int)
        for t in group:
            state = t.metadata.get("sponsorship_state", "?") if t.metadata else "?"
            by_spons[state] += 1
        mix = ", ".join(f"{s}={c}" for s, c in sorted(by_spons.items()))
        lines.append(f"    {group_label}: {mix}")

    # --- Carry by regime tier ---
    lines.append(f"\n  Carry vs EOD by Regime Tier:")
    for tier in ["A", "B"]:
        carry_in_tier = [t for t in carry_trades if t.regime_tier == tier]
        eod_in_tier = [t for t in eod_trades if t.regime_tier == tier]
        lines.append(f"    Tier {tier}: {len(carry_in_tier)} carry, {len(eod_in_tier)} EOD_FLATTEN")

    # --- Hold duration of carry trades ---
    if carry_trades:
        lines.append(f"\n  Carry Hold Duration:")
        hours = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in carry_trades]
        arr = np.array(hours)
        lines.append(f"    Mean: {np.mean(arr):.1f}h  |  Median: {np.median(arr):.1f}h")
        lines.append(f"    Min: {np.min(arr):.1f}h  |  Max: {np.max(arr):.1f}h")
        one_day = sum(1 for h in hours if h < 30)
        multi_day = sum(1 for h in hours if h >= 30)
        lines.append(f"    1-day carry: {one_day}  |  Multi-day: {multi_day}")

    # --- Carry R vs EOD R comparison ---
    if carry_trades and eod_trades:
        lines.append(f"\n  Carry vs EOD Performance:")
        carry_mean = np.mean([t.r_multiple for t in carry_trades])
        eod_mean = np.mean([t.r_multiple for t in eod_trades])
        lines.append(f"    Carry Mean R: {carry_mean:+.3f} (n={len(carry_trades)})")
        lines.append(f"    EOD Mean R: {eod_mean:+.3f} (n={len(eod_trades)})")
        lines.append(f"    Delta: {carry_mean - eod_mean:+.3f}R")

    # --- FR trades that could have been carry ---
    lines.append(f"\n  FLOW_REVERSAL Trades (multi-day holds suggest successful carry):")
    fr_multiday = [t for t in fr_trades if (t.exit_time - t.entry_time).total_seconds() > 24 * 3600]
    fr_sameday = [t for t in fr_trades if (t.exit_time - t.entry_time).total_seconds() <= 24 * 3600]
    if fr_multiday:
        lines.append(f"    Multi-day FR: {len(fr_multiday)}, Mean R={np.mean([t.r_multiple for t in fr_multiday]):+.3f}")
    if fr_sameday:
        lines.append(f"    Same-day FR: {len(fr_sameday)}, Mean R={np.mean([t.r_multiple for t in fr_sameday]):+.3f}")

    return "\n".join(lines)
