"""Priority 3: FLOW_REVERSAL Attribution Analysis.

Analyzes why FLOW_REVERSAL exits are so profitable and how they
differ from other exit types.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _hdr(title: str) -> str:
    return f"\n{'='*70}\n  {title}\n{'='*70}"


def _group_stats_line(label: str, group: list[TradeRecord]) -> str:
    if not group:
        return f"    {label}: (no trades)"
    n = len(group)
    wins = sum(1 for t in group if t.is_winner)
    rs = [t.r_multiple for t in group]
    total_pnl = sum(t.pnl_net for t in group)
    return (
        f"    {label}: n={n}, WR={wins/n:.1%}, "
        f"Mean R={np.mean(rs):+.3f}, Total PnL=${total_pnl:,.0f}"
    )


def iaric_flow_reversal_attribution(trades: list[TradeRecord]) -> str:
    """Deep analysis of FLOW_REVERSAL exit mechanism."""
    lines = [_hdr("FR-1  FLOW_REVERSAL Attribution Analysis")]

    fr_trades = [t for t in trades if t.exit_reason == "FLOW_REVERSAL"]
    eod_trades = [t for t in trades if t.exit_reason == "EOD_FLATTEN"]
    stop_trades = [t for t in trades if t.exit_reason == "CLOSE_STOP"]
    carry_trades = [t for t in trades if t.exit_reason == "CARRY_EXIT"]

    if not fr_trades:
        lines.append("  No FLOW_REVERSAL trades found.")
        return "\n".join(lines)

    total_pnl = sum(t.pnl_net for t in trades)
    fr_pnl = sum(t.pnl_net for t in fr_trades)
    fr_total_r = sum(t.r_multiple for t in fr_trades)

    lines.append(f"  FLOW_REVERSAL: {len(fr_trades)} trades ({len(fr_trades)/len(trades):.1%} of total)")
    lines.append(f"  FR PnL: ${fr_pnl:,.2f} ({fr_pnl/total_pnl*100:.0f}% of net profit)" if total_pnl else "  FR PnL: ${fr_pnl:,.2f}")
    lines.append(f"  FR Total R: {fr_total_r:+.2f}")
    lines.append(f"  FR Win Rate: {sum(1 for t in fr_trades if t.is_winner)/len(fr_trades):.1%}")

    # --- R at flow reversal ---
    lines.append(f"\n  R-Multiple Distribution at FLOW_REVERSAL Exit:")
    fr_rs = [t.r_multiple for t in fr_trades]
    arr = np.array(fr_rs)
    for pct, label in [(10, "P10"), (25, "P25"), (50, "P50"), (75, "P75"), (90, "P90")]:
        lines.append(f"    {label}: {np.percentile(arr, pct):+.3f}R")
    lines.append(f"    Mean: {np.mean(arr):+.3f}R  |  Std: {np.std(arr):.3f}R")

    # --- MFE at flow reversal (how much edge was captured) ---
    lines.append(f"\n  MFE Capture Efficiency (how much of the move was captured):")
    capture_ratios = []
    for t in fr_trades:
        mfe = t.metadata.get("mfe_r", 0) if t.metadata else 0
        if mfe == 0 and t.risk_per_share > 0 and t.max_favorable > 0:
            mfe = (t.max_favorable - t.entry_price) / t.risk_per_share
        if mfe > 0:
            capture_ratios.append(t.r_multiple / mfe)
    if capture_ratios:
        arr = np.array(capture_ratios)
        lines.append(f"    Mean capture: {np.mean(arr):.1%} of MFE")
        lines.append(f"    Median capture: {np.median(arr):.1%} of MFE")
        lines.append(f"    P25 capture: {np.percentile(arr, 25):.1%}")

    # --- Hold duration comparison ---
    lines.append(f"\n  Hold Duration by Exit Type:")
    for label, group in [("FLOW_REVERSAL", fr_trades), ("EOD_FLATTEN", eod_trades),
                         ("CLOSE_STOP", stop_trades), ("CARRY_EXIT", carry_trades)]:
        if group:
            hours = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in group]
            lines.append(f"    {label:<20s}: Mean {np.mean(hours):5.1f}h, Median {np.median(hours):5.1f}h (n={len(group)})")

    # --- Sector breakdown of flow reversals ---
    lines.append(f"\n  FLOW_REVERSAL by Sector:")
    lines.append(f"    {'Sector':<20s} {'n':>4s} {'WR':>6s} {'Mean R':>8s} {'PnL':>10s}")
    lines.append(f"    {'-'*52}")
    by_sector: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in fr_trades:
        by_sector[t.sector].append(t)
    for sector in sorted(by_sector, key=lambda s: sum(t.r_multiple for t in by_sector[s]), reverse=True):
        group = by_sector[sector]
        n = len(group)
        wr = sum(1 for t in group if t.is_winner) / n
        mean_r = np.mean([t.r_multiple for t in group])
        pnl = sum(t.pnl_net for t in group)
        lines.append(f"    {sector:<20s} {n:>4d} {wr:>5.1%} {mean_r:>+7.3f} {pnl:>+9,.0f}")

    # --- Regime tier breakdown ---
    lines.append(f"\n  FLOW_REVERSAL by Regime Tier:")
    by_tier: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in fr_trades:
        by_tier[t.regime_tier].append(t)
    for tier in sorted(by_tier):
        lines.append(_group_stats_line(f"Tier {tier}", by_tier[tier]))

    # --- Conviction breakdown ---
    lines.append(f"\n  FLOW_REVERSAL by Conviction Bucket:")
    by_conv: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in fr_trades:
        bucket = t.metadata.get("conviction_bucket", "?") if t.metadata else "?"
        by_conv[bucket].append(t)
    for bucket in sorted(by_conv):
        lines.append(_group_stats_line(bucket, by_conv[bucket]))

    # --- Sponsorship state ---
    lines.append(f"\n  FLOW_REVERSAL by Sponsorship State:")
    by_spons: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in fr_trades:
        state = t.metadata.get("sponsorship_state", "?") if t.metadata else "?"
        by_spons[state].append(t)
    for state in sorted(by_spons):
        lines.append(_group_stats_line(state, by_spons[state]))

    # --- Comparison: exit type profitability summary ---
    lines.append(f"\n  Exit Type P&L Attribution:")
    lines.append(f"    {'Exit Type':<25s} {'n':>5s} {'WR':>6s} {'Total R':>9s} {'Total PnL':>11s} {'% of Net':>9s}")
    lines.append(f"    {'-'*68}")
    for label, group in [("FLOW_REVERSAL", fr_trades), ("EOD_FLATTEN", eod_trades),
                         ("CLOSE_STOP", stop_trades), ("CARRY_EXIT", carry_trades)]:
        if group:
            n = len(group)
            wr = sum(1 for t in group if t.is_winner) / n
            total_r = sum(t.r_multiple for t in group)
            pnl = sum(t.pnl_net for t in group)
            pct = pnl / total_pnl * 100 if total_pnl else 0
            lines.append(f"    {label:<25s} {n:>5d} {wr:>5.1%} {total_r:>+8.2f} {pnl:>+10,.0f} {pct:>+8.1f}%")

    return "\n".join(lines)
