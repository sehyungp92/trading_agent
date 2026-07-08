"""Short-hold deep dive analysis for ALCB P8 diagnostics.

Examines why 1-6 bar holds (0-30min) have -31.09R total despite 68% WR.
The losses must be very large — this script dissects the 132 losers in
that bucket for common patterns and suggests mitigation.

Usage:
    from backtests.stock.analysis.alcb_short_hold_analysis import (
        short_hold_deep_dive,
    )
    report = short_hold_deep_dive(trades)
    print(report)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _meta(t: TradeRecord, key: str, default=None):
    """Safe metadata access."""
    if t.metadata:
        return t.metadata.get(key, default)
    return default


def _group_stats(trades: list[TradeRecord]) -> dict:
    """Compute stats for a trade group."""
    if not trades:
        return {"n": 0, "wr": 0, "mean_r": 0, "median_r": 0, "total_r": 0, "pf": 0}
    wins = [t for t in trades if t.r_multiple > 0]
    r_vals = [t.r_multiple for t in trades]
    gross_w = sum(r for r in r_vals if r > 0)
    gross_l = abs(sum(r for r in r_vals if r < 0))
    return {
        "n": len(trades),
        "wr": len(wins) / len(trades),
        "mean_r": float(np.mean(r_vals)),
        "median_r": float(np.median(r_vals)),
        "total_r": sum(r_vals),
        "pf": gross_w / gross_l if gross_l > 0 else float("inf"),
    }


def _fmt(s: dict) -> str:
    pf_str = f"{s['pf']:.2f}" if s["pf"] < 100 else "inf"
    return (f"n={s['n']:>4d}  WR={s['wr']:.1%}  mean_R={s['mean_r']:+.3f}  "
            f"med_R={s['median_r']:+.3f}  total_R={s['total_r']:+.1f}  PF={pf_str}")


def short_hold_deep_dive(trades: list[TradeRecord]) -> str:
    """Deep analysis of short-hold trades (<=6 bars / 30min).

    Returns report covering:
    - Hold duration distribution
    - Short-hold losers: exit reasons, magnitude, common features
    - MFE analysis: did losers ever have positive excursion?
    - Comparison of quick exits vs longer holds
    - Mitigation recommendations
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("SHORT-HOLD DEEP DIVE ANALYSIS")
    lines.append("=" * 70)

    # --- Section 1: Hold duration distribution ---
    lines.append("\n--- Hold Duration Distribution ---")
    buckets = [
        ("0-2 bars (0-10min)", lambda t: t.hold_bars <= 2),
        ("3-6 bars (10-30min)", lambda t: 3 <= t.hold_bars <= 6),
        ("7-12 bars (30-60min)", lambda t: 7 <= t.hold_bars <= 12),
        ("13-24 bars (1-2h)", lambda t: 13 <= t.hold_bars <= 24),
        ("25-48 bars (2-4h)", lambda t: 25 <= t.hold_bars <= 48),
        (">48 bars (4h+)", lambda t: t.hold_bars > 48),
    ]

    for label, pred in buckets:
        group = [t for t in trades if pred(t)]
        s = _group_stats(group)
        lines.append(f"  {label:<25s} {_fmt(s)}")

    # --- Section 2: Short-hold trades (<=6 bars) ---
    short = [t for t in trades if t.hold_bars <= 6]
    short_w = [t for t in short if t.r_multiple > 0]
    short_l = [t for t in short if t.r_multiple <= 0]

    lines.append(f"\n--- Short Hold (<=6 bars) Overview ---")
    lines.append(f"  Total: {len(short)} trades")
    lines.append(f"  Winners: {len(short_w)} ({len(short_w)/len(short):.1%} WR)")
    lines.append(f"  Losers:  {len(short_l)}")

    if short_w:
        w_r = [t.r_multiple for t in short_w]
        lines.append(f"\n  Winner R profile:")
        lines.append(f"    Mean: {np.mean(w_r):+.3f}  Median: {np.median(w_r):+.3f}")
        lines.append(f"    Total: {sum(w_r):+.1f}")

    if short_l:
        l_r = [t.r_multiple for t in short_l]
        lines.append(f"\n  Loser R profile:")
        lines.append(f"    Mean: {np.mean(l_r):+.3f}  Median: {np.median(l_r):+.3f}")
        lines.append(f"    Total: {sum(l_r):+.1f}")
        lines.append(f"    >>> Losers contribute {sum(l_r):+.1f}R despite being "
                     f"only {len(short_l)}/{len(short)} trades")

    # --- Section 3: Short-hold loser exit reasons ---
    lines.append(f"\n--- Short Loser Exit Reasons ---")
    exit_groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in short_l:
        exit_groups[t.exit_reason or "UNKNOWN"].append(t)

    for reason in sorted(exit_groups, key=lambda r: len(exit_groups[r]), reverse=True):
        group = exit_groups[reason]
        s = _group_stats(group)
        lines.append(f"  {reason:<25s} {_fmt(s)}")

    # --- Section 4: Short-hold loser MFE analysis ---
    lines.append(f"\n--- Short Loser MFE (Did They Ever Have Profit?) ---")
    if short_l:
        mfe_vals = [_meta(t, "mfe_r", 0) for t in short_l]
        had_profit = [t for t in short_l if _meta(t, "mfe_r", 0) > 0.1]
        no_profit = [t for t in short_l if _meta(t, "mfe_r", 0) <= 0.1]

        lines.append(f"  Losers with MFE > 0.1R: {len(had_profit)} ({len(had_profit)/len(short_l):.1%})")
        lines.append(f"  Losers with MFE <= 0.1R: {len(no_profit)} ({len(no_profit)/len(short_l):.1%})")
        lines.append(f"  MFE distribution: P10={np.percentile(mfe_vals, 10):.3f} "
                     f"P50={np.percentile(mfe_vals, 50):.3f} "
                     f"P90={np.percentile(mfe_vals, 90):.3f}")

        if had_profit:
            lines.append(f"  Losers that had profit: avg MFE={np.mean([_meta(t, 'mfe_r', 0) for t in had_profit]):.3f}R")
            lines.append(f"    → These could be saved with tighter profit protection")

    # --- Section 5: Feature comparison (short losers vs rest) ---
    lines.append(f"\n--- Short Loser Features vs Winners ---")
    rest = [t for t in trades if t.hold_bars > 6]

    for feat_name, feat_fn in [
        ("RVOL", lambda t: _meta(t, "rvol_at_entry", 0)),
        ("Score", lambda t: _meta(t, "momentum_score", 0)),
        ("Entry Type", lambda t: _meta(t, "entry_type", "?")),
    ]:
        if feat_name in ("RVOL", "Score"):
            sl_vals = [feat_fn(t) for t in short_l]
            sw_vals = [feat_fn(t) for t in short_w] if short_w else [0]
            rest_vals = [feat_fn(t) for t in rest]
            lines.append(f"  {feat_name}:")
            lines.append(f"    Short losers: {np.mean(sl_vals):.2f}")
            lines.append(f"    Short winners: {np.mean(sw_vals):.2f}")
            lines.append(f"    Rest of trades: {np.mean(rest_vals):.2f}")
        else:
            dist: dict[str, int] = defaultdict(int)
            for t in short_l:
                dist[feat_fn(t)] += 1
            lines.append(f"  {feat_name} distribution (short losers):")
            for val in sorted(dist, key=dist.get, reverse=True):
                lines.append(f"    {val}: {dist[val]} ({dist[val]/len(short_l):.0%})")

    # --- Section 6: Entry time analysis ---
    lines.append(f"\n--- Short Loser Entry Time ---")
    from zoneinfo import ZoneInfo
    et_tz = ZoneInfo("America/New_York")
    time_buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in short_l:
        try:
            et = t.entry_time.astimezone(et_tz)
            bucket = f"{et.hour:02d}:{(et.minute // 30) * 30:02d}"
            time_buckets[bucket].append(t)
        except Exception:
            pass

    for bucket in sorted(time_buckets):
        group = time_buckets[bucket]
        r_total = sum(t.r_multiple for t in group)
        lines.append(f"  {bucket}: {len(group)} trades, total_R={r_total:+.1f}")

    # --- Section 7: Mitigation recommendations ---
    lines.append(f"\n--- Mitigation Recommendations ---")
    if short_l:
        avg_loser_r = float(np.mean([t.r_multiple for t in short_l]))
        total_drag = sum(t.r_multiple for t in short_l)
        lines.append(f"  Short-hold loser drag: {total_drag:+.1f}R")
        lines.append(f"  Average loser R: {avg_loser_r:+.3f}")
        lines.append(f"\n  Options:")
        lines.append(f"  1. Quick exit after {6} bars if R < 0.2 (time-based stop)")
        lines.append(f"  2. Tighter stop for first 6 bars (reduce loss magnitude)")
        lines.append(f"  3. Higher entry bar minimum (skip first 1-2 bars)")

        # Estimate impact of quick exit
        would_save = [t for t in short_l if t.r_multiple < 0 and _meta(t, "mfe_r", 0) < 0.2]
        if would_save:
            saved_r = sum(t.r_multiple for t in would_save)
            lines.append(f"\n  Quick exit (6 bars, min 0.2R) would affect {len(would_save)} trades")
            lines.append(f"  Current cost of those trades: {saved_r:+.1f}R")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
