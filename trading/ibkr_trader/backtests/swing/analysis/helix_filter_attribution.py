"""Swing Helix filter attribution — shadow-based KEEP/REVIEW verdicts.

Consumes HelixShadowTracker output to evaluate whether each gate is
preventing more loss than it misses, or blocking profitable setups.
"""
from __future__ import annotations

import numpy as np
from collections import Counter

from backtests.swing.analysis.helix_shadow_tracker import FilterStats, HelixShadowTracker


def helix_filter_attribution_report(
    shadow_tracker: HelixShadowTracker,
    trades: list,
) -> str:
    """Generate filter attribution report with KEEP/REVIEW verdicts.

    Args:
        shadow_tracker: Populated tracker with rejections and simulation results.
        trades: Completed Helix trade records for baseline comparison.
    """
    lines = ["=" * 60]
    lines.append("  SWING HELIX FILTER ATTRIBUTION REPORT")
    lines.append("=" * 60)
    lines.append("")

    rejections = shadow_tracker.rejections
    results = shadow_tracker.results

    # Baseline stats from actual trades
    if trades:
        r_arr = np.array([getattr(t, 'r_multiple', 0.0) for t in trades])
        actual_wr = float(np.mean(r_arr > 0)) * 100
        actual_avg_r = float(np.mean(r_arr))
    else:
        actual_wr = 0.0
        actual_avg_r = 0.0

    lines.append(f"  Actual trades:             {len(trades)}")
    lines.append(f"  Actual WR%:                {actual_wr:.1f}%")
    lines.append(f"  Actual avg R:              {actual_avg_r:+.3f}")
    lines.append(f"  Total rejections tracked:  {len(rejections)}")
    lines.append(f"  Shadow simulations:        {len(results)}")
    lines.append("")

    # Rejection counts by filter
    rej_counts = Counter()
    for c in rejections:
        for name in c.filter_names:
            rej_counts[name] += 1

    if results:
        summaries = shadow_tracker.get_filter_summary()

        lines.append("  Per-Gate Attribution (shadow simulation):")
        lines.append("")
        header = (
            f"  {'Gate':30s} {'Blkd':>5s} {'Fill':>5s} {'WR%':>6s} "
            f"{'AvgR':>7s} {'>1R%':>5s} {'MissEV':>8s} "
            f"{'AvdLoss':>8s} {'NetEV':>8s} {'Verdict':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        all_gates = sorted(
            set(list(rej_counts.keys()) + list(summaries.keys())),
            key=lambda g: -rej_counts.get(g, 0),
        )

        keep_count = 0
        review_count = 0
        for gate in all_gates:
            count = rej_counts.get(gate, 0)
            s = summaries.get(gate)
            if s and s.filled_count > 0:
                net_ev = s.net_missed_expectancy - s.net_avoided_loss
                verdict = "KEEP" if net_ev < 0 else "REVIEW"
                if verdict == "KEEP":
                    keep_count += 1
                else:
                    review_count += 1

                # Shadow win rate
                filled_for_gate = [
                    r for r in results
                    if r.filled and gate in r.candidate.filter_names
                ]
                shadow_wr = 0.0
                if filled_for_gate:
                    shadow_wr = sum(1 for r in filled_for_gate if r.r_multiple > 0) / len(filled_for_gate) * 100

                lines.append(
                    f"  {gate:30s} {count:5d} {s.filled_count:5d} "
                    f"{shadow_wr:5.1f}% {s.avg_shadow_r:+7.3f} "
                    f"{s.pct_above_1r:4.0f}% "
                    f"{s.net_missed_expectancy:+8.1f} {s.net_avoided_loss:8.1f} "
                    f"{net_ev:+8.1f} {verdict:>8s}"
                )
            else:
                lines.append(
                    f"  {gate:30s} {count:5d}   N/A    N/A     N/A"
                    f"   N/A      N/A      N/A      N/A"
                )

        lines.append("")
        lines.append(f"  Summary: {keep_count} KEEP, {review_count} REVIEW gates")
        lines.append("")
        lines.append("  Verdict logic:")
        lines.append("    KEEP   = net EV < 0 → gate prevents more loss than it blocks")
        lines.append("    REVIEW = net EV >= 0 → gate may be over-filtering profitable setups")

    else:
        # No simulation results — show rejection counts only
        lines.append("  Per-Gate Rejection Counts (no shadow simulation):")
        lines.append("")
        header = f"  {'Gate':30s} {'Blocked':>7s} {'%':>6s}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        total = sum(rej_counts.values())
        for gate, count in rej_counts.most_common():
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {gate:30s} {count:7d} {pct:5.1f}%")

        lines.append("")
        lines.append("  (Run shadow simulation for KEEP/REVIEW verdicts)")

    # Breakdown by setup class
    lines.append("")
    lines.append("  Rejections by Setup Class:")
    cls_counts = Counter(c.setup_class for c in rejections)
    for cls, cnt in cls_counts.most_common():
        pct = cnt / len(rejections) * 100 if rejections else 0
        lines.append(f"    {cls or 'unknown':12s}: {cnt:5d} ({pct:.1f}%)")

    # Breakdown by symbol
    lines.append("")
    lines.append("  Rejections by Symbol:")
    sym_counts = Counter(c.symbol for c in rejections)
    for sym, cnt in sym_counts.most_common():
        pct = cnt / len(rejections) * 100 if rejections else 0
        lines.append(f"    {sym:12s}: {cnt:5d} ({pct:.1f}%)")

    # Breakdown by direction
    lines.append("")
    lines.append("  Rejections by Direction:")
    dir_counts = Counter("LONG" if c.direction == 1 else "SHORT" for c in rejections)
    for d_label, cnt in dir_counts.most_common():
        lines.append(f"    {d_label}: {cnt}")

    # Breakdown by origin timeframe
    lines.append("")
    lines.append("  Rejections by Origin TF:")
    tf_counts = Counter(c.origin_tf for c in rejections)
    for tf, cnt in tf_counts.most_common():
        lines.append(f"    {tf}: {cnt}")

    return "\n".join(lines)
