"""VdubusNQ v4.0 gating attribution — the critical deliverable.

Per-gate rejection analysis answering: "Is this filter gating profitable trades?"

For each gate:
- Count blocked
- % that would have hit +1R
- Avg virtual R-multiple
- Net virtual EV: missed winners - avoided losers
- Verdict: KEEP (net EV < 0) vs REVIEW (net EV > 0)
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from backtests.momentum.analysis.vdubus_shadow_tracker import VdubusFilterStats, VdubusShadowTracker
from backtests.momentum.engine.vdubus_engine import VdubusSignalEvent


def vdubus_filter_attribution_report(
    signal_events: list[VdubusSignalEvent],
    trades: list,
    shadow_tracker: VdubusShadowTracker | None = None,
) -> str:
    """Generate the gating attribution report.

    If shadow_tracker is provided, includes virtual R outcomes.
    Otherwise, reports only rejection counts from signal events.
    """
    lines = ["=" * 60]
    lines.append("  VdubusNQ v4.0 GATING ATTRIBUTION REPORT")
    lines.append("=" * 60)
    lines.append("")

    blocked = [e for e in signal_events if not e.passed_all]
    passed = [e for e in signal_events if e.passed_all]

    lines.append(f"  Total 15m evaluations:     {len(signal_events)}")
    lines.append(f"  Passed all gates:          {len(passed)}")
    lines.append(f"  Blocked by gates:          {len(blocked)}")
    lines.append(f"  Completed trades:          {len(trades)}")
    lines.append("")

    reasons = Counter(e.first_block_reason for e in blocked)

    if shadow_tracker and shadow_tracker.results:
        summaries = shadow_tracker.get_filter_summary()

        lines.append("  Per-Gate Attribution (with shadow trade simulation):")
        lines.append("")
        header = (
            f"  {'Gate':24s} {'Blocked':>7s} {'WouldFill':>9s} {'AvgR':>7s} "
            f"{'1R%':>5s} {'>1R%':>5s} {'NetEV':>8s} {'Verdict':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        all_gates = sorted(
            set(list(reasons.keys()) + list(summaries.keys())),
            key=lambda g: -reasons.get(g, 0),
        )

        for gate in all_gates:
            count = reasons.get(gate, 0)
            s = summaries.get(gate)
            if s and s.filled_count > 0:
                net_ev = s.net_missed_ev - s.net_avoided_loss
                verdict = "KEEP" if net_ev < 0 else "REVIEW"
                lines.append(
                    f"  {gate:24s} {count:7d} {s.filled_count:9d} "
                    f"{s.avg_shadow_r:+7.3f} "
                    f"{s.pct_reach_1r:4.0f}% {s.pct_above_1r:4.0f}% "
                    f"{net_ev:+8.1f} {verdict:>8s}"
                )
            else:
                lines.append(
                    f"  {gate:24s} {count:7d}       N/A     N/A   N/A   N/A      N/A"
                )

        lines.append("")
        lines.append("  Verdict interpretation:")
        lines.append("    KEEP   = gate prevents more loss than it misses (net EV < 0)")
        lines.append("    REVIEW = gate may be blocking profitable trades (net EV > 0)")

    else:
        lines.append("  Per-Gate Rejection Counts:")
        lines.append("")
        header = f"  {'Gate':24s} {'Blocked':>7s} {'% of Total':>10s}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        total_blocked = len(blocked)
        for gate, count in reasons.most_common():
            pct = count / total_blocked * 100 if total_blocked > 0 else 0
            lines.append(f"  {gate:24s} {count:7d} {pct:9.1f}%")

        lines.append("")
        lines.append("  (Enable shadow tracking for full virtual R analysis)")

    # Session breakdown of rejections
    lines.append("")
    lines.append("  Rejections by Session:")
    sess_counts = Counter(e.session for e in blocked)
    for sess, cnt in sess_counts.most_common():
        lines.append(f"    {sess}: {cnt}")

    # Sub-window breakdown
    lines.append("")
    lines.append("  Rejections by Sub-Window:")
    sw_counts = Counter(e.sub_window for e in blocked)
    for sw, cnt in sw_counts.most_common():
        lines.append(f"    {sw}: {cnt}")

    # Direction breakdown
    lines.append("")
    lines.append("  Rejections by Direction:")
    dir_counts = Counter("LONG" if e.direction == 1 else "SHORT" for e in blocked)
    for d, cnt in dir_counts.most_common():
        lines.append(f"    {d}: {cnt}")

    return "\n".join(lines)
