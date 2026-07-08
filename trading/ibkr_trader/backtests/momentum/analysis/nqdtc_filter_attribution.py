"""NQDTC v2.0 gating attribution — the critical deliverable.

Per-gate rejection analysis answering: "Is this filter gating profitable trades?"

For each gate:
- Count blocked
- % that would have hit TP1/TP2
- Avg virtual R-multiple
- Virtual max DD contribution
- Net virtual EV: "Gate X blocked 120 trades; 40 would have been winners; net EV was +0.12R"
"""
from __future__ import annotations

import numpy as np

from backtests.momentum.analysis.nqdtc_shadow_tracker import NQDTCFilterStats, NQDTCShadowTracker
from backtests.momentum.engine.nqdtc_engine import NQDTCSignalEvent


def nqdtc_filter_attribution_report(
    signal_events: list[NQDTCSignalEvent],
    trades: list,
    shadow_tracker: NQDTCShadowTracker | None = None,
) -> str:
    """Generate the gating attribution report.

    If shadow_tracker is provided, includes virtual R outcomes.
    Otherwise, reports only rejection counts from signal events.
    """
    lines = ["=" * 60]
    lines.append("  NQDTC v2.0 GATING ATTRIBUTION REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Rejection counts from signal events
    blocked = [e for e in signal_events if not e.passed_all]
    passed = [e for e in signal_events if e.passed_all]

    lines.append(f"  Total 30m evaluations:     {len(signal_events)}")
    lines.append(f"  Passed all gates:          {len(passed)}")
    lines.append(f"  Blocked by gates:          {len(blocked)}")
    lines.append(f"  Completed trades:          {len(trades)}")
    lines.append("")

    # Per-gate rejection table
    from collections import Counter
    reasons = Counter(e.first_block_reason for e in blocked)

    if shadow_tracker and shadow_tracker.results:
        summaries = shadow_tracker.get_filter_summary()

        lines.append("  Per-Gate Attribution (with shadow trade simulation):")
        lines.append("")
        header = (
            f"  {'Gate':24s} {'Blocked':>7s} {'WouldFill':>9s} {'AvgR':>7s} "
            f"{'TP1%':>5s} {'>1R%':>5s} {'NetEV':>8s} {'Verdict':>8s}"
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
                # Net EV = missed winners - avoided losers
                net_ev = s.net_missed_ev - s.net_avoided_loss
                verdict = "KEEP" if net_ev < 0 else "REVIEW"
                lines.append(
                    f"  {gate:24s} {count:7d} {s.filled_count:9d} "
                    f"{s.avg_shadow_r:+7.3f} "
                    f"{s.pct_reach_tp1:4.0f}% {s.pct_above_1r:4.0f}% "
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
        # Without shadow trades, just show rejection counts
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

    # Regime breakdown of rejections
    lines.append("")
    lines.append("  Rejections by Regime:")
    regime_counts = Counter(e.composite_regime for e in blocked)
    for regime, cnt in regime_counts.most_common():
        lines.append(f"    {regime}: {cnt}")

    return "\n".join(lines)
