"""ALCB filter attribution — per-gate KEEP/REVIEW verdicts.

Uses shadow tracker results to determine which gates are filtering
profitably (blocking losers) vs. filtering unprofitably (blocking winners).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker


@dataclass
class FilterStats:
    """Per-gate statistics from shadow simulation."""

    gate: str
    blocked_count: int = 0
    simulated_count: int = 0
    shadow_win_pct: float = 0.0
    shadow_avg_r: float = 0.0
    shadow_pct_reach_target: float = 0.0
    net_missed_ev: float = 0.0     # sum of positive shadow R (what we missed)
    net_avoided_loss: float = 0.0  # sum of negative shadow R (what we dodged)
    verdict: str = ""


def compute_filter_stats(tracker: ALCBShadowTracker) -> list[FilterStats]:
    """Compute per-gate statistics from completed shadow setups."""
    by_gate = tracker.get_filter_summary()
    results: list[FilterStats] = []

    for gate, shadows in sorted(by_gate.items()):
        fs = FilterStats(gate=gate, blocked_count=len(shadows))

        simulated = [s for s in shadows if s.simulated_exit]
        fs.simulated_count = len(simulated)

        if not simulated:
            results.append(fs)
            continue

        rs = [s.simulated_r for s in simulated]
        fs.shadow_avg_r = float(np.mean(rs))
        fs.shadow_win_pct = sum(1 for r in rs if r > 0) / len(rs)
        fs.shadow_pct_reach_target = sum(
            1 for s in simulated if s.simulated_exit == "TARGET_HIT"
        ) / len(simulated)
        fs.net_missed_ev = sum(r for r in rs if r > 0)
        fs.net_avoided_loss = sum(r for r in rs if r < 0)

        results.append(fs)

    return results


def alcb_filter_attribution_report(
    shadow_tracker: ALCBShadowTracker,
    actual_win_rate: float,
) -> str:
    """Generate per-gate KEEP/REVIEW verdict table.

    KEEP: gate's shadow_avg_r <= 0 OR shadow_win_pct <= actual_win_rate + 5%
    REVIEW: shadow_avg_r > 0 AND shadow_win_pct > actual_win_rate + 5%
    """
    stats = compute_filter_stats(shadow_tracker)
    if not stats:
        return "  No filter attribution data available."

    lines = [
        "",
        "=" * 70,
        "  Filter Attribution (Shadow Simulation)",
        "=" * 70,
        "",
        f"  {'Gate':<20s} {'Blocked':>7} {'Sim':>5} {'WR%':>6} {'Avg R':>7}"
        f" {'Tgt%':>6} {'Missed':>8} {'Avoided':>8} {'Verdict':>8}",
        "  " + "-" * 68,
    ]

    threshold = actual_win_rate + 0.05

    for fs in stats:
        # Determine verdict
        if fs.simulated_count == 0:
            fs.verdict = "N/A"
        elif fs.shadow_avg_r > 0 and fs.shadow_win_pct > threshold:
            fs.verdict = "REVIEW"
        else:
            fs.verdict = "KEEP"

        lines.append(
            f"  {fs.gate:<20s} {fs.blocked_count:>7} {fs.simulated_count:>5}"
            f" {fs.shadow_win_pct:>5.0%} {fs.shadow_avg_r:>7.2f}"
            f" {fs.shadow_pct_reach_target:>5.0%} {fs.net_missed_ev:>8.1f}"
            f" {fs.net_avoided_loss:>8.1f} {fs.verdict:>8}"
        )

    lines.append("")
    lines.append("  Verdict: KEEP = gate is filtering profitably, REVIEW = gate may be too aggressive")
    lines.append(f"  Threshold: shadow WR > {threshold:.0%} AND shadow avg R > 0 → REVIEW")

    # Funnel report
    lines.append("")
    lines.append(shadow_tracker.funnel_report())

    return "\n".join(lines)
