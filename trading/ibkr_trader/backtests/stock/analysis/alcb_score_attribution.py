"""Momentum score component attribution for ALCB P8 diagnostics.

Decomposes the momentum score into individual components and measures
each component's predictive power for trade outcome (R-multiple).
Answers: why is score 5 the best bucket, not 7?

Usage:
    from backtests.stock.analysis.alcb_score_attribution import (
        score_component_attribution,
    )
    report = score_component_attribution(trades)
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
        return {"n": 0, "wr": 0, "mean_r": 0, "total_r": 0, "pf": 0}
    wins = [t for t in trades if t.r_multiple > 0]
    wr = len(wins) / len(trades) if trades else 0
    r_vals = [t.r_multiple for t in trades]
    gross_w = sum(r for r in r_vals if r > 0)
    gross_l = abs(sum(r for r in r_vals if r < 0))
    return {
        "n": len(trades),
        "wr": wr,
        "mean_r": float(np.mean(r_vals)) if r_vals else 0,
        "total_r": sum(r_vals),
        "pf": gross_w / gross_l if gross_l > 0 else float("inf"),
    }


def _fmt_stats(s: dict) -> str:
    pf_str = f"{s['pf']:.2f}" if s["pf"] < 100 else "inf"
    return (f"  n={s['n']:>4d}  WR={s['wr']:.1%}  "
            f"mean_R={s['mean_r']:+.3f}  total_R={s['total_r']:+.1f}  PF={pf_str}")


def score_component_attribution(trades: list[TradeRecord]) -> str:
    """Analyze momentum score components for predictive power.

    Examines score_detail dict from each trade's MomentumSetup to decompose
    overall score into individual component contributions.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MOMENTUM SCORE COMPONENT ATTRIBUTION")
    lines.append("=" * 70)

    # --- Section 1: Score bucket performance ---
    lines.append("\n--- Score Bucket Performance ---")
    by_score: dict[int, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        score = _meta(t, "momentum_score", -1)
        if score >= 0:
            by_score[score].append(t)

    for score in sorted(by_score.keys()):
        s = _group_stats(by_score[score])
        lines.append(f"  Score {score}: {_fmt_stats(s)}")

    # --- Section 2: Component decomposition ---
    lines.append("\n--- Score Detail Component Analysis ---")
    lines.append("(Requires score_detail in trade metadata/momentum_setup)")

    component_r: dict[str, list[float]] = defaultdict(list)
    component_present: dict[str, list[float]] = defaultdict(list)
    component_absent: dict[str, list[float]] = defaultdict(list)
    n_with_detail = 0

    for t in trades:
        # Try to get score_detail from metadata or setup
        detail = _meta(t, "score_detail")
        if detail is None:
            setup = _meta(t, "momentum_setup")
            if hasattr(setup, "score_detail"):
                detail = setup.score_detail
        if detail is None:
            continue

        n_with_detail += 1
        r = t.r_multiple
        for comp_name, comp_val in detail.items():
            if isinstance(comp_val, (int, float)):
                component_r[comp_name].append((comp_val, r))
                if comp_val > 0:
                    component_present[comp_name].append(r)
                else:
                    component_absent[comp_name].append(r)

    lines.append(f"\n  Trades with score_detail: {n_with_detail}/{len(trades)}")

    if n_with_detail > 0:
        lines.append("\n  Component → Present vs Absent performance:")
        lines.append(f"  {'Component':<30s} {'Present':>8s} {'Absent':>8s} {'Delta':>8s} {'Corr':>8s}")
        lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for comp in sorted(component_present.keys()):
            present = component_present[comp]
            absent = component_absent[comp]
            avg_p = float(np.mean(present)) if present else 0
            avg_a = float(np.mean(absent)) if absent else 0
            delta = avg_p - avg_a

            # Correlation between component value and R
            pairs = component_r.get(comp, [])
            if len(pairs) >= 10:
                vals, rs = zip(*pairs)
                corr = float(np.corrcoef(vals, rs)[0, 1])
            else:
                corr = 0.0

            lines.append(
                f"  {comp:<30s} {avg_p:>+7.3f}R {avg_a:>+7.3f}R "
                f"{delta:>+7.3f}R {corr:>+7.3f}"
            )

        # --- Section 3: Optimal score threshold analysis ---
        lines.append("\n--- Optimal Score Threshold ---")
        lines.append("  Test: keeping only trades >= threshold")
        for thresh in range(2, 8):
            kept = [t for t in trades if _meta(t, "momentum_score", 0) >= thresh]
            if kept:
                s = _group_stats(kept)
                lines.append(f"  Score >= {thresh}: {_fmt_stats(s)}")

    # --- Section 4: Score × Entry Type ---
    lines.append("\n--- Score × Entry Type Cross-tab ---")
    entry_score: dict[str, dict[int, list[TradeRecord]]] = defaultdict(lambda: defaultdict(list))
    for t in trades:
        etype = _meta(t, "entry_type", t.entry_type or "UNKNOWN")
        score = _meta(t, "momentum_score", -1)
        if score >= 0:
            entry_score[etype][score].append(t)

    for etype in sorted(entry_score.keys()):
        lines.append(f"\n  {etype}:")
        for score in sorted(entry_score[etype].keys()):
            s = _group_stats(entry_score[etype][score])
            if s["n"] >= 5:
                lines.append(f"    Score {score}: {_fmt_stats(s)}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
