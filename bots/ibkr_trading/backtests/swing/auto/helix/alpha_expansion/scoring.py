"""Immutable seven-component score for Helix alpha expansion.

The scale keeps headroom around the round 1 baseline instead of saturating
the strongest existing dimensions. Candidate screening uses the repo's fast
independent replay path; phase adoption and end-of-round diagnostics are
recomputed with synchronized shared-capital replay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backtests.swing.auto.helix.scoring import HelixMetrics

IMMUTABLE_SCORE_WEIGHTS: dict[str, float] = {
    "net_profit": 0.26,
    "frequency": 0.18,
    "pf": 0.15,
    "exit_efficiency": 0.12,
    "waste_ratio": 0.11,
    "tail_preservation": 0.10,
    "inv_dd": 0.08,
}

DEFAULT_HARD_REJECTS: dict[str, float] = {
    "min_trades": 320.0,
    "min_pf": 2.0,
    "max_r_dd": 12.0,
    "min_tail_pct": 0.50,
    "min_regime_pf": 1.50,
    "min_net_return_pct": 100.0,
    "min_avg_r": 0.35,
    "min_exit_efficiency": 0.32,
}


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _scale(value: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0
    return _clip01((value - floor) / (ceiling - floor))


@dataclass(frozen=True)
class AlphaExpansionScore:
    """Frozen score with exactly seven weighted components."""

    net_profit_component: float = 0.0
    frequency_component: float = 0.0
    pf_component: float = 0.0
    exit_efficiency_component: float = 0.0
    waste_ratio_component: float = 0.0
    tail_preservation_component: float = 0.0
    inv_dd_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


def alpha_expansion_score(
    metrics: HelixMetrics,
    hard_rejects: dict[str, float] | None = None,
) -> AlphaExpansionScore:
    """Score synchronized Helix metrics with immutable weights."""
    hr = dict(DEFAULT_HARD_REJECTS)
    hr.update(hard_rejects or {})

    avg_r = metrics.total_r / metrics.total_trades if metrics.total_trades else 0.0

    if metrics.total_trades < hr["min_trades"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=f"Too few trades: {metrics.total_trades} < {hr['min_trades']:.0f}",
        )
    if metrics.profit_factor < hr["min_pf"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=f"PF too low: {metrics.profit_factor:.2f} < {hr['min_pf']:.2f}",
        )
    if metrics.max_r_dd > hr["max_r_dd"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=f"Max R DD too high: {metrics.max_r_dd:.2f} > {hr['max_r_dd']:.2f}",
        )
    if metrics.tail_pct < hr["min_tail_pct"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=f"Tail preservation too low: {metrics.tail_pct:.2f} < {hr['min_tail_pct']:.2f}",
        )
    if metrics.min_regime_pf < hr["min_regime_pf"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=f"Min regime PF too low: {metrics.min_regime_pf:.2f} < {hr['min_regime_pf']:.2f}",
        )
    if metrics.net_return_pct < hr["min_net_return_pct"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=f"Return too low: {metrics.net_return_pct:.1f}% < {hr['min_net_return_pct']:.1f}%",
        )
    if avg_r < hr["min_avg_r"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=f"Average R too low: {avg_r:.2f} < {hr['min_avg_r']:.2f}",
        )
    if metrics.exit_efficiency < hr["min_exit_efficiency"]:
        return AlphaExpansionScore(
            rejected=True,
            reject_reason=(
                f"Exit efficiency too low: {metrics.exit_efficiency:.2f} "
                f"< {hr['min_exit_efficiency']:.2f}"
            ),
        )

    net_profit_component = _clip01(
        math.log1p(max(metrics.net_return_pct, 0.0) / 100.0) / math.log1p(3.0)
    )
    frequency_component = _scale(float(metrics.total_trades), 300.0, 520.0)
    pf_component = _scale(metrics.profit_factor, 1.8, 3.8)
    exit_efficiency_component = _scale(metrics.exit_efficiency, 0.35, 0.65)
    waste_ratio_component = _scale(metrics.waste_ratio, 0.55, 0.85)
    tail_preservation_component = _scale(metrics.tail_pct, 0.50, 0.82)
    inv_dd_component = _scale(12.0 - metrics.max_r_dd, 0.0, 8.0)

    w = IMMUTABLE_SCORE_WEIGHTS
    total = (
        w["net_profit"] * net_profit_component
        + w["frequency"] * frequency_component
        + w["pf"] * pf_component
        + w["exit_efficiency"] * exit_efficiency_component
        + w["waste_ratio"] * waste_ratio_component
        + w["tail_preservation"] * tail_preservation_component
        + w["inv_dd"] * inv_dd_component
    )

    return AlphaExpansionScore(
        net_profit_component=net_profit_component,
        frequency_component=frequency_component,
        pf_component=pf_component,
        exit_efficiency_component=exit_efficiency_component,
        waste_ratio_component=waste_ratio_component,
        tail_preservation_component=tail_preservation_component,
        inv_dd_component=inv_dd_component,
        total=total,
    )
