"""Seven-component score for Helix right-then-stopped leakage control."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from backtests.swing.auto.helix.scoring import HelixMetrics

IMMUTABLE_SCORE_WEIGHTS: dict[str, float] = {
    "net_profit": 0.34,
    "pf": 0.16,
    "frequency": 0.14,
    "leak_control": 0.14,
    "tail_preservation": 0.10,
    "short_hold_quality": 0.06,
    "inv_dd": 0.06,
}

DEFAULT_HARD_REJECTS: dict[str, float] = {
    "min_trades": 320.0,
    "min_pf": 2.0,
    "max_r_dd": 12.0,
    "min_tail_pct": 0.50,
    "min_regime_pf": 1.50,
    "min_net_return_pct": 100.0,
    "min_avg_r": 0.35,
}


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _scale(value: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0
    return _clip01((value - floor) / (ceiling - floor))


@dataclass(frozen=True)
class LeakageControlScore:
    """Frozen score with exactly seven weighted components."""

    net_profit_component: float = 0.0
    leak_control_component: float = 0.0
    pf_component: float = 0.0
    frequency_component: float = 0.0
    tail_preservation_component: float = 0.0
    short_hold_quality_component: float = 0.0
    inv_dd_component: float = 0.0
    total: float = 0.0
    rejected: bool = False
    reject_reason: str = ""


def extract_leakage_metrics(metrics: HelixMetrics, all_trades: list[Any]) -> dict[str, float]:
    """Compute leakage diagnostics used by the optimizer score."""
    right_then_lost = [
        trade for trade in all_trades
        if float(getattr(trade, "mfe_r", 0.0) or 0.0) >= 0.5
        and float(getattr(trade, "r_multiple", 0.0) or 0.0) < 0.0
    ]
    short_right_then_lost = [
        trade for trade in right_then_lost
        if int(getattr(trade, "bars_held", 0) or 0) <= 10
    ]
    mid_right_then_lost = [
        trade for trade in right_then_lost
        if 10 < int(getattr(trade, "bars_held", 0) or 0) <= 30
    ]

    right_then_loss_r = abs(sum(float(trade.r_multiple) for trade in right_then_lost))
    right_then_leak_r = sum(
        float(getattr(trade, "mfe_r", 0.0) or 0.0) - float(getattr(trade, "r_multiple", 0.0) or 0.0)
        for trade in right_then_lost
    )
    gross_win_r = max(float(metrics.gross_win_r), 1e-9)
    return {
        "right_then_lost_count": float(len(right_then_lost)),
        "right_then_loss_r": right_then_loss_r,
        "right_then_leak_r": right_then_leak_r,
        "right_then_leak_ratio": right_then_leak_r / gross_win_r,
        "right_then_short_count": float(len(short_right_then_lost)),
        "right_then_short_loss_r": abs(sum(float(trade.r_multiple) for trade in short_right_then_lost)),
        "right_then_mid_count": float(len(mid_right_then_lost)),
        "right_then_mid_loss_r": abs(sum(float(trade.r_multiple) for trade in mid_right_then_lost)),
    }


def leakage_control_score(
    metrics: HelixMetrics,
    leakage_metrics: dict[str, float],
    hard_rejects: dict[str, float] | None = None,
) -> LeakageControlScore:
    """Score Helix metrics with a leakage-focused immutable objective."""
    hr = dict(DEFAULT_HARD_REJECTS)
    hr.update(hard_rejects or {})

    avg_r = metrics.total_r / metrics.total_trades if metrics.total_trades else 0.0
    if metrics.total_trades < hr["min_trades"]:
        return LeakageControlScore(
            rejected=True,
            reject_reason=f"Too few trades: {metrics.total_trades} < {hr['min_trades']:.0f}",
        )
    if metrics.profit_factor < hr["min_pf"]:
        return LeakageControlScore(
            rejected=True,
            reject_reason=f"PF too low: {metrics.profit_factor:.2f} < {hr['min_pf']:.2f}",
        )
    if metrics.max_r_dd > hr["max_r_dd"]:
        return LeakageControlScore(
            rejected=True,
            reject_reason=f"Max R DD too high: {metrics.max_r_dd:.2f} > {hr['max_r_dd']:.2f}",
        )
    if metrics.tail_pct < hr["min_tail_pct"]:
        return LeakageControlScore(
            rejected=True,
            reject_reason=f"Tail preservation too low: {metrics.tail_pct:.2f} < {hr['min_tail_pct']:.2f}",
        )
    if metrics.min_regime_pf < hr["min_regime_pf"]:
        return LeakageControlScore(
            rejected=True,
            reject_reason=f"Min regime PF too low: {metrics.min_regime_pf:.2f} < {hr['min_regime_pf']:.2f}",
        )
    if metrics.net_return_pct < hr["min_net_return_pct"]:
        return LeakageControlScore(
            rejected=True,
            reject_reason=f"Return too low: {metrics.net_return_pct:.1f}% < {hr['min_net_return_pct']:.1f}%",
        )
    if avg_r < hr["min_avg_r"]:
        return LeakageControlScore(
            rejected=True,
            reject_reason=f"Average R too low: {avg_r:.2f} < {hr['min_avg_r']:.2f}",
        )

    leak_ratio = float(leakage_metrics.get("right_then_leak_ratio", 1.0))
    net_profit_component = _clip01(
        math.log1p(max(metrics.net_return_pct, 0.0) / 100.0) / math.log1p(8.0)
    )
    leak_loss_r = float(leakage_metrics.get("right_then_loss_r", 999.0))
    leak_ratio_component = _clip01((0.30 - leak_ratio) / 0.16)
    leak_loss_component = _clip01((70.0 - leak_loss_r) / 20.0)
    leak_control_component = 0.5 * leak_ratio_component + 0.5 * leak_loss_component
    pf_component = _scale(metrics.profit_factor, 2.7, 4.4)
    frequency_component = _scale(float(metrics.total_trades), 360.0, 460.0)
    tail_preservation_component = _scale(metrics.tail_pct, 0.56, 0.76)
    short_hold_quality_component = _clip01(1.0 - metrics.short_hold_r / 80.0)
    inv_dd_component = _scale(12.0 - metrics.max_r_dd, 0.0, 8.0)

    w = IMMUTABLE_SCORE_WEIGHTS
    total = (
        w["net_profit"] * net_profit_component
        + w["leak_control"] * leak_control_component
        + w["pf"] * pf_component
        + w["frequency"] * frequency_component
        + w["tail_preservation"] * tail_preservation_component
        + w["short_hold_quality"] * short_hold_quality_component
        + w["inv_dd"] * inv_dd_component
    )
    return LeakageControlScore(
        net_profit_component=net_profit_component,
        leak_control_component=leak_control_component,
        pf_component=pf_component,
        frequency_component=frequency_component,
        tail_preservation_component=tail_preservation_component,
        short_hold_quality_component=short_hold_quality_component,
        inv_dd_component=inv_dd_component,
        total=total,
    )
