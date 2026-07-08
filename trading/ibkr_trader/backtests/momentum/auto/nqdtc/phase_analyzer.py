"""NQDTC phase analysis -- optional hooks for the PhaseRunner framework.

The NQDTC plugin delegates most analysis to the shared framework via
PhaseAnalysisPolicy callbacks. This module provides NQDTC-specific
analysis helpers if needed for standalone diagnostics.
"""
from __future__ import annotations

from .scoring import NQDTCMetrics
from .plugin import ULTIMATE_TARGETS


def assess_goal_progress(metrics: NQDTCMetrics) -> dict[str, dict]:
    """Assess progress toward ultimate targets."""
    progress: dict[str, dict] = {}
    mapping = {
        "net_return_pct": metrics.net_return_pct,
        "profit_factor": metrics.profit_factor,
        "max_dd_pct": metrics.max_dd_pct,
        "calmar": metrics.calmar,
        "total_trades": float(metrics.total_trades),
        "capture_ratio": metrics.capture_ratio,
        "win_rate": metrics.win_rate,
        "sharpe": metrics.sharpe,
    }
    for key, actual in mapping.items():
        target = ULTIMATE_TARGETS.get(key, 0.0)
        if target == 0:
            continue
        # For max_dd_pct, lower is better
        if key == "max_dd_pct":
            pct = (1 - actual / target) * 100 if target > 0 else 0
            met = actual <= target
        else:
            pct = (actual / target) * 100 if target > 0 else 0
            met = actual >= target
        progress[key] = {"target": target, "actual": actual, "pct": pct, "met": met}
    return progress


def identify_strengths_weaknesses(metrics: NQDTCMetrics) -> tuple[list[str], list[str]]:
    """Identify NQDTC-specific strengths and weaknesses."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    if metrics.profit_factor >= 2.0:
        strengths.append(f"Strong profit factor ({metrics.profit_factor:.2f})")
    elif metrics.profit_factor < 1.5:
        weaknesses.append(f"Low profit factor ({metrics.profit_factor:.2f})")

    if metrics.capture_ratio >= 0.40:
        strengths.append(f"Good MFE capture ({metrics.capture_ratio:.2f})")
    elif metrics.capture_ratio < 0.35:
        weaknesses.append(f"Low MFE capture ({metrics.capture_ratio:.2f}) -- exits leave alpha")

    if metrics.max_dd_pct <= 0.08:
        strengths.append(f"Controlled drawdown ({metrics.max_dd_pct:.1%})")
    elif metrics.max_dd_pct > 0.12:
        weaknesses.append(f"Elevated drawdown ({metrics.max_dd_pct:.1%})")

    if metrics.total_trades >= 300:
        strengths.append(f"Good trade frequency ({metrics.total_trades})")
    elif metrics.total_trades < 200:
        weaknesses.append(f"Low trade count ({metrics.total_trades})")

    if metrics.burst_trade_pct > 0.15:
        weaknesses.append(f"High burst clustering ({metrics.burst_trade_pct:.1%})")

    if metrics.eth_short_wr < 0.40 and metrics.eth_short_trades > 30:
        weaknesses.append(f"ETH shorts weak (WR={metrics.eth_short_wr:.0%}, {metrics.eth_short_trades} trades)")

    if metrics.calmar >= 10.0:
        strengths.append(f"Excellent calmar ({metrics.calmar:.1f})")
    elif metrics.calmar < 5.0:
        weaknesses.append(f"Low calmar ({metrics.calmar:.1f})")

    return strengths, weaknesses
