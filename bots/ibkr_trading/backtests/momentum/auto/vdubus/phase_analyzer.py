"""VdubusNQ phase analysis -- optional hooks for the PhaseRunner framework.

The VdubusNQ plugin delegates most analysis to the shared framework via
PhaseAnalysisPolicy callbacks. This module provides VdubusNQ-specific
analysis helpers for standalone diagnostics.
"""
from __future__ import annotations

from .scoring import VdubusMetrics
from .plugin import ULTIMATE_TARGETS


def assess_goal_progress(metrics: VdubusMetrics) -> dict[str, dict]:
    """Assess progress toward ultimate targets."""
    progress: dict[str, dict] = {}
    mapping = {
        "profit_factor": metrics.profit_factor,
        "max_dd_pct": metrics.max_dd_pct,
        "max_r_drawdown": metrics.max_r_drawdown,
        "r_calmar": metrics.r_calmar,
        "r_per_month": metrics.r_per_month,
        "total_trades": float(metrics.total_trades),
        "capture_ratio": metrics.capture_ratio,
        "sharpe": metrics.sharpe,
    }
    for key, actual in mapping.items():
        target = ULTIMATE_TARGETS.get(key, 0.0)
        if target == 0:
            continue
        # For drawdown metrics, lower is better.
        if key in {"max_dd_pct", "max_r_drawdown"}:
            pct = (1 - actual / target) * 100 if target > 0 else 0
            met = actual <= target
        else:
            pct = (actual / target) * 100 if target > 0 else 0
            met = actual >= target
        progress[key] = {"target": target, "actual": actual, "pct": pct, "met": met}
    return progress


def identify_strengths_weaknesses(metrics: VdubusMetrics) -> tuple[list[str], list[str]]:
    """Identify VdubusNQ-specific strengths and weaknesses."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    if metrics.profit_factor >= 3.0:
        strengths.append(f"Strong profit factor ({metrics.profit_factor:.2f})")
    elif metrics.profit_factor < 2.0:
        weaknesses.append(f"Low profit factor ({metrics.profit_factor:.2f})")

    if metrics.capture_ratio >= 0.55:
        strengths.append(f"Good MFE capture ({metrics.capture_ratio:.2f})")
    elif metrics.capture_ratio < 0.45:
        weaknesses.append(f"Low MFE capture ({metrics.capture_ratio:.2f}) -- exits leave alpha")

    if metrics.max_dd_pct <= 0.12:
        strengths.append(f"Controlled drawdown ({metrics.max_dd_pct:.1%})")
    elif metrics.max_dd_pct > 0.18:
        weaknesses.append(f"Elevated drawdown ({metrics.max_dd_pct:.1%})")

    if metrics.stale_exit_pct < 0.30:
        strengths.append(f"Low stale exit rate ({metrics.stale_exit_pct:.1%})")
    elif metrics.stale_exit_pct > 0.40:
        weaknesses.append(f"High stale exit rate ({metrics.stale_exit_pct:.1%})")

    if metrics.multi_session_pct >= 0.30:
        strengths.append(f"Good overnight holding ({metrics.multi_session_pct:.1%})")

    if metrics.fast_death_pct > 0.20:
        weaknesses.append(f"Many fast deaths ({metrics.fast_death_pct:.1%})")

    if metrics.r_per_month >= 2.5:
        strengths.append(f"Strong R throughput ({metrics.r_per_month:.2f} R/month)")
    elif metrics.r_per_month < 1.5:
        weaknesses.append(f"Low R throughput ({metrics.r_per_month:.2f} R/month)")

    if metrics.r_calmar >= 5.0:
        strengths.append(f"Strong R-Calmar ({metrics.r_calmar:.1f})")
    elif metrics.r_calmar < 3.0:
        weaknesses.append(f"Low R-Calmar ({metrics.r_calmar:.1f})")

    if metrics.sharpe >= 2.0:
        strengths.append(f"Strong sharpe ({metrics.sharpe:.2f})")
    elif metrics.sharpe < 1.2:
        weaknesses.append(f"Low sharpe ({metrics.sharpe:.2f})")

    return strengths, weaknesses
