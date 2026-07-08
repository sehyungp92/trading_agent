"""Scoring normalizers and composite score computation.

All PerformanceMetrics values are in percentage units (e.g. max_drawdown_pct=25.0
means 25%) or raw values. Normalizers map each metric to [0, 1].
"""

from __future__ import annotations

import math
from typing import Any


def normalize_coverage(metrics: dict[str, float]) -> float:
    """More trades = better coverage. 30 trades = 1.0."""
    trades = metrics.get("total_trades", 0)
    return min(trades / 30.0, 1.0)


def normalize_risk(metrics: dict[str, float]) -> float:
    """Lower drawdown = better. 0% dd -> 1.0, 50% dd -> 0.0."""
    dd = metrics.get("max_drawdown_pct", 50.0)
    return max(1.0 - dd / 50.0, 0.0)


def normalize_edge(metrics: dict[str, float]) -> float:
    """Higher profit factor = better edge. PF 7.0 -> 1.0."""
    pf = metrics.get("profit_factor", 0.0)
    return min(max((pf - 1.0) / 6.0, 0.0), 1.0)


def normalize_capture(metrics: dict[str, float]) -> float:
    """Exit efficiency — how much of MFE was captured."""
    eff = metrics.get("exit_efficiency", 0.0)
    return max(min(eff, 1.0), 0.0)


def normalize_expectancy(metrics: dict[str, float]) -> float:
    """Average R per trade. 0.6R/trade = 1.0, negative = 0.0."""
    expectancy = metrics.get("expectancy_r", 0.0)
    return min(max(expectancy / 0.6, 0.0), 1.0)


def normalize_hold(metrics: dict[str, float]) -> float:
    """Gaussian centered at 12 M15 bars (~3 hours). Penalizes too short/long."""
    bars = metrics.get("avg_bars_held", 0.0)
    return math.exp(-0.5 * ((bars - 12) / 8) ** 2)


def normalize_entry_quality(metrics: dict[str, float]) -> float:
    """Lower |avg MAE-R| = better entries. |MAE| 0 -> 1.0, |MAE| 1+ -> 0.0."""
    mae = abs(metrics.get("avg_mae_r", 0.0))
    return max(min(1.0 - mae, 1.0), 0.0)


normalize_exit_efficiency = normalize_capture


def normalize_calmar(metrics: dict[str, float]) -> float:
    """Calmar ratio normalizer. calmar 5+ -> 1.0."""
    calmar = metrics.get("calmar_ratio", 0.0)
    return min(max(calmar / 5.0, 0.0), 1.0)


def normalize_sharpe(metrics: dict[str, float]) -> float:
    """Sharpe ratio normalizer. sharpe 2.5+ -> 1.0."""
    sharpe = metrics.get("sharpe_ratio", 0.0)
    return min(max(sharpe / 2.5, 0.0), 1.0)


def normalize_returns(metrics: dict[str, float]) -> float:
    """Net return percentage. 10% return -> 1.0, negative -> 0."""
    ret = metrics.get("net_return_pct", 0.0)
    return min(max(ret / 10.0, 0.0), 1.0)


def normalize_stability(metrics: dict[str, float]) -> float:
    """Lower max drawdown duration = more stable. 2000 M15 bars (~20d) -> 0."""
    bars = metrics.get("max_drawdown_duration", 0)
    return max(1.0 - bars / 2000.0, 0.0)


NORMALIZERS: dict[str, Any] = {
    "coverage": normalize_coverage,
    "risk": normalize_risk,
    "edge": normalize_edge,
    "capture": normalize_capture,
    "expectancy": normalize_expectancy,
    "hold": normalize_hold,
    "entry_quality": normalize_entry_quality,
    "exit_efficiency": normalize_exit_efficiency,
    "returns": normalize_returns,
    "calmar": normalize_calmar,
    "sharpe": normalize_sharpe,
    "stability": normalize_stability,
}


def check_hard_rejects(
    metrics: dict[str, float],
    hard_rejects: dict[str, tuple[str, float]],
) -> tuple[bool, str]:
    """Check if metrics violate any hard reject thresholds.

    Returns (rejected, reason). If rejected is True, reason describes why.
    """
    for metric_name, (operator, threshold) in hard_rejects.items():
        value = metrics.get(metric_name, 0.0)
        passed = False
        if operator == ">=":
            passed = value >= threshold
        elif operator == "<=":
            passed = value <= threshold
        elif operator == ">":
            passed = value > threshold
        elif operator == "<":
            passed = value < threshold

        if not passed:
            return True, f"{metric_name}={value:.2f} failed {operator} {threshold:.2f}"

    return False, ""


def _normalize_with_ceiling(
    dimension: str, metrics: dict[str, float], ceiling: float,
) -> float:
    """Apply a custom ceiling to a normalizer dimension."""
    if dimension == "returns":
        return min(max(metrics.get("net_return_pct", 0.0) / ceiling, 0.0), 1.0)
    if dimension == "edge":
        return min(max((metrics.get("profit_factor", 0.0) - 1.0) / ceiling, 0.0), 1.0)
    if dimension == "coverage":
        return min(metrics.get("total_trades", 0.0) / ceiling, 1.0)
    if dimension == "calmar":
        return min(max(metrics.get("calmar_ratio", 0.0) / ceiling, 0.0), 1.0)
    if dimension == "sharpe":
        return min(max(metrics.get("sharpe_ratio", 0.0) / ceiling, 0.0), 1.0)
    if dimension == "risk":
        return max(1.0 - metrics.get("max_drawdown_pct", 50.0) / ceiling, 0.0)
    if dimension == "expectancy":
        return min(max(metrics.get("expectancy_r", 0.0) / ceiling, 0.0), 1.0)
    if dimension == "capture":
        return min(max(metrics.get("exit_efficiency", 0.0) / ceiling, 0.0), 1.0)
    if dimension == "entry_quality":
        mae = abs(metrics.get("avg_mae_r", 0.0))
        return max(min(1.0 - mae / ceiling, 1.0), 0.0)
    # Fallback to default normalizer
    normalizer = NORMALIZERS.get(dimension)
    return normalizer(metrics) if normalizer else 0.0


def composite_score(
    metrics: dict[str, float],
    weights: dict[str, float],
    hard_rejects: dict[str, tuple[str, float]] | None = None,
    ceilings: dict[str, float] | None = None,
) -> tuple[float, bool, str]:
    """Compute weighted composite score from metrics.

    Returns (score, rejected, reject_reason).
    If hard_rejects are specified and any fail, score is 0.0 and rejected is True.
    Optional *ceilings* dict overrides default normalizer ceilings per dimension
    (e.g. ``{"returns": 25.0, "edge": 14.0}``).
    """
    if hard_rejects:
        rejected, reason = check_hard_rejects(metrics, hard_rejects)
        if rejected:
            return 0.0, True, reason

    score = 0.0
    for dimension, weight in weights.items():
        if ceilings and dimension in ceilings:
            score += weight * _normalize_with_ceiling(dimension, metrics, ceilings[dimension])
        else:
            normalizer = NORMALIZERS.get(dimension)
            if normalizer:
                score += weight * normalizer(metrics)

    return score, False, ""
