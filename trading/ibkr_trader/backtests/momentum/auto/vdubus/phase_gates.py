"""VdubusNQ phase gate criteria -- single source of truth for all gate logic."""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion, GateResult

from .scoring import VdubusMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: VdubusMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*.

    Phase 4 uses *prior_phase_metrics* (from phase 3) for 90% no-regression.
    """
    r_per_month = metrics.r_per_month if metrics.r_per_month else metrics.avg_r * metrics.trades_per_month

    # Hard floors (all phases)
    criteria: list[GateCriterion] = [
        GateCriterion("hard_min_trades", 120.0, float(metrics.total_trades), metrics.total_trades >= 120),
        GateCriterion("hard_max_dd_pct", 0.26, metrics.max_dd_pct, metrics.max_dd_pct <= 0.26),
        GateCriterion("hard_min_pf", 1.55, metrics.profit_factor, metrics.profit_factor >= 1.55),
        GateCriterion("hard_min_avg_r", 0.18, metrics.avg_r, metrics.avg_r >= 0.18),
    ]

    if phase == 1:
        criteria.extend([
            GateCriterion("profit_factor", 1.8, metrics.profit_factor, metrics.profit_factor >= 1.8),
            GateCriterion("capture_ratio", 0.48, metrics.capture_ratio, metrics.capture_ratio >= 0.48),
            GateCriterion("r_per_month", 1.50, r_per_month, r_per_month >= 1.50),
            GateCriterion("sharpe", 1.0, metrics.sharpe, metrics.sharpe >= 1.0),
        ])
    elif phase == 2:
        criteria.extend([
            GateCriterion("total_trades", 165.0, float(metrics.total_trades), metrics.total_trades >= 165),
            GateCriterion("trades_per_month", 5.0, metrics.trades_per_month, metrics.trades_per_month >= 5.0),
            GateCriterion("r_per_month", 1.50, r_per_month, r_per_month >= 1.50),
            GateCriterion("sharpe", 1.2, metrics.sharpe, metrics.sharpe >= 1.2),
        ])
    elif phase == 3:
        criteria.extend([
            GateCriterion("fast_death_pct", 0.26, metrics.fast_death_pct, metrics.fast_death_pct <= 0.26),
            GateCriterion("profit_factor", 1.75, metrics.profit_factor, metrics.profit_factor >= 1.75),
            GateCriterion("avg_r", 0.25, metrics.avg_r, metrics.avg_r >= 0.25),
            GateCriterion("r_per_month", 1.50, r_per_month, r_per_month >= 1.50),
        ])
    elif phase == 4:
        criteria.extend([
            GateCriterion("total_trades", 165.0, float(metrics.total_trades), metrics.total_trades >= 165),
            GateCriterion("trades_per_month", 5.0, metrics.trades_per_month, metrics.trades_per_month >= 5.0),
            GateCriterion("profit_factor", 1.65, metrics.profit_factor, metrics.profit_factor >= 1.65),
            GateCriterion("r_per_month", 1.50, r_per_month, r_per_month >= 1.50),
        ])
    elif phase == 5:
        criteria.extend([
            GateCriterion("evening_avg_r_floor", -0.20, metrics.evening_avg_r, metrics.evening_avg_r >= -0.20),
            GateCriterion("r_calmar", 4.0, metrics.r_calmar, metrics.r_calmar >= 4.0),
            GateCriterion("max_dd_pct", 0.24, metrics.max_dd_pct, metrics.max_dd_pct <= 0.24),
            GateCriterion("r_per_month", 1.50, r_per_month, r_per_month >= 1.50),
        ])
    elif phase == 6:
        # 90% no-regression floor on P3 metrics
        if prior_phase_metrics:
            for key in ["profit_factor", "r_calmar", "r_per_month", "sharpe", "total_trades", "capture_ratio", "trades_per_month", "avg_r"]:
                target = float(prior_phase_metrics.get(key, 0.0)) * 0.90
                actual = float(getattr(metrics, key, 0.0))
                criteria.append(GateCriterion(f"no_regress_{key}", target, actual, actual >= target))
        else:
            criteria.append(GateCriterion("phase4_pass", 0.0, 1.0, True))

    return criteria


def check_phase_gate(
    phase: int,
    metrics: VdubusMetrics,
    greedy_result: dict | None = None,
    prior_phase_metrics: dict | None = None,
) -> GateResult:
    """Full gate check with failure categorization and recommendations."""
    criteria = gate_criteria_for_phase(phase, metrics, prior_phase_metrics)
    if all(c.passed for c in criteria):
        return GateResult(passed=True, criteria=tuple(criteria))

    category = _categorize_failure(metrics, criteria, greedy_result)
    recs = _get_recommendations(metrics, criteria, category)
    return GateResult(passed=False, criteria=tuple(criteria), failure_category=category, recommendations=tuple(recs))


def _categorize_failure(metrics: VdubusMetrics, criteria: list[GateCriterion], greedy_result: dict | None) -> str:
    if metrics.total_trades < 120 or metrics.max_dd_pct > 0.26 or metrics.profit_factor < 1.55:
        return "structural_issue"
    if greedy_result and greedy_result.get("total_candidates", 0) > 0 and greedy_result.get("accepted_count", 0) == 0:
        return "candidates_exhausted"
    near_miss = sum(1 for c in criteria if not c.passed and c.target > 0 and c.actual / c.target >= 0.85)
    if near_miss >= 2:
        return "diagnostic_needed"
    return "scoring_ineffective"


def _get_recommendations(metrics: VdubusMetrics, criteria: list[GateCriterion], category: str) -> list[str]:
    recs: list[str] = []
    if category == "structural_issue":
        if metrics.total_trades < 150:
            recs.append("Relax signal gates to increase trade count")
        if metrics.max_dd_pct > 0.24:
            recs.append("Improve exit capture or tighten poor-regime participation")
        if metrics.profit_factor < 1.5:
            recs.append("Fundamental edge issue -- review signal quality")
    elif category == "scoring_ineffective":
        recs.append("Adjust scoring weights to better reward gate-relevant metrics")
    elif category == "candidates_exhausted":
        recs.append("All candidates rejected -- expand experiment pool or relax hard rejects")
    elif category == "diagnostic_needed":
        for c in criteria:
            if not c.passed:
                recs.append(f"Near-miss on {c.name}: {c.actual:.3f} vs target {c.target:.3f}")
    return recs
