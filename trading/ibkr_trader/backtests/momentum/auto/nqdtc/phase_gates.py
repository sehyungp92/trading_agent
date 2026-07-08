"""NQDTC phase gate criteria.

The score ranks candidates, while these gates keep selection viable. They are
intentionally baseline-compatible because the previous strict PF/return rejects
zeroed the incumbent and caused every phase to accept no candidates.
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion, GateResult

from .scoring import NQDTCMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: NQDTCMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*."""
    criteria: list[GateCriterion] = [
        GateCriterion("hard_min_trades", 70.0, float(metrics.total_trades), metrics.total_trades >= 70),
        GateCriterion("hard_max_dd_pct", 0.26, metrics.max_dd_pct, metrics.max_dd_pct <= 0.26),
        GateCriterion("hard_min_pf", 1.40, metrics.profit_factor, metrics.profit_factor >= 1.40),
        GateCriterion("hard_min_avg_r", 0.16, metrics.avg_r, metrics.avg_r >= 0.16),
        GateCriterion("hard_min_capture", 0.28, metrics.capture_ratio, metrics.capture_ratio >= 0.28),
        GateCriterion(
            "hard_max_largest_win_share",
            0.45,
            metrics.largest_win_pnl_share,
            metrics.largest_win_pnl_share <= 0.45,
        ),
    ]

    if phase == 1:
        criteria.extend([
            GateCriterion("exit_capture_floor", 0.32, metrics.capture_ratio, metrics.capture_ratio >= 0.32),
            GateCriterion("net_return_floor", 120.0, metrics.net_return_pct, metrics.net_return_pct >= 120.0),
        ])
    elif phase == 2:
        criteria.extend([
            GateCriterion("pf_quality_floor", 1.50, metrics.profit_factor, metrics.profit_factor >= 1.50),
            GateCriterion("avg_r_quality_floor", 0.22, metrics.avg_r, metrics.avg_r >= 0.22),
        ])
    elif phase == 3:
        criteria.extend([
            GateCriterion("frequency_floor", 80.0, float(metrics.total_trades), metrics.total_trades >= 80),
            GateCriterion("net_return_floor", 110.0, metrics.net_return_pct, metrics.net_return_pct >= 110.0),
        ])
    elif phase == 4:
        criteria.extend([
            GateCriterion("frequency_floor", 80.0, float(metrics.total_trades), metrics.total_trades >= 80),
            GateCriterion("robust_return_floor", 80.0, metrics.robust_net_return_pct, metrics.robust_net_return_pct >= 80.0),
        ])
    elif phase == 5:
        criteria.extend([
            GateCriterion("frequency_floor", 90.0, float(metrics.total_trades), metrics.total_trades >= 90),
            GateCriterion("pf_quality_floor", 1.50, metrics.profit_factor, metrics.profit_factor >= 1.50),
            GateCriterion("robust_return_floor", 100.0, metrics.robust_net_return_pct, metrics.robust_net_return_pct >= 100.0),
        ])

    if prior_phase_metrics:
        for key, tolerance in {
            "profit_factor": 0.82,
            "net_return_pct": 0.85,
            "robust_net_return_pct": 0.82,
            "avg_r": 0.80,
            "capture_ratio": 0.80,
        }.items():
            target = float(prior_phase_metrics.get(key, 0.0)) * tolerance
            actual = float(getattr(metrics, key, 0.0))
            if target > 0:
                criteria.append(GateCriterion(f"no_regress_{key}", target, actual, actual >= target))
        prior_trades = float(prior_phase_metrics.get("total_trades", metrics.total_trades))
        trade_target = max(70.0, prior_trades * 0.80)
        criteria.append(GateCriterion("no_regress_total_trades", trade_target, float(metrics.total_trades), metrics.total_trades >= trade_target))
        prior_dd = float(prior_phase_metrics.get("max_dd_pct", metrics.max_dd_pct))
        dd_target = min(0.30, max(0.18, prior_dd * 1.25))
        criteria.append(GateCriterion("no_regress_max_dd_pct", dd_target, metrics.max_dd_pct, metrics.max_dd_pct <= dd_target))

    return criteria


def check_phase_gate(
    phase: int,
    metrics: NQDTCMetrics,
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


def _categorize_failure(metrics: NQDTCMetrics, criteria: list[GateCriterion], greedy_result: dict | None) -> str:
    hard_fails = [c for c in criteria if not c.passed and c.name.startswith("hard_")]
    if hard_fails:
        return "structural_issue"
    if greedy_result and greedy_result.get("total_candidates", 0) > 0 and greedy_result.get("accepted_count", 0) == 0:
        return "candidates_exhausted"
    near_miss = sum(1 for c in criteria if not c.passed and c.target > 0 and c.actual / c.target >= 0.85)
    if near_miss >= 2:
        return "diagnostic_needed"
    return "scoring_ineffective"


def _get_recommendations(metrics: NQDTCMetrics, criteria: list[GateCriterion], category: str) -> list[str]:
    recs: list[str] = []
    if category == "structural_issue":
        if metrics.total_trades < 70:
            recs.append("Entry/frequency levers are still too restrictive")
        if metrics.max_dd_pct > 0.26:
            recs.append("Tighten stop-width, MFE ratchet, or cooldown interactions")
        if metrics.profit_factor < 1.40:
            recs.append("Signal recovery is adding low-quality trades")
    elif category == "scoring_ineffective":
        recs.append("Keep score immutable and add candidates targeting the failed dimension")
    elif category == "candidates_exhausted":
        recs.append("All candidates rejected; inspect hard rejects before expanding search")
    elif category == "diagnostic_needed":
        for c in criteria:
            if not c.passed:
                recs.append(f"Near-miss on {c.name}: {c.actual:.3f} vs target {c.target:.3f}")
    return recs
