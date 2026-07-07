"""Downturn phase gate criteria — success thresholds + 4 failure categories."""
from __future__ import annotations

from dataclasses import dataclass

from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics


@dataclass(frozen=True)
class GateCriterion:
    """A single gate criterion with target and actual value."""
    name: str
    target: float
    actual: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    """Result of a phase gate check."""
    passed: bool
    criteria: tuple[GateCriterion, ...] = ()
    failure_category: str | None = None
    recommendations: tuple[str, ...] = ()


def check_phase_gate(
    phase: int,
    metrics: DownturnMetrics,
    greedy_result: dict | None = None,
    prior_phase_metrics: dict | None = None,
) -> GateResult:
    """Check if a phase's gate criteria are met.

    4 failure categories:
      scoring_ineffective — score improved but gate metrics didn't move
      candidates_exhausted — all candidates tried, none accepted
      structural_issue — fundamental problem
      diagnostic_needed — >=2 criteria within 85% of target
    """
    criteria = _get_criteria(phase, metrics, prior_phase_metrics)
    passed_count = sum(1 for c in criteria if c.passed)
    total = len(criteria)
    all_passed = passed_count == total

    if all_passed:
        return GateResult(passed=True, criteria=tuple(criteria))

    # Determine failure category
    category = _categorize_failure(phase, metrics, criteria, greedy_result)
    recs = _get_recommendations(phase, metrics, criteria, category)

    return GateResult(
        passed=False,
        criteria=tuple(criteria),
        failure_category=category,
        recommendations=tuple(recs),
    )


def _get_criteria(
    phase: int,
    metrics: DownturnMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Get phase-specific gate criteria."""
    if phase == 1:
        return [
            GateCriterion("signal_to_entry", 0.15, metrics.signal_to_entry_ratio,
                          metrics.signal_to_entry_ratio >= 0.15),
            GateCriterion("total_trades", 15, metrics.total_trades,
                          metrics.total_trades >= 15),
            GateCriterion("correction_pnl_pct", 5.0, metrics.correction_pnl_pct,
                          metrics.correction_pnl_pct >= 5.0),
        ]
    elif phase == 2:
        return [
            GateCriterion("exit_efficiency", 0.20, metrics.exit_efficiency,
                          metrics.exit_efficiency >= 0.20),
            GateCriterion("profit_factor", 1.3, metrics.profit_factor,
                          metrics.profit_factor >= 1.3),
            GateCriterion("correction_pnl_pct", 10.0, metrics.correction_pnl_pct,
                          metrics.correction_pnl_pct >= 10.0),
        ]
    elif phase == 3:
        return [
            GateCriterion("calmar", 1.0, metrics.calmar,
                          metrics.calmar >= 1.0),
            GateCriterion("max_dd_pct", 0.22, metrics.max_dd_pct,
                          metrics.max_dd_pct <= 0.22),
            GateCriterion("sharpe", 0.6, metrics.sharpe,
                          metrics.sharpe >= 0.6),
        ]
    elif phase == 4:
        # No regression: each metric within 90% of phase 3
        criteria = []
        if prior_phase_metrics:
            for key in ["calmar", "profit_factor", "sharpe", "correction_pnl_pct"]:
                target = _prior_metric(prior_phase_metrics, key) * 0.90
                actual = getattr(metrics, key, 0)
                criteria.append(GateCriterion(
                    f"no_regress_{key}", target, actual, actual >= target,
                ))
        if not criteria:
            criteria.append(GateCriterion("phase4_pass", 0, 1, True))
        return criteria

    elif phase == 5:
        # No regression from Phase 4
        criteria = []
        if prior_phase_metrics:
            for key in ["calmar", "profit_factor", "sharpe", "correction_pnl_pct"]:
                target = _prior_metric(prior_phase_metrics, key) * 0.90
                actual = getattr(metrics, key, 0)
                criteria.append(GateCriterion(
                    f"no_regress_{key}", target, actual, actual >= target,
                ))
        if not criteria:
            criteria.append(GateCriterion("phase5_pass", 0, 1, True))
        return criteria

    return []


def _categorize_failure(
    phase: int,
    metrics: DownturnMetrics,
    criteria: list[GateCriterion],
    greedy_result: dict | None,
) -> str:
    """Determine failure category."""
    # Structural issue
    if metrics.total_trades < 8 or metrics.max_dd_pct > 0.30:
        return "structural_issue"
    if metrics.correction_pnl_pct < 0:
        return "structural_issue"

    # Candidates exhausted
    if greedy_result:
        accepted = greedy_result.get("accepted_count", 0)
        total = greedy_result.get("total_candidates", 0)
        if total > 0 and accepted == 0:
            return "candidates_exhausted"

    # Diagnostic needed: >=2 criteria within 85% of target
    near_miss = 0
    for c in criteria:
        if not c.passed and c.target > 0:
            ratio = c.actual / c.target
            if ratio >= 0.85:
                near_miss += 1
    if near_miss >= 2:
        return "diagnostic_needed"

    return "scoring_ineffective"


def _get_recommendations(
    phase: int,
    metrics: DownturnMetrics,
    criteria: list[GateCriterion],
    category: str,
) -> list[str]:
    """Generate recommendations based on failure category."""
    recs = []
    if category == "structural_issue":
        if metrics.total_trades < 8:
            recs.append("Relax signal gates to increase trade count")
        if metrics.max_dd_pct > 0.30:
            recs.append("Reduce position sizing or add tighter stops")
        if metrics.correction_pnl_pct < 0:
            recs.append("Strategy loses money during corrections — fundamental signal issue")
    elif category == "scoring_ineffective":
        recs.append("Adjust scoring weights to better reward gate-relevant metrics")
    elif category == "candidates_exhausted":
        recs.append("All candidates rejected — expand experiment pool or relax hard rejects")
    elif category == "diagnostic_needed":
        failed = [c for c in criteria if not c.passed]
        for c in failed:
            recs.append(f"Near-miss on {c.name}: {c.actual:.3f} vs target {c.target:.3f}")
    return recs


def _prior_metric(prior_phase_metrics: dict, key: str) -> float:
    if key in prior_phase_metrics:
        return prior_phase_metrics[key]
    if key == "correction_pnl_pct":
        return prior_phase_metrics.get("correction_alpha_pct", 0)
    return prior_phase_metrics.get(key, 0)
