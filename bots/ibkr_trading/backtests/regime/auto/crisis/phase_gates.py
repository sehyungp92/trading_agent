"""Crisis detection phase gate criteria -- Round 3: recovery architecture.

Coverage is non-negotiable (7/7 crises) in all phases.  Progressive
tightening applies to FP rates, hard latency, early action latency, and
recovery speed.
Corrections (2 "C" type periods) are tracked but don't affect gate pass/fail.

  Phase 1: crises>=7, avg_latency<=30, action_latency<=20, warning_fp<=0.12
  Phase 2: crises>=7, avg_latency<=22, action_latency<=18, warning_fp<=0.08
  Phase 3: crises>=7, avg_latency<=20, action_latency<=17, warning_fp<=0.06
  Phase 4: crises>=7, avg_latency<=18.5, action_latency<=16.5, warning_fp<=0.05
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion, GateResult

from .scoring import CrisisMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: CrisisMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*."""
    criteria: list[GateCriterion] = []

    if phase == 1:
        # Phase 1: Recovery architecture -- structural discovery.
        criteria.extend([
            GateCriterion(
                "crises_detected", 7.0, float(metrics.crises_detected),
                metrics.crises_detected >= 7,
            ),
            GateCriterion(
                "avg_latency", 30.0, metrics.avg_latency,
                metrics.avg_latency <= 30.0,
            ),
            GateCriterion(
                "avg_action_latency", 20.0, metrics.avg_action_latency,
                metrics.avg_action_latency <= 20.0,
            ),
            GateCriterion(
                "warning_fp_rate", 0.12, metrics.warning_fp_rate,
                metrics.warning_fp_rate <= 0.12,
            ),
            GateCriterion(
                "preaction_fp_rate", 0.20, metrics.preaction_fp_rate,
                metrics.preaction_fp_rate <= 0.20,
            ),
        ])
    elif phase == 2:
        # Phase 2: Threshold re-optimization, add recovery gate.
        criteria.extend([
            GateCriterion(
                "crises_detected", 7.0, float(metrics.crises_detected),
                metrics.crises_detected >= 7,
            ),
            GateCriterion(
                "avg_latency", 22.0, metrics.avg_latency,
                metrics.avg_latency <= 22.0,
            ),
            GateCriterion(
                "avg_action_latency", 18.0, metrics.avg_action_latency,
                metrics.avg_action_latency <= 18.0,
            ),
            GateCriterion(
                "warning_fp_rate", 0.08, metrics.warning_fp_rate,
                metrics.warning_fp_rate <= 0.08,
            ),
            GateCriterion(
                "preaction_fp_rate", 0.16, metrics.preaction_fp_rate,
                metrics.preaction_fp_rate <= 0.16,
            ),
            GateCriterion(
                "avg_recovery_days", 20.0, metrics.avg_recovery_days,
                metrics.avg_recovery_days <= 20.0,
            ),
        ])
    elif phase == 3:
        # Phase 3: Correlation + yield + conjunction -- tighten FP and recovery.
        criteria.extend([
            GateCriterion(
                "crises_detected", 7.0, float(metrics.crises_detected),
                metrics.crises_detected >= 7,
            ),
            GateCriterion(
                "avg_latency", 20.0, metrics.avg_latency,
                metrics.avg_latency <= 20.0,
            ),
            GateCriterion(
                "avg_action_latency", 17.0, metrics.avg_action_latency,
                metrics.avg_action_latency <= 17.0,
            ),
            GateCriterion(
                "warning_fp_rate", 0.06, metrics.warning_fp_rate,
                metrics.warning_fp_rate <= 0.06,
            ),
            GateCriterion(
                "preaction_fp_rate", 0.14, metrics.preaction_fp_rate,
                metrics.preaction_fp_rate <= 0.14,
            ),
            GateCriterion(
                "crisis_fp_rate", 0.03, metrics.crisis_fp_rate,
                metrics.crisis_fp_rate <= 0.03,
            ),
            GateCriterion(
                "avg_recovery_days", 15.0, metrics.avg_recovery_days,
                metrics.avg_recovery_days <= 15.0,
            ),
        ])
    elif phase == 4:
        # Phase 4: Final targets with recovery.
        criteria.extend([
            GateCriterion(
                "crises_detected", 7.0, float(metrics.crises_detected),
                metrics.crises_detected >= 7,
            ),
            GateCriterion(
                "avg_latency", 18.5, metrics.avg_latency,
                metrics.avg_latency <= 18.5,
            ),
            GateCriterion(
                "avg_action_latency", 16.5, metrics.avg_action_latency,
                metrics.avg_action_latency <= 16.5,
            ),
            GateCriterion(
                "avg_advisory_latency", 8.0, metrics.avg_advisory_latency,
                metrics.avg_advisory_latency <= 8.0,
            ),
            GateCriterion(
                "warning_fp_rate", 0.05, metrics.warning_fp_rate,
                metrics.warning_fp_rate <= 0.05,
            ),
            GateCriterion(
                "preaction_fp_rate", 0.12, metrics.preaction_fp_rate,
                metrics.preaction_fp_rate <= 0.12,
            ),
            GateCriterion(
                "advisory_fp_rate", 0.45, metrics.advisory_fp_rate,
                metrics.advisory_fp_rate <= 0.45,
            ),
            GateCriterion(
                "crisis_fp_rate", 0.02, metrics.crisis_fp_rate,
                metrics.crisis_fp_rate <= 0.02,
            ),
            GateCriterion(
                "avg_recovery_days", 10.0, metrics.avg_recovery_days,
                metrics.avg_recovery_days <= 10.0,
            ),
        ])

    return criteria


def check_phase_gate(
    phase: int,
    metrics: CrisisMetrics,
    greedy_result: dict | None = None,
    prior_phase_metrics: dict | None = None,
) -> GateResult:
    """Full gate check with failure categorization and recommendations."""
    criteria = gate_criteria_for_phase(phase, metrics, prior_phase_metrics)
    if all(c.passed for c in criteria):
        return GateResult(passed=True, criteria=tuple(criteria))

    category = _categorize_failure(metrics, criteria, greedy_result)
    recs = _get_recommendations(metrics, criteria, category)
    return GateResult(
        passed=False, criteria=tuple(criteria),
        failure_category=category, recommendations=tuple(recs),
    )


def _categorize_failure(
    metrics: CrisisMetrics,
    criteria: list[GateCriterion],
    greedy_result: dict | None,
) -> str:
    if metrics.crises_detected < 7:
        return "coverage_gap"
    if metrics.warning_fp_rate > 0.15:
        return "fp_explosion"
    if metrics.preaction_fp_rate > 0.20:
        return "preaction_fp_explosion"
    if metrics.avg_action_latency > 20.0:
        return "slow_preaction"
    if hasattr(metrics, "avg_recovery_days") and metrics.avg_recovery_days > 25:
        return "slow_recovery"
    if greedy_result and greedy_result.get("accepted_count", 0) == 0:
        return "candidates_exhausted"
    return "tuning_needed"


def _get_recommendations(
    metrics: CrisisMetrics,
    criteria: list[GateCriterion],
    category: str,
) -> list[str]:
    recs: list[str] = []
    if category == "coverage_gap":
        recs.append("Lower threshold levels or relax conjunction requirements")
    elif category == "fp_explosion":
        recs.append("Raise threshold levels or tighten conjunction requirements")
    elif category == "preaction_fp_explosion":
        recs.append("Raise stress-formation minimum score or tighten shock/grind thresholds")
    elif category == "slow_preaction":
        recs.append("Lower shock/grind stress thresholds or add targeted stress-formation candidates")
    elif category == "slow_recovery":
        recs.append("Enable accelerated de-escalation or reduce hysteresis days")
    elif category == "candidates_exhausted":
        recs.append("Expand candidate pool or relax hard rejects")
    else:
        for c in criteria:
            if not c.passed:
                recs.append(
                    f"Near-miss on {c.name}: {c.actual:.3f} vs target {c.target:.3f}"
                )
    return recs
