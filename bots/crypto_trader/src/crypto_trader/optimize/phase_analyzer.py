"""Phase analysis — assesses progress and generates retry recommendations.

Pipeline:
1. Goal progress computation with strength/weakness classification
2. Scoring assessment (EFFECTIVE/INEFFECTIVE/MARGINAL/MISALIGNED)
3. Policy callback invocation
4. Decision logic (custom or fallback chain)
5. Weight redesign for improve_scoring recommendations
6. Structured report generation
"""

from __future__ import annotations

from typing import Any

from crypto_trader.optimize.phase_gates import _lower_is_better
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.optimize.types import (
    GateResult,
    GreedyResult,
    PhaseAnalysis,
    PhaseAnalysisPolicy,
    PhaseDecision,
)

# Scoring assessment taxonomy
EFFECTIVE = "EFFECTIVE"
INEFFECTIVE = "INEFFECTIVE"
MARGINAL = "MARGINAL"
MISALIGNED = "MISALIGNED"


def analyze_phase(
    phase: int,
    greedy_result: GreedyResult,
    metrics: dict[str, float],
    state: PhaseState,
    gate_result: GateResult,
    *,
    ultimate_targets: dict[str, float] | None = None,
    policy: PhaseAnalysisPolicy | None = None,
    current_weights: dict[str, float] | None = None,
    max_scoring_retries: int = 2,
    max_diagnostic_retries: int = 1,
) -> PhaseAnalysis:
    """Analyze a completed phase and recommend next action.

    Returns PhaseAnalysis with recommendation, goal progress, scoring assessment,
    strengths/weaknesses, diagnostic gaps, suggested experiments, and a text report.
    """
    policy = policy or PhaseAnalysisPolicy()
    ultimate_targets = ultimate_targets or {}
    current_weights = current_weights or {}
    max_scoring = max_scoring_retries
    max_diag = max_diagnostic_retries

    # 1. Goal progress
    goal_progress = _compute_goal_progress(metrics, ultimate_targets)

    # Classify strengths and weaknesses
    strengths = [m for m, p in goal_progress.items() if p["pct_of_target"] >= 80.0]
    weaknesses = [m for m, p in goal_progress.items() if p["pct_of_target"] < 60.0]

    # 2. Scoring assessment
    scoring_assessment = _assess_scoring(
        greedy_result, gate_result, metrics, policy, state, phase
    )

    # 3. Policy callbacks
    diagnostic_gaps: list[str] = []
    suggested_experiments = []
    extra: dict[str, Any] = {}

    if policy.diagnostic_gap_fn:
        diagnostic_gaps = policy.diagnostic_gap_fn(phase, metrics)

    if policy.suggest_experiments_fn:
        suggested_experiments = policy.suggest_experiments_fn(
            phase, metrics, weaknesses, state
        )

    if policy.build_extra_analysis_fn:
        extra = policy.build_extra_analysis_fn(
            phase, metrics, state, greedy_result
        )

    # 4. Decision logic
    scoring_weight_overrides = None
    recommendation_reason = ""

    # Try custom decision first
    decision: PhaseDecision | None = None
    if policy.decide_action_fn:
        decision = policy.decide_action_fn(
            phase, metrics, state, greedy_result, gate_result,
            current_weights, goal_progress, max_scoring, max_diag,
        )

    if decision is not None:
        # Validate feasibility (budget check)
        scoring_used = state.scoring_retries.get(phase, 0)
        diag_used = state.diagnostic_retries.get(phase, 0)

        if decision.action == "improve_scoring" and scoring_used >= max_scoring:
            recommendation = "advance"
            recommendation_reason = (
                f"Custom decision requested improve_scoring but budget exhausted "
                f"({scoring_used}/{max_scoring})"
            )
        elif decision.action == "improve_diagnostics" and diag_used >= max_diag:
            recommendation = "advance"
            recommendation_reason = (
                f"Custom decision requested improve_diagnostics but budget exhausted "
                f"({diag_used}/{max_diag})"
            )
        else:
            recommendation = decision.action
            recommendation_reason = decision.reason
            scoring_weight_overrides = decision.scoring_weight_overrides
            if decision.scoring_assessment_override:
                scoring_assessment = decision.scoring_assessment_override
            if decision.extra_diagnostic_gaps:
                diagnostic_gaps.extend(decision.extra_diagnostic_gaps)
            if decision.extra_suggested_experiments:
                suggested_experiments.extend(decision.extra_suggested_experiments)
    else:
        # Fallback chain
        recommendation, recommendation_reason = _fallback_recommendation(
            phase, gate_result, greedy_result, scoring_assessment,
            diagnostic_gaps, state, max_scoring, max_diag,
        )

    # 5. Weight redesign when recommending improve_scoring
    if recommendation == "improve_scoring" and scoring_weight_overrides is None:
        scoring_weight_overrides = _redesign_weights(
            policy, phase, current_weights, metrics,
            strengths, weaknesses, policy.focus_metrics,
        )

    # 6. Build report
    summary = _build_analysis_report(
        phase, recommendation, recommendation_reason, scoring_assessment,
        strengths, weaknesses, diagnostic_gaps, goal_progress,
        greedy_result, extra, policy,
    )

    return PhaseAnalysis(
        phase=phase,
        recommendation=recommendation,
        summary=summary,
        goal_progress=goal_progress,
        scoring_assessment=scoring_assessment,
        strengths=strengths,
        weaknesses=weaknesses,
        diagnostic_gaps=diagnostic_gaps,
        suggested_experiments=suggested_experiments,
        recommendation_reason=recommendation_reason,
        report=summary,
        scoring_weight_overrides=scoring_weight_overrides,
        extra=extra,
    )


def _compute_goal_progress(
    metrics: dict[str, float],
    targets: dict[str, float],
) -> dict[str, dict]:
    """Compute progress toward ultimate targets.

    Returns {metric: {target, actual, pct_of_target}}.
    For "lower is better" metrics, 100% means at or below target.
    Clamped at 200%.
    """
    progress: dict[str, dict] = {}
    for metric, target in targets.items():
        actual = metrics.get(metric, 0.0)
        if target == 0:
            pct = 100.0 if actual >= 0 else 0.0
        elif _lower_is_better(metric):
            # Lower actual = better. At target = 100%, above target = <100%
            pct = (target / actual) * 100.0 if actual > 0 else 100.0
        else:
            pct = (actual / target) * 100.0

        pct = min(pct, 200.0)
        progress[metric] = {"target": target, "actual": actual, "pct_of_target": pct}

    return progress


def _assess_scoring(
    greedy_result: GreedyResult,
    gate_result: GateResult,
    metrics: dict[str, float],
    policy: PhaseAnalysisPolicy,
    state: PhaseState,
    phase: int,
) -> str:
    """Assess quality of scoring using 4-level taxonomy.

    - EFFECTIVE: gate passed
    - INEFFECTIVE: 0 accepted or candidates_exhausted category
    - MARGINAL: low score delta or diagnostic_needed category
    - MISALIGNED: focus metrics far from target or not improving vs previous phase
    """
    if gate_result.passed:
        return EFFECTIVE

    if not greedy_result.accepted_experiments:
        return INEFFECTIVE

    if gate_result.failure_category == "candidates_exhausted":
        return INEFFECTIVE

    if gate_result.failure_category == "diagnostic_needed":
        return MARGINAL

    # Check if score delta is very low
    if greedy_result.base_score > 0:
        delta_pct = abs(greedy_result.final_score - greedy_result.base_score) / greedy_result.base_score
        if delta_pct < policy.min_effective_score_delta_pct:
            return MARGINAL

    # Check if focus metrics are far from target (MISALIGNED)
    if policy.focus_metrics:
        prev_metrics = state.get_phase_metrics(phase - 1) if phase > 1 else None
        for fm in policy.focus_metrics:
            actual = metrics.get(fm, 0.0)
            prev_actual = prev_metrics.get(fm, 0.0) if prev_metrics else 0.0
            # Not improving compared to previous phase
            if prev_metrics and actual <= prev_actual and actual > 0:
                return MISALIGNED

    return MARGINAL


def _fallback_recommendation(
    phase: int,
    gate_result: GateResult,
    greedy_result: GreedyResult,
    scoring_assessment: str,
    diagnostic_gaps: list[str],
    state: PhaseState,
    max_scoring: int,
    max_diag: int,
) -> tuple[str, str]:
    """Fallback recommendation chain.

    Returns (recommendation, reason).
    """
    scoring_used = state.scoring_retries.get(phase, 0)
    diag_used = state.diagnostic_retries.get(phase, 0)

    if gate_result.passed:
        return "advance", "Gate passed."

    if gate_result.failure_category == "diagnostic_needed" and diag_used < max_diag:
        return "improve_diagnostics", (
            f"Gate failure category is diagnostic_needed. "
            f"Diagnostic retry {diag_used + 1}/{max_diag}."
        )

    if scoring_assessment in (MISALIGNED, INEFFECTIVE) and scoring_used < max_scoring:
        return "improve_scoring", (
            f"Scoring assessment is {scoring_assessment}. "
            f"Scoring retry {scoring_used + 1}/{max_scoring}."
        )

    if scoring_assessment == MARGINAL and scoring_used < max_scoring:
        return "improve_scoring", (
            f"Scoring assessment is MARGINAL. "
            f"Scoring retry {scoring_used + 1}/{max_scoring}."
        )

    if diagnostic_gaps and diag_used < max_diag:
        return "improve_diagnostics", (
            f"{len(diagnostic_gaps)} diagnostic gap(s) identified. "
            f"Diagnostic retry {diag_used + 1}/{max_diag}."
        )

    # Scoring budget exhausted but diagnostic budget remains — try diagnostics
    if scoring_used >= max_scoring and diag_used < max_diag:
        return "improve_diagnostics", (
            f"Scoring budget exhausted ({scoring_used}/{max_scoring}). "
            f"Trying diagnostics retry {diag_used + 1}/{max_diag}."
        )

    # Budget exhausted — must advance
    if scoring_used >= max_scoring and diag_used >= max_diag:
        return "advance", (
            f"Retry budget exhausted (scoring: {scoring_used}/{max_scoring}, "
            f"diagnostics: {diag_used}/{max_diag}). Force advancing."
        )

    # Remaining scoring budget
    if scoring_used < max_scoring:
        return "improve_scoring", (
            f"Gate failed. Scoring retry {scoring_used + 1}/{max_scoring}."
        )

    return "advance", "No retry budget remaining."


def _redesign_weights(
    policy: PhaseAnalysisPolicy,
    phase: int,
    current_weights: dict[str, float],
    metrics: dict[str, float],
    strengths: list[str],
    weaknesses: list[str],
    focus_metrics: list[str],
) -> dict[str, float] | None:
    """Redesign scoring weights to address weaknesses.

    Uses custom callback if provided, otherwise:
    - 1.25x for focus metrics
    - 1.15x for weakness-related metrics
    - Renormalize to sum=1
    """
    if policy.redesign_scoring_weights_fn:
        result = policy.redesign_scoring_weights_fn(
            phase, current_weights, metrics, strengths, weaknesses,
        )
        if result is not None:
            return result

    if not current_weights:
        return None

    new_weights = dict(current_weights)

    # Boost focus metrics
    for fm in focus_metrics:
        for key in new_weights:
            if fm.lower() in key.lower() or key.lower() in fm.lower():
                new_weights[key] *= 1.25

    # Boost weakness-related metrics
    for wm in weaknesses:
        for key in new_weights:
            if wm.lower() in key.lower() or key.lower() in wm.lower():
                new_weights[key] *= 1.15

    # Renormalize
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    return new_weights


def _build_analysis_report(
    phase: int,
    recommendation: str,
    recommendation_reason: str,
    scoring_assessment: str,
    strengths: list[str],
    weaknesses: list[str],
    diagnostic_gaps: list[str],
    goal_progress: dict[str, dict],
    greedy_result: GreedyResult,
    extra: dict[str, Any],
    policy: PhaseAnalysisPolicy,
) -> str:
    """Build structured text report."""
    lines = [f"======== PHASE {phase} ANALYSIS ========"]
    lines.append(f"Recommendation: {recommendation}")
    lines.append(f"Reason: {recommendation_reason}")
    lines.append(f"Scoring assessment: {scoring_assessment}")

    lines.append(f"\nAccepted: {len(greedy_result.accepted_experiments)} experiments")
    lines.append(f"Final score: {greedy_result.final_score:.4f}")
    lines.append(f"Base score: {greedy_result.base_score:.4f}")
    lines.append(f"Rounds: {len(greedy_result.rounds)}")

    if strengths:
        lines.append(f"\nStrengths: {', '.join(strengths)}")
    if weaknesses:
        lines.append(f"Weaknesses: {', '.join(weaknesses)}")
    if diagnostic_gaps:
        lines.append(f"Diagnostic Gaps: {', '.join(diagnostic_gaps)}")

    if goal_progress:
        lines.append("\nGoal Progress:")
        for metric, p in goal_progress.items():
            lines.append(
                f"  {metric}: {p['actual']:.2f} / {p['target']:.2f} "
                f"({p['pct_of_target']:.0f}%)"
            )

    if extra:
        formatted = ""
        if policy.format_extra_analysis_fn:
            formatted = policy.format_extra_analysis_fn(extra)
        if formatted:
            lines.append(f"\nExtra Analysis:\n{formatted}")

    return "\n".join(lines)
