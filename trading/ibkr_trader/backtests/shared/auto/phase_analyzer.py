from __future__ import annotations

import json

from .greedy_optimizer import _delta_ratio as _score_delta_ratio
from .phase_gates import _lower_is_better
from .phase_state import PhaseState
from .plugin import PhaseAnalysisPolicy
from .types import Experiment, GateResult, GreedyResult, PhaseAnalysis


def _metric_value(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))


def _target_progress(name: str, actual: float, target: float) -> float:
    if target == 0:
        return 0.0
    if _lower_is_better(name):
        if actual <= 0:
            return 200.0
        return min((target / actual) * 100.0, 200.0)
    return min((actual / target) * 100.0, 200.0)


def _metric_improved(name: str, previous: float, current: float) -> bool:
    if _lower_is_better(name):
        return current < previous
    return current > previous


def _focus_metrics_far_from_target(
    focus_metrics: list[str],
    goal_progress: dict[str, dict],
) -> bool:
    if not focus_metrics:
        return False
    below_target = sum(
        1
        for metric_name in focus_metrics
        if goal_progress.get(metric_name, {}).get("pct_of_target", 0.0) < 60.0
    )
    threshold = max(1, (len(focus_metrics) + 1) // 2)
    return below_target >= threshold


def _assess_scoring(
    greedy_result: GreedyResult,
    gate_result: GateResult,
    *,
    score_delta_ratio: float,
    focus_metrics: list[str],
    goal_progress: dict[str, dict],
    previous_metrics: dict[str, float] | None,
    metrics: dict[str, float],
    min_effective_score_delta_pct: float,
) -> str:
    if gate_result.passed:
        return "EFFECTIVE"

    if greedy_result.accepted_count == 0 or gate_result.failure_category == "candidates_exhausted":
        return "INEFFECTIVE"

    if score_delta_ratio < min_effective_score_delta_pct:
        return "MARGINAL"

    if _focus_metrics_far_from_target(focus_metrics, goal_progress):
        return "MISALIGNED"

    if previous_metrics and focus_metrics:
        improved_focus = sum(
            1
            for metric_name in focus_metrics
            if _metric_improved(
                metric_name,
                _metric_value(previous_metrics, metric_name),
                _metric_value(metrics, metric_name),
            )
        )
        threshold = max(1, (len(focus_metrics) + 1) // 2)
        if improved_focus < threshold:
            return "MISALIGNED"

    if gate_result.failure_category == "diagnostic_needed":
        return "MARGINAL"

    if gate_result.failure_category == "scoring_ineffective":
        return "MISALIGNED"

    return "EFFECTIVE"


def _decision_is_feasible(
    action: str,
    *,
    scoring_retries: int,
    diagnostic_retries: int,
    max_scoring_retries: int,
    max_diagnostic_retries: int,
) -> bool:
    if action == "improve_scoring":
        return scoring_retries < max_scoring_retries
    if action == "improve_diagnostics":
        return diagnostic_retries < max_diagnostic_retries
    return True


def _fallback_recommendation(
    analysis: PhaseAnalysis,
    gate_result: GateResult,
    *,
    scoring_retries: int,
    diagnostic_retries: int,
    max_scoring_retries: int,
    max_diagnostic_retries: int,
) -> tuple[str, str]:
    if gate_result.passed:
        return "advance", "Gate passed."

    if gate_result.failure_category == "diagnostic_needed" and diagnostic_retries < max_diagnostic_retries:
        return (
            "improve_diagnostics",
            "Multiple gate criteria are near target; rerun enhanced diagnostics before changing the score.",
        )

    if analysis.scoring_assessment in {"MISALIGNED", "INEFFECTIVE"} and scoring_retries < max_scoring_retries:
        return "improve_scoring", "Greedy score is not aligned strongly enough with phase goals."

    if (
        analysis.scoring_assessment == "MARGINAL"
        and scoring_retries < max_scoring_retries
        and gate_result.failure_category != "diagnostic_needed"
    ):
        return "improve_scoring", "Greedy score is not aligned strongly enough with phase goals."

    if analysis.diagnostic_gaps and diagnostic_retries < max_diagnostic_retries:
        return "improve_diagnostics", "Phase weaknesses need deeper diagnostics."

    if analysis.suggested_experiments:
        return "advance", "Retry budget exhausted; carry suggested experiments into the next phase."
    return "advance", "Retry budget exhausted; advance with current best mutations."


def analyze_phase(
    phase: int,
    greedy_result: GreedyResult,
    metrics: dict[str, float],
    state: PhaseState,
    gate_result: GateResult,
    *,
    ultimate_targets: dict[str, float],
    policy: PhaseAnalysisPolicy,
    current_weights: dict[str, float] | None = None,
    max_scoring_retries: int = 2,
    max_diagnostic_retries: int = 1,
) -> PhaseAnalysis:
    analysis = PhaseAnalysis(phase=phase)

    for metric_name, target in ultimate_targets.items():
        actual = _metric_value(metrics, metric_name)
        analysis.goal_progress[metric_name] = {
            "target": target,
            "actual": actual,
            "pct_of_target": _target_progress(metric_name, actual, target),
        }

    for metric_name, progress in analysis.goal_progress.items():
        label = f"{metric_name}: {progress['actual']:.4f} ({progress['pct_of_target']:.0f}% of {progress['target']:.4f})"
        if progress["pct_of_target"] >= 80.0:
            analysis.strengths.append(label)
        elif progress["pct_of_target"] < 60.0:
            analysis.weaknesses.append(label)

    previous_metrics = state.get_phase_metrics(phase - 1) if phase > 1 else None
    score_delta_ratio = _score_delta_ratio(greedy_result.final_score, greedy_result.base_score)
    analysis.scoring_assessment = _assess_scoring(
        greedy_result,
        gate_result,
        score_delta_ratio=score_delta_ratio,
        focus_metrics=policy.focus_metrics,
        goal_progress=analysis.goal_progress,
        previous_metrics=previous_metrics,
        metrics=metrics,
        min_effective_score_delta_pct=policy.min_effective_score_delta_pct,
    )

    analysis.diagnostic_gaps = _dedupe_strings(
        policy.diagnostic_gap_fn(phase, metrics)
        if policy.diagnostic_gap_fn
        else []
    )
    analysis.suggested_experiments = _dedupe_experiments(
        policy.suggest_experiments_fn(phase, metrics, analysis.weaknesses, state)
        if policy.suggest_experiments_fn
        else []
    )

    if policy.build_extra_analysis_fn:
        analysis.extra = policy.build_extra_analysis_fn(phase, metrics, state, greedy_result)

    scoring_retries = state.scoring_retries.get(phase, 0)
    diagnostic_retries = state.diagnostic_retries.get(phase, 0)
    decision = None
    if policy.decide_action_fn:
        proposed_decision = policy.decide_action_fn(
            phase,
            metrics,
            state,
            greedy_result,
            gate_result,
            current_weights,
            analysis,
            max_scoring_retries,
            max_diagnostic_retries,
        )
        if proposed_decision:
            if proposed_decision.scoring_assessment_override:
                analysis.scoring_assessment = proposed_decision.scoring_assessment_override
            analysis.diagnostic_gaps = _dedupe_strings(
                [*analysis.diagnostic_gaps, *proposed_decision.extra_diagnostic_gaps]
            )
            analysis.suggested_experiments = _dedupe_experiments(
                [*analysis.suggested_experiments, *proposed_decision.extra_suggested_experiments]
            )
            if _decision_is_feasible(
                proposed_decision.action,
                scoring_retries=scoring_retries,
                diagnostic_retries=diagnostic_retries,
                max_scoring_retries=max_scoring_retries,
                max_diagnostic_retries=max_diagnostic_retries,
            ):
                decision = proposed_decision

    if decision:
        analysis.recommendation = decision.action
        analysis.recommendation_reason = decision.reason
        if analysis.recommendation == "improve_scoring":
            analysis.scoring_weight_overrides = decision.scoring_weight_overrides or _redesign_weights(
                policy,
                phase,
                current_weights,
                analysis,
                gate_result,
            )
    else:
        analysis.recommendation, analysis.recommendation_reason = _fallback_recommendation(
            analysis,
            gate_result,
            scoring_retries=scoring_retries,
            diagnostic_retries=diagnostic_retries,
            max_scoring_retries=max_scoring_retries,
            max_diagnostic_retries=max_diagnostic_retries,
        )
        if analysis.recommendation == "improve_scoring":
            analysis.scoring_weight_overrides = _redesign_weights(
                policy,
                phase,
                current_weights,
                analysis,
                gate_result,
            )

    analysis.report = _build_analysis_report(analysis, policy)
    return analysis


def _redesign_weights(
    policy: PhaseAnalysisPolicy,
    phase: int,
    current_weights: dict[str, float] | None,
    analysis: PhaseAnalysis,
    gate_result: GateResult,
) -> dict[str, float] | None:
    if policy.redesign_scoring_weights_fn:
        redesigned = policy.redesign_scoring_weights_fn(
            phase,
            current_weights,
            analysis,
            gate_result,
        )
        if redesigned is not None:
            return redesigned

    if not current_weights:
        return None

    weights = dict(current_weights)
    for metric_name in policy.focus_metrics:
        if metric_name in weights:
            weights[metric_name] *= 1.25
    for weakness in analysis.weaknesses:
        name = weakness.split(":", 1)[0]
        if name in weights:
            weights[name] *= 1.15
    total = sum(weights.values())
    if total > 0:
        weights = {key: value / total for key, value in weights.items()}
    return weights


def _build_analysis_report(analysis: PhaseAnalysis, policy: PhaseAnalysisPolicy) -> str:
    lines = [
        "=" * 70,
        f"PHASE {analysis.phase} ANALYSIS",
        "=" * 70,
        "",
        f"Recommendation: {analysis.recommendation}",
        f"Reason: {analysis.recommendation_reason}",
        f"Scoring assessment: {analysis.scoring_assessment}",
    ]

    if analysis.strengths:
        lines.extend(["", "Strengths:"])
        lines.extend(f"  + {entry}" for entry in analysis.strengths)

    if analysis.weaknesses:
        lines.extend(["", "Weaknesses:"])
        lines.extend(f"  - {entry}" for entry in analysis.weaknesses)

    if analysis.diagnostic_gaps:
        lines.extend(["", "Diagnostic Gaps:"])
        lines.extend(f"  ! {entry}" for entry in analysis.diagnostic_gaps)

    if analysis.suggested_experiments:
        lines.extend(["", f"Suggested Experiments ({len(analysis.suggested_experiments)}):"])
        lines.extend(
            f"  - {experiment.name}: {json.dumps(experiment.mutations, default=str)}"
            for experiment in analysis.suggested_experiments[:10]
        )

    if analysis.extra:
        lines.extend(["", "Extra Analysis:"])
        if policy.format_extra_analysis_fn:
            lines.extend(f"  {line}" for line in policy.format_extra_analysis_fn(analysis.extra))
        else:
            lines.extend(
                f"  {line}"
                for line in json.dumps(analysis.extra, indent=2, default=str).splitlines()
            )

    lines.extend(["", "Goal Progress:"])
    for metric_name, progress in analysis.goal_progress.items():
        lines.append(
            f"  {metric_name}: {progress['actual']:.4f} / {progress['target']:.4f} "
            f"({progress['pct_of_target']:.0f}%)"
        )
    return "\n".join(lines)


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _dedupe_experiments(experiments: list[Experiment]) -> list[Experiment]:
    deduped: list[Experiment] = []
    seen: set[str] = set()
    for experiment in experiments:
        if experiment.name in seen:
            continue
        seen.add(experiment.name)
        deduped.append(experiment)
    return deduped
