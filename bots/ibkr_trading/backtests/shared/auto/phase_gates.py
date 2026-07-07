from __future__ import annotations

from .types import GateCriterion, GateResult, GreedyResult

FAILURE_CATEGORIES = (
    "structural_issue",
    "candidates_exhausted",
    "scoring_ineffective",
    "diagnostic_needed",
)

_LOWER_IS_BETTER_TOKENS = (
    "drawdown",
    "latency",
    "false_positive",
    "right_then_lost",
    "top5_winner_share",
    "hold_bars",
)


def _lower_is_better(name: str) -> bool:
    normalized = name.lower()
    if normalized.startswith("max_") or normalized.endswith("_max"):
        return True
    parts = normalized.replace(".", "_").split("_")
    if "dd" in parts:
        return True
    return any(token in normalized for token in _LOWER_IS_BETTER_TOKENS)


def evaluate_gate(
    criteria: list[GateCriterion],
    greedy_result: GreedyResult | None = None,
) -> GateResult:
    if all(criterion.passed for criterion in criteria):
        return GateResult(passed=True, criteria=tuple(criteria))

    category = _categorize_failure(criteria, greedy_result)
    return GateResult(
        passed=False,
        criteria=tuple(criteria),
        failure_category=category,
        recommendations=tuple(_generate_recommendations(criteria, category)),
    )


def _categorize_failure(
    criteria: list[GateCriterion],
    greedy_result: GreedyResult | None,
) -> str:
    for criterion in criteria:
        if criterion.passed or criterion.target == 0:
            continue
        if criterion.name.startswith("hard_"):
            return "structural_issue"
        if _lower_is_better(criterion.name):
            if criterion.actual > criterion.target * 2.0:
                return "structural_issue"
        elif criterion.actual < criterion.target * 0.5:
            return "structural_issue"

    if greedy_result and greedy_result.accepted_count == 0:
        return "candidates_exhausted"

    near_miss = 0
    for criterion in criteria:
        if criterion.passed or criterion.target == 0:
            continue
        if criterion.name.startswith("hard_"):
            continue
        if _lower_is_better(criterion.name):
            if criterion.actual <= criterion.target * 1.15:
                near_miss += 1
        elif criterion.actual >= criterion.target * 0.85:
            near_miss += 1
    if near_miss >= 2:
        return "diagnostic_needed"

    return "scoring_ineffective"


def _generate_recommendations(criteria: list[GateCriterion], failure_category: str) -> list[str]:
    failing = [criterion for criterion in criteria if not criterion.passed]

    if failure_category == "structural_issue":
        return [f"{criterion.name} is far from target; revisit structural parameters." for criterion in failing]
    if failure_category == "candidates_exhausted":
        return ["No candidates were accepted; expand or reshape the experiment pool."]
    if failure_category == "diagnostic_needed":
        return ["Multiple criteria are close to target; inspect deeper diagnostics before re-scoring."] + [
            f"{criterion.name}: actual={criterion.actual:.4f}, target={criterion.target:.4f}"
            for criterion in failing
        ]
    return ["Score moved more than gate metrics; rebalance scoring weights toward failing criteria."]
