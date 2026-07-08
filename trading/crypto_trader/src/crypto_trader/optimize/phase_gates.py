"""Phase gate evaluation — determines if a phase's results meet quality criteria."""

from __future__ import annotations

from crypto_trader.optimize.types import GateCriterion, GateResult, GreedyResult


def evaluate_gate(
    criteria: list[GateCriterion],
    greedy_result: GreedyResult,
) -> GateResult:
    """Evaluate gate criteria against final metrics from greedy optimization.

    Returns GateResult with pass/fail for each criterion, overall verdict,
    and failure_category if the gate failed.
    """
    results: list[tuple[GateCriterion, float, bool]] = []
    failures: list[str] = []

    metrics = greedy_result.final_metrics

    for criterion in criteria:
        actual = metrics.get(criterion.metric, 0.0)
        passed = _check_operator(actual, criterion.operator, criterion.threshold)
        results.append((criterion, actual, passed))

        if not passed:
            failures.append(
                f"{criterion.metric}: {actual:.4f} {criterion.operator} "
                f"{criterion.threshold:.4f} FAILED"
            )

    gate_passed = len(failures) == 0
    category = None
    if not gate_passed:
        category = _categorize_failure(results, greedy_result)

    return GateResult(
        passed=gate_passed,
        criteria_results=results,
        failure_reasons=failures,
        failure_category=category,
    )


def _categorize_failure(
    criteria_results: list[tuple[GateCriterion, float, bool]],
    greedy_result: GreedyResult,
) -> str:
    """Categorize gate failure to guide retry strategy.

    Categories:
    - "structural_issue": any failing criterion is >2x or <0.5x target
    - "candidates_exhausted": greedy_result has 0 accepted experiments
    - "diagnostic_needed": 2+ criteria are near-miss (within 15% of target)
    - "scoring_ineffective": default
    """
    if not greedy_result.accepted_experiments:
        return "candidates_exhausted"

    failing = [(c, actual) for c, actual, passed in criteria_results if not passed]

    # Check for structural issues: far from target
    for criterion, actual in failing:
        target = criterion.threshold
        if target == 0:
            continue
        if _lower_is_better(criterion.metric):
            # For lower-is-better, structural if actual > 2x target
            if actual > 2.0 * target:
                return "structural_issue"
        else:
            # For higher-is-better, structural if actual < 0.5x target
            if actual < 0.5 * target:
                return "structural_issue"

    # Check for near-miss: within 15% of target
    near_miss_count = 0
    for criterion, actual in failing:
        target = criterion.threshold
        if target == 0:
            continue
        if _lower_is_better(criterion.metric):
            ratio = actual / target if target > 0 else 1.0
            if ratio <= 1.15:
                near_miss_count += 1
        else:
            ratio = actual / target if target > 0 else 0.0
            if ratio >= 0.85:
                near_miss_count += 1

    if near_miss_count >= 2:
        return "diagnostic_needed"

    return "scoring_ineffective"


def _lower_is_better(name: str) -> bool:
    """Determine if a metric is 'lower is better' based on name tokens."""
    lower = name.lower()
    if "drawdown" in lower or "latency" in lower:
        return True
    if lower in ("avg_mae_r",):
        return True
    return False


def suggest_scoring_adjustment(
    gate_result: GateResult,
    current_weights: dict[str, float],
) -> dict[str, float]:
    """Suggest scoring weight adjustments based on failure category.

    Returns new weights dict. Boosts dimensions related to failures.
    """
    category = gate_result.failure_category or "scoring_ineffective"
    new_weights = dict(current_weights)

    # Map old categories to dimensions for backward compat
    _CATEGORY_TO_DIMENSION = {
        "candidates_exhausted": "coverage",
        "structural_issue": "risk",
        "diagnostic_needed": "edge",
        "scoring_ineffective": "edge",
    }

    dim = _CATEGORY_TO_DIMENSION.get(category)
    if dim and dim in new_weights:
        _boost_dimension(new_weights, dim, 0.10)

    # Re-normalize to sum to 1.0
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    return new_weights


def _check_operator(actual: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return actual >= threshold
    elif operator == "<=":
        return actual <= threshold
    elif operator == ">":
        return actual > threshold
    elif operator == "<":
        return actual < threshold
    return False


def _boost_dimension(weights: dict[str, float], dim: str, amount: float) -> None:
    """Boost a dimension's weight, reducing others proportionally."""
    if dim not in weights:
        return
    old_val = weights[dim]
    weights[dim] = min(old_val + amount, 0.60)
    added = weights[dim] - old_val
    # Reduce others proportionally
    others = {k: v for k, v in weights.items() if k != dim and v > 0}
    if others:
        reduction_total = sum(others.values())
        for k in others:
            weights[k] -= added * (others[k] / reduction_total)
            weights[k] = max(weights[k], 0.01)
