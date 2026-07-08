"""Phase gate criteria for multi-phase regime optimization.

Each phase has success gates that must pass before advancing.
Gate failure is categorized to guide the decision loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backtests.regime.analysis.metrics import PortfolioMetrics
from backtests.regime.auto.phase_scoring import ALLOC_DIFF_FLOOR


@dataclass(frozen=True)
class GateCriterion:
    name: str
    target: float
    actual: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    passed: bool
    criteria: list[GateCriterion]
    failure_category: str | None = None
    recommendations: list[str] = field(default_factory=list)


def check_phase_gate(
    phase: int,
    metrics: PortfolioMetrics,
    regime_stats: dict,
    greedy_result: dict | None = None,
) -> GateResult:
    """Check whether a phase's success gate is met."""
    if phase == 1:
        return _gate_phase_1(metrics, regime_stats, greedy_result)
    if phase == 2:
        return _gate_phase_2(metrics, regime_stats, greedy_result)
    if phase == 3:
        return _gate_phase_3(metrics, regime_stats, greedy_result)
    if phase == 4:
        return _gate_phase_4(metrics, regime_stats, greedy_result)
    if phase == 5:
        return _gate_phase_5(metrics, regime_stats, greedy_result)
    raise ValueError(f"Unknown phase: {phase}")


def _gate_phase_1(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 1 gate: >=3 regimes >5%, transition rate >0.008, Sharpe >0.4."""
    dist = rs.get("dominant_dist", {})
    regimes_above_5 = sum(1 for v in dist.values() if v > 0.05)

    criteria = [
        GateCriterion("regimes_above_5pct", 3.0, float(regimes_above_5), regimes_above_5 >= 3),
        GateCriterion("transition_rate", 0.008, rs.get("transition_rate", 0.0), rs.get("transition_rate", 0.0) > 0.008),
        GateCriterion("sharpe", 0.4, metrics.sharpe, metrics.sharpe > 0.4),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_1_recommendations(criteria, rs),
    )


def _gate_phase_2(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 2 gate: all 4 regimes >3%, Sharpe >0.6, alloc differentiation."""
    dist = rs.get("dominant_dist", {})
    regimes_above_3 = sum(1 for v in dist.values() if v > 0.03)
    alloc_diff = rs.get("alloc_differentiation", 0.0)

    criteria = [
        GateCriterion("all_4_regimes_above_3pct", 4.0, float(regimes_above_3), regimes_above_3 >= 4),
        GateCriterion("sharpe", 0.6, metrics.sharpe, metrics.sharpe > 0.6),
        GateCriterion("alloc_differentiation", ALLOC_DIFF_FLOOR, alloc_diff, alloc_diff >= ALLOC_DIFF_FLOOR),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_2_recommendations(criteria, rs),
    )


def _gate_phase_3(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 3 gate: Sharpe >0.8, Calmar >0.5, max DD <20%, alloc diff, crisis accuracy."""
    alloc_diff = rs.get("alloc_differentiation", 0.0)
    crisis_acc = rs.get("crisis_accuracy", 0.0)

    criteria = [
        GateCriterion("sharpe", 0.8, metrics.sharpe, metrics.sharpe > 0.8),
        GateCriterion("calmar", 0.5, metrics.calmar, metrics.calmar > 0.5),
        GateCriterion("max_dd", 0.20, metrics.max_drawdown_pct, metrics.max_drawdown_pct < 0.20),
        GateCriterion("alloc_differentiation", ALLOC_DIFF_FLOOR, alloc_diff, alloc_diff >= ALLOC_DIFF_FLOOR),
        GateCriterion("crisis_accuracy", 0.30, crisis_acc, crisis_acc >= 0.30),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_3_recommendations(criteria, rs),
    )


def _gate_phase_4(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 4 gate: Sharpe >0.9, Calmar >0.8, max DD <15%, alloc diff, crisis accuracy."""
    alloc_diff = rs.get("alloc_differentiation", 0.0)
    crisis_acc = rs.get("crisis_accuracy", 0.0)

    criteria = [
        GateCriterion("sharpe", 0.9, metrics.sharpe, metrics.sharpe > 0.9),
        GateCriterion("calmar", 0.8, metrics.calmar, metrics.calmar > 0.8),
        GateCriterion("max_dd", 0.15, metrics.max_drawdown_pct, metrics.max_drawdown_pct < 0.15),
        GateCriterion("alloc_differentiation", ALLOC_DIFF_FLOOR, alloc_diff, alloc_diff >= ALLOC_DIFF_FLOOR),
        GateCriterion("crisis_accuracy", 0.30, crisis_acc, crisis_acc >= 0.30),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_4_recommendations(criteria, rs),
    )


def _gate_phase_5(metrics: PortfolioMetrics, rs: dict, gr: dict | None) -> GateResult:
    """Phase 5 gate: Sharpe >1.3, Calmar >1.0, max DD <10%, alloc diff.

    Tighter than Phase 4 because the R8 baseline already clears easily.
    """
    alloc_diff = rs.get("alloc_differentiation", 0.0)
    crisis_acc = rs.get("crisis_accuracy", 0.0)

    criteria = [
        GateCriterion("sharpe", 1.3, metrics.sharpe, metrics.sharpe > 1.3),
        GateCriterion("calmar", 1.0, metrics.calmar, metrics.calmar > 1.0),
        GateCriterion("max_dd", 0.10, metrics.max_drawdown_pct, metrics.max_drawdown_pct < 0.10),
        GateCriterion("alloc_differentiation", ALLOC_DIFF_FLOOR, alloc_diff, alloc_diff >= ALLOC_DIFF_FLOOR),
        GateCriterion("crisis_accuracy", 0.30, crisis_acc, crisis_acc >= 0.30),
    ]

    passed = all(c.passed for c in criteria)
    if passed:
        return GateResult(passed=True, criteria=criteria)

    return GateResult(
        passed=False,
        criteria=criteria,
        failure_category=_categorize_failure(criteria, rs, gr),
        recommendations=_phase_5_recommendations(criteria, rs),
    )


def _categorize_failure(
    criteria: list[GateCriterion],
    rs: dict,
    gr: dict | None,
) -> str:
    """Categorize a gate failure for the decision loop."""
    if rs.get("n_active_regimes", 0) < 2 or rs.get("transition_rate", 0.0) < 0.001:
        return "structural_issue"

    if gr is not None:
        n_rounds = gr.get("n_rounds", len(gr.get("rounds", [])))
        max_rounds = gr.get("max_rounds", 20)
        if n_rounds > 0 and n_rounds < max_rounds:
            close_count = sum(
                1
                for c in criteria
                if not c.passed and c.actual >= 0.8 * c.target
            )
            if close_count > 0:
                return "candidates_exhausted"

    return "scoring_ineffective"


def _phase_1_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "regimes_above_5pct":
            if rs.get("n_active_regimes", 0) < 2:
                recs.append("Only 2 regimes active: sticky prior may need to go even lower (try 2-3)")
            else:
                recs.append("3 regimes exist but not all exceed 5%: try a narrower sticky_diag sweep around the best value")
        elif c.name == "transition_rate":
            recs.append("Low transition rate: try a more aggressive rolling window (5y) or a lower refit_ll_tolerance")
        elif c.name == "sharpe":
            recs.append("Sharpe below target: acceptable early on, but avoid taking more regime complexity without better returns")
    return recs


def _phase_2_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "all_4_regimes_above_3pct":
            dist = rs.get("dominant_dist", {})
            weak = [k for k, v in dist.items() if v <= 0.03]
            recs.append(f"Weak regimes: {weak}; try feature combinations that separate those quadrants more clearly")
        elif c.name == "sharpe":
            recs.append("Sharpe below target: features may be adding noise, so try dropping collinear inputs")
        elif c.name == "alloc_differentiation":
            recs.append("Allocation differentiation too low: regimes produce nearly identical portfolios; try increasing posterior_temperature or reducing weight_smoothing_alpha")
    return recs


def _phase_3_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "sharpe":
            recs.append("Sharpe below target: reduce leverage or tighten the risk budget before adding more complexity")
        elif c.name == "calmar":
            recs.append("Calmar below target: revisit crisis weights, crisis z-window, or ventilator sensitivity")
        elif c.name == "max_dd":
            recs.append("Max DD too high: reduce L_max, raise sigma_floor, or speed up crisis response")
        elif c.name == "alloc_differentiation":
            recs.append("Allocation differentiation too low: regimes produce nearly identical portfolios; revisit posterior calibration or feature selection")
        elif c.name == "crisis_accuracy":
            recs.append("Crisis accuracy too low: model does not correctly classify known crises; try adjusting crisis_weights or crisis_logit_a")
    return recs


def _phase_4_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "sharpe":
            recs.append("Sharpe below target: fine-tuning is likely exhausted, so revisit Phase 3 risk controls")
        elif c.name == "calmar":
            recs.append("Calmar below target: Phase 4 should not worsen drawdown control, so retest the latest risk tweaks")
        elif c.name == "max_dd":
            recs.append("Max DD remains too high: revert the riskiest fine-tuning changes and re-check crisis settings")
        elif c.name == "alloc_differentiation":
            recs.append("Allocation differentiation collapsed during fine-tuning: revert recent weight_smoothing or posterior changes")
        elif c.name == "crisis_accuracy":
            recs.append("Crisis accuracy too low: fine-tuning degraded crisis detection; revert recent crisis_weights or logit changes")
    return recs


def _phase_5_recommendations(criteria: list[GateCriterion], rs: dict) -> list[str]:
    recs = []
    for c in criteria:
        if c.passed:
            continue
        if c.name == "sharpe":
            recs.append("Sharpe below target: budget changes alone cannot improve signal quality; consider reverting budget mutations")
        elif c.name == "calmar":
            recs.append("Calmar below target: defensive/stagflation budgets may be too aggressive; increase GLD/CASH allocations")
        elif c.name == "max_dd":
            recs.append("Max DD too high: reduce equity/crypto in Goldilocks and Reflation budgets, or increase Defensive TLT allocation")
        elif c.name == "alloc_differentiation":
            recs.append("Allocation differentiation too low: budget profiles across regimes are too similar; increase cross-regime spread")
        elif c.name == "crisis_accuracy":
            recs.append("Crisis accuracy too low: budget changes should not affect regime classification; check for HMM interference")
    return recs
