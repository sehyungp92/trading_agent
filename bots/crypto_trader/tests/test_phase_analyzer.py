"""Tests for phase_analyzer — goal progress, scoring assessment, policy callbacks, decisions."""

import pytest

from crypto_trader.optimize.phase_analyzer import (
    EFFECTIVE,
    INEFFECTIVE,
    MARGINAL,
    MISALIGNED,
    analyze_phase,
    _compute_goal_progress,
    _assess_scoring,
    _fallback_recommendation,
    _redesign_weights,
)
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.optimize.types import (
    Experiment,
    GateCriterion,
    GateResult,
    GreedyResult,
    GreedyRound,
    PhaseAnalysisPolicy,
    PhaseDecision,
    ScoredCandidate,
)


def _make_greedy_result(
    accepted: int = 1,
    score: float = 0.5,
    base_score: float = 0.1,
    metrics: dict | None = None,
) -> GreedyResult:
    accepted_list = [
        ScoredCandidate(
            experiment=Experiment(f"exp_{i}", {}), score=score, metrics=metrics or {},
        )
        for i in range(accepted)
    ]
    return GreedyResult(
        accepted_experiments=accepted_list,
        rejected_experiments=[],
        final_mutations={},
        final_score=score,
        final_metrics=metrics or {},
        base_score=base_score,
        rounds=[GreedyRound(1, 5, "exp_0", score, 50.0, True)],
    )


def _make_gate_result(
    passed: bool = True,
    failure_category: str | None = None,
) -> GateResult:
    return GateResult(
        passed=passed,
        criteria_results=[],
        failure_reasons=[] if passed else ["some failure"],
        failure_category=failure_category,
    )


class TestComputeGoalProgress:
    def test_higher_is_better(self):
        progress = _compute_goal_progress(
            {"total_trades": 40.0}, {"total_trades": 80.0}
        )
        assert progress["total_trades"]["pct_of_target"] == pytest.approx(50.0)

    def test_lower_is_better(self):
        progress = _compute_goal_progress(
            {"max_drawdown_pct": 15.0}, {"max_drawdown_pct": 20.0}
        )
        # 20/15 * 100 = 133.3% (better than target)
        assert progress["max_drawdown_pct"]["pct_of_target"] > 100.0

    def test_clamped_at_200(self):
        progress = _compute_goal_progress(
            {"total_trades": 200.0}, {"total_trades": 80.0}
        )
        assert progress["total_trades"]["pct_of_target"] == 200.0

    def test_zero_target(self):
        progress = _compute_goal_progress(
            {"metric": 5.0}, {"metric": 0.0}
        )
        assert progress["metric"]["pct_of_target"] == 100.0

    def test_includes_target_and_actual(self):
        progress = _compute_goal_progress(
            {"trades": 40.0}, {"trades": 80.0}
        )
        assert progress["trades"]["target"] == 80.0
        assert progress["trades"]["actual"] == 40.0


class TestAssessScoring:
    def test_effective_when_gate_passes(self):
        result = _assess_scoring(
            _make_greedy_result(), _make_gate_result(passed=True),
            {}, PhaseAnalysisPolicy(), PhaseState(), 1,
        )
        assert result == EFFECTIVE

    def test_ineffective_when_no_accepted(self):
        gr = _make_greedy_result(accepted=0)
        result = _assess_scoring(
            gr, _make_gate_result(passed=False),
            {}, PhaseAnalysisPolicy(), PhaseState(), 1,
        )
        assert result == INEFFECTIVE

    def test_ineffective_when_candidates_exhausted(self):
        result = _assess_scoring(
            _make_greedy_result(),
            _make_gate_result(passed=False, failure_category="candidates_exhausted"),
            {}, PhaseAnalysisPolicy(), PhaseState(), 1,
        )
        assert result == INEFFECTIVE

    def test_marginal_when_diagnostic_needed(self):
        result = _assess_scoring(
            _make_greedy_result(),
            _make_gate_result(passed=False, failure_category="diagnostic_needed"),
            {}, PhaseAnalysisPolicy(), PhaseState(), 1,
        )
        assert result == MARGINAL

    def test_marginal_when_low_delta(self):
        gr = _make_greedy_result(score=0.101, base_score=0.1)
        result = _assess_scoring(
            gr, _make_gate_result(passed=False),
            {}, PhaseAnalysisPolicy(min_effective_score_delta_pct=0.05),
            PhaseState(), 1,
        )
        assert result == MARGINAL

    def test_misaligned_when_focus_not_improving(self):
        state = PhaseState()
        state.phase_metrics[1] = {"win_rate": 60.0}
        policy = PhaseAnalysisPolicy(focus_metrics=["win_rate"])

        gr = _make_greedy_result(score=0.5, base_score=0.1)
        result = _assess_scoring(
            gr, _make_gate_result(passed=False),
            {"win_rate": 55.0},  # Lower than phase 1's 60.0
            policy, state, 2,
        )
        assert result == MISALIGNED


class TestFallbackRecommendation:
    def test_advance_when_gate_passed(self):
        rec, reason = _fallback_recommendation(
            1, _make_gate_result(passed=True),
            _make_greedy_result(), EFFECTIVE, [], PhaseState(), 2, 1,
        )
        assert rec == "advance"

    def test_improve_diagnostics_on_diagnostic_needed(self):
        rec, reason = _fallback_recommendation(
            1, _make_gate_result(passed=False, failure_category="diagnostic_needed"),
            _make_greedy_result(), MARGINAL, [], PhaseState(), 2, 1,
        )
        assert rec == "improve_diagnostics"

    def test_improve_scoring_on_ineffective(self):
        rec, reason = _fallback_recommendation(
            1, _make_gate_result(passed=False),
            _make_greedy_result(), INEFFECTIVE, [], PhaseState(), 2, 1,
        )
        assert rec == "improve_scoring"

    def test_improve_scoring_on_marginal(self):
        rec, reason = _fallback_recommendation(
            1, _make_gate_result(passed=False),
            _make_greedy_result(), MARGINAL, [], PhaseState(), 2, 1,
        )
        assert rec == "improve_scoring"

    def test_advance_when_budget_exhausted(self):
        state = PhaseState()
        state.scoring_retries[1] = 2
        state.diagnostic_retries[1] = 1
        rec, reason = _fallback_recommendation(
            1, _make_gate_result(passed=False),
            _make_greedy_result(), MARGINAL, [], state, 2, 1,
        )
        assert rec == "advance"
        assert "exhausted" in reason.lower()


class TestRedesignWeights:
    def test_default_boost(self):
        weights = {"coverage": 0.3, "risk": 0.3, "edge": 0.4}
        policy = PhaseAnalysisPolicy()
        new = _redesign_weights(
            policy, 1, weights, {},
            strengths=[], weaknesses=["coverage"],
            focus_metrics=[],
        )
        # Coverage should be boosted by 1.15x relative to others
        assert new is not None
        assert sum(new.values()) == pytest.approx(1.0)

    def test_focus_metrics_boosted(self):
        weights = {"coverage": 0.3, "risk": 0.3, "edge": 0.4}
        policy = PhaseAnalysisPolicy()
        new = _redesign_weights(
            policy, 1, weights, {},
            strengths=[], weaknesses=[],
            focus_metrics=["coverage"],
        )
        assert new is not None
        # Coverage boosted by 1.25x
        assert new["coverage"] > weights["coverage"]

    def test_custom_callback(self):
        custom_weights = {"a": 0.5, "b": 0.5}
        policy = PhaseAnalysisPolicy(
            redesign_scoring_weights_fn=lambda *args: custom_weights,
        )
        new = _redesign_weights(
            policy, 1, {"c": 1.0}, {},
            strengths=[], weaknesses=[],
            focus_metrics=[],
        )
        assert new == custom_weights

    def test_empty_weights_returns_none(self):
        policy = PhaseAnalysisPolicy()
        new = _redesign_weights(
            policy, 1, {}, {}, [], [], [],
        )
        assert new is None


class TestAnalyzePhase:
    def test_advance_on_gate_pass(self):
        analysis = analyze_phase(
            1, _make_greedy_result(), {"total_trades": 50.0},
            PhaseState(), _make_gate_result(passed=True),
            ultimate_targets={"total_trades": 80.0},
        )
        assert analysis.recommendation == "advance"
        assert analysis.phase == 1
        assert len(analysis.goal_progress) == 1

    def test_retry_on_gate_failure(self):
        analysis = analyze_phase(
            1, _make_greedy_result(), {"total_trades": 5.0},
            PhaseState(), _make_gate_result(passed=False),
            ultimate_targets={"total_trades": 80.0},
        )
        assert analysis.recommendation in ("improve_scoring", "improve_diagnostics")

    def test_strengths_and_weaknesses(self):
        analysis = analyze_phase(
            1, _make_greedy_result(),
            {"total_trades": 70.0, "win_rate": 20.0},
            PhaseState(), _make_gate_result(passed=True),
            ultimate_targets={"total_trades": 80.0, "win_rate": 55.0},
        )
        assert "total_trades" in analysis.strengths  # 70/80 = 87.5%
        assert "win_rate" in analysis.weaknesses  # 20/55 = 36.4%

    def test_report_not_empty(self):
        analysis = analyze_phase(
            1, _make_greedy_result(), {},
            PhaseState(), _make_gate_result(passed=True),
        )
        assert analysis.report != ""
        assert analysis.summary != ""
        assert "PHASE 1" in analysis.report

    def test_custom_decide_action_fn(self):
        def custom_decision(*args):
            return PhaseDecision(
                action="improve_scoring",
                reason="custom decision",
                scoring_weight_overrides={"custom": 1.0},
            )

        policy = PhaseAnalysisPolicy(decide_action_fn=custom_decision)
        analysis = analyze_phase(
            1, _make_greedy_result(), {},
            PhaseState(), _make_gate_result(passed=False),
            policy=policy,
        )
        assert analysis.recommendation == "improve_scoring"
        assert analysis.scoring_weight_overrides == {"custom": 1.0}

    def test_custom_decision_budget_check(self):
        """Custom decision requesting improve_scoring with exhausted budget -> advance."""
        def custom_decision(*args):
            return PhaseDecision(
                action="improve_scoring",
                reason="custom",
            )

        state = PhaseState()
        state.scoring_retries[1] = 2
        policy = PhaseAnalysisPolicy(decide_action_fn=custom_decision)

        analysis = analyze_phase(
            1, _make_greedy_result(), {},
            state, _make_gate_result(passed=False),
            policy=policy,
            max_scoring_retries=2,
        )
        assert analysis.recommendation == "advance"

    def test_diagnostic_gap_callback(self):
        def gap_fn(phase, metrics):
            return ["missing_volume_data", "no_funding_rate"]

        policy = PhaseAnalysisPolicy(diagnostic_gap_fn=gap_fn)
        analysis = analyze_phase(
            1, _make_greedy_result(), {},
            PhaseState(), _make_gate_result(passed=True),
            policy=policy,
        )
        assert "missing_volume_data" in analysis.diagnostic_gaps

    def test_suggest_experiments_callback(self):
        def suggest_fn(phase, metrics, weaknesses, state):
            return [Experiment("SUGGESTED_1", {"x": 1})]

        policy = PhaseAnalysisPolicy(suggest_experiments_fn=suggest_fn)
        analysis = analyze_phase(
            1, _make_greedy_result(), {},
            PhaseState(), _make_gate_result(passed=True),
            policy=policy,
        )
        assert len(analysis.suggested_experiments) == 1
        assert analysis.suggested_experiments[0].name == "SUGGESTED_1"

    def test_scoring_weight_overrides_on_improve_scoring(self):
        """When recommending improve_scoring, weight overrides are generated."""
        analysis = analyze_phase(
            1, _make_greedy_result(), {},
            PhaseState(), _make_gate_result(passed=False),
            current_weights={"coverage": 0.5, "risk": 0.5},
        )
        if analysis.recommendation == "improve_scoring":
            assert analysis.scoring_weight_overrides is not None
