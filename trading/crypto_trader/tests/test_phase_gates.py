"""Tests for phase_gates — gate evaluation, failure categorization, scoring adjustment."""

import pytest

from crypto_trader.optimize.phase_gates import (
    _lower_is_better,
    evaluate_gate,
    suggest_scoring_adjustment,
)
from crypto_trader.optimize.types import GateCriterion, GreedyResult, ScoredCandidate, Experiment


def _make_greedy_result(
    metrics: dict[str, float],
    accepted: int = 1,
) -> GreedyResult:
    accepted_list = [
        ScoredCandidate(
            experiment=Experiment(f"exp_{i}", {}), score=0.5, metrics=metrics,
        )
        for i in range(accepted)
    ]
    return GreedyResult(
        accepted_experiments=accepted_list,
        rejected_experiments=[],
        final_mutations={},
        final_score=0.5,
        final_metrics=metrics,
    )


class TestEvaluateGate:
    def test_all_pass(self):
        result = _make_greedy_result({"total_trades": 50.0, "max_drawdown_pct": 20.0})
        criteria = [
            GateCriterion("total_trades", ">=", 30.0),
            GateCriterion("max_drawdown_pct", "<=", 45.0),
        ]
        gate = evaluate_gate(criteria, result)
        assert gate.passed is True
        assert len(gate.failure_reasons) == 0
        assert gate.failure_category is None

    def test_one_fails(self):
        result = _make_greedy_result({"total_trades": 20.0, "max_drawdown_pct": 20.0})
        criteria = [
            GateCriterion("total_trades", ">=", 30.0),
            GateCriterion("max_drawdown_pct", "<=", 45.0),
        ]
        gate = evaluate_gate(criteria, result)
        assert gate.passed is False
        assert len(gate.failure_reasons) == 1
        assert "total_trades" in gate.failure_reasons[0]
        assert gate.failure_category is not None

    def test_all_fail(self):
        result = _make_greedy_result({"total_trades": 10.0, "max_drawdown_pct": 50.0})
        criteria = [
            GateCriterion("total_trades", ">=", 30.0),
            GateCriterion("max_drawdown_pct", "<=", 45.0),
        ]
        gate = evaluate_gate(criteria, result)
        assert gate.passed is False
        assert len(gate.failure_reasons) == 2

    def test_operators(self):
        result = _make_greedy_result({"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0})
        assert evaluate_gate([GateCriterion("a", ">", 4.0)], result).passed
        assert not evaluate_gate([GateCriterion("a", ">", 5.0)], result).passed
        assert evaluate_gate([GateCriterion("b", "<", 6.0)], result).passed
        assert not evaluate_gate([GateCriterion("b", "<", 5.0)], result).passed


class TestFailureCategory:
    def test_candidates_exhausted(self):
        """0 accepted experiments -> candidates_exhausted."""
        result = GreedyResult(
            accepted_experiments=[],
            rejected_experiments=[],
            final_mutations={},
            final_score=0.0,
            final_metrics={"total_trades": 10.0},
        )
        gate = evaluate_gate([GateCriterion("total_trades", ">=", 30.0)], result)
        assert gate.failure_category == "candidates_exhausted"

    def test_structural_issue_higher_is_better(self):
        """Actual < 0.5x target for higher-is-better -> structural_issue."""
        result = _make_greedy_result({"total_trades": 10.0})
        gate = evaluate_gate([GateCriterion("total_trades", ">=", 30.0)], result)
        assert gate.failure_category == "structural_issue"

    def test_structural_issue_lower_is_better(self):
        """Actual > 2x target for lower-is-better -> structural_issue."""
        result = _make_greedy_result({"max_drawdown_pct": 95.0})
        gate = evaluate_gate([GateCriterion("max_drawdown_pct", "<=", 30.0)], result)
        assert gate.failure_category == "structural_issue"

    def test_diagnostic_needed(self):
        """2+ near-miss criteria -> diagnostic_needed."""
        # Both within 15% of target (higher-is-better: 27/30=90%, 1.35/1.5=90%)
        result = _make_greedy_result(
            {"total_trades": 27.0, "profit_factor": 1.35}
        )
        criteria = [
            GateCriterion("total_trades", ">=", 30.0),
            GateCriterion("profit_factor", ">=", 1.5),
        ]
        gate = evaluate_gate(criteria, result)
        assert gate.failure_category == "diagnostic_needed"

    def test_scoring_ineffective_default(self):
        """Single non-structural failure -> scoring_ineffective."""
        # 20/30 = 67%, not structural (>50%) and only 1 near-miss
        result = _make_greedy_result({"total_trades": 20.0})
        gate = evaluate_gate([GateCriterion("total_trades", ">=", 30.0)], result)
        assert gate.failure_category == "scoring_ineffective"

    def test_passed_has_no_category(self):
        result = _make_greedy_result({"total_trades": 50.0})
        gate = evaluate_gate([GateCriterion("total_trades", ">=", 30.0)], result)
        assert gate.failure_category is None


class TestLowerIsBetter:
    def test_drawdown_metrics(self):
        assert _lower_is_better("max_drawdown_pct") is True
        assert _lower_is_better("max_drawdown_duration") is True

    def test_mae(self):
        assert _lower_is_better("avg_mae_r") is True

    def test_higher_is_better(self):
        assert _lower_is_better("total_trades") is False
        assert _lower_is_better("sharpe_ratio") is False
        assert _lower_is_better("win_rate") is False


class TestScoringAdjustment:
    def test_adjusts_weights_on_failure(self):
        result = _make_greedy_result({"total_trades": 10.0})
        gate = evaluate_gate([GateCriterion("total_trades", ">=", 30.0)], result)
        weights = {"coverage": 0.3, "risk": 0.3, "edge": 0.4}
        new_weights = suggest_scoring_adjustment(gate, weights)
        # Should adjust some weights
        assert new_weights != weights or gate.failure_category == "scoring_ineffective"

    def test_weights_sum_to_one(self):
        result = _make_greedy_result({"max_drawdown_pct": 50.0})
        gate = evaluate_gate([GateCriterion("max_drawdown_pct", "<=", 30.0)], result)
        weights = {"coverage": 0.3, "risk": 0.3, "edge": 0.4}
        new_weights = suggest_scoring_adjustment(gate, weights)
        assert sum(new_weights.values()) == pytest.approx(1.0)
