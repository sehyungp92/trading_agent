from __future__ import annotations

from trading_assistant_backtest.auto.greedy_optimizer import (
    best_passing_candidate,
    no_adoption_reason,
)
from trading_assistant_backtest.auto.phase_runner import run_phase
from trading_assistant_backtest.auto.types import Candidate, CandidateEvaluation, PhaseSpec
from trading_assistant_backtest.repair.ablation import build_ablation_matrix
from trading_assistant_backtest.repair.confirmatory import (
    build_confirmatory_variants,
    variant_payload,
)
from trading_assistant_backtest.repair.failure_analysis import analyze_failure
from trading_assistant_backtest.repair.perturbation import build_local_perturbations
from trading_assistant_backtest.repair.rollback import build_rollback_candidates
from trading_assistant_backtest.repair.targeted import build_targeted_candidates
from trading_assistant_backtest.repair.trigger import evaluate_selection_oos_repair_trigger


def test_phase_runner_retains_diagnostic_rejections() -> None:
    evaluations = run_phase(
        PhaseSpec(
            phase_id="signal_quality",
            candidate_families=["filter_repair", "filter_repair", "exit_repair"],
        )
    )

    assert [item.candidate.candidate_id for item in evaluations] == [
        "signal_quality-filter_repair-1",
        "signal_quality-exit_repair-2",
    ]
    assert all(item.passed is False for item in evaluations)
    assert "replay-backed evaluator" in no_adoption_reason(evaluations, "unused")


def test_greedy_selector_uses_passing_score_only() -> None:
    weak_pass = CandidateEvaluation(Candidate("weak", "family"), 0.1, True)
    strong_pass = CandidateEvaluation(Candidate("strong", "family"), 0.5, True)
    failed_high_score = CandidateEvaluation(Candidate("failed", "family"), 2.0, False)

    assert best_passing_candidate([weak_pass, failed_high_score, strong_pass]) == strong_pass


def test_failure_analysis_classifies_missing_replay_evaluator() -> None:
    analysis = analyze_failure(
        "run-1",
        rejected_candidates=[
            {
                "candidate_id": "signal_quality-filter_repair-1",
                "reason": "strategy plugin has not provided a replay-backed evaluator",
            }
        ],
        repair_triggered=True,
    )

    assert analysis["primary_failure"] == "missing_replay_evaluator"
    assert analysis["repair_triggered"] is True
    assert analysis["rejected_candidate_count"] == 1


def test_repair_candidate_builders_are_deterministic() -> None:
    analysis = {"primary_failure": "missing_replay_evaluator"}
    targeted = build_targeted_candidates(analysis)
    base = targeted[0]
    perturbations = build_local_perturbations(base, count=2)
    rollbacks = build_rollback_candidates([{"mutation_id": "mutation-a"}])
    variants = build_confirmatory_variants(base, [*perturbations, *rollbacks])
    rows = build_ablation_matrix(
        "run-1",
        [{"mutation_id": "mutation-a", "selection_oos_delta": -0.1}],
        reason="failed gate",
    )

    assert base.family == "replay_evaluator_enablement"
    assert [item.candidate_id for item in perturbations] == [
        "repair-replay_evaluator_enablement-perturb-1",
        "repair-replay_evaluator_enablement-perturb-2",
    ]
    assert rollbacks[0].candidate_id == "rollback-mutation-a"
    assert variants[0].candidate_id == "repair-replay_evaluator_enablement-confirm"
    assert rows == [
        {
            "run_id": "run-1",
            "stage": "ablation",
            "mutation_id": "mutation-a",
            "decision": "skip",
            "skip_reason": "failed gate",
            "in_sample_delta": 0.0,
            "selection_oos_delta": -0.1,
        }
    ]
    assert variant_payload(
        CandidateEvaluation(variants[0], objective_score=0.2, passed=True)
    )["deterministic_replay_passed"] is True


def test_selection_oos_repair_trigger_requires_fold_profile_for_normal_runs() -> None:
    payload = evaluate_selection_oos_repair_trigger(
        run_id="run-1",
        incumbent={"objective_score": -0.2, "trade_count": 2, "max_drawdown": 0.02},
        candidate=None,
        fold_profile={},
    )

    assert payload["triggered"] is False
    assert payload["status"] == "not_triggered"
    assert "no in-sample fold profile" in payload["reasons"][0]


def test_selection_oos_repair_trigger_uses_measured_degradation_thresholds() -> None:
    payload = evaluate_selection_oos_repair_trigger(
        run_id="run-1",
        incumbent={"objective_score": 1.0, "trade_count": 4, "max_drawdown": 0.04},
        candidate={"objective_score": 0.8, "trade_count": 1, "max_drawdown": 0.12},
        fold_profile={
            "mean_objective_score": 1.0,
            "min_objective_score": 0.95,
            "max_objective_score": 1.1,
            "mean_trade_count": 4,
            "mean_max_drawdown": 0.04,
        },
    )

    assert payload["triggered"] is True
    assert payload["status"] == "triggered"
    assert payload["measured_degradation"]["objective_delta_vs_fold_mean"] < -0.19
