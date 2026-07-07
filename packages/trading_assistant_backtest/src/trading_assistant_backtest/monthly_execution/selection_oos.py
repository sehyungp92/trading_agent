"""Selection-OOS evaluation helpers for monthly optimizer flows."""

from __future__ import annotations

from typing import Any

from trading_assistant_backtest.auto.types import CandidateEvaluation
from trading_assistant_backtest.contract_models import MonthlyRunManifest
from trading_assistant_backtest.monthly_execution.artifact_emitter import replay_summary
from trading_assistant_backtest.monthly_execution.replay_context import ReplayEvaluationContext
from trading_assistant_backtest.replay.types import ReplayResult
from trading_assistant_backtest.replay.windows import resolve_selection_oos_window


def selection_oos_evaluation(
    manifest: MonthlyRunManifest,
    *,
    replay_context: ReplayEvaluationContext,
    primary: CandidateEvaluation | None,
) -> tuple[dict[str, Any], CandidateEvaluation | None]:
    if not replay_context.replay_backed or replay_context.plugin is None:
        return (
            {
                "schema_version": "selection_oos_evaluation_v1",
                "run_id": manifest.run_id,
                "status": "blocked",
                "reason": replay_context.reason,
                "selection_oos_used_after_fold_ranking": True,
                "selection_oos_used_in_first_pass": False,
                "incumbent_selection_oos": {},
                "candidate_selection_oos": {},
            },
            None,
        )
    window = resolve_selection_oos_window(manifest)
    incumbent = _selection_oos_incumbent(manifest, replay_context=replay_context)
    candidate_eval = None
    candidate_summary: dict[str, Any] = {}
    if primary is not None:
        candidate_eval = _with_selection_oos_incumbent(
            replay_context.plugin.evaluate_candidate(primary.candidate, window),
            incumbent,
        )
        candidate_summary = {
            **candidate_eval.candidate.payload.get("selection_oos_replay_result", {}),
            "candidate_id": candidate_eval.candidate.candidate_id,
            "objective_score": candidate_eval.objective_score,
            "objective_delta_vs_incumbent": candidate_eval.candidate.payload.get(
                "selection_oos_delta_vs_incumbent",
                0.0,
            ),
            "reasons": candidate_eval.reasons,
        }
    return (
        {
            "schema_version": "selection_oos_evaluation_v1",
            "run_id": manifest.run_id,
            "status": "pass",
            "selection_oos_used_after_fold_ranking": True,
            "selection_oos_used_in_first_pass": False,
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "incumbent_selection_oos": replay_summary(incumbent),
            "candidate_selection_oos": candidate_summary,
            "primary_candidate_id": primary.candidate.candidate_id if primary else "",
        },
        candidate_eval,
    )


def evaluate_candidate_on_selection_oos(
    manifest: MonthlyRunManifest,
    *,
    replay_context: ReplayEvaluationContext,
    candidate,
) -> CandidateEvaluation | None:
    if replay_context.plugin is None or not replay_context.replay_backed:
        return None
    window = resolve_selection_oos_window(manifest)
    incumbent = _selection_oos_incumbent(manifest, replay_context=replay_context)
    return _with_selection_oos_incumbent(
        replay_context.plugin.evaluate_candidate(candidate, window),
        incumbent,
    )


def with_selection_oos_payload(
    fold_evaluation: CandidateEvaluation,
    selection_evaluation: CandidateEvaluation | None,
) -> CandidateEvaluation:
    if selection_evaluation is None:
        return fold_evaluation
    selection_payload = selection_evaluation.candidate.payload
    selection_replay = (
        selection_payload.get("selection_oos_replay_result")
        or _payload_replay_result(selection_evaluation)
    )
    selection_delta = float(
        selection_payload.get("selection_oos_delta_vs_incumbent")
        if selection_payload.get("selection_oos_delta_vs_incumbent") is not None
        else selection_payload.get("selection_oos_delta", 0.0)
    )
    payload = {
        **fold_evaluation.candidate.payload,
        "selection_oos_replay_result": selection_replay,
        "selection_oos_incumbent_result": selection_payload.get(
            "selection_oos_incumbent_result",
            {},
        ),
        "selection_oos_objective_score": selection_evaluation.objective_score,
        "selection_oos_incumbent_objective_score": selection_payload.get(
            "selection_oos_incumbent_objective_score",
            0.0,
        ),
        "selection_oos_delta": selection_delta,
        "selection_oos_delta_vs_incumbent": selection_delta,
        "latest_month_oos_improvement": selection_delta > 0.0,
    }
    return CandidateEvaluation(
        candidate=fold_evaluation.candidate.__class__(
            candidate_id=fold_evaluation.candidate.candidate_id,
            family=fold_evaluation.candidate.family,
            payload=payload,
        ),
        objective_score=fold_evaluation.objective_score,
        passed=fold_evaluation.passed and selection_evaluation.passed and selection_delta > -0.001,
        reasons=list(
            dict.fromkeys(
                [
                    *fold_evaluation.reasons,
                    *selection_evaluation.reasons,
                    f"confirmatory selection-OOS delta={selection_delta:.8f}",
                ]
            )
        ),
    )


def _selection_oos_incumbent(
    manifest: MonthlyRunManifest,
    *,
    replay_context: ReplayEvaluationContext,
) -> ReplayResult:
    if replay_context.selection_oos_incumbent is None:
        assert replay_context.plugin is not None
        window = resolve_selection_oos_window(manifest)
        replay_context.selection_oos_incumbent = replay_context.plugin.run_incumbent(
            window,
            replay_context.baseline,
        )
    return replay_context.selection_oos_incumbent


def _with_selection_oos_incumbent(
    evaluation: CandidateEvaluation,
    incumbent: ReplayResult,
) -> CandidateEvaluation:
    replay = _payload_replay_result(evaluation)
    incumbent_summary = replay_summary(incumbent)
    selection_delta = evaluation.objective_score - incumbent.objective_score
    payload = {
        **evaluation.candidate.payload,
        "selection_oos_replay_result": replay,
        "selection_oos_incumbent_result": incumbent_summary,
        "selection_oos_objective_score": evaluation.objective_score,
        "selection_oos_incumbent_objective_score": incumbent.objective_score,
        "selection_oos_delta": selection_delta,
        "selection_oos_delta_vs_incumbent": selection_delta,
        "latest_month_oos_improvement": selection_delta > 0.0,
        "selection_oos_used_in_first_pass": False,
    }
    return CandidateEvaluation(
        candidate=evaluation.candidate.__class__(
            candidate_id=evaluation.candidate.candidate_id,
            family=evaluation.candidate.family,
            payload=payload,
        ),
        objective_score=evaluation.objective_score,
        passed=evaluation.passed and selection_delta > -0.001,
        reasons=list(
            dict.fromkeys(
                [
                    *evaluation.reasons,
                    f"selection-OOS delta vs incumbent={selection_delta:.8f}",
                ]
            )
        ),
    )


def _payload_replay_result(evaluation: CandidateEvaluation | None) -> dict[str, Any]:
    if evaluation is None:
        return {}
    payload = evaluation.candidate.payload.get("replay_result", {})
    return payload if isinstance(payload, dict) else {}
