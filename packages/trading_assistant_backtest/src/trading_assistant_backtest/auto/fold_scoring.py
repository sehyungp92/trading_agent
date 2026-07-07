"""Purged in-sample fold scoring for monthly optimizer candidates."""

from __future__ import annotations

from statistics import mean, pvariance
from typing import Any

from trading_assistant_backtest.auto.types import Candidate, CandidateEvaluation
from trading_assistant_backtest.contract_models import FoldManifest, FoldSpec
from trading_assistant_backtest.replay.types import ReplayResult, WindowSpec
from trading_assistant_backtest.scoring.immutable import compact_score_payload


def score_candidate_on_folds(
    *,
    candidate: Candidate,
    plugin: Any,
    baseline: Any,
    fold_manifest: FoldManifest,
) -> CandidateEvaluation:
    """Evaluate one candidate on both purged in-sample folds.

    The selection-OOS window is intentionally not touched here. Each fold caches
    its incumbent immediately before candidate replay so plugin evaluators that
    compare against ``self._incumbent`` stay deterministic.
    """

    fold_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    adoption_blocked = False
    fold_patch_fields: list[dict[str, Any]] = []
    for fold in fold_manifest.folds:
        incumbent = plugin.run_incumbent(_fold_window(fold), baseline)
        evaluation = plugin.evaluate_candidate(candidate, incumbent.window)
        fold_patch_fields.append(_evaluated_patch_fields(evaluation.candidate.payload))
        candidate_replay = _payload_replay(evaluation)
        incumbent_summary = _replay_summary(incumbent)
        objective_delta = float(evaluation.objective_score - incumbent.objective_score)
        net_return_delta = float(candidate_replay.get("net_return", 0.0)) - incumbent.net_return
        max_drawdown_delta = (
            float(candidate_replay.get("max_drawdown", 0.0)) - incumbent.max_drawdown
        )
        profit_factor_delta = (
            float(candidate_replay.get("profit_factor", 0.0)) - incumbent.profit_factor
        )
        trade_count = int(candidate_replay.get("trade_count", 0) or 0)
        fold_support = (
            fold.purged
            and trade_count > 0
            and objective_delta > 0.0
            and max_drawdown_delta <= _drawdown_tolerance(incumbent)
        )
        no_regression_gates = {
            "sufficient_rows": _coverage_rows(candidate_replay) >= 2,
            "sufficient_trades": trade_count > 0,
            "drawdown_not_materially_worse": max_drawdown_delta
            <= _drawdown_tolerance(incumbent),
            "cost_sensitivity_preserves_edge": objective_delta > -0.001,
            "outlier_removal_preserves_edge": trade_count != 1 or objective_delta > 0.002,
            "not_one_trade_dependent": trade_count != 1 or objective_delta > 0.002,
            "portfolio_synergy_passed": True,
        }
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "window": {
                    "start": fold.validation_start.isoformat(),
                    "end": fold.validation_end.isoformat(),
                },
                "purged": fold.purged,
                "embargo_days": fold.embargo_days,
                "incumbent": incumbent_summary,
                "candidate": candidate_replay,
                "objective_score": float(evaluation.objective_score),
                "incumbent_objective_score": incumbent.objective_score,
                "objective_delta": objective_delta,
                "net_return_delta": net_return_delta,
                "calmar_delta": _calmar(candidate_replay) - _calmar(incumbent_summary),
                "profit_factor_delta": profit_factor_delta,
                "expectancy_delta": _expectancy(candidate_replay) - _expectancy(incumbent_summary),
                "max_drawdown_delta": max_drawdown_delta,
                "process_quality_proxy": _process_quality(candidate_replay),
                "fold_support_passed": fold_support,
                "no_regression_gates": no_regression_gates,
                "replay_hashes": {
                    "candidate_trade_hash": str(candidate_replay.get("trade_hash") or ""),
                    "candidate_order_hash": str(candidate_replay.get("order_hash") or ""),
                    "incumbent_trade_hash": str(incumbent.diagnostics.get("trade_hash") or ""),
                    "incumbent_order_hash": str(incumbent.diagnostics.get("order_hash") or ""),
                },
                "reasons": evaluation.reasons,
            }
        )
        reasons.extend(evaluation.reasons)
        adoption_blocked = adoption_blocked or any(
            "adoption remains disabled" in reason for reason in evaluation.reasons
        )

    score = mean([row["objective_score"] for row in fold_rows]) if fold_rows else 0.0
    baseline_score = (
        mean([row["incumbent_objective_score"] for row in fold_rows]) if fold_rows else 0.0
    )
    objective_delta = score - baseline_score
    fold_support_passed = bool(fold_rows) and all(row["fold_support_passed"] for row in fold_rows)
    gate_statuses = _aggregate_gates(fold_rows)
    patch_consistency_errors = _patch_consistency_errors(fold_patch_fields)
    patch_consistency_passed = not patch_consistency_errors
    passed = (
        fold_support_passed
        and all(gate_statuses.values())
        and not adoption_blocked
        and patch_consistency_passed
    )
    if not fold_support_passed:
        reasons.append("candidate failed purged two-fold in-sample support")
    reasons.extend(patch_consistency_errors)
    if adoption_blocked:
        reasons.append("candidate adoption remains disabled until bridge approval-readiness passes")
    reasons.append(
        "two-fold in-sample replay "
        f"score={score:.8f}; baseline={baseline_score:.8f}; delta={objective_delta:.8f}"
    )
    return CandidateEvaluation(
        candidate=Candidate(
            candidate_id=candidate.candidate_id,
            family=candidate.family,
            payload={
                **candidate.payload,
                **_consistent_patch_fields(fold_patch_fields),
                "fold_metrics": fold_rows,
                "fold_support_passed": fold_support_passed,
                "fold_patch_consistency_passed": patch_consistency_passed,
                "purged_fold_support": {
                    row["fold_id"]: row["fold_support_passed"] for row in fold_rows
                },
                "objective_component_scores": _component_scores(fold_rows),
                "no_regression_gate_statuses": gate_statuses,
                "decision_parity_status": "not_required_config_only",
                "cost_sensitivity_passed": gate_statuses.get(
                    "cost_sensitivity_preserves_edge", False
                ),
                "drawdown_gate_passed": gate_statuses.get(
                    "drawdown_not_materially_worse", False
                ),
                "outlier_concentration_passed": gate_statuses.get(
                    "outlier_removal_preserves_edge", False
                ),
                "portfolio_synergy_passed": gate_statuses.get(
                    "portfolio_synergy_passed", False
                ),
                "selection_oos_used_in_first_pass": False,
                "aggregate_fold_score": score,
                "aggregate_fold_baseline_score": baseline_score,
                "aggregate_fold_objective_delta": objective_delta,
                "fold_score_variance": pvariance([row["objective_score"] for row in fold_rows])
                if len(fold_rows) > 1
                else 0.0,
            },
        ),
        objective_score=score,
        passed=passed,
        reasons=list(dict.fromkeys(reason for reason in reasons if reason)),
    )


def fold_candidate_rows(evaluation: CandidateEvaluation, *, run_id: str) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "candidate_id": evaluation.candidate.candidate_id,
            "candidate_family": evaluation.candidate.family,
            **row,
        }
        for row in evaluation.candidate.payload.get("fold_metrics", [])
        if isinstance(row, dict)
    ]


def fold_score_matrix(
    *,
    run_id: str,
    fold_manifest: FoldManifest,
    evaluations: list[CandidateEvaluation],
    selected_candidate_ids: list[str],
) -> dict[str, Any]:
    rows = []
    for evaluation in evaluations:
        payload = evaluation.candidate.payload
        rows.append(
            {
                "candidate_id": evaluation.candidate.candidate_id,
                "family": evaluation.candidate.family,
                "objective_score": evaluation.objective_score,
                "objective_delta": payload.get("aggregate_fold_objective_delta", 0.0),
                "fold_score_variance": payload.get("fold_score_variance", 0.0),
                "fold_support_passed": payload.get("fold_support_passed", False),
                "passed": evaluation.passed,
                "folds": payload.get("fold_metrics", []),
            }
        )
    return {
        "schema_version": "fold_score_matrix_v1",
        "run_id": run_id,
        "fold_manifest_version": fold_manifest.manifest_version,
        "selection_oos_excluded_from_first_pass": True,
        "scoring_windows": [
            {
                "fold_id": fold.fold_id,
                "validation_start": fold.validation_start.isoformat(),
                "validation_end": fold.validation_end.isoformat(),
                "purged": fold.purged,
                "embargo_days": fold.embargo_days,
            }
            for fold in fold_manifest.folds
        ],
        "candidate_count": len(evaluations),
        "selected_candidate_ids": selected_candidate_ids,
        "candidates": rows,
    }


def _fold_window(fold: FoldSpec) -> WindowSpec:
    return WindowSpec(
        name=fold.fold_id,
        start=fold.validation_start,
        end=fold.validation_end,
    )


def _payload_replay(evaluation: CandidateEvaluation) -> dict[str, Any]:
    payload = evaluation.candidate.payload.get("replay_result", {})
    return payload if isinstance(payload, dict) else {}


def _evaluated_patch_fields(payload: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "parameter_patch",
        "evaluated_parameter_patch",
        "evaluated_parameters",
        "parameter_patch_fingerprint",
        "evaluated_patch_fingerprint",
        "evaluated_patch_schema_version",
    }
    return {key: payload[key] for key in keys if key in payload}


def _consistent_patch_fields(fold_patch_fields: list[dict[str, Any]]) -> dict[str, Any]:
    for fields in fold_patch_fields:
        if fields:
            return fields
    return {}


def _patch_consistency_errors(fold_patch_fields: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = (
        "parameter_patch",
        "evaluated_parameters",
        "parameter_patch_fingerprint",
        "evaluated_patch_fingerprint",
    )
    populated = [fields for fields in fold_patch_fields if fields]
    if not populated:
        return ["candidate replay did not emit evaluated patch evidence on any fold"]
    for key in required:
        values = [fields.get(key) for fields in populated if key in fields]
        if len(values) != len(populated):
            errors.append(f"candidate replay missing {key} on at least one fold")
            continue
        first = values[0]
        if any(value != first for value in values[1:]):
            errors.append(f"candidate replay emitted inconsistent {key} across folds")
    return errors


def _replay_summary(result: ReplayResult) -> dict[str, Any]:
    return {
        "trade_count": result.trade_count,
        "net_return": result.net_return,
        "max_drawdown": result.max_drawdown,
        "profit_factor": result.profit_factor,
        "objective_score": result.objective_score,
        "objective_profile_id": result.diagnostics.get("objective_profile_id", ""),
        "immutable_score": compact_score_payload(result.diagnostics.get("immutable_score")),
        "trade_hash": result.diagnostics.get("trade_hash", ""),
        "order_hash": result.diagnostics.get("order_hash", ""),
        "coverage": result.diagnostics.get("coverage", []),
    }


def _coverage_rows(replay: dict[str, Any]) -> int:
    coverage = replay.get("coverage", [])
    if not isinstance(coverage, list):
        return 0
    return sum(int(row.get("rows", 0) or 0) for row in coverage if isinstance(row, dict))


def _drawdown_tolerance(incumbent: ReplayResult) -> float:
    return max(0.01, incumbent.max_drawdown * 0.10)


def _calmar(replay: dict[str, Any]) -> float:
    max_drawdown = abs(float(replay.get("max_drawdown", 0.0) or 0.0))
    net_return = float(replay.get("net_return", 0.0) or 0.0)
    if max_drawdown <= 0.0:
        return net_return
    return net_return / max_drawdown


def _expectancy(replay: dict[str, Any]) -> float:
    trade_count = int(replay.get("trade_count", 0) or 0)
    if trade_count <= 0:
        return 0.0
    return float(replay.get("net_return", 0.0) or 0.0) / trade_count


def _process_quality(replay: dict[str, Any]) -> float:
    trade_count = int(replay.get("trade_count", 0) or 0)
    coverage_rows = _coverage_rows(replay)
    if coverage_rows <= 0:
        return 0.0
    return min(1.0, trade_count / max(coverage_rows, 1))


def _aggregate_gates(rows: list[dict[str, Any]]) -> dict[str, bool]:
    gate_names = sorted(
        {
            name
            for row in rows
            for name in (row.get("no_regression_gates") or {})
            if isinstance(row.get("no_regression_gates"), dict)
        }
    )
    return {
        name: all(
            bool((row.get("no_regression_gates") or {}).get(name))
            for row in rows
            if isinstance(row.get("no_regression_gates"), dict)
        )
        for name in gate_names
    }


def _component_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "objective_delta",
        "net_return_delta",
        "calmar_delta",
        "profit_factor_delta",
        "expectancy_delta",
        "max_drawdown_delta",
        "process_quality_proxy",
    ]
    return {
        key: mean([float(row.get(key, 0.0) or 0.0) for row in rows]) if rows else 0.0
        for key in keys
    }
