"""Repair sequence for monthly optimizer execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_assistant_backtest.artifact_writer import ArtifactWriter
from trading_assistant_backtest.auto.fold_scoring import score_candidate_on_folds
from trading_assistant_backtest.auto.greedy_optimizer import best_passing_candidate
from trading_assistant_backtest.auto.types import CandidateEvaluation
from trading_assistant_backtest.contract_models import MonthlyRunManifest, MonthlyRunMode
from trading_assistant_backtest.monthly_execution.replay_context import ReplayEvaluationContext
from trading_assistant_backtest.monthly_execution.selection_oos import (
    evaluate_candidate_on_selection_oos,
    with_selection_oos_payload,
)
from trading_assistant_backtest.repair.ablation import build_ablation_matrix
from trading_assistant_backtest.repair.failure_analysis import analyze_failure
from trading_assistant_backtest.repair.trigger import evaluate_selection_oos_repair_trigger


@dataclass(frozen=True)
class RepairSequenceResult:
    trigger_payload: dict[str, Any]
    repair_triggered: bool
    failure_analysis: dict[str, Any]
    accepted_mutation_chain: dict[str, Any]
    repair_evaluations: list[CandidateEvaluation]
    repair_selection_evaluations: list[CandidateEvaluation]
    repair_primary_winner: CandidateEvaluation | None


def run_repair_sequence(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    *,
    data_errors: list[str],
    rejected_candidate_rows: list[dict[str, Any]],
    phase_winner_fold_profile: dict[str, Any],
    selection_oos_payload: dict[str, Any],
    replay_context: ReplayEvaluationContext,
    fold_manifest: Any,
) -> RepairSequenceResult:
    artifact_root = writer.root
    trigger_payload = evaluate_selection_oos_repair_trigger(
        run_id=manifest.run_id,
        incumbent=selection_oos_payload.get("incumbent_selection_oos", {}),
        candidate=selection_oos_payload.get("candidate_selection_oos") or None,
        fold_profile=phase_winner_fold_profile,
        force_trigger=manifest.mode == MonthlyRunMode.SMOKE_REPAIR,
    )
    repair_triggered = bool(trigger_payload.get("triggered"))
    writer.write_json("selection_oos_repair_trigger.json", trigger_payload)

    failure_analysis = analyze_failure(
        manifest.run_id,
        data_errors=data_errors,
        rejected_candidates=rejected_candidate_rows,
        repair_triggered=repair_triggered,
    )
    failure_analysis["selection_oos_trigger_path"] = str(
        artifact_root / "selection_oos_repair_trigger.json"
    )
    failure_analysis["evidence_paths"] = [
        str(artifact_root / "fold_score_matrix.json"),
        str(artifact_root / "selection_oos_evaluation.json"),
        str(artifact_root / "selection_oos_repair_trigger.json"),
    ]
    writer.write_json("repair_failure_attribution.json", failure_analysis)

    accepted_mutation_chain = _load_accepted_mutation_chain(manifest, artifact_root)
    writer.write_json("accepted_mutation_chain.json", accepted_mutation_chain)

    repair_evaluations = _repair_evaluations(
        manifest,
        data_errors=data_errors,
        repair_triggered=repair_triggered,
        replay_context=replay_context,
        failure_analysis=failure_analysis,
        accepted_mutation_chain=accepted_mutation_chain,
        fold_manifest=fold_manifest,
    )
    writer.write_jsonl(
        "repair_candidate_results.jsonl",
        [
            _repair_candidate_result_row(evaluation, manifest, artifact_root)
            for evaluation in repair_evaluations
        ],
    )
    writer.write_json(
        "repair_checkpoint.json",
        _repair_checkpoint_payload(
            manifest,
            repair_triggered=repair_triggered,
            accepted_mutation_chain=accepted_mutation_chain,
            repair_evaluations=repair_evaluations,
        ),
    )

    repair_selection_evaluations = [
        enriched
        for evaluation in repair_evaluations
        for selection_eval in [
            evaluate_candidate_on_selection_oos(
                manifest,
                replay_context=replay_context,
                candidate=evaluation.candidate,
            )
        ]
        for enriched in [with_selection_oos_payload(evaluation, selection_eval)]
    ]
    return RepairSequenceResult(
        trigger_payload=trigger_payload,
        repair_triggered=repair_triggered,
        failure_analysis=failure_analysis,
        accepted_mutation_chain=accepted_mutation_chain,
        repair_evaluations=repair_evaluations,
        repair_selection_evaluations=repair_selection_evaluations,
        repair_primary_winner=best_passing_candidate(repair_selection_evaluations),
    )


def write_repair_ablation_matrix(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    *,
    accepted_mutation_chain: dict[str, Any],
    reason: str,
) -> None:
    writer.write_jsonl(
        "repair_ablation_matrix.jsonl",
        build_ablation_matrix(
            manifest.run_id,
            accepted_mutation_chain.get("accepted_mutations", [])
            if isinstance(accepted_mutation_chain, dict)
            else [],
            reason=reason,
        ),
    )


def _repair_evaluations(
    manifest: MonthlyRunManifest,
    *,
    data_errors: list[str],
    repair_triggered: bool,
    replay_context: ReplayEvaluationContext,
    failure_analysis: dict[str, Any],
    accepted_mutation_chain: dict[str, Any],
    fold_manifest: Any,
) -> list[CandidateEvaluation]:
    if not repair_triggered or data_errors or replay_context.plugin is None:
        return []
    repair_candidates = replay_context.plugin.build_repair_candidates(
        failure_analysis,
        accepted_mutation_chain,
    )
    return [
        score_candidate_on_folds(
            candidate=candidate,
            plugin=replay_context.plugin,
            baseline=replay_context.baseline,
            fold_manifest=fold_manifest,
        )
        for candidate in repair_candidates
    ]


def _repair_checkpoint_payload(
    manifest: MonthlyRunManifest,
    *,
    repair_triggered: bool,
    accepted_mutation_chain: dict[str, Any],
    repair_evaluations: list[CandidateEvaluation],
) -> dict[str, Any]:
    return {
        "schema_version": "repair_checkpoint_v1",
        "run_id": manifest.run_id,
        "repair_triggered": repair_triggered,
        "candidate_ids": [
            evaluation.candidate.candidate_id
            for evaluation in repair_evaluations
        ],
        "accepted_mutation_count": len(
            accepted_mutation_chain.get("accepted_mutations", [])
            if isinstance(accepted_mutation_chain, dict)
            else []
        ),
        "deterministic_resume_key": _stable_json_hash(
            {
                "run_id": manifest.run_id,
                "repair_candidates": [
                    evaluation.candidate.candidate_id for evaluation in repair_evaluations
                ],
                "accepted_mutation_chain": accepted_mutation_chain,
            }
        ),
    }


def _repair_candidate_result_row(
    evaluation: CandidateEvaluation,
    manifest: MonthlyRunManifest,
    artifact_root: Path,
) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "candidate_id": evaluation.candidate.candidate_id,
        "family": evaluation.candidate.family,
        "passed": evaluation.passed,
        "objective_score": evaluation.objective_score,
        "fold_support_passed": bool(evaluation.candidate.payload.get("fold_support_passed")),
        "reason": _evaluation_reason(evaluation),
        "evidence_paths": [
            str(artifact_root / "fold_score_matrix.json"),
            str(artifact_root / "repair_failure_attribution.json"),
            str(artifact_root / "accepted_mutation_chain.json"),
        ],
        "raw_payload": evaluation.candidate.payload,
    }


def _load_accepted_mutation_chain(
    manifest: MonthlyRunManifest,
    artifact_root: Path,
) -> dict[str, Any]:
    raw_items: list[dict[str, Any]] = []
    guidance = (
        manifest.monthly_search_guidance
        if isinstance(manifest.monthly_search_guidance, dict)
        else {}
    )
    for item in guidance.get("accepted_mutations", []) or []:
        if isinstance(item, dict):
            raw_items.append(item)
    for path_text in _accepted_mutation_source_paths(manifest, artifact_root):
        path = Path(path_text)
        if not path.exists():
            continue
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                payload = item.get("payload") if isinstance(item, dict) else None
                if isinstance(payload, dict):
                    item = payload
                if isinstance(item, dict):
                    raw_items.append(item)
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if isinstance(payload, dict):
                raw_items.extend(payload.get("accepted_mutations", []) or [])
                raw_items.extend(payload.get("records", []) or [])
            elif isinstance(payload, list):
                raw_items.extend(item for item in payload if isinstance(item, dict))
    mutations = _dedupe_mutations(
        [
            mutation
            for item in raw_items
            for mutation in [_normalize_accepted_mutation(item)]
            if mutation
        ]
    )
    return {
        "schema_version": "accepted_mutation_chain_v1",
        "run_id": manifest.run_id,
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "accepted_mutations": mutations,
        "source_paths": _accepted_mutation_source_paths(manifest, artifact_root),
    }


def _accepted_mutation_source_paths(
    manifest: MonthlyRunManifest,
    artifact_root: Path,
) -> list[str]:
    paths: list[str] = []
    for attr in (
        "strategy_change_ledger_path",
        "accepted_mutation_chain_path",
        "prior_rounds_manifest_path",
        "rounds_manifest_path",
    ):
        value = str(getattr(manifest, attr, "") or "").strip()
        if value:
            paths.append(value)
    guidance = (
        manifest.monthly_search_guidance
        if isinstance(manifest.monthly_search_guidance, dict)
        else {}
    )
    for value in guidance.get("accepted_mutation_source_paths", []) or []:
        if str(value):
            paths.append(str(value))
    candidate = artifact_root.parent / "strategy_change_ledger.jsonl"
    if candidate.exists():
        paths.append(str(candidate))
    return _dedupe_strings(paths)


def _normalize_accepted_mutation(item: dict[str, Any]) -> dict[str, Any] | None:
    record_type = str(item.get("record_type") or item.get("type") or "").lower()
    if record_type and record_type not in {
        "accepted_change",
        "implemented_change",
        "deployed_change",
        "record",
        "round",
        "adopted",
        "proposed_change",
    }:
        return None
    mutation_id = str(
        item.get("mutation_id")
        or item.get("candidate_id")
        or item.get("record_id")
        or item.get("round_id")
        or ""
    ).strip()
    if not mutation_id:
        return None
    mutation_diff = item.get("mutation_diff") if isinstance(item.get("mutation_diff"), dict) else {}
    return {
        "mutation_id": mutation_id,
        "first_accepted_round": str(
            item.get("first_accepted_round")
            or item.get("round_id")
            or item.get("run_month")
            or ""
        ),
        "strategy_scope": str(item.get("strategy_id") or item.get("strategy_scope") or ""),
        "config_scope": str(item.get("config_version") or item.get("new_config_version") or ""),
        "patch_path": str(item.get("patch_path") or item.get("config_patch_path") or ""),
        "parameter_diff": item.get("parameter_diff") or mutation_diff,
        "structural_diff": item.get("structural_diff") or item.get("file_changes") or [],
        "original_evidence_paths": item.get("evidence_paths", []) or [],
        "outcome_status": str(item.get("outcome_status") or item.get("monthly_status") or ""),
    }


def _dedupe_mutations(mutations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mutation in mutations:
        mutation_id = str(mutation.get("mutation_id") or "")
        if mutation_id in seen:
            continue
        seen.add(mutation_id)
        result.append(mutation)
    return result


def _stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _evaluation_reason(evaluation: CandidateEvaluation) -> str:
    return "; ".join(reason for reason in evaluation.reasons if reason)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
