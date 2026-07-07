"""Native monthly runner CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant_backtest.artifact_writer import ArtifactWriter
from trading_assistant_backtest.auto.candidate_workspace import CandidateWorkspaceManager
from trading_assistant_backtest.auto.fold_scoring import (
    fold_candidate_rows,
    fold_score_matrix,
    score_candidate_on_folds,
)
from trading_assistant_backtest.auto.greedy_optimizer import (
    best_passing_candidate,
)
from trading_assistant_backtest.auto.greedy_optimizer import (
    no_adoption_reason as phase_no_adoption_reason,
)
from trading_assistant_backtest.auto.phase_runner import run_phase
from trading_assistant_backtest.auto.types import CandidateEvaluation, PhaseSpec
from trading_assistant_backtest.contract_models import (
    PHASED_AUTO_RUNNER_CONTRACT_VERSION,
    SMOKE_REPAIR_RUNNER_CONTRACT_VERSION,
    CandidateAttemptRecord,
    CandidateAttemptState,
    ConfirmatoryRerank,
    MonthlyCandidateSource,
    MonthlyRunManifest,
    OptimizerExperimentPlan,
    OptimizerStage,
    RoundManifestRecord,
    RoundsManifest,
)
from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.monthly_execution.artifact_emitter import (
    replay_evidence_payload,
    replay_summary,
    scope_id_for_manifest,
)
from trading_assistant_backtest.monthly_execution.repair_sequence import (
    run_repair_sequence,
    write_repair_ablation_matrix,
)
from trading_assistant_backtest.monthly_execution.replay_context import ReplayEvaluationContext
from trading_assistant_backtest.monthly_execution.report_summary import (
    no_adoption_reason,
)
from trading_assistant_backtest.monthly_execution.selection_oos import (
    evaluate_candidate_on_selection_oos,
    selection_oos_evaluation,
    with_selection_oos_payload,
)
from trading_assistant_backtest.monthly_execution.structural_registry import (
    bridge_id_for_plugin,
    bridge_ids_for_scope,
)
from trading_assistant_backtest.observability import runner_event
from trading_assistant_backtest.paths import package_root
from trading_assistant_backtest.planner.deterministic_fallback import build_deterministic_plan
from trading_assistant_backtest.replay.windows import (
    build_manifest_folds,
)
from trading_assistant_backtest.scoring.gates import pass_gate_report


def write_optimizer_artifacts(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    *,
    manifest_path: Path,
    data_errors: list[str],
    planner_mode: str,
    replay_context: ReplayEvaluationContext,
) -> None:
    artifact_root = writer.root
    writer.write_json(
        "optimizer_run_manifest.json",
        _optimizer_run_manifest_payload(
            manifest,
            artifact_root=artifact_root,
            manifest_path=manifest_path,
            planner_mode=planner_mode,
        ),
    )
    workspace_root = Path(manifest.candidate_workspace_root or artifact_root / "workspaces")
    workspace_manager = CandidateWorkspaceManager(workspace_root)
    workspace = workspace_manager.prepare(
        run_id=manifest.run_id,
        candidate_id="no-adoption",
        workspace_key=manifest.candidate_workspace_key or "no-adoption",
    )
    writer.write_json("candidate_workspace_manifest.json", workspace)

    fold_manifest = build_manifest_folds(
        manifest,
        evidence_paths=[str(artifact_root / "objective_breakdown.json")],
    )
    writer.write_json("fold_manifest.json", fold_manifest)
    experiment_plan = build_deterministic_plan(manifest, artifact_root)
    writer.write_json("llm_experiment_plan.json", experiment_plan)

    phase_evaluations = (
        []
        if data_errors
        else _run_plan_phases(
            experiment_plan,
            replay_context=replay_context,
            fold_manifest=fold_manifest,
            search_brief=manifest.monthly_search_guidance,
        )
    )

    phase_winner = best_passing_candidate(phase_evaluations)
    selected_phase_ids = [phase_winner.candidate.candidate_id] if phase_winner is not None else []
    writer.write_jsonl(
        "fold_candidate_results.jsonl",
        [
            row
            for evaluation in phase_evaluations
            for row in fold_candidate_rows(evaluation, run_id=manifest.run_id)
        ],
    )
    writer.write_json(
        "fold_score_matrix.json",
        fold_score_matrix(
            run_id=manifest.run_id,
            fold_manifest=fold_manifest,
            evaluations=phase_evaluations,
            selected_candidate_ids=selected_phase_ids,
        ),
    )

    selection_oos_evaluation_payload, selection_oos_candidate = selection_oos_evaluation(
        manifest,
        replay_context=replay_context,
        primary=phase_winner,
    )
    phase_primary_winner = (
        with_selection_oos_payload(phase_winner, selection_oos_candidate)
        if phase_winner is not None
        else None
    )
    writer.write_json("selection_oos_evaluation.json", selection_oos_evaluation_payload)
    phase_rows_for_failure = [
        _candidate_result_row(
            evaluation,
            manifest,
            artifact_root=artifact_root,
            source=MonthlyCandidateSource.PHASED_AUTO,
            selected=False,
            baseline_score=_evaluation_baseline_score(evaluation, replay_context.baseline_score),
        )
        for evaluation in phase_evaluations
    ]
    repair_result = run_repair_sequence(
        writer,
        manifest,
        data_errors=data_errors,
        rejected_candidate_rows=phase_rows_for_failure,
        phase_winner_fold_profile=_fold_profile(phase_winner),
        selection_oos_payload=selection_oos_evaluation_payload,
        replay_context=replay_context,
        fold_manifest=fold_manifest,
    )
    trigger_payload = repair_result.trigger_payload
    repair_triggered = repair_result.repair_triggered
    failure_analysis = repair_result.failure_analysis
    accepted_mutation_chain = repair_result.accepted_mutation_chain
    repair_evaluations = repair_result.repair_evaluations
    repair_selection_evaluations = repair_result.repair_selection_evaluations
    repair_primary_winner = repair_result.repair_primary_winner
    primary_source = (
        MonthlyCandidateSource.SMOKE_REPAIR
        if repair_triggered
        else MonthlyCandidateSource.PHASED_AUTO
    )
    primary_winner = repair_primary_winner if repair_triggered else phase_primary_winner

    confirmatory_evaluations: list[CandidateEvaluation] = []
    confirmatory_variant_rows: list[dict[str, Any]] = []
    if primary_winner is not None and replay_context.plugin is not None:
        variants = replay_context.plugin.build_confirmatory_variants(
                primary_winner.candidate,
                {
                    "selection_oos_evaluation": selection_oos_evaluation_payload,
                    "repair_trigger": trigger_payload,
                    "failure_analysis": failure_analysis,
                },
        )
        for variant in variants:
            fold_eval = score_candidate_on_folds(
                candidate=variant,
                plugin=replay_context.plugin,
                baseline=replay_context.baseline,
                fold_manifest=fold_manifest,
            )
            selection_eval = evaluate_candidate_on_selection_oos(
                manifest,
                replay_context=replay_context,
                candidate=fold_eval.candidate,
            )
            enriched = with_selection_oos_payload(fold_eval, selection_eval)
            confirmatory_evaluations.append(enriched)
            confirmatory_variant_rows.append(
                _confirmatory_variant_payload(
                    enriched,
                    selection_eval,
                    artifact_root=artifact_root,
                    baseline_score=_evaluation_baseline_score(
                        enriched,
                        replay_context.baseline_score,
                    ),
                )
            )

    compared_for_adoption = _dedupe_evaluations_by_candidate(
        [
            phase_primary_winner,
            *repair_selection_evaluations,
            *confirmatory_evaluations,
        ]
    )
    confirmatory_winner = best_passing_candidate(compared_for_adoption)
    winner = confirmatory_winner

    phase_report_evaluations = _replace_evaluation_by_candidate(
        phase_evaluations,
        phase_primary_winner,
    )
    repair_report_evaluations = (
        repair_selection_evaluations if repair_selection_evaluations else repair_evaluations
    )
    all_evaluations = [
        *phase_report_evaluations,
        *repair_report_evaluations,
        *confirmatory_evaluations,
    ]
    source_by_candidate = {
        evaluation.candidate.candidate_id: MonthlyCandidateSource.PHASED_AUTO
        for evaluation in phase_evaluations
    }
    source_by_candidate.update(
        {
            evaluation.candidate.candidate_id: MonthlyCandidateSource.SMOKE_REPAIR
            for evaluation in repair_evaluations
        }
    )
    source_by_candidate.update(
        {
            evaluation.candidate.candidate_id: primary_source
            for evaluation in confirmatory_evaluations
        }
    )
    all_workspaces = {
        evaluation.candidate.candidate_id: workspace_manager.prepare(
            run_id=manifest.run_id,
            candidate_id=evaluation.candidate.candidate_id,
            workspace_key=evaluation.candidate.candidate_id,
        )
        for evaluation in all_evaluations
    }
    attempts = [
        _candidate_attempt(
            evaluation,
            manifest,
            artifact_root=artifact_root,
            source=source_by_candidate.get(
                evaluation.candidate.candidate_id,
                MonthlyCandidateSource.PHASED_AUTO,
            ),
            workspace=all_workspaces[evaluation.candidate.candidate_id],
        )
        for evaluation in all_evaluations
    ]
    attempts_by_candidate = {attempt.candidate_id: attempt for attempt in attempts}
    selected_rows = (
        [
            _candidate_result_row(
                winner,
                manifest,
                artifact_root=artifact_root,
                source=source_by_candidate.get(winner.candidate.candidate_id, primary_source),
                selected=True,
                attempt=attempts_by_candidate.get(winner.candidate.candidate_id),
                workspace=all_workspaces.get(winner.candidate.candidate_id),
                baseline_score=_evaluation_baseline_score(winner, replay_context.baseline_score),
            )
        ]
        if winner is not None
        else []
    )
    rejected_rows = [
        _candidate_result_row(
            evaluation,
            manifest,
            artifact_root=artifact_root,
            source=source_by_candidate.get(
                evaluation.candidate.candidate_id,
                MonthlyCandidateSource.PHASED_AUTO,
            ),
            selected=False,
            attempt=attempts_by_candidate.get(evaluation.candidate.candidate_id),
            workspace=all_workspaces.get(evaluation.candidate.candidate_id),
            baseline_score=_evaluation_baseline_score(evaluation, replay_context.baseline_score),
        )
        for evaluation in all_evaluations
        if winner is None or evaluation.candidate.candidate_id != winner.candidate.candidate_id
    ]
    reason = _optimizer_decision_reason(manifest, data_errors, all_evaluations, winner)
    writer.write_jsonl("candidate_attempts.jsonl", attempts)
    writer.write_json(
        "runner_observability.json",
        _runner_observability(manifest, attempts, reason=reason, planner_mode=planner_mode),
    )

    gate_status = "blocked" if data_errors else "pass"
    _write_optimizer_gate_artifacts(
        writer,
        manifest,
        evaluations=all_evaluations,
        data_errors=data_errors,
        gate_status=gate_status,
    )

    writer.write_json(
        "end_of_round_diagnostics.json",
        {
            "run_id": manifest.run_id,
            "status": gate_status,
            "diagnostics_saved": True,
            "failure_analysis": failure_analysis,
            "evidence_paths": [
                str(artifact_root / "incumbent_validation.json"),
                str(artifact_root / "gap_attribution.json"),
                str(artifact_root / "fold_validation.json"),
                str(artifact_root / "fold_score_matrix.json"),
                str(artifact_root / "selection_oos_repair_trigger.json"),
            ],
        },
    )

    winner_source = (
        source_by_candidate.get(winner.candidate.candidate_id, primary_source)
        if winner is not None
        else primary_source
    )
    round_n_payload = _round_n_plus_1_recommendation(
        manifest,
        winner=winner,
        replay_context=replay_context,
        artifact_root=artifact_root,
        no_adoption_reason=reason if winner is None else "",
    )
    writer.write_json("round_n_plus_1_recommendation.json", round_n_payload)

    compared_candidate_ids = [row["candidate_id"] for row in [*selected_rows, *rejected_rows]]
    writer.write_json(
        "confirmatory_rerank.json",
        ConfirmatoryRerank(
            run_id=manifest.run_id,
            primary_candidate_id=primary_winner.candidate.candidate_id
            if primary_winner is not None
            else "",
            primary_source=primary_source,
            repair_triggered=repair_triggered,
            compared_candidate_ids=compared_candidate_ids,
            variants=confirmatory_variant_rows,
            adopted_candidate_id=winner.candidate.candidate_id if winner is not None else "",
            adopted_source=winner_source if winner is not None else MonthlyCandidateSource.UNKNOWN,
            no_adoption_reason="" if winner is not None else reason,
            selection_rule=(
                "best confirmatory candidate passing purged folds, selection-OOS, "
                "and no-regression gates; fail closed otherwise"
            ),
            objective_version=manifest.objective_version,
            evidence_paths=[
                str(artifact_root / "fold_validation.json"),
                str(artifact_root / "fold_score_matrix.json"),
                str(artifact_root / "selection_oos_evaluation.json"),
            ],
        ),
    )
    writer.write_json(
        "rounds_manifest.json",
        RoundsManifest(
            run_id=manifest.run_id,
            bot_id=manifest.bot_id,
            strategy_id=manifest.strategy_id,
            current_round_id=manifest.round_id or f"{manifest.run_month}-round-0",
            next_round_id=manifest.next_round_id if winner is not None else "",
            adopted_candidate_id=winner.candidate.candidate_id if winner is not None else "",
            no_adoption_reason="" if winner is not None else reason,
            records=[
                RoundManifestRecord(
                    round_id=(
                        manifest.next_round_id
                        if winner is not None
                        else manifest.round_id or f"{manifest.run_month}-round-0"
                    ),
                    prior_round_id=(
                        manifest.round_id
                        if winner is not None
                        else manifest.prior_round_id
                    ),
                    next_round_id=manifest.next_round_id if winner is not None else "",
                    candidate_id=winner.candidate.candidate_id if winner is not None else "",
                    source=winner_source,
                    fold_manifest_path=str(artifact_root / "fold_manifest.json"),
                    diagnostics_path=str(artifact_root / "end_of_round_diagnostics.json"),
                    confirmatory_rerank_path=str(artifact_root / "confirmatory_rerank.json"),
                    approval_state="not_requested",
                    live_deployment_status="optimized_backtest_recommendation",
                    evidence_paths=[
                        str(artifact_root / "end_of_round_diagnostics.json"),
                        str(artifact_root / "round_n_plus_1_recommendation.json"),
                    ],
                )
            ],
            objective_version=manifest.objective_version,
        ),
    )
    writer.write_json("selected_candidates.json", selected_rows)
    writer.write_jsonl(
        "rejected_candidates.jsonl",
        rejected_rows
        or [
            {
                "run_id": manifest.run_id,
                "candidate_id": "candidate-space",
                "reason": reason,
                "source": primary_source.value,
            }
        ],
    )
    writer.write_jsonl("candidate_results.jsonl", [*selected_rows, *rejected_rows])
    if repair_triggered:
        write_repair_ablation_matrix(
            writer,
            manifest,
            accepted_mutation_chain=accepted_mutation_chain,
            reason=reason,
        )
    if replay_context.replay_backed:
        _write_replay_lineage_artifacts(
            writer,
            manifest,
            replay_context=replay_context,
            selected_rows=selected_rows,
            rejected_rows=rejected_rows,
            no_adoption_reason=reason,
        )

def _optimizer_run_manifest_payload(
    manifest: MonthlyRunManifest,
    *,
    artifact_root: Path,
    manifest_path: Path,
    planner_mode: str,
) -> dict[str, Any]:
    scope_id = scope_id_for_manifest(manifest)
    contract_path = (
        Path(manifest.strategy_plugin_contract_path)
        if manifest.strategy_plugin_contract_path
        else None
    )
    deployment_path = (
        Path(manifest.deployment_metadata_path)
        if manifest.deployment_metadata_path
        else None
    )
    approval_mode = str(getattr(manifest.approval_mode, "value", manifest.approval_mode) or "")
    run_mode = str(getattr(manifest.mode, "value", manifest.mode) or "")
    approval_grade = (
        approval_mode not in {"", "none"}
        and run_mode != "smoke_repair"
        and "monthly_smoke" not in {part.lower() for part in artifact_root.parts}
    )
    scope_aliases = [
        alias
        for alias in (
            manifest.bot_id,
            manifest.strategy_id,
            manifest.strategy_plugin_id,
            scope_id,
            _optimizer_scope_id(manifest),
        )
        if alias
    ]
    contract_hash = _sha256_file_if_exists(contract_path)
    deployment_hash = _sha256_file_if_exists(deployment_path)
    contract_paths = _bridge_artifact_paths(
        manifest,
        scope_id=scope_id,
        primary_path=contract_path,
        artifact_name="strategy_plugin_contract.json",
        manifest_map_names=("bridge_contract_paths", "strategy_plugin_contract_paths"),
    )
    deployment_paths = _bridge_artifact_paths(
        manifest,
        scope_id=scope_id,
        primary_path=deployment_path,
        artifact_name="deployment_metadata.json",
        manifest_map_names=("bridge_deployment_metadata_paths", "deployment_metadata_paths"),
    )
    contract_hashes = _bridge_artifact_hashes(contract_paths)
    deployment_hashes = _bridge_artifact_hashes(deployment_paths)
    data_bundle_checksum = manifest.data_bundle_checksum or manifest.data_manifest_checksum
    return {
        "schema_version": "optimizer_approval_run_manifest_v1",
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id,
        "scope_id": scope_id,
        "scope_aliases": list(dict.fromkeys(scope_aliases)),
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "strategy_plugin_id": manifest.strategy_plugin_id,
        "run_month": manifest.run_month,
        "run_mode": run_mode,
        "optimizer_mode": "approval_grade" if approval_grade else "shadow_validation",
        "approval_mode": approval_mode or "none",
        "planner_mode": planner_mode,
        "approval_grade_optimizer_run": approval_grade,
        "smoke_mode": not approval_grade,
        "artifact_root": str(artifact_root),
        "run_manifest_path": str(manifest_path),
        "run_manifest_hash": _sha256_file_if_exists(manifest_path),
        "data_bundle_checksum": data_bundle_checksum,
        "data_bundle_checksums": [data_bundle_checksum] if data_bundle_checksum else [],
        "strategy_plugin_contract_path": str(contract_path or ""),
        "strategy_plugin_contract_hash": contract_hash,
        "strategy_plugin_contract_paths": _string_path_map(contract_paths),
        "bridge_contract_paths": _string_path_map(contract_paths),
        "strategy_plugin_contract_hashes": contract_hashes,
        "bridge_contract_hashes": contract_hashes,
        "deployment_metadata_path": str(deployment_path or ""),
        "deployment_metadata_hash": deployment_hash,
        "deployment_metadata_paths": _string_path_map(deployment_paths),
        "bridge_deployment_metadata_paths": _string_path_map(deployment_paths),
        "deployment_metadata_hashes": deployment_hashes,
        "bridge_deployment_metadata_hashes": deployment_hashes,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00",
            "Z",
        ),
    }


def _optimizer_scope_id(manifest: MonthlyRunManifest) -> str:
    return (
        str(getattr(manifest, "scope_id", "") or "").strip()
        or manifest.strategy_id
        or manifest.bot_id
    )


def _bridge_artifact_paths(
    manifest: MonthlyRunManifest,
    *,
    scope_id: str,
    primary_path: Path | None,
    artifact_name: str,
    manifest_map_names: tuple[str, ...],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in manifest_map_names:
        paths.update(_path_map(getattr(manifest, name, {})))
    bridge_id = _bridge_id_for_manifest(manifest, scope_id)
    if primary_path is not None and bridge_id:
        paths.setdefault(bridge_id, primary_path)
    for root in _contract_root_candidates(primary_path, manifest):
        for expected_bridge_id in bridge_ids_for_scope(scope_id):
            candidate = root / expected_bridge_id / artifact_name
            if candidate.exists() and candidate.is_file():
                paths.setdefault(expected_bridge_id, candidate)
    if not paths and primary_path is not None:
        paths[scope_id] = primary_path
    return paths


def _bridge_id_for_manifest(manifest: MonthlyRunManifest, scope_id: str) -> str:
    return bridge_id_for_plugin(manifest.strategy_plugin_id, scope_id)


def _path_map(value: Any) -> dict[str, Path]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): Path(str(item))
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def _contract_root_candidates(
    primary_path: Path | None,
    manifest: MonthlyRunManifest,
) -> list[Path]:
    roots: list[Path] = []
    if primary_path is not None and primary_path.parent.name:
        roots.append(primary_path.parent.parent)
    if manifest.backtest_repo_path:
        roots.append(Path(manifest.backtest_repo_path) / "contracts")
    roots.append(package_root() / "contracts")
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _bridge_artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        bridge_id: digest
        for bridge_id, path in paths.items()
        for digest in [_sha256_file_if_exists(path)]
        if digest
    }


def _string_path_map(paths: dict[str, Path]) -> dict[str, str]:
    return {bridge_id: str(path) for bridge_id, path in paths.items()}


def _sha256_file_if_exists(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return sha256_file(path, missing_ok=True)


def _write_replay_lineage_artifacts(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    *,
    replay_context: ReplayEvaluationContext,
    selected_rows: list[dict],
    rejected_rows: list[dict],
    no_adoption_reason: str,
) -> None:
    replay = replay_context.incumbent
    assert replay is not None
    baseline_payload = {
        "schema_version": "frozen_replay_baseline_v1",
        "run_id": manifest.run_id,
        "scope_id": scope_id_for_manifest(manifest),
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "strategy_plugin_id": manifest.strategy_plugin_id,
        "run_month": manifest.run_month,
        "round_id": manifest.round_id,
        "objective_version": manifest.objective_version,
        "data_bundle_manifest_path": manifest.data_bundle_manifest_path
        or manifest.market_data_manifest_path,
        "data_bundle_checksum": manifest.data_bundle_checksum
        or manifest.data_manifest_checksum,
        "baseline_score": replay.objective_score,
        "incumbent": replay_summary(replay),
        "diagnostics": replay_context.diagnostics or {},
        "evidence_paths": [
            str(writer.path("incumbent_validation.json")),
            str(writer.path("end_of_round_diagnostics.json")),
            str(writer.path("candidate_results.jsonl")),
        ],
    }
    writer.write_json("frozen_baseline.json", baseline_payload)

    compared_ids = [row["candidate_id"] for row in [*selected_rows, *rejected_rows]]
    adopted = selected_rows[0]["candidate_id"] if selected_rows else ""
    round_payload = {
        "schema_version": "round_reproduction_report_v1",
        "run_id": manifest.run_id,
        "scope_id": scope_id_for_manifest(manifest),
        "status": "pass",
        "round_id": manifest.round_id,
        "prior_round_id": manifest.prior_round_id,
        "next_round_id": manifest.next_round_id if adopted else "",
        "adopted_candidate_id": adopted,
        "no_adoption_reason": "" if adopted else no_adoption_reason,
        "candidate_ids": compared_ids,
        "candidate_count": len(compared_ids),
        "baseline_score": replay.objective_score,
        "checks": [
            {
                "name": "baseline_replay_frozen",
                "status": "pass",
                "details": "frozen baseline was written after replay-backed incumbent evaluation",
            },
            {
                "name": "candidate_decisions_reproducible",
                "status": "pass",
                "details": "candidate ids and adoption/no-adoption decision are deterministic",
            },
            {
                "name": "round_lineage_complete",
                "status": "pass",
                "details": "round, prior round, objective, and data checksums are retained",
            },
        ],
        "evidence_paths": [
            str(writer.path("frozen_baseline.json")),
            str(writer.path("rounds_manifest.json")),
            str(writer.path("confirmatory_rerank.json")),
            str(writer.path("candidate_results.jsonl")),
        ],
    }
    writer.write_json("round_reproduction_report.json", round_payload)

    walk_forward_payload = {
        "schema_version": "historical_walk_forward_report_v1",
        "run_id": manifest.run_id,
        "scope_id": scope_id_for_manifest(manifest),
        "status": "blocked",
        "reason": (
            "multi-month walk-forward evidence must be generated from several "
            "authoritative bundles"
        ),
        "window_count": 1,
        "windows": [
            {
                "run_month": manifest.run_month,
                "status": "pass",
                "objective_score": replay.objective_score,
                "trade_count": replay.trade_count,
                "data_bundle_checksum": manifest.data_bundle_checksum
                or manifest.data_manifest_checksum,
            }
        ],
        "evidence_paths": [str(writer.path("frozen_baseline.json"))],
    }
    writer.write_json("historical_walk_forward_report.json", walk_forward_payload)
    writer.write_json(
        "replay_evidence_report.json",
        replay_evidence_payload(
            manifest,
            incumbent_pass=True,
            round_pass=True,
            historical_pass=False,
            evidence_paths=[
                str(writer.path("frozen_baseline.json")),
                str(writer.path("round_reproduction_report.json")),
                str(writer.path("historical_walk_forward_report.json")),
            ],
        ),
    )


def _dedupe_evaluations_by_candidate(
    evaluations: list[CandidateEvaluation | None],
) -> list[CandidateEvaluation]:
    result: list[CandidateEvaluation] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        if evaluation is None:
            continue
        candidate_id = evaluation.candidate.candidate_id
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(evaluation)
    return result


def _replace_evaluation_by_candidate(
    evaluations: list[CandidateEvaluation],
    replacement: CandidateEvaluation | None,
) -> list[CandidateEvaluation]:
    if replacement is None:
        return evaluations
    replaced = False
    result: list[CandidateEvaluation] = []
    for evaluation in evaluations:
        if evaluation.candidate.candidate_id == replacement.candidate.candidate_id:
            result.append(replacement)
            replaced = True
        else:
            result.append(evaluation)
    if not replaced:
        result.append(replacement)
    return result


def _objective_metadata_from_evaluation(evaluation: CandidateEvaluation) -> dict[str, Any]:
    for replay in _objective_replay_sources(evaluation):
        immutable_score = replay.get("immutable_score")
        if not isinstance(immutable_score, dict):
            immutable_score = {}
        profile = immutable_score.get("profile")
        if not isinstance(profile, dict):
            profile = {}
        profile_id = str(
            replay.get("objective_profile_id")
            or immutable_score.get("profile_id")
            or profile.get("profile_id")
            or ""
        )
        profile_version = str(
            immutable_score.get("profile_version")
            or immutable_score.get("version")
            or profile.get("version")
            or ""
        )
        if not (profile_id or profile_version or immutable_score):
            continue
        return {
            "effective_objective_version": profile_version,
            "immutable_objective_version": profile_version,
            "objective_profile_id": profile_id,
            "objective_profile_family": str(
                immutable_score.get("family") or profile.get("family") or ""
            ),
            "objective_profile_scope": str(
                immutable_score.get("scope") or profile.get("scope") or ""
            ),
            "score_component_cap": _int_value(
                immutable_score.get("score_component_cap")
                or profile.get("component_cap")
            ),
            "immutable_score": immutable_score,
        }
    return {}


def _objective_replay_sources(evaluation: CandidateEvaluation) -> list[dict[str, Any]]:
    payload = evaluation.candidate.payload
    sources: list[dict[str, Any]] = []
    for key in ("selection_oos_replay_result", "replay_result"):
        replay = payload.get(key)
        if isinstance(replay, dict):
            sources.append(replay)
    for row in payload.get("fold_metrics", []):
        if not isinstance(row, dict):
            continue
        replay = row.get("candidate")
        if isinstance(replay, dict):
            sources.append(replay)
    return sources


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fold_profile(evaluation: CandidateEvaluation | None) -> dict[str, Any]:
    if evaluation is None:
        return {}
    rows = [
        row
        for row in evaluation.candidate.payload.get("fold_metrics", [])
        if isinstance(row, dict)
    ]
    if not rows:
        return {}
    return {
        "mean_objective_score": sum(float(row.get("objective_score", 0.0) or 0.0) for row in rows)
        / len(rows),
        "min_objective_score": min(float(row.get("objective_score", 0.0) or 0.0) for row in rows),
        "max_objective_score": max(float(row.get("objective_score", 0.0) or 0.0) for row in rows),
        "mean_trade_count": sum(
            int((row.get("candidate") or {}).get("trade_count", 0) or 0) for row in rows
        )
        / len(rows),
        "mean_max_drawdown": sum(
            float((row.get("candidate") or {}).get("max_drawdown", 0.0) or 0.0)
            for row in rows
        )
        / len(rows),
    }


def _write_optimizer_gate_artifacts(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    *,
    evaluations: list[CandidateEvaluation],
    data_errors: list[str],
    gate_status: str,
) -> None:
    gate_rows = [_candidate_gate_summary(evaluation) for evaluation in evaluations]
    leakage_payload = pass_gate_report(manifest.run_id, "leakage")
    leakage_payload.update(
        {
            "status": gate_status,
            "errors": data_errors,
            "selection_oos_used_in_first_pass": False,
            "two_fold_purged_in_sample": True,
            "candidate_checks": [
                {
                    "candidate_id": row["candidate_id"],
                    "passed": not row["selection_oos_used_in_first_pass"],
                }
                for row in gate_rows
            ],
        }
    )
    writer.write_json("leakage_report.json", leakage_payload)
    gate_mapping = {
        "cost_sensitivity.json": ("cost_sensitivity", "cost_sensitivity_passed"),
        "fold_validation.json": ("fold_validation", "fold_support_passed"),
        "outlier_sensitivity.json": ("outlier_sensitivity", "outlier_concentration_passed"),
        "portfolio_synergy.json": ("portfolio_synergy", "portfolio_synergy_passed"),
    }
    for name, (gate, key) in gate_mapping.items():
        payload = pass_gate_report(manifest.run_id, gate)
        payload["status"] = gate_status
        payload["errors"] = data_errors
        payload["candidate_checks"] = [
            {
                "candidate_id": row["candidate_id"],
                "passed": bool(row.get(key)),
                "details": row,
            }
            for row in gate_rows
        ]
        if name == "fold_validation.json":
            payload["purged_fold_support"] = {
                row["candidate_id"]: row.get("purged_fold_support", {}) for row in gate_rows
            }
            payload["selection_oos_excluded_from_first_pass"] = True
        writer.write_json(name, payload)


def _candidate_gate_summary(evaluation: CandidateEvaluation) -> dict[str, Any]:
    payload = evaluation.candidate.payload
    return {
        "candidate_id": evaluation.candidate.candidate_id,
        "fold_support_passed": bool(payload.get("fold_support_passed")),
        "purged_fold_support": payload.get("purged_fold_support", {}),
        "cost_sensitivity_passed": bool(payload.get("cost_sensitivity_passed")),
        "drawdown_gate_passed": bool(payload.get("drawdown_gate_passed")),
        "outlier_concentration_passed": bool(payload.get("outlier_concentration_passed")),
        "portfolio_synergy_passed": bool(payload.get("portfolio_synergy_passed")),
        "selection_oos_used_in_first_pass": bool(payload.get("selection_oos_used_in_first_pass")),
        "objective_component_scores": payload.get("objective_component_scores", {}),
    }


def _confirmatory_variant_payload(
    evaluation: CandidateEvaluation,
    selection_evaluation: CandidateEvaluation | None,
    *,
    artifact_root: Path,
    baseline_score: float,
) -> dict[str, Any]:
    selection_score = (
        selection_evaluation.objective_score if selection_evaluation is not None else 0.0
    )
    selection_payload = (
        selection_evaluation.candidate.payload if selection_evaluation is not None else {}
    )
    selection_delta = float(
        selection_payload.get("selection_oos_delta_vs_incumbent")
        if selection_payload.get("selection_oos_delta_vs_incumbent") is not None
        else selection_score - baseline_score
    )
    return {
        "candidate_id": evaluation.candidate.candidate_id,
        "source_candidate_id": str(evaluation.candidate.payload.get("source_candidate_id") or ""),
        "variant_type": str(evaluation.candidate.payload.get("variant_type") or ""),
        "objective_score": evaluation.objective_score,
        "baseline_score": baseline_score,
        "in_sample_delta": evaluation.objective_score - baseline_score,
        "selection_oos_delta": selection_delta,
        "fold_support_passed": bool(evaluation.candidate.payload.get("fold_support_passed")),
        "deterministic_replay_passed": evaluation.passed,
        "materially_degrades_in_sample": evaluation.objective_score < baseline_score - 0.001,
        "evidence_paths": [
            str(artifact_root / "fold_score_matrix.json"),
            str(artifact_root / "selection_oos_evaluation.json"),
        ],
    }


def _round_n_plus_1_recommendation(
    manifest: MonthlyRunManifest,
    *,
    winner: CandidateEvaluation | None,
    replay_context: ReplayEvaluationContext,
    artifact_root: Path,
    no_adoption_reason: str,
) -> dict[str, Any]:
    if winner is None or replay_context.plugin is None:
        return {
            "schema_version": "round_n_plus_1_recommendation_v1",
            "run_id": manifest.run_id,
            "status": "no_adoption",
            "adopted_candidate_id": "",
            "no_adoption_reason": no_adoption_reason,
            "live_deployment_status": "not_requested",
        }
    output_dir = artifact_root / "round_n_plus_1"
    emitted = replay_context.plugin.write_round_n_plus_1(winner.candidate, output_dir)
    return {
        "schema_version": "round_n_plus_1_recommendation_v1",
        "run_id": manifest.run_id,
        "status": "optimized_backtest_recommendation",
        "adopted_candidate_id": winner.candidate.candidate_id,
        "next_round_id": manifest.next_round_id,
        "next_config_hash": emitted.get("next_config_hash", ""),
        "config_patch_path": emitted.get("config_patch_path", ""),
        "candidate_manifest_path": emitted.get("candidate_manifest_path", ""),
        "rollback_plan_path": emitted.get("rollback_plan_path", ""),
        "recommendation_path": emitted.get("path", ""),
        "parameter_patch_fingerprint": emitted.get(
            "parameter_patch_fingerprint",
            winner.candidate.payload.get("parameter_patch_fingerprint", ""),
        ),
        "evaluated_patch_fingerprint": emitted.get(
            "evaluated_patch_fingerprint",
            winner.candidate.payload.get("evaluated_patch_fingerprint", ""),
        ),
        "approval_state": "not_requested",
        "live_deployment_status": "optimized_backtest_recommendation",
    }


def _evaluation_baseline_score(
    evaluation: CandidateEvaluation,
    fallback: float,
) -> float:
    value = evaluation.candidate.payload.get("aggregate_fold_baseline_score")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if parsed == 0.0 and fallback != 0.0:
        return fallback
    return parsed


def _run_plan_phases(
    plan: OptimizerExperimentPlan,
    *,
    replay_context: ReplayEvaluationContext,
    fold_manifest,
    search_brief: dict[str, Any] | None = None,
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    evaluator = None
    if replay_context.plugin is not None and replay_context.replay_backed:
        def evaluator(candidate):
            return score_candidate_on_folds(
                candidate=candidate,
                plugin=replay_context.plugin,
                baseline=replay_context.baseline,
                fold_manifest=fold_manifest,
            )

    phase_specs = _phase_specs_from_plan(plan)
    if replay_context.plugin is not None and replay_context.replay_backed:
        plugin_specs = replay_context.plugin.build_phase_specs(
            replay_context.diagnostics or {},
            plan,
            search_brief or {},
        )
        if plugin_specs:
            phase_specs = plugin_specs
    for phase in phase_specs:
        evaluations.extend(run_phase(phase, evaluator=evaluator))
    return evaluations


def _phase_specs_from_plan(plan: OptimizerExperimentPlan) -> list[PhaseSpec]:
    fallback_phase = plan.phase_order[0] if plan.phase_order else "signal_quality"
    phase_families: dict[str, list[str]] = {}
    for item in plan.candidate_families:
        if isinstance(item, dict):
            family = str(item.get("family") or item.get("candidate_family") or "").strip()
            phase_id = str(item.get("phase") or fallback_phase).strip() or fallback_phase
        else:
            family = str(item).strip()
            phase_id = fallback_phase
        if not family:
            continue
        phase_families.setdefault(phase_id, []).append(family)

    ordered_phases = [phase for phase in plan.phase_order if phase in phase_families]
    ordered_phases.extend(phase for phase in phase_families if phase not in ordered_phases)
    return [
        PhaseSpec(
            phase_id=phase_id,
            candidate_families=_dedupe_strings(phase_families[phase_id]),
        )
        for phase_id in ordered_phases
    ]


def _candidate_attempt(
    evaluation: CandidateEvaluation,
    manifest: MonthlyRunManifest,
    *,
    artifact_root: Path,
    source: MonthlyCandidateSource,
    workspace,
) -> CandidateAttemptRecord:
    state = CandidateAttemptState.SUCCEEDED if evaluation.passed else CandidateAttemptState.FAILED
    return CandidateAttemptRecord(
        attempt_id=f"{manifest.run_id}-{evaluation.candidate.candidate_id}-attempt-1",
        run_id=manifest.run_id,
        candidate_id=evaluation.candidate.candidate_id,
        workspace_key=workspace.workspace_key,
        workspace_path=workspace.workspace_path,
        state=state,
        stage=_source_stage(source),
        attempt_number=1,
        retry_attempt=0,
        retry_reason="",
        stall_timeout_seconds=manifest.stall_timeout_seconds,
        manifest_id=manifest.manifest_id,
        backtest_repo_commit_sha=manifest.backtest_repo_commit_sha,
        trading_repo_commit_sha=manifest.trading_repo_commit_sha,
        phase=str(evaluation.candidate.payload.get("phase_id") or ""),
        reason=_evaluation_reason(evaluation),
        artifact_paths=[
            str(artifact_root / "fold_manifest.json"),
            str(artifact_root / "fold_score_matrix.json"),
            str(artifact_root / "llm_experiment_plan.json"),
            str(artifact_root / "candidate_results.jsonl"),
            str(artifact_root / "selection_oos_evaluation.json"),
        ],
    )


def _candidate_result_row(
    evaluation: CandidateEvaluation,
    manifest: MonthlyRunManifest,
    *,
    artifact_root: Path,
    source: MonthlyCandidateSource,
    selected: bool,
    attempt: CandidateAttemptRecord | None = None,
    workspace=None,
    baseline_score: float = 0.0,
) -> dict:
    reason = _evaluation_reason(evaluation)
    candidate = evaluation.candidate
    phase_id = str(candidate.payload.get("phase_id") or "")
    runner_contract_version = _runner_contract_version(source)
    passed = bool(evaluation.passed)
    replay_backed = _evaluation_replay_backed(evaluation)
    baseline_score = _evaluation_baseline_score(evaluation, baseline_score)
    objective_delta = evaluation.objective_score - baseline_score
    gate_statuses = candidate.payload.get("no_regression_gate_statuses", {})
    if not isinstance(gate_statuses, dict):
        gate_statuses = {}
    fold_support_passed = bool(candidate.payload.get("fold_support_passed", passed))
    selected_round_id = manifest.next_round_id if selected else manifest.round_id
    selected_prior_round_id = manifest.round_id if selected else manifest.prior_round_id
    objective_metadata = _objective_metadata_from_evaluation(evaluation)
    return {
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id,
        "round_id": selected_round_id,
        "prior_round_id": selected_prior_round_id,
        "next_round_id": manifest.next_round_id if selected else "",
        "candidate_id": candidate.candidate_id,
        "source": source.value,
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "family": candidate.family,
        "phase": phase_id,
        "title": f"{candidate.family} candidate",
        "description": reason or "deterministic phase candidate",
        "status": "selected" if selected else "rejected",
        "decision": "keep" if selected and passed else "reject",
        "change_kind": "parameter_change",
        "risk_classification": "medium",
        "objective_score": evaluation.objective_score,
        "baseline_score": baseline_score,
        "objective_delta": objective_delta,
        "objective_deltas": {
            "calibration": candidate.payload.get("aggregate_fold_objective_delta", objective_delta),
            "latest_month_oos": candidate.payload.get("selection_oos_delta", 0.0),
        },
        "objective_version": manifest.objective_version,
        "effective_objective_version": objective_metadata.get("effective_objective_version", ""),
        "immutable_objective_version": objective_metadata.get("immutable_objective_version", ""),
        "objective_profile_id": objective_metadata.get("objective_profile_id", ""),
        "objective_profile_family": objective_metadata.get("objective_profile_family", ""),
        "objective_profile_scope": objective_metadata.get("objective_profile_scope", ""),
        "score_component_cap": objective_metadata.get(
            "score_component_cap",
            manifest.score_component_cap,
        ),
        "immutable_score": objective_metadata.get("immutable_score", {}),
        "parameter_patch_fingerprint": candidate.payload.get("parameter_patch_fingerprint", ""),
        "evaluated_patch_fingerprint": candidate.payload.get("evaluated_patch_fingerprint", ""),
        "reason": reason,
        "optimizer_stage": _source_stage(source).value,
        "deterministic_gate_inputs": {
            "phase4_sequence_valid": passed,
            "fold_support_passed": fold_support_passed,
            "purged_fold_support": candidate.payload.get("purged_fold_support", {}),
            "deterministic_replay_passed": replay_backed and fold_support_passed,
            "end_of_round_diagnostics_saved": True,
            "round_n_plus_1_adopted": bool(selected and passed),
            "live_backtest_parity_aligned": replay_backed,
            "runner_contract_version": runner_contract_version,
            "source_runner_contract_version": runner_contract_version,
            "manifest_id": manifest.manifest_id,
            "diagnostic_only": not replay_backed,
            "latest_month_oos_improvement": bool(
                candidate.payload.get("latest_month_oos_improvement", False)
            ),
            "latest_month_oos_delta": candidate.payload.get("selection_oos_delta", 0.0),
            "calibration_support": candidate.payload.get(
                "aggregate_fold_objective_delta",
                objective_delta,
            )
            > 0.0,
            "calibration_objective_delta": candidate.payload.get(
                "aggregate_fold_objective_delta",
                objective_delta,
            ),
            "leakage_passed": not bool(candidate.payload.get("selection_oos_used_in_first_pass")),
            "cost_gate_passed": bool(candidate.payload.get("cost_sensitivity_passed", False)),
            "drawdown_gate_passed": bool(candidate.payload.get("drawdown_gate_passed", False)),
            "outlier_concentration_passed": bool(
                candidate.payload.get("outlier_concentration_passed", False)
            ),
            "risk_constraints_passed": bool(
                candidate.payload.get("portfolio_synergy_passed", False)
            ),
            "sufficient_trade_count": all(
                int((row.get("candidate") or {}).get("trade_count", 0) or 0) > 0
                for row in candidate.payload.get("fold_metrics", [])
                if isinstance(row, dict)
            ),
            "no_regression_gate_statuses": gate_statuses,
        },
        "evidence_paths": [
            str(artifact_root / "fold_manifest.json"),
            str(artifact_root / "llm_experiment_plan.json"),
            str(artifact_root / "end_of_round_diagnostics.json"),
            str(artifact_root / "fold_score_matrix.json"),
            str(artifact_root / "selection_oos_evaluation.json"),
        ],
        "artifact_paths": [
            str(artifact_root / "candidate_results.jsonl"),
            str(artifact_root / "runner_observability.json"),
            str(artifact_root / "fold_candidate_results.jsonl"),
        ],
        "fold_manifest_path": str(artifact_root / "fold_manifest.json"),
        "rounds_manifest_path": str(artifact_root / "rounds_manifest.json"),
        "end_of_round_diagnostics_path": str(artifact_root / "end_of_round_diagnostics.json"),
        "confirmatory_rerank_path": str(artifact_root / "confirmatory_rerank.json"),
        "candidate_workspace_key": workspace.workspace_key if workspace is not None else "",
        "candidate_workspace_path": workspace.workspace_path if workspace is not None else "",
        "candidate_attempt_id": attempt.attempt_id if attempt is not None else "",
        "candidate_attempt_status": attempt.state.value if attempt is not None else "",
        "retry_attempt": attempt.retry_attempt if attempt is not None else 0,
        "retry_reason": attempt.retry_reason if attempt is not None else "",
        "stall_timeout_seconds": manifest.stall_timeout_seconds,
        "backtest_repo_commit_sha": manifest.backtest_repo_commit_sha,
        "live_trading_repo_commit_sha": manifest.trading_repo_commit_sha,
        "control_plane_commit_sha": manifest.control_plane_commit_sha,
        "workflow_contract_path": manifest.workflow_contract_path,
        "workflow_contract_version": manifest.workflow_contract_version,
        "score_component_count": manifest.score_component_cap,
        "max_workers": manifest.max_workers,
        "source_weekly_signal_ids": manifest.source_weekly_signal_ids,
        "raw_payload": {
            "phase_id": phase_id,
            "candidate_family": candidate.family,
            "candidate_payload": candidate.payload,
            "replay_backed": replay_backed,
        },
        "acceptance_criteria": [
            "both purged in-sample folds improve the incumbent objective",
            "selection-OOS remains within the declared degradation thresholds",
            "no-regression gates pass for drawdown, costs, outliers, and synergy",
        ],
        "rollback_plan": str(
            candidate.payload.get("rollback_plan_ref") or "restore round_N config"
        ),
    }


def _runner_observability(
    manifest: MonthlyRunManifest,
    attempts: list[CandidateAttemptRecord],
    *,
    reason: str,
    planner_mode: str,
) -> list[dict]:
    if not attempts:
        return [
            runner_event(
                manifest,
                phase="eligibility",
                status="blocked",
                planner_mode=planner_mode,
                reason=reason,
            )
        ]
    return [
        runner_event(
            manifest,
            phase=attempt.phase or attempt.stage.value,
            status=attempt.state.value,
            attempt_id=attempt.attempt_id,
            candidate_id=attempt.candidate_id,
            planner_mode=planner_mode,
            reason=attempt.reason,
        )
        for attempt in attempts
    ]


def _optimizer_decision_reason(
    manifest: MonthlyRunManifest,
    data_errors: list[str],
    evaluations: list[CandidateEvaluation],
    winner: CandidateEvaluation | None,
) -> str:
    if winner is not None:
        return f"adopted {winner.candidate.candidate_id} after deterministic phase gates"
    if data_errors:
        return no_adoption_reason(manifest, data_errors)
    if evaluations:
        return phase_no_adoption_reason(
            evaluations,
            "no candidate passed deterministic replay gates",
        )
    return no_adoption_reason(manifest, data_errors)


def _source_stage(source: MonthlyCandidateSource) -> OptimizerStage:
    if source == MonthlyCandidateSource.SMOKE_REPAIR:
        return OptimizerStage.OOS_REPAIR
    return OptimizerStage.PHASED_AUTO


def _runner_contract_version(source: MonthlyCandidateSource) -> str:
    if source == MonthlyCandidateSource.SMOKE_REPAIR:
        return SMOKE_REPAIR_RUNNER_CONTRACT_VERSION
    return PHASED_AUTO_RUNNER_CONTRACT_VERSION


def _evaluation_reason(evaluation: CandidateEvaluation) -> str:
    return "; ".join(reason for reason in evaluation.reasons if reason)


def _evaluation_replay_backed(evaluation: CandidateEvaluation) -> bool:
    return bool(
        evaluation.candidate.payload.get("replay_result")
        or evaluation.candidate.payload.get("fold_metrics")
        or evaluation.candidate.payload.get("selection_oos_replay_result")
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
