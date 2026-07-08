"""Reusable fixture implementation of the external monthly runner contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import (
    PHASE4_OPTIMIZER_ARTIFACTS,
    REQUIRED_BACKTEST_ARTIFACTS,
    BacktestArtifactIndex,
)
from trading_assistant.schemas.decision_parity import DECISION_PARITY_DIMENSIONS
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest, MonthlyRunMode
from trading_assistant.skills.monthly_optimizer_runner import CandidateWorkspaceManager, build_two_fold_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--failure",
        choices=[
            "missing_index",
            "outside_path",
            "stale_artifact",
            "malformed_json",
            "missing_optimizer_diagnostics",
            "structural_missing_patch",
            "decision_parity_mismatch",
        ],
        default="",
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = MonthlyRunManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    root = Path(manifest.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_required(root, manifest, malformed_json=args.failure == "malformed_json")
    if manifest.mode == MonthlyRunMode.PHASED_AUTO:
        _write_optimizer(
            root,
            manifest,
            structural_missing_patch=args.failure == "structural_missing_patch",
            decision_parity_mismatch=args.failure == "decision_parity_mismatch",
        )
    if args.failure == "missing_optimizer_diagnostics":
        (root / "runner_observability.json").unlink(missing_ok=True)
    if args.failure == "stale_artifact":
        old = manifest_path.stat().st_mtime - 10
        os.utime(root / "incumbent_validation.json", (old, old))
    if args.failure == "missing_index":
        return 0
    artifacts = {
        name: str(root / name)
        for name in [*REQUIRED_BACKTEST_ARTIFACTS, *PHASE4_OPTIMIZER_ARTIFACTS]
        if (root / name).exists()
    }
    if args.failure == "outside_path":
        outside = root.parent / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        artifacts["coverage_manifest.json"] = str(outside)
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id if manifest.mode == MonthlyRunMode.PHASED_AUTO else "",
        artifact_root=str(root),
        artifacts=artifacts,
    )
    (root / "artifact_index.json").write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return 0


def _write_required(root: Path, manifest: MonthlyRunManifest, *, malformed_json: bool) -> None:
    payloads = {
        "coverage_manifest.json": {
            "run_id": manifest.run_id,
            "status": "pass",
            "data_bundle_checksum": manifest.data_bundle_checksum or manifest.data_manifest_checksum,
        },
        "incumbent_validation.json": {"run_id": manifest.run_id, "objective_delta": 0.0},
        "gap_attribution.json": {"run_id": manifest.run_id, "primary_category": "none"},
        "mode_decision.json": {
            "run_id": manifest.run_id,
            "status": "experiment" if manifest.mode == MonthlyRunMode.PHASED_AUTO else "no_change",
        },
        "replay_parity_report.json": {
            "bot_id": manifest.bot_id,
            "strategy_id": manifest.strategy_id,
            "run_month": manifest.run_month,
            "trade_count_live": 1,
            "trade_count_replay": 1,
            "entry_match_rate": 1.0,
            "exit_match_rate": 1.0,
            "side_quantity_match_rate": 1.0,
            "status": "pass",
        },
        "objective_breakdown.json": {
            "run_id": manifest.run_id,
            "objective_version": manifest.objective_version,
        },
        "selected_candidates.json": [],
        "exit_status.json": {"exit_code": 0, "timed_out": False},
    }
    for name, payload in payloads.items():
        path = root / name
        if malformed_json and name == "objective_breakdown.json":
            path.write_text("{not-json", encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
    (root / "candidate_results.jsonl").write_text("", encoding="utf-8")
    (root / "rejected_candidates.jsonl").write_text("", encoding="utf-8")
    (root / "monthly_report.md").write_text("fixture monthly report", encoding="utf-8")
    (root / "stdout.log").write_text("", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")


def _write_optimizer(
    root: Path,
    manifest: MonthlyRunManifest,
    *,
    structural_missing_patch: bool,
    decision_parity_mismatch: bool,
) -> None:
    _write_optimizer_run_manifest(root, manifest)
    candidate_id = "fixture-structural" if structural_missing_patch or decision_parity_mismatch else "fixture-candidate"
    workspace = CandidateWorkspaceManager(root / "workspaces").prepare(
        run_id=manifest.run_id,
        candidate_id=candidate_id,
        workspace_key=candidate_id,
        structural=structural_missing_patch,
    )
    fold_manifest = build_two_fold_manifest(
        run_id=manifest.run_id,
        run_month=manifest.run_month,
        in_sample_start=manifest.in_sample_start,
        in_sample_end=manifest.in_sample_end,
        selection_oos_start=manifest.selection_oos_start,
        selection_oos_end=manifest.selection_oos_end,
        evidence_paths=[str(root / "objective_breakdown.json")],
    )
    (root / "fold_manifest.json").write_text(fold_manifest.model_dump_json(indent=2), encoding="utf-8")
    (root / "llm_experiment_plan.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "score_components": ["expected_return", "calmar", "profit_factor"],
        "phase_order": ["signal_quality"],
        "candidate_families": [{"family": "filter_repair", "phase": "signal_quality"}],
        "gate_expectations": ["positive purged folds"],
        "overfit_risks": ["sparse sample"],
        "evidence_paths": [str(root / "gap_attribution.json"), manifest.monthly_search_brief_path],
        "source_weekly_signal_ids": manifest.source_weekly_signal_ids,
    }), encoding="utf-8")
    (root / "confirmatory_rerank.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "primary_candidate_id": candidate_id,
        "primary_source": "phased_auto",
        "compared_candidate_ids": [candidate_id],
        "variants": [{"candidate_id": candidate_id, "fold_support_passed": True, "deterministic_replay_passed": True}],
        "adopted_candidate_id": candidate_id,
        "selection_rule": "fixture",
        "evidence_paths": [str(root / "fold_validation.json")],
    }), encoding="utf-8")
    (root / "rounds_manifest.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "current_round_id": manifest.round_id or "round_1",
        "next_round_id": manifest.next_round_id or "round_2",
        "adopted_candidate_id": candidate_id,
        "records": [{
            "round_id": manifest.next_round_id or "round_2",
            "prior_round_id": manifest.round_id or "round_1",
            "candidate_id": candidate_id,
            "source": "phased_auto",
            "decision_parity_report_path": str(root / "decision_parity_report.json"),
        }],
    }), encoding="utf-8")
    candidate = _candidate_payload(root, manifest, candidate_id, workspace.workspace_path)
    if structural_missing_patch or decision_parity_mismatch:
        candidate["change_kind"] = "structural_change"
        candidate["file_changes"] = [{"file_path": "strategies/alpha.py", "kind": "modify"}]
        candidate["decision_parity_report_path"] = str(root / "decision_parity_report.json")
    if decision_parity_mismatch:
        for name in ("live_repo.patch", "backtest_adapter.patch"):
            (root / name).write_text("fixture patch", encoding="utf-8")
        candidate["live_repo_patch_path"] = str(root / "live_repo.patch")
        candidate["backtest_adapter_patch_path"] = str(root / "backtest_adapter.patch")
    (root / "selected_candidates.json").write_text(json.dumps([candidate]), encoding="utf-8")
    (root / "candidate_results.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    (root / "candidate_attempts.jsonl").write_text(json.dumps({
        "attempt_id": "attempt-1",
        "run_id": manifest.run_id,
        "candidate_id": candidate_id,
        "workspace_key": candidate_id,
        "workspace_path": workspace.workspace_path,
        "state": "succeeded",
        "stage": "confirmatory_follow_up",
        "attempt_number": 1,
        "manifest_id": manifest.manifest_id,
        "artifact_paths": [str(root / "confirmatory_rerank.json")],
    }) + "\n", encoding="utf-8")
    (root / "runner_observability.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "attempt_id": "attempt-1",
        "attempt_state": "succeeded",
        "phase": "confirmatory_follow_up",
    }), encoding="utf-8")
    _write_p6_p7(root, manifest, candidate_id)
    for name in [
        "leakage_report.json",
        "cost_sensitivity.json",
        "fold_validation.json",
        "outlier_sensitivity.json",
        "portfolio_synergy.json",
        "end_of_round_diagnostics.json",
        "candidate_workspace_manifest.json",
    ]:
        (root / name).write_text(json.dumps({"run_id": manifest.run_id, "status": "pass"}), encoding="utf-8")
    _write_decision_parity(root, manifest, candidate_id, mismatch=decision_parity_mismatch)


def _write_optimizer_run_manifest(root: Path, manifest: MonthlyRunManifest) -> None:
    run_manifest_path = root / "run_manifest.json"
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
    contract_paths = _manifest_path_map(
        manifest,
        ("bridge_contract_paths", "strategy_plugin_contract_paths"),
    )
    deployment_paths = _manifest_path_map(
        manifest,
        ("bridge_deployment_metadata_paths", "deployment_metadata_paths"),
    )
    if contract_path is not None:
        contract_paths.setdefault(_fixture_bridge_id(manifest), str(contract_path))
    if deployment_path is not None:
        deployment_paths.setdefault(_fixture_bridge_id(manifest), str(deployment_path))
    contract_hashes = _hash_path_map(contract_paths)
    deployment_hashes = _hash_path_map(deployment_paths)
    approval_mode = str(getattr(manifest.approval_mode, "value", manifest.approval_mode) or "")
    approval_evidence_mode = bool(getattr(manifest, "approval_evidence_mode", False))
    approval_grade = approval_mode not in {"", "none"} or approval_evidence_mode
    scope_id = _fixture_scope_id(manifest)
    (root / "optimizer_run_manifest.json").write_text(json.dumps({
        "schema_version": "optimizer_approval_run_manifest_v1",
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id,
        "scope_id": scope_id,
        "scope_aliases": [
            item for item in (
                manifest.bot_id,
                manifest.strategy_id,
                manifest.strategy_plugin_id,
                scope_id,
            ) if item
        ],
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "strategy_plugin_id": manifest.strategy_plugin_id,
        "run_month": manifest.run_month,
        "run_mode": manifest.mode.value,
        "optimizer_mode": "approval_grade" if approval_grade else "shadow_validation",
        "approval_mode": approval_mode or "none",
        "approval_evidence_mode": approval_evidence_mode,
        "approval_grade_optimizer_run": approval_grade,
        "smoke_mode": not approval_grade,
        "artifact_root": str(root),
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_hash": _sha256_file(run_manifest_path),
        "data_bundle_checksum": manifest.data_bundle_checksum or manifest.data_manifest_checksum,
        "data_bundle_checksums": [
            manifest.data_bundle_checksum or manifest.data_manifest_checksum
        ],
        "strategy_plugin_contract_path": str(contract_path or ""),
        "strategy_plugin_contract_hash": _sha256_file(contract_path),
        "strategy_plugin_contract_paths": contract_paths,
        "bridge_contract_paths": contract_paths,
        "strategy_plugin_contract_hashes": contract_hashes,
        "bridge_contract_hashes": contract_hashes,
        "deployment_metadata_path": str(deployment_path or ""),
        "deployment_metadata_hash": _sha256_file(deployment_path),
        "deployment_metadata_paths": deployment_paths,
        "bridge_deployment_metadata_paths": deployment_paths,
        "deployment_metadata_hashes": deployment_hashes,
        "bridge_deployment_metadata_hashes": deployment_hashes,
    }), encoding="utf-8")


def _manifest_path_map(manifest: MonthlyRunManifest, keys: tuple[str, ...]) -> dict[str, str]:
    for key in keys:
        value = getattr(manifest, key, {})
        if isinstance(value, dict):
            mapped = {
                str(item_key).strip(): str(item).strip()
                for item_key, item in value.items()
                if str(item_key).strip() and str(item).strip()
            }
            if mapped:
                return mapped
    return {}


def _fixture_scope_id(manifest: MonthlyRunManifest) -> str:
    return {
        "crypto-trend-v1": "crypto_trader_portfolio",
        "crypto-momentum-v1": "crypto_trader_portfolio",
        "crypto-breakout-v1": "crypto_trader_portfolio",
    }.get(manifest.strategy_plugin_id, manifest.strategy_id)


def _fixture_bridge_id(manifest: MonthlyRunManifest) -> str:
    return {
        "crypto-trend-v1": "crypto_trend_v1",
        "crypto-momentum-v1": "crypto_momentum_v1",
        "crypto-breakout-v1": "crypto_breakout_v1",
    }.get(manifest.strategy_plugin_id, manifest.strategy_plugin_id or manifest.strategy_id)


def _hash_path_map(paths: dict[str, str]) -> dict[str, str]:
    return {
        bridge_id: digest
        for bridge_id, path in paths.items()
        for digest in [_sha256_file(Path(path))]
        if digest
    }


def _fixture_config_patch() -> dict:
    return {
        "family": "filter_repair",
        "filter_threshold_bps_delta": -2.0,
        "position_weight_multiplier": 1.05,
    }


def _fixture_evaluated_parameters() -> dict:
    return {
        "threshold_bps": 8.0,
        "position_weight": 1.05,
        "max_positions": 1,
    }


def _fixture_patch_fingerprints() -> tuple[str, str]:
    patch = _fixture_config_patch()
    evaluated = {
        "parameter_patch": patch,
        "evaluated_parameters": _fixture_evaluated_parameters(),
    }
    return _stable_fixture_hash(patch), _stable_fixture_hash(evaluated)


def _stable_fixture_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_p6_p7(root: Path, manifest: MonthlyRunManifest, candidate_id: str) -> None:
    patch_fingerprint, evaluated_fingerprint = _fixture_patch_fingerprints()
    candidate_replay = {
        "trade_count": 4,
        "net_return": 1.12,
        "max_drawdown": 0.04,
        "profit_factor": 1.6,
        "objective_score": 1.12,
        "trade_hash": "fixture-candidate-trades",
        "order_hash": "fixture-candidate-orders",
        "coverage": [{"rows": 4}],
        "parameter_patch": _fixture_config_patch(),
        "evaluated_parameter_patch": _fixture_config_patch(),
        "parameter_patch_fingerprint": patch_fingerprint,
        "evaluated_patch_fingerprint": evaluated_fingerprint,
        "evaluated_parameters": _fixture_evaluated_parameters(),
    }
    fold_rows = [
        {
            "run_id": manifest.run_id,
            "candidate_id": candidate_id,
            "candidate_family": "filter_repair",
            "fold_id": "fold_1",
            "purged": True,
            "selection_oos_used_in_first_pass": False,
            "objective_delta": 0.06,
            "fold_support_passed": True,
            "candidate": candidate_replay,
        },
        {
            "run_id": manifest.run_id,
            "candidate_id": candidate_id,
            "candidate_family": "filter_repair",
            "fold_id": "fold_2",
            "purged": True,
            "selection_oos_used_in_first_pass": False,
            "objective_delta": 0.05,
            "fold_support_passed": True,
            "candidate": candidate_replay,
        },
    ]
    (root / "fold_candidate_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in fold_rows) + "\n",
        encoding="utf-8",
    )
    (root / "fold_score_matrix.json").write_text(json.dumps({
        "schema_version": "fold_score_matrix_v1",
        "run_id": manifest.run_id,
        "selection_oos_excluded_from_first_pass": True,
        "scoring_windows": [
            {"fold_id": "fold_1", "purged": True, "embargo_days": 5},
            {"fold_id": "fold_2", "purged": True, "embargo_days": 5},
        ],
        "candidate_count": 1,
        "selected_candidate_ids": [candidate_id],
        "candidates": [{"candidate_id": candidate_id, "fold_support_passed": True}],
    }), encoding="utf-8")
    (root / "selection_oos_evaluation.json").write_text(json.dumps({
        "schema_version": "selection_oos_evaluation_v1",
        "run_id": manifest.run_id,
        "status": "pass",
        "selection_oos_used_after_fold_ranking": True,
        "selection_oos_used_in_first_pass": False,
        "primary_candidate_id": candidate_id,
        "incumbent_selection_oos": {"objective_score": 1.0, "trade_count": 4, "max_drawdown": 0.05},
        "candidate_selection_oos": {"candidate_id": candidate_id, "objective_score": 1.12, "trade_count": 4, "max_drawdown": 0.04},
    }), encoding="utf-8")
    (root / "selection_oos_repair_trigger.json").write_text(json.dumps({
        "schema_version": "selection_oos_repair_trigger_v1",
        "run_id": manifest.run_id,
        "triggered": False,
        "status": "not_triggered",
        "thresholds": {
            "objective_drop_threshold": -0.05,
            "drawdown_increase_threshold": 0.05,
            "trade_count_collapse_ratio": 0.5,
        },
        "expected_is_fold_score_band": {"mean_objective_score": 1.1},
        "measured_degradation": {"objective_delta_vs_fold_mean": 0.02},
    }), encoding="utf-8")
    (root / "repair_failure_attribution.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "status": "complete",
        "repair_triggered": False,
    }), encoding="utf-8")
    (root / "accepted_mutation_chain.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "accepted_mutations": [],
    }), encoding="utf-8")
    (root / "repair_candidate_results.jsonl").write_text("", encoding="utf-8")
    (root / "repair_checkpoint.json").write_text(json.dumps({
        "schema_version": "repair_checkpoint_v1",
        "run_id": manifest.run_id,
        "repair_triggered": False,
        "candidate_ids": [],
        "deterministic_resume_key": "fixture-resume-key",
    }), encoding="utf-8")
    config_patch_path = root / "round_n_plus_1" / "config_patch.json"
    config_patch_path.parent.mkdir(parents=True, exist_ok=True)
    config_patch_path.write_text(json.dumps(_fixture_config_patch()), encoding="utf-8")
    (root / "round_n_plus_1_recommendation.json").write_text(json.dumps({
        "schema_version": "round_n_plus_1_recommendation_v1",
        "run_id": manifest.run_id,
        "status": "optimized_backtest_recommendation",
        "adopted_candidate_id": candidate_id,
        "next_round_id": manifest.next_round_id or "round_2",
        "live_deployment_status": "optimized_backtest_recommendation",
        "config_patch_path": str(config_patch_path),
        "parameter_patch_fingerprint": patch_fingerprint,
        "evaluated_patch_fingerprint": evaluated_fingerprint,
        "evaluated_parameters": _fixture_evaluated_parameters(),
    }), encoding="utf-8")


def _candidate_payload(root: Path, manifest: MonthlyRunManifest, candidate_id: str, workspace: str) -> dict:
    patch_fingerprint, evaluated_fingerprint = _fixture_patch_fingerprints()
    return {
        "candidate_id": candidate_id,
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id,
        "source": "phased_auto",
        "family": "filter_repair",
        "title": "Fixture candidate",
        "decision": "experiment",
        "objective_delta": 0.1,
        "objective_deltas": {"latest_month_oos": 0.1, "calibration": 0.05},
        "parameter_patch": _fixture_config_patch(),
        "evaluated_parameter_patch": _fixture_config_patch(),
        "evaluated_parameters": _fixture_evaluated_parameters(),
        "parameter_patch_fingerprint": patch_fingerprint,
        "evaluated_patch_fingerprint": evaluated_fingerprint,
        "candidate_workspace_key": candidate_id,
        "candidate_workspace_path": workspace,
        "candidate_attempt_id": "attempt-1",
        "candidate_attempt_status": "succeeded",
        "round_id": manifest.next_round_id or "round_2",
        "prior_round_id": manifest.round_id or "round_1",
        "next_round_id": manifest.next_round_id or "round_2",
        "backtest_repo_commit_sha": manifest.backtest_repo_commit_sha or "fixture-backtest-sha",
        "live_trading_repo_commit_sha": manifest.trading_repo_commit_sha or "fixture-live-sha",
        "control_plane_commit_sha": manifest.control_plane_commit_sha or "fixture-control-sha",
        "fold_manifest_path": str(root / "fold_manifest.json"),
        "rounds_manifest_path": str(root / "rounds_manifest.json"),
        "end_of_round_diagnostics_path": str(root / "end_of_round_diagnostics.json"),
        "confirmatory_rerank_path": str(root / "confirmatory_rerank.json"),
        "param_changes": [{"param_name": "threshold", "current": 1, "proposed": 2}],
        "acceptance_criteria": ["positive latest OOS"],
        "replay_or_experiment_plan": "shadow next month",
        "rollback_plan": "restore prior config",
        "evidence_paths": [str(root / "confirmatory_rerank.json")],
        "deterministic_gate_inputs": {
            "runner_contract_version": "phased_auto_runner_contract_v1",
            "phase4_sequence_valid": True,
            "round_n_plus_1_adopted": True,
            "end_of_round_diagnostics_saved": True,
            "live_backtest_parity_aligned": True,
            "latest_month_oos_improvement": True,
            "calibration_support": True,
            "fold_support_passed": True,
            "leakage_passed": True,
            "sufficient_trade_count": True,
            "cost_gate_passed": True,
            "drawdown_gate_passed": True,
            "outlier_concentration_passed": True,
            "risk_constraints_passed": True,
        },
    }


def _write_decision_parity(
    root: Path,
    manifest: MonthlyRunManifest,
    candidate_id: str,
    *,
    mismatch: bool,
) -> None:
    evidence = str(root / "end_of_round_diagnostics.json")
    (root / "decision_parity_report.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "candidate_id": candidate_id,
        "strategy_plugin_id": "wrong-plugin" if mismatch else manifest.strategy_plugin_id,
        "live_repo_commit_sha": manifest.trading_repo_commit_sha or "fixture-live-sha",
        "backtest_adapter_commit_sha": manifest.backtest_repo_commit_sha or "fixture-backtest-sha",
        "status": "pass",
        "evidence_paths": [evidence],
        "checks": [
            {
                "dimension": dimension,
                "status": "pass",
                "match_rate": 1.0,
                "mismatch_count": 0,
                "evidence_paths": [evidence],
            }
            for dimension in sorted(DECISION_PARITY_DIMENSIONS)
        ],
    }), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
