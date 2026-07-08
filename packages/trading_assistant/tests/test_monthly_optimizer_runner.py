from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import (
    PHASE4_OOS_REPAIR_ARTIFACTS,
    PHASE4_OPTIMIZER_ARTIFACTS,
    REQUIRED_BACKTEST_ARTIFACTS,
    BacktestArtifactIndex,
)
from trading_assistant.schemas.data_bundle_manifest import DataBundleManifest, DataBundleSlice, DataBundleStatus
from trading_assistant.schemas.market_data_manifest import MarketDataManifest
from trading_assistant.schemas.monthly_optimizer import (
    CandidateAttemptState,
    OptimizerSequenceStatus,
    OptimizerStage,
)
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.monthly_run_manifest import (
    MonthlyApprovalMode,
    MonthlyRunManifest,
    MonthlyRunMode,
)
from trading_assistant.schemas.strategy_plugin_contract import StrategyPluginContract, StrategyPluginMaturity
from trading_assistant.skills.backtest_runner_client import BacktestRunnerResult
from trading_assistant.skills.monthly_validation_orchestrator import (
    MonthlyValidationOrchestrator,
    MonthlyValidationRequest,
    _approval_evidence_artifact_base,
)
from trading_assistant.skills.monthly_optimizer_runner import (
    CandidateAttemptExecutor,
    CandidateAttemptStore,
    CandidateWorkspaceManager,
    MonthlyOptimizerRunner,
    build_two_fold_manifest,
)
from trading_assistant.skills.monthly_deployment_metadata import deployment_metadata_errors


def test_monthly_validation_request_defaults_to_optimizer_sequence() -> None:
    request = MonthlyValidationRequest(bot_id="bot1", strategy_id="strat1")

    assert request.optimizer_sequence_enabled is True


def test_approval_evidence_request_routes_to_scanned_artifact_base(tmp_path: Path) -> None:
    root = tmp_path / "trading_assistant_backtest" / "artifacts" / "monthly"
    request = MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        approval_evidence_mode=True,
    )

    assert _approval_evidence_artifact_base(root, request) == root / "approval_grade"
    assert (
        _approval_evidence_artifact_base(root / "approval_grade", request)
        == root / "approval_grade"
    )
    assert (
        _approval_evidence_artifact_base(
            root,
            MonthlyValidationRequest(bot_id="bot1", strategy_id="strat1"),
        )
        == root
    )


def _manifest(root: Path) -> MonthlyRunManifest:
    bundle = DataBundleManifest(
        data_repo_path=str(root / "market_data"),
        data_repo_commit_sha="fixture-data-sha",
        slice_manifests=[
            DataBundleSlice(
                manifest_path=str(root / "slice_market.json"),
                manifest_id="slice-1",
                source="fixture",
                market="equity",
                symbol="AAPL",
                timeframe="1m",
                checksum="slice-sha",
                calendar="XNYS",
                authoritative=True,
            )
        ],
        calendars=["XNYS"],
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        adjustment_policy="split_adjusted",
        status=DataBundleStatus.AUTHORITATIVE,
    )
    (root / "market.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return MonthlyRunManifest(
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        mode=MonthlyRunMode.PHASED_AUTO,
        bot_id="bot1",
        strategy_id="strat1",
        latest_month_start=date(2026, 4, 1),
        latest_month_end=date(2026, 4, 30),
        calibration_start=date(2026, 1, 1),
        calibration_end=date(2026, 3, 31),
        selection_oos_start=date(2026, 4, 1),
        selection_oos_end=date(2026, 4, 30),
        in_sample_start=date(2026, 1, 1),
        in_sample_end=date(2026, 3, 31),
        market_data_manifest_path=str(root / "market.json"),
        data_bundle_manifest_path=str(root / "market.json"),
        data_bundle_checksum=bundle.bundle_checksum,
        telemetry_manifest_path=str(root / "telemetry.json"),
        backtest_repo_path=str(root / "backtests"),
        artifact_root=str(root / "artifacts"),
        round_id="round_1",
        prior_round_id="round_0",
        next_round_id="round_2",
    )


def _approval_evidence_manifest(root: Path) -> MonthlyRunManifest:
    manifest = _manifest(root)
    contract_path = root / "strategy_plugin_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "plugin_id": "strat1-plugin",
                "backtest_adapter_path": "adapter.py",
                "config_schema_version": "config_v1",
                "decision_api_version": "decision_v1",
            }
        ),
        encoding="utf-8",
    )
    deployment_path = root / "deployment_metadata.json"
    deployment_path.write_text(
        json.dumps({"deployment_id": "deployment-1", "strategy_id": "strat1"}),
        encoding="utf-8",
    )
    return manifest.model_copy(
        update={
            "approval_evidence_mode": True,
            "strategy_plugin_contract_path": str(contract_path),
            "deployment_metadata_path": str(deployment_path),
        }
    )


def _phase4_index(manifest: MonthlyRunManifest) -> BacktestArtifactIndex:
    artifact_root = Path(manifest.artifact_root)
    return BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={
            name: str(artifact_root / name)
            for name in [
                *REQUIRED_BACKTEST_ARTIFACTS,
                *PHASE4_OPTIMIZER_ARTIFACTS,
            ]
        },
    )


def test_build_two_fold_manifest_keeps_latest_month_outside_scoring() -> None:
    manifest = build_two_fold_manifest(
        run_id="run1",
        run_month="2026-04",
        in_sample_start=date(2026, 1, 1),
        in_sample_end=date(2026, 3, 31),
        selection_oos_start=date(2026, 4, 1),
        selection_oos_end=date(2026, 4, 30),
        embargo_days=5,
    )

    assert len(manifest.folds) == 2
    assert all(fold.purged for fold in manifest.folds)
    assert manifest.folds[0].validation_end < manifest.folds[1].validation_start
    assert manifest.selection_oos_start > manifest.in_sample_end
    assert all(fold.embargo_days == 5 for fold in manifest.folds)


def test_deployment_metadata_gate_rejects_local_shadow_metadata_even_with_matching_hashes(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "strategy_plugin_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "plugin_id": "strat1-plugin",
                "backtest_adapter_path": "adapters/strat1.py",
                "config_schema_version": "config_v1",
                "decision_api_version": "decision_api_v1",
                "required_telemetry_schemas": ["trade_event_v1"],
            }
        ),
        encoding="utf-8",
    )
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "deployment_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "repo_url": "local://fixture",
                "source_control_origin": "local://fixture",
                "source_control_commit_sha": "live-sha",
                "source_control_worktree_clean": True,
                "deployed_commit_sha": "live-sha",
                "config_hash": "cfg-sha",
                "strategy_version": "strat-v1",
                "config_version": "cfg-v1",
                "telemetry_schema_version": "trade_event_v1",
                "strategy_plugin_contract_path": str(contract_path),
                "strategy_plugin_contract_hash": contract_hash,
                "runtime_entrypoint": "fixture.main",
                "runtime_instance_id": "instance-1",
                "runtime_host_fingerprint": "host-1",
                "live_runtime_started_at_utc": "2026-06-02T00:00:00Z",
                "emitted_at_utc": "2026-06-02T00:01:00Z",
                "emission_environment": "local",
                "metadata_source": "local_shadow_snapshot_v1",
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path).model_copy(
        update={
            "deployment_metadata_path": str(metadata_path),
            "strategy_plugin_contract_path": str(contract_path),
            "trading_repo_commit_sha": "live-sha",
            "config_hash": "cfg-sha",
            "strategy_version": "strat-v1",
            "config_version": "cfg-v1",
        }
    )

    errors = deployment_metadata_errors(
        manifest,
        missing_reason="deployment metadata required",
    )

    assert any("metadata_source" in error for error in errors)
    assert any("emission_environment" in error for error in errors)
    assert any("local://" in error for error in errors)


def test_workspace_manager_sanitizes_and_contains_candidate_workspace(tmp_path: Path) -> None:
    manager = CandidateWorkspaceManager(tmp_path / "workspaces")
    manifest = manager.prepare(
        run_id="run1",
        candidate_id="../candidate alpha",
        workspace_key="../candidate alpha",
        structural=True,
    )

    assert ".." not in manifest.workspace_key
    assert " " not in manifest.workspace_key
    Path(manifest.workspace_path).resolve().relative_to((tmp_path / "workspaces").resolve())
    assert Path(manifest.manifest_path).exists()
    assert manifest.cwd_enforced is True
    assert manifest.structural is True


def test_attempt_store_tracks_stall_retry_and_reconciliation(tmp_path: Path) -> None:
    workspace = CandidateWorkspaceManager(tmp_path / "workspaces").prepare(
        run_id="run1",
        candidate_id="cand1",
        workspace_key="cand1",
    )
    store = CandidateAttemptStore(tmp_path / "candidate_attempts.jsonl")
    claimed = store.claim(
        run_id="run1",
        candidate_id="cand1",
        workspace=workspace,
        manifest_id="manifest-a",
        stage=OptimizerStage.PHASED_AUTO,
        stall_timeout_seconds=10,
        backtest_repo_commit_sha="sha-a",
    )
    running = store.transition(claimed.attempt_id, CandidateAttemptState.RUNNING)
    stale_running = running.model_copy(update={
        "updated_at": datetime.now(timezone.utc) - timedelta(seconds=30),
    })
    store.append(stale_running)

    stalled = store.mark_stalled()
    assert stalled and stalled[0].state == CandidateAttemptState.STALLED

    retry = store.transition(claimed.attempt_id, CandidateAttemptState.RETRY_QUEUED, retry_reason="stall")
    assert retry.retry_attempt == 1
    assert store.retry_backoff_seconds(retry.retry_attempt) == 120

    canceled = store.reconcile(manifest_id="manifest-b")
    assert canceled and canceled[-1].state == CandidateAttemptState.CANCELED_BY_RECONCILIATION
    assert canceled[-1].reason == "run manifest changed"


def test_attempt_store_numbers_new_claims_after_terminal_attempts(tmp_path: Path) -> None:
    workspace = CandidateWorkspaceManager(tmp_path / "workspaces").prepare(
        run_id="run1",
        candidate_id="cand1",
        workspace_key="cand1",
    )
    store = CandidateAttemptStore(tmp_path / "candidate_attempts.jsonl")
    first = store.claim(
        run_id="run1",
        candidate_id="cand1",
        workspace=workspace,
        manifest_id="manifest-a",
    )
    store.transition(first.attempt_id, CandidateAttemptState.SUCCEEDED)

    second = store.claim(
        run_id="run1",
        candidate_id="cand1",
        workspace=workspace,
        manifest_id="manifest-a",
    )

    assert second.attempt_number == 2
    assert second.attempt_id != first.attempt_id


def test_optimizer_sequence_blocks_terminal_attempt_number_collision(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    attempts_path = artifact_root / "candidate_attempts.jsonl"
    attempt = json.loads(attempts_path.read_text(encoding="utf-8").splitlines()[0])
    collision = {
        **attempt,
        "attempt_id": "attempt-collision",
        "state": "succeeded",
    }
    attempts_path.write_text(
        json.dumps(attempt) + "\n" + json.dumps(collision) + "\n",
        encoding="utf-8",
    )
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any(
        "attempt_number collision" in reason and "attempt-collision" in reason
        for reason in result.blocking_reasons
    )


def test_attempt_executor_runs_with_cwd_equal_to_candidate_workspace(tmp_path: Path) -> None:
    workspace = CandidateWorkspaceManager(tmp_path / "workspaces").prepare(
        run_id="run1",
        candidate_id="cand1",
        workspace_key="cand1",
    )
    store = CandidateAttemptStore(tmp_path / "candidate_attempts.jsonl")
    attempt = store.claim(
        run_id="run1",
        candidate_id="cand1",
        workspace=workspace,
        manifest_id="manifest-a",
        stage=OptimizerStage.PHASED_AUTO,
        stall_timeout_seconds=10,
    )

    result = CandidateAttemptExecutor(store).run(
        attempt,
        [sys.executable, "-c", "from pathlib import Path; Path('cwd_marker.txt').write_text('ok')"],
    )

    assert result.state == CandidateAttemptState.SUCCEEDED
    assert Path(result.cwd) == Path(workspace.workspace_path)
    assert (Path(workspace.workspace_path) / "cwd_marker.txt").exists()
    assert store.latest_by_attempt()[attempt.attempt_id].state == CandidateAttemptState.SUCCEEDED


def test_optimizer_sequence_validates_repair_centered_round_adoption(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=True,
        adopted_candidate_id="cand-repair-local",
    )
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
            *PHASE4_OOS_REPAIR_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.SUCCEEDED
    assert result.repair_triggered is True
    assert result.adopted_candidate_id == "cand-repair-local"
    assert result.rounds_manifest_path.endswith("rounds_manifest.json")
    assert result.blocking_reasons == []


def test_optimizer_sequence_allows_repair_triggered_phase_winner_adoption(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=True,
        adopted_candidate_id="cand-phased",
        adopted_source="phased_auto",
    )
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
            *PHASE4_OOS_REPAIR_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.SUCCEEDED
    assert result.repair_triggered is True
    assert result.adopted_candidate_id == "cand-phased"
    assert result.blocking_reasons == []


def test_approval_optimizer_manifest_requires_complete_crypto_bridge_hash_sets(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "crypto_trend_contract.json"
    contract_path.write_text(json.dumps({"plugin_id": "crypto-trend-v1"}), encoding="utf-8")
    metadata_path = tmp_path / "crypto_trend_deployment.json"
    metadata_path.write_text(json.dumps({"plugin_id": "crypto-trend-v1"}), encoding="utf-8")
    manifest = _manifest(tmp_path).model_copy(
        update={
            "approval_mode": MonthlyApprovalMode.MANUAL_REQUIRED,
            "strategy_plugin_id": "crypto-trend-v1",
            "strategy_plugin_contract_path": str(contract_path),
            "deployment_metadata_path": str(metadata_path),
        }
    )
    artifact_root = Path(manifest.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "run_manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any(
        "missing strategy contract path(s)" in reason
        and "crypto_breakout_v1" in reason
        and "crypto_momentum_v1" in reason
        for reason in result.blocking_reasons
    )
    assert any(
        "missing deployment metadata path(s)" in reason
        and "crypto_breakout_v1" in reason
        and "crypto_momentum_v1" in reason
        for reason in result.blocking_reasons
    )


def test_optimizer_sequence_blocks_when_round_adoption_is_not_proven(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
        round_adopted_gate=False,
    )
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("round_n_plus_1_adopted" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_requires_p6_p7_evidence_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    for name in [
        "fold_score_matrix.json",
        "selection_oos_evaluation.json",
        "selection_oos_repair_trigger.json",
        "round_n_plus_1_recommendation.json",
    ]:
        (artifact_root / name).unlink()
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("fold_score_matrix.json" in reason for reason in result.blocking_reasons)
    assert any("selection_oos_evaluation.json" in reason for reason in result.blocking_reasons)
    assert any("round_n_plus_1_recommendation.json" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_blocks_unevaluated_round_patch_fingerprint(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    recommendation_path = artifact_root / "round_n_plus_1_recommendation.json"
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    recommendation["parameter_patch_fingerprint"] = "not-the-evaluated-patch"
    recommendation_path.write_text(json.dumps(recommendation), encoding="utf-8")
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("fingerprint" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_blocks_empty_confirmatory_variants_with_primary(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    confirmatory_path = artifact_root / "confirmatory_rerank.json"
    confirmatory = json.loads(confirmatory_path.read_text(encoding="utf-8"))
    confirmatory["variants"] = []
    confirmatory_path.write_text(json.dumps(confirmatory), encoding="utf-8")
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("variants cannot be empty" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_requires_selected_candidate_gate_proofs(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    selected_path = artifact_root / "selected_candidates.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected[0]["deterministic_gate_inputs"]["drawdown_gate_passed"] = False
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("drawdown gate" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_blocks_when_adopted_attempt_is_not_bound_to_candidate(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    attempts_path = artifact_root / "candidate_attempts.jsonl"
    attempt = json.loads(attempts_path.read_text(encoding="utf-8").splitlines()[0])
    attempt["candidate_id"] = "other-candidate"
    attempts_path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("different candidate" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_requires_search_brief_consumption_when_guidance_exists(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path).model_copy(update={
        "monthly_search_brief_path": str(tmp_path / "artifacts" / "monthly_search_brief.json"),
        "source_weekly_signal_ids": ["weekly-important"],
        "monthly_search_guidance": {
            "source_weekly_signal_ids": ["weekly-important"],
            "seed_candidates": [{"family": "filter_repair"}],
        },
    })
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    (artifact_root / "monthly_search_brief.json").write_text(
        json.dumps(manifest.monthly_search_guidance),
        encoding="utf-8",
    )
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("monthly_search_brief_path" in reason for reason in result.blocking_reasons)
    assert any("source_weekly_signal_ids" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_requires_search_brief_guidance_to_shape_plan(tmp_path: Path) -> None:
    base_manifest = _manifest(tmp_path)
    artifact_root = Path(base_manifest.artifact_root)
    manifest = base_manifest.model_copy(update={
        "monthly_search_brief_path": str(artifact_root / "monthly_search_brief.json"),
        "source_weekly_signal_ids": ["weekly-1"],
        "monthly_search_guidance": {
            "source_weekly_signal_ids": ["weekly-1"],
            "seed_candidates": [{"family": "entry_timing"}],
            "rollback_candidates": [{"family": "stop_loss"}],
            "negative_priors": [{"family": "weak_signal"}],
            "plan_requirements": {
                "candidate_families": ["entry_timing"],
                "rollback_families": ["stop_loss"],
                "negative_prior_families": ["weak_signal"],
            },
        },
    })
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    (artifact_root / "monthly_search_brief.json").write_text(
        json.dumps(manifest.monthly_search_guidance),
        encoding="utf-8",
    )
    plan_path = artifact_root / "llm_experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["evidence_paths"].append(str(artifact_root / "monthly_search_brief.json"))
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("candidate_families" in reason and "entry_timing" in reason for reason in result.blocking_reasons)
    assert any("rollback_families" in reason and "stop_loss" in reason for reason in result.blocking_reasons)
    assert any("negative_prior_families" in reason and "weak_signal" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_requires_every_search_brief_requirement(tmp_path: Path) -> None:
    base_manifest = _manifest(tmp_path)
    artifact_root = Path(base_manifest.artifact_root)
    manifest = base_manifest.model_copy(update={
        "monthly_search_brief_path": str(artifact_root / "monthly_search_brief.json"),
        "source_weekly_signal_ids": ["weekly-1", "weekly-2"],
        "monthly_search_guidance": {
            "source_weekly_signal_ids": ["weekly-1", "weekly-2"],
            "seed_candidates": [{"family": "entry_timing"}, {"family": "filter_repair"}],
            "plan_requirements": {
                "candidate_families": ["entry_timing", "filter_repair"],
            },
        },
    })
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-phased",
    )
    (artifact_root / "monthly_search_brief.json").write_text(
        json.dumps(manifest.monthly_search_guidance),
        encoding="utf-8",
    )
    plan_path = artifact_root / "llm_experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["evidence_paths"].append(str(artifact_root / "monthly_search_brief.json"))
    plan["source_weekly_signal_ids"] = ["weekly-1"]
    plan["candidate_families"] = [{"family": "entry_timing", "phase": "signal_quality"}]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("weekly-2" in reason for reason in result.blocking_reasons)
    assert any("filter_repair" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_allows_deterministic_no_adoption_without_attempts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_no_adoption_artifacts(artifact_root=artifact_root, manifest=manifest)
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.NO_ADOPTION
    assert result.no_adoption_reason == "insufficient mature replay plugin sample size"
    assert result.selected_candidate_ids == []
    assert result.blocking_reasons == []
    optimizer_manifest = json.loads(
        (artifact_root / "optimizer_run_manifest.json").read_text(encoding="utf-8")
    )
    assert optimizer_manifest["approval_grade_optimizer_run"] is False
    assert optimizer_manifest["smoke_mode"] is True


def test_optimizer_sequence_allows_shadow_approval_evidence_without_adoption(
    tmp_path: Path,
) -> None:
    manifest = _approval_evidence_manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "run_manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _write_no_adoption_artifacts(artifact_root=artifact_root, manifest=manifest)
    index = _phase4_index(manifest)

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)
    optimizer_manifest = json.loads(
        (artifact_root / "optimizer_run_manifest.json").read_text(encoding="utf-8")
    )

    assert result.status == OptimizerSequenceStatus.NO_ADOPTION
    assert result.no_adoption_reason == "insufficient mature replay plugin sample size"
    assert result.selected_candidate_ids == []
    assert result.blocking_reasons == []
    assert optimizer_manifest["approval_mode"] == "none"
    assert optimizer_manifest["approval_evidence_mode"] is True
    assert optimizer_manifest["approval_grade_optimizer_run"] is True
    assert optimizer_manifest["smoke_mode"] is False


def test_optimizer_sequence_blocks_stale_run_manifest_hash_in_approval_evidence_mode(
    tmp_path: Path,
) -> None:
    manifest = _approval_evidence_manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = artifact_root / "run_manifest.json"
    run_manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    _write_no_adoption_artifacts(artifact_root=artifact_root, manifest=manifest)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["config_hash"] = "stale-after-optimizer-manifest"
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    index = _phase4_index(manifest)

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("run_manifest_hash" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_blocks_structural_adoption_without_patch_lineage(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-structural",
        structural_without_patches=True,
    )
    index = BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(artifact_root),
        artifacts={name: str(artifact_root / name) for name in [
            *REQUIRED_BACKTEST_ARTIFACTS,
            *PHASE4_OPTIMIZER_ARTIFACTS,
        ]},
    )

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("structural candidate missing required lineage" in reason for reason in result.blocking_reasons)


def test_optimizer_sequence_blocks_structural_adoption_with_invalid_metadata_after_promotion(
    tmp_path: Path,
) -> None:
    contract_path = _write_strategy_plugin_contract(tmp_path)
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "deployment_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "repo_url": "local://fixture",
                "source_control_origin": "local://fixture",
                "source_control_commit_sha": "fixture-live-sha",
                "source_control_worktree_clean": True,
                "deployed_commit_sha": "fixture-live-sha",
                "config_hash": "cfg-sha",
                "strategy_version": "strat-v1",
                "config_version": "cfg-v1",
                "telemetry_schema_version": "trade_event_v1",
                "strategy_plugin_contract_path": str(contract_path),
                "strategy_plugin_contract_hash": contract_hash,
                "runtime_entrypoint": "fixture.main",
                "runtime_instance_id": "instance-1",
                "runtime_host_fingerprint": "host-1",
                "live_runtime_started_at_utc": "2026-06-02T00:00:00Z",
                "emitted_at_utc": "2026-06-02T00:01:00Z",
                "emission_environment": "local",
                "metadata_source": "local_shadow_snapshot_v1",
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path).model_copy(
        update={
            "strategy_plugin_id": "strat1-plugin",
            "strategy_plugin_contract_path": str(contract_path),
            "deployment_metadata_path": str(metadata_path),
            "trading_repo_commit_sha": "fixture-live-sha",
            "backtest_repo_commit_sha": "fixture-backtest-sha",
            "config_hash": "cfg-sha",
            "strategy_version": "strat-v1",
            "config_version": "cfg-v1",
        }
    )
    artifact_root = Path(manifest.artifact_root)
    _write_phase4_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        repair_triggered=False,
        adopted_candidate_id="cand-structural",
        structural_with_patches=True,
    )
    index = _phase4_index(manifest)

    result = MonthlyOptimizerRunner().validate_artifacts(manifest, index)

    assert result.status == OptimizerSequenceStatus.BLOCKED
    assert any("metadata_source" in reason for reason in result.blocking_reasons)
    assert any("local://" in reason for reason in result.blocking_reasons)


def test_orchestrator_integrates_optimizer_sequence_result(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_monthly_inputs(tmp_path)
    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
    )

    class FakeRunner:
        def run(self, manifest: MonthlyRunManifest, manifest_path: Path) -> BacktestRunnerResult:
            artifact_root = Path(manifest.artifact_root)
            _write_phase4_artifacts(
                artifact_root=artifact_root,
                manifest=manifest,
                repair_triggered=False,
                adopted_candidate_id="cand-phased",
                consume_search_guidance=True,
            )
            index = BacktestArtifactIndex(
                run_id=manifest.run_id,
                manifest_id=manifest.manifest_id,
                artifact_root=str(artifact_root),
                artifacts={name: str(artifact_root / name) for name in [
                    *REQUIRED_BACKTEST_ARTIFACTS,
                    *PHASE4_OPTIMIZER_ARTIFACTS,
                ]},
            )
            (artifact_root / "artifact_index.json").write_text(
                index.model_dump_json(indent=2),
                encoding="utf-8",
            )
            return BacktestRunnerResult(success=True, artifact_index=index)

    orchestrator.runner = FakeRunner()
    plugin_contract_path = _write_strategy_plugin_contract(tmp_path)
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        optimizer_sequence_enabled=True,
        strategy_plugin_id="strat1-plugin",
        strategy_plugin_contract_path=plugin_contract_path,
        shadow=True,
    ))

    assert result.optimizer_sequence_status == OptimizerSequenceStatus.SUCCEEDED.value
    assert result.adopted_candidate_id == "cand-phased"
    assert Path(result.optimizer_sequence_result_path).exists()
    assert result.blocking_reasons == []


def test_orchestrator_passes_direct_data_bundle_to_optimizer(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_monthly_inputs(tmp_path)
    data_bundle_path, data_bundle = _write_data_bundle(market_root)
    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
    )
    captured: dict[str, MonthlyRunManifest] = {}

    class FakeRunner:
        def run(self, manifest: MonthlyRunManifest, manifest_path: Path) -> BacktestRunnerResult:
            captured["manifest"] = manifest
            artifact_root = Path(manifest.artifact_root)
            _write_phase4_artifacts(
                artifact_root=artifact_root,
                manifest=manifest,
                repair_triggered=False,
                adopted_candidate_id="cand-phased",
                consume_search_guidance=True,
            )
            index = BacktestArtifactIndex(
                run_id=manifest.run_id,
                manifest_id=manifest.manifest_id,
                artifact_root=str(artifact_root),
                artifacts={name: str(artifact_root / name) for name in [
                    *REQUIRED_BACKTEST_ARTIFACTS,
                    *PHASE4_OPTIMIZER_ARTIFACTS,
                ]},
            )
            (artifact_root / "artifact_index.json").write_text(
                index.model_dump_json(indent=2),
                encoding="utf-8",
            )
            return BacktestRunnerResult(success=True, artifact_index=index)

    orchestrator.runner = FakeRunner()
    plugin_contract_path = _write_strategy_plugin_contract(tmp_path)

    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        optimizer_sequence_enabled=True,
        strategy_plugin_id="strat1-plugin",
        strategy_plugin_contract_path=plugin_contract_path,
        data_bundle_manifest_path=data_bundle_path,
        data_bundle_checksum=data_bundle.bundle_checksum,
        shadow=True,
    ))

    assert result.blocking_reasons == []
    manifest = captured["manifest"]
    assert manifest.market_data_manifest_path == str(data_bundle_path)
    assert manifest.data_bundle_manifest_path == str(data_bundle_path)
    assert manifest.data_bundle_checksum == data_bundle.bundle_checksum
    assert manifest.data_manifest_checksum == data_bundle.bundle_checksum
    assert manifest.in_sample_start == date(2026, 1, 1)
    assert manifest.in_sample_end == date(2026, 3, 31)
    assert not (Path(manifest.artifact_root) / "data_bundle_manifest.json").exists()


def test_orchestrator_blocks_direct_diagnostics_data_bundle(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_monthly_inputs(tmp_path)
    data_bundle_path, _ = _write_data_bundle(
        market_root,
        status=DataBundleStatus.DIAGNOSTICS_ONLY,
        diagnostics_only_reason="data repo commit SHA missing",
    )
    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
    )

    class RaisingRunner:
        def run(self, manifest: MonthlyRunManifest, manifest_path: Path) -> BacktestRunnerResult:
            raise AssertionError("runner must not start for diagnostics-only data bundle")

    orchestrator.runner = RaisingRunner()
    plugin_contract_path = _write_strategy_plugin_contract(tmp_path)

    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        optimizer_sequence_enabled=True,
        strategy_plugin_id="strat1-plugin",
        strategy_plugin_contract_path=plugin_contract_path,
        data_bundle_manifest_path=data_bundle_path,
        shadow=True,
    ))

    assert any(
        "data bundle is not authoritative: diagnostics_only" in reason
        for reason in result.blocking_reasons
    )
    manifest = json.loads(Path(result.run_manifest_path).read_text(encoding="utf-8"))
    assert manifest["market_data_manifest_path"] == str(data_bundle_path)
    assert manifest["data_bundle_manifest_path"] == str(data_bundle_path)


def test_orchestrator_blocks_direct_bundle_without_latest_month(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_monthly_inputs(tmp_path)
    data_bundle_path, _ = _write_data_bundle(
        market_root,
        end_ts=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )
    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
    )

    class RaisingRunner:
        def run(self, manifest: MonthlyRunManifest, manifest_path: Path) -> BacktestRunnerResult:
            raise AssertionError("runner must not start when the data bundle lacks latest-month coverage")

    orchestrator.runner = RaisingRunner()
    plugin_contract_path = _write_strategy_plugin_contract(tmp_path)

    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        optimizer_sequence_enabled=True,
        strategy_plugin_id="strat1-plugin",
        strategy_plugin_contract_path=plugin_contract_path,
        data_bundle_manifest_path=data_bundle_path,
        shadow=True,
    ))

    assert "data bundle does not cover latest-month selection window" in result.blocking_reasons
    assert result.gap_attribution.primary_category.value == "data_gap"


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


def _write_optimizer_run_manifest(
    *,
    artifact_root: Path,
    manifest: MonthlyRunManifest,
) -> None:
    run_manifest_path = artifact_root / "run_manifest.json"
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
    (artifact_root / "optimizer_run_manifest.json").write_text(json.dumps({
        "schema_version": "optimizer_approval_run_manifest_v1",
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id,
        "scope_id": scope_id,
        "scope_aliases": [
            item
            for item in (manifest.bot_id, manifest.strategy_id, manifest.strategy_plugin_id, scope_id)
            if item
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
        "artifact_root": str(artifact_root),
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_hash": _sha256_file(run_manifest_path),
        "data_bundle_checksum": manifest.data_bundle_checksum
        or manifest.data_manifest_checksum,
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


def _sha256_file(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_phase4_artifacts(
    *,
    artifact_root: Path,
    manifest: MonthlyRunManifest,
    repair_triggered: bool,
    adopted_candidate_id: str,
    adopted_source: str | None = None,
    round_adopted_gate: bool = True,
    structural_without_patches: bool = False,
    structural_with_patches: bool = False,
    consume_search_guidance: bool = False,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    workspace_root = artifact_root / "workspaces"
    workspace = CandidateWorkspaceManager(workspace_root).prepare(
        run_id=manifest.run_id,
        candidate_id=adopted_candidate_id,
        workspace_key=adopted_candidate_id,
    )
    (artifact_root / "candidate_workspace_manifest.json").write_text(
        workspace.model_dump_json(indent=2),
        encoding="utf-8",
    )
    fold_manifest = build_two_fold_manifest(
        run_id=manifest.run_id,
        run_month=manifest.run_month,
        in_sample_start=manifest.in_sample_start,
        in_sample_end=manifest.in_sample_end,
        selection_oos_start=manifest.selection_oos_start,
        selection_oos_end=manifest.selection_oos_end,
        evidence_paths=[str(artifact_root / "objective_breakdown.json")],
    )
    (artifact_root / "fold_manifest.json").write_text(
        fold_manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    plan_evidence = [str(artifact_root / "gap_attribution.json")]
    if consume_search_guidance and manifest.monthly_search_brief_path:
        plan_evidence.append(manifest.monthly_search_brief_path)
    (artifact_root / "llm_experiment_plan.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "score_components": ["expected_return", "calmar", "profit_factor"],
        "phase_order": ["signal_quality", "trade_management"],
        "candidate_families": _fixture_candidate_families(
            manifest,
            consume_search_guidance=consume_search_guidance,
        ),
        "gate_expectations": ["positive purged folds", "cost sensitivity"],
        "overfit_risks": ["sparse latest month"],
        "evidence_paths": plan_evidence,
        "source_weekly_signal_ids": (
            manifest.source_weekly_signal_ids
            if consume_search_guidance and manifest.source_weekly_signal_ids
            else ["weekly-1"]
        ),
    }))
    candidate_source = adopted_source or ("smoke_repair" if repair_triggered else "phased_auto")
    runner_contract = (
        "smoke_repair_runner_contract_v1"
        if candidate_source == "smoke_repair" else "phased_auto_runner_contract_v1"
    )
    patch_fingerprint, evaluated_fingerprint = _fixture_patch_fingerprints()
    live_repo_patch = artifact_root / "live_repo.patch"
    adapter_patch = artifact_root / "backtest_adapter.patch"
    if structural_with_patches:
        live_repo_patch.write_text("diff --git a/strategy.py b/strategy.py\n", encoding="utf-8")
        adapter_patch.write_text("diff --git a/adapter.py b/adapter.py\n", encoding="utf-8")
    structural_candidate = structural_without_patches or structural_with_patches
    candidate = {
        "candidate_id": adopted_candidate_id,
        "run_id": manifest.run_id,
        "manifest_id": manifest.manifest_id,
        "source": candidate_source,
        "family": "filter_repair",
        "title": "Adopt optimized backtest candidate",
        "description": "Repair-centered local follow-up wins immutable objective.",
        "decision": "experiment",
        "change_kind": "structural_change" if structural_candidate else "parameter_change",
        "objective_score": 1.22,
        "baseline_score": 1.0,
        "objective_delta": 0.22,
        "objective_deltas": {"latest_month_oos": 0.14, "calibration": 0.08},
        "parameter_patch": _fixture_config_patch(),
        "evaluated_parameter_patch": _fixture_config_patch(),
        "evaluated_parameters": _fixture_evaluated_parameters(),
        "parameter_patch_fingerprint": patch_fingerprint,
        "evaluated_patch_fingerprint": evaluated_fingerprint,
        "candidate_workspace_key": workspace.workspace_key,
        "candidate_workspace_path": workspace.workspace_path,
        "candidate_attempt_id": "attempt-1",
        "candidate_attempt_status": "succeeded",
        "retry_attempt": 0,
        "stall_timeout_seconds": 900,
        "round_id": "round_2",
        "prior_round_id": "round_1",
        "next_round_id": "round_2",
        "backtest_repo_commit_sha": manifest.backtest_repo_commit_sha or "fixture-backtest-sha",
        "live_trading_repo_commit_sha": manifest.trading_repo_commit_sha or "fixture-live-sha",
        "control_plane_commit_sha": manifest.control_plane_commit_sha or "fixture-control-sha",
        "fold_manifest_path": str(artifact_root / "fold_manifest.json"),
        "rounds_manifest_path": str(artifact_root / "rounds_manifest.json"),
        "end_of_round_diagnostics_path": str(artifact_root / "end_of_round_diagnostics.json"),
        "confirmatory_rerank_path": str(artifact_root / "confirmatory_rerank.json"),
        "decision_parity_report_path": str(artifact_root / "decision_parity_report.json"),
        "live_repo_patch_path": str(live_repo_patch) if structural_with_patches else "",
        "backtest_adapter_patch_path": str(adapter_patch) if structural_with_patches else "",
        "optimizer_stage": "round_adoption",
        "score_component_count": 3,
        "max_workers": 2,
        "source_weekly_signal_ids": ["weekly-1"],
        "param_changes": [{"param_name": "entry_filter", "current": True, "proposed": False}],
        "file_changes": (
            [{"file_path": "strategies/alpha.py", "kind": "modify", "summary": "Add signal split"}]
            if structural_candidate else []
        ),
        "acceptance_criteria": ["next completed month remains positive"],
        "replay_or_experiment_plan": "Run approval-gated shadow deployment after human review.",
        "rollback_plan": "Restore round_1 config.",
        "evidence_paths": [
            str(artifact_root / "confirmatory_rerank.json"),
            str(artifact_root / "end_of_round_diagnostics.json"),
        ],
        "deterministic_gate_inputs": {
            "runner_contract_version": runner_contract,
            "phase4_sequence_valid": True,
            "round_n_plus_1_adopted": round_adopted_gate,
            "confirmatory_follow_up_passed": True,
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
    (artifact_root / "selected_candidates.json").write_text(json.dumps([candidate]), encoding="utf-8")
    (artifact_root / "candidate_results.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    (artifact_root / "rejected_candidates.jsonl").write_text(
        json.dumps({"candidate_id": "weak-candidate", "reason": "failed cost sensitivity"}) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "confirmatory_rerank.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "primary_candidate_id": "cand-repair" if repair_triggered else adopted_candidate_id,
        "primary_source": "smoke_repair" if repair_triggered else "phased_auto",
        "repair_triggered": repair_triggered,
        "compared_candidate_ids": [
            "incumbent",
            "cand-repair" if repair_triggered else adopted_candidate_id,
            adopted_candidate_id,
        ],
        "variants": [{
            "candidate_id": adopted_candidate_id,
            "source_candidate_id": "cand-repair" if repair_triggered else adopted_candidate_id,
            "variant_type": "local_parameter_perturbation",
            "objective_score": 1.22,
            "baseline_score": 1.0,
            "in_sample_delta": 0.08,
            "selection_oos_delta": 0.14,
            "fold_support_passed": True,
            "deterministic_replay_passed": True,
            "evidence_paths": [str(artifact_root / "fold_validation.json")],
        }],
        "adopted_candidate_id": adopted_candidate_id,
        "adopted_source": candidate_source,
        "selection_rule": "best selection-OOS without material in-sample deterioration",
        "evidence_paths": [str(artifact_root / "fold_validation.json")],
    }), encoding="utf-8")
    (artifact_root / "rounds_manifest.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "current_round_id": "round_1",
        "next_round_id": "round_2",
        "adopted_candidate_id": adopted_candidate_id,
        "records": [{
            "round_id": "round_2",
            "prior_round_id": "round_1",
            "candidate_id": adopted_candidate_id,
            "source": candidate_source,
            "config_version": "round_2_cfg",
            "fold_manifest_path": str(artifact_root / "fold_manifest.json"),
            "diagnostics_path": str(artifact_root / "end_of_round_diagnostics.json"),
            "confirmatory_rerank_path": str(artifact_root / "confirmatory_rerank.json"),
            "decision_parity_report_path": str(artifact_root / "decision_parity_report.json"),
            "approval_state": "not_requested",
            "live_deployment_status": "optimized_backtest_recommendation",
            "evidence_paths": [str(artifact_root / "end_of_round_diagnostics.json")],
        }],
    }), encoding="utf-8")

    attempt = {
        "attempt_id": "attempt-1",
        "run_id": manifest.run_id,
        "candidate_id": adopted_candidate_id,
        "workspace_key": workspace.workspace_key,
        "workspace_path": workspace.workspace_path,
        "state": "succeeded",
        "stage": "confirmatory_follow_up",
        "attempt_number": 1,
        "retry_attempt": 0,
        "stall_timeout_seconds": 900,
        "manifest_id": manifest.manifest_id,
        "artifact_paths": [str(artifact_root / "confirmatory_rerank.json")],
    }
    (artifact_root / "candidate_attempts.jsonl").write_text(json.dumps(attempt) + "\n", encoding="utf-8")
    (artifact_root / "runner_observability.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "candidate_id": adopted_candidate_id,
        "attempt_id": "attempt-1",
        "workspace": workspace.workspace_path,
        "attempt_state": "succeeded",
        "phase": "confirmatory_follow_up",
        "timeout_status": "ok",
        "parity_status": "pass",
    }), encoding="utf-8")
    _write_p6_p7_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        candidate_id=adopted_candidate_id,
        repair_triggered=repair_triggered,
        no_adoption_reason="",
        candidate_source=candidate_source,
    )

    (artifact_root / "decision_parity_report.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "candidate_id": adopted_candidate_id,
        "strategy_plugin_id": manifest.strategy_plugin_id,
        "live_repo_commit_sha": manifest.trading_repo_commit_sha or "fixture-live-sha",
        "backtest_adapter_commit_sha": manifest.backtest_repo_commit_sha or "fixture-backtest-sha",
        "status": "pass",
        "checks": [
            {
                "dimension": dimension,
                "status": "pass",
                "match_rate": 1.0,
                "mismatch_count": 0,
                "evidence_paths": [str(artifact_root / "end_of_round_diagnostics.json")],
            }
            for dimension in [
                "signals",
                "filters",
                "entries",
                "exits",
                "stops",
                "sizing",
                "risk_caps",
                "order_intent",
            ]
        ],
        "evidence_paths": [str(artifact_root / "end_of_round_diagnostics.json")],
    }), encoding="utf-8")

    for name in [
        "coverage_manifest.json",
        "incumbent_validation.json",
        "gap_attribution.json",
        "mode_decision.json",
        "replay_parity_report.json",
        "objective_breakdown.json",
        "monthly_report.md",
        "stdout.log",
        "stderr.log",
        "exit_status.json",
        "leakage_report.json",
        "cost_sensitivity.json",
        "fold_validation.json",
        "outlier_sensitivity.json",
        "portfolio_synergy.json",
        "end_of_round_diagnostics.json",
    ]:
        path = artifact_root / name
        if path.exists():
            continue
        if name == "mode_decision.json":
            path.write_text(json.dumps({
                "status": "experiment",
                "mode": "smoke_repair" if repair_triggered else "phased_auto",
            }), encoding="utf-8")
        elif name == "coverage_manifest.json":
            path.write_text(json.dumps({
                "status": "pass",
                "run_id": manifest.run_id,
                "data_bundle_checksum": manifest.data_bundle_checksum,
            }), encoding="utf-8")
        elif name == "replay_parity_report.json":
            path.write_text(json.dumps({
                "bot_id": manifest.bot_id,
                "strategy_id": manifest.strategy_id,
                "run_month": manifest.run_month,
                "trade_count_live": 1,
                "trade_count_replay": 1,
                "entry_match_rate": 1.0,
                "exit_match_rate": 1.0,
                "side_quantity_match_rate": 1.0,
                "status": "pass",
            }), encoding="utf-8")
        elif name.endswith(".md") or name.endswith(".log"):
            path.write_text(name, encoding="utf-8")
        else:
            path.write_text(json.dumps({"status": "pass", "run_id": manifest.run_id}), encoding="utf-8")
    if repair_triggered:
        (artifact_root / "repair_ablation_matrix.jsonl").write_text(
            json.dumps({"mutation_id": "m1", "decision": "keep"}) + "\n",
            encoding="utf-8",
        )


def _fixture_candidate_families(
    manifest: MonthlyRunManifest,
    *,
    consume_search_guidance: bool,
) -> list[dict[str, str]]:
    families = [{"family": "filter_repair", "phase": "signal_quality"}]
    if not consume_search_guidance:
        return families
    requirements = manifest.monthly_search_guidance.get("plan_requirements") or {}
    if not isinstance(requirements, dict):
        return families
    for family in requirements.get("candidate_families") or []:
        family_name = str(family or "").strip()
        if family_name and not any(row["family"] == family_name for row in families):
            families.append({"family": family_name, "phase": "signal_quality"})
    return families


def _write_no_adoption_artifacts(*, artifact_root: Path, manifest: MonthlyRunManifest) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    fold_manifest = build_two_fold_manifest(
        run_id=manifest.run_id,
        run_month=manifest.run_month,
        in_sample_start=manifest.in_sample_start,
        in_sample_end=manifest.in_sample_end,
        selection_oos_start=manifest.selection_oos_start,
        selection_oos_end=manifest.selection_oos_end,
        evidence_paths=[str(artifact_root / "objective_breakdown.json")],
    )
    (artifact_root / "fold_manifest.json").write_text(
        fold_manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (artifact_root / "llm_experiment_plan.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "score_components": ["expected_return", "calmar", "profit_factor"],
        "phase_order": ["signal_quality"],
        "candidate_families": [{"family": "filter_repair", "phase": "signal_quality"}],
        "gate_expectations": ["positive purged folds"],
        "overfit_risks": ["insufficient sample"],
        "evidence_paths": [str(artifact_root / "gap_attribution.json")],
    }), encoding="utf-8")
    (artifact_root / "confirmatory_rerank.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "primary_candidate_id": "",
        "primary_source": "phased_auto",
        "repair_triggered": False,
        "compared_candidate_ids": [],
        "variants": [],
        "no_adoption_reason": "insufficient mature replay plugin sample size",
        "selection_rule": "fail closed when no replay-backed candidate is mature",
        "evidence_paths": [str(artifact_root / "fold_validation.json")],
    }), encoding="utf-8")
    (artifact_root / "rounds_manifest.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "current_round_id": "round_1",
        "next_round_id": "",
        "no_adoption_reason": "insufficient mature replay plugin sample size",
        "records": [],
    }), encoding="utf-8")
    (artifact_root / "selected_candidates.json").write_text("[]", encoding="utf-8")
    (artifact_root / "candidate_results.jsonl").write_text("", encoding="utf-8")
    (artifact_root / "rejected_candidates.jsonl").write_text(
        json.dumps({"candidate_id": "candidate-space", "reason": "insufficient mature replay plugin sample size"}) + "\n",
        encoding="utf-8",
    )
    (artifact_root / "runner_observability.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "phase": "eligibility",
        "attempt_state": "not_started",
        "reason": "insufficient mature replay plugin sample size",
    }), encoding="utf-8")
    _write_p6_p7_artifacts(
        artifact_root=artifact_root,
        manifest=manifest,
        candidate_id="candidate-space",
        repair_triggered=False,
        no_adoption_reason="insufficient mature replay plugin sample size",
    )
    for name in [
        "coverage_manifest.json",
        "incumbent_validation.json",
        "gap_attribution.json",
        "mode_decision.json",
        "replay_parity_report.json",
        "objective_breakdown.json",
        "monthly_report.md",
        "stdout.log",
        "stderr.log",
        "exit_status.json",
        "fold_validation.json",
        "end_of_round_diagnostics.json",
    ]:
        path = artifact_root / name
        if name.endswith(".md") or name.endswith(".log"):
            path.write_text(name, encoding="utf-8")
        elif name == "coverage_manifest.json":
            path.write_text(json.dumps({
                "status": "pass",
                "run_id": manifest.run_id,
                "data_bundle_checksum": manifest.data_bundle_checksum,
            }), encoding="utf-8")
        else:
            path.write_text(json.dumps({"status": "pass", "run_id": manifest.run_id}), encoding="utf-8")


def _write_p6_p7_artifacts(
    *,
    artifact_root: Path,
    manifest: MonthlyRunManifest,
    candidate_id: str,
    repair_triggered: bool,
    no_adoption_reason: str,
    candidate_source: str = "phased_auto",
) -> None:
    _write_optimizer_run_manifest(artifact_root=artifact_root, manifest=manifest)
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
    (artifact_root / "fold_candidate_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in fold_rows) + "\n",
        encoding="utf-8",
    )
    adopted = "" if no_adoption_reason else candidate_id
    (artifact_root / "fold_score_matrix.json").write_text(json.dumps({
        "schema_version": "fold_score_matrix_v1",
        "run_id": manifest.run_id,
        "selection_oos_excluded_from_first_pass": True,
        "scoring_windows": [
            {"fold_id": "fold_1", "purged": True, "embargo_days": 5},
            {"fold_id": "fold_2", "purged": True, "embargo_days": 5},
        ],
        "candidate_count": 1,
        "selected_candidate_ids": [candidate_id] if adopted else [],
        "candidates": [{"candidate_id": candidate_id, "fold_support_passed": True}],
    }), encoding="utf-8")
    (artifact_root / "selection_oos_evaluation.json").write_text(json.dumps({
        "schema_version": "selection_oos_evaluation_v1",
        "run_id": manifest.run_id,
        "status": "pass" if adopted else "blocked",
        "selection_oos_used_after_fold_ranking": True,
        "selection_oos_used_in_first_pass": False,
        "primary_candidate_id": adopted,
        "incumbent_selection_oos": {"objective_score": 1.0, "trade_count": 4, "max_drawdown": 0.05},
        "candidate_selection_oos": (
            {"candidate_id": candidate_id, "objective_score": 1.14, "trade_count": 4, "max_drawdown": 0.04}
            if adopted else {}
        ),
    }), encoding="utf-8")
    (artifact_root / "selection_oos_repair_trigger.json").write_text(json.dumps({
        "schema_version": "selection_oos_repair_trigger_v1",
        "run_id": manifest.run_id,
        "triggered": repair_triggered,
        "status": "triggered" if repair_triggered else "not_triggered",
        "thresholds": {
            "objective_drop_threshold": -0.05,
            "drawdown_increase_threshold": 0.05,
            "trade_count_collapse_ratio": 0.5,
        },
        "expected_is_fold_score_band": {"mean_objective_score": 1.1},
        "measured_degradation": {"objective_delta_vs_fold_mean": -0.08 if repair_triggered else 0.02},
    }), encoding="utf-8")
    (artifact_root / "repair_failure_attribution.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "status": "complete",
        "repair_triggered": repair_triggered,
        "reason": no_adoption_reason,
    }), encoding="utf-8")
    (artifact_root / "accepted_mutation_chain.json").write_text(json.dumps({
        "run_id": manifest.run_id,
        "accepted_mutations": (
            [{"mutation_id": "accepted-1", "strategy_scope": manifest.strategy_id}]
            if repair_triggered else []
        ),
    }), encoding="utf-8")
    repair_rows = [
        {
            "run_id": manifest.run_id,
            "candidate_id": candidate_id,
            "source": candidate_source,
            "accepted_mutation_count": 1,
        }
    ] if repair_triggered else []
    (artifact_root / "repair_candidate_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in repair_rows),
        encoding="utf-8",
    )
    (artifact_root / "repair_checkpoint.json").write_text(json.dumps({
        "schema_version": "repair_checkpoint_v1",
        "run_id": manifest.run_id,
        "repair_triggered": repair_triggered,
        "candidate_ids": [candidate_id] if repair_triggered else [],
        "deterministic_resume_key": "fixture-resume-key",
    }), encoding="utf-8")
    config_patch_path = artifact_root / "round_n_plus_1" / "config_patch.json"
    if adopted:
        config_patch_path.parent.mkdir(parents=True, exist_ok=True)
        config_patch_path.write_text(json.dumps(_fixture_config_patch()), encoding="utf-8")
    (artifact_root / "round_n_plus_1_recommendation.json").write_text(json.dumps({
        "schema_version": "round_n_plus_1_recommendation_v1",
        "run_id": manifest.run_id,
        "status": "no_adoption" if no_adoption_reason else "optimized_backtest_recommendation",
        "adopted_candidate_id": adopted,
        "no_adoption_reason": no_adoption_reason,
        "next_round_id": manifest.next_round_id if adopted else "",
        "live_deployment_status": "not_requested"
        if no_adoption_reason else "optimized_backtest_recommendation",
        "config_patch_path": str(config_patch_path) if adopted else "",
        "parameter_patch_fingerprint": patch_fingerprint if adopted else "",
        "evaluated_patch_fingerprint": evaluated_fingerprint if adopted else "",
        "evaluated_parameters": _fixture_evaluated_parameters() if adopted else {},
    }), encoding="utf-8")


def _write_monthly_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    curated = tmp_path / "curated"
    findings = tmp_path / "memory" / "findings"
    bot_dir = curated / "2026-04-02" / "bot1"
    bot_dir.mkdir(parents=True)
    trade = TradeEvent.model_validate({
        "trade_id": "t1",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "pair": "AAPL",
        "side": "LONG",
        "entry_time": "2026-04-02T10:00:00Z",
        "exit_time": "2026-04-02T11:00:00Z",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "position_size": 1.0,
        "pnl": 1.0,
        "pnl_pct": 1.0,
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "parameter_set_id": "ps1",
    })
    (bot_dir / "trades.jsonl").write_text(trade.model_dump_json() + "\n", encoding="utf-8")

    market_root = tmp_path / "market_data"
    market_root.mkdir(parents=True)
    manifest_path = market_root / "manifests" / "bot1" / "strat1" / "2026-04.coverage_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(MarketDataManifest(
        source="fixture",
        market="equity",
        symbol="AAPL",
        timeframe="1m",
        start_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 4, 30, tzinfo=timezone.utc),
        expected_bars=100,
        actual_bars=100,
        usable_for_authoritative_validation=True,
        checksum="market-sha",
        session_calendar="XNYS",
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        adjustment_policy="split_adjusted",
    ).model_dump_json(indent=2), encoding="utf-8")
    _git_commit_all(market_root)
    repo = tmp_path / "backtests"
    repo.mkdir()
    return curated, findings, market_root, repo


def _write_strategy_plugin_contract(tmp_path: Path) -> Path:
    path = tmp_path / "strategy_plugin_contract.json"
    path.write_text(StrategyPluginContract(
        plugin_id="strat1-plugin",
        live_repo_path=str(tmp_path / "live"),
        live_repo_commit_sha="fixture-live-sha",
        backtest_adapter_path="adapters/strat1.py",
        backtest_adapter_commit_sha="fixture-backtest-sha",
        config_schema_version="config_v1",
        decision_api_version="decision_api_v1",
        required_telemetry_schemas=["trade_event_v1"],
        supported_symbols=["AAPL"],
        supported_timeframes=["1m"],
        parity_fixture_set=[str(tmp_path / "parity_fixture.json")],
        maturity=StrategyPluginMaturity.APPROVAL_READY,
    ).model_dump_json(indent=2), encoding="utf-8")
    return path


def _write_data_bundle(
    market_root: Path,
    *,
    status: DataBundleStatus = DataBundleStatus.AUTHORITATIVE,
    diagnostics_only_reason: str = "",
    start_ts: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc),
    end_ts: datetime = datetime(2026, 4, 30, tzinfo=timezone.utc),
) -> tuple[Path, DataBundleManifest]:
    bundle = DataBundleManifest(
        data_repo_path=str(market_root),
        data_repo_commit_sha=(
            "fixture-data-sha"
            if status == DataBundleStatus.AUTHORITATIVE
            else ""
        ),
        slice_manifests=[
            DataBundleSlice(
                manifest_path=str(
                    market_root / "manifests" / "bot1" / "strat1" / "2026-04.coverage_manifest.json"
                ),
                manifest_id="slice-1",
                source="fixture",
                market="equity",
                symbol="AAPL",
                timeframe="1m",
                start_ts=start_ts,
                end_ts=end_ts,
                checksum="market-sha",
                calendar="XNYS",
                authoritative=status == DataBundleStatus.AUTHORITATIVE,
            )
        ],
        calendars=["XNYS"],
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        adjustment_policy="split_adjusted",
        status=status,
        diagnostics_only_reason=diagnostics_only_reason,
    )
    path = market_root / "bundles" / "bot1-strat1-2026-04.data_bundle_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return path, bundle


def _git_commit_all(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture data"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
