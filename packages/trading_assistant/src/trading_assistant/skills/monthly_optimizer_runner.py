"""Control-plane runner for the Phase 4 monthly optimizer sequence.

The sibling backtest repository owns replay/search execution.  This module owns
the durable orchestration contract around that external runner: workspace and
attempt state, manifest freezing, artifact validation, and round_N+1 adoption
checks before the normal monthly candidate pipeline can trust the output.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_assistant.schemas.backtest_artifacts import (
    BacktestArtifactIndex,
    PHASE4_OOS_REPAIR_ARTIFACTS,
    PHASE4_OPTIMIZER_ARTIFACTS,
)
from trading_assistant.schemas.data_bundle_manifest import DataBundleManifest, DataBundleStatus
from trading_assistant.schemas.decision_parity import DecisionParityReport
from trading_assistant.schemas.monthly_candidates import MonthlyCandidateSource, MonthlyImprovementCandidate
from trading_assistant.schemas.monthly_optimizer import (
    CandidateAttemptRecord,
    CandidateAttemptState,
    CandidateWorkspaceManifest,
    ConfirmatoryRerank,
    FoldManifest,
    FoldSpec,
    MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION,
    MonthlyOptimizerSequenceResult,
    OptimizerExperimentPlan,
    OptimizerSequenceStatus,
    OptimizerStage,
    RoundsManifest,
    sanitize_workspace_key,
)
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest, MonthlyRunMode
from trading_assistant.schemas.strategy_plugin_contract import StrategyPluginContract
from trading_assistant.skills.backtest_runner_client import BacktestRunnerClient
from trading_assistant.skills.monthly_artifact_contract import MonthlyArtifactContract
from trading_assistant.skills.monthly_deployment_metadata import deployment_metadata_errors


ACTIVE_ATTEMPT_STATES = {
    CandidateAttemptState.CLAIMED,
    CandidateAttemptState.RUNNING,
    CandidateAttemptState.RETRY_QUEUED,
}
TERMINAL_ATTEMPT_STATES = {
    CandidateAttemptState.RELEASED,
    CandidateAttemptState.SUCCEEDED,
    CandidateAttemptState.FAILED,
    CandidateAttemptState.TIMED_OUT,
    CandidateAttemptState.STALLED,
    CandidateAttemptState.CANCELED_BY_RECONCILIATION,
}

CORE_PHASE4_SEQUENCE_ARTIFACTS = [
    "optimizer_run_manifest.json",
    "fold_manifest.json",
    "fold_candidate_results.jsonl",
    "fold_score_matrix.json",
    "selection_oos_evaluation.json",
    "selection_oos_repair_trigger.json",
    "repair_failure_attribution.json",
    "accepted_mutation_chain.json",
    "repair_candidate_results.jsonl",
    "repair_checkpoint.json",
    "rounds_manifest.json",
    "round_n_plus_1_recommendation.json",
    "end_of_round_diagnostics.json",
    "llm_experiment_plan.json",
    "fold_validation.json",
    "confirmatory_rerank.json",
    "runner_observability.json",
]

ADOPTION_GATE_ARTIFACTS = [
    "leakage_report.json",
    "cost_sensitivity.json",
    "outlier_sensitivity.json",
    "portfolio_synergy.json",
]

OPTIMIZER_BRIDGE_IDS_BY_SCOPE = {
    "crypto_trader_portfolio": (
        "crypto_trend_v1",
        "crypto_momentum_v1",
        "crypto_breakout_v1",
    ),
}

OPTIMIZER_SCOPE_BY_PLUGIN_ID = {
    "crypto-trend-v1": "crypto_trader_portfolio",
    "crypto-momentum-v1": "crypto_trader_portfolio",
    "crypto-breakout-v1": "crypto_trader_portfolio",
}


@dataclass(frozen=True)
class CandidateCommandResult:
    attempt_id: str
    state: CandidateAttemptState
    return_code: int
    cwd: str
    stdout_path: str = ""
    stderr_path: str = ""
    error: str = ""


class CandidateWorkspaceManager:
    """Creates deterministic path-contained candidate workspaces."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        *,
        run_id: str,
        candidate_id: str,
        workspace_key: str = "",
        structural: bool = False,
    ) -> CandidateWorkspaceManifest:
        safe_key = sanitize_workspace_key(workspace_key or f"{run_id}-{candidate_id}")
        workspace_path = (self.workspace_root / safe_key).resolve()
        workspace_path.relative_to(self.workspace_root.resolve())
        workspace_path.mkdir(parents=True, exist_ok=True)
        manifest_path = workspace_path / "candidate_workspace_manifest.json"
        manifest = CandidateWorkspaceManifest(
            run_id=run_id,
            candidate_id=candidate_id,
            workspace_key=safe_key,
            workspace_root=str(self.workspace_root),
            workspace_path=str(workspace_path),
            manifest_path=str(manifest_path),
            structural=structural,
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest


class CandidateAttemptStore:
    """Append-only state log for Symphony-style candidate attempts."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CandidateAttemptRecord) -> CandidateAttemptRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record

    def load(self) -> list[CandidateAttemptRecord]:
        if not self.path.exists():
            return []
        records: list[CandidateAttemptRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(CandidateAttemptRecord.model_validate(json.loads(line)))
        return records

    def latest_by_attempt(self) -> dict[str, CandidateAttemptRecord]:
        latest: dict[str, CandidateAttemptRecord] = {}
        for record in self.load():
            latest[record.attempt_id] = record
        return latest

    def claim(
        self,
        *,
        run_id: str,
        candidate_id: str,
        workspace: CandidateWorkspaceManifest,
        manifest_id: str,
        stage: OptimizerStage = OptimizerStage.PHASED_AUTO,
        stall_timeout_seconds: int = 0,
        backtest_repo_commit_sha: str = "",
        trading_repo_commit_sha: str = "",
    ) -> CandidateAttemptRecord:
        records = self.load()
        for record in _latest_attempts(records).values():
            if (
                record.run_id == run_id
                and record.candidate_id == candidate_id
                and record.state in ACTIVE_ATTEMPT_STATES
            ):
                raise ValueError(
                    f"candidate {candidate_id} already has active attempt {record.attempt_id}"
                )
        attempt_number = 1 + max(
            (
                record.attempt_number
                for record in records
                if record.run_id == run_id and record.candidate_id == candidate_id
            ),
            default=0,
        )
        attempt_id = _attempt_id(run_id, candidate_id, attempt_number)
        return self.append(CandidateAttemptRecord(
            attempt_id=attempt_id,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace_key=workspace.workspace_key,
            workspace_path=workspace.workspace_path,
            state=CandidateAttemptState.CLAIMED,
            stage=stage,
            attempt_number=attempt_number,
            stall_timeout_seconds=stall_timeout_seconds,
            manifest_id=manifest_id,
            backtest_repo_commit_sha=backtest_repo_commit_sha,
            trading_repo_commit_sha=trading_repo_commit_sha,
        ))

    def transition(
        self,
        attempt_id: str,
        state: CandidateAttemptState,
        *,
        reason: str = "",
        retry_reason: str = "",
        subprocess_pid: int | None = None,
        artifact_paths: list[str] | None = None,
        parity_status: str = "",
    ) -> CandidateAttemptRecord:
        latest = self.latest_by_attempt().get(attempt_id)
        if latest is None:
            raise ValueError(f"unknown attempt_id: {attempt_id}")
        retry_attempt = latest.retry_attempt
        if state == CandidateAttemptState.RETRY_QUEUED:
            retry_attempt += 1
        return self.append(latest.model_copy(update={
            "state": state,
            "reason": reason,
            "retry_reason": retry_reason or latest.retry_reason,
            "retry_attempt": retry_attempt,
            "subprocess_pid": subprocess_pid if subprocess_pid is not None else latest.subprocess_pid,
            "artifact_paths": artifact_paths if artifact_paths is not None else latest.artifact_paths,
            "parity_status": parity_status or latest.parity_status,
            "updated_at": datetime.now(timezone.utc),
        }))

    def mark_stalled(self, *, now: datetime | None = None) -> list[CandidateAttemptRecord]:
        current_time = now or datetime.now(timezone.utc)
        stalled: list[CandidateAttemptRecord] = []
        for record in self.latest_by_attempt().values():
            if record.state != CandidateAttemptState.RUNNING:
                continue
            if record.stall_timeout_seconds <= 0:
                continue
            age = current_time - record.updated_at
            if age.total_seconds() >= record.stall_timeout_seconds:
                stalled.append(self.transition(
                    record.attempt_id,
                    CandidateAttemptState.STALLED,
                    reason="attempt exceeded stall_timeout_seconds",
                ))
        return stalled

    def reconcile(
        self,
        *,
        manifest_id: str,
        backtest_repo_commit_sha: str = "",
        trading_repo_commit_sha: str = "",
        eligible_candidate_ids: set[str] | None = None,
    ) -> list[CandidateAttemptRecord]:
        canceled: list[CandidateAttemptRecord] = []
        for record in self.latest_by_attempt().values():
            if record.state not in ACTIVE_ATTEMPT_STATES:
                continue
            reason = ""
            if record.manifest_id and record.manifest_id != manifest_id:
                reason = "run manifest changed"
            elif (
                backtest_repo_commit_sha
                and record.backtest_repo_commit_sha
                and record.backtest_repo_commit_sha != backtest_repo_commit_sha
            ):
                reason = "backtest repo SHA drifted"
            elif (
                trading_repo_commit_sha
                and record.trading_repo_commit_sha
                and record.trading_repo_commit_sha != trading_repo_commit_sha
            ):
                reason = "trading repo SHA drifted"
            elif eligible_candidate_ids is not None and record.candidate_id not in eligible_candidate_ids:
                reason = "candidate is no longer eligible"
            if reason:
                canceled.append(self.transition(
                    record.attempt_id,
                    CandidateAttemptState.CANCELED_BY_RECONCILIATION,
                    reason=reason,
                ))
        return canceled

    @staticmethod
    def retry_backoff_seconds(retry_attempt: int, *, base_seconds: int = 60, max_seconds: int = 900) -> int:
        attempt = max(0, retry_attempt)
        return min(max_seconds, int(base_seconds * math.pow(2, attempt)))


class CandidateAttemptExecutor:
    """Runs candidate-local work with cwd pinned to the candidate workspace."""

    def __init__(self, store: CandidateAttemptStore) -> None:
        self.store = store

    def run(
        self,
        attempt: CandidateAttemptRecord,
        command: list[str],
        *,
        timeout_seconds: int | None = None,
    ) -> CandidateCommandResult:
        if not command:
            raise ValueError("candidate attempt command cannot be empty")
        workspace = Path(attempt.workspace_path).resolve()
        if not workspace.exists() or not workspace.is_dir():
            failed = self.store.transition(
                attempt.attempt_id,
                CandidateAttemptState.FAILED,
                reason="candidate workspace is missing",
            )
            return CandidateCommandResult(
                attempt_id=attempt.attempt_id,
                state=failed.state,
                return_code=-1,
                cwd=str(workspace),
                error="candidate workspace is missing",
            )

        running = self.store.transition(attempt.attempt_id, CandidateAttemptState.RUNNING)
        stdout_path = workspace / f"{sanitize_workspace_key(attempt.attempt_id)}.stdout.log"
        stderr_path = workspace / f"{sanitize_workspace_key(attempt.attempt_id)}.stderr.log"
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds or attempt.stall_timeout_seconds or None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            timed_out = self.store.transition(
                running.attempt_id,
                CandidateAttemptState.TIMED_OUT,
                reason="candidate command timed out",
                artifact_paths=[str(stdout_path), str(stderr_path)],
            )
            return CandidateCommandResult(
                attempt_id=attempt.attempt_id,
                state=timed_out.state,
                return_code=-1,
                cwd=str(workspace),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error="candidate command timed out",
            )

        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        state = (
            CandidateAttemptState.SUCCEEDED
            if completed.returncode == 0
            else CandidateAttemptState.FAILED
        )
        final = self.store.transition(
            running.attempt_id,
            state,
            reason="" if completed.returncode == 0 else f"exit code {completed.returncode}",
            artifact_paths=[str(stdout_path), str(stderr_path)],
        )
        return CandidateCommandResult(
            attempt_id=attempt.attempt_id,
            state=final.state,
            return_code=completed.returncode,
            cwd=str(workspace),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


class MonthlyOptimizerRunner:
    """Runs and validates the external Phase 4 optimizer contract."""

    def __init__(self, runner: BacktestRunnerClient | None = None) -> None:
        self.runner = runner or BacktestRunnerClient(timeout_seconds=6 * 3600)

    def prepare_manifest(self, manifest: MonthlyRunManifest) -> MonthlyRunManifest:
        updates: dict[str, Any] = {
            "mode": MonthlyRunMode.PHASED_AUTO,
            "workflow_contract_version": (
                manifest.workflow_contract_version
                if manifest.workflow_contract_version != "monthly_incumbent_validation_v1"
                else MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION
            ),
            "max_workers": manifest.max_workers or 2,
            "score_component_cap": min(manifest.score_component_cap or 7, 7),
            "expected_outputs": _dedupe([
                *manifest.expected_outputs,
                *CORE_PHASE4_SEQUENCE_ARTIFACTS,
            ]),
            "output_artifact_names": _dedupe([
                *manifest.output_artifact_names,
                *PHASE4_OPTIMIZER_ARTIFACTS,
                *PHASE4_OOS_REPAIR_ARTIFACTS,
            ]),
        }
        if manifest.selection_oos_start is None:
            updates["selection_oos_start"] = manifest.latest_month_start
            updates["selection_oos_end"] = manifest.latest_month_end
        if manifest.in_sample_start is None and manifest.calibration_start is not None:
            updates["in_sample_start"] = manifest.calibration_start
            updates["in_sample_end"] = manifest.calibration_end
        artifact_root = Path(manifest.artifact_root)
        if not manifest.fold_manifest_path:
            updates["fold_manifest_path"] = str(artifact_root / "fold_manifest.json")
        if not manifest.rounds_manifest_path:
            updates["rounds_manifest_path"] = str(artifact_root / "rounds_manifest.json")
        if not manifest.end_of_round_diagnostics_path:
            updates["end_of_round_diagnostics_path"] = str(
                artifact_root / "end_of_round_diagnostics.json"
            )
        if not manifest.candidate_workspace_root:
            updates["candidate_workspace_root"] = str(artifact_root / "workspaces")
        if not manifest.checkpoint_path:
            updates["checkpoint_path"] = str(artifact_root / "optimizer_checkpoint.json")
        if not manifest.cache_path:
            updates["cache_path"] = str(artifact_root / "optimizer_cache")
        return manifest.model_copy(update=updates)

    def run(
        self,
        manifest: MonthlyRunManifest,
        manifest_path: Path,
        *,
        write_manifest: bool = True,
    ) -> MonthlyOptimizerSequenceResult:
        frozen = self.prepare_manifest(manifest)
        manifest_path = Path(manifest_path)
        if write_manifest:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(frozen.model_dump_json(indent=2), encoding="utf-8")
        runner_result = self.runner.run(frozen, manifest_path)
        artifact_index = runner_result.artifact_index
        if not runner_result.success or artifact_index is None:
            return MonthlyOptimizerSequenceResult(
                run_id=frozen.run_id,
                status=OptimizerSequenceStatus.FAILED,
                blocking_reasons=[runner_result.error or "external optimizer runner failed"],
                artifact_index_path=str(Path(frozen.artifact_root) / "artifact_index.json"),
                evidence_paths=[str(manifest_path)],
            )
        return self.validate_artifacts(frozen, artifact_index, manifest_path=manifest_path)

    def validate_artifacts(
        self,
        manifest: MonthlyRunManifest,
        artifact_index: BacktestArtifactIndex,
        *,
        manifest_path: Path | None = None,
    ) -> MonthlyOptimizerSequenceResult:
        errors: list[str] = []
        evidence_paths: list[str] = []
        artifact_contract = MonthlyArtifactContract.from_index(
            artifact_index,
            manifest=manifest,
        )
        artifact_index_path = Path(artifact_index.artifact_root) / "artifact_index.json"
        if artifact_index_path.exists():
            evidence_paths.append(str(artifact_index_path))
        if manifest_path is not None:
            evidence_paths.append(str(manifest_path))

        missing = artifact_contract.missing_named_artifacts(CORE_PHASE4_SEQUENCE_ARTIFACTS)
        if missing:
            errors.append(f"missing core phase4 optimizer artifacts: {', '.join(missing)}")

        fold_manifest = artifact_contract.load_model("fold_manifest.json", FoldManifest, errors)
        experiment_plan = artifact_contract.load_model(
            "llm_experiment_plan.json",
            OptimizerExperimentPlan,
            errors,
        )
        confirmatory = artifact_contract.load_model(
            "confirmatory_rerank.json",
            ConfirmatoryRerank,
            errors,
        )
        rounds_manifest = artifact_contract.load_model("rounds_manifest.json", RoundsManifest, errors)
        optimizer_run_manifest = artifact_contract.load_json_object(
            "optimizer_run_manifest.json",
            errors,
        )
        fold_candidate_results = artifact_contract.load_jsonl(
            "fold_candidate_results.jsonl",
            errors,
        )
        fold_score_matrix = artifact_contract.load_json_object(
            "fold_score_matrix.json",
            errors,
        )
        selection_oos_evaluation = artifact_contract.load_json_object(
            "selection_oos_evaluation.json",
            errors,
        )
        selection_oos_repair_trigger = artifact_contract.load_json_object(
            "selection_oos_repair_trigger.json",
            errors,
        )
        repair_failure_attribution = artifact_contract.load_json_object(
            "repair_failure_attribution.json",
            errors,
        )
        accepted_mutation_chain = artifact_contract.load_json_object(
            "accepted_mutation_chain.json",
            errors,
        )
        repair_candidate_results = artifact_contract.load_jsonl(
            "repair_candidate_results.jsonl",
            errors,
        )
        repair_checkpoint = artifact_contract.load_json_object(
            "repair_checkpoint.json",
            errors,
        )
        round_n_plus_1_recommendation = artifact_contract.load_json_object(
            "round_n_plus_1_recommendation.json",
            errors,
        )
        selected = [
            MonthlyImprovementCandidate.from_raw(item)
            for item in artifact_contract.load_candidate_rows("selected_candidates.json", errors)
        ]
        rejected = artifact_contract.load_jsonl("rejected_candidates.jsonl", errors)
        attempts = _load_attempts_from_rows(
            artifact_contract.load_jsonl("candidate_attempts.jsonl", errors),
            errors,
        )

        _validate_manifest_alignment(
            manifest=manifest,
            fold_manifest=fold_manifest,
            experiment_plan=experiment_plan,
            confirmatory=confirmatory,
            rounds_manifest=rounds_manifest,
            errors=errors,
        )
        _validate_data_bundle_alignment(manifest, artifact_index, errors)
        _validate_optimizer_run_manifest(
            optimizer_run_manifest,
            manifest=manifest,
            artifact_index=artifact_index,
            errors=errors,
        )
        _validate_attempts(
            attempts,
            manifest,
            errors,
            allow_empty=bool(confirmatory and confirmatory.no_adoption_reason and not selected),
        )
        _validate_runner_observability(artifact_index, manifest, attempts, errors)
        _validate_optimizer_p6_p7_evidence(
            manifest=manifest,
            fold_manifest=fold_manifest,
            confirmatory=confirmatory,
            rounds_manifest=rounds_manifest,
            selected=selected,
            fold_candidate_results=fold_candidate_results,
            fold_score_matrix=fold_score_matrix,
            selection_oos_evaluation=selection_oos_evaluation,
            selection_oos_repair_trigger=selection_oos_repair_trigger,
            repair_failure_attribution=repair_failure_attribution,
            accepted_mutation_chain=accepted_mutation_chain,
            repair_candidate_results=repair_candidate_results,
            repair_checkpoint=repair_checkpoint,
            round_n_plus_1_recommendation=round_n_plus_1_recommendation,
            errors=errors,
        )
        _validate_optimizer_decision(
            manifest=manifest,
            confirmatory=confirmatory,
            rounds_manifest=rounds_manifest,
            selected=selected,
            artifact_index=artifact_index,
            attempts=attempts,
            errors=errors,
        )

        repair_triggered = bool(confirmatory and confirmatory.repair_triggered)
        if repair_triggered:
            repair_missing = artifact_contract.missing_named_artifacts(PHASE4_OOS_REPAIR_ARTIFACTS)
            if repair_missing:
                errors.append(f"missing OOS repair artifacts: {', '.join(repair_missing)}")

        paths = {
            "fold_manifest_path": artifact_contract.path_str("fold_manifest.json"),
            "optimizer_run_manifest_path": artifact_contract.path_str(
                "optimizer_run_manifest.json",
            ),
            "experiment_plan_path": artifact_contract.path_str("llm_experiment_plan.json"),
            "candidate_attempts_path": artifact_contract.path_str("candidate_attempts.jsonl"),
            "runner_observability_path": artifact_contract.path_str(
                "runner_observability.json",
            ),
            "repair_ablation_matrix_path": artifact_contract.path_str(
                "repair_ablation_matrix.jsonl",
            ),
            "confirmatory_rerank_path": artifact_contract.path_str("confirmatory_rerank.json"),
            "rounds_manifest_path": artifact_contract.path_str("rounds_manifest.json"),
            "end_of_round_diagnostics_path": artifact_contract.path_str(
                "end_of_round_diagnostics.json",
            ),
        }
        evidence_paths.extend(path for path in paths.values() if path)
        if fold_manifest:
            evidence_paths.extend(artifact_contract.existing_paths(
                path for fold in fold_manifest.folds for path in fold.evidence_paths
            ))
        if experiment_plan:
            evidence_paths.extend(artifact_contract.existing_paths(experiment_plan.evidence_paths))
        if confirmatory:
            evidence_paths.extend(artifact_contract.existing_paths(confirmatory.evidence_paths))
        if rounds_manifest:
            evidence_paths.extend(artifact_contract.existing_paths(
                path for record in rounds_manifest.records for path in record.evidence_paths
            ))

        no_adoption_reason = ""
        adopted_candidate_id = ""
        if confirmatory:
            adopted_candidate_id = confirmatory.adopted_candidate_id
            no_adoption_reason = confirmatory.no_adoption_reason
        status = OptimizerSequenceStatus.SUCCEEDED
        if errors:
            status = OptimizerSequenceStatus.BLOCKED
        elif no_adoption_reason:
            status = OptimizerSequenceStatus.NO_ADOPTION

        return MonthlyOptimizerSequenceResult(
            run_id=manifest.run_id,
            status=status,
            adopted_candidate_id=adopted_candidate_id,
            no_adoption_reason=no_adoption_reason,
            repair_triggered=repair_triggered,
            blocking_reasons=errors,
            artifact_index_path=str(artifact_index_path),
            selected_candidate_ids=[candidate.candidate_id for candidate in selected],
            rejected_candidate_count=len(rejected),
            evidence_paths=_dedupe(evidence_paths),
            **paths,
        )


def build_two_fold_manifest(
    *,
    run_id: str,
    run_month: str,
    in_sample_start: date,
    in_sample_end: date,
    selection_oos_start: date,
    selection_oos_end: date,
    embargo_days: int = 5,
    evidence_paths: list[str] | None = None,
) -> FoldManifest:
    """Build the monthly two-fold purged in-sample manifest."""
    total_days = (in_sample_end - in_sample_start).days + 1
    if total_days < 2:
        raise ValueError("two-fold phased-auto requires at least two in-sample days")
    split = in_sample_start + timedelta(days=(total_days // 2) - 1)
    second_start = split + timedelta(days=1)
    evidence = evidence_paths or []
    return FoldManifest(
        run_id=run_id,
        run_month=run_month,
        in_sample_start=in_sample_start,
        in_sample_end=in_sample_end,
        selection_oos_start=selection_oos_start,
        selection_oos_end=selection_oos_end,
        folds=[
            FoldSpec(
                fold_id="fold_1",
                training_start=second_start,
                training_end=in_sample_end,
                validation_start=in_sample_start,
                validation_end=split,
                embargo_days=embargo_days,
                evidence_paths=evidence,
            ),
            FoldSpec(
                fold_id="fold_2",
                training_start=in_sample_start,
                training_end=split,
                validation_start=second_start,
                validation_end=in_sample_end,
                embargo_days=embargo_days,
                evidence_paths=evidence,
            ),
        ],
    )


def _attempt_id(run_id: str, candidate_id: str, attempt_number: int) -> str:
    raw = f"{run_id}:{candidate_id}:{attempt_number}"
    return "attempt-" + sanitize_workspace_key(raw)


def _missing_named_artifacts(index: BacktestArtifactIndex, names: list[str]) -> list[str]:
    return MonthlyArtifactContract.from_index(index).missing_named_artifacts(names)


def _load_attempts_from_rows(
    rows: list[dict[str, Any]],
    errors: list[str],
) -> list[CandidateAttemptRecord]:
    attempts: list[CandidateAttemptRecord] = []
    for row in rows:
        try:
            attempts.append(CandidateAttemptRecord.model_validate(row))
        except Exception as exc:
            errors.append(f"invalid candidate_attempts.jsonl row: {exc}")
    return attempts


def _latest_attempts(attempts: list[CandidateAttemptRecord]) -> dict[str, CandidateAttemptRecord]:
    latest: dict[str, CandidateAttemptRecord] = {}
    for attempt in attempts:
        current = latest.get(attempt.attempt_id)
        if current is None or attempt.updated_at >= current.updated_at:
            latest[attempt.attempt_id] = attempt
    return latest


def _validate_attempts(
    attempts: list[CandidateAttemptRecord],
    manifest: MonthlyRunManifest,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not attempts:
        if allow_empty:
            return
        errors.append("candidate_attempts.jsonl must include at least one attempt")
        return
    latest = _latest_attempts(attempts)
    attempt_ids_by_number: dict[tuple[str, str, int], set[str]] = {}
    for attempt in attempts:
        if attempt.run_id != manifest.run_id:
            errors.append(f"attempt {attempt.attempt_id} run_id does not match run manifest")
        if attempt.manifest_id and attempt.manifest_id != manifest.manifest_id:
            errors.append(f"attempt {attempt.attempt_id} manifest_id does not match run manifest")
        if (
            manifest.backtest_repo_commit_sha
            and attempt.backtest_repo_commit_sha
            and attempt.backtest_repo_commit_sha != manifest.backtest_repo_commit_sha
        ):
            errors.append(f"attempt {attempt.attempt_id} backtest repo SHA does not match run manifest")
        if (
            manifest.trading_repo_commit_sha
            and attempt.trading_repo_commit_sha
            and attempt.trading_repo_commit_sha != manifest.trading_repo_commit_sha
        ):
            errors.append(f"attempt {attempt.attempt_id} trading repo SHA does not match run manifest")
        number_key = (attempt.run_id, attempt.candidate_id, attempt.attempt_number)
        attempt_ids_by_number.setdefault(number_key, set()).add(attempt.attempt_id)
        if attempt.workspace_key != sanitize_workspace_key(attempt.workspace_key):
            errors.append(f"attempt {attempt.attempt_id} has unsafe workspace key")
        _validate_path_under_root(
            attempt.workspace_path,
            manifest.candidate_workspace_root or str(Path(manifest.artifact_root) / "workspaces"),
            f"attempt {attempt.attempt_id} workspace",
            errors,
        )
        for artifact_path in attempt.artifact_paths:
            _validate_path_under_any_root(
                artifact_path,
                [
                    manifest.artifact_root,
                    manifest.candidate_workspace_root or str(Path(manifest.artifact_root) / "workspaces"),
                ],
                f"attempt {attempt.attempt_id} artifact",
                errors,
            )
    for (_run_id, candidate_id, attempt_number), attempt_ids in attempt_ids_by_number.items():
        if len(attempt_ids) > 1:
            errors.append(
                "attempt_number collision for candidate "
                f"{candidate_id}: {attempt_number} used by attempts {', '.join(sorted(attempt_ids))}"
            )
    for attempt in latest.values():
        if attempt.state not in TERMINAL_ATTEMPT_STATES:
            errors.append(f"attempt {attempt.attempt_id} is not terminal: {attempt.state.value}")


def _validate_manifest_alignment(
    *,
    manifest: MonthlyRunManifest,
    fold_manifest: FoldManifest | None,
    experiment_plan: OptimizerExperimentPlan | None,
    confirmatory: ConfirmatoryRerank | None,
    rounds_manifest: RoundsManifest | None,
    errors: list[str],
) -> None:
    if fold_manifest is not None:
        if fold_manifest.run_id != manifest.run_id:
            errors.append("fold_manifest run_id does not match run manifest")
        if fold_manifest.run_month != manifest.run_month:
            errors.append("fold_manifest run_month does not match run manifest")
        if manifest.in_sample_start and fold_manifest.in_sample_start != manifest.in_sample_start:
            errors.append("fold_manifest in_sample_start does not match run manifest")
        if manifest.in_sample_end and fold_manifest.in_sample_end != manifest.in_sample_end:
            errors.append("fold_manifest in_sample_end does not match run manifest")
        if manifest.selection_oos_start and fold_manifest.selection_oos_start != manifest.selection_oos_start:
            errors.append("fold_manifest selection_oos_start does not match run manifest")
        if manifest.selection_oos_end and fold_manifest.selection_oos_end != manifest.selection_oos_end:
            errors.append("fold_manifest selection_oos_end does not match run manifest")
    if experiment_plan is not None:
        if experiment_plan.run_id != manifest.run_id:
            errors.append("llm_experiment_plan run_id does not match run manifest")
        _append_missing_paths(
            experiment_plan.evidence_paths,
            "llm_experiment_plan evidence",
            errors,
        )
        _validate_search_brief_consumed(manifest, experiment_plan, errors)
    if confirmatory is not None:
        if confirmatory.run_id != manifest.run_id:
            errors.append("confirmatory_rerank run_id does not match run manifest")
        _append_missing_paths(confirmatory.evidence_paths, "confirmatory_rerank evidence", errors)
    if rounds_manifest is not None:
        if rounds_manifest.run_id != manifest.run_id:
            errors.append("rounds_manifest run_id does not match run manifest")
        if rounds_manifest.bot_id and rounds_manifest.bot_id != manifest.bot_id:
            errors.append("rounds_manifest bot_id does not match run manifest")
        if rounds_manifest.strategy_id and rounds_manifest.strategy_id != manifest.strategy_id:
            errors.append("rounds_manifest strategy_id does not match run manifest")


def _validate_data_bundle_alignment(
    manifest: MonthlyRunManifest,
    artifact_index: BacktestArtifactIndex,
    errors: list[str],
) -> None:
    bundle_path = Path(manifest.data_bundle_manifest_path or manifest.market_data_manifest_path)
    if not bundle_path.exists():
        errors.append("data bundle manifest path does not exist")
        return
    try:
        bundle = DataBundleManifest.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))
    except Exception as exc:
        errors.append(f"invalid data bundle manifest: {exc}")
        return
    if bundle.status != DataBundleStatus.AUTHORITATIVE:
        errors.append(f"data bundle is not authoritative: {bundle.status.value}")
    expected_checksum = manifest.data_bundle_checksum or manifest.data_manifest_checksum
    if expected_checksum and bundle.bundle_checksum != expected_checksum:
        errors.append("data bundle checksum does not match run manifest")
    coverage_path = artifact_index.artifact_path("coverage_manifest.json")
    if coverage_path is None or not coverage_path.exists():
        return
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except Exception:
        return
    emitted_checksum = _find_checksum(coverage)
    if not emitted_checksum:
        errors.append("coverage_manifest.json must include data bundle checksum for optimizer runs")
    elif emitted_checksum != bundle.bundle_checksum:
        errors.append("coverage_manifest.json data bundle checksum does not match run manifest")


def _validate_optimizer_run_manifest(
    payload: dict[str, Any],
    *,
    manifest: MonthlyRunManifest,
    artifact_index: BacktestArtifactIndex,
    errors: list[str],
) -> None:
    if not payload:
        errors.append("optimizer_run_manifest.json must contain approval evidence provenance")
        return
    if payload.get("schema_version") != "optimizer_approval_run_manifest_v1":
        errors.append(
            "optimizer_run_manifest.json schema_version must be optimizer_approval_run_manifest_v1"
        )
    expected = {
        "run_id": manifest.run_id,
        "run_month": manifest.run_month,
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "data_bundle_checksum": manifest.data_bundle_checksum
        or manifest.data_manifest_checksum,
    }
    for field, value in expected.items():
        if value and str(payload.get(field) or "").strip() != value:
            errors.append(f"optimizer_run_manifest.json {field} does not match run manifest")
    artifact_root = str(payload.get("artifact_root") or "").strip()
    if artifact_root and Path(artifact_root).resolve() != Path(artifact_index.artifact_root).resolve():
        errors.append("optimizer_run_manifest.json artifact_root does not match artifact index")
    approval_mode = str(getattr(manifest.approval_mode, "value", manifest.approval_mode) or "")
    if approval_mode in {"", "none"}:
        return
    if payload.get("approval_grade_optimizer_run") is not True:
        errors.append("approval-mode optimizer run must set approval_grade_optimizer_run=true")
    if payload.get("smoke_mode") is not False:
        errors.append("approval-mode optimizer run must set smoke_mode=false")
    if str(payload.get("run_mode") or "").strip() == MonthlyRunMode.SMOKE_REPAIR.value:
        errors.append("approval-mode optimizer evidence cannot use smoke_repair run_mode")
    for path_key, hash_key in (
        ("run_manifest_path", "run_manifest_hash"),
        ("strategy_plugin_contract_path", "strategy_plugin_contract_hash"),
        ("deployment_metadata_path", "deployment_metadata_hash"),
    ):
        _validate_manifest_hash(payload, path_key, hash_key, errors)
    _validate_optimizer_bridge_hash_sets(payload, manifest=manifest, errors=errors)


def _validate_manifest_hash(
    payload: dict[str, Any],
    path_key: str,
    hash_key: str,
    errors: list[str],
) -> None:
    path_text = str(payload.get(path_key) or "").strip()
    expected_hash = str(payload.get(hash_key) or "").strip()
    if not path_text or not expected_hash:
        errors.append(f"optimizer_run_manifest.json missing {path_key}/{hash_key}")
        return
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        errors.append(f"optimizer_run_manifest.json {path_key} does not exist")
        return
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        errors.append(f"optimizer_run_manifest.json {hash_key} does not match {path_key}")


def _validate_optimizer_bridge_hash_sets(
    payload: dict[str, Any],
    *,
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> None:
    expected_scope = _optimizer_scope_from_manifest(manifest)
    payload_scope = str(payload.get("scope_id") or "").strip()
    if expected_scope and payload_scope and payload_scope != expected_scope:
        errors.append("optimizer_run_manifest.json scope_id does not match strategy plugin scope")
    scope_id = _optimizer_scope_id_for_manifest(payload, manifest)
    expected_bridge_ids = set(OPTIMIZER_BRIDGE_IDS_BY_SCOPE.get(scope_id, ()))
    contract_paths = _first_string_map(
        payload,
        ("bridge_contract_paths", "strategy_plugin_contract_paths"),
    )
    contract_hashes = _first_string_map(
        payload,
        ("bridge_contract_hashes", "strategy_plugin_contract_hashes"),
    )
    deployment_paths = _first_string_map(
        payload,
        ("bridge_deployment_metadata_paths", "deployment_metadata_paths"),
    )
    deployment_hashes = _first_string_map(
        payload,
        ("bridge_deployment_metadata_hashes", "deployment_metadata_hashes"),
    )
    _validate_manifest_hash_map(
        paths=contract_paths,
        hashes=contract_hashes,
        label="strategy contract",
        errors=errors,
    )
    _validate_manifest_hash_map(
        paths=deployment_paths,
        hashes=deployment_hashes,
        label="deployment metadata",
        errors=errors,
    )
    if expected_bridge_ids:
        _require_bridge_hash_set(
            expected_bridge_ids,
            paths=contract_paths,
            hashes=contract_hashes,
            label="strategy contract",
            errors=errors,
        )
        _require_bridge_hash_set(
            expected_bridge_ids,
            paths=deployment_paths,
            hashes=deployment_hashes,
            label="deployment metadata",
            errors=errors,
        )

    manifest_contract_paths = _manifest_path_map(
        manifest,
        ("bridge_contract_paths", "strategy_plugin_contract_paths"),
    )
    manifest_deployment_paths = _manifest_path_map(
        manifest,
        ("bridge_deployment_metadata_paths", "deployment_metadata_paths"),
    )
    _compare_manifest_path_hashes(
        manifest_contract_paths,
        actual_hashes=contract_hashes,
        label="manifest-declared strategy contract",
        errors=errors,
    )
    _compare_manifest_path_hashes(
        manifest_deployment_paths,
        actual_hashes=deployment_hashes,
        label="manifest-declared deployment metadata",
        errors=errors,
    )


def _optimizer_scope_id_for_manifest(
    payload: dict[str, Any],
    manifest: MonthlyRunManifest,
) -> str:
    manifest_scope = _optimizer_scope_from_manifest(manifest)
    payload_scope = str(payload.get("scope_id") or "").strip()
    return (
        manifest_scope
        or payload_scope
        or str(getattr(manifest, "scope_id", "") or "").strip()
        or manifest.strategy_id
        or manifest.bot_id
    )


def _optimizer_scope_from_manifest(manifest: MonthlyRunManifest) -> str:
    return OPTIMIZER_SCOPE_BY_PLUGIN_ID.get(manifest.strategy_plugin_id, "")


def _first_string_map(
    payload: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            mapped = {
                str(item_key).strip(): str(item).strip()
                for item_key, item in value.items()
                if str(item_key).strip() and str(item).strip()
            }
            if mapped:
                return mapped
    return {}


def _manifest_path_map(
    manifest: MonthlyRunManifest,
    keys: tuple[str, ...],
) -> dict[str, str]:
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


def _validate_manifest_hash_map(
    *,
    paths: dict[str, str],
    hashes: dict[str, str],
    label: str,
    errors: list[str],
) -> None:
    if not paths and not hashes:
        return
    for bridge_id, path_text in paths.items():
        expected_hash = hashes.get(bridge_id, "")
        if not expected_hash:
            errors.append(f"optimizer_run_manifest.json missing {label} hash for {bridge_id}")
            continue
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            errors.append(f"optimizer_run_manifest.json {label} path for {bridge_id} does not exist")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            errors.append(
                f"optimizer_run_manifest.json {label} hash for {bridge_id} does not match path"
            )
    extra_hashes = sorted(set(hashes) - set(paths))
    if extra_hashes:
        errors.append(
            f"optimizer_run_manifest.json {label} hash map has no path for: "
            + ", ".join(extra_hashes)
        )


def _require_bridge_hash_set(
    expected_bridge_ids: set[str],
    *,
    paths: dict[str, str],
    hashes: dict[str, str],
    label: str,
    errors: list[str],
) -> None:
    missing_paths = sorted(expected_bridge_ids - set(paths))
    if missing_paths:
        errors.append(
            f"optimizer_run_manifest.json missing {label} path(s) for: "
            + ", ".join(missing_paths)
        )
    missing_hashes = sorted(expected_bridge_ids - set(hashes))
    if missing_hashes:
        errors.append(
            f"optimizer_run_manifest.json missing {label} hash(es) for: "
            + ", ".join(missing_hashes)
        )


def _compare_manifest_path_hashes(
    paths: dict[str, str],
    *,
    actual_hashes: dict[str, str],
    label: str,
    errors: list[str],
) -> None:
    for bridge_id, path_text in paths.items():
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            errors.append(f"optimizer_run_manifest.json {label} path for {bridge_id} does not exist")
            continue
        expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hashes.get(bridge_id) != expected_hash:
            errors.append(
                f"optimizer_run_manifest.json {label} hash for {bridge_id} "
                "does not match run manifest"
            )


def _validate_search_brief_consumed(
    manifest: MonthlyRunManifest,
    experiment_plan: OptimizerExperimentPlan,
    errors: list[str],
) -> None:
    if not manifest.monthly_search_brief_path and not manifest.source_weekly_signal_ids:
        return
    expected_ids = {str(item) for item in manifest.source_weekly_signal_ids if str(item)}
    if not expected_ids:
        guidance_ids = manifest.monthly_search_guidance.get("source_weekly_signal_ids", [])
        expected_ids = {str(item) for item in guidance_ids if str(item)}
    guidance_has_content = bool(expected_ids) or any(
        manifest.monthly_search_guidance.get(key)
        for key in (
            "phase_order_hints",
            "priority_families",
            "seed_candidates",
            "negative_priors",
            "rollback_candidates",
        )
    )
    if manifest.monthly_search_brief_path:
        brief_path = Path(manifest.monthly_search_brief_path)
        if not brief_path.exists():
            errors.append("monthly_search_brief_path does not exist")
        elif guidance_has_content and str(brief_path) not in experiment_plan.evidence_paths:
            errors.append("llm_experiment_plan evidence_paths must cite monthly_search_brief_path")
    if expected_ids:
        consumed_ids = {str(item) for item in experiment_plan.source_weekly_signal_ids if str(item)}
        missing_ids = expected_ids - consumed_ids
        if missing_ids:
            errors.append(
                "llm_experiment_plan does not consume monthly search brief "
                f"source_weekly_signal_ids: {', '.join(sorted(missing_ids))}"
            )
    requirements = manifest.monthly_search_guidance.get("plan_requirements") or {}
    if not isinstance(requirements, dict):
        requirements = {}
    _validate_plan_requirement_families(
        experiment_plan,
        _string_set(requirements.get("candidate_families")),
        "candidate_families",
        errors,
    )
    _validate_plan_requirement_families(
        experiment_plan,
        _string_set(requirements.get("rollback_families")),
        "rollback_families",
        errors,
        require_caution=True,
    )
    _validate_plan_requirement_families(
        experiment_plan,
        _string_set(requirements.get("negative_prior_families")),
        "negative_prior_families",
        errors,
        require_caution=True,
    )


def _validate_plan_requirement_families(
    experiment_plan: OptimizerExperimentPlan,
    required: set[str],
    label: str,
    errors: list[str],
    *,
    require_caution: bool = False,
) -> None:
    if not required:
        return
    candidate_families = {
        str(item.get("family") or item.get("category") or "").strip().lower()
        for item in experiment_plan.candidate_families
        if isinstance(item, dict)
    }
    missing = {family for family in required if family.lower() not in candidate_families}
    if require_caution:
        caution_text = " ".join([
            *experiment_plan.gate_expectations,
            *experiment_plan.overfit_risks,
            json.dumps(experiment_plan.candidate_families, sort_keys=True, default=str),
        ]).lower()
        missing = {family for family in missing if family.lower() not in caution_text}
    if not missing:
        return
    errors.append(
        "llm_experiment_plan does not reflect monthly search brief "
        f"{label}: {', '.join(sorted(missing))}"
    )


def _validate_runner_observability(
    index: BacktestArtifactIndex,
    manifest: MonthlyRunManifest,
    attempts: list[CandidateAttemptRecord],
    errors: list[str],
) -> None:
    path = index.artifact_path("runner_observability.json")
    if path is None or not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid runner_observability.json: {exc}")
        return
    entries = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(entry, dict) for entry in entries):
        errors.append("runner_observability.json must contain an object or list of objects")
        return
    attempt_ids = {attempt.attempt_id for attempt in attempts}
    observed_attempt_ids: set[str] = set()
    for entry in entries:
        if str(entry.get("run_id") or "") != manifest.run_id:
            errors.append("runner_observability run_id does not match run manifest")
        attempt_id = str(entry.get("attempt_id") or "")
        if attempt_id:
            observed_attempt_ids.add(attempt_id)
        if attempt_id and attempt_ids and attempt_id not in attempt_ids:
            errors.append(f"runner_observability references unknown attempt_id: {attempt_id}")
        if attempt_ids and not str(entry.get("attempt_state") or ""):
            errors.append("runner_observability entries with attempts must include attempt_state")
    missing_observed_attempts = attempt_ids - observed_attempt_ids
    if missing_observed_attempts:
        errors.append(
            "runner_observability missing attempt entries: "
            + ", ".join(sorted(missing_observed_attempts))
        )


def _validate_optimizer_p6_p7_evidence(
    *,
    manifest: MonthlyRunManifest,
    fold_manifest: FoldManifest | None,
    confirmatory: ConfirmatoryRerank | None,
    rounds_manifest: RoundsManifest | None,
    selected: list[MonthlyImprovementCandidate],
    fold_candidate_results: list[dict[str, Any]],
    fold_score_matrix: dict[str, Any],
    selection_oos_evaluation: dict[str, Any],
    selection_oos_repair_trigger: dict[str, Any],
    repair_failure_attribution: dict[str, Any],
    accepted_mutation_chain: dict[str, Any],
    repair_candidate_results: list[dict[str, Any]],
    repair_checkpoint: dict[str, Any],
    round_n_plus_1_recommendation: dict[str, Any],
    errors: list[str],
) -> None:
    deterministic_no_adoption = bool(
        (confirmatory and confirmatory.no_adoption_reason)
        or (rounds_manifest and rounds_manifest.no_adoption_reason)
        or round_n_plus_1_recommendation.get("status") == "no_adoption"
    )
    if fold_manifest is not None:
        if len(fold_manifest.folds) != 2:
            errors.append("P6 evidence requires exactly two in-sample folds")
        if not all(fold.purged for fold in fold_manifest.folds):
            errors.append("P6 evidence requires purged in-sample folds")

    if fold_score_matrix:
        if str(fold_score_matrix.get("run_id") or "") not in {"", manifest.run_id}:
            errors.append("fold_score_matrix run_id does not match run manifest")
        if fold_score_matrix.get("selection_oos_excluded_from_first_pass") is not True:
            errors.append("fold_score_matrix must prove selection-OOS exclusion from first pass")
        scoring_windows = fold_score_matrix.get("scoring_windows") or []
        if not isinstance(scoring_windows, list) or len(scoring_windows) != 2:
            errors.append("fold_score_matrix must include two scoring windows")
        elif not all(
            isinstance(window, dict) and window.get("purged") is True
            for window in scoring_windows
        ):
            errors.append("fold_score_matrix scoring windows must be purged")
        if int(fold_score_matrix.get("candidate_count") or 0) <= 0 and not deterministic_no_adoption:
            errors.append("fold_score_matrix must include at least one scored candidate")
        matrix_candidates = fold_score_matrix.get("candidates") or []
        if isinstance(matrix_candidates, list) and matrix_candidates:
            if int(fold_score_matrix.get("candidate_count") or 0) != len(matrix_candidates):
                errors.append("fold_score_matrix candidate_count must match candidates length")

    if not fold_candidate_results:
        if not deterministic_no_adoption:
            errors.append("fold_candidate_results.jsonl must include first-pass fold rows")
    else:
        fold_ids = {
            str(row.get("fold_id") or "")
            for row in fold_candidate_results
            if isinstance(row, dict) and row.get("fold_id")
        }
        if len(fold_ids) < 2:
            errors.append("fold_candidate_results.jsonl must cover both purged folds")
        for row in fold_candidate_results:
            if str(row.get("run_id") or "") not in {"", manifest.run_id}:
                errors.append("fold_candidate_results row run_id does not match run manifest")
            if row.get("selection_oos_used_in_first_pass") is True:
                errors.append("fold_candidate_results rows must not use selection-OOS first pass")
                break
        missing_replay_rows = [
            row for row in fold_candidate_results if not isinstance(row.get("candidate"), dict)
        ]
        if missing_replay_rows:
            errors.append("fold_candidate_results rows must include candidate replay summaries")
        elif any(
            not str((row.get("candidate") or {}).get("evaluated_patch_fingerprint") or "")
            for row in fold_candidate_results
        ):
            errors.append(
                "fold_candidate_results candidate replays must include evaluated_patch_fingerprint"
            )
        elif any(
            not str((row.get("candidate") or {}).get("parameter_patch_fingerprint") or "")
            for row in fold_candidate_results
        ):
            errors.append(
                "fold_candidate_results candidate replays must include parameter_patch_fingerprint"
            )

    if selection_oos_evaluation:
        if str(selection_oos_evaluation.get("run_id") or "") not in {"", manifest.run_id}:
            errors.append("selection_oos_evaluation run_id does not match run manifest")
        if selection_oos_evaluation.get("selection_oos_used_after_fold_ranking") is not True:
            errors.append("selection_oos_evaluation must run after fold ranking")
        if selection_oos_evaluation.get("selection_oos_used_in_first_pass") is True:
            errors.append("selection_oos_evaluation must not mark selection-OOS as first pass")
        if not deterministic_no_adoption and not selection_oos_evaluation.get("candidate_selection_oos"):
            errors.append("selection_oos_evaluation must include the selected fold winner replay")

    repair_triggered = bool(confirmatory and confirmatory.repair_triggered)
    if selection_oos_repair_trigger:
        status = str(selection_oos_repair_trigger.get("status") or "")
        if str(selection_oos_repair_trigger.get("run_id") or "") not in {"", manifest.run_id}:
            errors.append("selection_oos_repair_trigger run_id does not match run manifest")
        if status not in {"triggered", "not_triggered"}:
            errors.append("selection_oos_repair_trigger status must be triggered or not_triggered")
        if not isinstance(selection_oos_repair_trigger.get("thresholds"), dict):
            errors.append("selection_oos_repair_trigger must include thresholds")
        if not isinstance(selection_oos_repair_trigger.get("measured_degradation"), dict):
            errors.append("selection_oos_repair_trigger must include measured_degradation")
        trigger_flag = bool(selection_oos_repair_trigger.get("triggered"))
        if repair_triggered and not trigger_flag:
            errors.append("repair-triggered confirmatory rerank requires measured OOS repair trigger")
        if not repair_triggered and trigger_flag:
            errors.append("repair trigger cannot be true when confirmatory_rerank repair_triggered=false")

    if confirmatory is not None:
        if confirmatory.primary_candidate_id and not confirmatory.variants:
            errors.append("confirmatory_rerank variants cannot be empty when a primary exists")

    if not repair_failure_attribution:
        errors.append("repair_failure_attribution.json must contain failure analysis")
    elif str(repair_failure_attribution.get("run_id") or "") not in {"", manifest.run_id}:
        errors.append("repair_failure_attribution run_id does not match run manifest")

    if not accepted_mutation_chain:
        errors.append("accepted_mutation_chain.json must contain accepted-mutation ledger context")
    elif not isinstance(accepted_mutation_chain.get("accepted_mutations", []), list):
        errors.append("accepted_mutation_chain accepted_mutations must be a list")

    if not repair_checkpoint:
        errors.append("repair_checkpoint.json must contain deterministic repair checkpoint")
    else:
        if str(repair_checkpoint.get("run_id") or "") not in {"", manifest.run_id}:
            errors.append("repair_checkpoint run_id does not match run manifest")
        if bool(repair_checkpoint.get("repair_triggered")) != repair_triggered:
            errors.append("repair_checkpoint repair_triggered must match confirmatory_rerank")

    if repair_triggered and not repair_candidate_results:
        errors.append("repair-triggered optimizer result must include repair candidate rows")

    if round_n_plus_1_recommendation:
        if str(round_n_plus_1_recommendation.get("run_id") or "") not in {"", manifest.run_id}:
            errors.append("round_n_plus_1_recommendation run_id does not match run manifest")
        status = str(round_n_plus_1_recommendation.get("status") or "")
        if status not in {"optimized_backtest_recommendation", "no_adoption"}:
            errors.append("round_n_plus_1_recommendation status is invalid")
        recommended_id = str(round_n_plus_1_recommendation.get("adopted_candidate_id") or "")
        no_adoption_reason = str(round_n_plus_1_recommendation.get("no_adoption_reason") or "")
        if status == "optimized_backtest_recommendation":
            if not recommended_id:
                errors.append("round_N+1 recommendation requires adopted_candidate_id")
            if confirmatory and recommended_id != confirmatory.adopted_candidate_id:
                errors.append("round_N+1 recommendation does not match confirmatory adoption")
            if rounds_manifest and recommended_id != rounds_manifest.adopted_candidate_id:
                errors.append("round_N+1 recommendation does not match rounds manifest")
            selected_matches = [
                candidate for candidate in selected if candidate.candidate_id == recommended_id
            ]
            if len(selected_matches) != 1:
                errors.append("round_N+1 recommendation requires exactly one matching selected candidate")
            else:
                selected_candidate = selected_matches[0]
                if confirmatory:
                    expected_source = (
                        confirmatory.adopted_source
                        if confirmatory.adopted_source != MonthlyCandidateSource.UNKNOWN
                        else confirmatory.primary_source
                    )
                    if selected_candidate.source != expected_source:
                        errors.append(
                            "round_N+1 selected candidate source does not match confirmatory adoption"
                        )
                if rounds_manifest:
                    round_source = _round_adopted_source(rounds_manifest, recommended_id)
                    if (
                        round_source != MonthlyCandidateSource.UNKNOWN
                        and selected_candidate.source != round_source
                    ):
                        errors.append(
                            "round_N+1 selected candidate source does not match rounds manifest"
                        )
                _validate_round_n_patch_evidence(
                    recommendation=round_n_plus_1_recommendation,
                    selected_candidate=selected_candidate,
                    artifact_root=Path(manifest.artifact_root),
                    errors=errors,
                )
        if status == "no_adoption" and not no_adoption_reason:
            errors.append("round_N+1 no-adoption recommendation requires a reason")
        if status == "no_adoption" and selected:
            errors.append("round_N+1 no-adoption recommendation cannot emit selected candidates")


def _validate_optimizer_decision(
    *,
    manifest: MonthlyRunManifest,
    confirmatory: ConfirmatoryRerank | None,
    rounds_manifest: RoundsManifest | None,
    selected: list[MonthlyImprovementCandidate],
    artifact_index: BacktestArtifactIndex,
    attempts: list[CandidateAttemptRecord],
    errors: list[str],
) -> None:
    if confirmatory is None or rounds_manifest is None:
        return
    if confirmatory.adopted_candidate_id != rounds_manifest.adopted_candidate_id:
        errors.append("confirmatory rerank and rounds manifest adopted_candidate_id differ")
    if confirmatory.no_adoption_reason != rounds_manifest.no_adoption_reason:
        errors.append("confirmatory rerank and rounds manifest no_adoption_reason differ")
    if confirmatory.no_adoption_reason:
        if selected:
            errors.append("no-adoption optimizer result must not emit selected candidates")
        return
    if confirmatory.adopted_candidate_id:
        adoption_missing = _missing_named_artifacts(artifact_index, ADOPTION_GATE_ARTIFACTS)
        if adoption_missing:
            errors.append(f"missing adoption gate artifacts: {', '.join(adoption_missing)}")
        if len(selected) != 1:
            errors.append("optimizer adoption requires exactly one selected candidate")
            return
        candidate = selected[0]
        if candidate.candidate_id != confirmatory.adopted_candidate_id:
            errors.append("selected candidate is not the confirmatory adopted candidate")
        expected_source = (
            confirmatory.adopted_source
            if confirmatory.adopted_source != MonthlyCandidateSource.UNKNOWN
            else confirmatory.primary_source
        )
        if candidate.source != expected_source:
            errors.append(f"adopted candidate source must be {expected_source.value}")
        _validate_candidate_lineage(candidate, manifest, errors)
        _validate_adopted_candidate_attempt(candidate, attempts, manifest, errors)
        if not _runner_contract_matches(candidate):
            errors.append("adopted candidate is missing the correct runner_contract_version")
        gate_inputs = candidate.deterministic_gate_inputs
        if gate_inputs.get("phase4_sequence_valid") is not True:
            errors.append("adopted candidate must set phase4_sequence_valid=true")
        if gate_inputs.get("round_n_plus_1_adopted") is not True:
            errors.append("adopted candidate must set round_n_plus_1_adopted=true")
        if gate_inputs.get("end_of_round_diagnostics_saved") is not True:
            errors.append("adopted candidate must prove end-of-round diagnostics were saved")
        if gate_inputs.get("live_backtest_parity_aligned") is not True:
            errors.append("adopted candidate must prove live/backtest parity alignment")
        for gate_key, message in (
            ("fold_support_passed", "adopted candidate must pass purged fold support"),
            ("calibration_support", "adopted candidate must prove calibration support"),
            ("leakage_passed", "adopted candidate must pass leakage checks"),
            ("cost_gate_passed", "adopted candidate must pass cost sensitivity"),
            ("drawdown_gate_passed", "adopted candidate must pass drawdown gate"),
            ("outlier_concentration_passed", "adopted candidate must pass outlier gate"),
            ("risk_constraints_passed", "adopted candidate must pass risk constraints"),
            ("sufficient_trade_count", "adopted candidate must prove sufficient trade count"),
        ):
            if gate_inputs.get(gate_key) is not True:
                errors.append(message)
        no_regression = gate_inputs.get("no_regression_gate_statuses")
        if isinstance(no_regression, dict):
            failed = [name for name, passed in no_regression.items() if passed is not True]
            if failed:
                errors.append(
                    "adopted candidate no-regression gates failed: "
                    + ", ".join(sorted(failed))
                )
        try:
            latest_oos_delta = float(gate_inputs.get("latest_month_oos_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            latest_oos_delta = 0.0
        if latest_oos_delta < -0.001:
            errors.append("adopted candidate selection-OOS delta materially degrades incumbent")
        for attr, name in (
            ("fold_manifest_path", "fold_manifest_path"),
            ("rounds_manifest_path", "rounds_manifest_path"),
            ("end_of_round_diagnostics_path", "end_of_round_diagnostics_path"),
            ("confirmatory_rerank_path", "confirmatory_rerank_path"),
        ):
            if not getattr(candidate, attr):
                errors.append(f"adopted candidate missing {name}")
        if candidate.score_component_count and candidate.score_component_count > manifest.score_component_cap:
            errors.append("adopted candidate exceeds score_component_cap")
        if _is_structural_candidate(candidate):
            _validate_structural_candidate(candidate, artifact_index, manifest, errors)
        _validate_path_under_root(
            candidate.candidate_workspace_path,
            manifest.candidate_workspace_root or str(Path(manifest.artifact_root) / "workspaces"),
            "adopted candidate workspace",
            errors,
        )
        if candidate.decision_parity_report_path:
            _validate_path_under_root(
                candidate.decision_parity_report_path,
                artifact_index.artifact_root,
                "adopted candidate decision parity report",
                errors,
            )


def _validate_adopted_candidate_attempt(
    candidate: MonthlyImprovementCandidate,
    attempts: list[CandidateAttemptRecord],
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> None:
    latest = _latest_attempts(attempts)
    if not candidate.candidate_attempt_id:
        errors.append("adopted candidate must include candidate_attempt_id")
        return
    attempt = latest.get(candidate.candidate_attempt_id)
    if attempt is None:
        errors.append("adopted candidate candidate_attempt_id is missing from attempt ledger")
        return
    if attempt.run_id != manifest.run_id:
        errors.append("adopted candidate attempt run_id does not match run manifest")
    if not attempt.manifest_id:
        errors.append("adopted candidate attempt missing manifest_id")
    elif attempt.manifest_id != manifest.manifest_id:
        errors.append("adopted candidate attempt manifest_id does not match run manifest")
    if attempt.candidate_id != candidate.candidate_id:
        errors.append("adopted candidate attempt belongs to a different candidate")
    if attempt.state != CandidateAttemptState.SUCCEEDED:
        errors.append(f"adopted candidate attempt must be succeeded, got {attempt.state.value}")
    if candidate.candidate_attempt_status and candidate.candidate_attempt_status != attempt.state.value:
        errors.append("adopted candidate attempt status does not match latest attempt ledger state")


def _validate_candidate_lineage(
    candidate: MonthlyImprovementCandidate,
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> None:
    if candidate.run_id != manifest.run_id:
        errors.append("adopted candidate run_id does not match run manifest")
    if candidate.manifest_id != manifest.manifest_id:
        errors.append("adopted candidate manifest_id does not match run manifest")
    for field_name in ("round_id", "prior_round_id", "next_round_id"):
        if not str(getattr(candidate, field_name) or "").strip():
            errors.append(f"adopted candidate missing {field_name}")
    if manifest.round_id and candidate.prior_round_id != manifest.round_id:
        errors.append("adopted candidate prior_round_id does not match run manifest round_id")
    if manifest.next_round_id and candidate.next_round_id != manifest.next_round_id:
        errors.append("adopted candidate next_round_id does not match run manifest")
    if not candidate.backtest_repo_commit_sha:
        errors.append("adopted candidate missing backtest_repo_commit_sha")
    elif (
        manifest.backtest_repo_commit_sha
        and candidate.backtest_repo_commit_sha != manifest.backtest_repo_commit_sha
    ):
        errors.append("adopted candidate backtest repo SHA does not match run manifest")
    live_sha = candidate.live_trading_repo_commit_sha or candidate.code_sha
    if not live_sha:
        errors.append("adopted candidate missing live_trading_repo_commit_sha")
    elif manifest.trading_repo_commit_sha and live_sha != manifest.trading_repo_commit_sha:
        errors.append("adopted candidate live trading repo SHA does not match run manifest")
    if not candidate.control_plane_commit_sha:
        errors.append("adopted candidate missing control_plane_commit_sha")
    elif (
        manifest.control_plane_commit_sha
        and candidate.control_plane_commit_sha != manifest.control_plane_commit_sha
    ):
        errors.append("adopted candidate control-plane SHA does not match run manifest")


def _runner_contract_matches(candidate: MonthlyImprovementCandidate) -> bool:
    version = str(
        candidate.deterministic_gate_inputs.get("runner_contract_version")
        or candidate.deterministic_gate_inputs.get("source_runner_contract_version")
        or candidate.workflow_contract_version
        or ""
    )
    expected = {
        MonthlyCandidateSource.SMOKE_REPAIR: "smoke_repair_runner_contract_v1",
        MonthlyCandidateSource.PHASED_AUTO: "phased_auto_runner_contract_v1",
    }.get(candidate.source, "")
    return bool(expected and version == expected)


def _round_adopted_source(
    rounds_manifest: RoundsManifest,
    adopted_candidate_id: str,
) -> MonthlyCandidateSource:
    for record in rounds_manifest.records:
        if record.candidate_id == adopted_candidate_id:
            return record.source
    return MonthlyCandidateSource.UNKNOWN


def _validate_round_n_patch_evidence(
    *,
    recommendation: dict[str, Any],
    selected_candidate: MonthlyImprovementCandidate,
    artifact_root: Path,
    errors: list[str],
) -> None:
    selected_evaluated = _candidate_payload_fingerprint(
        selected_candidate,
        "evaluated_patch_fingerprint",
    )
    recommended_evaluated = str(recommendation.get("evaluated_patch_fingerprint") or "")
    selected_parameter = _candidate_payload_fingerprint(
        selected_candidate,
        "parameter_patch_fingerprint",
    )
    recommended_parameter = str(recommendation.get("parameter_patch_fingerprint") or "")
    if not selected_evaluated:
        errors.append("selected candidate missing evaluated_patch_fingerprint")
    if not recommended_evaluated:
        errors.append("round_N+1 recommendation missing evaluated_patch_fingerprint")
    if selected_evaluated and recommended_evaluated and selected_evaluated != recommended_evaluated:
        errors.append("round_N+1 evaluated patch fingerprint does not match selected candidate")
    if not selected_parameter:
        errors.append("selected candidate missing parameter_patch_fingerprint")
    if not recommended_parameter:
        errors.append("round_N+1 recommendation missing parameter_patch_fingerprint")
    if selected_parameter and recommended_parameter and selected_parameter != recommended_parameter:
        errors.append("round_N+1 parameter patch fingerprint does not match selected candidate")

    selected_patch = _candidate_payload_dict(
        selected_candidate,
        ("evaluated_parameter_patch", "parameter_patch", "config_patch"),
    )
    selected_parameters = _candidate_payload_dict(
        selected_candidate,
        ("evaluated_parameters",),
    )
    if not selected_patch:
        errors.append("selected candidate missing concrete parameter_patch")
    elif selected_parameter and _canonical_patch_hash(selected_patch) != selected_parameter:
        errors.append("selected candidate parameter patch fingerprint is not canonical")
    if selected_patch and not selected_parameters:
        errors.append("selected candidate missing evaluated_parameters")
    elif selected_patch and selected_evaluated:
        recomputed_selected = _canonical_evaluated_patch_fingerprint(
            selected_patch,
            selected_parameters,
        )
        if recomputed_selected != selected_evaluated:
            errors.append("selected candidate evaluated patch fingerprint is not canonical")

    patch_path_text = str(recommendation.get("config_patch_path") or "")
    if not patch_path_text:
        errors.append("round_N+1 recommendation missing config_patch_path")
        return
    patch_path = Path(patch_path_text)
    if not patch_path.is_absolute():
        patch_path = artifact_root / patch_path
    if not patch_path.exists():
        errors.append("round_N+1 recommendation config_patch_path does not exist")
        return
    try:
        patch_payload = json.loads(patch_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid round_N+1 config_patch_path: {exc}")
        return
    if not isinstance(patch_payload, dict):
        errors.append("round_N+1 config patch must be a JSON object")
        return
    patch_fingerprint = _canonical_patch_hash(patch_payload)
    if recommended_parameter and patch_fingerprint != recommended_parameter:
        errors.append("round_N+1 config patch fingerprint does not match recommendation")
    evaluated_parameters = recommendation.get("evaluated_parameters")
    if not isinstance(evaluated_parameters, dict) or not evaluated_parameters:
        errors.append("round_N+1 recommendation missing evaluated_parameters")
        return
    if recommended_evaluated:
        recomputed_recommended = _canonical_evaluated_patch_fingerprint(
            patch_payload,
            evaluated_parameters,
        )
        if recomputed_recommended != recommended_evaluated:
            errors.append(
                "round_N+1 evaluated patch fingerprint does not match config patch and evaluated parameters"
            )


def _candidate_payload_fingerprint(
    candidate: MonthlyImprovementCandidate,
    key: str,
) -> str:
    raw = candidate.raw_payload if isinstance(candidate.raw_payload, dict) else {}
    direct = str(raw.get(key) or "")
    if direct:
        return direct
    nested_raw = raw.get("raw_payload")
    if isinstance(nested_raw, dict):
        nested_direct = str(nested_raw.get(key) or "")
        if nested_direct:
            return nested_direct
        candidate_payload = nested_raw.get("candidate_payload")
        if isinstance(candidate_payload, dict):
            return str(candidate_payload.get(key) or "")
    candidate_payload = raw.get("candidate_payload")
    if isinstance(candidate_payload, dict):
        return str(candidate_payload.get(key) or "")
    return ""


def _candidate_payload_dict(
    candidate: MonthlyImprovementCandidate,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    raw = candidate.raw_payload if isinstance(candidate.raw_payload, dict) else {}
    for key in keys:
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    nested_raw = raw.get("raw_payload")
    if isinstance(nested_raw, dict):
        for key in keys:
            value = nested_raw.get(key)
            if isinstance(value, dict):
                return value
        candidate_payload = nested_raw.get("candidate_payload")
        if isinstance(candidate_payload, dict):
            for key in keys:
                value = candidate_payload.get(key)
                if isinstance(value, dict):
                    return value
    candidate_payload = raw.get("candidate_payload")
    if isinstance(candidate_payload, dict):
        for key in keys:
            value = candidate_payload.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _canonical_patch_hash(value: dict[str, Any]) -> str:
    return _stable_json_hash(_normalize_patch(value))


def _canonical_evaluated_patch_fingerprint(
    patch: dict[str, Any],
    evaluated_parameters: dict[str, Any],
) -> str:
    return _stable_json_hash(
        {
            "parameter_patch": _normalize_patch(patch),
            "evaluated_parameters": _normalize_patch(evaluated_parameters),
        }
    )


def _normalize_patch(value: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        if isinstance(item, dict):
            normalized[str(key)] = _normalize_patch(item)
        elif isinstance(item, list):
            normalized[str(key)] = [
                _normalize_patch(element)
                if isinstance(element, dict)
                else _normalize_scalar(element)
                for element in item
            ]
        else:
            normalized[str(key)] = _normalize_scalar(item)
    return normalized


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 10)
    return str(value)


def _stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_structural_candidate(candidate: MonthlyImprovementCandidate) -> bool:
    return (
        candidate.change_kind == "structural_change"
        or bool(candidate.file_changes)
        or bool(candidate.live_repo_patch_path)
        or bool(candidate.backtest_adapter_patch_path)
    )


def _validate_structural_candidate(
    candidate: MonthlyImprovementCandidate,
    artifact_index: BacktestArtifactIndex,
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> None:
    missing = [
        name for name, value in (
            ("live_repo_patch_path", candidate.live_repo_patch_path),
            ("backtest_adapter_patch_path", candidate.backtest_adapter_patch_path),
            ("decision_parity_report_path", candidate.decision_parity_report_path),
        )
        if not value
    ]
    if missing:
        errors.append(f"structural candidate missing required lineage: {', '.join(missing)}")
    for label, path in (
        ("live repo patch", candidate.live_repo_patch_path),
        ("backtest adapter patch", candidate.backtest_adapter_patch_path),
        ("config patch", candidate.config_patch_path),
        ("decision parity report", candidate.decision_parity_report_path),
    ):
        if path:
            _validate_path_under_root(path, artifact_index.artifact_root, label, errors)
            if not Path(path).exists():
                errors.append(f"structural candidate {label} path does not exist")
    if candidate.decision_parity_report_path and Path(candidate.decision_parity_report_path).exists():
        try:
            report = DecisionParityReport.model_validate(
                json.loads(Path(candidate.decision_parity_report_path).read_text(encoding="utf-8"))
            )
        except Exception as exc:
            errors.append(f"structural candidate decision parity report is invalid: {exc}")
            return
        if report.run_id != candidate.run_id:
            errors.append("structural candidate decision parity run_id does not match candidate")
        if report.candidate_id != candidate.candidate_id:
            errors.append("structural candidate decision parity candidate_id does not match candidate")
        if not report.eligible_for_structural_approval:
            errors.append("structural candidate decision parity report is not pass")
        _validate_decision_parity_evidence(report, artifact_index, errors)
        if manifest.strategy_plugin_id and report.strategy_plugin_id != manifest.strategy_plugin_id:
            errors.append("structural candidate decision parity strategy_plugin_id does not match run manifest")
        contract = _load_strategy_plugin_contract(manifest, errors)
        if contract is None:
            errors.append("structural candidate requires strategy plugin contract evidence")
        else:
            if not contract.eligible_for_approval:
                errors.append("structural candidate strategy plugin contract is not approval-ready")
            errors.extend(
                deployment_metadata_errors(
                    manifest,
                    missing_reason="structural candidate requires deployment metadata evidence",
                )
            )
            if report.strategy_plugin_id != contract.plugin_id:
                errors.append("structural candidate decision parity strategy_plugin_id does not match plugin contract")
            if report.live_repo_commit_sha != contract.live_repo_commit_sha:
                errors.append("structural candidate decision parity live repo SHA does not match plugin contract")
            if report.backtest_adapter_commit_sha != contract.backtest_adapter_commit_sha:
                errors.append("structural candidate decision parity backtest adapter SHA does not match plugin contract")


def _load_strategy_plugin_contract(
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> StrategyPluginContract | None:
    if not manifest.strategy_plugin_contract_path:
        return None
    path = Path(manifest.strategy_plugin_contract_path)
    if not path.exists() or not path.is_file():
        errors.append("strategy plugin contract path is missing")
        return None
    try:
        return StrategyPluginContract.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        errors.append(f"strategy plugin contract is invalid: {exc}")
        return None


def _validate_decision_parity_evidence(
    report: DecisionParityReport,
    artifact_index: BacktestArtifactIndex,
    errors: list[str],
) -> None:
    evidence_paths = _dedupe([
        *report.evidence_paths,
        *[
            path
            for check in report.checks
            for path in check.evidence_paths
        ],
    ])
    for path in evidence_paths:
        _validate_path_under_root(path, artifact_index.artifact_root, "decision parity evidence", errors)
        if not Path(path).exists():
            errors.append(f"decision parity evidence path does not exist: {path}")


def _validate_path_under_root(path: str, root: str, label: str, errors: list[str]) -> None:
    if not path:
        return
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        errors.append(f"{label} is outside configured root")


def _validate_path_under_any_root(
    path: str,
    roots: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not path:
        return
    resolved = Path(path).resolve()
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return
        except (OSError, ValueError):
            continue
    errors.append(f"{label} is outside configured root")


def _append_missing_paths(paths: list[str], label: str, errors: list[str]) -> None:
    missing = [path for path in paths if path and not Path(path).exists()]
    if missing:
        errors.append(f"{label} paths do not exist: {', '.join(missing[:5])}")


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, list | tuple | set):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _find_checksum(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("bundle_checksum", "data_bundle_checksum", "checksum"):
            if value.get(key):
                return str(value[key])
        for item in value.values():
            found = _find_checksum(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_checksum(item)
            if found:
                return found
    return ""


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
