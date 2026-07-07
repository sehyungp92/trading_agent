"""Monthly optimizer runner sequence schemas.

These models describe the control-plane contract for the external monthly
optimizer.  The actual replay/search implementation lives behind the backtest
runner boundary; these schemas make the emitted artifacts measurable and
auditable before candidate ingestion or approval routing.
"""
from __future__ import annotations

import hashlib
import string
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from trading_assistant.schemas.monthly_candidates import MonthlyCandidateSource
from trading_assistant.schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION


MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION = "monthly_optimizer_workflow_contract_v1"
PHASED_AUTO_RUNNER_CONTRACT_VERSION = "phased_auto_runner_contract_v1"
SMOKE_REPAIR_RUNNER_CONTRACT_VERSION = "smoke_repair_runner_contract_v1"
TWO_FOLD_PURGED_MANIFEST_VERSION = "two_fold_purged_is_manifest_v1"


class OptimizerStage(str, Enum):
    DIAGNOSTICS = "diagnostics"
    LLM_EXPERIMENT_PLAN = "llm_experiment_plan"
    PHASED_AUTO = "phased_auto"
    OOS_REPAIR = "oos_repair"
    CONFIRMATORY_FOLLOW_UP = "confirmatory_follow_up"
    ROUND_ADOPTION = "round_adoption"


class CandidateAttemptState(str, Enum):
    UNCLAIMED = "unclaimed"
    CLAIMED = "claimed"
    RUNNING = "running"
    RETRY_QUEUED = "retry_queued"
    RELEASED = "released"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    STALLED = "stalled"
    CANCELED_BY_RECONCILIATION = "canceled_by_reconciliation"


class OptimizerSequenceStatus(str, Enum):
    SUCCEEDED = "succeeded"
    NO_ADOPTION = "no_adoption"
    BLOCKED = "blocked"
    FAILED = "failed"


class FoldSpec(BaseModel):
    fold_id: str
    training_start: date
    training_end: date
    validation_start: date
    validation_end: date
    embargo_days: int = 0
    purged: bool = True
    evidence_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_dates(self) -> "FoldSpec":
        if self.training_end < self.training_start:
            raise ValueError("training_end must be >= training_start")
        if self.validation_end < self.validation_start:
            raise ValueError("validation_end must be >= validation_start")
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        return self


class FoldManifest(BaseModel):
    run_id: str
    run_month: str
    in_sample_start: date
    in_sample_end: date
    selection_oos_start: date
    selection_oos_end: date
    folds: list[FoldSpec]
    manifest_version: str = TWO_FOLD_PURGED_MANIFEST_VERSION
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_manifest(self) -> "FoldManifest":
        if len(self.folds) != 2:
            raise ValueError("monthly phased-auto requires exactly two purged in-sample folds")
        if self.in_sample_end < self.in_sample_start:
            raise ValueError("in_sample_end must be >= in_sample_start")
        if self.selection_oos_end < self.selection_oos_start:
            raise ValueError("selection_oos_end must be >= selection_oos_start")
        if self.selection_oos_start <= self.in_sample_end:
            raise ValueError("selection-OOS must start after the in-sample window")
        for fold in self.folds:
            if not fold.purged:
                raise ValueError(f"{fold.fold_id} is not marked purged")
            if fold.validation_start < self.in_sample_start or fold.validation_end > self.in_sample_end:
                raise ValueError(f"{fold.fold_id} validation window is outside in-sample data")
        return self


class OptimizerExperimentPlan(BaseModel):
    run_id: str
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    score_components: list[str] = Field(default_factory=list)
    phase_order: list[str] = Field(default_factory=list)
    candidate_families: list[dict[str, Any]] = Field(default_factory=list)
    structural_candidates: list[dict[str, Any]] = Field(default_factory=list)
    gate_expectations: list[str] = Field(default_factory=list)
    overfit_risks: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    source_weekly_signal_ids: list[str] = Field(default_factory=list)
    plan_version: str = "monthly_optimizer_experiment_plan_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_plan(self) -> "OptimizerExperimentPlan":
        if len(self.score_components) > 7:
            raise ValueError("optimizer objective may use at most seven score components")
        if not self.evidence_paths:
            raise ValueError("optimizer experiment plan requires evidence_paths")
        if not self.candidate_families:
            raise ValueError("optimizer experiment plan requires candidate_families")
        if not self.gate_expectations:
            raise ValueError("optimizer experiment plan requires gate_expectations")
        if not self.overfit_risks:
            raise ValueError("optimizer experiment plan requires overfit_risks")
        return self


class CandidateWorkspaceManifest(BaseModel):
    run_id: str
    candidate_id: str
    workspace_key: str
    workspace_root: str
    workspace_path: str
    cwd_enforced: bool = True
    manifest_path: str = ""
    structural: bool = False
    live_repo_patch_path: str = ""
    backtest_adapter_patch_path: str = ""
    config_patch_path: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    manifest_version: str = "candidate_workspace_manifest_v1"

    @model_validator(mode="after")
    def _validate_workspace(self) -> "CandidateWorkspaceManifest":
        root = Path(self.workspace_root)
        workspace = Path(self.workspace_path)
        try:
            workspace.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("workspace_path must stay under workspace_root") from exc
        if self.workspace_key != sanitize_workspace_key(self.workspace_key):
            raise ValueError("workspace_key is not sanitized")
        if workspace.name != self.workspace_key:
            raise ValueError("workspace_path basename must match workspace_key")
        return self


class CandidateAttemptRecord(BaseModel):
    attempt_id: str
    run_id: str
    candidate_id: str
    workspace_key: str
    workspace_path: str
    state: CandidateAttemptState = CandidateAttemptState.UNCLAIMED
    stage: OptimizerStage = OptimizerStage.PHASED_AUTO
    attempt_number: int = 1
    retry_attempt: int = 0
    retry_reason: str = ""
    stall_timeout_seconds: int = 0
    subprocess_pid: int | None = None
    manifest_id: str = ""
    backtest_repo_commit_sha: str = ""
    trading_repo_commit_sha: str = ""
    phase: str = ""
    reason: str = ""
    artifact_paths: list[str] = Field(default_factory=list)
    parity_status: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_attempt(self) -> "CandidateAttemptRecord":
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be >= 1")
        if self.retry_attempt < 0:
            raise ValueError("retry_attempt cannot be negative")
        if self.stall_timeout_seconds < 0:
            raise ValueError("stall_timeout_seconds cannot be negative")
        if self.workspace_key != sanitize_workspace_key(self.workspace_key):
            raise ValueError("workspace_key is not sanitized")
        return self


class ConfirmatoryVariant(BaseModel):
    candidate_id: str
    source_candidate_id: str = ""
    variant_type: str = ""
    objective_score: float = 0.0
    baseline_score: float = 0.0
    in_sample_delta: float = 0.0
    selection_oos_delta: float = 0.0
    fold_support_passed: bool = False
    deterministic_replay_passed: bool = False
    materially_degrades_in_sample: bool = False
    evidence_paths: list[str] = Field(default_factory=list)


class ConfirmatoryRerank(BaseModel):
    run_id: str
    primary_candidate_id: str = ""
    primary_source: MonthlyCandidateSource = MonthlyCandidateSource.PHASED_AUTO
    repair_triggered: bool = False
    compared_candidate_ids: list[str] = Field(default_factory=list)
    variants: list[ConfirmatoryVariant] = Field(default_factory=list)
    adopted_candidate_id: str = ""
    adopted_source: MonthlyCandidateSource = MonthlyCandidateSource.UNKNOWN
    no_adoption_reason: str = ""
    selection_rule: str = ""
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    evidence_paths: list[str] = Field(default_factory=list)
    rerank_version: str = "confirmatory_rerank_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_rerank(self) -> "ConfirmatoryRerank":
        if bool(self.adopted_candidate_id) == bool(self.no_adoption_reason):
            raise ValueError("confirmatory rerank must have exactly one adoption or no-adoption reason")
        if self.repair_triggered and self.primary_source != MonthlyCandidateSource.SMOKE_REPAIR:
            raise ValueError("repair-triggered confirmatory rerank must center a smoke/OOS-repair candidate")
        if not self.repair_triggered and self.primary_source != MonthlyCandidateSource.PHASED_AUTO:
            raise ValueError("non-repair confirmatory rerank must center the phased-auto winner")
        if self.adopted_candidate_id and self.adopted_candidate_id not in set(self.compared_candidate_ids):
            raise ValueError("adopted candidate must be listed in compared_candidate_ids")
        if self.primary_candidate_id and self.primary_candidate_id not in set(self.compared_candidate_ids):
            raise ValueError("primary candidate must be listed in compared_candidate_ids")
        if self.adopted_candidate_id and self.adopted_source == MonthlyCandidateSource.UNKNOWN:
            self.adopted_source = self.primary_source
        return self


class RoundManifestRecord(BaseModel):
    round_id: str
    prior_round_id: str = ""
    next_round_id: str = ""
    candidate_id: str = ""
    source: MonthlyCandidateSource = MonthlyCandidateSource.UNKNOWN
    strategy_version: str = ""
    config_version: str = ""
    parameter_set_id: str = ""
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    fold_manifest_path: str = ""
    diagnostics_path: str = ""
    confirmatory_rerank_path: str = ""
    decision_parity_report_path: str = ""
    approval_state: str = "not_requested"
    live_deployment_status: str = "optimized_backtest_recommendation"
    evidence_paths: list[str] = Field(default_factory=list)


class RoundsManifest(BaseModel):
    run_id: str
    bot_id: str = ""
    strategy_id: str = ""
    current_round_id: str
    next_round_id: str
    adopted_candidate_id: str = ""
    no_adoption_reason: str = ""
    records: list[RoundManifestRecord] = Field(default_factory=list)
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    manifest_version: str = "rounds_manifest_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_rounds(self) -> "RoundsManifest":
        if bool(self.adopted_candidate_id) == bool(self.no_adoption_reason):
            raise ValueError("rounds manifest must have exactly one adoption or no-adoption reason")
        if self.adopted_candidate_id and not self.next_round_id:
            raise ValueError("adopted rounds manifest requires next_round_id")
        if self.adopted_candidate_id:
            matching = [
                record for record in self.records
                if record.round_id == self.next_round_id
                and record.candidate_id == self.adopted_candidate_id
            ]
            if len(matching) != 1:
                raise ValueError("rounds manifest must link the adopted candidate to exactly one next round")
            if matching[0].live_deployment_status == "live_deployed":
                raise ValueError("round_N+1 must not be recorded as a live deployment")
        return self


class MonthlyOptimizerSequenceResult(BaseModel):
    run_id: str
    status: OptimizerSequenceStatus
    adopted_candidate_id: str = ""
    no_adoption_reason: str = ""
    repair_triggered: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    artifact_index_path: str = ""
    fold_manifest_path: str = ""
    experiment_plan_path: str = ""
    candidate_attempts_path: str = ""
    runner_observability_path: str = ""
    repair_ablation_matrix_path: str = ""
    confirmatory_rerank_path: str = ""
    rounds_manifest_path: str = ""
    end_of_round_diagnostics_path: str = ""
    selected_candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidate_count: int = 0
    evidence_paths: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def sanitize_workspace_key(value: str) -> str:
    """Return the stable path-safe workspace key used by optimizer attempts."""
    raw = str(value or "").strip()
    allowed = set(string.ascii_letters + string.digits + "-_")
    safe = "".join(ch if ch in allowed else "_" for ch in raw)
    safe = safe.strip("._-")
    if not safe:
        safe = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12] if raw else "candidate"
    if len(safe) > 64:
        safe = f"{safe[:51]}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"
    return safe
