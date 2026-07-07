"""Canonical contract models for the trading workspace."""

from __future__ import annotations

import hashlib
import json
import string
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OBJECTIVE_WEIGHTS_VERSION = "objective_weights_v1"
MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION = "monthly_optimizer_workflow_contract_v1"
PHASED_AUTO_RUNNER_CONTRACT_VERSION = "phased_auto_runner_contract_v1"
SMOKE_REPAIR_RUNNER_CONTRACT_VERSION = "smoke_repair_runner_contract_v1"
TWO_FOLD_PURGED_MANIFEST_VERSION = "two_fold_purged_is_manifest_v1"

REQUIRED_BACKTEST_ARTIFACTS = [
    "coverage_manifest.json",
    "incumbent_validation.json",
    "gap_attribution.json",
    "mode_decision.json",
    "replay_parity_report.json",
    "objective_breakdown.json",
    "candidate_results.jsonl",
    "selected_candidates.json",
    "rejected_candidates.jsonl",
    "monthly_report.md",
    "stdout.log",
    "stderr.log",
    "exit_status.json",
]

PHASE4_OPTIMIZER_ARTIFACTS = [
    "optimizer_run_manifest.json",
    "leakage_report.json",
    "cost_sensitivity.json",
    "fold_validation.json",
    "outlier_sensitivity.json",
    "portfolio_synergy.json",
    "fold_manifest.json",
    "rounds_manifest.json",
    "end_of_round_diagnostics.json",
    "llm_experiment_plan.json",
    "candidate_workspace_manifest.json",
    "candidate_attempts.jsonl",
    "runner_observability.json",
    "confirmatory_rerank.json",
    "fold_candidate_results.jsonl",
    "fold_score_matrix.json",
    "selection_oos_evaluation.json",
    "selection_oos_repair_trigger.json",
    "repair_failure_attribution.json",
    "accepted_mutation_chain.json",
    "repair_candidate_results.jsonl",
    "repair_checkpoint.json",
    "round_n_plus_1_recommendation.json",
    "replay_evaluator_report.json",
    "frozen_baseline.json",
    "round_reproduction_report.json",
    "historical_walk_forward_report.json",
    "replay_evidence_report.json",
]

PHASE4_OOS_REPAIR_ARTIFACTS = ["repair_ablation_matrix.jsonl"]

PHASE4_STRUCTURAL_CANDIDATE_ARTIFACTS = [
    "structural_candidate_plan.json",
    "structural_selection_gate.json",
    "live_repo_patch.diff",
    "backtest_adapter_patch.diff",
    "config_patch.diff",
    "decision_parity_report.json",
]

LEARNING_INSTRUMENTATION_ARTIFACTS = [
    "learning_sufficiency_manifest.json",
    "strategy_discovery_packet.json",
]

OPTIONAL_BACKTEST_ARTIFACTS = [
    *PHASE4_OPTIMIZER_ARTIFACTS,
    *PHASE4_OOS_REPAIR_ARTIFACTS,
    *PHASE4_STRUCTURAL_CANDIDATE_ARTIFACTS,
    *LEARNING_INSTRUMENTATION_ARTIFACTS,
]

JSON_ARTIFACTS = {
    name
    for name in [*REQUIRED_BACKTEST_ARTIFACTS, *OPTIONAL_BACKTEST_ARTIFACTS]
    if name.endswith(".json")
}
JSONL_ARTIFACTS = {
    name
    for name in [*REQUIRED_BACKTEST_ARTIFACTS, *OPTIONAL_BACKTEST_ARTIFACTS]
    if name.endswith(".jsonl")
}

DECISION_PARITY_DIMENSIONS = {
    "signals",
    "filters",
    "entries",
    "exits",
    "stops",
    "sizing",
    "risk_caps",
    "order_intent",
}


class MonthlyRunMode(str, Enum):
    INCUMBENT_VALIDATION = "incumbent_validation"
    SMOKE_REPAIR = "smoke_repair"
    PHASED_AUTO = "phased_auto"
    STRUCTURAL_REVIEW = "structural_review"
    OUTCOME_MEASUREMENT = "outcome_measurement"


class MonthlyApprovalMode(str, Enum):
    NONE = "none"
    EXPERIMENT = "experiment"
    MANUAL_REQUIRED = "manual_required"


class MonthlyCandidateSource(str, Enum):
    SMOKE_REPAIR = "smoke_repair"
    PHASED_AUTO = "phased_auto"
    MODEL_REVIEW = "model_review"
    UNKNOWN = "unknown"


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


class OptimizerStage(str, Enum):
    DIAGNOSTICS = "diagnostics"
    LLM_EXPERIMENT_PLAN = "llm_experiment_plan"
    PHASED_AUTO = "phased_auto"
    OOS_REPAIR = "oos_repair"
    CONFIRMATORY_FOLLOW_UP = "confirmatory_follow_up"
    ROUND_ADOPTION = "round_adoption"


class DataBundleStatus(str, Enum):
    AUTHORITATIVE = "authoritative"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    BLOCKED = "blocked"


class StrategyPluginMaturity(str, Enum):
    DIAGNOSTIC = "diagnostic"
    SHADOW_VALIDATED = "shadow_validated"
    APPROVAL_READY = "approval_ready"


class DecisionParityStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class AssistantLineage(BaseModel):
    weekly_signal_ids: list[str] = Field(default_factory=list)
    monthly_search_brief_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    experiment_id: str = ""
    variant_id: str = ""
    parameter_set_id: str = ""
    deployment_id: str = ""
    strategy_change_record_ids: list[str] = Field(default_factory=list)
    monthly_outcome_id: str = ""


class MonthlyRunManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    run_month: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mode: MonthlyRunMode = MonthlyRunMode.INCUMBENT_VALIDATION
    bot_id: str
    strategy_id: str
    strategy_version: str = ""
    config_version: str = ""
    config_hash: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_ids: list[str] = Field(default_factory=list)
    assistant_lineage: AssistantLineage = Field(default_factory=AssistantLineage)
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    latest_month_start: date
    latest_month_end: date
    calibration_start: date | None = None
    calibration_end: date | None = None
    selection_oos_start: date | None = None
    selection_oos_end: date | None = None
    market_data_manifest_path: str
    telemetry_manifest_path: str
    learning_sufficiency_manifest_path: str = ""
    learning_sufficiency_status: str = ""
    supported_learning_capabilities: list[str] = Field(default_factory=list)
    blocked_learning_capabilities: list[str] = Field(default_factory=list)
    strategy_discovery_packet_path: str = ""
    backtest_repo_path: str = ""
    backtest_repo_commit_sha: str = ""
    trading_repo_path: str = ""
    trading_repo_branch: str = ""
    trading_repo_commit_sha: str = ""
    deployment_metadata_path: str = ""
    deployment_metadata_paths: dict[str, str] = Field(default_factory=dict)
    bridge_deployment_metadata_paths: dict[str, str] = Field(default_factory=dict)
    expected_session_paths: list[str] = Field(default_factory=list)
    runtime_support_paths: list[str] = Field(default_factory=list)
    control_plane_commit_sha: str = ""
    backtest_command: list[str] = Field(default_factory=list)
    artifact_root: str
    strategy_plugin_id: str = ""
    strategy_plugin_contract_path: str = ""
    strategy_plugin_contract_paths: dict[str, str] = Field(default_factory=dict)
    bridge_contract_paths: dict[str, str] = Field(default_factory=dict)
    strategy_plugin_contract_version: str = ""
    round_id: str = ""
    prior_round_id: str = ""
    next_round_id: str = ""
    round_n_strategy_config_path: str = ""
    round_n_strategy_config_version: str = ""
    round_n_portfolio_config_path: str = ""
    round_n_portfolio_config_version: str = ""
    data_manifest_checksum: str = ""
    data_bundle_manifest_path: str = ""
    data_bundle_checksum: str = ""
    in_sample_start: date | None = None
    in_sample_end: date | None = None
    fold_manifest_path: str = ""
    rounds_manifest_path: str = ""
    end_of_round_diagnostics_path: str = ""
    candidate_workspace_root: str = ""
    candidate_workspace_key: str = ""
    candidate_workspace_manifest_path: str = ""
    candidate_attempt_id: str = ""
    candidate_attempt_number: int = 0
    candidate_attempt_status: str = ""
    retry_reason: str = ""
    stall_timeout_seconds: int = 0
    max_workers: int = 2
    score_component_cap: int = 7
    checkpoint_path: str = ""
    cache_path: str = ""
    outcome_prior_snapshot_path: str = ""
    monthly_search_brief_path: str = ""
    monthly_search_brief_id: str = ""
    source_weekly_signal_ids: list[str] = Field(default_factory=list)
    monthly_search_guidance: dict[str, Any] = Field(default_factory=dict)
    workflow_contract_path: str = ""
    workflow_contract_version: str = "monthly_incumbent_validation_v1"
    source_runner_contract_versions: list[str] = Field(
        default_factory=lambda: [
            SMOKE_REPAIR_RUNNER_CONTRACT_VERSION,
            PHASED_AUTO_RUNNER_CONTRACT_VERSION,
        ]
    )
    output_artifact_names: list[str] = Field(default_factory=list)
    required_json_schemas: list[str] = Field(default_factory=list)
    approval_mode: MonthlyApprovalMode = MonthlyApprovalMode.NONE
    expected_outputs: list[str] = Field(default_factory=list)
    manifest_version: str = "monthly_run_manifest_v1"

    @field_validator("run_month")
    @classmethod
    def _validate_run_month(cls, value: str) -> str:
        parts = value.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            raise ValueError("run_month must be YYYY-MM")
        month = int(parts[1])
        if not 1 <= month <= 12:
            raise ValueError("run_month month must be 01..12")
        return value

    @model_validator(mode="after")
    def _validate_windows(self) -> MonthlyRunManifest:
        if self.latest_month_end < self.latest_month_start:
            raise ValueError("latest_month_end must be >= latest_month_start")
        for start, end, name in (
            (self.calibration_start, self.calibration_end, "calibration"),
            (self.selection_oos_start, self.selection_oos_end, "selection_oos"),
            (self.in_sample_start, self.in_sample_end, "in_sample"),
        ):
            if (start is None) != (end is None):
                raise ValueError(f"{name}_start and {name}_end must be set together")
            if start and end and end < start:
                raise ValueError(f"{name}_end must be >= {name}_start")
        if (
            self.in_sample_end
            and self.selection_oos_start
            and self.selection_oos_start <= self.in_sample_end
        ):
            raise ValueError("selection_oos_start must be after in_sample_end")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.score_component_cap < 1 or self.score_component_cap > 7:
            raise ValueError("score_component_cap must be between 1 and 7")
        if self.mode == MonthlyRunMode.PHASED_AUTO and (
            not self.workflow_contract_version
            or self.workflow_contract_version == "monthly_incumbent_validation_v1"
        ):
            self.workflow_contract_version = MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION
        return self

    @property
    def manifest_id(self) -> str:
        raw = "|".join(
            [
                self.run_id,
                self.run_month,
                self.bot_id,
                self.strategy_id,
                self.mode.value,
                self.latest_month_start.isoformat(),
                self.latest_month_end.isoformat(),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def optimizer_mode(self) -> bool:
        return self.mode in {
            MonthlyRunMode.PHASED_AUTO,
            MonthlyRunMode.SMOKE_REPAIR,
            MonthlyRunMode.STRUCTURAL_REVIEW,
        }


class DataBundleSlice(BaseModel):
    model_config = ConfigDict(extra="allow")

    manifest_path: str
    manifest_id: str = ""
    source: str = ""
    market: str = ""
    symbol: str
    timeframe: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    checksum: str = ""
    calendar: str = ""
    authoritative: bool = False


class DataBundleManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    bundle_id: str = ""
    data_repo_path: str = ""
    data_repo_commit_sha: str = ""
    data_repo_branch: str = ""
    slice_manifests: list[DataBundleSlice]
    bundle_checksum: str = ""
    calendars: list[str] = Field(default_factory=list)
    fee_model_version: str = ""
    slippage_model_version: str = ""
    adjustment_policy: str = ""
    status: DataBundleStatus = DataBundleStatus.DIAGNOSTICS_ONLY
    diagnostics_only_reason: str = ""
    schema_version: str = "data_bundle_manifest_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _normalize(self) -> DataBundleManifest:
        if not self.slice_manifests:
            raise ValueError("data bundle requires at least one slice manifest")
        if not self.calendars:
            self.calendars = sorted(
                {item.calendar for item in self.slice_manifests if item.calendar}
            )
        if not self.bundle_checksum:
            raw = "|".join(
                [
                    self.data_repo_commit_sha,
                    self.fee_model_version,
                    self.slippage_model_version,
                    self.adjustment_policy,
                    *[
                        "|".join([item.manifest_id, item.symbol, item.timeframe, item.checksum])
                        for item in self.slice_manifests
                    ],
                ]
            )
            self.bundle_checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if not self.bundle_id:
            raw = "|".join(
                [
                    self.data_repo_commit_sha,
                    self.bundle_checksum,
                    ",".join(f"{item.symbol}:{item.timeframe}" for item in self.slice_manifests),
                ]
            )
            self.bundle_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self

    def authoritative_contract_errors(self) -> list[str]:
        missing: list[str] = []
        for attr in (
            "data_repo_commit_sha",
            "bundle_checksum",
            "fee_model_version",
            "slippage_model_version",
            "adjustment_policy",
        ):
            if not str(getattr(self, attr, "") or "").strip():
                missing.append(attr)
        if not self.calendars:
            missing.append("calendars")
        for index, item in enumerate(self.slice_manifests):
            if not item.checksum:
                missing.append(f"slice_manifests[{index}].checksum")
            if not item.calendar:
                missing.append(f"slice_manifests[{index}].calendar")
            if not item.authoritative:
                missing.append(f"slice_manifests[{index}].authoritative")
        return missing

    @property
    def usable_for_authoritative_validation(self) -> bool:
        return (
            self.status == DataBundleStatus.AUTHORITATIVE
            and not self.authoritative_contract_errors()
        )


class StrategyPluginContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    plugin_id: str
    live_repo_path: str = ""
    live_repo_commit_sha: str = ""
    backtest_adapter_path: str
    backtest_adapter_commit_sha: str = ""
    config_schema_version: str
    decision_api_version: str
    required_telemetry_schemas: list[str] = Field(default_factory=list)
    supported_symbols: list[str] = Field(default_factory=list)
    supported_timeframes: list[str] = Field(default_factory=list)
    parity_fixture_set: list[str] = Field(default_factory=list)
    maturity: StrategyPluginMaturity = StrategyPluginMaturity.DIAGNOSTIC
    contract_id: str = ""
    contract_version: str = "strategy_plugin_contract_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _normalize(self) -> StrategyPluginContract:
        self.plugin_id = self.plugin_id.strip()
        self.live_repo_path = self.live_repo_path.strip()
        self.live_repo_commit_sha = self.live_repo_commit_sha.strip()
        self.backtest_adapter_path = self.backtest_adapter_path.strip()
        self.backtest_adapter_commit_sha = self.backtest_adapter_commit_sha.strip()
        self.config_schema_version = self.config_schema_version.strip()
        self.decision_api_version = self.decision_api_version.strip()
        self.required_telemetry_schemas = sorted(
            {schema.strip() for schema in self.required_telemetry_schemas if schema.strip()}
        )
        self.supported_symbols = sorted(
            {symbol.strip().upper() for symbol in self.supported_symbols if symbol.strip()}
        )
        self.supported_timeframes = sorted(
            {timeframe.strip() for timeframe in self.supported_timeframes if timeframe.strip()}
        )
        self.parity_fixture_set = [path.strip() for path in self.parity_fixture_set if path.strip()]
        missing = [
            attr
            for attr in (
                "plugin_id",
                "backtest_adapter_path",
                "config_schema_version",
                "decision_api_version",
            )
            if not str(getattr(self, attr) or "").strip()
        ]
        if missing:
            raise ValueError(
                "strategy plugin contract missing required fields: " + ", ".join(missing)
            )
        if self.maturity in {
            StrategyPluginMaturity.SHADOW_VALIDATED,
            StrategyPluginMaturity.APPROVAL_READY,
        }:
            mature_missing = self.maturity_contract_errors()
            if mature_missing:
                raise ValueError(
                    "mature strategy plugin contract missing required fields: "
                    + ", ".join(sorted(mature_missing))
                )
        if not self.contract_id:
            raw = "|".join(
                [
                    self.plugin_id,
                    self.live_repo_commit_sha,
                    self.backtest_adapter_path,
                    self.backtest_adapter_commit_sha,
                    self.config_schema_version,
                    self.decision_api_version,
                    self.maturity.value,
                ]
            )
            self.contract_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self

    def maturity_contract_errors(self) -> list[str]:
        missing: list[str] = []
        for attr in (
            "live_repo_path",
            "live_repo_commit_sha",
            "backtest_adapter_path",
            "backtest_adapter_commit_sha",
            "config_schema_version",
            "decision_api_version",
        ):
            if not str(getattr(self, attr, "") or "").strip():
                missing.append(attr)
        for attr in (
            "required_telemetry_schemas",
            "supported_symbols",
            "supported_timeframes",
            "parity_fixture_set",
        ):
            if not getattr(self, attr):
                missing.append(attr)
        return missing

    @property
    def eligible_for_optimizer(self) -> bool:
        return self.maturity in {
            StrategyPluginMaturity.SHADOW_VALIDATED,
            StrategyPluginMaturity.APPROVAL_READY,
        } and not self.maturity_contract_errors()

    @property
    def eligible_for_approval(self) -> bool:
        return (
            self.maturity == StrategyPluginMaturity.APPROVAL_READY
            and not self.maturity_contract_errors()
        )


class BacktestArtifactIndex(BaseModel):
    run_id: str
    manifest_id: str = ""
    artifact_root: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    index_version: str = "backtest_artifact_index_v1"

    @model_validator(mode="after")
    def _fill_default_paths(self) -> BacktestArtifactIndex:
        for name in REQUIRED_BACKTEST_ARTIFACTS + OPTIONAL_BACKTEST_ARTIFACTS:
            self.artifacts.setdefault(name, str(Path(self.artifact_root) / name))
        return self

    def artifact_path(self, name: str) -> Path | None:
        raw = str(self.artifacts.get(name, "") or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = Path(self.artifact_root) / path
        return path

    def missing_required(self) -> list[str]:
        return [
            name
            for name in REQUIRED_BACKTEST_ARTIFACTS
            if self.artifact_path(name) is None or not self.artifact_path(name).exists()
        ]

    def present_optional_artifacts(self) -> list[str]:
        present: list[str] = []
        for name in OPTIONAL_BACKTEST_ARTIFACTS:
            path = self.artifact_path(name)
            if path is not None and path.exists():
                present.append(name)
        return present

    def validation_errors(
        self,
        *,
        expected_run_id: str = "",
        expected_manifest_id: str = "",
        require_manifest_id: bool = False,
    ) -> list[str]:
        errors: list[str] = []
        if expected_run_id and self.run_id != expected_run_id:
            errors.append(f"artifact index run_id mismatch: {self.run_id!r} != {expected_run_id!r}")
        if expected_manifest_id:
            if require_manifest_id and not self.manifest_id:
                errors.append("artifact index manifest_id is required for optimizer runs")
            elif self.manifest_id and self.manifest_id != expected_manifest_id:
                errors.append(
                    "artifact index manifest_id mismatch: "
                    f"{self.manifest_id!r} != {expected_manifest_id!r}"
                )
        if missing := self.missing_required():
            errors.append(f"missing required artifacts: {', '.join(missing)}")
        root = Path(self.artifact_root).resolve()
        outside: list[str] = []
        malformed: list[str] = []
        for name in [*REQUIRED_BACKTEST_ARTIFACTS, *self.present_optional_artifacts()]:
            path = self.artifact_path(name)
            if path is None:
                continue
            try:
                path.resolve().relative_to(root)
            except (OSError, ValueError):
                outside.append(name)
            if not path.exists():
                continue
            if name in JSON_ARTIFACTS:
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    malformed.append(name)
            if name in JSONL_ARTIFACTS:
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            json.loads(line)
                except Exception:
                    malformed.append(name)
        if outside:
            errors.append(f"artifact paths outside artifact_root: {', '.join(outside)}")
        if malformed:
            errors.append(f"malformed required artifacts: {', '.join(malformed)}")
        return errors


class BacktestExitStatus(BaseModel):
    exit_code: int = 0
    timed_out: bool = False
    error: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None


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
    def _validate_dates(self) -> FoldSpec:
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
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_manifest(self) -> FoldManifest:
        if len(self.folds) != 2:
            raise ValueError("monthly phased-auto requires exactly two purged in-sample folds")
        if self.selection_oos_start <= self.in_sample_end:
            raise ValueError("selection-OOS must start after the in-sample window")
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
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_plan(self) -> OptimizerExperimentPlan:
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    manifest_version: str = "candidate_workspace_manifest_v1"

    @model_validator(mode="after")
    def _validate_workspace(self) -> CandidateWorkspaceManifest:
        root = Path(self.workspace_root)
        workspace = Path(self.workspace_path)
        workspace.resolve().relative_to(root.resolve())
        if self.workspace_key != sanitize_workspace_key(self.workspace_key):
            raise ValueError("workspace_key is not sanitized")
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_attempt(self) -> CandidateAttemptRecord:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be >= 1")
        if self.retry_attempt < 0:
            raise ValueError("retry_attempt cannot be negative")
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
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_rerank(self) -> ConfirmatoryRerank:
        if bool(self.adopted_candidate_id) == bool(self.no_adoption_reason):
            raise ValueError(
                "confirmatory rerank must have exactly one adoption or no-adoption reason"
            )
        if self.repair_triggered and self.primary_source != MonthlyCandidateSource.SMOKE_REPAIR:
            raise ValueError(
                "repair-triggered confirmatory rerank must center a smoke/OOS-repair candidate"
            )
        if not self.repair_triggered and self.primary_source != MonthlyCandidateSource.PHASED_AUTO:
            raise ValueError("non-repair confirmatory rerank must center the phased-auto winner")
        if self.adopted_candidate_id and self.adopted_candidate_id not in set(
            self.compared_candidate_ids
        ):
            raise ValueError("adopted candidate must be listed in compared_candidate_ids")
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
    next_round_id: str = ""
    adopted_candidate_id: str = ""
    no_adoption_reason: str = ""
    records: list[RoundManifestRecord] = Field(default_factory=list)
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    manifest_version: str = "rounds_manifest_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_rounds(self) -> RoundsManifest:
        if bool(self.adopted_candidate_id) == bool(self.no_adoption_reason):
            raise ValueError("rounds manifest must have exactly one adoption or no-adoption reason")
        if self.adopted_candidate_id and not self.next_round_id:
            raise ValueError("adopted rounds manifest requires next_round_id")
        return self


class DecisionParityCheck(BaseModel):
    dimension: str
    status: DecisionParityStatus = DecisionParityStatus.INSUFFICIENT_DATA
    match_rate: float = 0.0
    mismatch_count: int = 0
    notes: str = ""
    evidence_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> DecisionParityCheck:
        self.dimension = self.dimension.strip().lower()
        self.match_rate = max(0.0, min(float(self.match_rate), 1.0))
        if self.mismatch_count < 0:
            raise ValueError("mismatch_count cannot be negative")
        return self


class DecisionParityReport(BaseModel):
    run_id: str
    candidate_id: str
    strategy_plugin_id: str = ""
    live_repo_commit_sha: str = ""
    backtest_adapter_commit_sha: str = ""
    checks: list[DecisionParityCheck]
    status: DecisionParityStatus = DecisionParityStatus.INSUFFICIENT_DATA
    min_required_match_rate: float = 1.0
    evidence_paths: list[str] = Field(default_factory=list)
    report_version: str = "decision_parity_report_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_report(self) -> DecisionParityReport:
        missing = DECISION_PARITY_DIMENSIONS - {check.dimension for check in self.checks}
        if missing:
            raise ValueError(
                "decision parity report missing dimensions: " + ", ".join(sorted(missing))
            )
        self.min_required_match_rate = max(0.0, min(float(self.min_required_match_rate), 1.0))
        if self.status == DecisionParityStatus.PASS:
            missing_required = [
                name
                for name in (
                    "strategy_plugin_id",
                    "live_repo_commit_sha",
                    "backtest_adapter_commit_sha",
                    "evidence_paths",
                )
                if not getattr(self, name)
            ]
            if missing_required:
                raise ValueError(
                    "decision parity pass missing required fields: "
                    + ", ".join(sorted(missing_required))
                )
            failed = [
                check.dimension
                for check in self.checks
                if check.status != DecisionParityStatus.PASS
                or check.match_rate < self.min_required_match_rate
                or check.mismatch_count != 0
            ]
            if failed:
                raise ValueError(
                    "decision parity marked pass but dimensions failed: "
                    + ", ".join(sorted(failed))
                )
            missing_evidence = [
                check.dimension for check in self.checks if not check.evidence_paths
            ]
            if missing_evidence:
                raise ValueError(
                    "decision parity pass missing dimension evidence: "
                    + ", ".join(sorted(missing_evidence))
                )
        return self

    @property
    def eligible_for_structural_approval(self) -> bool:
        return self.status == DecisionParityStatus.PASS


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


def sanitize_workspace_key(value: str) -> str:
    raw = str(value or "").strip()
    allowed = set(string.ascii_letters + string.digits + "-_")
    safe = "".join(ch if ch in allowed else "_" for ch in raw)
    safe = safe.strip("._-")
    if not safe:
        safe = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12] if raw else "candidate"
    if len(safe) > 64:
        safe = f"{safe[:51]}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"
    return safe
