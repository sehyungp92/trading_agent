"""Monthly candidate generation and approval evidence schemas."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from trading_assistant.schemas.assistant_lineage import AssistantLineage, assistant_lineage_from_fields
from trading_assistant.schemas.objective_weights import (
    IMMUTABLE_MONTHLY_OBJECTIVE_VERSION,
    OBJECTIVE_WEIGHTS_VERSION,
)


class MonthlyCandidateSource(str, Enum):
    SMOKE_REPAIR = "smoke_repair"
    PHASED_AUTO = "phased_auto"
    MODEL_REVIEW = "model_review"
    UNKNOWN = "unknown"


class MonthlyRiskClassification(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonthlyCandidateDecision(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    EXPERIMENT = "experiment"
    DEFER = "defer"


class MonthlyGateSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class MonthlyGateCheck(BaseModel):
    name: str
    passed: bool
    severity: MonthlyGateSeverity = MonthlyGateSeverity.HARD
    reason: str = ""
    evidence_paths: list[str] = Field(default_factory=list)


class MonthlyCandidateGateReport(BaseModel):
    candidate_id: str
    source: MonthlyCandidateSource = MonthlyCandidateSource.UNKNOWN
    passed: bool = False
    checks: list[MonthlyGateCheck] = Field(default_factory=list)
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    effective_objective_version: str = ""
    objective_profile_id: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive_passed(self) -> "MonthlyCandidateGateReport":
        self.passed = bool(self.checks) and all(
            check.passed or check.severity != MonthlyGateSeverity.HARD
            for check in self.checks
        )
        return self

    @property
    def blocking_reasons(self) -> list[str]:
        return [
            check.reason or check.name
            for check in self.checks
            if not check.passed and check.severity == MonthlyGateSeverity.HARD
        ]


class MonthlyImprovementCandidate(BaseModel):
    """A replay-backed candidate emitted by smoke-repair or phased-auto."""

    candidate_id: str = ""
    source: MonthlyCandidateSource = MonthlyCandidateSource.UNKNOWN
    bot_id: str = ""
    strategy_id: str = ""
    family: str = ""
    title: str = ""
    description: str = ""
    decision: MonthlyCandidateDecision = MonthlyCandidateDecision.REPAIR
    change_kind: str = "parameter_change"
    risk_classification: MonthlyRiskClassification = MonthlyRiskClassification.MEDIUM
    objective_score: float = 0.0
    baseline_score: float = 0.0
    objective_delta: float = 0.0
    objective_deltas: dict[str, float] = Field(default_factory=dict)
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    effective_objective_version: str = ""
    immutable_objective_version: str = ""
    objective_profile_id: str = ""
    objective_profile_family: str = ""
    objective_profile_scope: str = ""
    score_component_cap: int = 0
    immutable_score: dict[str, Any] = Field(default_factory=dict)
    deterministic_gate_inputs: dict[str, Any] = Field(default_factory=dict)
    evidence_paths: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    param_changes: list[dict[str, Any]] = Field(default_factory=list)
    file_changes: list[dict[str, Any]] = Field(default_factory=list)
    proposed_changes: list[dict[str, Any]] = Field(default_factory=list)
    planned_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    replay_or_experiment_plan: str = ""
    rollback_plan: str = ""
    candidate_workspace_key: str = ""
    candidate_workspace_path: str = ""
    candidate_attempt_id: str = ""
    candidate_attempt_status: str = ""
    retry_attempt: int = 0
    retry_reason: str = ""
    stall_timeout_seconds: int = 0
    run_id: str = ""
    manifest_id: str = ""
    round_id: str = ""
    prior_round_id: str = ""
    next_round_id: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    assistant_lineage: AssistantLineage = Field(default_factory=AssistantLineage)
    code_sha: str = ""
    backtest_repo_commit_sha: str = ""
    live_trading_repo_commit_sha: str = ""
    control_plane_commit_sha: str = ""
    workflow_contract_path: str = ""
    workflow_contract_version: str = ""
    live_repo_patch_path: str = ""
    backtest_adapter_patch_path: str = ""
    config_patch_path: str = ""
    decision_parity_report_path: str = ""
    fold_manifest_path: str = ""
    rounds_manifest_path: str = ""
    end_of_round_diagnostics_path: str = ""
    confirmatory_rerank_path: str = ""
    optimizer_stage: str = ""
    score_component_count: int = 0
    max_workers: int = 0
    is_window: str = ""
    selection_oos_window: str = ""
    checkpoint_path: str = ""
    source_weekly_signal_ids: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        *,
        bot_id: str = "",
        strategy_id: str = "",
        default_source: MonthlyCandidateSource = MonthlyCandidateSource.UNKNOWN,
    ) -> "MonthlyImprovementCandidate":
        payload = dict(raw)
        source = _coerce_source(payload.get("source") or payload.get("candidate_source") or payload.get("mode"), default_source)
        risk = _coerce_risk(payload.get("risk_classification") or payload.get("risk_class") or payload.get("risk"))
        gates = payload.get("deterministic_gate_inputs") or payload.get("gate_inputs") or payload.get("gates") or {}
        if not isinstance(gates, dict):
            gates = {}

        objective_deltas = payload.get("objective_deltas") or payload.get("objective_impact") or {}
        if not isinstance(objective_deltas, dict):
            objective_deltas = {}
        objective_deltas = {
            str(key): float(value)
            for key, value in objective_deltas.items()
            if _is_number(value)
        }

        objective_score = _float(payload.get("objective_score") or payload.get("composite_score"))
        baseline_score = _float(payload.get("baseline_score") or payload.get("incumbent_score"))
        objective_delta = _float(payload.get("objective_delta"))
        if objective_delta == 0.0 and objective_score and baseline_score:
            objective_delta = objective_score - baseline_score
        if objective_delta == 0.0:
            objective_delta = _first_delta(objective_deltas)

        immutable_score = _dict_value(payload.get("immutable_score"))
        objective_breakdown = _dict_value(payload.get("objective_breakdown"))
        if not immutable_score and objective_breakdown:
            immutable_score = _score_payload_from_breakdown(objective_breakdown)
        profile_payload = _dict_value(immutable_score.get("profile"))
        objective_profile_id = str(
            payload.get("objective_profile_id")
            or immutable_score.get("profile_id")
            or profile_payload.get("profile_id")
            or ""
        )
        effective_objective_version = str(
            payload.get("effective_objective_version")
            or payload.get("immutable_objective_version")
            or immutable_score.get("profile_version")
            or immutable_score.get("version")
            or profile_payload.get("version")
            or ""
        )
        if not effective_objective_version and objective_profile_id:
            effective_objective_version = IMMUTABLE_MONTHLY_OBJECTIVE_VERSION
        immutable_objective_version = str(
            payload.get("immutable_objective_version")
            or immutable_score.get("profile_version")
            or effective_objective_version
            or ""
        )

        candidate = cls(
            candidate_id=str(payload.get("candidate_id") or payload.get("id") or ""),
            source=source,
            bot_id=str(payload.get("bot_id") or bot_id),
            strategy_id=str(payload.get("strategy_id") or strategy_id),
            family=str(payload.get("family") or payload.get("candidate_family") or ""),
            title=str(payload.get("title") or payload.get("name") or "Monthly improvement candidate"),
            description=str(payload.get("description") or payload.get("summary") or ""),
            decision=_coerce_decision(payload.get("decision") or payload.get("status")),
            change_kind=str(payload.get("change_kind") or payload.get("proposal_type") or "parameter_change"),
            risk_classification=risk,
            objective_score=objective_score,
            baseline_score=baseline_score,
            objective_delta=objective_delta,
            objective_deltas=objective_deltas,
            objective_version=str(payload.get("objective_version") or OBJECTIVE_WEIGHTS_VERSION),
            effective_objective_version=effective_objective_version,
            immutable_objective_version=immutable_objective_version,
            objective_profile_id=objective_profile_id,
            objective_profile_family=str(
                payload.get("objective_profile_family")
                or immutable_score.get("family")
                or profile_payload.get("family")
                or ""
            ),
            objective_profile_scope=str(
                payload.get("objective_profile_scope")
                or immutable_score.get("scope")
                or profile_payload.get("scope")
                or ""
            ),
            score_component_cap=_int(
                payload.get("score_component_cap")
                or payload.get("score_component_count")
                or immutable_score.get("score_component_cap")
                or profile_payload.get("component_cap")
            ),
            immutable_score=immutable_score,
            deterministic_gate_inputs={**gates, **_top_level_gate_inputs(payload)},
            evidence_paths=_string_list(payload.get("evidence_paths")),
            artifact_paths=_string_list(payload.get("artifact_paths")),
            param_changes=_dict_list(payload.get("param_changes")),
            file_changes=_dict_list(payload.get("file_changes")),
            proposed_changes=_dict_list(payload.get("proposed_changes")),
            planned_files=_string_list(payload.get("planned_files") or payload.get("affected_files")),
            acceptance_criteria=_string_list(payload.get("acceptance_criteria")),
            replay_or_experiment_plan=str(payload.get("replay_or_experiment_plan") or payload.get("experiment_plan") or ""),
            rollback_plan=str(payload.get("rollback_plan") or ""),
            candidate_workspace_key=str(payload.get("candidate_workspace_key") or ""),
            candidate_workspace_path=str(payload.get("candidate_workspace_path") or ""),
            candidate_attempt_id=str(payload.get("candidate_attempt_id") or ""),
            candidate_attempt_status=str(payload.get("candidate_attempt_status") or ""),
            retry_attempt=_int(payload.get("retry_attempt")),
            retry_reason=str(payload.get("retry_reason") or ""),
            stall_timeout_seconds=_int(payload.get("stall_timeout_seconds")),
            run_id=str(payload.get("run_id") or ""),
            manifest_id=str(payload.get("manifest_id") or ""),
            round_id=str(payload.get("round_id") or ""),
            prior_round_id=str(payload.get("prior_round_id") or ""),
            next_round_id=str(payload.get("next_round_id") or ""),
            deployment_id=str(payload.get("deployment_id") or ""),
            parameter_set_id=str(payload.get("parameter_set_id") or ""),
            proposal_ids=_string_list(payload.get("proposal_ids") or payload.get("proposal_id")),
            suggestion_ids=_string_list(payload.get("suggestion_ids") or payload.get("suggestion_id")),
            hypothesis_ids=_string_list(payload.get("hypothesis_ids") or payload.get("hypothesis_id")),
            assistant_lineage=AssistantLineage.model_validate(payload.get("assistant_lineage") or {}),
            code_sha=str(payload.get("code_sha") or ""),
            backtest_repo_commit_sha=str(payload.get("backtest_repo_commit_sha") or ""),
            live_trading_repo_commit_sha=str(payload.get("live_trading_repo_commit_sha") or ""),
            control_plane_commit_sha=str(payload.get("control_plane_commit_sha") or ""),
            workflow_contract_path=str(payload.get("workflow_contract_path") or ""),
            workflow_contract_version=str(payload.get("workflow_contract_version") or ""),
            live_repo_patch_path=str(payload.get("live_repo_patch_path") or ""),
            backtest_adapter_patch_path=str(payload.get("backtest_adapter_patch_path") or ""),
            config_patch_path=str(payload.get("config_patch_path") or ""),
            decision_parity_report_path=str(payload.get("decision_parity_report_path") or ""),
            fold_manifest_path=str(payload.get("fold_manifest_path") or ""),
            rounds_manifest_path=str(payload.get("rounds_manifest_path") or ""),
            end_of_round_diagnostics_path=str(payload.get("end_of_round_diagnostics_path") or ""),
            confirmatory_rerank_path=str(payload.get("confirmatory_rerank_path") or ""),
            optimizer_stage=str(payload.get("optimizer_stage") or ""),
            score_component_count=_int(payload.get("score_component_count")),
            max_workers=_int(payload.get("max_workers")),
            is_window=str(payload.get("is_window") or payload.get("in_sample_window") or ""),
            selection_oos_window=str(payload.get("selection_oos_window") or ""),
            checkpoint_path=str(payload.get("checkpoint_path") or ""),
            source_weekly_signal_ids=_string_list(payload.get("source_weekly_signal_ids")),
            raw_payload=payload,
        )
        if not candidate.candidate_id:
            candidate.candidate_id = _candidate_id(candidate)
        candidate._derive_assistant_lineage()
        return candidate

    @model_validator(mode="after")
    def _derive_assistant_lineage(self) -> "MonthlyImprovementCandidate":
        derived = assistant_lineage_from_fields(
            weekly_signal_ids=self.source_weekly_signal_ids,
            proposal_ids=self.proposal_ids,
            suggestion_ids=self.suggestion_ids,
            hypothesis_ids=self.hypothesis_ids,
            parameter_set_id=self.parameter_set_id,
            deployment_id=self.deployment_id,
        )
        if not self.assistant_lineage.has_any():
            self.assistant_lineage = derived
        else:
            self.assistant_lineage = AssistantLineage.model_validate({
                **derived.model_dump(),
                **self.assistant_lineage.model_dump(),
                "weekly_signal_ids": [
                    *derived.weekly_signal_ids,
                    *self.assistant_lineage.weekly_signal_ids,
                ],
                "proposal_ids": [*derived.proposal_ids, *self.assistant_lineage.proposal_ids],
                "suggestion_ids": [*derived.suggestion_ids, *self.assistant_lineage.suggestion_ids],
                "hypothesis_ids": [*derived.hypothesis_ids, *self.assistant_lineage.hypothesis_ids],
            })
        return self


class MonthlyApprovalEvidencePacket(BaseModel):
    """Human-readable and machine-readable evidence for approval routing."""

    request_id: str = ""
    proposal_id: str = ""
    suggestion_id: str = ""
    assistant_lineage: AssistantLineage = Field(default_factory=AssistantLineage)
    candidate_id: str
    run_id: str
    run_month: str
    bot_id: str
    strategy_id: str
    strategy_change_record_id: str = ""
    title: str
    reason_for_change: str = ""
    incumbent_validation_summary: str = ""
    smoke_or_phased_evidence: str = ""
    objective_deltas: dict[str, float] = Field(default_factory=dict)
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    effective_objective_version: str = ""
    objective_profile_id: str = ""
    score_component_cap: int = 0
    latest_month_behavior: str = ""
    calibration_support: str = ""
    data_coverage_status: str = ""
    replay_parity_status: str = ""
    risk_classification: MonthlyRiskClassification = MonthlyRiskClassification.MEDIUM
    rollback_plan: str = ""
    artifact_paths: list[str] = Field(default_factory=list)
    model_review_path: str = ""
    model_review_validation_path: str = ""
    evidence_verification_path: str = ""
    evidence_verification_id: str = ""
    evidence_verification_verdict: str = ""
    human_summary: str = ""
    machine_readable_payload: dict[str, Any] = Field(default_factory=dict)
    approval_ready: bool = False
    approval_suppressed_reasons: list[str] = Field(default_factory=list)
    approval_packet_path: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive_assistant_lineage(self) -> "MonthlyApprovalEvidencePacket":
        derived = assistant_lineage_from_fields(
            proposal_ids=_string_list(self.proposal_id),
            suggestion_ids=_string_list(self.suggestion_id),
            strategy_change_record_ids=_string_list(self.strategy_change_record_id),
        )
        if not self.assistant_lineage.has_any():
            self.assistant_lineage = derived
        else:
            self.assistant_lineage = AssistantLineage.model_validate({
                **derived.model_dump(),
                **self.assistant_lineage.model_dump(),
                "proposal_ids": [*derived.proposal_ids, *self.assistant_lineage.proposal_ids],
                "suggestion_ids": [*derived.suggestion_ids, *self.assistant_lineage.suggestion_ids],
                "strategy_change_record_ids": [
                    *derived.strategy_change_record_ids,
                    *self.assistant_lineage.strategy_change_record_ids,
                ],
            })
        return self


class MonthlyCandidateProcessingResult(BaseModel):
    run_id: str
    run_month: str
    bot_id: str
    strategy_id: str
    selected_candidates: list[MonthlyImprovementCandidate] = Field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = Field(default_factory=list)
    gate_reports: list[MonthlyCandidateGateReport] = Field(default_factory=list)
    approval_packets: list[MonthlyApprovalEvidencePacket] = Field(default_factory=list)
    approval_request_ids: list[str] = Field(default_factory=list)
    gate_passed_candidate_count: int = 0
    approval_ready_candidate_count: int = 0
    candidate_summary_path: str = ""
    gate_report_path: str = ""
    approval_packet_paths: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    model_review_path: str = ""
    model_review_validation_path: str = ""
    evidence_verification_paths: list[str] = Field(default_factory=list)
    evidence_verification_verdicts: dict[str, str] = Field(default_factory=dict)
    model_review_valid: bool | None = None
    model_review_issues: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _coerce_source(value: Any, default: MonthlyCandidateSource) -> MonthlyCandidateSource:
    raw = str(value or "").strip().lower()
    aliases = {
        "smoke": MonthlyCandidateSource.SMOKE_REPAIR,
        "smoke_repair": MonthlyCandidateSource.SMOKE_REPAIR,
        "repair": MonthlyCandidateSource.SMOKE_REPAIR,
        "phased": MonthlyCandidateSource.PHASED_AUTO,
        "phased_auto": MonthlyCandidateSource.PHASED_AUTO,
        "auto": MonthlyCandidateSource.PHASED_AUTO,
        "model": MonthlyCandidateSource.MODEL_REVIEW,
        "model_review": MonthlyCandidateSource.MODEL_REVIEW,
    }
    return aliases.get(raw, default)


def _coerce_risk(value: Any) -> MonthlyRiskClassification:
    raw = str(value or "").strip().lower()
    try:
        return MonthlyRiskClassification(raw)
    except ValueError:
        return MonthlyRiskClassification.MEDIUM


def _coerce_decision(value: Any) -> MonthlyCandidateDecision:
    raw = str(value or "").strip().lower()
    try:
        return MonthlyCandidateDecision(raw)
    except ValueError:
        return MonthlyCandidateDecision.REPAIR


def _candidate_id(candidate: MonthlyImprovementCandidate) -> str:
    raw = "|".join([
        candidate.bot_id,
        candidate.strategy_id,
        candidate.source.value,
        candidate.effective_objective_version,
        candidate.objective_profile_id,
        candidate.family,
        candidate.title,
        str(candidate.objective_delta),
        ",".join(candidate.evidence_paths[:2]),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _first_delta(values: dict[str, float]) -> float:
    for key in ("canonical", "composite", "latest_month_oos", "latest_month", "calibration"):
        if key in values:
            return values[key]
    return next(iter(values.values()), 0.0)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _float(value: Any) -> float:
    if _is_number(value):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    return []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list | tuple):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _score_payload_from_breakdown(breakdown: dict[str, Any]) -> dict[str, Any]:
    profile = _dict_value(breakdown.get("profile"))
    payload = {
        "profile_id": str(breakdown.get("objective_profile_id") or ""),
        "profile_version": str(
            breakdown.get("effective_objective_version")
            or breakdown.get("immutable_objective_version")
            or ""
        ),
        "objective_score": _float(breakdown.get("objective_score")),
        "rejected": bool(breakdown.get("hard_rejected", False)),
        "reject_reasons": breakdown.get("reject_reasons", []),
        "score_component_cap": _int(breakdown.get("score_component_cap")),
        "renormalized_components": breakdown.get("renormalized_components", []),
    }
    if profile:
        payload["profile"] = profile
    return payload


def _top_level_gate_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    gate_keys = {
        "latest_month_oos_improvement",
        "latest_month_oos_delta",
        "latest_month_objective_delta",
        "calibration_support",
        "calibration_supported",
        "calibration_objective_delta",
        "leakage_passed",
        "no_leakage",
        "sufficient_trade_count",
        "trade_count",
        "sparse_sample_classification",
        "cost_gate_passed",
        "realistic_costs_passed",
        "drawdown_gate_passed",
        "max_drawdown_delta_pct",
        "outlier_concentration_passed",
        "outlier_win_concentration",
        "risk_constraints_passed",
        "portfolio_risk_constraints_passed",
        "runner_contract_version",
        "source_runner_contract_version",
        "phase4_sequence_valid",
        "round_n_plus_1_adopted",
        "confirmatory_follow_up_passed",
        "end_of_round_diagnostics_saved",
        "live_backtest_parity_aligned",
        "fold_support_passed",
        "purged_fold_support",
        "positive_fold_support",
        "folds_positive",
        "score_component_count",
        "source_weekly_signal_ids",
        "manifest_id",
    }
    return {key: payload[key] for key in gate_keys if key in payload}
