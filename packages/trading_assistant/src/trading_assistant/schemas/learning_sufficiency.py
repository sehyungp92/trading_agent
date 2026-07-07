"""Learning-sufficiency authority schema for monthly instrumentation."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class LearningEligibility(str, Enum):
    LEARNING_AUTHORITATIVE = "learning_authoritative"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    INSUFFICIENT_LINEAGE = "insufficient_lineage"
    INSUFFICIENT_JOINS = "insufficient_joins"
    INSUFFICIENT_DENOMINATORS = "insufficient_denominators"
    INSUFFICIENT_AFTER_COSTS = "insufficient_after_costs"
    INSUFFICIENT_OUTCOMES = "insufficient_outcomes"
    INSUFFICIENT_SHADOW_EVIDENCE = "insufficient_shadow_evidence"


class CoverageStatus(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class LearningCapabilityAuthority(str, Enum):
    LEARNING_AUTHORITATIVE = "learning_authoritative"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class RuntimeEvidenceSupportState(str, Enum):
    UNSUPPORTED = "unsupported"
    SUPPORTED_BUT_UNOBSERVED = "supported_but_unobserved"
    OBSERVED = "observed"


class ExpectedLearningValue(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CoverageCheck(BaseModel):
    """Deterministic coverage result for one learning prerequisite."""

    check_id: str
    status: CoverageStatus = CoverageStatus.UNKNOWN
    observed_count: int = 0
    required_count: int = 0
    coverage_ratio: float = 0.0
    min_required_ratio: float = 1.0
    required_fields: list[str] = Field(default_factory=list)
    observed_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    missing_event_types: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    reason: str = ""
    declared_not_applicable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "CoverageCheck":
        if self.observed_count < 0:
            raise ValueError("observed_count cannot be negative")
        if self.required_count < 0:
            raise ValueError("required_count cannot be negative")
        if self.required_count and not self.coverage_ratio:
            self.coverage_ratio = self.observed_count / self.required_count
        self.coverage_ratio = max(0.0, min(float(self.coverage_ratio), 1.0))
        self.min_required_ratio = max(0.0, min(float(self.min_required_ratio), 1.0))
        self.required_fields = _dedupe(self.required_fields)
        self.observed_fields = _dedupe(self.observed_fields)
        if not self.missing_fields and self.required_fields:
            observed = set(self.observed_fields)
            self.missing_fields = [field for field in self.required_fields if field not in observed]
        else:
            self.missing_fields = _dedupe(self.missing_fields)
        self.missing_event_types = _dedupe(self.missing_event_types)
        self.evidence_paths = _dedupe(self.evidence_paths)
        if self.declared_not_applicable:
            self.status = CoverageStatus.NOT_APPLICABLE
        elif self.status == CoverageStatus.UNKNOWN:
            if self.required_count == 0 and not self.required_fields and not self.missing_event_types:
                self.status = CoverageStatus.UNKNOWN
            elif (
                self.coverage_ratio >= self.min_required_ratio
                and not self.missing_fields
                and not self.missing_event_types
            ):
                self.status = CoverageStatus.PASS
            elif self.observed_count > 0 or self.observed_fields:
                self.status = CoverageStatus.PARTIAL
            else:
                self.status = CoverageStatus.MISSING
        return self

    @property
    def satisfies_learning_authority(self) -> bool:
        return self.status == CoverageStatus.PASS or (
            self.status == CoverageStatus.NOT_APPLICABLE and self.declared_not_applicable
        )


class LearningCapabilityStatus(BaseModel):
    """Capability-level authority derived from one or more coverage checks."""

    capability_id: str
    status: LearningCapabilityAuthority = LearningCapabilityAuthority.BLOCKED
    required_checks: list[str] = Field(default_factory=list)
    satisfied_checks: list[str] = Field(default_factory=list)
    blocking_checks: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "LearningCapabilityStatus":
        self.required_checks = _dedupe(self.required_checks)
        self.satisfied_checks = _dedupe(self.satisfied_checks)
        self.blocking_checks = _dedupe(self.blocking_checks)
        self.evidence_paths = _dedupe(self.evidence_paths)
        self.blocking_reasons = _dedupe(self.blocking_reasons)
        if self.status == LearningCapabilityAuthority.LEARNING_AUTHORITATIVE and self.blocking_checks:
            raise ValueError("learning-authoritative capability cannot list blocking_checks")
        return self


class LearningGap(BaseModel):
    """Actionable instrumentation gap ranked by blocked learning value."""

    gap_id: str = ""
    bot_id: str = ""
    strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    event_type: str = ""
    missing_field: str = ""
    blocked_learning_capability: str
    expected_learning_value: ExpectedLearningValue = ExpectedLearningValue.MEDIUM
    frequency: int = 0
    evidence_paths: list[str] = Field(default_factory=list)
    remediation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "LearningGap":
        if self.frequency < 0:
            raise ValueError("frequency cannot be negative")
        self.evidence_paths = _dedupe(self.evidence_paths)
        if not self.gap_id:
            raw = "|".join([
                self.bot_id,
                self.strategy_id,
                self.family_id,
                self.portfolio_id,
                self.event_type,
                self.missing_field,
                self.blocked_learning_capability,
            ])
            self.gap_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self


class RuntimeEvidenceSupport(BaseModel):
    """Configured runtime support plus observation state for one evidence class."""

    evidence_class: str
    support_state: RuntimeEvidenceSupportState = RuntimeEvidenceSupportState.UNSUPPORTED
    configured_event_types: list[str] = Field(default_factory=list)
    observed_event_count: int = 0
    support_source_paths: list[str] = Field(default_factory=list)
    observed_evidence_paths: list[str] = Field(default_factory=list)
    event_value_classifications: dict[str, str] = Field(default_factory=dict)
    declared_unavailable: bool = False
    reason: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "RuntimeEvidenceSupport":
        self.configured_event_types = _dedupe(self.configured_event_types)
        self.support_source_paths = _dedupe(self.support_source_paths)
        self.observed_evidence_paths = _dedupe(self.observed_evidence_paths)
        self.event_value_classifications = {
            str(event_type): str(value_class)
            for event_type, value_class in sorted(self.event_value_classifications.items())
            if str(event_type) and str(value_class)
        }
        self.observed_event_count = max(0, self.observed_event_count)
        if self.support_state == RuntimeEvidenceSupportState.OBSERVED and self.observed_event_count <= 0:
            self.support_state = RuntimeEvidenceSupportState.SUPPORTED_BUT_UNOBSERVED
        return self


class LearningSufficiencyManifest(BaseModel):
    """Authority artifact for whether a bot/strategy/month can support learning."""

    manifest_id: str = ""
    bot_id: str
    strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    run_month: str
    window_start: date
    window_end: date
    telemetry_manifest_path: str = ""
    telemetry_authoritative_eligibility: str = ""
    eligibility: LearningEligibility = LearningEligibility.DIAGNOSTICS_ONLY
    event_counts_by_type: dict[str, int] = Field(default_factory=dict)
    required_event_coverage: dict[str, CoverageCheck] = Field(default_factory=dict)
    lineage_coverage: CoverageCheck = Field(
        default_factory=lambda: CoverageCheck(check_id="lineage_coverage")
    )
    join_coverage: dict[str, CoverageCheck] = Field(default_factory=dict)
    denominator_coverage: dict[str, CoverageCheck] = Field(default_factory=dict)
    after_cost_coverage: CoverageCheck = Field(
        default_factory=lambda: CoverageCheck(check_id="after_cost_coverage")
    )
    counterfactual_coverage: CoverageCheck = Field(
        default_factory=lambda: CoverageCheck(check_id="counterfactual_coverage")
    )
    proposal_trace_coverage: CoverageCheck = Field(
        default_factory=lambda: CoverageCheck(check_id="proposal_trace_coverage")
    )
    deployment_metadata_coverage: CoverageCheck = Field(
        default_factory=lambda: CoverageCheck(check_id="deployment_metadata_coverage")
    )
    runtime_evidence_support: dict[str, RuntimeEvidenceSupport] = Field(default_factory=dict)
    capability_status: dict[str, LearningCapabilityStatus] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    known_gaps: list[LearningGap] = Field(default_factory=list)
    blocked_learning_capabilities: list[str] = Field(default_factory=list)
    supported_learning_capabilities: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    total_events: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    manifest_version: str = "learning_sufficiency_manifest_v1"

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
    def _normalize(self) -> "LearningSufficiencyManifest":
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        if not self.total_events:
            self.total_events = sum(self.event_counts_by_type.values())
        for event_type, count in self.event_counts_by_type.items():
            if count < 0:
                raise ValueError(f"event count cannot be negative for {event_type!r}")
        self.blocked_learning_capabilities = _dedupe(self.blocked_learning_capabilities)
        self.supported_learning_capabilities = _dedupe(self.supported_learning_capabilities)
        for capability_id, status in self.capability_status.items():
            if status.status == LearningCapabilityAuthority.LEARNING_AUTHORITATIVE:
                self.supported_learning_capabilities = _append_unique(
                    self.supported_learning_capabilities,
                    capability_id,
                )
            elif status.status == LearningCapabilityAuthority.BLOCKED:
                self.blocked_learning_capabilities = _append_unique(
                    self.blocked_learning_capabilities,
                    capability_id,
                )
        self.evidence_paths = _dedupe([
            *self.evidence_paths,
            *([self.telemetry_manifest_path] if self.telemetry_manifest_path else []),
            *[
                path
                for support in self.runtime_evidence_support.values()
                for path in [*support.support_source_paths, *support.observed_evidence_paths]
            ],
        ])
        if not self.manifest_id:
            raw = "|".join([
                self.bot_id,
                self.strategy_id,
                self.family_id,
                self.portfolio_id,
                self.run_month,
                self.window_start.isoformat(),
                self.window_end.isoformat(),
                str(self.total_events),
            ])
            self.manifest_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self

    @property
    def is_learning_authoritative(self) -> bool:
        return self.eligibility == LearningEligibility.LEARNING_AUTHORITATIVE


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _append_unique(values: list[str], value: str) -> list[str]:
    text = str(value or "")
    if text and text not in values:
        values.append(text)
    return values
