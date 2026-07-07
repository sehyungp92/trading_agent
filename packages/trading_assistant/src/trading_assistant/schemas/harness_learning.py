"""Schemas for offline harness evaluation artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HarnessExecutionMode(str, Enum):
    """How an executable benchmark case should be replayed."""

    DETERMINISTIC_ONLY = "deterministic_only"
    RECORDED_AGENT = "recorded_agent"
    PROVIDER_SANDBOX = "provider_sandbox"


class HarnessExecutionInput(BaseModel):
    """Frozen input needed to replay a benchmark case through harness boundaries."""

    case_id: str
    workflow: str = ""
    bot_id: str = ""
    strategy_id: str = ""
    source_run_id: str = ""
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    prompt_inputs: dict[str, Any] = Field(default_factory=dict)
    retrieval_profile: dict[str, Any] = Field(default_factory=dict)
    allowed_artifacts: list[str] = Field(default_factory=list)
    expected_behavior: str = ""
    forbidden_behavior: list[str] = Field(default_factory=list)
    recorded_response: str = ""
    recorded_structured_output: dict[str, Any] = Field(default_factory=dict)
    materialized_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HarnessExecutionOutput(BaseModel):
    """Result of replaying one benchmark case through parser/validator boundaries."""

    case_id: str
    variant_name: str = "baseline"
    execution_mode: HarnessExecutionMode = HarnessExecutionMode.DETERMINISTIC_ONLY
    prompt_package_hash: str = ""
    variant_config_hash: str = ""
    changed_components: list[str] = Field(default_factory=list)
    retrieved_card_ids: list[str] = Field(default_factory=list)
    retrieved_playbook_ids: list[str] = Field(default_factory=list)
    raw_response_path: str = ""
    raw_response_fixture_id: str = ""
    parse_success: bool = False
    fallback_parse_used: bool = False
    dropped_counts: dict[str, int] = Field(default_factory=dict)
    approved_item_count: int = 0
    blocked_item_count: int = 0
    validator_notes: str = ""
    evidence_refs_used: list[str] = Field(default_factory=list)
    invalid_evidence_refs: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: int = 0
    governance_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HarnessVariant(BaseModel):
    """A named harness configuration variant for offline evaluation."""

    name: str
    prompt_patch: str = ""
    retrieval_mode: str = "baseline"
    validator_profile: str = "baseline"
    route_profile: str = "configured"
    enabled: bool = True
    anti_overfitting_note: str = ""
    complexity_score: float = 1.0
    requires_execution: bool = False

    @property
    def changed_components(self) -> list[str]:
        """Return the harness components changed from the reference baseline."""
        changed: list[str] = []
        if self.prompt_patch:
            changed.append("prompt_patch")
        if self.retrieval_mode != "baseline":
            changed.append("retrieval")
        if self.validator_profile != "baseline":
            changed.append("validator")
        if self.route_profile != "configured":
            changed.append("provider_route")
        return changed


class HarnessCaseResult(BaseModel):
    """Scored result for one executable benchmark case."""

    case_id: str
    source: str = ""
    severity: str = ""
    score: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    governance_failures: list[str] = Field(default_factory=list)
    evidence_refs_checked: list[str] = Field(default_factory=list)
    rationale: str = ""


class HarnessEvalResult(BaseModel):
    """Offline benchmark comparison result for a harness variant."""

    variant_name: str
    benchmark_count: int = 0
    execution_case_count: int = 0
    fallback_case_count: int = 0
    aggregate_score: float = 0.0
    baseline_score: float | None = None
    score_delta: float | None = None
    per_source: dict[str, float] = Field(default_factory=dict)
    per_metric: dict[str, float] = Field(default_factory=dict)
    case_results: list[HarnessCaseResult] = Field(default_factory=list)
    governance_failures: list[str] = Field(default_factory=list)
    kept: bool = False
    rationale: str = ""
    program_ref: str = "memory/policies/v1/harness_program.md"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HarnessExperimentLedgerEntry(BaseModel):
    """Durable keep/discard record for a candidate harness experiment."""

    experiment_id: str
    variant_name: str
    baseline_variant_name: str = "baseline"
    hypothesis: str = ""
    changed_components: list[str] = Field(default_factory=list)
    benchmark_count: int = 0
    execution_case_count: int = 0
    fallback_case_count: int = 0
    aggregate_score: float = 0.0
    baseline_score: float = 0.0
    score_delta: float = 0.0
    kept: bool = False
    discard_reason: str = ""
    governance_regressions: list[str] = Field(default_factory=list)
    anti_overfitting_assessment: str = ""
    future_warning_tags: list[str] = Field(default_factory=list)
    program_ref: str = "memory/policies/v1/harness_program.md"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
