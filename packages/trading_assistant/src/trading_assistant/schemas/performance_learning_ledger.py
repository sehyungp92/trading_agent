"""Compact performance-learning projection records.

These records are evidence-memory projections. They preserve links and learning
authority from source ledgers, but they do not approve, deploy, or rewrite any
trading source of truth.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PerformanceRecordType(str, Enum):
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"


class SourceCadence(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    FOLLOW_UP = "follow_up"
    HARNESS = "harness"


class LearningLayer(str, Enum):
    SENSOR_CONTEXT = "sensor_context"
    BOUNDED_SEARCH_PRIOR = "bounded_search_prior"
    TRADING_AUTHORITY = "trading_authority"
    PERSISTENCE_CONFIRMATION = "persistence_confirmation"
    HARNESS_META_LEARNING = "harness_meta_learning"


class AuthorityLevel(str, Enum):
    DIAGNOSTIC = "diagnostic"
    ADVISORY_PRIOR = "advisory_prior"
    MONTHLY_REPLAY_AUTHORITY = "monthly_replay_authority"
    EARLY_WARNING = "early_warning"
    PERSISTENCE_CONFIRMATION = "persistence_confirmation"
    BENCHMARK_ONLY = "benchmark_only"


class DecisionStage(str, Enum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    MEASURED = "measured"
    FOLLOW_UP = "follow_up"
    ROLLBACK = "rollback"
    REJECTED = "rejected"


class PerformanceMetricDeltas(BaseModel):
    objective: float | None = None
    return_: float | None = Field(default=None, alias="return")
    drawdown: float | None = None
    turnover: float | None = None
    cost: float | None = None
    slippage: float | None = None
    confidence: float | None = None

    model_config = {"populate_by_name": True}

    def has_any(self) -> bool:
        return any(value is not None for value in self.model_dump().values())


class StrategySliceContext(BaseModel):
    regime: str = ""
    symbol: str = ""
    session: str = ""
    side: str = ""
    liquidity: str = ""
    sample_size: int | None = None
    trade_count: int | None = None
    cost_bps: float | None = None
    failure_mode: str = ""

    def has_any(self) -> bool:
        return any(value not in ("", None, [], {}) for value in self.model_dump().values())


class PortfolioInteractionContext(BaseModel):
    allocation_weights: dict[str, float] = Field(default_factory=dict)
    risk_budgets: dict[str, float] = Field(default_factory=dict)
    exposure: dict[str, float] = Field(default_factory=dict)
    correlation: dict[str, float] = Field(default_factory=dict)
    drawdown_overlap: dict[str, float] = Field(default_factory=dict)
    crowding: str = ""
    cannibalization: str = ""
    marginal_contribution: dict[str, float] = Field(default_factory=dict)
    concentration: str = ""
    liquidity_constraints: list[str] = Field(default_factory=list)

    def has_any(self) -> bool:
        return any(value not in ("", None, [], {}) for value in self.model_dump().values())


class IntendedLearningEffects(BaseModel):
    outcome_prior_update: str = ""
    search_allocation_change: str = ""
    evidence_gate_calibration: str = ""
    oos_repair_focus: str = ""
    rollback_priority: str = ""
    quarantine: str = ""
    watch: str = ""
    notes: list[str] = Field(default_factory=list)

    def has_any(self) -> bool:
        return any(value not in ("", None, [], {}) for value in self.model_dump().values())


class PerformanceSourceRecord(BaseModel):
    kind: str
    id: str = ""
    path: str = ""


class PerformanceLearningRecord(BaseModel):
    record_id: str = ""
    record_type: PerformanceRecordType
    scope: str = ""
    bot_id: str = ""
    strategy_id: str = ""
    portfolio_id: str = ""
    source_cadence: SourceCadence
    learning_layer: LearningLayer
    authority_level: AuthorityLevel
    decision_stage: DecisionStage
    material_approval_evidence: bool = False
    loop_run_id: str = ""
    run_id: str = ""
    run_month: str = ""
    task_id: str = ""
    agent_run_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    strategy_change_record_ids: list[str] = Field(default_factory=list)
    approval_request_id: str = ""
    deployment_id: str = ""
    data_bundle_id: str = ""
    objective_version: str = ""
    scoring_profile: str = ""
    verifier_version: str = ""
    artifact_authority_version: str = ""
    monthly_search_brief_path: str = ""
    learning_sufficiency_manifest_id: str = ""
    learning_sufficiency_manifest_path: str = ""
    learning_sufficiency_status: str = ""
    learning_capability_status: dict[str, str] = Field(default_factory=dict)
    supported_learning_capabilities: list[str] = Field(default_factory=list)
    blocked_learning_capability_ids: list[str] = Field(default_factory=list)
    source_weekly_signal_ids: list[str] = Field(default_factory=list)
    brief_attribution_ids: list[str] = Field(default_factory=list)
    strategy_config_diff: dict[str, Any] = Field(default_factory=dict)
    portfolio_allocation_diff: dict[str, Any] = Field(default_factory=dict)
    expected_deltas: PerformanceMetricDeltas = Field(default_factory=PerformanceMetricDeltas)
    realized_after_cost_deltas: PerformanceMetricDeltas = Field(default_factory=PerformanceMetricDeltas)
    verdict: str = ""
    intended_learning_effects: IntendedLearningEffects = Field(default_factory=IntendedLearningEffects)
    strategy_slice: StrategySliceContext = Field(default_factory=StrategySliceContext)
    portfolio_context: PortfolioInteractionContext = Field(default_factory=PortfolioInteractionContext)
    evidence_paths: list[str] = Field(default_factory=list)
    blocker_reasons: list[str] = Field(default_factory=list)
    rollback_status: str = ""
    summary: str = ""
    event_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_records: list[PerformanceSourceRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_authority(self) -> PerformanceLearningRecord:
        expected = _EXPECTED_AUTHORITY[self.source_cadence]
        if self.learning_layer != expected[0] or self.authority_level != expected[1]:
            raise ValueError(
                "source_cadence, learning_layer, and authority_level are inconsistent "
                f"for {self.source_cadence.value}"
            )
        if self.material_approval_evidence and (
            self.source_cadence != SourceCadence.MONTHLY
            or self.authority_level != AuthorityLevel.MONTHLY_REPLAY_AUTHORITY
        ):
            raise ValueError(
                "only monthly replay-authority records may satisfy material approval evidence"
            )
        if self.record_type == PerformanceRecordType.PORTFOLIO and not self.portfolio_id:
            self.portfolio_id = self.scope or self.bot_id or "portfolio"
        if not self.scope:
            self.scope = self.strategy_id or self.portfolio_id or self.bot_id or "global"
        if not self.record_id:
            self.record_id = self._make_record_id()
        return self

    def _make_record_id(self) -> str:
        source_key = _stable_source_key(self)
        raw = "|".join([
            self.record_type.value,
            self.decision_stage.value,
            self.scope,
            self.bot_id,
            self.strategy_id,
            self.portfolio_id,
            source_key,
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def summary_line(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type.value,
            "scope": self.scope,
            "bot_id": self.bot_id,
            "strategy_id": self.strategy_id,
            "portfolio_id": self.portfolio_id,
            "source_cadence": self.source_cadence.value,
            "learning_layer": self.learning_layer.value,
            "authority_level": self.authority_level.value,
            "decision_stage": self.decision_stage.value,
            "proposal_ids": self.proposal_ids[:5],
            "strategy_change_record_ids": self.strategy_change_record_ids[:5],
            "expected_deltas": _non_empty_dict(self.expected_deltas.model_dump(by_alias=True)),
            "realized_after_cost_deltas": _non_empty_dict(
                self.realized_after_cost_deltas.model_dump(by_alias=True)
            ),
            "verdict": self.verdict,
            "learning_sufficiency_status": self.learning_sufficiency_status,
            "supported_learning_capabilities": self.supported_learning_capabilities[:5],
            "blocked_learning_capability_ids": self.blocked_learning_capability_ids[:5],
            "source_weekly_signal_ids": self.source_weekly_signal_ids[:5],
            "evidence_paths": self.evidence_paths[:5],
            "summary": self.summary,
        }


_EXPECTED_AUTHORITY: dict[SourceCadence, tuple[LearningLayer, AuthorityLevel]] = {
    SourceCadence.DAILY: (LearningLayer.SENSOR_CONTEXT, AuthorityLevel.DIAGNOSTIC),
    SourceCadence.WEEKLY: (LearningLayer.BOUNDED_SEARCH_PRIOR, AuthorityLevel.ADVISORY_PRIOR),
    SourceCadence.MONTHLY: (LearningLayer.TRADING_AUTHORITY, AuthorityLevel.MONTHLY_REPLAY_AUTHORITY),
    SourceCadence.FOLLOW_UP: (
        LearningLayer.PERSISTENCE_CONFIRMATION,
        AuthorityLevel.PERSISTENCE_CONFIRMATION,
    ),
    SourceCadence.HARNESS: (LearningLayer.HARNESS_META_LEARNING, AuthorityLevel.BENCHMARK_ONLY),
}


def authority_for_cadence(cadence: SourceCadence | str) -> tuple[LearningLayer, AuthorityLevel]:
    return _EXPECTED_AUTHORITY[SourceCadence(cadence)]


def _non_empty_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _stable_source_key(record: PerformanceLearningRecord) -> str:
    """Durable projection identity, excluding mutable summaries and timestamps."""

    source_ids = [
        f"{source.kind}:{source.id}"
        for source in record.source_records
        if source.kind in _PROJECTION_SOURCE_KINDS and source.id
    ]
    if source_ids:
        return "source:" + ",".join(sorted(source_ids))
    if record.strategy_change_record_ids:
        return "strategy_change:" + ",".join(sorted(record.strategy_change_record_ids))
    if record.proposal_ids:
        return "proposal:" + ",".join(sorted(record.proposal_ids))
    if record.approval_request_id:
        return f"approval:{record.approval_request_id}"
    if record.deployment_id:
        return f"deployment:{record.deployment_id}"
    if record.run_id:
        return f"run:{record.run_id}"
    source_ids = [
        f"{source.kind}:{source.id}"
        for source in record.source_records
        if source.kind and source.id
    ]
    if source_ids:
        return "source:" + ",".join(sorted(source_ids))
    return "scope:" + record.scope


_PROJECTION_SOURCE_KINDS = {
    "proposal_candidate",
    "proposal_evaluation",
    "proposal_outcome",
    "strategy_change",
    "strategy_monthly_verdict",
    "strategy_follow_up_verdict",
    "portfolio_outcome",
}
