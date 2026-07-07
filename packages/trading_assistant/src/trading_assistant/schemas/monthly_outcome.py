"""Authoritative monthly outcome measurement schemas."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from trading_assistant.schemas.assistant_lineage import AssistantLineage, assistant_lineage_from_fields
from trading_assistant.schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION


class OutcomeSource(str, Enum):
    EARLY_WARNING = "early_warning"
    MONTHLY = "monthly"
    FOLLOW_UP = "follow_up"


class MonthlyOutcomeVerdict(str, Enum):
    KEEP = "keep"
    WATCH = "watch"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"
    INCONCLUSIVE = "inconclusive"


class OutcomeDataSufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    SPARSE = "sparse"
    INSUFFICIENT = "insufficient"


class FollowUpTrigger(str, Enum):
    THREE_MONTHS = "three_months"
    MIN_TRADE_COUNT = "min_trade_count"


class MonthlyOutcomeRecord(BaseModel):
    """One source-aware verdict for a deployed strategy/config change."""

    outcome_id: str = ""
    source: OutcomeSource = OutcomeSource.MONTHLY
    bot_id: str
    strategy_id: str
    run_id: str = ""
    run_month: str = ""
    workflow: str = "monthly_validation"
    source_provider: str = ""
    source_model: str = ""
    strategy_change_record_id: str = ""
    deployment_id: str = ""
    config_version: str = ""
    strategy_version: str = ""
    commit_sha: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_ids: list[str] = Field(default_factory=list)
    assistant_lineage: AssistantLineage = Field(default_factory=AssistantLineage)
    mutation_family: str = ""
    category: str = ""
    verdict: MonthlyOutcomeVerdict = MonthlyOutcomeVerdict.INCONCLUSIVE
    live_vs_expected_objective_delta: float = 0.0
    trade_frequency_delta: float = 0.0
    drawdown_delta: float = 0.0
    execution_slippage_delta: float = 0.0
    objective_deltas: dict[str, float] = Field(default_factory=dict)
    gap_attribution: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_sufficiency: OutcomeDataSufficiency = OutcomeDataSufficiency.INSUFFICIENT
    recommended_next_action: str = ""
    evidence_paths: list[str] = Field(default_factory=list)
    minimum_trade_count_met: bool = False
    persistence_confirmed: bool = False
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("run_month")
    @classmethod
    def _validate_run_month(cls, value: str) -> str:
        if not value:
            return value
        parts = value.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            raise ValueError("run_month must be YYYY-MM")
        month = int(parts[1])
        if not 1 <= month <= 12:
            raise ValueError("run_month month must be 01..12")
        return value

    @model_validator(mode="after")
    def _ensure_outcome_id(self) -> "MonthlyOutcomeRecord":
        if not self.outcome_id:
            self.outcome_id = make_monthly_outcome_id(
                source=self.source,
                bot_id=self.bot_id,
                strategy_id=self.strategy_id,
                run_id=self.run_id,
                run_month=self.run_month,
                strategy_change_record_id=self.strategy_change_record_id,
                deployment_id=self.deployment_id,
                proposal_ids=self.proposal_ids,
                suggestion_ids=self.suggestion_ids,
            )
        if not self.objective_deltas:
            self.objective_deltas = {
                "live_vs_expected": self.live_vs_expected_objective_delta,
                "trade_frequency": self.trade_frequency_delta,
                "drawdown": self.drawdown_delta,
                "execution_slippage": self.execution_slippage_delta,
            }
        derived = assistant_lineage_from_fields(
            proposal_ids=self.proposal_ids,
            suggestion_ids=self.suggestion_ids,
            deployment_id=self.deployment_id,
            strategy_change_record_ids=[self.strategy_change_record_id],
            monthly_outcome_id=self.outcome_id,
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

    @property
    def is_authoritative(self) -> bool:
        return self.source in {OutcomeSource.MONTHLY, OutcomeSource.FOLLOW_UP}

    @property
    def is_negative(self) -> bool:
        return self.verdict in {
            MonthlyOutcomeVerdict.REPAIR,
            MonthlyOutcomeVerdict.ROLLBACK,
            MonthlyOutcomeVerdict.QUARANTINE,
        }

    @property
    def is_positive_prior_eligible(self) -> bool:
        if self.verdict != MonthlyOutcomeVerdict.KEEP:
            return False
        if self.source == OutcomeSource.FOLLOW_UP and self.persistence_confirmed:
            return True
        return self.minimum_trade_count_met and self.confidence >= 0.8


class FollowUpOutcomeSchedule(BaseModel):
    """A pending persistence check for an already-recorded monthly verdict."""

    outcome_id: str
    bot_id: str
    strategy_id: str
    strategy_change_record_id: str = ""
    deployment_id: str = ""
    trigger: FollowUpTrigger = FollowUpTrigger.THREE_MONTHS
    due_run_month: str = ""
    min_trade_count: int = 0
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def make_monthly_outcome_id(
    *,
    source: OutcomeSource | str,
    bot_id: str,
    strategy_id: str,
    run_id: str = "",
    run_month: str = "",
    strategy_change_record_id: str = "",
    deployment_id: str = "",
    proposal_ids: list[str] | None = None,
    suggestion_ids: list[str] | None = None,
) -> str:
    source_value = source.value if isinstance(source, OutcomeSource) else str(source)
    raw = "|".join([
        source_value,
        bot_id,
        strategy_id,
        run_id,
        run_month,
        strategy_change_record_id,
        deployment_id,
        ",".join(sorted(proposal_ids or [])),
        ",".join(sorted(suggestion_ids or [])),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
