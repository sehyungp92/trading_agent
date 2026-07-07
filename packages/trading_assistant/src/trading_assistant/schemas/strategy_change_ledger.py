"""Strategy-level change ledger schemas."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StrategyChangeRecordType(str, Enum):
    MONTHLY_REVIEW = "monthly_review"
    PROPOSED_CHANGE = "proposed_change"
    ACCEPTED_CHANGE = "accepted_change"
    IMPLEMENTED_CHANGE = "implemented_change"
    DEPLOYED_CHANGE = "deployed_change"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"
    REPAIR = "repair"
    WATCH = "watch"
    NO_CHANGE = "no_change"
    ONE_MONTH_VERDICT = "one_month_verdict"
    FOLLOW_UP_VERDICT = "follow_up_verdict"


class RollbackStatus(str, Enum):
    NONE = "none"
    WATCHING = "watching"
    RECOMMENDED = "recommended"
    EXECUTED = "executed"


class StrategyChangeRecord(BaseModel):
    """One auditable strategy-level decision or lifecycle transition."""

    record_id: str = ""
    bot_id: str
    strategy_id: str
    record_type: StrategyChangeRecordType
    strategy_version: str = ""
    prior_config_version: str = ""
    new_config_version: str = ""
    mutation_diff: dict[str, Any] = Field(default_factory=dict)
    source_proposal_ids: list[str] = Field(default_factory=list)
    source_suggestion_ids: list[str] = Field(default_factory=list)
    approval_request_id: str | None = None
    pr_url: str | None = None
    commit_sha: str | None = None
    deployment_id: str | None = None
    deployed_at: datetime | None = None
    evidence_paths: list[str] = Field(default_factory=list)
    objective_deltas: dict[str, float] = Field(default_factory=dict)
    decision_reason: str = ""
    monthly_verdict: dict[str, Any] | None = None
    follow_up_verdict: dict[str, Any] | None = None
    rollback_status: RollbackStatus = RollbackStatus.NONE
    monthly_status: str = ""
    run_id: str = ""
    run_month: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _ensure_id(self) -> StrategyChangeRecord:
        if not self.record_id:
            self.record_id = make_strategy_change_record_id(
                bot_id=self.bot_id,
                strategy_id=self.strategy_id,
                record_type=self.record_type,
                source_proposal_ids=self.source_proposal_ids,
                approval_request_id=self.approval_request_id or "",
                deployment_id=self.deployment_id or "",
                run_id=self.run_id,
                run_month=self.run_month,
                created_date=self.created_at.date(),
            )
        return self


def make_strategy_change_record_id(
    *,
    bot_id: str,
    strategy_id: str,
    record_type: StrategyChangeRecordType | str,
    source_proposal_ids: list[str] | None = None,
    approval_request_id: str = "",
    deployment_id: str = "",
    run_id: str = "",
    run_month: str = "",
    created_date: date | None = None,
) -> str:
    """Deterministic 16-char record id for idempotent monthly reruns.

    Durable lifecycle links such as proposal ids, approval ids, deployment ids,
    run ids, or run months define the identity. The creation date is only a
    fallback for ad-hoc records that have no stable link key.
    """

    record_value = record_type.value if isinstance(record_type, StrategyChangeRecordType) else str(record_type)
    link_key = (
        ",".join(sorted(source_proposal_ids or []))
        or approval_request_id
        or deployment_id
        or run_id
        or run_month
    )
    if not link_key:
        link_key = (created_date or datetime.now(timezone.utc).date()).isoformat()
    raw = "|".join([bot_id, strategy_id, record_value, link_key])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
