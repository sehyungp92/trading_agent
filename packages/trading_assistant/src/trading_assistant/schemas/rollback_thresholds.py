"""Versioned rollback threshold policy schemas."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class RollbackRecommendationAction(str, Enum):
    WATCH = "watch"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"


class RollbackThresholds(BaseModel):
    """Numeric policy for rollback/quarantine recommendations."""

    policy_version: str = "rollback_thresholds_v1"
    min_confidence: float = 0.6
    repair_objective_delta: float = -0.05
    rollback_objective_delta: float = -0.15
    repair_drawdown_delta: float = 0.05
    rollback_drawdown_delta: float = 0.15
    repair_execution_slippage_delta: float = 0.03
    rollback_execution_slippage_delta: float = 0.08
    quarantine_negative_count: int = 2


class RollbackRecommendation(BaseModel):
    """Approval-gated recommendation emitted from monthly or early-warning evidence."""

    recommendation_id: str = ""
    bot_id: str
    strategy_id: str = ""
    strategy_change_record_id: str = ""
    deployment_id: str = ""
    action: RollbackRecommendationAction
    reason: str
    confidence: float = 0.0
    requires_approval: bool = True
    evidence_paths: list[str] = Field(default_factory=list)
    source_outcome_id: str = ""
    policy_version: str = "rollback_thresholds_v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
