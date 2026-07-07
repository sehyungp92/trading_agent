# schemas/structural_experiment.py
"""Structural experiment schemas — track experiments with falsifiable acceptance criteria.

Ensures structural proposals have measurable success criteria and are
evaluated against real outcomes before being promoted or abandoned.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class AcceptanceCriteria(BaseModel):
    """A single falsifiable acceptance criterion for an experiment."""

    metric: str
    direction: str = "improve"  # "improve" or "not_degrade"
    minimum_change: float = 0.0
    observation_window_days: int = 14
    minimum_trade_count: int = 20
    baseline_value: Optional[float] = None


class ExperimentRecord(BaseModel):
    """A tracked structural experiment with lifecycle and acceptance criteria."""

    experiment_id: str
    bot_id: str
    title: str
    description: str = ""
    hypothesis_id: Optional[str] = None
    suggestion_id: Optional[str] = None
    proposal_run_id: str = ""
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    acceptance_criteria: list[AcceptanceCriteria] = Field(default_factory=list)
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    activated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    criteria_met: list[bool] = Field(default_factory=list)
    actual_values: list[float] = Field(default_factory=list)
    resolution_notes: str = ""

    @property
    def all_criteria_met(self) -> bool:
        return bool(self.criteria_met) and all(self.criteria_met)

    @property
    def is_evaluable(self) -> bool:
        if self.status != ExperimentStatus.ACTIVE or not self.activated_at:
            return False
        activated = self.activated_at
        if activated.tzinfo is None:
            activated = activated.replace(tzinfo=timezone.utc)
        max_window = max(
            (c.observation_window_days for c in self.acceptance_criteria),
            default=14,
        )
        return datetime.now(timezone.utc) >= activated + timedelta(days=max_window)
