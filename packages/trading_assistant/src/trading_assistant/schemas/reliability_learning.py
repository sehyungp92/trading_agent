# schemas/reliability_learning.py
"""Reliability learning schemas — track bug fix interventions and recurrence.

Enables closed-loop learning for reliability improvements: record interventions,
track recurrences, auto-verify fixes, and surface chronic bug classes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BugClass(str, Enum):
    CONNECTION = "connection"
    DATA_INTEGRITY = "data_integrity"
    TIMING = "timing"
    CONFIG = "config"
    LOGIC = "logic"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class InterventionStatus(str, Enum):
    OPEN = "open"
    VERIFIED = "verified"
    RECURRED = "recurred"
    SUPERSEDED = "superseded"


class ReliabilityIntervention(BaseModel):
    """A recorded bug fix intervention with observation tracking."""

    intervention_id: str
    bot_id: str
    bug_class: BugClass
    error_category: str = ""
    triage_run_id: str = ""
    fix_description: str = ""
    root_cause: str = ""
    source_file: str = ""
    pr_url: str = ""
    status: InterventionStatus = InterventionStatus.OPEN
    opened_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    verified_at: Optional[datetime] = None
    observation_window_days: int = 14
    recurrence_count: int = 0
    last_recurrence_at: Optional[datetime] = None

    @property
    def is_within_observation(self) -> bool:
        now = datetime.now(timezone.utc)
        opened = self.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return now < opened + timedelta(days=self.observation_window_days)


class ReliabilityScorecard(BaseModel):
    """Aggregate reliability stats for a single bug class."""

    bug_class: BugClass
    intervention_count: int = 0
    verified_count: int = 0
    recurrence_rate: float = 0.0
    avg_observation_days: float = 0.0


class ReliabilitySummary(BaseModel):
    """Summary of reliability across all bug classes."""

    scorecards_by_class: dict[str, ReliabilityScorecard] = Field(default_factory=dict)
    chronic_bug_classes: list[str] = Field(default_factory=list)  # 3+ recurrences
    total_open: int = 0
