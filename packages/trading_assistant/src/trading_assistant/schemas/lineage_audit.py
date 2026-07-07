"""Lineage audit schemas."""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class LineageSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    BLOCKING = "blocking"


class LineageGapReport(BaseModel):
    """Missing-lineage report for one bot/strategy/window."""

    bot_id: str
    strategy_id: str = ""
    window_start: date
    window_end: date
    total_events: int = 0
    missing_field_counts: dict[str, int] = Field(default_factory=dict)
    lineage_coverage_ratio: float = 0.0
    severity: LineageSeverity = LineageSeverity.OK
    recommended_action: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def blocks_authoritative_validation(self) -> bool:
        return self.severity == LineageSeverity.BLOCKING
