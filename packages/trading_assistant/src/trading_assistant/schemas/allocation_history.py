"""Schema for tracking historical allocation changes."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class AllocationSource(str, Enum):
    MANUAL = "manual"
    SUGGESTED = "suggested"


class AllocationRecord(BaseModel):
    """Single point-in-time allocation for a bot/strategy."""
    date: str  # YYYY-MM-DD
    bot_id: str
    strategy_id: str = ""
    allocation_pct: float
    unit_risk_pct: float = 0.0
    heat_cap_r: float = 0.0
    source: AllocationSource = AllocationSource.MANUAL
    reason: str = ""


class BotAllocationSnapshot(BaseModel):
    """Per-bot allocation snapshot pairing recommended vs actual."""
    bot_id: str
    recommended_pct: float
    actual_pct: float
    drift_pct: float = 0.0       # recommended - actual (signed)
    abs_drift_pct: float = 0.0   # |drift_pct|


class AllocationSnapshot(BaseModel):
    """Point-in-time snapshot of portfolio allocation vs recommendation."""
    date: str                     # YYYY-MM-DD (week_start)
    week_start: str
    week_end: str
    bot_allocations: list[BotAllocationSnapshot] = []
    total_drift_pct: float = 0.0  # sum(|drift|) / 2 (normalized)
    max_single_drift_pct: float = 0.0
    source: str = "weekly_handler"
