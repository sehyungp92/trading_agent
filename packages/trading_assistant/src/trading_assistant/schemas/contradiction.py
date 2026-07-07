"""Contradiction detection schemas — temporal inconsistencies across daily reports."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ContradictionType(str, Enum):
    REGIME_DIRECTION_CONFLICT = "regime_direction_conflict"
    FACTOR_QUALITY_DIVERGENCE = "factor_quality_divergence"
    EXIT_PROCESS_CONFLICT = "exit_process_conflict"
    RISK_EXPOSURE_CONFLICT = "risk_exposure_conflict"


class ContradictionItem(BaseModel):
    """A single detected contradiction between two days."""

    type: ContradictionType
    bot_id: str
    description: str
    day_a: str
    day_b: str
    severity: str = "medium"  # low / medium / high
    evidence: dict = {}


class ContradictionReport(BaseModel):
    """Report of all contradictions found across a lookback window."""

    date: str
    lookback_days: int = 3
    items: list[ContradictionItem] = []
    bots_analyzed: list[str] = []
