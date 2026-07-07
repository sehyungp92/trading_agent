# schemas/pattern_library.py
"""Cross-bot pattern library — catalogues successful structural innovations for transfer."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class PatternCategory(str, Enum):
    FILTER = "filter"
    EXIT_RULE = "exit_rule"
    ENTRY_SIGNAL = "entry_signal"
    POSITION_SIZING = "position_sizing"
    REGIME_GATE = "regime_gate"
    RISK_MANAGEMENT = "risk_management"
    COORDINATION = "coordination"


class PatternStatus(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    IMPLEMENTED = "implemented"
    TRANSFERRED = "transferred"
    REJECTED = "rejected"


class PatternEntry(BaseModel):
    """A catalogued structural innovation that may transfer between bots/strategies."""

    pattern_id: str = ""
    title: str
    category: PatternCategory
    status: PatternStatus = PatternStatus.PROPOSED
    source_bot: str  # Bot where the pattern was first observed/implemented
    source_strategy: str = ""
    target_bots: list[str] = []  # Bots where the pattern could apply
    description: str = ""
    evidence: str = ""  # Quantified evidence (trades, period, metrics)
    estimated_impact: str = ""  # Expected Calmar/PnL impact if transferred
    implementation_notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validated_at: str = ""  # YYYY-MM-DD when promoted to VALIDATED
    linked_suggestion_id: str = ""
