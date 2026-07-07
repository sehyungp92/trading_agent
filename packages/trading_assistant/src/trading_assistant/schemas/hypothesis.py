# schemas/hypothesis.py
"""Hypothesis record schema with lifecycle tracking."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class HypothesisRecord(BaseModel):
    """A hypothesis with lifecycle counters for adaptive learning."""

    id: str
    title: str
    category: str
    description: str
    evidence_required: str = ""
    reversibility: str = "moderate"
    estimated_complexity: str = "medium"
    # Lifecycle tracking
    times_proposed: int = 0
    times_accepted: int = 0
    times_rejected: int = 0
    outcomes_positive: int = 0
    outcomes_negative: int = 0
    status: Literal["active", "candidate", "retired"] = "active"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_proposed_at: str = ""

    @property
    def effectiveness(self) -> float:
        """Score: (positive - negative) / max(proposed, 1)."""
        denom = max(self.times_proposed, 1)
        return (self.outcomes_positive - self.outcomes_negative) / denom
