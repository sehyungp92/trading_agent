# schemas/outcome_reasoning.py
"""Outcome reasoning schemas — causal analysis of why suggestions worked or didn't."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class OutcomeReasoning(BaseModel):
    """Causal reasoning about a measured suggestion outcome."""
    suggestion_id: str
    genuine_effect: bool | None = None  # Was the measured effect genuinely caused? None = inconclusive
    mechanism: str = ""           # How did it work (or fail)?
    transferable: bool = False    # Can this insight be applied to other bots?
    lessons_learned: str = ""     # Key takeaway for future decisions
    revised_confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # Updated confidence
    market_context: str = ""      # What market conditions existed during measurement?
    confounders: list[str] = []   # Identified confounding factors
    reasoned_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class OutcomeReasoningReport(BaseModel):
    """Collection of outcome reasonings from a single agent invocation."""
    run_id: str = ""
    date: str = ""  # when reasoning was performed
    reasonings: list[OutcomeReasoning] = []
