# schemas/corrections.py
"""Human correction schemas — structured feedback from Telegram/Discord replies.

Written to memory/findings/corrections.jsonl. Included in future analysis prompts
so Claude learns the human's mental model over time (Ralph Loop V2).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CorrectionType(str, Enum):
    TRADE_RECLASSIFY = "trade_reclassify"
    REGIME_OVERRIDE = "regime_override"
    POSITIVE_REINFORCEMENT = "positive_reinforcement"
    ALLOCATION_CHANGE = "allocation_change"
    SUGGESTION_ACCEPT = "suggestion_accept"
    SUGGESTION_REJECT = "suggestion_reject"
    FREE_TEXT = "free_text"


class HumanCorrection(BaseModel):
    """A single human correction or feedback item."""

    correction_type: CorrectionType
    original_report_id: str
    target_id: Optional[str] = None  # trade_id or bot_id being corrected
    raw_text: str
    structured_correction: dict = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
