# schemas/correction_patterns.py
"""Schemas for correction pattern extraction — recurring human feedback patterns."""
from __future__ import annotations

from pydantic import BaseModel


class CorrectionPattern(BaseModel):
    """A recurring pattern extracted from multiple human corrections."""

    pattern_id: str  # auto-generated hash
    correction_type: str  # from CorrectionType enum
    target: str  # bot_id or "all"
    description: str  # "You keep misclassifying regime X as Y for bot Z"
    count: int  # how many corrections match
    first_seen: str  # oldest correction timestamp
    last_seen: str  # most recent
    example_texts: list[str] = []  # up to 3 raw_text examples


class CorrectionPatternReport(BaseModel):
    """Report of extracted correction patterns."""

    extracted_at: str
    total_corrections_analyzed: int
    patterns: list[CorrectionPattern] = []
