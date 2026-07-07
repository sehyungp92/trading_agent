"""Session persistence schemas for analysis runs (H5)."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    """Record of a single LLM invocation within an analysis session."""

    session_id: str
    agent_type: str  # e.g. "daily_analysis", "weekly_analysis", "monthly_validation", "triage"
    prompt_hash: str = ""  # SHA256 of the prompt package for dedup/tracking
    response_summary: str = ""  # First 500 chars of the response
    token_usage: dict = Field(default_factory=dict)  # {"input_tokens": N, "output_tokens": N}
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)
