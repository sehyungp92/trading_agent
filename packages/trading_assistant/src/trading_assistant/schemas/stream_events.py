"""SSE stream event schema for real-time monitoring (H4)."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    """Event broadcast via SSE for real-time monitoring."""

    sequence: int
    event_type: str  # e.g. "event_processing_start", "event_processing_complete", "event_processing_error", "alert", "analysis_start", "analysis_complete"
    data: dict = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
