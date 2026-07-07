"""Schema for the enriched relay /health response."""
from __future__ import annotations

from pydantic import BaseModel


class RelayHealthResponse(BaseModel):
    """Typed mirror of the relay's enriched /health response."""
    status: str = "ok"
    pending_events: int = 0
    per_bot_pending: dict[str, int] = {}
    last_event_per_bot: dict[str, str] = {}
    oldest_pending_age_seconds: float = 0.0
    db_size_bytes: int = 0
    uptime_seconds: float = 0.0
