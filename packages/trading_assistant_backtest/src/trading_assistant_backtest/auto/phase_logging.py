"""Phase logging helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def event(stage: str, message: str, **fields) -> dict:
    return {"ts": datetime.now(UTC).isoformat(), "stage": stage, "message": message, **fields}
