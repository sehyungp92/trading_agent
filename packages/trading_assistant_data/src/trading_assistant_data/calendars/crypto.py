"""Crypto 24/7 UTC calendar."""

from __future__ import annotations

from datetime import datetime, timezone

from .core import CalendarDefinition


CALENDAR_ID = "crypto_utc_24_7_v1"


def calendar_definition() -> CalendarDefinition:
    return CalendarDefinition(
        calendar_id=CALENDAR_ID,
        timezone="UTC",
        session_open="00:00",
        session_close="00:00",
        weekdays=(0, 1, 2, 3, 4, 5, 6),
        holidays=frozenset(),
        version="v1",
        market="crypto_perp",
        source="exchange 24/7 UTC convention",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

