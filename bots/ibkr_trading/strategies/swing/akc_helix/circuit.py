"""Circuit breaker accounting helpers."""
from __future__ import annotations

from datetime import datetime

from .models import CircuitBreakerState


def roll_circuit_breaker_window(
    cb: CircuitBreakerState,
    bar_time: datetime,
) -> CircuitBreakerState:
    """Reset realized-R buckets when the calendar day or ISO week changes."""
    day_key = bar_time.date().isoformat()
    iso = bar_time.isocalendar()
    week_key = f"{iso.year}-W{iso.week:02d}"

    if cb.daily_bucket != day_key:
        cb.daily_bucket = day_key
        cb.daily_realized_r = 0.0
    if cb.weekly_bucket != week_key:
        cb.weekly_bucket = week_key
        cb.weekly_realized_r = 0.0
    return cb
