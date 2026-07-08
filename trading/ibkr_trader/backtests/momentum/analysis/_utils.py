"""Shared utilities for momentum analysis modules."""
from __future__ import annotations

from datetime import datetime, time, timedelta

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("US/Eastern")
except ImportError:
    _ET = None


def utc_to_et(dt_val) -> datetime | None:
    """Convert a UTC datetime/timestamp to Eastern Time (DST-aware).

    Falls back to fixed UTC-5 if zoneinfo is not available.
    """
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        dt = dt_val
    else:
        try:
            import pandas as pd
            dt = pd.Timestamp(dt_val).to_pydatetime()
        except Exception:
            return None

    if _ET is not None:
        try:
            # If dt is naive, assume UTC
            if dt.tzinfo is None:
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(_ET).replace(tzinfo=None)
        except Exception:
            pass

    # Fallback: fixed UTC-5 (EST)
    return dt.replace(tzinfo=None) - timedelta(hours=5)


def classify_session(et_dt: datetime) -> str:
    """Classify an Eastern-Time datetime into a session window.

    Windows: ETH-Asia (18:00-02:00), ETH-Europe (02:00-08:00),
    Pre-Market (08:00-09:30), RTH-Open (09:30-10:30),
    RTH-Core (10:30-14:00), RTH-Close (14:00-16:00),
    Evening (16:00-18:00).
    """
    t = et_dt.time()
    if t >= time(18, 0) or t < time(2, 0):
        return "ETH-Asia"
    if time(2, 0) <= t < time(8, 0):
        return "ETH-Europe"
    if time(8, 0) <= t < time(9, 30):
        return "Pre-Market"
    if time(9, 30) <= t < time(10, 30):
        return "RTH-Open"
    if time(10, 30) <= t < time(14, 0):
        return "RTH-Core"
    if time(14, 0) <= t < time(16, 0):
        return "RTH-Close"
    if time(16, 0) <= t < time(18, 0):
        return "Evening"
    return "RTH-Core"


SESSION_ORDER = [
    "ETH-Asia", "ETH-Europe", "Pre-Market",
    "RTH-Open", "RTH-Core", "RTH-Close", "Evening",
]


def parse_dt(val) -> datetime | None:
    """Parse a datetime from various types (datetime, numpy.datetime64, pd.Timestamp)."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        import pandas as pd
        return pd.Timestamp(val).to_pydatetime()
    except Exception:
        return None


def trade_date(trade) -> object | None:
    """Extract a date from a trade's entry_time."""
    entry = getattr(trade, "entry_time", None)
    dt = parse_dt(entry)
    return dt.date() if dt else None
