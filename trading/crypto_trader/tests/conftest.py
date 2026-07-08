"""Shared test fixtures and helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Bar, TimeFrame


def make_bar(
    ts: datetime | str,
    o: float,
    h: float,
    l: float,
    c: float,
    v: float = 100.0,
    tf: TimeFrame = TimeFrame.M15,
    sym: str = "BTC",
) -> Bar:
    """Create a Bar for testing."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    return Bar(timestamp=ts, symbol=sym, open=o, high=h, low=l, close=c, volume=v, timeframe=tf)
