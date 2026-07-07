"""Canonical replay frame placeholders.

Strategy plugins can adapt data bundle slices into their native frame formats behind this
module. The monthly runner only depends on normalized metadata until a real plugin is
wired to production strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ReplayFrame:
    symbol: str
    timeframe: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    row_count: int = 0
