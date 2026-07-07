"""Clock abstractions for wall-clock and simulated time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for time sources."""

    def now(self) -> datetime: ...
    def is_backtest(self) -> bool: ...


class SimClock:
    """Simulated clock that advances via explicit calls.

    The StrategyEngine calls advance() on each bar — the feed does NOT.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self._current_time = start or datetime.min.replace(tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._current_time

    def advance(self, ts: datetime) -> None:
        """Advance simulated time to the given timestamp."""
        self._current_time = ts

    def is_backtest(self) -> bool:
        return True

    def snapshot_state(self) -> dict[str, datetime]:
        """Return the current simulated timestamp."""
        return {"current_time": self._current_time}

    def restore_state(self, snapshot: dict[str, datetime]) -> None:
        """Restore a timestamp captured by :meth:`snapshot_state`."""
        self._current_time = snapshot["current_time"]


class WallClock:
    """Real wall-clock time source."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def is_backtest(self) -> bool:
        return False
