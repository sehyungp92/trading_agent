"""Event blackout calendar support for the unified runtime scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class EventWindow:
    name: str
    start_utc: datetime
    end_utc: datetime
    cooldown_bars: int = 3
    max_extension_minutes: int = 60


class EventCalendar:
    """Calendar of economic blackout windows."""

    def __init__(self, windows: list[EventWindow] | None = None):
        self._windows = sorted(windows or [], key=lambda window: window.start_utc)

    @property
    def windows(self) -> tuple[EventWindow, ...]:
        return tuple(self._windows)

    def is_blocked(self, now_utc: datetime) -> bool:
        return self.current_event(now_utc) is not None

    def current_event(self, now_utc: datetime) -> Optional[EventWindow]:
        for window in self._windows:
            if window.start_utc <= now_utc <= window.end_utc:
                return window
            if window.start_utc > now_utc:
                break
        return None

    def next_event(self, now_utc: datetime) -> Optional[EventWindow]:
        for window in self._windows:
            if window.start_utc > now_utc:
                return window
        return None

