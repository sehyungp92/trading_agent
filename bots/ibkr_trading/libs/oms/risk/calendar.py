"""Event calendar for blackout windows."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class EventWindow:
    name: str  # "CPI", "NFP", "FOMC"
    start_utc: datetime
    end_utc: datetime
    cooldown_bars: int = 3
    max_extension_minutes: int = 60


class EventCalendar:
    """Calendar of economic event blackout windows.
    Loaded from config/calendars.yaml at startup.
    """

    def __init__(self, windows: list[EventWindow] | None = None):
        self._windows = sorted(windows or [], key=lambda w: w.start_utc)

    def is_blocked(self, now_utc: datetime) -> bool:
        for w in self._windows:
            if w.start_utc <= now_utc <= w.end_utc:
                return True
            if w.start_utc > now_utc:
                break
        return False

    def current_event(self, now_utc: datetime) -> Optional[EventWindow]:
        """Return the event window that now_utc falls within, or None."""
        for w in self._windows:
            if w.start_utc <= now_utc <= w.end_utc:
                return w
            if w.start_utc > now_utc:
                break
        return None

    def next_event(self, now_utc: datetime) -> Optional[EventWindow]:
        for w in self._windows:
            if w.start_utc > now_utc:
                return w
        return None

    def add_window(self, window: EventWindow) -> None:
        self._windows.append(window)
        self._windows.sort(key=lambda w: w.start_utc)
