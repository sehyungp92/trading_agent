"""Unit tests for libs.config.event_calendar — EventCalendar class."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from libs.config.event_calendar import EventCalendar, EventWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _make_window(
    name: str,
    start: datetime,
    end: datetime,
    cooldown_bars: int = 3,
    max_extension_minutes: int = 60,
) -> EventWindow:
    return EventWindow(
        name=name,
        start_utc=start,
        end_utc=end,
        cooldown_bars=cooldown_bars,
        max_extension_minutes=max_extension_minutes,
    )


# ---------------------------------------------------------------------------
# Tests — is_blocked (window matching)
# ---------------------------------------------------------------------------

class TestIsBlocked:
    """Verify is_blocked returns correct bool during and outside windows."""

    def test_blocked_inside_window(self) -> None:
        window = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        cal = EventCalendar([window])

        now = _utc(2026, 3, 12, 8, 30)
        assert cal.is_blocked(now) is True

    def test_not_blocked_before_window(self) -> None:
        window = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        cal = EventCalendar([window])

        now = _utc(2026, 3, 12, 7, 59)
        assert cal.is_blocked(now) is False

    def test_not_blocked_after_window(self) -> None:
        window = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        cal = EventCalendar([window])

        now = _utc(2026, 3, 12, 9, 1)
        assert cal.is_blocked(now) is False

    def test_blocked_at_exact_start(self) -> None:
        window = _make_window("NFP", _utc(2026, 4, 3, 12, 30), _utc(2026, 4, 3, 13, 30))
        cal = EventCalendar([window])

        assert cal.is_blocked(_utc(2026, 4, 3, 12, 30)) is True

    def test_blocked_at_exact_end(self) -> None:
        window = _make_window("NFP", _utc(2026, 4, 3, 12, 30), _utc(2026, 4, 3, 13, 30))
        cal = EventCalendar([window])

        assert cal.is_blocked(_utc(2026, 4, 3, 13, 30)) is True

    def test_multiple_windows_second_blocks(self) -> None:
        w1 = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        w2 = _make_window("FOMC", _utc(2026, 3, 19, 14, 0), _utc(2026, 3, 19, 15, 0))
        cal = EventCalendar([w2, w1])  # deliberately out of order

        assert cal.is_blocked(_utc(2026, 3, 19, 14, 30)) is True
        assert cal.is_blocked(_utc(2026, 3, 12, 8, 30)) is True


# ---------------------------------------------------------------------------
# Tests — next_event
# ---------------------------------------------------------------------------

class TestNextEvent:
    """Verify next_event returns the nearest upcoming window."""

    def test_returns_nearest_upcoming(self) -> None:
        w1 = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        w2 = _make_window("FOMC", _utc(2026, 3, 19, 14, 0), _utc(2026, 3, 19, 15, 0))
        cal = EventCalendar([w1, w2])

        now = _utc(2026, 3, 11)
        result = cal.next_event(now)
        assert result is not None
        assert result.name == "CPI"

    def test_skips_past_windows(self) -> None:
        w1 = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        w2 = _make_window("FOMC", _utc(2026, 3, 19, 14, 0), _utc(2026, 3, 19, 15, 0))
        cal = EventCalendar([w1, w2])

        now = _utc(2026, 3, 13)  # after CPI, before FOMC
        result = cal.next_event(now)
        assert result is not None
        assert result.name == "FOMC"

    def test_returns_none_when_all_past(self) -> None:
        w1 = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        cal = EventCalendar([w1])

        now = _utc(2026, 4, 1)
        assert cal.next_event(now) is None

    def test_does_not_return_current_window(self) -> None:
        w1 = _make_window("CPI", _utc(2026, 3, 12, 8, 0), _utc(2026, 3, 12, 9, 0))
        w2 = _make_window("FOMC", _utc(2026, 3, 19, 14, 0), _utc(2026, 3, 19, 15, 0))
        cal = EventCalendar([w1, w2])

        # now is inside CPI window — next_event should return FOMC, not CPI
        now = _utc(2026, 3, 12, 8, 30)
        result = cal.next_event(now)
        assert result is not None
        assert result.name == "FOMC"


# ---------------------------------------------------------------------------
# Tests — empty calendar
# ---------------------------------------------------------------------------

class TestEmptyCalendar:
    """An empty calendar should never block."""

    def test_is_blocked_returns_false(self) -> None:
        cal = EventCalendar()
        assert cal.is_blocked(_utc(2026, 6, 1, 12, 0)) is False

    def test_next_event_returns_none(self) -> None:
        cal = EventCalendar()
        assert cal.next_event(_utc(2026, 6, 1, 12, 0)) is None

    def test_current_event_returns_none(self) -> None:
        cal = EventCalendar()
        assert cal.current_event(_utc(2026, 6, 1, 12, 0)) is None

    def test_windows_is_empty_tuple(self) -> None:
        cal = EventCalendar()
        assert cal.windows == ()

    def test_explicit_empty_list(self) -> None:
        cal = EventCalendar([])
        assert cal.is_blocked(_utc(2026, 6, 1)) is False
        assert cal.windows == ()
