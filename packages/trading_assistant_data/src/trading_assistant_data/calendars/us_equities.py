"""US equities regular-session calendar approximation."""

from __future__ import annotations

from datetime import date, datetime, time, timezone

from .core import CalendarDefinition


CALENDAR_ID = "us_equities_xnys_xnas_v1"
EXTENDED_SESSION_OPEN = "04:00"
EXTENDED_SESSION_CLOSE = "20:00"
EARLY_CLOSE_EXTENDED_SESSION_CLOSE = "17:00"
EARLY_CLOSE_TIME = "13:00"
EARLY_CLOSE_DATES = frozenset(
    {
        date(2021, 11, 26),
        date(2022, 11, 25),
        date(2023, 7, 3),
        date(2023, 11, 24),
        date(2024, 7, 3),
        date(2024, 11, 29),
        date(2024, 12, 24),
        date(2025, 7, 3),
        date(2025, 11, 28),
        date(2025, 12, 24),
        date(2026, 11, 27),
        date(2026, 12, 24),
    }
)


def calendar_definition() -> CalendarDefinition:
    return CalendarDefinition(
        calendar_id=CALENDAR_ID,
        timezone="America/New_York",
        session_open="09:30",
        session_close="16:00",
        weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset(
            {
                date(2021, 1, 1),
                date(2021, 1, 18),
                date(2021, 2, 15),
                date(2021, 4, 2),
                date(2021, 5, 31),
                date(2021, 7, 5),
                date(2021, 9, 6),
                date(2021, 11, 25),
                date(2021, 12, 24),
                date(2022, 1, 17),
                date(2022, 2, 21),
                date(2022, 4, 15),
                date(2022, 5, 30),
                date(2022, 6, 20),
                date(2022, 7, 4),
                date(2022, 9, 5),
                date(2022, 11, 24),
                date(2022, 12, 26),
                date(2023, 1, 2),
                date(2023, 1, 16),
                date(2023, 2, 20),
                date(2023, 4, 7),
                date(2023, 5, 29),
                date(2023, 6, 19),
                date(2023, 7, 4),
                date(2023, 9, 4),
                date(2023, 11, 23),
                date(2023, 12, 25),
                date(2024, 1, 1),
                date(2024, 1, 15),
                date(2024, 2, 19),
                date(2024, 3, 29),
                date(2024, 5, 27),
                date(2024, 6, 19),
                date(2024, 7, 4),
                date(2024, 9, 2),
                date(2024, 11, 28),
                date(2024, 12, 25),
                date(2025, 1, 1),
                date(2025, 1, 9),
                date(2025, 1, 20),
                date(2025, 2, 17),
                date(2025, 4, 18),
                date(2025, 5, 26),
                date(2025, 6, 19),
                date(2025, 7, 4),
                date(2025, 9, 1),
                date(2025, 11, 27),
                date(2025, 12, 25),
                date(2026, 1, 1),
                date(2026, 1, 19),
                date(2026, 2, 16),
                date(2026, 4, 3),
                date(2026, 5, 25),
                date(2026, 6, 19),
                date(2026, 7, 3),
                date(2026, 9, 7),
                date(2026, 11, 26),
                date(2026, 12, 25),
            }
        ),
        version="v1",
        market="us_equity",
        source="NYSE/Nasdaq regular-session holiday and early-close seed",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def session_close_for_date(value: date) -> time:
    """Return the modeled regular-session close for a US equity trading date."""

    if value in EARLY_CLOSE_DATES:
        return _parse_time(EARLY_CLOSE_TIME)
    return _parse_time(calendar_definition().session_close)


def extended_session_close_for_date(value: date) -> time:
    """Return the modeled IBKR extended-hours close for a US equity trading date."""

    if value in EARLY_CLOSE_DATES:
        return _parse_time(EARLY_CLOSE_EXTENDED_SESSION_CLOSE)
    return _parse_time(EXTENDED_SESSION_CLOSE)


def extended_session_open() -> time:
    return _parse_time(EXTENDED_SESSION_OPEN)


def _parse_time(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))
