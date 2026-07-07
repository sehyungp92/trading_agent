"""CME equity-index futures calendar approximation for manifest diagnostics."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

from .core import CalendarDefinition


CALENDAR_ID = "cme_equity_index_futures_v1"
RULE_AUTHORITY_REL_PATH = Path("data/market_rules/cme/equity_index_futures_v1.json")

_HOLIDAYS = frozenset(
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
)

_CLOSED_RANGES_UTC = (
    ("2024-03-28T22:00:00+00:00", "2024-03-29T20:55:00+00:00"),
    ("2024-05-27T17:00:00+00:00", "2024-05-27T20:55:00+00:00"),
    ("2024-06-19T17:00:00+00:00", "2024-06-19T20:55:00+00:00"),
    ("2024-07-03T17:15:00+00:00", "2024-07-03T20:55:00+00:00"),
    ("2024-07-04T17:00:00+00:00", "2024-07-04T20:55:00+00:00"),
    ("2024-09-02T17:00:00+00:00", "2024-09-02T20:55:00+00:00"),
    ("2024-11-28T18:00:00+00:00", "2024-11-28T21:55:00+00:00"),
    ("2024-12-24T18:15:00+00:00", "2024-12-24T21:55:00+00:00"),
    ("2024-12-24T23:00:00+00:00", "2024-12-25T21:55:00+00:00"),
    ("2024-12-31T23:00:00+00:00", "2025-01-01T21:55:00+00:00"),
    ("2025-01-09T14:30:00+00:00", "2025-01-09T21:55:00+00:00"),
    ("2025-01-20T18:00:00+00:00", "2025-01-20T21:55:00+00:00"),
    ("2025-02-17T18:00:00+00:00", "2025-02-17T21:55:00+00:00"),
    ("2025-04-17T22:00:00+00:00", "2025-04-18T20:55:00+00:00"),
    ("2025-05-26T17:00:00+00:00", "2025-05-26T20:55:00+00:00"),
    ("2025-06-19T17:00:00+00:00", "2025-06-19T20:55:00+00:00"),
    ("2025-07-03T17:15:00+00:00", "2025-07-03T20:55:00+00:00"),
    ("2025-07-04T17:00:00+00:00", "2025-07-04T20:55:00+00:00"),
    ("2025-09-01T17:00:00+00:00", "2025-09-01T20:55:00+00:00"),
    ("2025-11-27T18:00:00+00:00", "2025-11-27T21:55:00+00:00"),
    ("2025-12-24T18:15:00+00:00", "2025-12-24T21:55:00+00:00"),
    ("2025-12-24T23:00:00+00:00", "2025-12-25T21:55:00+00:00"),
    ("2025-12-31T23:00:00+00:00", "2026-01-01T21:55:00+00:00"),
    ("2026-01-19T18:00:00+00:00", "2026-01-19T21:55:00+00:00"),
    ("2026-02-16T18:00:00+00:00", "2026-02-16T21:55:00+00:00"),
    ("2026-04-03T13:15:00+00:00", "2026-04-03T20:55:00+00:00"),
)


def calendar_definition() -> CalendarDefinition:
    return CalendarDefinition(
        calendar_id=CALENDAR_ID,
        timezone="America/Chicago",
        session_open="17:00",
        session_close="16:00",
        weekdays=(6, 0, 1, 2, 3),
        holidays=_HOLIDAYS,
        version="v1",
        market="cme_futures",
        source=str(RULE_AUTHORITY_REL_PATH),
        closed_ranges_utc=_CLOSED_RANGES_UTC,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def rule_authority_path(repo_root: Path) -> Path:
    return Path(repo_root) / RULE_AUTHORITY_REL_PATH


def rule_authority_checksum(repo_root: Path) -> str:
    path = rule_authority_path(repo_root)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
