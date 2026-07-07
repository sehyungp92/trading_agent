"""KRX equities calendar with explicit holiday-file requirement."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from .core import CalendarDefinition
from .core import TIMEFRAME_MINUTES


CALENDAR_ID = "krx_equities_v1"
KIS_INTRADAY_CALENDAR_ID = "krx_equities_kis_intraday_continuous_auction_v1"


def calendar_definition(holidays_path: Path | None = None) -> CalendarDefinition:
    holidays = _load_holidays(holidays_path) if holidays_path else set()
    return CalendarDefinition(
        calendar_id=CALENDAR_ID,
        timezone="Asia/Seoul",
        session_open="09:00",
        session_close="15:30",
        weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset(holidays),
        version="v1",
        market="krx_equity",
        source="KRX market closing calendar; no weekday-only fallback is authoritative",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def kis_intraday_calendar_definition(holidays_path: Path | None = None) -> CalendarDefinition:
    holidays = _load_holidays(holidays_path) if holidays_path else set()
    return CalendarDefinition(
        calendar_id=KIS_INTRADAY_CALENDAR_ID,
        timezone="Asia/Seoul",
        session_open="09:00",
        session_close="15:30",
        weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset(holidays),
        version="v1",
        market="krx_equity",
        source=(
            "KIS intraday bars are modeled through the KRX regular stock session "
            "close at 15:30 KST, matching the legacy k_stock_trader updater"
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def kis_intraday_expected_bar_opens(
    calendar: CalendarDefinition,
    timeframe: str,
    start_ts: datetime,
    end_ts: datetime,
) -> pd.DatetimeIndex:
    """Expected KIS intraday timestamps under the legacy updater convention.

    KIS/k_stock intraday files carry continuous auction bars through 15:19 KST
    plus a final 15:30 KST closing-auction bar. Higher timeframes are derived
    by flooring those 1m timestamps to the requested interval.
    """

    if end_ts < start_ts:
        return pd.DatetimeIndex([], tz="UTC")
    minutes = TIMEFRAME_MINUTES[timeframe.lower()]
    start_utc = _as_utc_timestamp(start_ts)
    end_utc = _as_utc_timestamp(end_ts)
    start_local = start_utc.tz_convert(calendar.timezone)
    end_local = end_utc.tz_convert(calendar.timezone)
    current = start_local.date()
    final = end_local.date()
    expected: list[pd.Timestamp] = []
    while current <= final:
        if calendar.is_trading_day(current):
            session_open = pd.Timestamp(f"{current.isoformat()} {calendar.session_open}", tz=calendar.timezone)
            close_auction = pd.Timestamp(f"{current.isoformat()} {calendar.session_close}", tz=calendar.timezone)
            continuous_end = close_auction - pd.Timedelta(minutes=11)
            minute_index = pd.date_range(start=session_open, end=continuous_end, freq="1min")
            auction_index = pd.DatetimeIndex([close_auction])
            source_minutes = minute_index.append(auction_index)
            bucketed = source_minutes.floor(f"{minutes}min").unique().sort_values().tz_convert("UTC")
            expected.extend(bucketed[(bucketed >= start_utc) & (bucketed <= end_utc)])
        current += timedelta(days=1)
    return pd.DatetimeIndex(expected).sort_values().unique()


def _load_holidays(path: Path) -> set[date]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    holidays: set[date] = set()
    for dates in payload.values():
        if not isinstance(dates, list):
            continue
        for item in dates:
            holidays.add(date.fromisoformat(str(item)))
    return holidays


def _as_utc_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
