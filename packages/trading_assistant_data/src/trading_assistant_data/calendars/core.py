"""Small deterministic session calendars used by manifest validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "1m_bid_ask": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "daily": 1440,
    "funding_1h": 60,
    "funding_8h": 480,
}


@dataclass(frozen=True)
class CalendarDefinition:
    calendar_id: str
    timezone: str
    session_open: str
    session_close: str
    weekdays: tuple[int, ...]
    holidays: frozenset[date]
    version: str
    market: str = ""
    source: str = ""
    breaks: tuple[tuple[str, str], ...] = ()
    closed_ranges_utc: tuple[tuple[str, str], ...] = ()
    generated_at: str = ""

    @property
    def session_minutes(self) -> int:
        start = _parse_time(self.session_open)
        end = _parse_time(self.session_close)
        start_dt = datetime.combine(date.today(), start)
        end_dt = datetime.combine(date.today(), end)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        minutes = (end_dt - start_dt).seconds // 60
        for break_start, break_end in self.breaks:
            minutes -= (
                datetime.combine(date.today(), _parse_time(break_end))
                - datetime.combine(date.today(), _parse_time(break_start))
            ).seconds // 60
        return minutes

    def is_trading_day(self, value: date) -> bool:
        return value.weekday() in self.weekdays and value not in self.holidays


def load_calendar_definition(path: Path) -> CalendarDefinition:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalendarDefinition(
        calendar_id=str(payload["calendar_id"]),
        timezone=str(payload["timezone"]),
        session_open=str(payload["session_open"]),
        session_close=str(payload["session_close"]),
        weekdays=tuple(int(item) for item in payload.get("weekdays", [])),
        holidays=frozenset(date.fromisoformat(item) for item in payload.get("holidays", [])),
        version=str(payload["version"]),
        market=str(payload.get("market", "")),
        source=str(payload.get("holiday_source", "")),
        breaks=tuple(tuple(item) for item in payload.get("breaks", [])),
        closed_ranges_utc=tuple(tuple(item) for item in payload.get("closed_ranges_utc", [])),
        generated_at=str(payload.get("generated_at", "")),
    )


def expected_bars(
    calendar: CalendarDefinition,
    timeframe: str,
    start_ts: datetime,
    end_ts: datetime,
) -> int:
    """Count expected bar opens between start and end, inclusive."""

    if _is_daily_timeframe(timeframe):
        return len(expected_trading_dates(calendar, start_ts, end_ts))
    return len(expected_bar_opens(calendar, timeframe, start_ts, end_ts))


def expected_trading_dates(
    calendar: CalendarDefinition,
    start_ts: datetime,
    end_ts: datetime,
) -> list[date]:
    """Return expected local trading dates between two timestamps."""

    if end_ts < start_ts:
        return []
    start_utc = _as_utc_timestamp(start_ts)
    end_utc = _as_utc_timestamp(end_ts)
    tz = ZoneInfo(calendar.timezone)
    start_local = start_utc.tz_convert(tz).to_pydatetime()
    end_local = end_utc.tz_convert(tz).to_pydatetime()
    current = start_local.date()
    final = end_local.date()
    dates: list[date] = []
    while current <= final:
        if calendar.is_trading_day(current):
            dates.append(current)
        current += timedelta(days=1)
    return dates


def expected_bar_opens(
    calendar: CalendarDefinition,
    timeframe: str,
    start_ts: datetime,
    end_ts: datetime,
) -> pd.DatetimeIndex:
    """Return expected UTC bar-open timestamps for an intraday timeframe."""

    if end_ts < start_ts:
        return pd.DatetimeIndex([], tz="UTC")
    minutes = TIMEFRAME_MINUTES[timeframe.lower()]
    start_utc = _as_utc_timestamp(start_ts)
    end_utc = _as_utc_timestamp(end_ts)
    if _is_24_7(calendar):
        return pd.date_range(start=start_utc, end=end_utc, freq=f"{minutes}min")

    tz = ZoneInfo(calendar.timezone)
    start_local = start_utc.tz_convert(tz).to_pydatetime()
    end_local = end_utc.tz_convert(tz).to_pydatetime()
    current = start_local.date()
    final = end_local.date()
    expected: list[pd.Timestamp] = []
    while current <= final:
        if calendar.is_trading_day(current):
            session_open = datetime.combine(current, _parse_time(calendar.session_open), tz)
            session_close = datetime.combine(current, _parse_time(calendar.session_close), tz)
            if session_close <= session_open:
                session_close += timedelta(days=1)
            opens = pd.date_range(
                start=session_open,
                end=session_close - timedelta(minutes=minutes),
                freq=f"{minutes}min",
            )
            if opens.size:
                for break_start, break_end in calendar.breaks:
                    local_break_start = datetime.combine(current, _parse_time(break_start), tz)
                    local_break_end = datetime.combine(current, _parse_time(break_end), tz)
                    opens = opens[(opens < local_break_start) | (opens >= local_break_end)]
                opens_utc = opens.tz_convert("UTC")
                for closed_start, closed_end in calendar.closed_ranges_utc:
                    closed_start_utc = _as_utc_timestamp(datetime.fromisoformat(closed_start))
                    closed_end_utc = _as_utc_timestamp(datetime.fromisoformat(closed_end))
                    opens_utc = opens_utc[
                        (opens_utc < closed_start_utc) | (opens_utc > closed_end_utc)
                    ]
                expected.extend(opens_utc[(opens_utc >= start_utc) & (opens_utc <= end_utc)])
        current += timedelta(days=1)
    return pd.DatetimeIndex(expected).sort_values()


def _parse_time(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def _is_24_7(calendar: CalendarDefinition) -> bool:
    return set(calendar.weekdays) == set(range(7)) and calendar.session_open == calendar.session_close


def _is_daily_timeframe(timeframe: str) -> bool:
    value = timeframe.lower()
    return value in {"1d", "daily"} or value.startswith("1d_") or value.endswith("_panama")


def _as_utc_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
