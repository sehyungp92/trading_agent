"""Market calendar with holiday and half-day awareness for US equity and CME futures.

Pure-stdlib implementation, no external dependencies.
Year-cached holiday computation for fast repeated lookups.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from enum import Enum
from functools import lru_cache
from zoneinfo import ZoneInfo


class AssetClass(Enum):
    EQUITY = "equity"
    CME_FUTURES = "cme_futures"


def _observe(day: date) -> date:
    """Apply observed-holiday rules for weekend dates."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    first_occurrence = first + timedelta(days=delta)
    return first_occurrence + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    delta = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=delta)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _good_friday(year: int) -> date:
    return _easter_sunday(year) - timedelta(days=2)


@lru_cache(maxsize=16)
def _equity_holidays(year: int) -> frozenset[date]:
    holidays: list[date] = [
        _observe(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _good_friday(year),
        _last_weekday(year, 5, 0),
        _observe(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observe(date(year, 12, 25)),
    ]
    if year >= 2022:
        holidays.append(_observe(date(year, 6, 19)))
    return frozenset(holidays)


@lru_cache(maxsize=16)
def _cme_holidays(year: int) -> frozenset[date]:
    return frozenset(
        [
            _observe(date(year, 1, 1)),
            _good_friday(year),
            _last_weekday(year, 5, 0),
            _observe(date(year, 7, 4)),
            _nth_weekday(year, 9, 0, 1),
            _nth_weekday(year, 11, 3, 4),
            _observe(date(year, 12, 25)),
        ]
    )


@lru_cache(maxsize=16)
def _half_days(year: int) -> frozenset[date]:
    days: list[date] = []

    july4_observed = _observe(date(year, 7, 4))
    before_july4 = july4_observed - timedelta(days=1)
    if before_july4.weekday() < 5:
        days.append(before_july4)

    thanksgiving = _nth_weekday(year, 11, 3, 4)
    days.append(thanksgiving + timedelta(days=1))

    xmas_observed = _observe(date(year, 12, 25))
    xmas_eve = xmas_observed - timedelta(days=1)
    if xmas_eve.weekday() < 5:
        days.append(xmas_eve)

    return frozenset(days)


class MarketCalendar:
    """Holiday and half-day calendar for US equity and CME futures markets."""

    def is_market_holiday(self, day: date, asset_class: AssetClass = AssetClass.EQUITY) -> bool:
        if day.weekday() >= 5:
            return True
        holidays = _equity_holidays(day.year) if asset_class == AssetClass.EQUITY else _cme_holidays(day.year)
        return day in holidays

    def is_half_day(self, day: date, asset_class: AssetClass = AssetClass.EQUITY) -> bool:
        if asset_class != AssetClass.EQUITY or day.weekday() >= 5:
            return False
        return day in _half_days(day.year)

    def is_trading_day(self, day: date, asset_class: AssetClass = AssetClass.EQUITY) -> bool:
        if day.weekday() >= 5:
            return False
        holidays = _equity_holidays(day.year) if asset_class == AssetClass.EQUITY else _cme_holidays(day.year)
        return day not in holidays

    def next_trading_day(self, day: date, asset_class: AssetClass = AssetClass.EQUITY) -> date:
        candidate = day + timedelta(days=1)
        while not self.is_trading_day(candidate, asset_class):
            candidate += timedelta(days=1)
        return candidate

    def market_close_time_et(self, day: date, asset_class: AssetClass = AssetClass.EQUITY) -> time:
        if self.is_half_day(day, asset_class):
            return time(13, 0)
        return time(16, 0)

    def is_entry_blocked(
        self,
        now_utc: datetime,
        asset_class: AssetClass = AssetClass.EQUITY,
    ) -> str | None:
        now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
        today = now_et.date()

        if self.is_market_holiday(today, asset_class):
            return f"Market holiday: {today}"

        if self.is_half_day(today, asset_class) and now_et.hour >= 12:
            return f"Half-day session: no new entries after 12:00 ET ({today})"

        if asset_class == AssetClass.EQUITY:
            if now_et.time() < time(9, 30):
                return f"Pre-market: entries blocked before 09:30 ET ({today})"
            close = self.market_close_time_et(today, asset_class)
            if now_et.time() >= close:
                return f"Post-market: entries blocked after {close.strftime('%H:%M')} ET ({today})"

        return None

