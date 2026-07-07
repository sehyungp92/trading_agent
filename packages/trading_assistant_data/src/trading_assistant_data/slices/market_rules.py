"""Market calendar rule adapters for data slices."""

from __future__ import annotations

from pathlib import Path

from trading_assistant_data.calendars.cme import calendar_definition as _cme_calendar
from trading_assistant_data.calendars.crypto import calendar_definition as _crypto_calendar
from trading_assistant_data.calendars.krx import calendar_definition as _krx_calendar
from trading_assistant_data.calendars.krx import (
    kis_intraday_calendar_definition as _krx_kis_intraday_calendar,
)
from trading_assistant_data.calendars.us_equities import (
    calendar_definition as _us_equities_calendar,
)


def crypto_calendar():
    return _crypto_calendar()


def cme_calendar():
    return _cme_calendar()


def krx_calendar(holiday_path: Path | None = None):
    return _krx_calendar(holiday_path)


def krx_kis_intraday_calendar(holiday_path: Path | None = None):
    return _krx_kis_intraday_calendar(holiday_path)


def us_equities_calendar():
    return _us_equities_calendar()
