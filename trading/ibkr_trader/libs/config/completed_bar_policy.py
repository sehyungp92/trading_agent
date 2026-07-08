"""Shared completed-bar availability rules for live and backtest paths."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np

UTC = timezone.utc
RTH_CLOSE = time(16, 0)
ETH_DAILY_CLOSE = time(17, 0)
_USE_RTH_DEFAULT_TZ = "America/New_York"

_BAR_SIZE_TO_DELTA: dict[str, timedelta] = {
    "5 mins": timedelta(minutes=5),
    "5 min": timedelta(minutes=5),
    "15 mins": timedelta(minutes=15),
    "15 min": timedelta(minutes=15),
    "30 mins": timedelta(minutes=30),
    "30 min": timedelta(minutes=30),
    "1 hour": timedelta(hours=1),
    "1 hours": timedelta(hours=1),
    "4 hour": timedelta(hours=4),
    "4 hours": timedelta(hours=4),
}


def normalize_bar_size(bar_size_setting: str) -> str:
    return " ".join(str(bar_size_setting).strip().lower().split())


def is_daily_bar_size(bar_size_setting: str) -> bool:
    return normalize_bar_size(bar_size_setting) == "1 day"


def intraday_bar_timedelta(bar_size_setting: str) -> timedelta | None:
    return _BAR_SIZE_TO_DELTA.get(normalize_bar_size(bar_size_setting))


def align_completed_higher_timeframe_indices(
    lower_times: Sequence[Any] | np.ndarray,
    higher_times: Sequence[Any] | np.ndarray,
    *,
    unavailable_index: int = -1,
) -> np.ndarray:
    """Map lower-TF timestamps to the most recent completed higher-TF bar.

    ``unavailable_index`` controls bars before the first completed higher-TF
    bar. The strict default keeps unavailable higher-TF context explicit;
    legacy callers that deliberately pre-seed context can pass ``0``.
    """
    higher = np.asarray(higher_times)
    lower = np.asarray(lower_times)
    if len(higher) == 0:
        return np.full(len(lower), unavailable_index, dtype=np.int64)
    idx = np.searchsorted(higher, lower, side="left").astype(np.int64) - 1
    idx = np.minimum(idx, len(higher) - 1)
    if unavailable_index >= 0:
        return np.maximum(idx, unavailable_index)
    return np.where(idx < 0, unavailable_index, idx).astype(np.int64, copy=False)


def align_completed_daily_session_indices(
    lower_times: Sequence[Any] | np.ndarray,
    daily_times: Sequence[Any] | np.ndarray,
    *,
    unavailable_index: int = -1,
) -> np.ndarray:
    """Map intraday timestamps to the most recent completed daily session."""
    return align_completed_higher_timeframe_indices(
        _normalize_dates(lower_times),
        _normalize_dates(daily_times),
        unavailable_index=unavailable_index,
    )


def filter_completed_live_bars(
    bars: Sequence[Any],
    *,
    bar_size_setting: str,
    use_rth: bool,
    end_datetime: Any = "",
    as_of: datetime | None = None,
    market_tz: str = _USE_RTH_DEFAULT_TZ,
) -> list[Any]:
    """Drop the in-progress tail bar for live historical requests.

    IBKR's ``reqHistoricalData`` with ``endDateTime=""`` and
    ``keepUpToDate=False`` returns an in-progress last bar. This helper makes
    that policy explicit and deterministic for both live and test callers.
    """
    if not bars:
        return []
    if end_datetime not in ("", None):
        return list(bars)

    filtered = list(bars)
    current_time = _ensure_aware(as_of or datetime.now(UTC))

    if is_daily_bar_size(bar_size_setting):
        return _filter_daily_tail(
            filtered,
            current_time=current_time,
            use_rth=use_rth,
            market_tz=market_tz,
        )

    bar_delta = intraday_bar_timedelta(bar_size_setting)
    if bar_delta is None:
        return filtered

    while filtered:
        last_ts = _coerce_datetime(getattr(filtered[-1], "date", None))
        if last_ts is None:
            break
        if last_ts + bar_delta <= current_time:
            break
        filtered.pop()
    return filtered


def _filter_daily_tail(
    bars: list[Any],
    *,
    current_time: datetime,
    use_rth: bool,
    market_tz: str,
) -> list[Any]:
    if not bars:
        return bars

    zone = ZoneInfo(market_tz)
    session_now = current_time.astimezone(zone)
    current_session_date = session_now.date()
    daily_close = RTH_CLOSE if use_rth else ETH_DAILY_CLOSE
    session_complete = session_now.weekday() < 5 and session_now.time() >= daily_close

    while bars:
        last_date = _coerce_date(getattr(bars[-1], "date", None), default_tz=zone)
        if last_date is None:
            break
        if last_date < current_session_date:
            break
        if last_date == current_session_date and session_complete:
            break
        bars.pop()
    return bars


def _normalize_dates(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[D]")

    normalized: list[np.datetime64] = []
    for value in values:
        coerced = _coerce_date(value)
        if coerced is None:
            raise ValueError(f"Unsupported timestamp value: {value!r}")
        normalized.append(np.datetime64(coerced.isoformat()))
    return np.asarray(normalized, dtype="datetime64[D]")


def _coerce_date(value: Any, *, default_tz: timezone | ZoneInfo = UTC) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    coerced = _coerce_datetime(value, default_tz=default_tz)
    if coerced is None:
        return None
    return coerced.astimezone(default_tz).date()


def _coerce_datetime(value: Any, *, default_tz: timezone | ZoneInfo = UTC) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_aware(value, default_tz=default_tz)
    if isinstance(value, np.datetime64):
        epoch_seconds = value.astype("datetime64[s]").astype(np.int64)
        return datetime.fromtimestamp(int(epoch_seconds), tz=UTC)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=default_tz)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit() and len(text) == 8:
            parsed = datetime.strptime(text, "%Y%m%d")
            return parsed.replace(tzinfo=default_tz)
        for candidate in (text, text.replace(" ", "T", 1)):
            try:
                return _ensure_aware(datetime.fromisoformat(candidate), default_tz=default_tz)
            except ValueError:
                pass
        for fmt in ("%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=default_tz)
            except ValueError:
                continue
    return None


def _ensure_aware(
    value: datetime,
    *,
    default_tz: timezone | ZoneInfo = UTC,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=default_tz)
    return value
