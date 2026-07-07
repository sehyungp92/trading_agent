"""Canonical candle timestamp helpers for backtest/live parity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from crypto_trader.core.models import TimeFrame


@dataclass(frozen=True, slots=True)
class CandleTimes:
    """Open, close, and strategy availability times for a completed candle."""

    open_time: datetime
    close_time: datetime
    available_at: datetime


def ensure_utc(ts: datetime) -> datetime:
    """Return ``ts`` as a UTC-aware datetime, assuming UTC for naive inputs."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def timeframe_delta(tf: TimeFrame) -> timedelta:
    return timedelta(minutes=tf.minutes)


def candle_times_from_timestamp(
    timestamp: datetime,
    tf: TimeFrame,
    *,
    timestamp_policy: str,
) -> CandleTimes:
    """Normalize a source timestamp into canonical completed-candle times."""
    ts = ensure_utc(timestamp)
    duration = timeframe_delta(tf)
    if timestamp_policy == "open_time":
        open_time = ts
        close_time = ts + duration
    elif timestamp_policy == "close_time":
        close_time = ts
        open_time = ts - duration
    else:
        raise ValueError(f"Unsupported timestamp_policy={timestamp_policy!r}")
    return CandleTimes(
        open_time=open_time,
        close_time=close_time,
        available_at=close_time,
    )


def candle_open_from_ms(
    *,
    tf: TimeFrame,
    open_ms: int | float | None = None,
    close_ms: int | float | None = None,
) -> datetime:
    """Return a candle open time from exchange millisecond fields."""
    if open_ms is not None:
        return ensure_utc(datetime.fromtimestamp(float(open_ms) / 1000, tz=timezone.utc))
    if close_ms is None:
        raise ValueError("expected either open_ms or close_ms")
    close_time = ensure_utc(datetime.fromtimestamp(float(close_ms) / 1000, tz=timezone.utc))
    return close_time - timeframe_delta(tf)


def higher_timeframe_open(primary_open: datetime, higher_tf: TimeFrame) -> datetime:
    """Return the UTC open time of the higher-TF candle containing ``primary_open``."""
    ts = ensure_utc(primary_open).replace(second=0, microsecond=0)
    if higher_tf == TimeFrame.D1:
        return ts.replace(hour=0, minute=0)

    period = higher_tf.minutes
    minutes_since_midnight = ts.hour * 60 + ts.minute
    start_minutes = (minutes_since_midnight // period) * period
    return ts.replace(
        hour=start_minutes // 60,
        minute=start_minutes % 60,
    )


def completes_higher_timeframe(
    primary_open: datetime,
    primary_tf: TimeFrame,
    higher_tf: TimeFrame,
) -> bool:
    """True when a primary bar is the final sub-bar of ``higher_tf``."""
    if higher_tf.minutes <= primary_tf.minutes:
        return False
    if higher_tf.minutes % primary_tf.minutes != 0:
        return False
    primary_close = ensure_utc(primary_open) + timeframe_delta(primary_tf)
    higher_close = higher_timeframe_open(primary_open, higher_tf) + timeframe_delta(higher_tf)
    return primary_close == higher_close
