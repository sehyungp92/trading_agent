"""Data preprocessing: gap filling, timezone normalization, alignment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from libs.config.completed_bar_policy import (
    align_completed_daily_session_indices,
    align_completed_higher_timeframe_indices,
)


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to Regular Trading Hours only (09:30-16:00 ET).

    Removes pre-market and after-hours bars so that signal detection
    runs exclusively on tradeable bars.  Bars at 09:00 ET are included
    to provide indicator context. Input bars are expected to be timestamped
    by interval start; replay loaders may shift them to close availability.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    # Keep bars from 09:00 (540) through 15:59 (959) on weekdays
    mask = (minutes >= 540) & (minutes < 960) & (idx_et.weekday < 5)
    return df.loc[mask]


def normalize_timezone(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Ensure DatetimeIndex is in the specified timezone."""
    if df.index.tz is None:
        df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)
    return df


def fill_gaps(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Forward-fill missing timestamps, mark gaps.

    Inserts rows for missing timestamps with NaN OHLCV and a ``gap=True``
    column.  Existing rows get ``gap=False``.
    """
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_idx)
    df["gap"] = df["close"].isna()
    # Forward-fill close only (for indicator seeding); leave OHLCV as NaN for gap bars
    df["close"] = df["close"].ffill()
    return df


def mark_invalid_blocks(df: pd.DataFrame, max_consecutive: int = 5) -> pd.DataFrame:
    """Mark contiguous gap blocks longer than max_consecutive.

    Adds ``invalid=True`` for bars inside blocks where the engine should skip
    entry evaluation.
    """
    if "gap" not in df.columns:
        df["invalid"] = False
        return df

    gap_groups = (~df["gap"]).cumsum()
    gap_lengths = df.groupby(gap_groups)["gap"].transform("sum")
    df["invalid"] = df["gap"] & (gap_lengths > max_consecutive)
    return df


def _vectorized_align(lower_times, higher_times) -> np.ndarray:
    """Compatibility wrapper over the shared completed-bar alignment policy."""
    return align_completed_higher_timeframe_indices(
        lower_times,
        higher_times,
        unavailable_index=0,
    )


def align_daily_to_hourly(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> np.ndarray:
    """Map each hourly bar to the index of the most recent *completed* daily bar.

    Returns an integer array of length ``len(hourly_df)`` where each element
    is the positional index into ``daily_df``.  This prevents look-ahead bias:
    a daily bar is only available after its close.

    Daily bars are assumed to represent the close of that calendar day.
    An hourly bar at time ``t`` uses the daily bar whose date is strictly
    before ``t``'s date (i.e., yesterday's daily bar during the current day,
    switching to today's daily bar only on the first bar of the next day).
    """
    return align_completed_daily_session_indices(
        hourly_df.index.values,
        daily_df.index.values,
        unavailable_index=0,
    )


def align_daily_to_5m(
    five_min_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> np.ndarray:
    """Map each 5-minute bar to the index of the most recent *completed* daily bar.

    Uses date-normalised alignment (same pattern as ``align_daily_to_hourly``)
    to avoid the look-ahead bias that occurs when ``align_higher_tf_to_5m``
    is used with daily bars (whose left-edge label is today's date at 00:00 UTC,
    making today's incomplete daily bar appear complete).
    """
    return align_completed_daily_session_indices(
        five_min_df.index.values,
        daily_df.index.values,
        unavailable_index=0,
    )


def align_daily_to_15m(
    fifteen_min_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> np.ndarray:
    """Map each 15-minute bar to the index of the most recent *completed* daily bar.

    Uses date-normalised alignment (same pattern as ``align_daily_to_hourly``)
    to avoid the look-ahead bias that occurs when ``align_higher_tf_to_15m``
    is used with daily bars.
    """
    return align_completed_daily_session_indices(
        fifteen_min_df.index.values,
        daily_df.index.values,
        unavailable_index=0,
    )


@dataclass
class NumpyBars:
    """Contiguous numpy arrays extracted from a DataFrame."""

    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    times: np.ndarray  # datetime64

    def __len__(self) -> int:
        return len(self.closes)


def resample_1h_to_4h(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV bars to 4H using standard aggregation.

    Uses UTC boundaries (00:00, 04:00, 08:00, 12:00, 16:00, 20:00).
    The resulting DataFrame has the same timezone as the input.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in hourly_df.columns:
        agg["volume"] = "sum"

    resampled = hourly_df.resample("4h", offset="0h", label="right").agg(agg)
    # Drop rows where all OHLC are NaN (incomplete periods at boundaries)
    resampled = resampled.dropna(subset=["open", "close"])
    return resampled


def align_4h_to_hourly(
    hourly_df: pd.DataFrame,
    four_hour_df: pd.DataFrame,
) -> np.ndarray:
    """Map each hourly bar to the index of the most recent *completed* 4H bar.

    Returns an integer array of length ``len(hourly_df)`` where each element
    is the positional index into ``four_hour_df``.

    A 4H bar is available only after its close. For example, the 4H bar
    closing at 04:00 is available starting from the 05:00 hourly bar.
    The hourly bars at 01:00-04:00 use the previous 4H bar (closing at 00:00).
    """
    return _vectorized_align(hourly_df.index.values, four_hour_df.index.values)


def build_numpy_arrays(df: pd.DataFrame) -> NumpyBars:
    """Extract OHLCV columns as contiguous float64 arrays."""
    return NumpyBars(
        opens=np.ascontiguousarray(df["open"].values, dtype=np.float64),
        highs=np.ascontiguousarray(df["high"].values, dtype=np.float64),
        lows=np.ascontiguousarray(df["low"].values, dtype=np.float64),
        closes=np.ascontiguousarray(df["close"].values, dtype=np.float64),
        volumes=np.ascontiguousarray(df["volume"].values, dtype=np.float64) if "volume" in df.columns else np.zeros(len(df)),
        times=df.index.values,
    )


# ---------------------------------------------------------------------------
# 1-minute data support (Apex strategy backtest)
# ---------------------------------------------------------------------------


def filter_eth(df: pd.DataFrame, include_evening: bool = False) -> pd.DataFrame:
    """Filter DataFrame to Extended Trading Hours for NQ futures.

    Primary window: 04:00-16:15 ET on weekdays.
    Evening window: 18:00-19:45 ET (optional).
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    # Primary: 04:00 (240) to 16:15 (975) on weekdays
    primary = (minutes >= 240) & (minutes < 975) & (idx_et.weekday < 5)
    if include_evening:
        # Evening: 18:00 (1080) to 19:45 (1185)
        evening = (minutes >= 1080) & (minutes < 1185) & (idx_et.weekday < 5)
        return df.loc[primary | evening]
    return df.loc[primary]


def resample_1m_to_1h(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars to 1H using standard OHLCV aggregation.

    Uses UTC-aligned hour boundaries.  Drops incomplete bars.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in minute_df.columns:
        agg["volume"] = "sum"
    resampled = minute_df.resample("1h", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def resample_1m_to_4h(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars to 4H using standard OHLCV aggregation."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in minute_df.columns:
        agg["volume"] = "sum"
    resampled = minute_df.resample("4h", offset="0h", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def resample_1m_to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars to daily bars.

    Uses RTH (09:30-16:00 ET) bars only for the daily aggregation to match
    how daily bars are conventionally defined for NQ futures.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = minute_df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    rth_mask = (minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)
    rth_df = minute_df.loc[rth_mask]

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in rth_df.columns:
        agg["volume"] = "sum"
    resampled = rth_df.resample("1D").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def align_higher_tf_to_minute(
    minute_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
) -> np.ndarray:
    """Map each minute bar to the index of the most recent *completed* higher-TF bar.

    Returns an integer array of length ``len(minute_df)`` where each element
    is the positional index into ``higher_tf_df``.  No look-ahead: a higher-TF
    bar is only available after its close timestamp.
    """
    return _vectorized_align(minute_df.index.values, higher_tf_df.index.values)


# ---------------------------------------------------------------------------
# 5-minute data support (NQDTC strategy backtest)
# ---------------------------------------------------------------------------


def resample_5m_to_30m(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute bars to 30-minute using standard OHLCV aggregation.

    Uses UTC-aligned 30-minute boundaries. Drops incomplete bars.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in five_min_df.columns:
        agg["volume"] = "sum"
    resampled = five_min_df.resample("30min", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def resample_5m_to_1h(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute bars to 1H using standard OHLCV aggregation."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in five_min_df.columns:
        agg["volume"] = "sum"
    resampled = five_min_df.resample("1h", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def resample_5m_to_4h(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute bars to 4H using standard OHLCV aggregation."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in five_min_df.columns:
        agg["volume"] = "sum"
    resampled = five_min_df.resample("4h", offset="0h", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def resample_5m_to_daily(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute bars to daily bars.

    Uses RTH (09:30-16:00 ET) bars only for daily aggregation to match
    how daily bars are conventionally defined for NQ futures.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = five_min_df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    rth_mask = (minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)
    rth_df = five_min_df.loc[rth_mask]

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in rth_df.columns:
        agg["volume"] = "sum"
    resampled = rth_df.resample("1D").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def align_higher_tf_to_5m(
    five_min_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
) -> np.ndarray:
    """Map each 5-minute bar to the index of the most recent *completed* higher-TF bar.

    Returns an integer array of length ``len(five_min_df)`` where each element
    is the positional index into ``higher_tf_df``.  No look-ahead: a higher-TF
    bar is only available after its close timestamp.
    """
    return _vectorized_align(five_min_df.index.values, higher_tf_df.index.values)


# ---------------------------------------------------------------------------
# 15-minute data support (VdubusNQ strategy backtest)
# ---------------------------------------------------------------------------


def filter_vdubus_session(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to VdubusNQ trading hours: RTH 09:30-16:00 + Evening 19:00-22:30 ET.

    Includes 09:00-09:30 for indicator context (entry is restricted until 09:40).
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    # Primary RTH: 09:00 (540) through 16:00 (960) on weekdays
    rth = (minutes >= 540) & (minutes < 960) & (idx_et.weekday < 5)
    # Evening: 19:00 (1140) through 22:30 (1350)
    evening = (minutes >= 1140) & (minutes < 1350) & (idx_et.weekday < 5)
    return df.loc[rth | evening]


def resample_15m_to_1h(fifteen_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute bars to 1H using standard OHLCV aggregation.

    Uses UTC-aligned hour boundaries with right-edge labeling to prevent
    look-ahead bias. Drops incomplete bars.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in fifteen_min_df.columns:
        agg["volume"] = "sum"
    resampled = fifteen_min_df.resample("1h", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def resample_15m_to_4h(fifteen_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute bars to 4H using standard OHLCV aggregation."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in fifteen_min_df.columns:
        agg["volume"] = "sum"
    resampled = fifteen_min_df.resample("4h", offset="0h", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def resample_15m_to_daily(fifteen_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute bars to daily bars.

    Uses RTH (09:30-16:00 ET) bars only for daily aggregation to match
    how daily bars are conventionally defined for NQ futures.
    """
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = fifteen_min_df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    rth_mask = (minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)
    rth_df = fifteen_min_df.loc[rth_mask]

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in rth_df.columns:
        agg["volume"] = "sum"
    resampled = rth_df.resample("1D").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def align_higher_tf_to_15m(
    fifteen_min_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
) -> np.ndarray:
    """Map each 15-minute bar to the index of the most recent *completed* higher-TF bar.

    Returns an integer array of length ``len(fifteen_min_df)`` where each element
    is the positional index into ``higher_tf_df``.  No look-ahead: a higher-TF
    bar is only available after its close timestamp.
    """
    return _vectorized_align(fifteen_min_df.index.values, higher_tf_df.index.values)


def resample_5m_to_15m(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute bars to 15-minute using standard OHLCV aggregation.

    Used for micro-trigger alignment (VdubusNQ).
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in five_min_df.columns:
        agg["volume"] = "sum"
    resampled = five_min_df.resample("15min", label="right").agg(agg)
    return resampled.dropna(subset=["open", "close"])


def align_5m_to_15m(
    five_min_df: pd.DataFrame,
    fifteen_min_df: pd.DataFrame,
) -> np.ndarray:
    """Map each 5-minute bar to the index of the containing 15-minute bar.

    Each 5m bar maps to the 15m bar whose close time encompasses it.
    Used for micro-trigger inner loop in VdubusNQ backtest.
    """
    return _vectorized_align(five_min_df.index.values, fifteen_min_df.index.values)
