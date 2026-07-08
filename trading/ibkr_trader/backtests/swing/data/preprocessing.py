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
    to provide indicator context (entry is still restricted by the engine
    until 09:45 ET).
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


def _vectorized_align(
    target_times: np.ndarray,
    source_times: np.ndarray,
    *,
    unavailable_index: int = -1,
) -> np.ndarray:
    """Compatibility wrapper over the shared completed-bar alignment policy."""
    return align_completed_higher_timeframe_indices(
        target_times,
        source_times,
        unavailable_index=unavailable_index,
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
        unavailable_index=-1,
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
    """Resample completed 1H OHLCV bars to completed 4H bars.

    Uses UTC boundaries (00:00, 04:00, 08:00, 12:00, 16:00, 20:00).
    Output rows are labeled at the 4H window close/availability time and only
    include windows with four valid 1H OHLC bars.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in hourly_df.columns:
        agg["volume"] = "sum"

    resampler = hourly_df.resample("4h", offset="0h", label="right", closed="left")
    resampled = resampler.agg(agg)
    complete_ohlc = hourly_df[["open", "high", "low", "close"]].notna().all(axis=1)
    source_counts = complete_ohlc.astype(int).resample(
        "4h",
        offset="0h",
        label="right",
        closed="left",
    ).sum()
    resampled = resampled.loc[source_counts >= 4]
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
