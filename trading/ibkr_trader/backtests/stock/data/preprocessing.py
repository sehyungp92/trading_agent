"""Data preprocessing: gap filling, timezone normalization, alignment, resampling.

Reusable functions from momentum backtest + stock-specific additions
(30m→4h for ALCB, 5m→30m for IARIC).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core utilities (shared with momentum)
# ---------------------------------------------------------------------------


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to Regular Trading Hours only (09:30-16:00 ET).

    Bars at 09:00 ET are included for indicator context.
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
    """Forward-fill missing timestamps, mark gaps."""
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_idx)
    df["gap"] = df["close"].isna()
    df["close"] = df["close"].ffill()
    return df


def mark_invalid_blocks(df: pd.DataFrame, max_consecutive: int = 5) -> pd.DataFrame:
    """Mark contiguous gap blocks longer than max_consecutive."""
    if "gap" not in df.columns:
        df["invalid"] = False
        return df

    gap_groups = (~df["gap"]).cumsum()
    gap_lengths = df.groupby(gap_groups)["gap"].transform("sum")
    df["invalid"] = df["gap"] & (gap_lengths > max_consecutive)
    return df


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
# Daily ↔ intraday alignment
# ---------------------------------------------------------------------------


def align_daily_to_intraday(
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> np.ndarray:
    """Map each intraday bar to the index of the most recent *completed* daily bar.

    Returns an integer array of length ``len(intraday_df)`` where each element
    is the positional index into ``daily_df``.  No look-ahead: a daily bar is
    only available after its close (today's daily bar is available starting
    from the first bar of the next day).
    """
    daily_dates = daily_df.index.normalize()
    intraday_dates = intraday_df.index.normalize()

    idx_map = np.empty(len(intraday_df), dtype=np.int64)
    daily_pos = 0

    for i in range(len(intraday_df)):
        h_date = intraday_dates[i]
        while daily_pos < len(daily_dates) - 1 and daily_dates[daily_pos + 1] < h_date:
            daily_pos += 1
        if daily_dates[daily_pos] < h_date:
            idx_map[i] = daily_pos
        elif daily_pos > 0:
            idx_map[i] = daily_pos - 1
        else:
            idx_map[i] = 0

    return idx_map


def align_higher_tf(
    lower_tf_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
) -> np.ndarray:
    """Map each lower-TF bar to the index of the most recent *completed* higher-TF bar.

    Returns an integer array of length ``len(lower_tf_df)`` where each element
    is the positional index into ``higher_tf_df``.  No look-ahead.
    """
    htf_times = higher_tf_df.index
    ltf_times = lower_tf_df.index

    idx_map = np.empty(len(lower_tf_df), dtype=np.int64)
    htf_pos = 0

    for i in range(len(ltf_times)):
        t = ltf_times[i]
        while htf_pos < len(htf_times) - 1 and htf_times[htf_pos + 1] < t:
            htf_pos += 1
        if htf_times[htf_pos] < t:
            idx_map[i] = htf_pos
        elif htf_pos > 0:
            idx_map[i] = htf_pos - 1
        else:
            idx_map[i] = 0

    return idx_map


# ---------------------------------------------------------------------------
# Stock-specific resampling
# ---------------------------------------------------------------------------


_OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def _agg_dict(df: pd.DataFrame) -> dict:
    agg = dict(_OHLCV_AGG)
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return agg


def resample_30m_to_4h(thirty_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 30-minute bars to 4H using standard OHLCV aggregation.

    ALCB uses 4H bars constructed from 8×30m bars. Uses UTC-aligned 4H
    boundaries (00:00, 04:00, 08:00, 12:00, 16:00, 20:00).
    """
    resampled = thirty_min_df.resample("4h", offset="0h").agg(_agg_dict(thirty_min_df))
    return resampled.dropna(subset=["open", "close"])


def resample_5m_to_30m(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute bars to 30-minute using standard OHLCV aggregation.

    IARIC uses 30m bars for AVWAP breakdown exit checks.
    """
    resampled = five_min_df.resample("30min").agg(_agg_dict(five_min_df))
    return resampled.dropna(subset=["open", "close"])


def resample_5m_to_1h(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute bars to 1H."""
    resampled = five_min_df.resample("1h").agg(_agg_dict(five_min_df))
    return resampled.dropna(subset=["open", "close"])


def resample_30m_to_daily(thirty_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 30m bars to daily bars using RTH only (09:30-16:00 ET)."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = thirty_min_df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    rth_mask = (minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)
    rth_df = thirty_min_df.loc[rth_mask]

    resampled = rth_df.resample("1D").agg(_agg_dict(rth_df))
    return resampled.dropna(subset=["open", "close"])


def resample_5m_to_daily(five_min_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5m bars to daily bars using RTH only (09:30-16:00 ET)."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    idx_et = five_min_df.index.tz_convert(et)
    minutes = idx_et.hour * 60 + idx_et.minute
    rth_mask = (minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)
    rth_df = five_min_df.loc[rth_mask]

    resampled = rth_df.resample("1D").agg(_agg_dict(rth_df))
    return resampled.dropna(subset=["open", "close"])


# ---------------------------------------------------------------------------
# Rolling 4H bar construction (ALCB intraday replay)
# ---------------------------------------------------------------------------


def rolling_30m_to_4h(bars_30m: list[tuple], bar_count: int = 8) -> list[tuple]:
    """Build 4H bars from a rolling window of 30m bars.

    Used during intraday replay: as each 30m bar arrives, check if we have
    a complete 8-bar group and emit a 4H bar.

    Args:
        bars_30m: List of (time, open, high, low, close, volume) tuples.
        bar_count: Number of 30m bars per 4H bar (default 8).

    Returns:
        List of 4H bar tuples. May be empty if not enough bars yet.
    """
    if len(bars_30m) < bar_count:
        return []

    result = []
    for start in range(0, len(bars_30m) - bar_count + 1, bar_count):
        chunk = bars_30m[start:start + bar_count]
        t = chunk[0][0]
        o = chunk[0][1]
        h = max(b[2] for b in chunk)
        lo = min(b[3] for b in chunk)
        c = chunk[-1][4]
        v = sum(b[5] for b in chunk)
        result.append((t, o, h, lo, c, v))
    return result


# ---------------------------------------------------------------------------
# Trading calendar helpers
# ---------------------------------------------------------------------------


def get_trading_dates(daily_df: pd.DataFrame) -> list:
    """Extract unique trading dates from a daily bar DataFrame."""
    return sorted(daily_df.index.normalize().unique().tolist())


def trading_days_between(start: pd.Timestamp, end: pd.Timestamp, trading_dates: list) -> int:
    """Count trading days between two timestamps (exclusive of start, inclusive of end)."""
    start_date = start.normalize()
    end_date = end.normalize()
    return sum(1 for d in trading_dates if start_date < d <= end_date)
