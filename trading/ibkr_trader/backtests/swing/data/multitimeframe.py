"""Multi-timeframe data utilities for 15m-primary swing strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample_15m_to_30m(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample IBKR start-stamped 15m bars to start-stamped 30m bars.

    The ETF parquet files downloaded from IBKR use bar start timestamps. This
    keeps the derived 30m bars on the same convention as the existing 1h files:
    a bar stamped 09:30 covers [09:30, 10:00) and is only visible after 10:00.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df_15m.columns:
        agg["volume"] = "sum"
    result = df_15m.resample("30min", label="left", closed="left").agg(agg)
    return result.dropna(subset=["open", "close"])


def resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample IBKR start-stamped 1h bars to start-stamped 4h bars."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df_1h.columns:
        agg["volume"] = "sum"
    result = df_1h.resample("4h", label="left", closed="left").agg(agg)
    return result.dropna(subset=["open", "close"])


def align_30m_to_15m(df_15m: pd.DataFrame, df_30m: pd.DataFrame) -> np.ndarray:
    return _align_start_stamped_completed(df_15m.index, df_30m.index, "15min", "30min")


def align_1h_to_15m(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> np.ndarray:
    return _align_start_stamped_completed(df_15m.index, df_1h.index, "15min", "1h")


def align_4h_to_15m(df_15m: pd.DataFrame, df_4h: pd.DataFrame) -> np.ndarray:
    return _align_start_stamped_completed(df_15m.index, df_4h.index, "15min", "4h")


def align_15m_to_30m(df_15m: pd.DataFrame, df_30m: pd.DataFrame) -> np.ndarray:
    return align_30m_to_15m(df_15m, df_30m)


def align_15m_to_1h(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> np.ndarray:
    return align_1h_to_15m(df_15m, df_1h)


def align_15m_to_4h(df_15m: pd.DataFrame, df_4h: pd.DataFrame) -> np.ndarray:
    return align_4h_to_15m(df_15m, df_4h)


def align_daily_to_15m(df_15m: pd.DataFrame, df_daily: pd.DataFrame) -> np.ndarray:
    return _align_daily_previous_session(df_15m.index, df_daily.index)


def _align_start_stamped_completed(
    lower_times: pd.DatetimeIndex,
    higher_times: pd.DatetimeIndex,
    lower_freq: str,
    higher_freq: str,
) -> np.ndarray:
    """Map each lower start timestamp to the latest completed higher bar."""
    if len(higher_times) == 0:
        return np.full(len(lower_times), -1, dtype=np.int64)
    lower_close = pd.DatetimeIndex(lower_times) + pd.Timedelta(lower_freq)
    higher_close = pd.DatetimeIndex(higher_times) + pd.Timedelta(higher_freq)
    idx = np.searchsorted(higher_close.values, lower_close.values, side="right").astype(np.int64) - 1
    return np.minimum(idx, len(higher_close) - 1)


def _align_daily_previous_session(lower_times: pd.DatetimeIndex, daily_times: pd.DatetimeIndex) -> np.ndarray:
    """Map intraday bars to the previous completed daily session."""
    lower_dates = pd.DatetimeIndex(lower_times).normalize().values.astype("datetime64[D]")
    daily_dates = pd.DatetimeIndex(daily_times).normalize().values.astype("datetime64[D]")
    if len(daily_dates) == 0:
        return np.full(len(lower_dates), -1, dtype=np.int64)
    idx = np.searchsorted(daily_dates, lower_dates, side="left").astype(np.int64) - 1
    return np.minimum(idx, len(daily_dates) - 1)
