"""Parquet-based bar data caching."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_bars(df: pd.DataFrame, path: Path) -> None:
    """Write a bar DataFrame to Parquet with timestamp index preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)


def load_bars(path: Path) -> pd.DataFrame:
    """Read a Parquet bar file, returning a DataFrame with DatetimeIndex."""
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse the index as datetime
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def bar_path(data_dir: Path, symbol: str, timeframe: str) -> Path:
    """Canonical path for cached bar file."""
    return data_dir / f"{symbol}_{timeframe}.parquet"


def load_or_download(
    symbol: str,
    timeframe: str,
    data_dir: Path,
) -> pd.DataFrame | None:
    """Load from cache if it exists, otherwise return None.

    Actual download is handled separately via the async downloader.
    """
    path = bar_path(data_dir, symbol, timeframe)
    if path.exists():
        return load_bars(path)
    return None
