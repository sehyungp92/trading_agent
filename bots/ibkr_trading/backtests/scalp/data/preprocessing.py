from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class NumpyBars:
    times: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))
    opens: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    highs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    lows: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    closes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    volumes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class NumpyTicks:
    timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))
    prices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    sizes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    sides: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    bid_prices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    ask_prices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __len__(self) -> int:
        return len(self.timestamps)


def empty_bars() -> NumpyBars:
    return NumpyBars()


def empty_ticks() -> NumpyTicks:
    return NumpyTicks()


def build_numpy_bars(df: pd.DataFrame) -> NumpyBars:
    if df.empty:
        return empty_bars()
    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" in frame.columns:
            frame.index = pd.to_datetime(frame["timestamp"], utc=True)
        elif "time" in frame.columns:
            frame.index = pd.to_datetime(frame["time"], utc=True)
        else:
            frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC")
    else:
        frame.index = frame.index.tz_localize("UTC")

    def col(*names: str, default: float = 0.0) -> np.ndarray:
        for name in names:
            if name in frame.columns:
                return frame[name].to_numpy(dtype=np.float64)
        return np.full(len(frame), default, dtype=np.float64)

    return NumpyBars(
        times=frame.index.to_numpy(dtype="datetime64[ns]"),
        opens=col("open", "Open"),
        highs=col("high", "High"),
        lows=col("low", "Low"),
        closes=col("close", "Close"),
        volumes=col("volume", "Volume"),
    )


def build_numpy_ticks(df: pd.DataFrame) -> NumpyTicks:
    if df.empty:
        return empty_ticks()
    frame = df.copy()
    if "timestamp" in frame.columns:
        times = pd.to_datetime(frame["timestamp"], utc=True)
    elif "time" in frame.columns:
        times = pd.to_datetime(frame["time"], utc=True)
    else:
        times = pd.to_datetime(frame.index, utc=True)

    def col(*names: str, default: float = 0.0) -> np.ndarray:
        for name in names:
            if name in frame.columns:
                return frame[name].to_numpy(dtype=np.float64)
        return np.full(len(frame), default, dtype=np.float64)

    prices = col("price", "last", "close")
    sizes = col("size", "volume", default=1.0)
    sides = col("side", default=0.0).astype(np.int8)
    return NumpyTicks(
        timestamps=times.to_numpy(dtype="datetime64[ns]"),
        prices=prices,
        sizes=sizes,
        sides=sides,
        bid_prices=col("bid", "bid_price"),
        ask_prices=col("ask", "ask_price"),
    )


def load_bar_data(data_dir: str | Path, symbol: str) -> dict[str, NumpyBars]:
    root = Path(data_dir)
    symbol = symbol.upper()
    return {
        timeframe: build_numpy_bars(_read_first_existing(_bar_candidates(root, symbol, timeframe)))
        for timeframe in ("1m", "5m", "1h", "4h", "daily")
    }


def load_tick_data(data_dir: str | Path, symbol: str) -> NumpyTicks | None:
    root = Path(data_dir)
    symbol = symbol.upper()
    df = _read_first_existing(
        [
            root / f"{symbol}_ticks.parquet",
            root / f"{symbol}_ticks.csv",
            root / symbol.lower() / "ticks.parquet",
            root / symbol.lower() / "ticks.csv",
            root / symbol / "ticks.parquet",
            root / symbol / "ticks.csv",
        ]
    )
    return None if df.empty else build_numpy_ticks(df)


def _bar_candidates(root: Path, symbol: str, timeframe: str) -> list[Path]:
    suffixes = {
        "daily": ("1d", "daily", "1D"),
        "4h": ("4h", "4H"),
        "1h": ("1h", "1H"),
        "5m": ("5m", "5min"),
        "1m": ("1m", "1min"),
    }[timeframe]
    paths: list[Path] = []
    for suffix in suffixes:
        paths.extend(
            [
                root / f"{symbol}_{suffix}.parquet",
                root / f"{symbol}_{suffix}.csv",
                root / symbol.lower() / f"{suffix}.parquet",
                root / symbol.lower() / f"{suffix}.csv",
                root / symbol / f"{suffix}.parquet",
                root / symbol / f"{suffix}.csv",
            ]
        )
    return paths


def _read_first_existing(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    return pd.DataFrame()
