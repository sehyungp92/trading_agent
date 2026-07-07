"""Storage, merge, and gap helpers for IBKR historical bars."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .models import Gap


OHLC_COLUMNS = ("open", "high", "low", "close")


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "time" in frame.columns:
            frame.index = pd.to_datetime(frame["time"], utc=True)
        elif "timestamp" in frame.columns:
            frame.index = pd.to_datetime(frame["timestamp"], utc=True)
        else:
            frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    return frame.sort_index()


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return ensure_utc_index(pd.read_parquet(path, engine="pyarrow"))


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_parquet(tmp, engine="pyarrow", index=True)
    tmp.replace(path)


def merge_frames(*frames: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    populated = [ensure_utc_index(frame) for frame in frames if frame is not None and not frame.empty]
    if not populated:
        return pd.DataFrame()
    merged = pd.concat(populated)
    merged = merged[~merged.index.duplicated(keep=keep)]
    return merged.sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = ensure_utc_index(df)
    if frame.empty:
        return frame
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    available = {column: func for column, func in agg.items() if column in frame.columns}
    resampled = frame.resample(rule, label="right", closed="right").agg(available)
    return resampled.dropna(
        subset=[column for column in OHLC_COLUMNS if column in resampled.columns],
        how="any",
    )


def gap_threshold(timeframe: str) -> pd.Timedelta:
    if timeframe == "1m":
        return pd.Timedelta(hours=72)
    if timeframe in {"5m", "15m", "30m", "1h", "4h"}:
        return pd.Timedelta(days=5)
    return pd.Timedelta(days=10)


def detect_large_gaps(df: pd.DataFrame, timeframe: str) -> list[Gap]:
    frame = ensure_utc_index(df)
    if len(frame) < 2:
        return []
    threshold = gap_threshold(timeframe)
    diffs = frame.index.to_series().diff().dropna()
    gaps: list[Gap] = []
    for ts, _delta in diffs[diffs > threshold].items():
        previous_pos = frame.index.get_loc(ts) - 1
        previous_ts = frame.index[previous_pos]
        gaps.append(
            Gap(
                start=previous_ts.to_pydatetime(),
                end=ts.to_pydatetime(),
                expected_frequency=timeframe,
            )
        )
    return gaps


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)

