"""Hyperliquid parquet store helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_assistant_data.checksums import stable_row_hashes


def candle_open_from_ms(value: int | float) -> pd.Timestamp:
    return pd.to_datetime(int(value), unit="ms", utc=True)


def canonicalize_candles(
    frame: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    source_file: str = "",
) -> pd.DataFrame:
    source = frame.copy()
    ts = _column(source, "ts", "t", "time", "timestamp")
    open_ = _column(source, "open", "o")
    high = _column(source, "high", "h")
    low = _column(source, "low", "l")
    close = _column(source, "close", "c")
    volume = _column(source, "volume", "v")
    raw = pd.DataFrame(
        {
            "ts": source[ts],
            "open": source[open_],
            "high": source[high],
            "low": source[low],
            "close": source[close],
            "volume": source[volume],
        }
    )
    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(raw["ts"], unit="ms", utc=True),
            "timestamp_exchange": pd.to_datetime(raw["ts"], unit="ms", utc=True).astype(str),
            "symbol": symbol.upper(),
            "market": "crypto_perp",
            "source": "hyperliquid",
            "timeframe": interval,
            "kind": "trades",
            "open": raw["open"].astype("float64"),
            "high": raw["high"].astype("float64"),
            "low": raw["low"].astype("float64"),
            "close": raw["close"].astype("float64"),
            "volume": raw["volume"].astype("float64"),
            "source_file": source_file,
            "source_ts_ms": raw["ts"].astype("int64"),
        }
    )
    out["source_row_hash"] = stable_row_hashes(raw)
    return out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")


def canonicalize_funding(
    frame: pd.DataFrame,
    *,
    symbol: str,
    source_file: str = "",
) -> pd.DataFrame:
    source = frame.copy()
    ts = _column(source, "ts", "time", "timestamp")
    rate = _column(source, "rate", "fundingRate", "funding_rate")
    raw = pd.DataFrame({"ts": source[ts], "rate": source[rate]})
    timestamp_utc = pd.to_datetime(raw["ts"], unit="ms", utc=True).dt.floor("h")
    out = pd.DataFrame(
        {
            "timestamp_utc": timestamp_utc,
            "timestamp_exchange": timestamp_utc.astype(str),
            "symbol": symbol.upper(),
            "market": "crypto_perp",
            "source": "hyperliquid",
            "rate": raw["rate"].astype("float64"),
            "source_file": source_file,
            "source_ts_ms": raw["ts"].astype("int64"),
        }
    )
    out["source_row_hash"] = stable_row_hashes(raw)
    return out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")


def _column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Hyperliquid frame missing one of columns: {list(names)}")


def write_monthly_partitions(frame: pd.DataFrame, root: Path, *parts: str) -> list[Path]:
    paths: list[Path] = []
    if frame.empty:
        return paths
    by_month = frame.assign(
        year=frame["timestamp_utc"].dt.year.astype(str),
        month=frame["timestamp_utc"].dt.month.map(lambda value: f"{int(value):02d}"),
    )
    for (year, month), group in by_month.groupby(["year", "month"], sort=True):
        path = Path(root).joinpath(*parts, f"year={year}", f"month={month}", "part.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year", "month"]).to_parquet(path, engine="pyarrow", index=False)
        paths.append(path)
    return paths
