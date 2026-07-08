"""Parquet-based storage for candle and funding data."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()

# Expected schemas
CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
CANDLE_DTYPES = {
    "ts": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}
FUNDING_COLUMNS = ["ts", "rate"]
FUNDING_DTYPES = {"ts": "int64", "rate": "float64"}


class ParquetStore:
    """File-based storage for market data in Parquet format.

    Directory layout:
      {base_dir}/candles/{coin}/{interval}.parquet
      {base_dir}/funding/{coin}.parquet
    """

    def __init__(self, base_dir: Path | str = Path("data")) -> None:
        self.base_dir = Path(base_dir)

    def _candle_path(self, coin: str, interval: str) -> Path:
        return self.base_dir / "candles" / coin / f"{interval}.parquet"

    def _funding_path(self, coin: str) -> Path:
        return self.base_dir / "funding" / f"{coin}.parquet"

    def _atomic_write(self, df: pd.DataFrame, path: Path) -> None:
        """Write DataFrame to Parquet atomically via tmp + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(str(tmp_path), str(path))

    # -----------------------------------------------------------------------
    # Candles
    # -----------------------------------------------------------------------

    def save_candles(self, coin: str, interval: str, df: pd.DataFrame) -> None:
        """Save candle data, merging with existing data if present."""
        path = self._candle_path(coin, interval)
        existing = self.load_candles(coin, interval)

        if existing is not None:
            df = pd.concat([existing, df], ignore_index=True)

        df = df.astype(CANDLE_DTYPES)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        self._atomic_write(df, path)
        log.info("store.candles_saved", coin=coin, interval=interval, rows=len(df))

    def load_candles(self, coin: str, interval: str) -> pd.DataFrame | None:
        """Load candle data, or None if not stored."""
        path = self._candle_path(coin, interval)
        if not path.exists():
            return None
        df = pd.read_parquet(path, engine="pyarrow")
        return df.astype(CANDLE_DTYPES)

    def get_last_timestamp(self, coin: str, interval: str) -> int | None:
        """Get the last candle timestamp (ms epoch), or None."""
        df = self.load_candles(coin, interval)
        if df is None or df.empty:
            return None
        return int(df["ts"].iloc[-1])

    # -----------------------------------------------------------------------
    # Funding
    # -----------------------------------------------------------------------

    def save_funding(self, coin: str, df: pd.DataFrame) -> None:
        """Save funding rate data, merging with existing."""
        path = self._funding_path(coin)
        existing = self.load_funding(coin)

        if existing is not None:
            df = pd.concat([existing, df], ignore_index=True)

        df = df.astype(FUNDING_DTYPES)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        self._atomic_write(df, path)
        log.info("store.funding_saved", coin=coin, rows=len(df))

    def load_funding(self, coin: str) -> pd.DataFrame | None:
        """Load funding rate data, or None if not stored."""
        path = self._funding_path(coin)
        if not path.exists():
            return None
        df = pd.read_parquet(path, engine="pyarrow")
        return df.astype(FUNDING_DTYPES)
