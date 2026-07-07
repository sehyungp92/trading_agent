"""Hyperliquid data downloader with incremental fetching and rate limiting."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone

import pandas as pd
import structlog

from crypto_trader.data.store import ParquetStore

log = structlog.get_logger()


class HyperliquidDownloader:
    """Downloads candle and funding data from Hyperliquid REST API.

    Uses backward pagination for candles (newest → oldest) with 500 candles
    per request. Supports incremental mode via ParquetStore timestamps.
    """

    CANDLES_PER_REQUEST = 500

    def __init__(
        self,
        store: ParquetStore,
        rate_limit: float = 0.2,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
    ) -> None:
        self.store = store
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._info = None

    @property
    def info(self):
        """Lazy-init Hyperliquid Info client."""
        if self._info is None:
            from hyperliquid.info import Info
            self._info = Info(skip_ws=True)
        return self._info

    def _rate_limited_call(self, fn, *args, **kwargs):
        """Call fn with rate limiting and exponential backoff on 429s."""
        for attempt in range(self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                time.sleep(self.rate_limit)
                return result
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    wait = min(
                        self.backoff_base * (2 ** attempt) + random.uniform(0, 1),
                        self.backoff_max,
                    )
                    log.warning("rate_limited", attempt=attempt, wait=wait)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded")

    # -----------------------------------------------------------------------
    # Candles
    # -----------------------------------------------------------------------

    @staticmethod
    def _candles_to_df(raw: list[dict]) -> pd.DataFrame:
        """Convert SDK candle response to DataFrame with standard schema."""
        if not raw:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        records = []
        for c in raw:
            records.append({
                "ts": int(c["t"]),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        return pd.DataFrame(records)

    def download_candles(
        self,
        coin: str,
        interval: str,
        start_ts: int,
        end_ts: int,
    ) -> pd.DataFrame:
        """Download candles via backward pagination from end_ts to start_ts.

        Returns a DataFrame with all candles in the range, sorted by ts.
        """
        all_dfs: list[pd.DataFrame] = []
        cursor = end_ts
        total_fetched = 0

        while cursor > start_ts:
            raw = self._rate_limited_call(
                self.info.candles_snapshot,
                coin,
                interval,
                start_ts,
                cursor,
            )

            if not raw:
                break

            df = self._candles_to_df(raw)
            all_dfs.append(df)
            total_fetched += len(df)

            # Move cursor backward: before the earliest candle in this batch
            earliest_ts = int(df["ts"].min())
            if earliest_ts >= cursor:
                # No progress — avoid infinite loop
                break
            cursor = earliest_ts - 1

            log.debug(
                "candles.batch",
                coin=coin,
                interval=interval,
                batch_size=len(df),
                total=total_fetched,
            )

        if not all_dfs:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        # Filter to requested range
        result = result[(result["ts"] >= start_ts) & (result["ts"] <= end_ts)]
        log.info("candles.downloaded", coin=coin, interval=interval, rows=len(result))
        return result

    # -----------------------------------------------------------------------
    # Funding
    # -----------------------------------------------------------------------

    @staticmethod
    def _funding_to_df(raw: list[dict]) -> pd.DataFrame:
        """Convert SDK funding response to DataFrame."""
        if not raw:
            return pd.DataFrame(columns=["ts", "rate"])
        records = []
        for f in raw:
            records.append({
                "ts": int(f["time"]),
                "rate": float(f["fundingRate"]),
            })
        return pd.DataFrame(records)

    def download_funding(
        self,
        coin: str,
        start_ts: int,
        end_ts: int,
    ) -> pd.DataFrame:
        """Download funding rate history for a coin."""
        all_dfs: list[pd.DataFrame] = []
        cursor = start_ts
        total_fetched = 0

        while cursor < end_ts:
            raw = self._rate_limited_call(
                self.info.funding_history,
                coin,
                cursor,
                end_ts,
            )

            if not raw:
                break

            df = self._funding_to_df(raw)
            all_dfs.append(df)
            total_fetched += len(df)

            # Move cursor forward past the latest in this batch
            latest_ts = int(df["ts"].max())
            if latest_ts <= cursor:
                break
            cursor = latest_ts + 1

            log.debug("funding.batch", coin=coin, batch_size=len(df), total=total_fetched)

        if not all_dfs:
            return pd.DataFrame(columns=["ts", "rate"])

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        log.info("funding.downloaded", coin=coin, rows=len(result))
        return result

    # -----------------------------------------------------------------------
    # High-level
    # -----------------------------------------------------------------------

    def download_and_store(
        self,
        coin: str,
        interval: str,
        days: int = 90,
        incremental: bool = True,
    ) -> None:
        """Download candles and store to Parquet. Supports incremental mode."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - (days * 86_400_000)

        if incremental:
            last_ts = self.store.get_last_timestamp(coin, interval)
            if last_ts is not None:
                # Start from last known timestamp + 1ms
                start_ms = max(start_ms, last_ts + 1)
                log.info("incremental.resume", coin=coin, interval=interval, from_ts=start_ms)

        if start_ms >= now_ms:
            log.info("candles.up_to_date", coin=coin, interval=interval)
            return

        df = self.download_candles(coin, interval, start_ms, now_ms)
        if not df.empty:
            self.store.save_candles(coin, interval, df)

    def download_and_store_funding(
        self,
        coin: str,
        days: int = 90,
    ) -> None:
        """Download funding rates and store to Parquet."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - (days * 86_400_000)

        existing = self.store.load_funding(coin)
        if existing is not None and not existing.empty:
            last_ts = int(existing["ts"].iloc[-1])
            start_ms = max(start_ms, last_ts + 1)

        if start_ms >= now_ms:
            log.info("funding.up_to_date", coin=coin)
            return

        df = self.download_funding(coin, start_ms, now_ms)
        if not df.empty:
            self.store.save_funding(coin, df)
