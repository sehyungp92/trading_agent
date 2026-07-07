"""IBKR Historical Data Provider — bridges sync instrumentation with async IBKR.

Used by PostExitTracker and MissedOpportunityLogger for backfill operations.
Runs async IBKR requests via run_coroutine_threadsafe from executor threads.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("instrumentation.ibkr_provider")


class IBKRHistoricalProvider:
    """Provides historical price data via IBKR for backfill operations.

    Designed to be called from threads running in an executor, with the
    IBKR connection managed on the main asyncio event loop.

    Usage::

        loop = asyncio.get_running_loop()
        provider = IBKRHistoricalProvider(session.ib, contract_factory, loop)

        # From executor thread:
        price = provider.get_price_at("QQQ", some_timestamp)
        candles = provider.get_ohlcv("QQQ", "5m", since_ms, 300)
    """

    def __init__(
        self,
        ib,
        contract_factory,
        loop: asyncio.AbstractEventLoop,
        historical_requester: Callable[..., Awaitable[object]] | None = None,
    ):
        self._ib = ib
        self._factory = contract_factory
        self._loop = loop
        self._historical_requester = historical_requester
        self._contract_cache: dict[str, object] = {}

    def get_price_at(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get close price at a specific timestamp. Synchronous (thread-safe)."""
        future = asyncio.run_coroutine_threadsafe(
            self._fetch_price(symbol, timestamp), self._loop,
        )
        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.debug("get_price_at(%s, %s) failed: %s", symbol, timestamp, e)
            return None

    def get_ohlcv(
        self,
        pair: str,
        timeframe: str = "5m",
        since: int | None = None,
        limit: int = 300,
    ) -> list | None:
        """Get OHLCV candles. Synchronous (thread-safe).

        Returns list of [timestamp_ms, open, high, low, close, volume] lists,
        matching the interface expected by MissedOpportunityLogger._compute_outcomes.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._fetch_ohlcv(pair, timeframe, since, limit), self._loop,
        )
        try:
            return future.result(timeout=60)
        except Exception as e:
            logger.debug("get_ohlcv(%s) failed: %s", pair, e)
            return None

    async def _get_contract(self, symbol: str):
        """Get or cache a qualified contract for a symbol."""
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]
        try:
            contract, _ = await self._factory.resolve(symbol)
            self._contract_cache[symbol] = contract
            return contract
        except Exception:
            try:
                contract = self._factory.build_contract(symbol)
            except Exception:
                return None
            qualified = await self._ib.qualifyContractsAsync(contract)
            if qualified:
                self._contract_cache[symbol] = qualified[0]
                return qualified[0]
            return None

    async def _fetch_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Fetch close price at a specific timestamp via IBKR historical data."""
        contract = await self._get_contract(symbol)
        if contract is None:
            return None

        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        end_dt = timestamp.strftime("%Y%m%d %H:%M:%S") + " UTC"

        bars = await self._request_historical_data(
            contract,
            endDateTime=end_dt,
            durationStr="3600 S",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
        )
        if bars:
            return bars[-1].close
        return None

    async def _fetch_ohlcv(
        self, pair: str, timeframe: str, since: int | None, limit: int,
    ) -> list | None:
        """Fetch OHLCV candles for missed opportunity backfill."""
        contract = await self._get_contract(pair)
        if contract is None:
            return None

        # Map timeframe to IBKR bar size
        bar_size_map = {
            "5m": "5 mins",
            "15m": "15 mins",
            "1h": "1 hour",
            "1d": "1 day",
        }
        bar_size = bar_size_map.get(timeframe, "5 mins")

        # Compute end datetime and duration from since/limit
        if since is not None:
            start_time = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
            # Estimate end time based on limit and bar size
            bar_minutes = {"5m": 5, "15m": 15, "1h": 60, "1d": 1440}.get(timeframe, 5)
            duration_seconds = limit * bar_minutes * 60
            end_time = start_time + timedelta(seconds=duration_seconds)
            # Clamp to now
            now = datetime.now(timezone.utc)
            if end_time > now:
                end_time = now
            end_dt = end_time.strftime("%Y%m%d %H:%M:%S") + " UTC"
            duration_str = f"{duration_seconds} S"
        else:
            end_dt = ""
            duration_str = f"{limit * 5 * 60} S"

        bars = await self._request_historical_data(
            contract,
            endDateTime=end_dt,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
        )

        if not bars:
            return None

        # Convert to [timestamp_ms, open, high, low, close, volume] format
        return [
            [
                int(bar.date.timestamp() * 1000) if hasattr(bar.date, 'timestamp')
                else int(datetime.fromisoformat(str(bar.date)).timestamp() * 1000),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ]
            for bar in bars
        ]

    async def _request_historical_data(self, contract, **kwargs):
        if self._historical_requester is not None:
            return await self._historical_requester(
                contract,
                request_kind="backfill",
                **kwargs,
            )
        # Raw IB fallback for standalone instrumentation tools/tests.
        return await self._ib.reqHistoricalDataAsync(contract, timeout=45, **kwargs)
