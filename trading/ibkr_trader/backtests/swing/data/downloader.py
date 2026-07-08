"""Swing-family IBKR downloader compatibility facade."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtests.shared.data.ibkr.bars import (
    bars_to_frame,
    build_legacy_chunked_contfuture_contract,
    download_historical_bars,
    duration_to_timedelta,
    request_bars_with_retry,
    timeframe_to_ibkr,
)
from backtests.shared.data.ibkr.models import BarDownloadRequest
from backtests.shared.data.ibkr.pacing import RequestPacer

from .cache import bar_path, save_bars

logger = logging.getLogger(__name__)

_MAX_BARS_PER_REQUEST = 2000
_PACING_DELAY = 12.0
_PACING_SLEEP = 65
_MAX_RETRIES = 5
_PACER = RequestPacer(min_interval_seconds=_PACING_DELAY)


def _timeframe_to_ibkr(timeframe: str) -> str:
    return timeframe_to_ibkr(timeframe)


def _ibkr_bar_size_to_timeframe(bar_size: str) -> str:
    reverse = {
        "1 min": "1m",
        "5 mins": "5m",
        "15 mins": "15m",
        "30 mins": "30m",
        "1 hour": "1h",
        "4 hours": "4h",
        "1 day": "1d",
    }
    return reverse.get(bar_size, bar_size)


def _duration_to_days(duration: str) -> int:
    return max(1, duration_to_timedelta(duration).days)


def _chunk_step(duration: str) -> timedelta:
    return duration_to_timedelta(duration)


def _chunk_filename(symbol: str, timeframe: str, end_dt: datetime) -> str:
    return f"{symbol}_{timeframe}_{end_dt.strftime('%Y%m%d_%H%M%S')}.parquet"


def _bars_to_df(bars) -> pd.DataFrame:
    return bars_to_frame(list(bars or []))


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_stock(symbol: str, exchange: str, currency: str = "USD"):
    from ib_async import Stock

    return Stock(symbol=symbol, exchange=exchange, currency=currency)


def _build_legacy_contfuture(ib, symbol: str, exchange: str, trading_class: str):
    from ib_async import ContFuture

    return ContFuture(symbol=symbol, exchange=exchange, tradingClass=trading_class or symbol)


async def _resolve_chunked_contract(ib, symbol: str, exchange: str, trading_class: str):
    request = BarDownloadRequest(
        symbol=symbol,
        timeframe="1m",
        exchange=exchange,
        trading_class=trading_class,
        allow_contfuture_legacy=True,
    )
    return await build_legacy_chunked_contfuture_contract(ib, request)


async def _request_with_retry(
    ib,
    contract,
    end_dt: datetime | str,
    duration: str,
    bar_size: str,
    use_rth: bool,
) -> list:
    return await request_bars_with_retry(
        ib,
        contract,
        end_dt=end_dt,
        duration=duration,
        timeframe=_ibkr_bar_size_to_timeframe(bar_size),
        what_to_show="TRADES",
        use_rth=use_rth,
        pacer=_PACER,
    )


async def download_historical(
    ib,
    symbol: str,
    timeframe: str,
    duration: str,
    exchange: str,
    trading_class: str = "",
    rth_only: bool = False,
    output_dir: Path = Path("backtest/data/raw"),
    sec_type: str = "FUT",
    primary_exchange: str = "",
) -> pd.DataFrame:
    """Download historical bars through the shared IBKR downloader.

    Futures calls here are legacy compatibility diagnostics. Family approval
    refreshes must use explicit production-source paths instead.
    """
    return await download_historical_bars(
        ib,
        BarDownloadRequest(
            symbol=symbol,
            timeframe=timeframe,
            duration=duration,
            exchange=exchange,
            trading_class=trading_class or symbol,
            use_rth=rth_only,
            output_dir=output_dir,
            sec_type=sec_type,
            primary_exchange=primary_exchange,
            allow_contfuture_legacy=sec_type.upper() == "FUT",
        ),
        pacer=_PACER,
    )


async def download_all_symbols(
    symbols: list[str],
    configs: dict,
    duration: str = "5 Y",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, dict[str, Path]]:
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=99, timeout=20)
    result: dict[str, dict[str, Path]] = {}
    try:
        for sym in symbols:
            cfg = configs[sym]
            result[sym] = {}
            for timeframe, rth_only in [("1h", False), ("1d", True)]:
                df = await download_historical(
                    ib,
                    sym,
                    timeframe,
                    duration,
                    exchange=cfg.exchange,
                    trading_class=cfg.trading_class,
                    rth_only=rth_only,
                    output_dir=output_dir,
                    sec_type=cfg.sec_type,
                    primary_exchange=cfg.primary_exchange,
                )
                path = bar_path(output_dir, sym, timeframe)
                save_bars(df, path)
                result[sym][timeframe] = path
                logger.info("Saved %s %s -> %s (%d bars)", sym, timeframe, path, len(df))
    finally:
        ib.disconnect()
    return result
