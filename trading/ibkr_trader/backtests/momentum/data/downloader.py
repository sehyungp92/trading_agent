"""Momentum-family IBKR downloader compatibility facade."""

from __future__ import annotations

import asyncio
import logging
import shutil
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

from .cache import bar_path, load_bars, save_bars

logger = logging.getLogger(__name__)

_MAX_BARS_PER_REQUEST = 2000
_PACING_DELAY = 12.0
_PACING_SLEEP = 65
_MAX_RETRIES = 4
_CLIENT_ID = 100
_PACER = RequestPacer(min_interval_seconds=_PACING_DELAY)
MOMENTUM_DERIVED_TIMEFRAMES = ("15m", "30m", "1h", "4h", "1d")


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


async def _reconnect(
    ib,
    host: str = "127.0.0.1",
    port: int = 7496,
    client_id: int = _CLIENT_ID,
    timeout: int = 30,
) -> None:
    try:
        ib.disconnect()
    except Exception:
        pass
    await asyncio.sleep(5)
    await ib.connectAsync(host, port, clientId=client_id, timeout=timeout)


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

    Futures calls here are legacy compatibility diagnostics. Approval refreshes
    use the central momentum sync physical-contract/Panama path instead.
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


def derive_aligned_momentum_timeframes(
    symbol: str = "NQ",
    output_dir: Path = Path("backtest/data/raw"),
    targets: tuple[str, ...] = MOMENTUM_DERIVED_TIMEFRAMES,
    backup_existing: bool = False,
) -> dict[str, Path]:
    """Derive momentum compatibility files from the canonical 5m series.

    Momentum strategies treat 5m as the canonical NQ/MNQ base.  Derived flat
    files keep Vdubus and utility loaders from mixing independently stitched
    higher-timeframe IBKR downloads with that base series.
    """
    from backtests.momentum.data.preprocessing import (
        normalize_timezone,
        resample_5m_to_15m,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )

    symbol = symbol.upper()
    base_path = bar_path(output_dir, symbol, "5m")
    if not base_path.exists():
        raise FileNotFoundError(f"Missing canonical 5m data for {symbol}: {base_path}")

    base = normalize_timezone(load_bars(base_path))
    resamplers = {
        "15m": resample_5m_to_15m,
        "30m": resample_5m_to_30m,
        "1h": resample_5m_to_1h,
        "4h": resample_5m_to_4h,
        "1d": resample_5m_to_daily,
    }
    result: dict[str, Path] = {}
    for target in targets:
        target_key = target.lower()
        if target_key == "5m":
            result[target_key] = base_path
            continue
        resampler = resamplers.get(target_key)
        if resampler is None:
            raise ValueError(f"Unsupported momentum derived timeframe: {target}")
        derived = resampler(base)
        path = bar_path(output_dir, symbol, target_key)
        if backup_existing and path.exists():
            backup_path = path.with_name(f"{path.stem}_direct{path.suffix}")
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
                logger.info("Backed up existing %s %s -> %s", symbol, target_key, backup_path)
        save_bars(derived, path)
        result[target_key] = path
        logger.info("Derived aligned %s %s from 5m -> %s (%d bars)", symbol, target_key, path, len(derived))
    return result


def check_aligned_momentum_timeframes(
    symbol: str = "NQ",
    output_dir: Path = Path("backtest/data/raw"),
    targets: tuple[str, ...] = MOMENTUM_DERIVED_TIMEFRAMES,
):
    """Check momentum flat files against the strategy's 5m-derived bars."""
    from backtests.momentum.data.preprocessing import (
        normalize_timezone,
        resample_5m_to_15m,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )
    from backtests.shared.data.ibkr.alignment import compare_derived_frame_alignment

    symbol = symbol.upper()
    base_path = bar_path(output_dir, symbol, "5m")
    if not base_path.exists():
        raise FileNotFoundError(f"Missing canonical 5m data for {symbol}: {base_path}")
    base = normalize_timezone(load_bars(base_path))
    resamplers = {
        "15m": resample_5m_to_15m,
        "30m": resample_5m_to_30m,
        "1h": resample_5m_to_1h,
        "4h": resample_5m_to_4h,
        "1d": resample_5m_to_daily,
    }

    results = []
    for target in targets:
        target_key = target.lower()
        resampler = resamplers.get(target_key)
        if resampler is None:
            raise ValueError(f"Unsupported momentum alignment timeframe: {target}")
        target_path = bar_path(output_dir, symbol, target_key)
        target_frame = load_bars(target_path) if target_path.exists() else pd.DataFrame()
        results.append(
            compare_derived_frame_alignment(
                symbol=symbol,
                derived=resampler(base),
                target=target_frame,
                base_timeframe="5m",
                target_timeframe=target_key,
                base_rows=len(base),
            )
        )
    return results


async def download_all_symbols(
    symbols: list[str],
    configs: dict,
    duration: str = "5 Y",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, dict[str, Path]]:
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=_CLIENT_ID, timeout=30)
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


async def download_apex_data(
    symbol: str = "NQ",
    duration: str = "5 Y",
    exchange: str = "CME",
    trading_class: str = "NQ",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, Path]:
    result = await _download_futures_timeframes(
        symbol=symbol,
        timeframes=[("5m", False)],
        duration=duration,
        exchange=exchange,
        trading_class=trading_class,
        output_dir=output_dir,
        client_id=_CLIENT_ID,
    )
    result.update(derive_aligned_momentum_timeframes(symbol, output_dir))
    return result


async def download_nqdtc_data(
    symbol: str = "NQ",
    duration: str = "5 Y",
    exchange: str = "CME",
    trading_class: str = "NQ",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, Path]:
    result = await _download_futures_timeframes(
        symbol=symbol,
        timeframes=[("5m", False)],
        duration=duration,
        exchange=exchange,
        trading_class=trading_class,
        output_dir=output_dir,
        client_id=_CLIENT_ID,
    )
    result.update(derive_aligned_momentum_timeframes(symbol, output_dir))
    return result


async def download_vdubus_data(
    symbol: str = "NQ",
    duration: str = "5 Y",
    exchange: str = "CME",
    output_dir: Path = Path("backtest/data/raw"),
) -> dict[str, Path]:
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=_CLIENT_ID, timeout=30)
    result: dict[str, Path] = {}
    try:
        nq_5m = await download_historical(
            ib,
            symbol,
            "5m",
            duration,
            exchange=exchange,
            trading_class=symbol,
            output_dir=output_dir,
            sec_type="FUT",
        )
        path_5m = bar_path(output_dir, symbol, "5m")
        save_bars(nq_5m, path_5m)
        result["5m"] = path_5m
        result.update(derive_aligned_momentum_timeframes(symbol, output_dir, targets=("15m",)))

        es_path = bar_path(output_dir, "ES", "1d")
        if es_path.exists():
            result["ES_1d"] = es_path
        else:
            es_daily = await download_historical(
                ib,
                "ES",
                "1d",
                duration,
                exchange=exchange,
                trading_class="ES",
                rth_only=True,
                output_dir=output_dir,
                sec_type="FUT",
            )
            save_bars(es_daily, es_path)
            result["ES_1d"] = es_path
    finally:
        ib.disconnect()
    return result


async def _download_futures_timeframes(
    *,
    symbol: str,
    timeframes: list[tuple[str, bool]],
    duration: str,
    exchange: str,
    trading_class: str,
    output_dir: Path,
    client_id: int,
) -> dict[str, Path]:
    from ib_async import IB

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=client_id, timeout=30)
    result: dict[str, Path] = {}
    try:
        for timeframe, rth_only in timeframes:
            path = bar_path(output_dir, symbol, timeframe)
            df = await download_historical(
                ib,
                symbol,
                timeframe,
                duration,
                exchange=exchange,
                trading_class=trading_class,
                rth_only=rth_only,
                output_dir=output_dir,
                sec_type="FUT",
            )
            save_bars(df, path)
            result[timeframe] = path
            logger.info("Saved %s %s -> %s (%d bars)", symbol, timeframe, path, len(df))
    finally:
        ib.disconnect()
    return result
