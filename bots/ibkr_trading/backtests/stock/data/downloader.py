"""Stock universe downloader — wrapper over momentum's IBKR downloader.

Downloads daily bars for all S&P 500 constituents + reference symbols,
plus intraday bars (30m for ALCB, 5m for IARIC) for tradable subsets.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd

from backtests.momentum.data.downloader import _CLIENT_ID, _PACING_DELAY
from backtests.shared.data.ibkr.bars import download_historical_bars as _shared_download_historical_bars
from backtests.shared.data.ibkr.models import BarDownloadRequest
from backtests.shared.data.ibkr.pacing import RequestPacer
from backtests.stock.data.cache import bar_path, save_bars
from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS

logger = logging.getLogger(__name__)
_PACER = RequestPacer(min_interval_seconds=_PACING_DELAY)

# Reference symbols for market/sector regime computation
REFERENCE_SYMBOLS = [
    "SPY", "VIX", "HYG",
    "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLB", "XLI", "XLU", "XLRE", "XLC",
]

# Universe expansion: force 30m download for underrepresented Financials + Industrials
ALCB_FORCED_INTRADAY = [
    # Financials
    "MS", "WFC", "C", "SCHW", "AXP",
    # Industrials
    "BA", "RTX", "HON", "DE", "UPS", "UNP", "GE", "LMT", "ETN", "FDX",
]

# Sector ETF mapping (for sector-level regime fields)
SECTOR_ETFS = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


async def download_historical(
    ib,
    symbol: str,
    timeframe: str,
    duration: str,
    exchange: str,
    trading_class: str = "",
    rth_only: bool = False,
    output_dir: Path = Path("backtests/stock/data/raw"),
    sec_type: str = "STK",
    primary_exchange: str = "",
) -> pd.DataFrame:
    """Compatibility wrapper over the shared IBKR downloader."""
    return await _shared_download_historical_bars(
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
        ),
        pacer=_PACER,
    )


def get_universe_symbols() -> list[tuple[str, str, str]]:
    """Return full universe: (symbol, sector, exchange) from SP500_CONSTITUENTS."""
    return list(SP500_CONSTITUENTS)


def get_all_download_symbols() -> list[str]:
    """Return flat list of all symbols to download (SP500 + reference)."""
    sp500 = [s[0] for s in SP500_CONSTITUENTS]
    # Add reference symbols that aren't already in SP500
    existing = set(sp500)
    all_syms = list(sp500)
    for ref in REFERENCE_SYMBOLS:
        if ref not in existing:
            all_syms.append(ref)
            existing.add(ref)
    return all_syms


async def download_stock_universe(
    timeframes: list[str] | None = None,
    symbols: list[str] | None = None,
    duration: str = "5 Y",
    output_dir: Path = Path("backtests/stock/data/raw"),
    host: str = "127.0.0.1",
    port: int = 7496,
    skip_existing: bool = True,
) -> dict[str, list[str]]:
    """Download historical bar data for the stock universe.

    Args:
        timeframes: List of timeframes to download (default: ["1d"]).
        symbols: Override symbol list (default: full SP500 + reference).
        duration: IBKR duration string (default "5 Y" for daily).
        output_dir: Directory for parquet files.
        host: IBKR gateway host.
        port: IBKR gateway port.
        skip_existing: Skip symbols with existing parquet files.

    Returns:
        Dict mapping timeframe to list of successfully downloaded symbols.
    """
    from ib_async import IB

    if timeframes is None:
        timeframes = ["1d"]
    if symbols is None:
        symbols = get_all_download_symbols()

    ib = IB()
    await ib.connectAsync(host, port, clientId=_CLIENT_ID, timeout=30)

    result: dict[str, list[str]] = {tf: [] for tf in timeframes}
    total = len(symbols) * len(timeframes)
    done = 0
    errors: list[str] = []

    try:
        for tf in timeframes:
            rth_only = tf == "1d"
            # Pass the full requested duration through — the momentum
            # downloader's backward-walking logic handles IBKR per-request
            # limits by chunking automatically for intraday timeframes.
            tf_duration = duration

            for sym in symbols:
                done += 1
                path = bar_path(output_dir, sym, tf)

                if skip_existing and path.exists():
                    result[tf].append(sym)
                    if done % 50 == 0:
                        logger.info("[%d/%d] Skipping %s %s (cached)", done, total, sym, tf)
                    continue

                try:
                    logger.info("[%d/%d] Downloading %s %s ...", done, total, sym, tf)
                    df = await download_historical(
                        ib, sym, tf, tf_duration,
                        exchange="SMART",
                        rth_only=rth_only,
                        output_dir=output_dir,
                        sec_type="STK",
                    )
                    save_bars(df, path)
                    result[tf].append(sym)
                    logger.info(
                        "[%d/%d] Saved %s %s: %d bars (%s -> %s)",
                        done, total, sym, tf, len(df),
                        df.index[0].strftime("%Y-%m-%d"),
                        df.index[-1].strftime("%Y-%m-%d"),
                    )
                except Exception as e:
                    errors.append(f"{sym} {tf}: {e}")
                    logger.warning("[%d/%d] Failed %s %s: %s", done, total, sym, tf, e)

                await asyncio.sleep(_PACING_DELAY)

    finally:
        ib.disconnect()

    if errors:
        logger.warning("Download completed with %d errors:", len(errors))
        for err in errors[:20]:
            logger.warning("  %s", err)
        if len(errors) > 20:
            logger.warning("  ... and %d more", len(errors) - 20)

    for tf in timeframes:
        logger.info("Downloaded %d/%d symbols for %s", len(result[tf]), len(symbols), tf)

    return result


async def download_intraday_for_tradable(
    tradable_alcb: list[str],
    tradable_iaric: list[str],
    output_dir: Path = Path("backtests/stock/data/raw"),
    host: str = "127.0.0.1",
    port: int = 7496,
) -> dict[str, list[str]]:
    """Pass 2: Download intraday bars for symbols identified as tradable.

    Args:
        tradable_alcb: Symbols that appeared in ALCB CandidateArtifact.
        tradable_iaric: Symbols that appeared in IARIC WatchlistArtifact.
        output_dir: Directory for parquet files.

    Returns:
        Dict mapping timeframe to list of successfully downloaded symbols.
    """
    result: dict[str, list[str]] = {"30m": [], "5m": []}

    # Merge forced intraday symbols into ALCB tradable list
    alcb_combined = list(dict.fromkeys(tradable_alcb + ALCB_FORCED_INTRADAY))
    if alcb_combined:
        logger.info("Downloading 30m bars for %d ALCB-tradable symbols (%d forced)...",
                     len(alcb_combined), len(ALCB_FORCED_INTRADAY))
        r = await download_stock_universe(
            timeframes=["30m"],
            symbols=alcb_combined,
            duration="1 Y",
            output_dir=output_dir,
            host=host,
            port=port,
        )
        result["30m"] = r.get("30m", [])

    if tradable_iaric:
        logger.info("Downloading 5m bars for %d IARIC-tradable symbols...", len(tradable_iaric))
        r = await download_stock_universe(
            timeframes=["5m"],
            symbols=tradable_iaric,
            duration="1 Y",
            output_dir=output_dir,
            host=host,
            port=port,
        )
        result["5m"] = r.get("5m", [])

    return result
