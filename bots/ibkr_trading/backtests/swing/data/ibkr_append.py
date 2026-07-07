"""Append recent data to swing parquet files using IBKR gateway.

Requires IB Gateway or TWS running on localhost:7496.

Usage: python -m backtests.swing.data.ibkr_append [--port 7496]

Loads existing parquet files, determines the last timestamp, downloads
incremental data from IBKR, and appends. Uses the same download
infrastructure as the original downloader.py to ensure schema parity.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtests.swing.data.cache import bar_path, load_bars, save_bars
from backtests.swing.data.downloader import (
    _bars_to_df,
    _build_stock,
    _request_with_retry,
    _timeframe_to_ibkr,
    _PACING_DELAY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "raw"

# ETF symbols and their IBKR config
SYMBOL_CONFIGS = {
    "QQQ": {"exchange": "SMART", "primary_exchange": "NASDAQ"},
    "GLD": {"exchange": "SMART", "primary_exchange": "ARCA"},
}

# Timeframes: (key, ibkr_rth_only)
# hourly: all hours (rth_only=False) to match existing IBKR data
# daily: RTH only (rth_only=True) to match existing IBKR data
TIMEFRAMES = [
    ("1h", False),
    ("1d", True),
]


def _days_since(ts: pd.Timestamp) -> int:
    """Days between timestamp and now."""
    now = datetime.now(timezone.utc)
    last = ts.to_pydatetime()
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last).days


def _duration_for_gap(days: int, timeframe: str) -> str:
    """Build an IBKR duration string that covers the gap plus buffer."""
    days = max(days + 5, 10)  # buffer for weekends/holidays
    if timeframe == "1d":
        if days > 365:
            return f"{(days // 365) + 1} Y"
        return f"{days} D"
    else:
        # Hourly: use months for larger gaps
        if days > 60:
            months = (days // 30) + 1
            return f"{months} M"
        return f"{days} D"


async def append_symbol(
    ib,
    symbol: str,
    timeframe: str,
    rth_only: bool,
) -> int:
    """Download and append incremental data for one symbol+timeframe."""
    path = bar_path(DATA_DIR, symbol, timeframe)
    if not path.exists():
        logger.warning("SKIP %s_%s — file not found at %s", symbol, timeframe, path)
        return 0

    existing = load_bars(path)
    last_ts = existing.index[-1]
    gap_days = _days_since(last_ts)

    if gap_days <= 1:
        logger.info("%s_%s: already up to date (last=%s)", symbol, timeframe, last_ts)
        return 0

    logger.info(
        "%s_%s: %d rows, last=%s, gap=%d days",
        symbol, timeframe, len(existing), last_ts, gap_days,
    )

    # Build contract
    cfg = SYMBOL_CONFIGS[symbol]
    stock = _build_stock(symbol, cfg["exchange"])
    from ib_async import IB
    qualified = await ib.qualifyContractsAsync(stock)
    if not qualified:
        logger.error("Could not qualify Stock for %s", symbol)
        return 0
    stock = qualified[0]

    # Download
    duration = _duration_for_gap(gap_days, timeframe)
    bar_size = _timeframe_to_ibkr(timeframe)
    logger.info("Requesting %s %s duration=%s rth=%s", symbol, timeframe, duration, rth_only)

    bars = await _request_with_retry(ib, stock, "", duration, bar_size, rth_only)
    if not bars:
        logger.warning("No data returned for %s %s", symbol, timeframe)
        return 0

    new_df = _bars_to_df(bars)

    # Keep only rows strictly after existing last timestamp
    new_only = new_df[new_df.index > last_ts]
    if new_only.empty:
        logger.info("%s_%s: no new bars after %s", symbol, timeframe, last_ts)
        return 0

    # Concat, dedup, sort
    combined = pd.concat([existing, new_only])
    combined = combined[~combined.index.duplicated(keep="first")]  # keep IBKR original
    combined = combined.sort_index()

    # Schema validation
    assert list(combined.columns) == list(existing.columns), \
        f"Column mismatch: {list(combined.columns)} vs {list(existing.columns)}"
    assert combined["volume"].dtype == existing["volume"].dtype, \
        f"Volume dtype mismatch: {combined['volume'].dtype} vs {existing['volume'].dtype}"

    # Save
    save_bars(combined, path)
    rows_added = len(combined) - len(existing)
    logger.info(
        "%s_%s: +%d rows → %d total, new range: %s to %s",
        symbol, timeframe, rows_added, len(combined),
        new_only.index[0], new_only.index[-1],
    )
    return rows_added


async def main(port: int = 7496, client_id: int = 99):
    from ib_async import IB

    ib = IB()
    logger.info("Connecting to IB Gateway on localhost:%d ...", port)
    await ib.connectAsync("127.0.0.1", port, clientId=client_id, timeout=20)
    logger.info("Connected.")

    total = 0
    try:
        for symbol in SYMBOL_CONFIGS:
            for tf, rth in TIMEFRAMES:
                total += await append_symbol(ib, symbol, tf, rth)
                await asyncio.sleep(_PACING_DELAY)
    finally:
        ib.disconnect()

    logger.info("Done. Total rows added: %d", total)

    # Verification
    logger.info("--- Verification ---")
    for symbol in SYMBOL_CONFIGS:
        for tf, _ in TIMEFRAMES:
            path = bar_path(DATA_DIR, symbol, tf)
            if path.exists():
                df = load_bars(path)
                logger.info(
                    "  %s_%s: %d rows, %s → %s, vol dtype=%s",
                    symbol, tf, len(df), df.index[0], df.index[-1], df["volume"].dtype,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append IBKR data to swing parquet files")
    parser.add_argument("--port", type=int, default=7496, help="IB Gateway port")
    parser.add_argument("--client-id", type=int, default=99, help="IBKR client ID")
    args = parser.parse_args()
    asyncio.run(main(port=args.port, client_id=args.client_id))
