"""Download missing 5m and 30m bars for symbols with no intraday data.

Requires IBKR Gateway/TWS running on 127.0.0.1:7496.

Usage:
    python -m backtests.stock.data.redownload_5m
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from backtests.stock.data.downloader import download_stock_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

SYMBOLS = ["CAT", "HD", "GS", "WMT", "BLK"]
TIMEFRAMES = ["5m", "30m"]

OUTPUT_DIR = Path("backtests/stock/data/raw")


async def main() -> None:
    logger.info(
        "Downloading %s bars for %d symbols...",
        TIMEFRAMES, len(SYMBOLS),
    )

    result = await download_stock_universe(
        timeframes=TIMEFRAMES,
        symbols=SYMBOLS,
        duration="2 Y",
        output_dir=OUTPUT_DIR,
        skip_existing=False,
    )

    for tf in TIMEFRAMES:
        downloaded = result.get(tf, [])
        failed = set(SYMBOLS) - set(downloaded)
        logger.info("%s: downloaded %d/%d", tf, len(downloaded), len(SYMBOLS))
        if failed:
            logger.warning("%s failed: %s", tf, sorted(failed))

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
