"""Incremental update of stock data (1d/5m/30m) using shared bars.py infrastructure.

Only downloads the gap between existing data end and now. Much faster than re-downloading
full 2Y history via download_top100.py --force.

Usage:
    python -m backtests.stock.data.update_intraday                  # all: 1d, 30m, 5m
    python -m backtests.stock.data.update_intraday --timeframe 1d   # daily only
    python -m backtests.stock.data.update_intraday --timeframe 30m
    python -m backtests.stock.data.update_intraday --timeframe 5m
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from backtests.shared.data.ibkr.bars import connect_ib, download_historical_bars
from backtests.shared.data.ibkr.models import BarDownloadRequest, ConnectionSettings
from backtests.shared.data.ibkr.pacing import RequestPacer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# The 98 backtested symbols from strategies/stock/live_universe.py
BACKTESTED_SYMBOLS = [
    "A", "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "AMAT", "AMD", "AMGN",
    "AMZN", "AVGO", "BAC", "BDX", "BIO", "BLK", "BSX", "CAT", "CDNS", "CDW",
    "CI", "COR", "CRM", "CRWD", "CSCO", "DHR", "DXCM", "ELV", "EPAM", "EW",
    "FSLR", "FTNT", "GEN", "GILD", "GOOG", "GOOGL", "GS", "HCA", "HD", "HPE",
    "HPQ", "HSIC", "IBM", "IDXX", "INTU", "IQV", "ISRG", "IT", "JNJ", "JPM",
    "KEYS", "KLAC", "LH", "LLY", "LRCX", "MA", "MCHP", "MCK", "MDT", "META",
    "MPWR", "MRK", "MSFT", "MSI", "MTD", "MU", "NFLX", "NOW", "NTAP", "NVDA",
    "NXPI", "ON", "ORCL", "PANW", "PFE", "PTC", "QCOM", "QRVO", "REGN", "RMD",
    "ROP", "SNPS", "SWKS", "SYK", "TDY", "TECH", "TER", "TMO", "TRMB", "TSLA",
    "TXN", "UNH", "V", "VRTX", "WMT", "ZBRA", "ZTS",
]

OUTPUT_DIR = Path("backtests/stock/data/raw")

# Reference symbols needed for regime/sector computation in stock backtests
REFERENCE_SYMBOLS = [
    "SPY", "HYG",
    "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLB", "XLI", "XLU", "XLRE", "XLC",
]


async def update_stock_data(
    timeframes: list[str],
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 113,
) -> None:
    """Incrementally update stock data for all backtested symbols."""
    settings = ConnectionSettings(host=host, port=port, client_id=client_id)
    pacer = RequestPacer()
    ib = await connect_ib(settings)

    try:
        for tf in timeframes:
            # For 1d, also include reference symbols (SPY, HYG, sector ETFs)
            if tf == "1d":
                seen = set(BACKTESTED_SYMBOLS)
                symbols = list(BACKTESTED_SYMBOLS)
                for ref in REFERENCE_SYMBOLS:
                    if ref not in seen:
                        symbols.append(ref)
                        seen.add(ref)
            else:
                symbols = list(BACKTESTED_SYMBOLS)

            # 1d uses RTH; intraday uses ETH
            use_rth = tf == "1d"

            logger.info("[stock %s] %d symbols to update (rth=%s)", tf, len(symbols), use_rth)

            for i, sym in enumerate(symbols, 1):
                output_path = OUTPUT_DIR / f"{sym}_{tf}.parquet"
                request = BarDownloadRequest(
                    symbol=sym,
                    timeframe=tf,
                    sec_type="STK",
                    exchange="SMART",
                    what_to_show="TRADES",
                    use_rth=use_rth,
                    duration="2 Y",
                    end=datetime.now(timezone.utc),
                    output_dir=OUTPUT_DIR,
                    family="stock",
                )
                logger.info("[stock %s] (%d/%d) %s", tf, i, len(symbols), sym)
                try:
                    result = await download_historical_bars(
                        ib,
                        request,
                        output_path=output_path,
                        pacer=pacer,
                        dry_run=False,
                        latest_only=True,
                    )
                    if result and result.rows:
                        logger.info("  -> %s %s: %d rows [%s .. %s]",
                                    result.symbol, result.timeframe,
                                    result.rows, result.start, result.end)
                    elif result:
                        logger.info("  -> %s %s: already up to date", sym, tf)
                except Exception as e:
                    logger.error("  -> %s %s FAILED: %s", sym, tf, e)
                    continue
    finally:
        ib.disconnect()


def main() -> None:
    timeframes = ["1d", "30m", "5m"]
    client_id = 114

    if "--timeframe" in sys.argv:
        idx = sys.argv.index("--timeframe")
        if idx + 1 < len(sys.argv):
            timeframes = [sys.argv[idx + 1]]

    if "--client-id" in sys.argv:
        idx = sys.argv.index("--client-id")
        if idx + 1 < len(sys.argv):
            client_id = int(sys.argv[idx + 1])

    asyncio.run(update_stock_data(timeframes, client_id=client_id))


if __name__ == "__main__":
    main()
