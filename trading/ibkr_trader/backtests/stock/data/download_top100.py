"""Download daily + intraday data for top 100 S&P 500 names.

Uses backward-walking stitching for intraday timeframes to get full 2Y history.
Processes each symbol with a fresh IBKR connection to avoid session timeouts.

Usage:
    python -m backtests.stock.data.download_top100
    python -m backtests.stock.data.download_top100 --force
    python -m backtests.stock.data.download_top100 --phase 5m   # only 5m
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from backtests.momentum.data.downloader import (
    _CLIENT_ID,
    _PACING_DELAY,
    bar_path,
    download_historical,
    save_bars,
)
from strategies.stock.alcb.universe_constituents import SP500_CONSTITUENTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ALL_TOP_100 = [sym for sym, _, _ in SP500_CONSTITUENTS[:100]]
OUTPUT_DIR = Path("backtests/stock/data/raw")

# Symbols that failed 30m download (IBKR ticker resolution issues).
# Deprioritised: processed last so working symbols finish first.
_DEPRIORITISED = {
    "ANSS", "SWKS", "PTC", "TRMB", "TER", "NTAP", "GEN", "JNPR",
    "EPAM", "QRVO", "LLY", "WAT", "BAX", "HOLX", "ALGN", "PODD",
    "MOH", "CRL", "RVTY", "VTRS", "INCY", "CTLT",
}
TOP_100 = [s for s in _ALL_TOP_100 if s not in _DEPRIORITISED] + \
          [s for s in _ALL_TOP_100 if s in _DEPRIORITISED]

# How many symbols to process per IBKR connection before reconnecting.
# Keeps sessions short to avoid IBKR's idle-disconnect.
_BATCH_SIZE = {
    "1d": 50,   # daily is fast — 1 request per symbol
    "30m": 10,  # 30m needs ~3 chunks per symbol
    "5m": 5,    # 5m needs ~100 chunks per symbol — ~15 min per batch of 5
}


async def _download_batch(
    symbols: list[str],
    timeframe: str,
    duration: str,
    rth_only: bool,
    skip_existing: bool,
    batch_label: str,
) -> tuple[list[str], list[str]]:
    """Download a batch of symbols with one IBKR connection.

    Returns (succeeded, failed) symbol lists.
    """
    from ib_async import IB

    ib = IB()
    succeeded: list[str] = []
    failed: list[str] = []

    try:
        await ib.connectAsync("127.0.0.1", 7496, clientId=_CLIENT_ID, timeout=30)

        for i, sym in enumerate(symbols, 1):
            path = bar_path(OUTPUT_DIR, sym, timeframe)
            if skip_existing and path.exists():
                succeeded.append(sym)
                continue

            try:
                logger.info("[%s %d/%d] Downloading %s %s ...",
                            batch_label, i, len(symbols), sym, timeframe)
                df = await download_historical(
                    ib, sym, timeframe, duration,
                    exchange="SMART",
                    rth_only=rth_only,
                    output_dir=OUTPUT_DIR,
                    sec_type="STK",
                )
                save_bars(df, path)
                succeeded.append(sym)
                span = (df.index[-1] - df.index[0]).days
                logger.info("[%s %d/%d] Saved %s %s: %d bars, %d days",
                            batch_label, i, len(symbols), sym, timeframe,
                            len(df), span)
            except Exception as e:
                failed.append(sym)
                logger.warning("[%s %d/%d] Failed %s %s: %s",
                               batch_label, i, len(symbols), sym, timeframe, e)

            await asyncio.sleep(_PACING_DELAY)
    except Exception as e:
        logger.error("Connection error in batch %s: %s", batch_label, e)
        # Mark remaining unprocessed symbols as failed
        processed = set(succeeded) | set(failed)
        for sym in symbols:
            if sym not in processed:
                failed.append(sym)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    return succeeded, failed


async def download_phase(
    symbols: list[str],
    timeframe: str,
    duration: str,
    skip_existing: bool,
) -> tuple[int, int]:
    """Download one timeframe for all symbols, batched for resilience."""
    rth_only = timeframe == "1d"
    batch_size = _BATCH_SIZE.get(timeframe, 5)

    # Filter out already-downloaded symbols BEFORE batching so we don't
    # waste IBKR connections just to skip files.
    skipped = 0
    if skip_existing:
        needed = []
        for sym in symbols:
            if bar_path(OUTPUT_DIR, sym, timeframe).exists():
                skipped += 1
            else:
                needed.append(sym)
        if skipped:
            logger.info("Skipping %d symbols with existing %s files, %d to download",
                        skipped, timeframe, len(needed))
    else:
        needed = list(symbols)

    total_ok = skipped
    total_fail = 0
    consecutive_empty = 0  # batches where every symbol failed (connection issue)

    for batch_start in range(0, len(needed), batch_size):
        batch = needed[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(needed) + batch_size - 1) // batch_size
        label = f"batch {batch_num}/{total_batches}"

        logger.info("=== %s %s: %s (%d symbols) ===",
                     timeframe, duration, label, len(batch))

        ok, fail = await _download_batch(
            batch, timeframe, duration, rth_only, False, label,
        )
        total_ok += len(ok)
        total_fail += len(fail)

        # Track consecutive full-batch failures (connection issues).
        if len(ok) == 0 and len(fail) > 0:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                remaining = len(needed) - batch_start - batch_size
                logger.error(
                    "3 consecutive batch failures — IBKR likely rate-limiting. "
                    "Aborting with %d symbols remaining. Re-run to resume.",
                    max(remaining, 0),
                )
                total_fail += max(remaining, 0)
                break
        else:
            consecutive_empty = 0

        # Pause between batches to let IBKR settle.
        # 5m is extra aggressive (~100 chunks/symbol) so needs longer cooldown.
        if batch_start + batch_size < len(needed):
            pause = 15 if timeframe == "5m" else 3
            await asyncio.sleep(pause)

    return total_ok, total_fail


async def main() -> None:
    force = "--force" in sys.argv

    # Allow running a single phase: --phase 1d / --phase 30m / --phase 5m
    phase_filter = None
    if "--phase" in sys.argv:
        idx = sys.argv.index("--phase")
        if idx + 1 < len(sys.argv):
            phase_filter = sys.argv[idx + 1]

    phases = [
        ("1d", "2 Y"),
        ("30m", "2 Y"),
        ("5m", "2 Y"),
    ]
    if phase_filter:
        phases = [(tf, dur) for tf, dur in phases if tf == phase_filter]

    logger.info("Downloading %d symbols, phases: %s, force=%s",
                len(TOP_100), [tf for tf, _ in phases], force)

    results: dict[str, tuple[int, int]] = {}
    for tf, dur in phases:
        logger.info("\n>>> PHASE: %s (%s) <<<", tf, dur)
        ok, fail = await download_phase(TOP_100, tf, dur, skip_existing=not force)
        results[tf] = (ok, fail)
        logger.info("Phase %s done: %d ok, %d failed", tf, ok, fail)

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    for tf, (ok, fail) in results.items():
        status = "OK" if fail == 0 else f"{fail} FAILED"
        print(f"  {tf:5s}: {ok}/{ok + fail} symbols  [{status}]")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
