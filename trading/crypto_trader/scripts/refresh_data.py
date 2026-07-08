"""Incremental data refresh for Hyperliquid perpetual futures.

Downloads the latest candles (all timeframes) and funding rates for BTC, ETH, SOL.
Resumes from the last stored timestamp — no gaps, no overlaps.

Schedule via Windows Task Scheduler (every 3 days) or cron:
  python scripts/refresh_data.py

Gap detection runs after each download and logs warnings if any intervals
are missing candles.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog

# Ensure project root is importable when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_trader.data.downloader import HyperliquidDownloader
from crypto_trader.data.store import ParquetStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COINS = ["BTC", "ETH", "SOL"]
INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Interval duration in milliseconds — used for gap detection
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# How far back to look (days) when no existing data exists.
# Only matters for first-ever download; incremental runs ignore this.
FALLBACK_DAYS = 1200

DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def check_gaps(store: ParquetStore, coin: str, interval: str) -> int:
    """Check for timestamp gaps in stored candles. Returns gap count."""
    df = store.load_candles(coin, interval)
    if df is None or len(df) < 2:
        return 0

    expected_step = INTERVAL_MS[interval]
    timestamps = df["ts"].values
    diffs = timestamps[1:] - timestamps[:-1]

    gaps = 0
    for i, diff in enumerate(diffs):
        if diff > expected_step * 1.5:  # Allow 50% tolerance for DST/maintenance
            gap_start = datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc)
            gap_end = datetime.fromtimestamp(timestamps[i + 1] / 1000, tz=timezone.utc)
            missing = int(diff / expected_step) - 1
            log.warning(
                "gap_detected",
                coin=coin,
                interval=interval,
                gap_start=str(gap_start),
                gap_end=str(gap_end),
                missing_candles=missing,
            )
            gaps += missing
    return gaps

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def refresh() -> None:
    """Run incremental download for all coins × intervals + funding."""
    store = ParquetStore(base_dir=DATA_DIR)
    downloader = HyperliquidDownloader(store=store)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    total_new = 0
    total_gaps = 0
    errors: list[str] = []

    log.info("refresh.start", coins=COINS, intervals=INTERVALS)

    for coin in COINS:
        for interval in INTERVALS:
            last_ts = store.get_last_timestamp(coin, interval)

            if last_ts is not None:
                start_ms = last_ts + 1
                staleness_hours = (now_ms - last_ts) / 3_600_000
                log.info(
                    "refresh.candles",
                    coin=coin,
                    interval=interval,
                    resuming_from=str(datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)),
                    staleness_hours=round(staleness_hours, 1),
                )
            else:
                start_ms = now_ms - (FALLBACK_DAYS * 86_400_000)
                log.info("refresh.candles.fresh", coin=coin, interval=interval)

            if start_ms >= now_ms:
                log.info("refresh.candles.up_to_date", coin=coin, interval=interval)
                continue

            try:
                before_ts = store.get_last_timestamp(coin, interval)
                df = downloader.download_candles(coin, interval, start_ms, now_ms)
                if not df.empty:
                    store.save_candles(coin, interval, df)
                    after_ts = store.get_last_timestamp(coin, interval)
                    new_rows = len(df)
                    total_new += new_rows
                    log.info(
                        "refresh.candles.done",
                        coin=coin,
                        interval=interval,
                        new_rows=new_rows,
                    )
                else:
                    log.info("refresh.candles.no_new", coin=coin, interval=interval)

                # Gap check
                gap_count = check_gaps(store, coin, interval)
                total_gaps += gap_count

            except Exception:
                log.exception("refresh.candles.error", coin=coin, interval=interval)
                errors.append(f"{coin}/{interval}")

        # Funding
        try:
            log.info("refresh.funding", coin=coin)
            downloader.download_and_store_funding(coin, days=FALLBACK_DAYS)
        except Exception:
            log.exception("refresh.funding.error", coin=coin)
            errors.append(f"{coin}/funding")

    # Summary
    log.info(
        "refresh.complete",
        new_candles=total_new,
        gaps_found=total_gaps,
        errors=errors if errors else "none",
    )

    if errors:
        log.error("refresh.had_errors", failed=errors)
        sys.exit(1)


if __name__ == "__main__":
    refresh()
