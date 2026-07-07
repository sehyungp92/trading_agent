"""Generate IARIC/ALCB artifacts using a temporary IB connection."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator

from ib_async import IB

logger = logging.getLogger(__name__)

RESEARCH_CLIENT_ID = int(os.environ.get("STOCK_RESEARCH_CLIENT_ID", "35"))


@asynccontextmanager
async def _research_ib(
    host: str = "127.0.0.1",
    port: int = 4002,
) -> AsyncIterator[IB]:
    """Temporary IB connection for research data fetching."""
    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=RESEARCH_CLIENT_ID, timeout=60)
        logger.info("Research IB connected (client_id=%d)", RESEARCH_CLIENT_ID)
        yield ib
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Research IB disconnected")


async def generate_iaric(trade_date: date, ib: IB) -> bool:
    """Run IARIC research + selection. Returns True on success."""
    from strategies.stock.iaric.research_generator import generate_research_snapshot
    from strategies.stock.iaric.research import run_daily_selection

    try:
        await generate_research_snapshot(trade_date, ib)
        run_daily_selection(trade_date)
        logger.info("IARIC artifact generated for %s", trade_date)
        return True
    except Exception:
        logger.error("IARIC artifact generation failed", exc_info=True)
        return False


async def generate_alcb(trade_date: date, ib: IB) -> bool:
    """Run ALCB research + selection. Returns True on success."""
    from strategies.stock.alcb.research_generator import generate_research_snapshot
    from strategies.stock.alcb.research import run_daily_selection

    try:
        await generate_research_snapshot(trade_date, ib=ib)
        run_daily_selection(trade_date)
        logger.info("ALCB artifact generated for %s", trade_date)
        return True
    except Exception:
        logger.error("ALCB artifact generation failed", exc_info=True)
        return False


async def ensure_artifacts(
    trade_date: date,
    missing: list[str],
    host: str = "127.0.0.1",
    port: int = 4002,
) -> dict[str, bool]:
    """Generate artifacts for missing strategies. Returns {sid: success}."""
    results: dict[str, bool] = {}
    t0 = time.monotonic()
    async with _research_ib(host=host, port=port) as ib:
        if "IARIC_v1" in missing:
            logger.info("Starting IARIC artifact generation...")
            results["IARIC_v1"] = await generate_iaric(trade_date, ib)
            logger.info("IARIC done in %.0fs", time.monotonic() - t0)
        if "ALCB_v1" in missing:
            t1 = time.monotonic()
            logger.info("Starting ALCB artifact generation...")
            results["ALCB_v1"] = await generate_alcb(trade_date, ib)
            logger.info("ALCB done in %.0fs", time.monotonic() - t1)
    logger.info("All artifact generation completed in %.0fs", time.monotonic() - t0)
    return results


async def _main() -> None:
    """Standalone entrypoint: generate both IARIC and ALCB artifacts for today."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    today = date.today()
    et_today = datetime.now(ZoneInfo("America/New_York")).date()
    if today != et_today:
        logger.info("UTC date %s differs from ET date %s -- using ET", today, et_today)
    today = et_today

    host = os.environ.get("IB_HOST", "127.0.0.1")
    port = int(os.environ.get("IB_PORT", "4002"))

    logger.info(
        "Generating stock artifacts for %s (IB %s:%d, client_id=%d)",
        today, host, port, RESEARCH_CLIENT_ID,
    )

    results = await ensure_artifacts(
        today,
        missing=["IARIC_v1", "ALCB_v1"],
        host=host,
        port=port,
    )

    failed = [sid for sid, ok in results.items() if not ok]
    if failed:
        logger.error("Artifact generation FAILED for: %s", failed)
        raise SystemExit(1)

    logger.info("All artifacts generated successfully: %s", list(results.keys()))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
