"""Capture concurrent positions across sibling strategies at trade entry time.

Used by all three family trade loggers to populate correlated_pairs_detail,
giving the Trading Assistant visibility into cross-strategy position overlap.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

logger = logging.getLogger("instrumentation.correlation_snapshot")


def run_async_safely(coro, *, timeout: float = 2.0):
    """Run an async coroutine from synchronous code, even inside a running loop.

    If an event loop is already running (e.g. inside an OMS async context),
    delegates to a background thread via ThreadPoolExecutor.  Otherwise
    uses ``asyncio.run()`` directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run directly
        return asyncio.run(coro)
    # Already inside an async context — run in a separate thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout)


async def capture_concurrent_positions(
    pg_store: Any,
    family_id: str,
    current_strategy_id: str,
    current_symbol: str,
    sibling_strategy_ids: list[str],
) -> list[dict]:
    """Query open positions from sibling strategies within the same family.

    Returns a list of dicts, one per sibling open position::

        {
            "sibling_strategy_id": "AKC_HELIX",
            "sibling_symbol": "QQQ",
            "sibling_direction": "LONG",
            "same_symbol": True,
        }

    This is a read-only query on the positions table.  Callers must wrap
    in try/except so instrumentation failures never block trades.
    """
    if not pg_store or not sibling_strategy_ids:
        return []

    # Exclude current strategy from sibling list
    siblings = [s for s in sibling_strategy_ids if s != current_strategy_id]
    if not siblings:
        return []

    rows = await pg_store.fetch(
        """
        SELECT strategy_id, instrument_symbol,
               CASE WHEN net_qty > 0 THEN 'LONG'
                    WHEN net_qty < 0 THEN 'SHORT'
                    ELSE 'FLAT' END AS direction
        FROM positions
        WHERE strategy_id = ANY($1)
          AND net_qty != 0
        """,
        siblings,
    )

    return [
        {
            "sibling_strategy_id": row["strategy_id"],
            "sibling_symbol": row["instrument_symbol"],
            "sibling_direction": row["direction"],
            "same_symbol": row["instrument_symbol"] == current_symbol,
        }
        for row in rows
    ]


def capture_concurrent_positions_from_coordinator(
    coordinator: Any,
    current_strategy_id: str,
    current_symbol: str,
) -> list[dict]:
    """Extract concurrent positions from an in-process StrategyCoordinator.

    Used by the swing family where all 5 strategies share a single OMS
    and positions are available in-memory via the coordinator.
    """
    if coordinator is None:
        return []

    result = []
    try:
        # StrategyCoordinator tracks positions per strategy_id
        for strategy_id, positions in coordinator.get_all_positions().items():
            if strategy_id == current_strategy_id:
                continue
            for pos in positions:
                if isinstance(pos, dict):
                    symbol = pos.get("symbol", "")
                    qty = pos.get("net_qty", 0)
                else:
                    symbol = getattr(pos, "symbol", "")
                    qty = getattr(pos, "net_qty", 0)
                if qty == 0:
                    continue
                direction = "LONG" if qty > 0 else "SHORT"
                result.append({
                    "sibling_strategy_id": strategy_id,
                    "sibling_symbol": symbol,
                    "sibling_direction": direction,
                    "same_symbol": symbol == current_symbol,
                })
    except Exception as e:
        logger.warning("Failed to read coordinator positions: %s", e)

    return result
