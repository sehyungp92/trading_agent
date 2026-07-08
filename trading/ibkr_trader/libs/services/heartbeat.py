"""Heartbeat service for strategy and adapter health monitoring.

Strategies and adapters should call these functions periodically
to update their health state in the database.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping, Optional

from ..oms.persistence.postgres import PgStore
from ..oms.persistence.schema import StrategyStateRow, AdapterStateRow
from .decision_codes import is_known as _is_known_decision_code

logger = logging.getLogger(__name__)

# Warned (strategy_id, code) pairs are remembered so the 15s heartbeat loop
# does not spam the log. Reset only on process restart -- intentional, since
# a fresh deploy is exactly when an operator wants to see drift.
_warned_decision_codes: set[tuple[str, str]] = set()


class HeartbeatService:
    """Manages health state updates for strategies and adapters."""

    def __init__(self, store: PgStore):
        self._store = store

    async def strategy_heartbeat(
        self,
        strategy_id: str,
        mode: str = "RUNNING",
        heat_r: Decimal = Decimal("0"),
        daily_pnl_r: Decimal = Decimal("0"),
        last_decision_code: Optional[str] = None,
        last_decision_details: Optional[dict] = None,
        last_seen_bar_ts: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update strategy health state. Call every 10-30 seconds."""
        if last_decision_code is not None and not _is_known_decision_code(last_decision_code):
            key = (strategy_id, last_decision_code)
            if key not in _warned_decision_codes:
                _warned_decision_codes.add(key)
                logger.warning(
                    "Unknown decision_code emitted by strategy=%s code=%r -- "
                    "see libs/services/decision_codes.py for the canonical taxonomy",
                    strategy_id,
                    last_decision_code,
                )
        row = StrategyStateRow(
            strategy_id=strategy_id,
            mode=mode,
            last_heartbeat_ts=datetime.now(timezone.utc),
            heat_r=heat_r,
            daily_pnl_r=daily_pnl_r,
            last_decision_code=last_decision_code,
            last_decision_details_json=json.dumps(last_decision_details)
            if last_decision_details
            else "{}",
            last_seen_bar_ts=last_seen_bar_ts,
            last_error=error,
            last_error_ts=datetime.now(timezone.utc) if error else None,
        )
        await self._store.upsert_strategy_state(row)

    async def adapter_heartbeat(
        self,
        adapter_id: str,
        connected: bool,
        broker: str = "IBKR",
    ) -> None:
        """Update adapter connection state. Call every 10-30 seconds."""
        row = AdapterStateRow(
            adapter_id=adapter_id,
            broker=broker,
            connected=connected,
            last_heartbeat_ts=datetime.now(timezone.utc),
        )
        await self._store.upsert_adapter_state(row)

    async def record_disconnect(
        self,
        adapter_id: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record adapter disconnect event."""
        await self._store.record_adapter_disconnect(adapter_id, error_code, error_message)

    async def record_reconnect(self, adapter_id: str) -> None:
        """Record adapter reconnection."""
        await self._store.record_adapter_connect(adapter_id)


async def emit_heartbeat(
    store: PgStore,
    strategy_id: str,
    heat_r: float = 0.0,
    daily_pnl_r: float = 0.0,
    mode: str = "RUNNING",
    decision_code: str = None,
) -> None:
    """Simple heartbeat for strategy engines."""
    svc = HeartbeatService(store)
    await svc.strategy_heartbeat(
        strategy_id=strategy_id,
        mode=mode,
        heat_r=Decimal(str(heat_r)),
        daily_pnl_r=Decimal(str(daily_pnl_r)),
        last_decision_code=decision_code,
    )


async def emit_family_heartbeats(
    heartbeat: HeartbeatService,
    family_id: str,
    strategy_payloads: Iterable[Mapping[str, Any]],
    adapter_connected: bool | None = None,
    timeout_s: float = 5.0,
) -> None:
    """Emit strategy heartbeats concurrently with bounded per-write latency."""
    payloads = list(strategy_payloads)

    async def _emit_strategy(payload: Mapping[str, Any]) -> None:
        sid = str(payload.get("strategy_id", "unknown"))
        try:
            await asyncio.wait_for(
                heartbeat.strategy_heartbeat(**dict(payload)),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("Strategy heartbeat timed out: family=%s strategy=%s", family_id, sid)
        except Exception:
            logger.debug(
                "Strategy heartbeat failed: family=%s strategy=%s",
                family_id,
                sid,
                exc_info=True,
            )

    if payloads:
        await asyncio.gather(*(_emit_strategy(payload) for payload in payloads))

    if adapter_connected is None:
        return
    try:
        await asyncio.wait_for(
            heartbeat.adapter_heartbeat(family_id, connected=adapter_connected),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("Adapter heartbeat timed out: family=%s", family_id)
    except Exception:
        logger.debug("Adapter heartbeat failed: family=%s", family_id, exc_info=True)
