"""Order timeout monitor for stuck transient states.

Scans for orders stuck in ROUTED or CANCEL_REQUESTED beyond configurable
thresholds without inventing terminal broker state.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..models.order import OrderStatus
from .state_machine import transition

if TYPE_CHECKING:
    from ..events.bus import EventBus
    from ..execution.router import ExecutionRouter
    from ..persistence.repository import OMSRepository

logger = logging.getLogger(__name__)

DEFAULT_ROUTED_TIMEOUT_S = 30.0
DEFAULT_CANCEL_REQUESTED_TIMEOUT_S = 15.0
DEFAULT_SCAN_INTERVAL_S = 5.0


class OrderTimeoutMonitor:
    """Background task that detects and escalates stuck transient orders."""

    def __init__(
        self,
        repo: "OMSRepository",
        bus: "EventBus",
        router: "ExecutionRouter | None" = None,
        routed_timeout_s: float = DEFAULT_ROUTED_TIMEOUT_S,
        cancel_timeout_s: float = DEFAULT_CANCEL_REQUESTED_TIMEOUT_S,
        scan_interval_s: float = DEFAULT_SCAN_INTERVAL_S,
    ):
        self._repo = repo
        self._bus = bus
        self._router = router
        self._routed_timeout_s = routed_timeout_s
        self._cancel_timeout_s = cancel_timeout_s
        self._scan_interval_s = scan_interval_s
        self._task: asyncio.Task | None = None
        self._running = False
        self._db_timeout_streak = 0
        self._db_degraded = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "OrderTimeoutMonitor started: routed=%ss, cancel=%ss, scan=%ss",
            self._routed_timeout_s,
            self._cancel_timeout_s,
            self._scan_interval_s,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        backoff = self._scan_interval_s
        while self._running:
            try:
                await self._scan_stuck_orders()
                if self._db_degraded:
                    logger.info(
                        "OMS timeout monitor database connectivity recovered after %d consecutive timeouts",
                        self._db_timeout_streak,
                    )
                self._db_timeout_streak = 0
                self._db_degraded = False
                backoff = self._scan_interval_s
                await asyncio.sleep(self._scan_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._is_db_timeout(exc):
                    self._db_timeout_streak += 1
                    if self._db_timeout_streak >= 3 and not self._db_degraded:
                        self._db_degraded = True
                        logger.error(
                            "OMS timeout monitor degraded: repeated database acquire timeouts (%d consecutive)",
                            self._db_timeout_streak,
                        )
                logger.warning("Timeout monitor error (retry in %.0fs): %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    @staticmethod
    def _is_db_timeout(exc: Exception) -> bool:
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, TimeoutError):
                return True
            current = current.__cause__ or current.__context__
        return False

    async def _scan_stuck_orders(self) -> None:
        now = datetime.now(timezone.utc)
        working_orders = await self._repo.get_all_working_orders()

        for order in working_orders:
            if order.status == OrderStatus.ROUTED:
                ref_time = order.submitted_at or order.created_at
                if ref_time and (now - ref_time).total_seconds() > self._routed_timeout_s:
                    logger.warning(
                        "Order %s stuck in ROUTED for >%ss; requesting broker cancel",
                        order.oms_order_id,
                        self._routed_timeout_s,
                    )
                    if transition(order, OrderStatus.CANCEL_REQUESTED):
                        order.last_update_at = now
                        await self._repo.save_order_and_event(
                            order,
                            "TIMEOUT_CANCEL_REQUESTED",
                            {"reason": "routed_timeout", "timeout_s": self._routed_timeout_s},
                        )
                        if self._router is not None:
                            await self._router.cancel(order)

            elif order.status == OrderStatus.CANCEL_REQUESTED:
                ref_time = order.last_update_at or order.created_at
                if ref_time and (now - ref_time).total_seconds() > self._cancel_timeout_s:
                    logger.warning(
                        "Order %s stuck in CANCEL_REQUESTED for >%ss; reconciliation required",
                        order.oms_order_id,
                        self._cancel_timeout_s,
                    )
                    order.last_update_at = now
                    await self._repo.save_order_and_event(
                        order,
                        "TIMEOUT_CANCEL_RECONCILE_REQUIRED",
                        {"reason": "cancel_timeout", "timeout_s": self._cancel_timeout_s},
                    )
