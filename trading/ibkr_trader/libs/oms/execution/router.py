"""Execution router for order dispatch."""
import asyncio
import inspect
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

from libs.market_data.futures_roll import with_contract_expiry_for_order

try:
    from libs.broker_ibkr.throttler import CongestionError
except Exception:  # pragma: no cover - broker package is present in runtime
    CongestionError = None  # type: ignore[assignment]

from ..models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType
from ..models.position import Position
from ..engine.state_machine import transition

if TYPE_CHECKING:
    from ..events.bus import EventBus
    from ..persistence.repository import OMSRepository

logger = logging.getLogger(__name__)

DRAIN_INTERVAL_SEC = 1.0
QUEUE_TTL_SECONDS = 300  # H2: queued orders expire after 5 minutes


class OrderPriority(IntEnum):
    STOP_EXIT = 0  # Highest
    CANCEL = 1
    REPLACE = 2
    NEW_ENTRY = 3  # Lowest


class ExecutionRouter:
    """Routes RISK_APPROVED orders to the broker adapter with priority queuing.
    Priority: stops/exits > cancels > replaces > new entries.
    """

    def __init__(
        self,
        adapter,
        repo: "OMSRepository",
        bus: "EventBus | None" = None,
        pre_submit_recheck: Callable[[OMSOrder], Awaitable[str | None]] | None = None,
        claimant_id: str | None = None,
    ):
        self._adapter = adapter  # IBKRExecutionAdapter
        self._repo = repo
        self._bus = bus
        self._pre_submit_recheck = pre_submit_recheck
        self._claimant_id = claimant_id or f"router-{uuid.uuid4()}"
        self._queue: list[tuple[OrderPriority, OMSOrder, dict]] = []
        self._drain_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background queue drain loop."""
        if self._running:
            return
        self._running = True
        self._drain_task = asyncio.create_task(self._drain_loop())
        logger.info("ExecutionRouter drain loop started")

    async def stop(self) -> None:
        """Stop the background queue drain loop."""
        self._running = False
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        await self._release_owned_queue_claims()
        logger.info("ExecutionRouter drain loop stopped")

    async def _drain_loop(self) -> None:
        """Background loop that drains queue when adapter is not congested."""
        while self._running:
            try:
                await self._recover_inflight_queued_orders()
                await self._expire_stale_queued_orders()
                if not self._adapter.is_congested:
                    await self.drain_queue()
                await asyncio.sleep(DRAIN_INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in drain loop: {e}")

    async def route(self, order: OMSOrder) -> None:
        """Route a risk-approved order to the adapter."""
        priority = self._get_priority(order)

        if self._adapter.is_congested:
            if order.role == OrderRole.ENTRY and priority > OrderPriority.CANCEL:
                logger.warning(f"Adapter congested; queueing {order.oms_order_id}")
                await self._queue_order(order, priority, "adapter_congested")
                return

        await self._submit_to_adapter(order)

    async def cancel(self, order: OMSOrder) -> None:
        if order.broker_order_id is None:
            return
        await self._adapter.cancel_order(order.broker_order_id, order.perm_id or 0)

    async def replace(
        self,
        order: OMSOrder,
        new_qty: Optional[int],
        new_limit_price: Optional[float],
        new_stop_price: Optional[float],
    ) -> None:
        if order.broker_order_id is None:
            return
        await self._adapter.replace_order(
            order.broker_order_id, new_qty, new_limit_price, new_stop_price
        )

    async def flatten(self, position: Position) -> OMSOrder:
        """Submit a market order to flatten a position. Returns the created order."""
        from ..models.instrument_registry import InstrumentRegistry
        side = OrderSide.SELL if position.net_qty > 0 else OrderSide.BUY
        qty = abs(int(position.net_qty))
        instrument = InstrumentRegistry.get(position.instrument_symbol)
        order = OMSOrder(
            strategy_id=position.strategy_id,
            account_id=position.account_id,
            instrument=instrument,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            role=OrderRole.EXIT,
            status=OrderStatus.RISK_APPROVED,
            remaining_qty=qty,
            created_at=datetime.now(timezone.utc),
        )
        await self._submit_to_adapter(order)
        return order

    async def _submit_to_adapter(self, order: OMSOrder) -> bool:
        if order.instrument is not None:
            order.instrument = with_contract_expiry_for_order(
                order.instrument,
                order_role=order.role.value,
                as_of=datetime.now(timezone.utc),
            )
        if not transition(order, OrderStatus.ROUTED):
            logger.warning(
                "Cannot route order %s: transition from %s to ROUTED is invalid",
                order.oms_order_id, order.status,
            )
            return False
        order.submitted_at = datetime.now(timezone.utc)
        await self._repo.save_order(order)

        try:
            ref = await self._submit_order_to_adapter(order)
        except Exception as exc:
            if (
                order.role == OrderRole.ENTRY
                and self._is_retryable_congestion_error(exc)
                and transition(order, OrderStatus.RISK_APPROVED)
            ):
                order.submitted_at = None
                order.last_update_at = datetime.now(timezone.utc)
                logger.warning(
                    "Broker submission for %s hit retryable congestion; requeueing",
                    order.oms_order_id,
                )
                await self._repo.save_order_and_event(
                    order,
                    "BROKER_SUBMIT_CONGESTED",
                    {
                        "error_type": exc.__class__.__name__,
                        "error": str(exc) or exc.__class__.__name__,
                    },
                )
                await self._queue_order(
                    order,
                    self._get_priority(order),
                    "adapter_congested_retry",
                )
                return False
            logger.exception(
                "Broker submission failed for order %s; rolling back to REJECTED",
                order.oms_order_id,
            )
            message = str(exc) or exc.__class__.__name__
            transition(order, OrderStatus.REJECTED)
            order.reject_reason = message
            order.last_update_at = datetime.now(timezone.utc)
            await self._repo.save_order_and_event(
                order,
                "BROKER_SUBMIT_FAILED",
                {
                    "error_type": exc.__class__.__name__,
                    "error": message,
                    "instrument_backed": order.instrument is not None,
                },
            )
            if self._bus is not None:
                self._bus.emit_order_event(order)
            return False

        order.broker_order_id = ref.broker_order_id
        order.perm_id = ref.perm_id
        await self._repo.save_order(order)
        return True

    async def _submit_order_to_adapter(self, order: OMSOrder):
        contract_expiry = order.instrument.contract_expiry if order.instrument else ""
        return await self._adapter.submit_order(
            oms_order_id=order.oms_order_id,
            contract_symbol=order.instrument.root if order.instrument else "",
            contract_expiry=contract_expiry,
            action=order.side.value,
            order_type=order.order_type.value,
            qty=order.qty,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            tif=order.tif,
            oca_group=order.oca_group,
            oca_type=order.oca_type,
            client_order_id=order.client_order_id or None,
            instrument=order.instrument,
        )

    def _has_async_repo_method(self, name: str) -> bool:
        method = getattr(self._repo, name, None)
        return bool(method and inspect.iscoroutinefunction(method))

    @staticmethod
    def _is_retryable_congestion_error(exc: Exception) -> bool:
        if CongestionError is not None and isinstance(exc, CongestionError):
            return True
        text = f"{exc.__class__.__name__} {exc}".lower()
        return "congest" in text or "pacing" in text

    async def _release_owned_queue_claims(self) -> None:
        if not self._has_async_repo_method("release_queued_claims"):
            return
        try:
            await self._repo.release_queued_claims(self._claimant_id)
        except Exception:
            logger.warning(
                "Failed to release queued order claims for %s",
                self._claimant_id,
                exc_info=True,
            )

    async def _queue_order(
        self,
        order: OMSOrder,
        priority: OrderPriority,
        reason: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=QUEUE_TTL_SECONDS)
        if order.status != OrderStatus.QUEUED and not transition(order, OrderStatus.QUEUED):
            logger.warning(
                "Cannot queue order %s from status %s",
                order.oms_order_id,
                order.status.value,
            )
            return
        order.queued_at = order.queued_at or now
        order.queue_priority = int(priority)
        order.queue_reason = reason
        order.queue_expires_at = expires_at
        order.queue_claimed_by = ""
        order.queue_claimed_at = None
        order.queue_claim_expires_at = None
        order.last_update_at = now

        if self._has_async_repo_method("mark_order_queued"):
            queued = await self._repo.mark_order_queued(
                order.oms_order_id,
                int(priority),
                reason,
                now,
                expires_at,
            )
            if queued is not None:
                order.status = queued.status
                order.queued_at = queued.queued_at
                order.queue_priority = queued.queue_priority
                order.queue_reason = queued.queue_reason
                order.queue_attempt = queued.queue_attempt
                order.queue_expires_at = queued.queue_expires_at
                order.last_update_at = queued.last_update_at
            else:
                persisted = await self._get_persisted_order(order.oms_order_id)
                if persisted is not None and persisted.status not in {
                    OrderStatus.RISK_APPROVED,
                    OrderStatus.QUEUED,
                }:
                    order.status = persisted.status
                    logger.warning(
                        "Queue persistence skipped for %s because DB status is %s",
                        order.oms_order_id,
                        persisted.status.value,
                    )
                    return
                if self._has_async_repo_method("save_order_and_event"):
                    await self._repo.save_order_and_event(
                        order,
                        "ORDER_QUEUED",
                        {
                            "priority": int(priority),
                            "reason": reason,
                            "queued_at": now.isoformat(),
                            "expires_at": expires_at.isoformat(),
                            "fallback": True,
                        },
                    )
                elif self._has_async_repo_method("save_order"):
                    await self._repo.save_order(order)
        else:
            self._queue.append((priority, order, {"queued_at": now}))
            if hasattr(self._repo, "save_order_and_event"):
                await self._repo.save_order_and_event(
                    order,
                    "ORDER_QUEUED",
                    {
                        "priority": int(priority),
                        "reason": reason,
                        "queued_at": now.isoformat(),
                        "expires_at": expires_at.isoformat(),
                    },
                )
        if self._bus is not None:
            self._bus.emit_order_event(order)

    async def _get_persisted_order(self, order_id: str) -> OMSOrder | None:
        method = getattr(self._repo, "get_order", None)
        if not method or not inspect.iscoroutinefunction(method):
            return None
        try:
            return await method(order_id)
        except Exception:
            logger.warning("Failed to read persisted order %s", order_id, exc_info=True)
            return None

    async def _recheck_queued_order(self, order: OMSOrder) -> str | None:
        if order.role != OrderRole.ENTRY or self._pre_submit_recheck is None:
            return None
        try:
            result = self._pre_submit_recheck(order, conn=None)  # type: ignore[misc,call-arg]
            if inspect.isawaitable(result):
                return await result
            return result
        except TypeError:
            result = self._pre_submit_recheck(order)
            if inspect.isawaitable(result):
                return await result
            return result

    @staticmethod
    def _get_priority(order: OMSOrder) -> OrderPriority:
        if order.role in {OrderRole.STOP, OrderRole.EXIT}:
            return OrderPriority.STOP_EXIT
        return OrderPriority.NEW_ENTRY

    async def drain_queue(self) -> None:
        """Process queued orders by priority when adapter is no longer congested.

        H2: Skip and expire orders older than QUEUE_TTL_SECONDS.
        """
        if self._adapter.is_congested:
            return
        await self._recover_inflight_queued_orders()
        await self._expire_stale_queued_orders()
        if self._has_async_repo_method("claim_queued_orders"):
            await self._drain_persisted_queue()
            return
        self._queue.sort(key=lambda x: x[0])
        while self._queue and not self._adapter.is_congested:
            _, order, _ = self._queue.pop(0)
            if order.status == OrderStatus.QUEUED:
                transition(order, OrderStatus.RISK_APPROVED)
                order.dequeued_at = datetime.now(timezone.utc)
            denial = await self._recheck_queued_order(order)
            if denial:
                if order.status == OrderStatus.RISK_APPROVED:
                    transition(order, OrderStatus.REJECTED)
                order.reject_reason = denial
                order.queue_denial_reason = denial
                order.last_update_at = datetime.now(timezone.utc)
                await self._repo.save_order_and_event(
                    order,
                    "QUEUED_ORDER_DENIED",
                    {"reason": denial},
                )
                if self._bus is not None:
                    self._bus.emit_order_event(order)
                    self._bus.emit_risk_denial(order.strategy_id, order.oms_order_id, denial)
                continue
            await self._submit_to_adapter(order)

    async def _drain_persisted_queue(self) -> None:
        while not self._adapter.is_congested:
            now = datetime.now(timezone.utc)
            claimed = await self._repo.claim_queued_orders(
                limit=10,
                claimant_id=self._claimant_id,
                now=now,
            )
            if not claimed:
                return
            for order in claimed:
                if self._adapter.is_congested:
                    await self._repo.release_queued_order(
                        order.oms_order_id,
                        self._claimant_id,
                    )
                    return
                denial = await self._recheck_queued_order(order)
                if denial:
                    denied = await self._repo.mark_queued_order_denied(
                        order.oms_order_id,
                        self._claimant_id,
                        denial,
                    )
                    if denied is not None:
                        order = denied
                    else:
                        order.status = OrderStatus.REJECTED
                        order.reject_reason = denial
                        order.queue_denial_reason = denial
                    if self._bus is not None:
                        self._bus.emit_order_event(order)
                        self._bus.emit_risk_denial(
                            order.strategy_id,
                            order.oms_order_id,
                            denial,
                        )
                    continue
                if self._adapter.is_congested:
                    await self._repo.release_queued_order(
                        order.oms_order_id,
                        self._claimant_id,
                    )
                    return
                await self._submit_claimed_queued_order(order)

    async def _submit_claimed_queued_order(self, order: OMSOrder) -> bool:
        recovered = await self._recover_existing_broker_submission(order)
        if recovered is not None:
            return recovered

        if order.instrument is not None:
            order.instrument = with_contract_expiry_for_order(
                order.instrument,
                order_role=order.role.value,
                as_of=datetime.now(timezone.utc),
            )

        started_at = datetime.now(timezone.utc)
        if self._has_async_repo_method("mark_queued_order_submit_started"):
            started = await self._repo.mark_queued_order_submit_started(
                order.oms_order_id,
                self._claimant_id,
                started_at,
            )
            if started is None:
                return False
            order = started
            if self._bus is not None:
                self._bus.emit_order_event(order)

        try:
            ref = await self._submit_order_to_adapter(order)
        except Exception as exc:
            if order.role == OrderRole.ENTRY and self._is_retryable_congestion_error(exc):
                order.status = OrderStatus.QUEUED
                order.submitted_at = None
                order.dequeued_at = None
                order.last_update_at = datetime.now(timezone.utc)
                if self._has_async_repo_method("save_order_and_event"):
                    await self._repo.save_order_and_event(
                        order,
                        "BROKER_SUBMIT_CONGESTED",
                        {
                            "error_type": exc.__class__.__name__,
                            "error": str(exc) or exc.__class__.__name__,
                            "claimant_id": self._claimant_id,
                        },
                    )
                await self._repo.mark_order_queued(
                    order.oms_order_id,
                    int(self._get_priority(order)),
                    "adapter_congested_retry",
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc) + timedelta(seconds=QUEUE_TTL_SECONDS),
                )
                return False

            message = str(exc) or exc.__class__.__name__
            logger.exception(
                "Broker submission failed for queued order %s; rejecting",
                order.oms_order_id,
            )
            rejected = await self._repo.mark_queued_order_denied(
                order.oms_order_id,
                self._claimant_id,
                message,
            )
            if rejected is not None and self._bus is not None:
                self._bus.emit_order_event(rejected)
            return False

        submitted_at = datetime.now(timezone.utc)
        submitted = await self._repo.mark_queued_order_submitted(
            order.oms_order_id,
            self._claimant_id,
            ref.broker_order_id,
            ref.perm_id,
            submitted_at,
            order.dequeued_at or started_at,
        )
        if submitted is None:
            logger.error(
                "Queued order %s was submitted to broker but could not be durably marked ROUTED",
                order.oms_order_id,
            )
            return False
        if self._bus is not None:
            self._bus.emit_order_event(submitted)
        return True

    async def _recover_inflight_queued_orders(self) -> None:
        if not self._has_async_repo_method("recover_inflight_queued_orders"):
            return
        recovered = await self._repo.recover_inflight_queued_orders(
            datetime.now(timezone.utc),
            "inflight queue drain recovered after restart",
        )
        if self._bus is not None:
            for order in recovered:
                self._bus.emit_order_event(order)

    async def _recover_existing_broker_submission(
        self,
        order: OMSOrder,
    ) -> bool | None:
        if not order.submitted_at and int(order.queue_attempt or 0) <= 1:
            return None

        broker_order, checked = await self._find_broker_order_by_ref(order)
        if broker_order is not None:
            now = datetime.now(timezone.utc)
            submitted = await self._repo.mark_queued_order_submitted(
                order.oms_order_id,
                self._claimant_id,
                getattr(broker_order, "broker_order_id", None),
                getattr(broker_order, "perm_id", None),
                order.submitted_at or now,
                order.dequeued_at or now,
            )
            if submitted is not None and self._bus is not None:
                self._bus.emit_order_event(submitted)
            return submitted is not None

        if order.submitted_at and not checked:
            logger.warning(
                "Queued order %s has submit-in-flight recovery pending because "
                "broker open-order snapshot is unavailable; leaving it queued "
                "and claimed for later recovery",
                order.oms_order_id,
            )
            return False

        if order.submitted_at:
            reason = (
                "queued broker submit recovery found no matching broker order; "
                "rejecting to avoid duplicate submit"
            )
            rejected = await self._repo.mark_queued_order_denied(
                order.oms_order_id,
                self._claimant_id,
                reason,
            )
            if rejected is not None and self._bus is not None:
                self._bus.emit_order_event(rejected)
            return False

        return None

    async def _find_broker_order_by_ref(self, order: OMSOrder) -> tuple[object | None, bool]:
        request_open_orders = getattr(self._adapter, "request_open_orders", None)
        if not callable(request_open_orders):
            return None, False
        refs = {ref for ref in (order.client_order_id, order.oms_order_id) if ref}
        if not refs:
            return None, True
        try:
            result = request_open_orders()
            broker_orders = await result if inspect.isawaitable(result) else result
        except Exception:
            logger.warning(
                "Failed to fetch broker open orders for queued recovery of %s",
                order.oms_order_id,
                exc_info=True,
            )
            return None, False
        for broker_order in broker_orders or []:
            if str(getattr(broker_order, "order_ref", "") or "") in refs:
                return broker_order, True
        return None, True

    async def _expire_stale_queued_orders(self) -> None:
        """Expire stale queued orders even if adapter congestion persists."""
        now = datetime.now(timezone.utc)
        if self._has_async_repo_method("expire_due_queued_orders"):
            expired = await self._repo.expire_due_queued_orders(now)
            if self._bus is not None:
                for order in expired:
                    self._bus.emit_order_event(order)
        fresh: list[tuple[OrderPriority, OMSOrder, dict]] = []

        for priority, order, meta in self._queue:
            queued_at = meta.get("queued_at")
            if queued_at and (now - queued_at).total_seconds() > QUEUE_TTL_SECONDS:
                logger.warning(
                    f"Expiring stale queued order {order.oms_order_id} "
                    f"(queued {(now - queued_at).total_seconds():.0f}s ago)"
                )
                if transition(order, OrderStatus.EXPIRED):
                    order.last_update_at = now
                    await self._repo.save_order_and_event(
                        order,
                        "QUEUE_EXPIRED",
                        {"queued_seconds": (now - queued_at).total_seconds()},
                    )
                    if self._bus is not None:
                        self._bus.emit_order_event(order)
                else:
                    fresh.append((priority, order, meta))
            else:
                fresh.append((priority, order, meta))

        self._queue = fresh
