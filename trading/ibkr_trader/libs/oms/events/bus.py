"""Event bus for OMS event distribution."""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from ..models.events import OMSEvent, OMSEventType
from ..models.order import OMSOrder, OrderStatus

logger = logging.getLogger(__name__)

_STATUS_TO_EVENT: dict[OrderStatus, OMSEventType] = {
    OrderStatus.CREATED: OMSEventType.ORDER_CREATED,
    OrderStatus.RISK_APPROVED: OMSEventType.ORDER_RISK_APPROVED,
    OrderStatus.QUEUED: OMSEventType.ORDER_QUEUED,
    OrderStatus.ROUTED: OMSEventType.ORDER_ROUTED,
    OrderStatus.ACKED: OMSEventType.ORDER_ACKED,
    OrderStatus.WORKING: OMSEventType.ORDER_WORKING,
    OrderStatus.PARTIALLY_FILLED: OMSEventType.ORDER_PARTIALLY_FILLED,
    OrderStatus.FILLED: OMSEventType.ORDER_FILLED,
    OrderStatus.CANCELLED: OMSEventType.ORDER_CANCELLED,
    OrderStatus.REJECTED: OMSEventType.ORDER_REJECTED,
    OrderStatus.EXPIRED: OMSEventType.ORDER_EXPIRED,
}


class EventBus:
    """Simple async event bus. Strategies subscribe by strategy_id."""

    def __init__(self, clock=None):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._global_subscribers: list[asyncio.Queue] = []
        self._clock = clock

    def _now(self) -> datetime:
        if self._clock is None:
            return datetime.now(timezone.utc)
        ts = self._clock()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def subscribe(self, strategy_id: str) -> asyncio.Queue:
        """Returns an asyncio.Queue that receives OMSEvent objects for this strategy."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[strategy_id].append(q)
        return q

    def subscribe_all(self) -> asyncio.Queue:
        """Subscribe to all events (for dashboard/logging)."""
        q: asyncio.Queue = asyncio.Queue()
        self._global_subscribers.append(q)
        return q

    def unsubscribe(self, strategy_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscription."""
        if strategy_id in self._subscribers:
            try:
                self._subscribers[strategy_id].remove(queue)
            except ValueError:
                pass

    def unsubscribe_all(self, queue: asyncio.Queue) -> None:
        """Remove a global subscription."""
        try:
            self._global_subscribers.remove(queue)
        except ValueError:
            pass

    def emit_order_event(self, order: OMSOrder) -> None:
        event_type = _STATUS_TO_EVENT.get(order.status)
        if not event_type:
            return  # No event for intermediate states (CANCEL_REQUESTED, etc.)

        event = OMSEvent(
            event_type=event_type,
            timestamp=order.last_update_at or order.created_at or self._now(),
            strategy_id=order.strategy_id,
            oms_order_id=order.oms_order_id,
            payload={
                "symbol": order.instrument.symbol if order.instrument else "",
                "status": order.status.value,
                "qty": order.qty,
                "filled_qty": order.filled_qty,
                "remaining_qty": order.remaining_qty,
                "avg_fill_price": order.avg_fill_price,
                "reject_reason": order.reject_reason,
                "client_order_id": order.client_order_id,
                "broker_order_id": order.broker_order_id,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "role": order.role.value,
                "queued_at": order.queued_at.isoformat() if order.queued_at else None,
                "queue_reason": order.queue_reason,
                "queue_priority": order.queue_priority,
                "queue_attempt": order.queue_attempt,
                "queue_expires_at": (
                    order.queue_expires_at.isoformat()
                    if order.queue_expires_at
                    else None
                ),
                "dequeued_at": order.dequeued_at.isoformat() if order.dequeued_at else None,
                "queue_denial_reason": order.queue_denial_reason,
            },
        )
        self._dispatch(event)

    def emit_fill_event(
        self, strategy_id: str, oms_order_id: str, fill_data: dict
    ) -> None:
        event = OMSEvent(
            event_type=OMSEventType.FILL,
            timestamp=self._now(),
            strategy_id=strategy_id,
            oms_order_id=oms_order_id,
            payload=fill_data,
        )
        self._dispatch(event)

    def emit_position_update(
        self, strategy_id: str, oms_order_id: str, payload: dict
    ) -> None:
        event = OMSEvent(
            event_type=OMSEventType.POSITION_UPDATE,
            timestamp=self._now(),
            strategy_id=strategy_id,
            oms_order_id=oms_order_id,
            payload=payload,
        )
        self._dispatch(event)

    def emit_risk_denial(
        self, strategy_id: str, oms_order_id: str, reason: str
    ) -> None:
        event = OMSEvent(
            event_type=OMSEventType.RISK_DENIAL,
            timestamp=self._now(),
            strategy_id=strategy_id,
            oms_order_id=oms_order_id,
            payload={"reason": reason},
        )
        self._dispatch(event)

    def emit_risk_decision(
        self, strategy_id: str, oms_order_id: str, payload: dict
    ) -> None:
        event = OMSEvent(
            event_type=OMSEventType.RISK_DECISION,
            timestamp=self._now(),
            strategy_id=strategy_id,
            oms_order_id=oms_order_id,
            payload=payload,
        )
        self._dispatch(event)

    def emit_risk_halt(self, strategy_id: str, reason: str) -> None:
        payload = {
            "reason": reason,
            "strategy_id": strategy_id,
            "halt_scope": "strategy" if strategy_id else "portfolio",
        }
        provider = getattr(self, "_current_oms_lineage", None)
        if callable(provider):
            try:
                from libs.instrumentation.event_contract import enrich_payload

                payload = enrich_payload(
                    payload,
                    lineage=provider(),
                    event_type="risk_halt",
                    scope="oms",
                )
            except Exception:
                pass
        event = OMSEvent(
            event_type=OMSEventType.RISK_HALT,
            timestamp=self._now(),
            strategy_id=strategy_id,
            payload=payload,
        )
        self._dispatch(event)

    def emit_reconciliation_event(self, payload: dict, strategy_id: str = "") -> None:
        event = OMSEvent(
            event_type=OMSEventType.RECONCILIATION_ALERT,
            timestamp=self._now(),
            strategy_id=strategy_id,
            payload=payload,
        )
        self._dispatch(event)

    def emit_coordination_event(
        self, target_strategy: str, event_type: str, **kwargs
    ) -> None:
        """Emit a coordination event to a specific strategy."""
        event = OMSEvent(
            event_type=OMSEventType.COORDINATION,
            timestamp=self._now(),
            strategy_id=target_strategy,
            payload={"coordination_type": event_type, **kwargs},
        )
        self._dispatch(event)
        logger.info(
            "Coordination event %s → %s: %s",
            event_type, target_strategy, kwargs,
        )

    def _dispatch(self, event: OMSEvent) -> None:
        if event.strategy_id:
            targets = list(self._subscribers.get(event.strategy_id, []))
        else:
            targets = [
                q
                for queues in self._subscribers.values()
                for q in queues
            ]

        for q in targets:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Event queue full for strategy {event.strategy_id}")
        for q in self._global_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Global event queue full")
