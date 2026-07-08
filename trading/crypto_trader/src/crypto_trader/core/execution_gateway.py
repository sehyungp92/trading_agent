"""Broker-compatible execution gateway for canonical parity capture."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus
from crypto_trader.core.execution_adapter import ExecutionAdapter
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, OrderType
from crypto_trader.core.runtime_types import (
    DecisionContext,
    ExecutionReport,
    ExecutionReportKind,
    OrderIntent,
)


class ExecutionGateway:
    """Preserve the legacy broker API while emitting canonical order reports."""

    def __init__(
        self,
        *,
        adapter: ExecutionAdapter,
        broker: Any,
        events: EventBus | None = None,
        oms_store: Any | None = None,
        immediate_fill_sync: Callable[[str], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._broker = broker
        self._events = events
        self._oms = oms_store
        self._immediate_fill_sync = immediate_fill_sync
        self._decision_context: DecisionContext | None = None
        self._last_reports: list[ExecutionReport] = []
        self._pending_immediate_fill_syncs: list[str] = []

    @property
    def adapter(self) -> ExecutionAdapter:
        return self._adapter

    @property
    def last_reports(self) -> list[ExecutionReport]:
        return list(self._last_reports)

    def begin_decision_context(self, context: DecisionContext) -> None:
        self._decision_context = context

    def end_decision_context(self, context: DecisionContext) -> None:
        if self._decision_context is context:
            self._decision_context = None

    def submit_order(self, order: Order) -> str:
        """Submit through the adapter and return the strategy-visible order id."""
        context = self._decision_context
        submitted_at = _now()
        if context is not None:
            order.metadata.setdefault("decision_id", context.decision_id)
            order.metadata.setdefault("bar_id", context.metadata.get("bar_id"))
            order.metadata.setdefault("decision_time", context.decision_time.isoformat())
        order.metadata.setdefault("submitted_at", submitted_at.isoformat())
        intent = OrderIntent.from_order(order, context)
        order.metadata.setdefault("intent_id", intent.intent_id)
        if context is not None:
            context.record_order()
        self._emit("order_intent", intent.to_dict(), submitted_at)

        reports = self._adapter.submit(intent)
        self._last_reports = reports
        for report in reports:
            self._apply_report_to_order(order, report)
            self._record_report(report)
            self._emit("execution", report.to_dict(), report.timestamp)

        visible_id = _visible_order_id(order, reports, intent)
        if visible_id:
            order.metadata.setdefault("client_order_id", visible_id)
        if _should_sync_immediate_fill(order, reports) and self._immediate_fill_sync is not None:
            self._pending_immediate_fill_syncs.append(visible_id)
        return visible_id

    def record_rejected_order(
        self,
        order: Order,
        *,
        reject_reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a locally rejected order in canonical streams and the OMS."""
        context = self._decision_context
        submitted_at = _now()
        if context is not None:
            order.metadata.setdefault("decision_id", context.decision_id)
            order.metadata.setdefault("bar_id", context.metadata.get("bar_id"))
            order.metadata.setdefault("decision_time", context.decision_time.isoformat())
        if metadata:
            order.metadata.update(metadata)
        order.status = OrderStatus.REJECTED
        order.metadata.setdefault("submitted_at", submitted_at.isoformat())
        intent = OrderIntent.from_order(order, context)
        order.metadata.setdefault("intent_id", intent.intent_id)
        if context is not None:
            context.record_order()
        self._emit("order_intent", intent.to_dict(), submitted_at)

        client_order_id = intent.client_order_id or intent.intent_id or order.order_id
        report = ExecutionReport(
            report_id=f"local_reject_{client_order_id}",
            kind=ExecutionReportKind.REJECTED,
            timestamp=submitted_at,
            symbol=order.symbol,
            side=order.side,
            client_order_id=client_order_id,
            order_status=OrderStatus.REJECTED,
            qty=order.qty,
            reject_reason=reject_reason,
            metadata={
                **intent.metadata,
                **dict(metadata or {}),
                "intent_id": intent.intent_id,
                "strategy_id": intent.strategy_id,
                "decision_id": intent.decision_id,
                "order_type": intent.order_type.value,
                "reduce_only": intent.reduce_only,
                "time_in_force": intent.time_in_force,
                "ttl_bars": intent.ttl_bars,
                "oca_group": intent.oca_group,
                "bracket_group": intent.bracket_group,
            },
        )
        self._last_reports = [report]
        self._apply_report_to_order(order, report)
        self._record_report(report)
        self._emit("execution", report.to_dict(), report.timestamp)
        return client_order_id

    def drain_immediate_fill_syncs(self) -> None:
        """Run queued immediate fill syncs after strategy state has settled."""
        if self._immediate_fill_sync is None:
            self._pending_immediate_fill_syncs.clear()
            return
        while self._pending_immediate_fill_syncs:
            order_id = self._pending_immediate_fill_syncs.pop(0)
            self._immediate_fill_sync(order_id)

    def cancel_order(self, order_id: str) -> bool:
        reports = self._adapter.cancel(order_id)
        self._last_reports = reports
        for report in reports:
            self._record_report(report)
            self._emit("execution", report.to_dict(), report.timestamp)
        return any(report.order_status == OrderStatus.CANCELLED for report in reports)

    def cancel_all(self, symbol: str = "") -> int:
        cancelled = 0
        for order in self.get_open_orders(symbol):
            if self.cancel_order(order.order_id):
                cancelled += 1
        return cancelled

    def expire_ttl_orders_for_bar(self, bar: Bar) -> list[ExecutionReport]:
        expire_fn = getattr(self._adapter, "expire_ttl_orders_for_bar", None)
        if not callable(expire_fn):
            return []

        reports = list(expire_fn(bar))
        self._persist_ttl_state()
        if not reports:
            return []

        self._last_reports = reports
        for report in reports:
            self._record_report(report)
            self._emit("execution", report.to_dict(), report.timestamp)
        return reports

    def seed_ttl_orders_from_open_orders(self) -> int:
        seed_fn = getattr(self._adapter, "seed_ttl_orders_from_open_orders", None)
        if not callable(seed_fn):
            return 0
        seeded = int(seed_fn())
        self._persist_ttl_state()
        return seeded

    def seed_ttl_orders(self, orders: list[Order]) -> int:
        seed_fn = getattr(self._adapter, "seed_ttl_orders", None)
        if not callable(seed_fn):
            return self.seed_ttl_orders_from_open_orders()
        seeded = int(seed_fn(orders))
        self._persist_ttl_state()
        return seeded

    def clear_ttl_for_fill(self, fill: Fill) -> bool:
        clear_fn = getattr(self._adapter, "clear_ttl_for_fill", None)
        if not callable(clear_fn):
            return False
        cleared = bool(clear_fn(fill))
        if not cleared:
            return False
        if self._ttl_order_still_active(fill):
            self._persist_ttl_state()
        else:
            self._mark_ttl_tracking_cleared(fill)
        return True

    def get_position(self, symbol: str):
        return self._broker.get_position(symbol)

    def get_positions(self) -> list:
        return self._broker.get_positions()

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        return self._broker.get_open_orders(symbol)

    def get_equity(self) -> float:
        return self._broker.get_equity()

    def get_fills_since(self, since: datetime) -> list:
        return self._broker.get_fills_since(since)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._broker, name)

    def _apply_report_to_order(self, order: Order, report: ExecutionReport) -> None:
        if report.order_status is not None:
            order.status = report.order_status
        if report.client_order_id and not order.order_id:
            order.order_id = report.client_order_id
        elif report.client_order_id and report.client_order_id != order.order_id:
            order.metadata.setdefault("broker_order_id", order.order_id)
            order.order_id = report.client_order_id
        if report.exchange_order_id:
            order.metadata.setdefault("exchange_order_id", report.exchange_order_id)

    def _record_report(self, report: ExecutionReport) -> None:
        if self._oms is None:
            return
        record_fn = getattr(self._oms, "record_execution_report", None)
        if record_fn is not None:
            record_fn(report)
        if report.metadata.get("ttl_cancel_failed"):
            self._record_ttl_cancel_discrepancy(report)
        upsert_fn = getattr(self._oms, "upsert_order", None)
        if upsert_fn is not None and report.client_order_id:
            metadata = report.metadata
            order_metadata = {**report.to_dict(), **metadata}
            ttl_report = any(
                key in metadata
                for key in ("ttl_bars", "ttl_bars_alive", "ttl_remaining_qty", "ttl_cancel_failed")
            )
            if ttl_report:
                for key in ("ttl_bars", "ttl_bars_alive", "ttl_remaining_qty", "ttl_cancel_failed"):
                    if key in metadata:
                        order_metadata[key] = metadata[key]
                order_metadata["ttl_tracking_active"] = report.order_status not in {
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                }
            upsert_fn(
                client_order_id=report.client_order_id,
                exchange_order_id=report.exchange_order_id,
                strategy_id=str(metadata.get("strategy_id") or ""),
                symbol=report.symbol,
                side=report.side.value if report.side is not None else "",
                order_type=str(metadata.get("order_type") or ""),
                status=report.order_status.value if report.order_status is not None else report.kind.value.upper(),
                role=str(metadata.get("role") or metadata.get("tag") or ""),
                decision_id=str(metadata.get("decision_id") or ""),
                position_instance_id=str(metadata.get("position_instance_id") or ""),
                reduce_only=bool(metadata.get("reduce_only", False)),
                oca_group=metadata.get("oca_group"),
                bracket_group=metadata.get("bracket_group"),
                metadata=order_metadata,
            )

    def _persist_ttl_state(self) -> None:
        if self._oms is None:
            return
        active_fn = getattr(self._adapter, "active_ttl_orders", None)
        update_fn = getattr(self._oms, "update_order_metadata", None)
        if not callable(active_fn) or not callable(update_fn):
            return

        for state in active_fn():
            client_order_id = str(state.get("client_order_id") or "")
            if not client_order_id:
                continue
            metadata = dict(state.get("metadata") or {})
            metadata.update({
                "ttl_bars": state.get("ttl_bars"),
                "ttl_bars_alive": state.get("ttl_bars_alive", 0),
                "ttl_remaining_qty": state.get("remaining_qty", state.get("qty", 0.0)),
                "ttl_tracking_active": True,
            })
            update_fn(
                client_order_id,
                metadata_updates=metadata,
                status=OrderStatus.WORKING.value,
            )

    def _ttl_order_still_active(self, fill: Fill) -> bool:
        active_fn = getattr(self._adapter, "active_ttl_orders", None)
        if not callable(active_fn):
            return False
        fill_ids = {str(order_id) for order_id in (fill.order_id, fill.exchange_order_id) if order_id}
        for state in active_fn():
            state_ids = {
                str(order_id)
                for order_id in (
                    state.get("client_order_id"),
                    state.get("exchange_order_id"),
                )
                if order_id
            }
            if fill_ids & state_ids:
                return True
        return False

    def _mark_ttl_tracking_cleared(self, fill: Fill) -> None:
        if self._oms is None:
            return
        update_fn = getattr(self._oms, "update_order_metadata", None)
        if not callable(update_fn):
            return
        updates = {
            "ttl_tracking_active": False,
            "ttl_remaining_qty": 0.0,
        }
        if update_fn(
            fill.order_id,
            metadata_updates=updates,
            status=OrderStatus.FILLED.value,
        ):
            return
        if fill.exchange_order_id:
            update_fn(
                fill.exchange_order_id,
                metadata_updates=updates,
                status=OrderStatus.FILLED.value,
            )

    def _record_ttl_cancel_discrepancy(self, report: ExecutionReport) -> None:
        if self._oms is None:
            return
        record_fn = getattr(self._oms, "record_discrepancy", None)
        if not callable(record_fn):
            return

        client_order_id = report.client_order_id
        list_fn = getattr(self._oms, "list_unresolved_discrepancies", None)
        if callable(list_fn):
            for discrepancy in list_fn():
                if (
                    discrepancy.get("kind") == "ttl_cancel_failed"
                    and (discrepancy.get("metadata") or {}).get("client_order_id") == client_order_id
                ):
                    return

        record_fn(
            kind="ttl_cancel_failed",
            description="TTL emulation attempted to cancel an expired live order, but the broker rejected the cancel.",
            symbol=report.symbol,
            strategy_id=str(report.metadata.get("strategy_id") or ""),
            severity="warning",
            metadata={
                "client_order_id": client_order_id,
                "exchange_order_id": report.exchange_order_id,
                "ttl_bars": report.metadata.get("ttl_bars"),
                "ttl_bars_alive": report.metadata.get("ttl_bars_alive"),
                "reject_reason": report.reject_reason,
            },
        )

    def _emit(self, stream: str, payload: dict[str, Any], timestamp: datetime) -> None:
        if self._events is None:
            return
        self._events.emit(CanonicalRuntimeEvent(
            timestamp=timestamp,
            stream=stream,
            payload=payload,
        ))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _visible_order_id(
    order: Order,
    reports: list[ExecutionReport],
    intent: OrderIntent,
) -> str:
    for report in reports:
        if report.client_order_id:
            return report.client_order_id
    if order.order_id:
        return order.order_id
    return intent.client_order_id or intent.intent_id


def _should_sync_immediate_fill(order: Order, reports: list[ExecutionReport]) -> bool:
    if order.tag != "entry":
        return False
    if order.order_type != OrderType.MARKET:
        return False
    return any(report.order_status != OrderStatus.REJECTED for report in reports)
