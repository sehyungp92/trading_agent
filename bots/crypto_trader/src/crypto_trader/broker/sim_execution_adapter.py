"""Execution adapter wrapper for ``SimBroker``."""

from __future__ import annotations

from datetime import datetime, timezone

from crypto_trader.core.execution_adapter import ExecutionCapabilities
from crypto_trader.core.models import Fill, Order, OrderStatus, OrderType
from crypto_trader.core.runtime_types import (
    ExecutionReport,
    ExecutionReportKind,
    OrderIntent,
)


class SimExecutionAdapter:
    """Translate canonical order intents to the existing simulated broker."""

    capabilities = ExecutionCapabilities(
        stop_limit=True,
        oca=True,
        ttl=True,
        partial_fills=True,
    )

    def __init__(self, broker) -> None:
        self._broker = broker
        self._broker_id_by_client_id: dict[str, str] = {}
        self._client_id_by_broker_id: dict[str, str] = {}

    def submit(self, intent: OrderIntent) -> list[ExecutionReport]:
        order = _order_from_intent(intent)
        requested_client_id = intent.client_order_id
        result_id = self._broker.submit_order(order)
        client_order_id = requested_client_id or result_id or intent.intent_id
        if result_id:
            self._remember_order_ids(client_order_id, result_id)
            order.metadata["client_order_id"] = client_order_id
            order.metadata["strategy_id"] = intent.strategy_id
            kind = ExecutionReportKind.ACCEPTED
        else:
            kind = ExecutionReportKind.REJECTED

        return [_order_report(
            kind=kind,
            intent=intent,
            order=order,
            client_order_id=client_order_id,
            reject_reason="" if result_id else "sim_broker_rejected",
        )]

    def cancel(self, client_order_id: str) -> list[ExecutionReport]:
        broker_order_id = self._broker_id_by_client_id.get(client_order_id, client_order_id)
        ok = self._broker.cancel_order(broker_order_id)
        return [ExecutionReport(
            report_id=f"sim_cancel_{client_order_id}",
            kind=ExecutionReportKind.CANCELLED if ok else ExecutionReportKind.REJECTED,
            timestamp=datetime.now(timezone.utc),
            symbol="",
            client_order_id=client_order_id,
            exchange_order_id=broker_order_id,
            order_status=OrderStatus.CANCELLED if ok else OrderStatus.REJECTED,
            reject_reason="" if ok else "cancel_rejected",
        )]

    def sync_open_orders(self) -> list[ExecutionReport]:
        reports = []
        active = (OrderStatus.PENDING, OrderStatus.WORKING)
        deferred = [
            order for order in getattr(self._broker, "_deferred_orders", [])
            if order.status in active
        ]
        for order in [*self._broker.get_open_orders(), *deferred]:
            client_order_id = (
                order.metadata.get("client_order_id")
                or self._client_id_by_broker_id.get(order.order_id)
                or order.order_id
            )
            self._remember_order_ids(str(client_order_id), order.order_id)
            reports.append(_order_report(
                kind=ExecutionReportKind.RESTING,
                intent=None,
                order=order,
                client_order_id=str(client_order_id),
            ))
        return reports

    def sync_positions(self) -> list[dict]:
        return [position.__dict__.copy() for position in self._broker.get_positions()]

    def sync_fills(self, watermark: datetime) -> list[ExecutionReport]:
        reports = [
            _fill_report(
                fill,
                client_order_id=self._client_id_by_broker_id.get(fill.order_id, fill.order_id),
            )
            for fill in self._broker.get_fills_since(watermark)
        ]
        drain_cancelled = getattr(self._broker, "drain_cancelled_oca_orders", None)
        if callable(drain_cancelled):
            for order in drain_cancelled():
                client_order_id = (
                    order.metadata.get("client_order_id")
                    or self._client_id_by_broker_id.get(order.order_id)
                    or order.order_id
                )
                reports.append(_order_report(
                    kind=ExecutionReportKind.CANCELLED,
                    intent=None,
                    order=order,
                    client_order_id=str(client_order_id),
                ))
        return reports

    def _remember_order_ids(self, client_order_id: str, broker_order_id: str) -> None:
        if not client_order_id or not broker_order_id:
            return
        self._broker_id_by_client_id[client_order_id] = broker_order_id
        self._client_id_by_broker_id[broker_order_id] = client_order_id


def _order_from_intent(intent: OrderIntent) -> Order:
    metadata = {
        "strategy_id": intent.strategy_id,
        "decision_id": intent.decision_id,
        "reduce_only": intent.reduce_only,
        "oca_group": intent.oca_group,
        "bracket_group": intent.bracket_group,
        **intent.risk_metadata,
        **intent.metadata,
    }
    if intent.client_order_id:
        metadata["client_order_id"] = intent.client_order_id
    return Order(
        order_id=intent.client_order_id,
        symbol=intent.symbol,
        side=intent.side,
        order_type=intent.order_type,
        qty=intent.qty,
        limit_price=intent.limit_price,
        stop_price=intent.stop_price,
        tag=str(intent.metadata.get("tag") or ""),
        oca_group=intent.oca_group,
        time_in_force=intent.time_in_force,
        ttl_bars=intent.ttl_bars,
        metadata=metadata,
    )


def _order_report(
    *,
    kind: ExecutionReportKind,
    intent: OrderIntent | None,
    order: Order,
    client_order_id: str,
    reject_reason: str = "",
) -> ExecutionReport:
    return ExecutionReport(
        report_id=f"sim_{kind.value}_{client_order_id or order.order_id}",
        kind=kind,
        timestamp=datetime.now(timezone.utc),
        symbol=order.symbol,
        side=order.side,
        client_order_id=client_order_id,
        exchange_order_id=order.order_id,
        order_status=order.status,
        qty=order.qty,
        reject_reason=reject_reason,
        metadata=_report_metadata(intent) if intent is not None else dict(order.metadata),
    )


def _report_metadata(intent: OrderIntent) -> dict:
    metadata = dict(intent.metadata)
    metadata.update({
        "intent_id": intent.intent_id,
        "strategy_id": intent.strategy_id,
        "decision_id": intent.decision_id,
        "order_type": intent.order_type.value,
        "reduce_only": intent.reduce_only,
        "oca_group": intent.oca_group,
        "bracket_group": intent.bracket_group,
    })
    return metadata


def _fill_report(fill: Fill, *, client_order_id: str) -> ExecutionReport:
    return ExecutionReport(
        report_id=f"sim_fill_{fill.order_id}_{int(fill.timestamp.timestamp() * 1000)}",
        kind=ExecutionReportKind.FILL,
        timestamp=fill.timestamp,
        symbol=fill.symbol,
        side=fill.side,
        client_order_id=client_order_id,
        exchange_order_id=fill.exchange_order_id or fill.order_id,
        fill_id=fill.exchange_fill_id or f"{fill.order_id}:{int(fill.timestamp.timestamp() * 1000)}",
        order_status=OrderStatus.FILLED,
        filled_qty=fill.qty,
        fill_price=fill.fill_price,
        commission=fill.commission,
        metadata={
            "tag": fill.tag,
            "broker_order_id": fill.order_id,
            **dict(fill.raw),
        },
    )
