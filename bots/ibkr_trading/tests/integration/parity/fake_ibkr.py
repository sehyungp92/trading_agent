from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from libs.broker_ibkr.models.types import BrokerOrderRef


class FakeIBKRCache:
    def __init__(self) -> None:
        self.contracts: dict[int, Any] = {}
        self._broker_to_oms: dict[int, str] = {}
        self._acked: set[str] = set()
        self._seen_fills: set[str] = set()

    def register_order(self, oms_order_id: str, broker_order_id: int, perm_id: int = 0) -> None:
        self._broker_to_oms[int(broker_order_id)] = oms_order_id

    def lookup_oms_id(self, broker_order_id: int) -> str | None:
        return self._broker_to_oms.get(int(broker_order_id))

    def mark_acked(self, oms_order_id: str) -> None:
        self._acked.add(oms_order_id)

    def is_acked(self, oms_order_id: str) -> bool:
        return oms_order_id in self._acked

    def mark_fill_seen(self, exec_id: str) -> None:
        self._seen_fills.add(exec_id)

    def is_fill_seen(self, exec_id: str) -> bool:
        return exec_id in self._seen_fills


class FakeIBKRExecutionAdapter:
    """Small IBKR adapter test double that drives production-style callbacks."""

    def __init__(self, *, auto_ack: bool = True) -> None:
        self.auto_ack = auto_ack
        self.is_congested = False
        self.cache = FakeIBKRCache()
        self.submitted: list[dict[str, Any]] = []
        self.cancelled: list[tuple[int, int]] = []
        self.replaced: list[dict[str, Any]] = []
        self.open_orders: list[Any] = []
        self.positions: list[Any] = []
        self.executions: list[Any] = []
        self._next_broker_order_id = 10_000

        self.on_ack = lambda *args: None
        self.on_reject = lambda *args: None
        self.on_fill = lambda *args: None
        self.on_status = lambda *args: None
        self.on_positions_snapshot = lambda *args: None

    async def submit_order(
        self,
        oms_order_id: str,
        contract_symbol: str = "",
        contract_expiry: str = "",
        action: str = "",
        order_type: str = "",
        qty: int = 0,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: str = "DAY",
        oca_group: str = "",
        oca_type: int = 0,
        client_order_id: str | None = None,
        instrument: Any | None = None,
    ) -> BrokerOrderRef:
        self._next_broker_order_id += 1
        ref = BrokerOrderRef(self._next_broker_order_id, self._next_broker_order_id + 1_000_000, 0)
        self.cache.register_order(oms_order_id, ref.broker_order_id, ref.perm_id)
        self.submitted.append(
            {
                "oms_order_id": oms_order_id,
                "contract_symbol": contract_symbol,
                "contract_expiry": contract_expiry,
                "action": action,
                "order_type": order_type,
                "qty": qty,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "tif": tif,
                "oca_group": oca_group,
                "oca_type": oca_type,
                "client_order_id": client_order_id,
                "instrument": instrument,
                "ref": ref,
            }
        )
        if self.auto_ack:
            self.cache.mark_acked(oms_order_id)
            self.on_ack(oms_order_id, ref)
        return ref

    async def cancel_order(self, broker_order_id: int, perm_id: int = 0) -> None:
        self.cancelled.append((broker_order_id, perm_id))

    async def replace_order(
        self,
        broker_order_id: int,
        new_qty: int | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> BrokerOrderRef:
        ref = BrokerOrderRef(broker_order_id, broker_order_id + 1_000_000, 0)
        self.replaced.append(
            {
                "broker_order_id": broker_order_id,
                "new_qty": new_qty,
                "new_limit_price": new_limit_price,
                "new_stop_price": new_stop_price,
                "ref": ref,
            }
        )
        return ref

    async def request_open_orders(self) -> list[Any]:
        return list(self.open_orders)

    async def request_positions(self) -> list[Any]:
        return list(self.positions)

    async def request_executions(self, since_ts: datetime | None = None) -> list[Any]:
        if since_ts is None:
            return list(self.executions)
        return [
            execution
            for execution in self.executions
            if _execution_time(execution) is None or _execution_time(execution) >= since_ts
        ]

    async def rebuild_cache(
        self,
        oms_order_id_resolver,
        fill_exists_check=None,
        fill_importer=None,
    ) -> None:
        for execution in self.executions:
            broker_order_id = int(getattr(execution, "broker_order_id"))
            oms_order_id = (
                getattr(execution, "oms_order_id", None)
                or self.cache.lookup_oms_id(broker_order_id)
                or await oms_order_id_resolver(broker_order_id)
            )
            if not oms_order_id:
                continue
            self.cache.register_order(oms_order_id, broker_order_id, getattr(execution, "perm_id", 0) or 0)
            exec_id = getattr(execution, "exec_id")
            if fill_exists_check is not None and await fill_exists_check(exec_id):
                self.cache.mark_fill_seen(exec_id)
                continue
            if fill_importer is not None and await fill_importer(oms_order_id, execution):
                self.cache.mark_fill_seen(exec_id)

    def emit_status(self, broker_order_id: int, status: str, remaining: float = 0.0) -> None:
        oms_order_id = self.cache.lookup_oms_id(broker_order_id)
        if oms_order_id:
            self.on_status(oms_order_id, status, remaining)

    def emit_fill(
        self,
        broker_order_id: int,
        *,
        exec_id: str,
        price: float,
        qty: float,
        commission: float = 0.0,
        fill_time: datetime | None = None,
    ) -> None:
        if self.cache.is_fill_seen(exec_id):
            return
        self.cache.mark_fill_seen(exec_id)
        oms_order_id = self.cache.lookup_oms_id(broker_order_id)
        if oms_order_id:
            self.on_fill(
                oms_order_id,
                exec_id,
                price,
                qty,
                fill_time or datetime.now(timezone.utc),
                commission,
            )

    def emit_reject(
        self,
        broker_order_id: int,
        reason: str,
        error_code: int = 0,
        retryable: bool = False,
    ) -> None:
        oms_order_id = self.cache.lookup_oms_id(broker_order_id)
        if oms_order_id:
            self.on_reject(oms_order_id, reason, error_code, retryable)


def with_oms_order_id(execution: Any, oms_order_id: str) -> Any:
    if hasattr(execution, "__dataclass_fields__"):
        try:
            return replace(execution, oms_order_id=oms_order_id)
        except TypeError:
            payload = {field: getattr(execution, field) for field in execution.__dataclass_fields__}
            payload["oms_order_id"] = oms_order_id
            return SimpleNamespace(**payload)
    try:
        setattr(execution, "oms_order_id", oms_order_id)
        return execution
    except Exception:
        payload = dict(getattr(execution, "__dict__", {}))
        payload["oms_order_id"] = oms_order_id
        return SimpleNamespace(**payload)


def _execution_time(execution: Any) -> datetime | None:
    return getattr(execution, "fill_time", None) or getattr(execution, "timestamp", None)
