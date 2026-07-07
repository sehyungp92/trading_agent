from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from libs.oms.engine.fill_processor import FillProcessor
from libs.oms.execution.router import ExecutionRouter, OrderPriority
from libs.oms.models.instrument import Instrument
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType, RiskContext
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.reconciliation.orchestrator import ReconciliationOrchestrator

sys.path.append(str(Path(__file__).resolve().parent))
from fake_ibkr import FakeIBKRExecutionAdapter  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_oms_restart_imports_offline_fills_before_next_bar() -> None:
    repo = InMemoryRepository()
    order = _order(status=OrderStatus.ROUTED)
    await repo.save_order(order)
    adapter = FakeIBKRExecutionAdapter()
    adapter.executions = [_exec("EXEC-1", broker_order_id=10001, qty=2, price=101.25)]
    orchestrator = ReconciliationOrchestrator(
        adapter,
        repo,
        _Bus(),
        fill_processor=FillProcessor(repo),
    )

    await orchestrator.startup_reconciliation()

    imported = await repo.get_order("OMS-1")
    assert await repo.fill_exists("EXEC-1") is True
    assert imported.status is OrderStatus.FILLED
    assert imported.filled_qty == 2
    assert imported.remaining_qty == 0
    assert imported.avg_fill_price == 101.25
    assert adapter.cache.is_fill_seen("EXEC-1") is True


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_oms_restart_does_not_double_import_fills() -> None:
    repo = InMemoryRepository()
    await repo.save_order(_order(status=OrderStatus.ROUTED))
    adapter = FakeIBKRExecutionAdapter()
    adapter.executions = [_exec("EXEC-1", broker_order_id=10001, qty=2, price=101.25)]
    orchestrator = ReconciliationOrchestrator(
        adapter,
        repo,
        _Bus(),
        fill_processor=FillProcessor(repo),
    )

    await orchestrator.startup_reconciliation()
    await orchestrator.startup_reconciliation()

    imported = await repo.get_order("OMS-1")
    assert len(repo._fills) == 1
    assert imported.filled_qty == 2
    assert imported.remaining_qty == 0


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_fill_processor_walks_routed_to_acked_to_filled() -> None:
    repo = InMemoryRepository()
    await repo.save_order(_order(status=OrderStatus.ROUTED))

    inserted = await FillProcessor(repo).process_fill(
        oms_order_id="OMS-1",
        broker_fill_id="EXEC-1",
        price=101.25,
        qty=2,
        timestamp=datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
        fees=1.24,
    )

    order = await repo.get_order("OMS-1")
    assert inserted is True
    assert order.status is OrderStatus.FILLED
    assert order.filled_qty == 2
    assert order.remaining_qty == 0


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_queued_order_resumes_after_router_restart() -> None:
    repo = InMemoryRepository()
    order = _order(status=OrderStatus.RISK_APPROVED)
    order.broker_order_id = None
    order.perm_id = None
    order.risk_context = RiskContext(
        stop_for_risk=95.0,
        planned_entry_price=100.0,
        risk_dollars=10.0,
    )
    await repo.save_order(order)

    congested = _SubmitAdapter(is_congested=True)
    router_before_restart = ExecutionRouter(congested, repo)
    await router_before_restart.route(order)

    queued = await repo.get_order(order.oms_order_id)
    assert queued.status is OrderStatus.QUEUED
    assert queued.queued_at is not None
    assert queued.queue_priority == int(OrderPriority.NEW_ENTRY)

    resumed = _SubmitAdapter(is_congested=False)
    router_after_restart = ExecutionRouter(
        resumed,
        repo,
        claimant_id="after-restart",
        pre_submit_recheck=lambda _order, conn=None: None,
    )
    await router_after_restart.drain_queue()

    routed = await repo.get_order(order.oms_order_id)
    assert routed.status is OrderStatus.ROUTED
    assert routed.queued_at == queued.queued_at
    assert routed.dequeued_at is not None
    assert routed.queue_claimed_by == ""
    assert resumed.submitted == [order.oms_order_id]


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_inflight_queued_drain_restart_recovers_broker_ref_without_resubmit() -> None:
    repo = InMemoryRepository()
    order = _order(status=OrderStatus.RISK_APPROVED)
    order.broker_order_id = None
    order.perm_id = None
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "forced_replay_congestion",
        queued_at,
        queued_at + timedelta(seconds=300),
    )
    order.status = OrderStatus.ROUTED
    order.dequeued_at = datetime.now(timezone.utc)
    order.submitted_at = order.dequeued_at
    order.broker_order_id = None
    order.perm_id = None
    await repo.save_order(order)

    resumed = _SubmitAdapter(
        is_congested=False,
        broker_orders=[
            SimpleNamespace(
                order_ref=order.client_order_id,
                broker_order_id=31001,
                perm_id=41001,
            )
        ],
    )
    router_after_restart = ExecutionRouter(
        resumed,
        repo,
        claimant_id="after-restart",
        pre_submit_recheck=lambda _order, conn=None: None,
    )
    await router_after_restart.drain_queue()

    routed = await repo.get_order(order.oms_order_id)
    assert routed.status is OrderStatus.ROUTED
    assert routed.broker_order_id == 31001
    assert routed.perm_id == 41001
    assert resumed.submitted == []


@pytest.mark.asyncio
@pytest.mark.parity_nightly
async def test_queued_order_expiry_lifecycle_name_is_replayable() -> None:
    repo = InMemoryRepository()
    order = _order(status=OrderStatus.RISK_APPROVED)
    order.broker_order_id = None
    order.perm_id = None
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "forced_replay_congestion",
        queued_at,
        queued_at,
    )

    expired = await repo.expire_due_queued_orders(
        queued_at.replace(microsecond=queued_at.microsecond + 1)
        if queued_at.microsecond < 999999
        else queued_at
    )

    assert [order.status.value for order in expired] == ["EXPIRED"]
    assert [event["event_type"] for event in repo._events] == [
        "ORDER_QUEUED",
        "QUEUED_ORDER_EXPIRED",
    ]


def _order(*, status: OrderStatus) -> OMSOrder:
    return OMSOrder(
        oms_order_id="OMS-1",
        client_order_id="CLIENT-1",
        strategy_id="PARITY_RESTART",
        account_id="DU123",
        instrument=_instrument(),
        side=OrderSide.BUY,
        qty=2,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        status=status,
        broker_order_id=10001,
        perm_id=20001,
        remaining_qty=2,
    )


def _instrument() -> Instrument:
    return Instrument(
        symbol="MNQ",
        root="MNQ",
        venue="CME",
        tick_size=0.25,
        tick_value=0.5,
        multiplier=2.0,
    )


def _exec(exec_id: str, *, broker_order_id: int, qty: float, price: float) -> SimpleNamespace:
    return SimpleNamespace(
        exec_id=exec_id,
        broker_order_id=broker_order_id,
        perm_id=20001,
        symbol="MNQ",
        side="BOT",
        qty=qty,
        price=price,
        fill_time=datetime(2026, 5, 20, 14, 31, tzinfo=timezone.utc),
        commission=1.24,
    )


class _Bus:
    def emit_risk_halt(self, *_args, **_kwargs) -> None:
        return None

    def emit_order_event(self, *_args, **_kwargs) -> None:
        return None


class _SubmitAdapter:
    def __init__(self, *, is_congested: bool, broker_orders: list | None = None) -> None:
        self.is_congested = is_congested
        self.submitted: list[str] = []
        self.broker_orders = list(broker_orders or [])

    async def submit_order(self, **kwargs):
        self.submitted.append(kwargs["oms_order_id"])
        return SimpleNamespace(broker_order_id=30001, perm_id=40001)

    async def request_open_orders(self):
        return list(self.broker_orders)
