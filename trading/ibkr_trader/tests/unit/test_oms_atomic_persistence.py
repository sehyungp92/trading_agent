from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from libs.oms.engine.fill_processor import FillProcessor
from libs.oms.engine.timeout_monitor import OrderTimeoutMonitor
from libs.oms.execution.router import ExecutionRouter, OrderPriority, QUEUE_TTL_SECONDS
from libs.oms.engine.state_machine import transition
from libs.oms.intent.handler import IntentHandler
from libs.oms.models.instrument import Instrument
from libs.oms.models.intent import Intent, IntentResult, IntentType, PreapprovedFamilyDecision
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType, RiskContext
from libs.oms.models.risk_state import PortfolioRiskState
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.persistence.repository import OMSPersistenceInvariantError, OMSRepository
from libs.oms.services.factory import (
    _hydrate_open_positions_from_repo,
    _import_fill_through_adapter_callback,
    _wire_adapter_callbacks,
    _wire_adapter_callbacks_multi,
)
from libs.broker_ibkr.state.cache import IBCache
from libs.broker_ibkr.throttler import CongestionError
from libs.risk.account_risk_gate import AccountRiskGate


def _instrument(symbol: str = "QQQ") -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.01,
        tick_value=0.01,
        multiplier=1.0,
        point_value=1.0,
        currency="USD",
        primary_exchange="NASDAQ",
        sec_type="STK",
    )


def _entry_order(symbol: str = "QQQ") -> OMSOrder:
    return OMSOrder(
        oms_order_id=f"oms-{symbol.lower()}",
        client_order_id=f"client-{symbol.lower()}",
        strategy_id="TEST",
        account_id="DU123",
        instrument=_instrument(symbol),
        side=OrderSide.BUY,
        qty=10,
        order_type=OrderType.LIMIT,
        limit_price=100.0,
        role=OrderRole.ENTRY,
        status=OrderStatus.CREATED,
    )


def _preapproved_decision(
    order: OMSOrder,
    *,
    original_qty: int,
    approved_qty: int,
    status: str = "accepted",
) -> PreapprovedFamilyDecision:
    return PreapprovedFamilyDecision(
        candidate_key=f"{order.strategy_id}|{order.instrument.symbol}|ENTRY|{order.side.value}|1",
        family_surface="unit_test_family_replay",
        strategy_id=order.strategy_id,
        symbol=order.instrument.symbol,
        side=order.side.value,
        role=order.role.value,
        sequence=1,
        original_qty=original_qty,
        approved_qty=approved_qty,
        status=status,
    )


class _Acquire:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _BrokenConnection:
    async def execute(self, *args, **kwargs):
        raise RuntimeError("fk violation")


class _BrokenPool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self):
        return _Acquire(self._conn)


class _AccountGateConn:
    def __init__(self, *, daily_realized: float = 0.0, weekly_realized: float = 0.0) -> None:
        self.locked = False
        self.queries: list[str] = []
        self.daily_realized = daily_realized
        self.weekly_realized = weekly_realized

    async def execute(self, query, *args):
        if "pg_advisory_xact_lock" in query:
            self.locked = True

    async def fetchrow(self, query, *args):
        self.queries.append(query)
        if "FROM positions" in query:
            return {"total_open_risk_dollars": 200.0}
        if "FROM orders" in query:
            return {"pending_entry_risk_dollars": 150.0}
        if "FROM risk_daily_portfolio" in query:
            if "total_weekly_realized_dollars" in query:
                return {"total_weekly_realized_dollars": self.weekly_realized}
            return {"total_daily_realized_dollars": self.daily_realized}
        raise AssertionError(query)


class _HydrationFailRepository:
    async def get_positions_for_strategies(self, strategy_ids):
        raise RuntimeError("db down")


class _DelayedWorkingSaveRepository(InMemoryRepository):
    def __init__(self) -> None:
        super().__init__()
        self.working_save_started = asyncio.Event()
        self.release_working_save = asyncio.Event()
        self.saved_statuses: list[OrderStatus] = []

    async def save_order(self, order: OMSOrder) -> None:
        self.saved_statuses.append(order.status)
        if order.status == OrderStatus.WORKING and not self.release_working_save.is_set():
            self.working_save_started.set()
            await self.release_working_save.wait()
        await super().save_order(order)


class _ExecutionSnapshot:
    def __init__(self, executions: list[SimpleNamespace]) -> None:
        self._executions = executions

    async def fetch_open_orders(self) -> list:
        return []

    async def fetch_executions(self):
        return self._executions


async def _wait_for_order_status(
    repo: InMemoryRepository,
    oms_order_id: str,
    expected: OrderStatus,
) -> OMSOrder:
    for _ in range(200):
        order = await repo.get_order(oms_order_id)
        if order is not None and order.status == expected:
            return order
        await asyncio.sleep(0.01)
    raise AssertionError(f"Order {oms_order_id} never reached {expected.value}")


@pytest.mark.asyncio
async def test_intent_handler_denial_uses_atomic_helper() -> None:
    risk = MagicMock()
    risk.check_entry = AsyncMock(return_value="risk denied")
    router = MagicMock()
    router.route = AsyncMock()
    repo = MagicMock()
    repo.get_order_id_by_client_order_id = AsyncMock(return_value=None)
    repo.get_positions = AsyncMock(return_value=[])
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_risk_denial = MagicMock()
    bus.emit_order_event = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order()

    receipt = await handler.submit(
        Intent(intent_type=IntentType.NEW_ORDER, strategy_id="TEST", order=order)
    )

    assert receipt.result == IntentResult.DENIED
    repo.save_order_and_event.assert_awaited_once()
    saved_order, event_type, payload = repo.save_order_and_event.await_args.args
    assert saved_order.status == OrderStatus.REJECTED
    assert event_type == "RISK_DENIED"
    assert payload == {"reason": "risk denied"}


@pytest.mark.asyncio
async def test_intent_handler_approval_uses_atomic_helper() -> None:
    risk = MagicMock()
    risk.check_entry = AsyncMock(return_value=None)
    risk.check_account_gate = AsyncMock(return_value=None)
    router = MagicMock()
    router.route = AsyncMock()
    repo = MagicMock()
    repo.get_order_id_by_client_order_id = AsyncMock(return_value=None)
    repo.get_positions = AsyncMock(return_value=[])
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_risk_denial = MagicMock()
    bus.emit_order_event = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("GLD")

    receipt = await handler.submit(
        Intent(intent_type=IntentType.NEW_ORDER, strategy_id="TEST", order=order)
    )

    assert receipt.result == IntentResult.ACCEPTED
    repo.save_order_and_event.assert_awaited_once()
    saved_order, event_type, payload = repo.save_order_and_event.await_args.args
    assert saved_order.status == OrderStatus.RISK_APPROVED
    assert event_type == "RISK_APPROVED"
    assert payload == {}
    risk.check_entry.assert_awaited_once()
    assert risk.check_entry.await_args.kwargs["skip_account_gate"] is True
    risk.check_account_gate.assert_awaited_once()


@pytest.mark.asyncio
async def test_intent_handler_preapproved_order_uses_public_submission_path() -> None:
    risk = MagicMock()
    risk.check_entry = AsyncMock()
    risk.check_preapproved_entry = AsyncMock(return_value=None)
    risk.check_account_gate = AsyncMock(return_value=None)
    router = MagicMock()
    router.route = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()
    bus.emit_risk_denial = MagicMock()
    bus.emit_order_event = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("MSFT")
    order.risk_context = RiskContext(
        stop_for_risk=99.0,
        planned_entry_price=100.0,
        risk_dollars=10.0,
        portfolio_size_mult=0.5,
    )

    receipt = await handler.submit(
        Intent(
            intent_type=IntentType.PREAPPROVED_ORDER,
            strategy_id="TEST",
            order=order,
            preapproved_family_decision=_preapproved_decision(order, original_qty=10, approved_qty=10),
        )
    )

    assert receipt.result == IntentResult.ACCEPTED
    assert order.qty == 10
    risk.check_entry.assert_not_awaited()
    risk.check_preapproved_entry.assert_awaited_once_with(order)
    risk.check_account_gate.assert_awaited_once()
    router.route.assert_awaited_once_with(order)
    saved = await repo.get_order(order.oms_order_id)
    assert saved is not None
    assert saved.status == OrderStatus.RISK_APPROVED
    assert [event["event_type"] for event in repo._events] == ["RISK_APPROVED"]
    bus.emit_order_event.assert_called_once_with(order)


@pytest.mark.asyncio
async def test_intent_handler_preapproved_order_requires_family_decision() -> None:
    risk = MagicMock()
    risk.check_preapproved_entry = AsyncMock(return_value=None)
    risk.check_account_gate = AsyncMock(return_value=None)
    router = MagicMock()
    router.route = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("MSFT")

    receipt = await handler.submit(
        Intent(
            intent_type=IntentType.PREAPPROVED_ORDER,
            strategy_id="TEST",
            order=order,
        )
    )

    assert receipt.result == IntentResult.DENIED
    assert "preapproved_family_decision" in (receipt.denial_reason or "")
    risk.check_preapproved_entry.assert_not_awaited()
    router.route.assert_not_awaited()


@pytest.mark.asyncio
async def test_intent_handler_preapproved_order_rejects_inconsistent_decision() -> None:
    risk = MagicMock()
    risk.check_preapproved_entry = AsyncMock(return_value=None)
    risk.check_account_gate = AsyncMock(return_value=None)
    router = MagicMock()
    router.route = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("MSFT")

    receipt = await handler.submit(
        Intent(
            intent_type=IntentType.PREAPPROVED_ORDER,
            strategy_id="TEST",
            order=order,
            preapproved_family_decision=_preapproved_decision(
                order,
                original_qty=10,
                approved_qty=9,
                status="accepted",
            ),
        )
    )

    assert receipt.result == IntentResult.DENIED
    assert "preserve quantity" in (receipt.denial_reason or "")
    risk.check_preapproved_entry.assert_not_awaited()
    router.route.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "original_qty", "approved_qty", "expected_reason"),
    [
        ("reduced", 10, 10, "must reduce"),
        ("rejected", 10, 1, "Invalid preapproved family decision status"),
        ("accepted", 10, 9, "preserve quantity"),
    ],
)
async def test_intent_handler_preapproved_order_rejects_invalid_family_payloads(
    status: str,
    original_qty: int,
    approved_qty: int,
    expected_reason: str,
) -> None:
    risk = MagicMock()
    risk.check_preapproved_entry = AsyncMock(return_value=None)
    risk.check_account_gate = AsyncMock(return_value=None)
    router = MagicMock()
    router.route = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("MSFT")

    receipt = await handler.submit(
        Intent(
            intent_type=IntentType.PREAPPROVED_ORDER,
            strategy_id="TEST",
            order=order,
            preapproved_family_decision=_preapproved_decision(
                order,
                original_qty=original_qty,
                approved_qty=approved_qty,
                status=status,
            ),
        )
    )

    assert receipt.result == IntentResult.DENIED
    assert expected_reason in (receipt.denial_reason or "")
    risk.check_preapproved_entry.assert_not_awaited()
    router.route.assert_not_awaited()


@pytest.mark.asyncio
async def test_intent_handler_account_gate_denial_does_not_persist_approval() -> None:
    risk = MagicMock()
    risk.check_entry = AsyncMock(return_value=None)
    risk.check_account_gate = AsyncMock(return_value="Account gate: heat cap")
    router = MagicMock()
    router.route = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()
    bus.emit_risk_denial = MagicMock()
    bus.emit_order_event = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("AAPL")

    receipt = await handler.submit(
        Intent(intent_type=IntentType.NEW_ORDER, strategy_id="TEST", order=order)
    )

    assert receipt.result == IntentResult.DENIED
    saved = await repo.get_order(order.oms_order_id)
    assert saved is not None
    assert saved.status == OrderStatus.REJECTED
    assert [event["event_type"] for event in repo._events] == ["RISK_DENIED"]
    router.route.assert_not_awaited()


@pytest.mark.asyncio
async def test_intent_handler_account_gate_sees_multiplier_adjusted_risk() -> None:
    async def _check_entry(order: OMSOrder, *, skip_account_gate: bool = False):
        assert skip_account_gate is True
        order.risk_context.portfolio_size_mult = 0.5
        order.risk_context.risk_dollars = 100.0
        return None

    async def _check_account_gate(order: OMSOrder, conn=None):
        assert order.qty == 5
        assert order.remaining_qty == 5
        assert order.risk_context.risk_dollars == pytest.approx(50.0)
        assert conn is None
        return None

    risk = MagicMock()
    risk.check_entry = AsyncMock(side_effect=_check_entry)
    risk.check_account_gate = AsyncMock(side_effect=_check_account_gate)
    router = MagicMock()
    router.route = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()

    handler = IntentHandler(risk, router, repo, bus)
    order = _entry_order("TSLA")
    order.risk_context = RiskContext(
        stop_for_risk=90.0,
        planned_entry_price=100.0,
        risk_dollars=100.0,
    )

    receipt = await handler.submit(
        Intent(intent_type=IntentType.NEW_ORDER, strategy_id="TEST", order=order)
    )

    assert receipt.result == IntentResult.ACCEPTED
    router.route.assert_awaited_once()


@pytest.mark.asyncio
async def test_account_risk_gate_counts_pending_entry_reservations() -> None:
    gate = AccountRiskGate(MagicMock(), heat_cap_R=2.0, daily_stop_R=3.0, account_urd=200.0)
    conn = _AccountGateConn()

    decision = await gate.check_entry("momentum", risk_dollars=100.0, conn=conn)

    assert conn.locked is True
    assert not decision.approved
    assert "Account heat cap" in (decision.reason or "")
    assert "reserved 1.75R" in (decision.reason or "")


@pytest.mark.asyncio
async def test_account_risk_gate_subtracts_reserved_queued_order_on_recheck() -> None:
    gate = AccountRiskGate(MagicMock(), heat_cap_R=2.0, daily_stop_R=3.0, account_urd=200.0)
    conn = _AccountGateConn()

    decision = await gate.check_entry(
        "momentum",
        risk_dollars=100.0,
        conn=conn,
        reserved_risk_dollars=150.0,
    )

    assert decision.approved


@pytest.mark.asyncio
async def test_account_risk_gate_denies_global_standdown_without_db_queries() -> None:
    gate = AccountRiskGate(
        MagicMock(),
        heat_cap_R=2.0,
        daily_stop_R=3.0,
        weekly_stop_R=5.0,
        account_urd=200.0,
        global_standdown=True,
    )
    conn = _AccountGateConn()

    decision = await gate.check_entry("stock", risk_dollars=100.0, conn=conn)

    assert not decision.approved
    assert decision.reason == "Global stand-down active"
    assert conn.locked is False
    assert conn.queries == []


@pytest.mark.asyncio
async def test_account_risk_gate_denies_account_weekly_stop() -> None:
    gate = AccountRiskGate(
        MagicMock(),
        heat_cap_R=99.0,
        daily_stop_R=3.0,
        weekly_stop_R=5.0,
        account_urd=200.0,
    )
    conn = _AccountGateConn(weekly_realized=-1_000.0)

    decision = await gate.check_entry("swing", risk_dollars=100.0, conn=conn)

    assert conn.locked is True
    assert not decision.approved
    assert "Account weekly stop" in (decision.reason or "")
    assert "realized -5.00R" in (decision.reason or "")


@pytest.mark.asyncio
async def test_account_risk_gate_skips_disabled_daily_and_weekly_stops() -> None:
    gate = AccountRiskGate(
        MagicMock(),
        heat_cap_R=99.0,
        daily_stop_R=0.0,
        weekly_stop_R=0.0,
        account_urd=200.0,
    )
    conn = _AccountGateConn(daily_realized=-10_000.0, weekly_realized=-10_000.0)

    decision = await gate.check_entry("momentum", risk_dollars=100.0, conn=conn)

    assert decision.approved
    assert not any("risk_daily_portfolio" in query for query in conn.queries)


@pytest.mark.asyncio
async def test_execution_router_queue_expiry_uses_atomic_helper() -> None:
    adapter = MagicMock()
    adapter.is_congested = True
    repo = MagicMock()
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    router = ExecutionRouter(adapter, repo, bus)
    now = datetime.now(timezone.utc)
    order = _entry_order()
    order.status = OrderStatus.RISK_APPROVED
    router._queue = [
        (
            OrderPriority.NEW_ENTRY,
            order,
            {"queued_at": now - timedelta(seconds=301)},
        )
    ]

    await router._expire_stale_queued_orders()

    assert order.status == OrderStatus.EXPIRED
    repo.save_order_and_event.assert_awaited_once()
    assert router._queue == []


@pytest.mark.asyncio
async def test_congested_entry_persists_as_queued() -> None:
    adapter = MagicMock()
    adapter.is_congested = True
    adapter.submit_order = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    router = ExecutionRouter(adapter, repo, bus)
    order = _entry_order("AAPL")
    order.status = OrderStatus.RISK_APPROVED
    order.risk_context = RiskContext(
        stop_for_risk=95.0,
        planned_entry_price=100.0,
        risk_dollars=50.0,
    )
    await repo.save_order(order)

    await router.route(order)

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status == OrderStatus.QUEUED
    assert persisted.queued_at is not None
    assert persisted.queue_priority == int(OrderPriority.NEW_ENTRY)
    assert persisted.queue_reason == "adapter_congested"
    assert persisted.queue_expires_at is not None
    adapter.submit_order.assert_not_awaited()
    bus.emit_order_event.assert_called_with(order)


@pytest.mark.asyncio
async def test_queued_entry_reserves_pending_risk_and_working_capacity() -> None:
    repo = InMemoryRepository()
    order = _entry_order("AMD")
    order.status = OrderStatus.QUEUED
    order.risk_context = RiskContext(
        stop_for_risk=95.0,
        planned_entry_price=100.0,
        risk_dollars=125.0,
    )
    await repo.save_order(order)

    assert await repo.get_pending_entry_risk_R(25.0) == pytest.approx(5.0)
    assert await repo.count_working_orders(order.strategy_id) == 1


@pytest.mark.asyncio
async def test_queued_entry_expires_from_repository_queue() -> None:
    adapter = MagicMock()
    adapter.is_congested = True
    repo = InMemoryRepository()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    router = ExecutionRouter(adapter, repo, bus)
    order = _entry_order("TSLA")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc) - timedelta(seconds=QUEUE_TTL_SECONDS + 1)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )

    await router._expire_stale_queued_orders()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status == OrderStatus.EXPIRED
    assert persisted.queue_denial_reason == "queue TTL expired"
    bus.emit_order_event.assert_called()


@pytest.mark.asyncio
async def test_queued_entry_re_risk_denial_blocks_submit() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock()
    repo = InMemoryRepository()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_risk_denial = MagicMock()

    async def deny(_order, conn=None):
        return "Global stand-down active"

    router = ExecutionRouter(adapter, repo, bus, pre_submit_recheck=deny)
    order = _entry_order("NVDA")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        datetime.now(timezone.utc),
        datetime.now(timezone.utc) + timedelta(seconds=QUEUE_TTL_SECONDS),
    )

    await router.drain_queue()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status == OrderStatus.REJECTED
    assert persisted.reject_reason == "Global stand-down active"
    adapter.submit_order.assert_not_awaited()
    bus.emit_risk_denial.assert_called_once()


@pytest.mark.asyncio
async def test_queued_entry_submit_preserves_audit_and_clears_claim() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(
        return_value=SimpleNamespace(broker_order_id=404, perm_id=505)
    )
    repo = InMemoryRepository()
    router = ExecutionRouter(adapter, repo, pre_submit_recheck=AsyncMock(return_value=None))
    order = _entry_order("META")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )

    await router.drain_queue()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status == OrderStatus.ROUTED
    assert persisted.queued_at == queued_at
    assert persisted.dequeued_at is not None
    assert persisted.queue_claimed_by == ""
    assert persisted.queue_claimed_at is None
    assert persisted.queue_claim_expires_at is None
    adapter.submit_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_persisted_queue_drain_keeps_order_claimed_until_broker_ref_is_saved() -> None:
    repo = InMemoryRepository()
    order = _entry_order("IBM")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )

    async def submit_order(**kwargs):
        during_submit = await repo.get_order(kwargs["oms_order_id"])
        assert during_submit.status is OrderStatus.QUEUED
        assert during_submit.queue_claimed_by == "crash-safe-router"
        assert during_submit.broker_order_id is None
        return SimpleNamespace(broker_order_id=606, perm_id=707)

    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(side_effect=submit_order)
    router = ExecutionRouter(adapter, repo, claimant_id="crash-safe-router")

    await router.drain_queue()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status is OrderStatus.ROUTED
    assert persisted.broker_order_id == 606
    assert persisted.queue_claimed_by == ""


@pytest.mark.asyncio
async def test_submit_inflight_queue_claim_cannot_be_stolen_or_expired_before_recovery() -> None:
    repo = InMemoryRepository()
    order = _entry_order("ORCL")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )
    claimed = await repo.claim_queued_orders(
        limit=1,
        claimant_id="worker-a",
        now=queued_at,
    )
    assert claimed
    started = await repo.mark_queued_order_submit_started(
        order.oms_order_id,
        "worker-a",
        queued_at,
    )
    assert started is not None
    assert started.status is OrderStatus.QUEUED
    assert started.submitted_at is not None

    after_claim_ttl = queued_at + timedelta(seconds=31)
    stolen = await repo.claim_queued_orders(
        limit=1,
        claimant_id="worker-b",
        now=after_claim_ttl,
    )
    assert stolen == []

    expired = await repo.expire_due_queued_orders(
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS + 1)
    )
    assert expired == []
    during_submit = await repo.get_order(order.oms_order_id)
    assert during_submit.status is OrderStatus.QUEUED
    assert during_submit.queue_claimed_by == "worker-a"
    assert during_submit.submitted_at is not None

    during_submit.queue_claim_expires_at = queued_at - timedelta(seconds=1)
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.request_open_orders = AsyncMock(
        return_value=[
            SimpleNamespace(
                order_ref=order.client_order_id,
                broker_order_id=6060,
                perm_id=7070,
            )
        ]
    )
    adapter.submit_order = AsyncMock()
    router = ExecutionRouter(adapter, repo, claimant_id="worker-b")

    await router.drain_queue()

    recovered = await repo.get_order(order.oms_order_id)
    assert recovered.status is OrderStatus.ROUTED
    assert recovered.broker_order_id == 6060
    assert recovered.queue_claimed_by == ""
    adapter.submit_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_inflight_recovery_stays_pending_when_broker_snapshot_unavailable() -> None:
    repo = InMemoryRepository()
    order = _entry_order("SAP")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )
    claimed = await repo.claim_queued_orders(
        limit=1,
        claimant_id="worker-a",
        now=queued_at,
    )
    assert claimed
    started = await repo.mark_queued_order_submit_started(
        order.oms_order_id,
        "worker-a",
        queued_at,
    )
    assert started is not None
    started.queue_claim_expires_at = queued_at - timedelta(seconds=1)

    adapter = MagicMock()
    adapter.is_congested = False
    adapter.request_open_orders = AsyncMock(side_effect=RuntimeError("IB snapshot down"))
    adapter.submit_order = AsyncMock()
    router = ExecutionRouter(adapter, repo, claimant_id="worker-b")

    await router.drain_queue()

    pending = await repo.get_order(order.oms_order_id)
    assert pending.status is OrderStatus.QUEUED
    assert pending.broker_order_id is None
    assert pending.submitted_at == queued_at
    assert pending.reject_reason == ""
    assert pending.queue_claimed_by == "worker-b"
    adapter.request_open_orders.assert_awaited_once()
    adapter.submit_order.assert_not_awaited()

    pending.queue_claim_expires_at = queued_at - timedelta(seconds=1)
    adapter.request_open_orders = AsyncMock(
        return_value=[
            SimpleNamespace(
                order_ref=order.client_order_id,
                broker_order_id=5151,
                perm_id=6161,
            )
        ]
    )
    await router.drain_queue()

    recovered = await repo.get_order(order.oms_order_id)
    assert recovered.status is OrderStatus.ROUTED
    assert recovered.broker_order_id == 5151
    assert recovered.reject_reason == ""
    adapter.submit_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_inflight_dequeued_order_recovers_after_restart_and_submits_once() -> None:
    repo = InMemoryRepository()
    order = _entry_order("ADBE")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )
    claimed = await repo.claim_queued_orders(
        limit=1,
        claimant_id="before-crash",
        now=datetime.now(timezone.utc),
    )
    assert claimed
    await repo.mark_queued_order_dequeued(
        order.oms_order_id,
        "before-crash",
        datetime.now(timezone.utc),
    )

    adapter = MagicMock()
    adapter.is_congested = False
    adapter.request_open_orders = AsyncMock(return_value=[])
    adapter.submit_order = AsyncMock(
        return_value=SimpleNamespace(broker_order_id=808, perm_id=909)
    )
    router = ExecutionRouter(adapter, repo, claimant_id="after-crash")

    await router.drain_queue()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status is OrderStatus.ROUTED
    assert persisted.broker_order_id == 808
    adapter.submit_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_inflight_routed_without_broker_id_recovers_matching_broker_ref_without_resubmit() -> None:
    repo = InMemoryRepository()
    order = _entry_order("INTC")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )
    order.status = OrderStatus.ROUTED
    order.dequeued_at = datetime.now(timezone.utc)
    order.submitted_at = order.dequeued_at
    order.broker_order_id = None
    order.perm_id = None
    await repo.save_order(order)

    adapter = MagicMock()
    adapter.is_congested = False
    adapter.request_open_orders = AsyncMock(
        return_value=[
            SimpleNamespace(
                order_ref=order.client_order_id,
                broker_order_id=1001,
                perm_id=1002,
            )
        ]
    )
    adapter.submit_order = AsyncMock()
    router = ExecutionRouter(adapter, repo, claimant_id="after-crash")

    await router.drain_queue()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status is OrderStatus.ROUTED
    assert persisted.broker_order_id == 1001
    assert persisted.perm_id == 1002
    adapter.submit_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_inflight_routed_without_broker_ref_rejects_instead_of_duplicate_submit() -> None:
    repo = InMemoryRepository()
    order = _entry_order("CSCO")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )
    order.status = OrderStatus.ROUTED
    order.dequeued_at = datetime.now(timezone.utc)
    order.submitted_at = order.dequeued_at
    order.broker_order_id = None
    await repo.save_order(order)

    adapter = MagicMock()
    adapter.is_congested = False
    adapter.request_open_orders = AsyncMock(return_value=[])
    adapter.submit_order = AsyncMock()
    router = ExecutionRouter(adapter, repo, claimant_id="after-crash")

    await router.drain_queue()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status is OrderStatus.REJECTED
    assert "avoid duplicate submit" in persisted.reject_reason
    adapter.submit_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_drain_workers_submit_claimed_order_once() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(
        return_value=SimpleNamespace(broker_order_id=707, perm_id=808)
    )
    repo = InMemoryRepository()
    order = _entry_order("MSFT")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    now = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        now,
        now + timedelta(seconds=QUEUE_TTL_SECONDS),
    )
    router_a = ExecutionRouter(adapter, repo, claimant_id="worker-a")
    router_b = ExecutionRouter(adapter, repo, claimant_id="worker-b")

    await asyncio.gather(router_a.drain_queue(), router_b.drain_queue())

    adapter.submit_order.assert_awaited_once()
    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status == OrderStatus.ROUTED


@pytest.mark.asyncio
async def test_queued_submit_failure_does_not_emit_submitted_audit() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(side_effect=RuntimeError("submit exploded"))
    repo = InMemoryRepository()
    router = ExecutionRouter(adapter, repo, pre_submit_recheck=AsyncMock(return_value=None))
    order = _entry_order("SHOP")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )

    await router.drain_queue()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status == OrderStatus.REJECTED
    assert persisted.reject_reason == "submit exploded"
    assert "QUEUED_ORDER_SUBMITTED" not in [
        event["event_type"] for event in repo._events
    ]


@pytest.mark.asyncio
async def test_retryable_submit_congestion_requeues_entry() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(side_effect=CongestionError("orders congested"))
    repo = InMemoryRepository()
    router = ExecutionRouter(adapter, repo)
    order = _entry_order("CRM")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)

    submitted = await router._submit_to_adapter(order)

    persisted = await repo.get_order(order.oms_order_id)
    assert submitted is False
    assert persisted.status == OrderStatus.QUEUED
    assert persisted.submitted_at is None
    assert persisted.queue_reason == "adapter_congested_retry"
    assert [event["event_type"] for event in repo._events] == [
        "BROKER_SUBMIT_CONGESTED",
        "ORDER_QUEUED",
    ]


@pytest.mark.asyncio
async def test_router_stop_releases_owned_queue_claims() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    repo = InMemoryRepository()
    order = _entry_order("ORCL")
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)
    queued_at = datetime.now(timezone.utc)
    await repo.mark_order_queued(
        order.oms_order_id,
        int(OrderPriority.NEW_ENTRY),
        "adapter_congested",
        queued_at,
        queued_at + timedelta(seconds=QUEUE_TTL_SECONDS),
    )
    claimed = await repo.claim_queued_orders(
        limit=1,
        claimant_id="router-a",
        now=datetime.now(timezone.utc),
    )
    assert claimed[0].queue_claimed_by == "router-a"

    await ExecutionRouter(adapter, repo, claimant_id="router-a").stop()

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.queue_claimed_by == ""
    assert persisted.queue_claimed_at is None
    assert persisted.queue_claim_expires_at is None


@pytest.mark.asyncio
async def test_protective_exit_bypasses_entry_queue_and_recheck() -> None:
    adapter = MagicMock()
    adapter.is_congested = True
    adapter.submit_order = AsyncMock(
        return_value=SimpleNamespace(broker_order_id=909, perm_id=1001)
    )
    repo = InMemoryRepository()
    recheck = AsyncMock(return_value="should not run")
    router = ExecutionRouter(adapter, repo, pre_submit_recheck=recheck)
    order = _entry_order("SPY")
    order.role = OrderRole.EXIT
    order.status = OrderStatus.RISK_APPROVED
    await repo.save_order(order)

    await router.route(order)

    persisted = await repo.get_order(order.oms_order_id)
    assert persisted.status == OrderStatus.ROUTED
    adapter.submit_order.assert_awaited_once()
    recheck.assert_not_awaited()


def test_order_state_machine_allows_queue_lifecycle_not_direct_fill() -> None:
    order = _entry_order("QQQ")
    order.status = OrderStatus.RISK_APPROVED

    assert transition(order, OrderStatus.QUEUED)
    assert not transition(order, OrderStatus.FILLED)
    assert transition(order, OrderStatus.RISK_APPROVED)


@pytest.mark.asyncio
async def test_execution_router_submit_failure_persists_rejection_event_and_emits_bus() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(side_effect=RuntimeError("submit exploded"))
    repo = MagicMock()
    repo.save_order = AsyncMock()
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    router = ExecutionRouter(adapter, repo, bus)
    order = _entry_order("MSFT")
    order.status = OrderStatus.RISK_APPROVED
    order.remaining_qty = order.qty

    await router._submit_to_adapter(order)

    assert order.status == OrderStatus.REJECTED
    assert order.reject_reason == "submit exploded"
    assert order.last_update_at is not None
    repo.save_order.assert_awaited_once()
    repo.save_order_and_event.assert_awaited_once()
    saved_order, event_type, payload = repo.save_order_and_event.await_args.args
    assert saved_order is order
    assert event_type == "BROKER_SUBMIT_FAILED"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "submit exploded"
    assert payload["instrument_backed"] is True
    bus.emit_order_event.assert_called_once_with(order)


@pytest.mark.asyncio
async def test_execution_router_passes_instrument_to_adapter_submit() -> None:
    adapter = MagicMock()
    adapter.is_congested = False
    adapter.submit_order = AsyncMock(
        return_value=SimpleNamespace(broker_order_id=101, perm_id=202)
    )
    repo = MagicMock()
    repo.save_order = AsyncMock()
    router = ExecutionRouter(adapter, repo)
    order = _entry_order("GLD")
    order.status = OrderStatus.RISK_APPROVED
    order.remaining_qty = order.qty

    await router._submit_to_adapter(order)

    assert adapter.submit_order.await_args.kwargs["instrument"] is order.instrument
    assert order.broker_order_id == 101
    assert order.perm_id == 202


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "timestamp_attr", "payload_reason", "expected_status", "expected_event"),
    [
        (
            OrderStatus.ROUTED,
            "submitted_at",
            "routed_timeout",
            OrderStatus.CANCEL_REQUESTED,
            "TIMEOUT_CANCEL_REQUESTED",
        ),
        (
            OrderStatus.CANCEL_REQUESTED,
            "last_update_at",
            "cancel_timeout",
            OrderStatus.CANCEL_REQUESTED,
            "TIMEOUT_CANCEL_RECONCILE_REQUIRED",
        ),
    ],
)
async def test_timeout_monitor_uses_atomic_helper(
    status: OrderStatus,
    timestamp_attr: str,
    payload_reason: str,
    expected_status: OrderStatus,
    expected_event: str,
) -> None:
    repo = MagicMock()
    repo.save_order_and_event = AsyncMock()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    router = MagicMock()
    router.cancel = AsyncMock()
    monitor = OrderTimeoutMonitor(repo, bus, router, routed_timeout_s=30.0, cancel_timeout_s=15.0)
    order = _entry_order("NFLX")
    order.status = status
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=60)
    setattr(order, timestamp_attr, stale_ts)
    order.created_at = stale_ts
    repo.get_all_working_orders = AsyncMock(return_value=[order])

    await monitor._scan_stuck_orders()

    assert order.status == expected_status
    repo.save_order_and_event.assert_awaited_once()
    _, event_type, payload = repo.save_order_and_event.await_args.args
    assert event_type == expected_event
    assert payload["reason"] == payload_reason
    if status == OrderStatus.ROUTED:
        router.cancel.assert_awaited_once_with(order)
    else:
        router.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_fill_processor_uses_atomic_fill_helper_only() -> None:
    repo = MagicMock()
    repo.fill_exists = AsyncMock(return_value=False)
    repo.save_fill = AsyncMock()
    repo.save_order_fill_and_event = AsyncMock(return_value=True)
    order = _entry_order("QQQ")
    order.status = OrderStatus.WORKING
    order.remaining_qty = 10
    repo.get_order = AsyncMock(return_value=order)
    processor = FillProcessor(repo)

    await processor.process_fill(
        oms_order_id=order.oms_order_id,
        broker_fill_id="exec-1",
        price=101.25,
        qty=10,
        timestamp=datetime.now(timezone.utc),
        fees=1.5,
    )

    repo.save_fill.assert_not_awaited()
    repo.save_order_fill_and_event.assert_awaited_once()
    saved_order, fill, event_type, payload = repo.save_order_fill_and_event.await_args.args
    assert saved_order.status == OrderStatus.FILLED
    assert fill.broker_fill_id == "exec-1"
    assert event_type == "FILL"
    assert payload["qty"] == 10


@pytest.mark.asyncio
async def test_fill_processor_ignores_duplicate_fill_after_race() -> None:
    repo = MagicMock()
    repo.fill_exists = AsyncMock(return_value=False)
    repo.save_order_fill_and_event = AsyncMock(return_value=False)
    order = _entry_order("SPY")
    order.status = OrderStatus.WORKING
    order.remaining_qty = 10
    repo.get_order = AsyncMock(return_value=order)
    processor = FillProcessor(repo)

    await processor.process_fill(
        oms_order_id=order.oms_order_id,
        broker_fill_id="exec-race",
        price=101.25,
        qty=10,
        timestamp=datetime.now(timezone.utc),
        fees=1.5,
    )

    repo.save_order_fill_and_event.assert_awaited_once()
    assert order.filled_qty == 0.0
    assert order.remaining_qty == 10


@pytest.mark.asyncio
async def test_fill_processor_serializes_distinct_partial_fills_per_order() -> None:
    repo = InMemoryRepository()
    order = _entry_order("MSFT")
    order.status = OrderStatus.WORKING
    order.remaining_qty = 10
    await repo.save_order(order)
    processor = FillProcessor(repo)
    now = datetime.now(timezone.utc)

    await asyncio.gather(
        processor.process_fill(
            oms_order_id=order.oms_order_id,
            broker_fill_id="exec-1",
            price=100.0,
            qty=5,
            timestamp=now,
        ),
        processor.process_fill(
            oms_order_id=order.oms_order_id,
            broker_fill_id="exec-2",
            price=101.0,
            qty=5,
            timestamp=now + timedelta(seconds=1),
        ),
    )

    updated = await repo.get_order(order.oms_order_id)
    assert updated is not None
    assert updated.filled_qty == 10
    assert updated.remaining_qty == 0
    assert updated.status == OrderStatus.FILLED


@pytest.mark.asyncio
async def test_fill_processor_accepts_status_first_terminal_state_without_warning(caplog) -> None:
    repo = InMemoryRepository()
    order = _entry_order("NVDA")
    order.status = OrderStatus.FILLED
    order.remaining_qty = 10
    await repo.save_order(order)
    processor = FillProcessor(repo)
    caplog.set_level(logging.WARNING)

    await processor.process_fill(
        oms_order_id=order.oms_order_id,
        broker_fill_id="exec-status-first",
        price=101.25,
        qty=10,
        timestamp=datetime.now(timezone.utc),
        fees=1.5,
    )

    updated = await repo.get_order(order.oms_order_id)
    assert updated is not None
    assert updated.status == OrderStatus.FILLED
    assert updated.filled_qty == 10
    assert "Invalid transition" not in caplog.text
    assert "Fill transition rejected" not in caplog.text


@pytest.mark.asyncio
async def test_single_oms_callbacks_serialize_status_ack_then_fill(caplog) -> None:
    repo = _DelayedWorkingSaveRepository()
    order = _entry_order("QQQ")
    order.status = OrderStatus.ROUTED
    order.remaining_qty = order.qty
    await repo.save_order(order)
    repo.saved_statuses.clear()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    adapter = SimpleNamespace()
    caplog.set_level(logging.WARNING)

    _wire_adapter_callbacks(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states={},
        portfolio_risk_state=PortfolioRiskState(trade_date=date.today()),
        unit_risk_dollars=100.0,
        open_positions={},
    )

    adapter.on_status(order.oms_order_id, "Submitted", order.qty)
    await asyncio.wait_for(repo.working_save_started.wait(), timeout=1.0)
    adapter.on_ack(order.oms_order_id, SimpleNamespace(broker_order_id=77))
    adapter.on_fill(
        order.oms_order_id,
        "exec-single",
        101.0,
        order.qty,
        datetime.now(timezone.utc),
        0.0,
    )
    repo.release_working_save.set()

    updated = await _wait_for_order_status(repo, order.oms_order_id, OrderStatus.FILLED)

    assert updated.filled_qty == order.qty
    assert updated.remaining_qty == 0
    assert updated.acked_at is not None
    assert updated.broker_order_ref is not None
    assert repo.saved_statuses[0] == OrderStatus.WORKING
    assert repo.saved_statuses[-1] == OrderStatus.FILLED
    assert "Invalid transition" not in caplog.text
    assert "Fill transition rejected" not in caplog.text


@pytest.mark.asyncio
async def test_multi_oms_callbacks_serialize_status_ack_then_fill(caplog) -> None:
    repo = _DelayedWorkingSaveRepository()
    order = _entry_order("GLD")
    order.status = OrderStatus.ROUTED
    order.remaining_qty = order.qty
    await repo.save_order(order)
    repo.saved_statuses.clear()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    adapter = SimpleNamespace()
    coordinator = MagicMock()
    caplog.set_level(logging.WARNING)

    _wire_adapter_callbacks_multi(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states={},
        portfolio_risk_state=PortfolioRiskState(trade_date=date.today()),
        unit_risk_map={order.strategy_id: 100.0},
        open_positions={},
        coordinator=coordinator,
        portfolio_urd=100.0,
    )

    adapter.on_status(order.oms_order_id, "Submitted", order.qty)
    await asyncio.wait_for(repo.working_save_started.wait(), timeout=1.0)
    adapter.on_ack(order.oms_order_id, SimpleNamespace(broker_order_id=88))
    adapter.on_fill(
        order.oms_order_id,
        "exec-multi",
        99.5,
        order.qty,
        datetime.now(timezone.utc),
        0.0,
    )
    repo.release_working_save.set()

    updated = await _wait_for_order_status(repo, order.oms_order_id, OrderStatus.FILLED)

    assert updated.filled_qty == order.qty
    assert updated.remaining_qty == 0
    assert updated.acked_at is not None
    assert updated.broker_order_ref is not None
    assert repo.saved_statuses[0] == OrderStatus.WORKING
    assert repo.saved_statuses[-1] == OrderStatus.FILLED
    assert "Invalid transition" not in caplog.text
    assert "Fill transition rejected" not in caplog.text


@pytest.mark.asyncio
async def test_offline_import_uses_single_oms_fill_side_effects() -> None:
    repo = InMemoryRepository()
    order = _entry_order("QQQ")
    order.status = OrderStatus.WORKING
    order.remaining_qty = order.qty
    order.risk_context = RiskContext(
        stop_for_risk=99.0,
        planned_entry_price=100.0,
        risk_dollars=100.0,
    )
    await repo.save_order(order)
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    adapter = SimpleNamespace()
    strategy_risk_states: dict = {}
    portfolio_risk_state = PortfolioRiskState(trade_date=date.today())

    _wire_adapter_callbacks(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states=strategy_risk_states,
        portfolio_risk_state=portfolio_risk_state,
        unit_risk_dollars=100.0,
        open_positions={},
    )

    exec_report = SimpleNamespace(
        broker_order_id=77,
        perm_id=8801,
        exec_id="exec-offline-single",
        price=101.0,
        qty=order.qty,
        fill_time=datetime.now(timezone.utc),
        commission=1.25,
    )
    cache = IBCache()
    await cache.rebuild_from_broker(
        _ExecutionSnapshot([exec_report]),
        oms_order_id_resolver=AsyncMock(return_value=order.oms_order_id),
        fill_exists_check=repo.fill_exists,
        fill_importer=lambda oms_id, er: _import_fill_through_adapter_callback(
            adapter, repo, oms_id, er,
        ),
    )

    assert cache.is_fill_seen(exec_report.exec_id)
    bus.emit_fill_event.assert_called_once()
    assert strategy_risk_states[order.strategy_id].open_risk_R == pytest.approx(1.0)
    assert portfolio_risk_state.open_risk_R == pytest.approx(1.0)
    positions = await repo.get_all_positions()
    assert len(positions) == 1
    assert positions[0].net_qty == order.qty
    assert positions[0].open_risk_R == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_adapter_importer_reports_duplicate_fill_without_side_effects() -> None:
    repo = InMemoryRepository()
    order = _entry_order("QQQ")
    order.status = OrderStatus.WORKING
    order.remaining_qty = order.qty
    order.risk_context = RiskContext(
        stop_for_risk=99.0,
        planned_entry_price=100.0,
        risk_dollars=100.0,
    )
    await repo.save_order(order)
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    adapter = SimpleNamespace()

    _wire_adapter_callbacks(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states={},
        portfolio_risk_state=PortfolioRiskState(trade_date=date.today()),
        unit_risk_dollars=100.0,
        open_positions={},
    )
    exec_report = SimpleNamespace(
        broker_order_id=77,
        perm_id=8801,
        exec_id="exec-duplicate-import",
        price=101.0,
        qty=order.qty,
        fill_time=datetime.now(timezone.utc),
        commission=0.0,
    )

    first = await _import_fill_through_adapter_callback(
        adapter, repo, order.oms_order_id, exec_report,
    )
    duplicate = await _import_fill_through_adapter_callback(
        adapter, repo, order.oms_order_id, exec_report,
    )

    assert first is True
    assert duplicate is False
    bus.emit_fill_event.assert_called_once()


@pytest.mark.asyncio
async def test_offline_import_uses_multi_oms_fill_side_effects() -> None:
    repo = InMemoryRepository()
    order = _entry_order("GLD")
    order.status = OrderStatus.WORKING
    order.remaining_qty = order.qty
    order.risk_context = RiskContext(
        stop_for_risk=98.0,
        planned_entry_price=100.0,
        risk_dollars=100.0,
    )
    await repo.save_order(order)
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    adapter = SimpleNamespace()
    coordinator = MagicMock()
    strategy_risk_states: dict = {}
    portfolio_risk_state = PortfolioRiskState(trade_date=date.today())

    _wire_adapter_callbacks_multi(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states=strategy_risk_states,
        portfolio_risk_state=portfolio_risk_state,
        unit_risk_map={order.strategy_id: 100.0},
        open_positions={},
        coordinator=coordinator,
        portfolio_urd=100.0,
    )

    exec_report = SimpleNamespace(
        broker_order_id=88,
        perm_id=8802,
        exec_id="exec-offline-multi",
        price=99.5,
        qty=order.qty,
        fill_time=datetime.now(timezone.utc),
        commission=0.0,
    )
    cache = IBCache()
    await cache.rebuild_from_broker(
        _ExecutionSnapshot([exec_report]),
        oms_order_id_resolver=AsyncMock(return_value=order.oms_order_id),
        fill_exists_check=repo.fill_exists,
        fill_importer=lambda oms_id, er: _import_fill_through_adapter_callback(
            adapter, repo, oms_id, er,
        ),
    )

    assert cache.is_fill_seen(exec_report.exec_id)
    bus.emit_fill_event.assert_called_once()
    coordinator.on_fill.assert_called_once()
    coordinator.on_position_update.assert_called_once()
    assert strategy_risk_states[order.strategy_id].open_risk_R == pytest.approx(1.0)
    positions = await repo.get_all_positions()
    assert len(positions) == 1
    assert positions[0].net_qty == order.qty


@pytest.mark.asyncio
async def test_multi_oms_paper_equity_ref_tracks_entry_and_exit_commissions(monkeypatch) -> None:
    repo = InMemoryRepository()
    entry = _entry_order("GLD")
    entry.status = OrderStatus.WORKING
    entry.remaining_qty = entry.qty
    entry.risk_context = RiskContext(
        stop_for_risk=90.0,
        planned_entry_price=100.0,
        risk_dollars=100.0,
    )
    await repo.save_order(entry)

    paper_value = 10_000.0
    calls: list[tuple[float, float, str, float]] = []

    async def fake_apply_paper_pnl(
        pool,
        pnl: float,
        commission: float,
        *,
        account_scope: str,
        initial_equity: float,
    ) -> float:
        nonlocal paper_value
        calls.append((pnl, commission, account_scope, initial_equity))
        paper_value += pnl - commission
        return paper_value

    monkeypatch.setattr(
        "libs.persistence.paper_equity.apply_paper_pnl",
        fake_apply_paper_pnl,
    )

    adapter = SimpleNamespace()
    bus = MagicMock()
    bus.emit_order_event = MagicMock()
    bus.emit_fill_event = MagicMock()
    coordinator = MagicMock()
    live_equity = [paper_value]
    open_positions: dict = {}

    _wire_adapter_callbacks_multi(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states={},
        portfolio_risk_state=PortfolioRiskState(trade_date=date.today()),
        unit_risk_map={entry.strategy_id: 100.0},
        open_positions=open_positions,
        coordinator=coordinator,
        portfolio_urd=100.0,
        live_equity=live_equity,
        paper_equity_pool=object(),
        paper_equity_scope="swing",
        paper_initial_equity=10_000.0,
    )

    task = adapter.on_fill(
        entry.oms_order_id,
        "exec-entry-paper",
        100.0,
        entry.qty,
        datetime.now(timezone.utc),
        1.25,
    )
    await asyncio.wait_for(task, timeout=1.0)
    assert live_equity[0] == pytest.approx(9_998.75)

    exit_order = _entry_order("GLD")
    exit_order.oms_order_id = "oms-gld-exit"
    exit_order.client_order_id = "client-gld-exit"
    exit_order.side = OrderSide.SELL
    exit_order.role = OrderRole.EXIT
    exit_order.status = OrderStatus.WORKING
    exit_order.remaining_qty = exit_order.qty
    exit_order.risk_context = None
    await repo.save_order(exit_order)

    task = adapter.on_fill(
        exit_order.oms_order_id,
        "exec-exit-paper",
        103.0,
        exit_order.qty,
        datetime.now(timezone.utc),
        1.0,
    )
    await asyncio.wait_for(task, timeout=1.0)

    assert live_equity[0] == pytest.approx(10_027.75)
    assert calls == [
        (0.0, 1.25, "swing", 10_000.0),
        (30.0, 1.0, "swing", 10_000.0),
    ]


@pytest.mark.asyncio
async def test_failed_offline_import_leaves_exec_unmarked() -> None:
    exec_report = SimpleNamespace(
        broker_order_id=99,
        perm_id=8803,
        exec_id="exec-failed-import",
        price=100.0,
        qty=1,
        fill_time=datetime.now(timezone.utc),
        commission=0.0,
    )
    cache = IBCache()

    await cache.rebuild_from_broker(
        _ExecutionSnapshot([exec_report]),
        oms_order_id_resolver=AsyncMock(return_value="oms-failed"),
        fill_exists_check=AsyncMock(return_value=False),
        fill_importer=AsyncMock(return_value=False),
    )

    assert not cache.is_fill_seen(exec_report.exec_id)


@pytest.mark.asyncio
async def test_single_oms_unknown_exit_fill_halts_without_flat_position() -> None:
    repo = InMemoryRepository()
    order = _entry_order("QQQ")
    order.role = OrderRole.EXIT
    order.side = OrderSide.SELL
    order.status = OrderStatus.WORKING
    order.remaining_qty = order.qty
    order.risk_context = None
    await repo.save_order(order)
    bus = MagicMock()
    bus.emit_fill_event = MagicMock()
    bus.emit_risk_halt = MagicMock()
    adapter = SimpleNamespace()
    halt_trading = AsyncMock()

    _wire_adapter_callbacks(
        adapter=adapter,
        bus=bus,
        repo=repo,
        fill_proc=FillProcessor(repo),
        router=MagicMock(),
        strategy_risk_states={},
        portfolio_risk_state=PortfolioRiskState(trade_date=date.today()),
        unit_risk_dollars=100.0,
        open_positions={},
        halt_trading=halt_trading,
    )

    adapter.on_fill(
        order.oms_order_id,
        "exec-unknown-exit",
        101.0,
        order.qty,
        datetime.now(timezone.utc),
        0.0,
    )

    updated = await _wait_for_order_status(repo, order.oms_order_id, OrderStatus.FILLED)

    assert updated.filled_qty == order.qty
    assert repo._positions == {}
    bus.emit_fill_event.assert_called_once()
    bus.emit_risk_halt.assert_called_once()
    halt_trading.assert_awaited_once()


@pytest.mark.asyncio
async def test_open_position_hydration_is_strict_for_db_backed_services() -> None:
    with pytest.raises(RuntimeError, match="open_positions hydration failed"):
        await _hydrate_open_positions_from_repo(
            open_positions={},
            repo=_HydrationFailRepository(),
            strategy_ids=["TEST"],
            unit_risk_dollars_for=lambda _sid: 100.0,
            strict=True,
        )


@pytest.mark.asyncio
async def test_save_event_raises_invariant_on_orphan_fk() -> None:
    repo = OMSRepository(_BrokenPool(_BrokenConnection()))
    repo._is_fk_violation = lambda exc: True  # type: ignore[method-assign]

    with pytest.raises(OMSPersistenceInvariantError, match="save_event failed"):
        await repo.save_event("oms-missing", "RISK_APPROVED", {})


@pytest.mark.asyncio
async def test_save_fill_raises_invariant_on_orphan_fk() -> None:
    repo = OMSRepository(_BrokenPool(_BrokenConnection()))
    repo._is_fk_violation = lambda exc: True  # type: ignore[method-assign]

    with pytest.raises(OMSPersistenceInvariantError, match="save_fill failed"):
        await repo.save_fill(
            fill=MagicMock(
                oms_order_id="oms-missing",
                fill_id="fill-1",
                broker_fill_id="exec-1",
                price=1.0,
                qty=1.0,
                timestamp=datetime.now(timezone.utc),
                fees=0.0,
            )
        )
