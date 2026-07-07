from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from libs.broker_ibkr.models.types import BrokerOrderStatus, OrderStatusEvent
from libs.broker_ibkr.reconciler.discrepancy_policy import (
    DiscrepancyAction,
    DiscrepancyPolicy,
)
from libs.broker_ibkr.reconciler.sync import Discrepancy, ReconcilerSync
from libs.broker_ibkr.state.cache import IBCache
from libs.oms.models.order import OMSOrder, OrderStatus
from libs.oms.reconciliation.authority import (
    InMemoryReconciliationAuthority,
    ReconciliationAuthorityScope,
)
from libs.oms.reconciliation.orchestrator import ReconciliationOrchestrator


def _broker_order(
    order_ref: str,
    broker_order_id: int = 101,
    *,
    account: str = "",
    client_id: int | None = None,
) -> OrderStatusEvent:
    return OrderStatusEvent(
        broker_order_id=broker_order_id,
        perm_id=9001,
        status=BrokerOrderStatus.SUBMITTED,
        filled_qty=0.0,
        remaining_qty=1.0,
        avg_fill_price=0.0,
        order_ref=order_ref,
        account=account,
        client_id=client_id,
    )


class _Repo:
    def __init__(self, order: OMSOrder | None = None) -> None:
        self.order = order
        self.saved: list[OMSOrder] = []

    async def get_order(self, oms_order_id: str) -> OMSOrder | None:
        if self.order and self.order.oms_order_id == oms_order_id:
            return self.order
        return None

    async def save_order(self, order: OMSOrder) -> None:
        self.saved.append(order)


class _Bus:
    def __init__(self) -> None:
        self.order_events: list[OMSOrder] = []
        self.risk_halts: list[str] = []

    def emit_order_event(self, order: OMSOrder) -> None:
        self.order_events.append(order)

    def emit_risk_halt(self, strategy_id: str, reason: str) -> None:
        self.risk_halts.append(reason)


class _Adapter:
    def __init__(self) -> None:
        self.cache = IBCache()
        self.cancelled: list[tuple[int, int]] = []

    async def cancel_order(self, broker_order_id: int, perm_id: int = 0) -> None:
        self.cancelled.append((broker_order_id, perm_id))


def test_blank_ref_broker_orphan_halts_by_default() -> None:
    sync = ReconcilerSync(DiscrepancyPolicy())

    discrepancies = sync.reconcile_orders(
        [_broker_order("")],
        oms_working_ids=set(),
        known_order_refs={},
    )

    assert len(discrepancies) == 1
    assert discrepancies[0].action == DiscrepancyAction.HALT_AND_ALERT
    assert discrepancies[0].details["account"] == ""
    assert discrepancies[0].details["client_id"] is None


def test_blank_ref_broker_orphan_can_be_cancellable_with_explicit_policy() -> None:
    sync = ReconcilerSync(
        DiscrepancyPolicy(unknown_order_orphan=DiscrepancyAction.CANCEL)
    )

    discrepancies = sync.reconcile_orders(
        [_broker_order("")],
        oms_working_ids=set(),
        known_order_refs={},
    )

    assert len(discrepancies) == 1
    assert discrepancies[0].action == DiscrepancyAction.CANCEL


def test_reconcile_orders_halts_foreign_account_before_orphan_policy() -> None:
    sync = ReconcilerSync(
        DiscrepancyPolicy(unknown_order_orphan=DiscrepancyAction.CANCEL)
    )

    discrepancies = sync.reconcile_orders(
        [_broker_order("", account="U999", client_id=7)],
        oms_working_ids=set(),
        known_order_refs={},
        managed_account_id="U123",
        managed_client_id=7,
    )

    assert len(discrepancies) == 1
    assert discrepancies[0].type == "foreign_order"
    assert discrepancies[0].action == DiscrepancyAction.HALT_AND_ALERT
    assert discrepancies[0].details["scope_mismatch"] == "account"
    assert discrepancies[0].details["account"] == "U999"


def test_reconcile_orders_halts_foreign_client_before_orphan_policy() -> None:
    sync = ReconcilerSync(
        DiscrepancyPolicy(unknown_order_orphan=DiscrepancyAction.CANCEL)
    )

    discrepancies = sync.reconcile_orders(
        [_broker_order("", account="U123", client_id=99)],
        oms_working_ids=set(),
        known_order_refs={},
        managed_account_id="U123",
        managed_client_id=7,
    )

    assert len(discrepancies) == 1
    assert discrepancies[0].type == "foreign_order"
    assert discrepancies[0].action == DiscrepancyAction.HALT_AND_ALERT
    assert discrepancies[0].details["scope_mismatch"] == "client_id"
    assert discrepancies[0].details["client_id"] == 99


def test_reconcile_orders_halts_non_owned_order_ref_prefix() -> None:
    sync = ReconcilerSync(DiscrepancyPolicy())

    discrepancies = sync.reconcile_orders(
        [_broker_order("MANUAL-123", account="U123", client_id=7)],
        oms_working_ids=set(),
        known_order_refs={},
        managed_account_id="U123",
        managed_client_id=7,
        owned_order_ref_prefixes=("OMS-",),
    )

    assert len(discrepancies) == 1
    assert discrepancies[0].type == "foreign_order"
    assert discrepancies[0].action == DiscrepancyAction.HALT_AND_ALERT
    assert discrepancies[0].details["scope_mismatch"] == "order_ref_prefix"
    assert discrepancies[0].details["order_ref"] == "MANUAL-123"


def test_orchestrator_reads_owned_order_ref_prefixes(monkeypatch) -> None:
    monkeypatch.setenv("OMS_OWNED_ORDER_REF_PREFIXES", "OMS-, LIVE- ,")

    assert ReconciliationOrchestrator._owned_order_ref_prefixes() == ("OMS-", "LIVE-")


def test_reconcile_orders_normalizes_broker_id_inputs() -> None:
    sync = ReconcilerSync(DiscrepancyPolicy())

    discrepancies = sync.reconcile_orders(
        [_broker_order("", broker_order_id="101")],  # type: ignore[arg-type]
        oms_working_ids={"101"},  # type: ignore[arg-type]
        known_order_refs={},
    )

    assert discrepancies == []


@pytest.mark.asyncio
async def test_known_order_ref_repairs_mapping_without_halt() -> None:
    order = OMSOrder(
        oms_order_id="oms-1",
        client_order_id="client-1",
        strategy_id="S1",
        status=OrderStatus.RISK_APPROVED,
        qty=1,
    )
    repo = _Repo(order)
    bus = _Bus()
    adapter = _Adapter()
    halted: list[str] = []
    async def _halt(reason: str) -> None:
        halted.append(reason)
    orchestrator = ReconciliationOrchestrator(
        adapter,
        repo,
        bus,
        halt_trading=_halt,
    )
    discrepancy = Discrepancy(
        "repair_order_mapping",
        DiscrepancyAction.REPAIR_MAPPING,
        {"order": _broker_order("client-1"), "order_ref": "client-1", "oms_order_id": "oms-1"},
    )

    await orchestrator._apply_discrepancies([discrepancy])

    assert halted == []
    assert bus.risk_halts == []
    assert order.broker_order_id == 101
    assert order.perm_id == 9001
    assert order.status == OrderStatus.WORKING
    assert adapter.cache.lookup_oms_id(101) == "oms-1"
    assert repo.saved == [order]
    assert bus.order_events == [order]


@pytest.mark.asyncio
async def test_unknown_nonblank_order_ref_halts() -> None:
    sync = ReconcilerSync(DiscrepancyPolicy())
    discrepancies = sync.reconcile_orders(
        [_broker_order("mystery-ref")],
        oms_working_ids=set(),
        known_order_refs={},
    )
    bus = _Bus()
    halted: list[str] = []
    async def _halt(reason: str) -> None:
        halted.append(reason)
    orchestrator = ReconciliationOrchestrator(
        _Adapter(),
        _Repo(),
        bus,
        halt_trading=_halt,
    )

    await orchestrator._apply_discrepancies(discrepancies)

    assert discrepancies[0].action == DiscrepancyAction.IMPORT
    assert halted
    assert bus.risk_halts


@pytest.mark.asyncio
async def test_position_quantity_mismatch_halts() -> None:
    bus = _Bus()
    halted: list[str] = []
    async def _halt(reason: str) -> None:
        halted.append(reason)
    orchestrator = ReconciliationOrchestrator(
        _Adapter(),
        _Repo(),
        bus,
        halt_trading=_halt,
    )
    discrepancy = Discrepancy(
        "position_mismatch",
        DiscrepancyAction.ADJUST_POSITION,
        {"symbol": "MNQ", "broker_qty": 2.0, "oms_qty": 1.0},
    )

    await orchestrator._apply_discrepancies([discrepancy])

    assert halted
    assert "position mismatch" in halted[0].lower()
    assert bus.risk_halts == halted


@pytest.mark.asyncio
async def test_non_authoritative_orchestrator_skips_mutating_actions() -> None:
    bus = _Bus()
    halted: list[str] = []

    async def _halt(reason: str) -> None:
        halted.append(reason)

    adapter = _Adapter()
    orchestrator = ReconciliationOrchestrator(
        adapter,
        _Repo(),
        bus,
        halt_trading=_halt,
        family_id="momentum",
        owner_id="momentum:NQ_REGIME",
        reconciliation_authoritative=False,
    )
    discrepancy = Discrepancy(
        "unknown_broker_order",
        DiscrepancyAction.CANCEL,
        {"order": _broker_order("mystery-ref")},
    )

    await orchestrator._apply_discrepancies([discrepancy])

    assert adapter.cancelled == []
    assert halted == []
    assert bus.risk_halts == []


@pytest.mark.asyncio
async def test_authority_lease_unavailable_skips_mutating_actions() -> None:
    authority = InMemoryReconciliationAuthority()
    scope = ReconciliationAuthorityScope(
        broker="IBKR",
        account_id="unknown",
        client_id=-1,
        family_id="momentum",
        recon_kind="manual",
    )
    owner_a = await authority.acquire(scope, "owner-a", ttl_seconds=60)
    assert owner_a is not None

    adapter = _Adapter()
    orchestrator = ReconciliationOrchestrator(
        adapter,
        _Repo(),
        _Bus(),
        family_id="momentum",
        owner_id="owner-b",
        reconciliation_authoritative=True,
        authority=authority,
    )
    discrepancy = Discrepancy(
        "unknown_broker_order",
        DiscrepancyAction.CANCEL,
        {"order": _broker_order("mystery-ref")},
    )

    await orchestrator._apply_discrepancies([discrepancy])

    assert adapter.cancelled == []


@pytest.mark.asyncio
async def test_in_memory_authority_recovers_stale_lease() -> None:
    authority = InMemoryReconciliationAuthority()
    scope = ReconciliationAuthorityScope(
        broker="IBKR",
        account_id="DU123",
        client_id=7,
        family_id="stock",
        recon_kind="periodic",
    )

    owner_a = await authority.acquire(scope, "owner-a", ttl_seconds=0.01)
    owner_b_initial = await authority.acquire(scope, "owner-b", ttl_seconds=60)
    await asyncio.sleep(0.02)
    owner_b_recovered = await authority.acquire(scope, "owner-b", ttl_seconds=60)

    assert owner_a is not None
    assert owner_b_initial is None
    assert owner_b_recovered is not None
    assert owner_b_recovered.owner_id == "owner-b"
    assert authority.is_authoritative(owner_b_recovered)


def test_reconciler_authority_wiring_is_present_in_factories_and_coordinators() -> None:
    factory = Path("libs/oms/services/factory.py").read_text(encoding="utf-8")
    service = Path("libs/oms/services/oms_service.py").read_text(encoding="utf-8")
    orchestrator = Path("libs/oms/reconciliation/orchestrator.py").read_text(encoding="utf-8")
    momentum = Path("strategies/momentum/coordinator.py").read_text(encoding="utf-8")
    stock = Path("strategies/stock/coordinator.py").read_text(encoding="utf-8")

    assert "reconciliation_authoritative" in factory
    assert "ReconciliationAuthority" in factory
    assert "OMS_RECONCILIATION_AUTHORITY_LEASES" in factory
    assert "enable_reconciliation_authority_lease" in factory
    assert "authority=_reconciliation_authority" in factory
    assert "reconciliation_authoritative" in orchestrator
    assert "is_authoritative" in service
    assert "reconciliation_authoritative = sid == authoritative_strategy_id" in momentum
    assert "reconciliation_authoritative = sid == authoritative_strategy_id" in stock
    assert "getattr(reconciler, \"is_authoritative\", True)" in momentum
    assert "getattr(reconciler, \"is_authoritative\", True)" in stock
