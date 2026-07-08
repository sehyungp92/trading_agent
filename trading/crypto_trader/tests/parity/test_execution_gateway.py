"""Tests for broker-compatible execution gateway parity capture."""

from datetime import datetime, timezone

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.broker.sim_execution_adapter import SimExecutionAdapter
from crypto_trader.core.events import CanonicalRuntimeEvent, EventBus
from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, OrderType, Side, TimeFrame
from crypto_trader.core.runtime_types import DecisionContext, ExecutionReport, ExecutionReportKind
from crypto_trader.live.oms_store import OmsStore


def test_execution_gateway_emits_intent_and_report_without_changing_broker_api() -> None:
    broker = SimBroker(initial_equity=10_000.0)
    events = EventBus()
    canonical = []
    events.subscribe(CanonicalRuntimeEvent, canonical.append)
    gateway = ExecutionGateway(
        adapter=SimExecutionAdapter(broker),
        broker=broker,
        events=events,
    )
    context = DecisionContext(
        decision_id="d1",
        strategy_id="momentum",
        symbol="BTC",
        timeframe=TimeFrame.M15,
        decision_time=datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc),
        decision_key="momentum|BTC|15m|2026-05-24T12:15:00+00:00",
    )
    gateway.begin_decision_context(context)

    visible_id = gateway.submit_order(Order(
        order_id="",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.LIMIT,
        qty=0.1,
        limit_price=100.0,
        tag="entry",
        metadata={"strategy_id": "momentum"},
    ))

    assert visible_id
    assert context.action == "order"
    assert [event.stream for event in canonical] == ["order_intent", "execution"]
    assert canonical[0].payload["decision_id"] == "d1"
    assert broker.get_open_orders()[0].metadata["client_order_id"] == visible_id


def test_execution_gateway_cancels_strategy_visible_sim_id() -> None:
    broker = SimBroker(initial_equity=10_000.0)
    gateway = ExecutionGateway(
        adapter=SimExecutionAdapter(broker),
        broker=broker,
    )
    visible_id = gateway.submit_order(Order(
        order_id="strategy_limit_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.LIMIT,
        qty=0.1,
        limit_price=100.0,
        tag="entry",
        metadata={"strategy_id": "momentum"},
    ))

    assert visible_id == "strategy_limit_1"
    assert broker.get_open_orders()[0].order_id == "1"
    assert gateway.cancel_order(visible_id) is True
    assert broker.get_open_orders() == []


def test_execution_gateway_persists_order_role_to_oms(tmp_path) -> None:
    broker = SimBroker(initial_equity=10_000.0)
    oms = OmsStore(tmp_path)
    gateway = ExecutionGateway(
        adapter=SimExecutionAdapter(broker),
        broker=broker,
        oms_store=oms,
    )

    visible_id = gateway.submit_order(Order(
        order_id="strategy_stop_1",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.1,
        stop_price=90.0,
        tag="protective_stop",
        metadata={"strategy_id": "momentum"},
    ))
    row = oms.get_order(visible_id)
    oms.close()

    assert row is not None
    assert row["role"] == "protective_stop"
    assert row["order_type"] == OrderType.STOP.value
    assert row["exchange_order_id"] == "1"


class _TtlStateAdapter:
    def __init__(self, *, reports=None) -> None:
        self.state = {
            "client_order_id": "ttl_1",
            "exchange_order_id": "101",
            "strategy_id": "trend",
            "symbol": "BTC",
            "side": Side.LONG.value,
            "order_type": OrderType.STOP.value,
            "qty": 0.1,
            "remaining_qty": 0.1,
            "ttl_bars": 2,
            "ttl_bars_alive": 0,
            "metadata": {
                "strategy_id": "trend",
                "order_type": OrderType.STOP.value,
                "tag": "entry",
                "ttl_bars": 2,
            },
        }
        self._reports = reports or []

    def submit(self, intent):
        return []

    def cancel(self, client_order_id):
        return []

    def sync_open_orders(self):
        return []

    def sync_positions(self):
        return []

    def sync_fills(self, watermark):
        return []

    def expire_ttl_orders_for_bar(self, bar):
        self.state["ttl_bars_alive"] = 1
        self.state["metadata"]["ttl_bars_alive"] = 1
        return list(self._reports)

    def active_ttl_orders(self):
        return [self.state]


class _TtlPartialFillAdapter(_TtlStateAdapter):
    def clear_ttl_for_fill(self, fill):
        self.state["remaining_qty"] = 0.6
        self.state["metadata"]["ttl_remaining_qty"] = 0.6
        return True


class _TtlFullFillAdapter(_TtlStateAdapter):
    def clear_ttl_for_fill(self, fill):
        return True

    def active_ttl_orders(self):
        return []


def test_execution_gateway_persists_ttl_age_without_noisy_report(tmp_path) -> None:
    oms = OmsStore(tmp_path)
    oms.upsert_order(
        client_order_id="ttl_1",
        exchange_order_id="101",
        strategy_id="trend",
        symbol="BTC",
        side=Side.LONG.value,
        order_type=OrderType.STOP.value,
        status=OrderStatus.WORKING.value,
        role="entry",
        metadata={"ttl_bars": 2},
    )
    gateway = ExecutionGateway(
        adapter=_TtlStateAdapter(),
        broker=SimBroker(initial_equity=10_000.0),
        oms_store=oms,
    )

    reports = gateway.expire_ttl_orders_for_bar(_bar())
    row = oms.get_order("ttl_1")
    oms.close()

    assert reports == []
    assert row["status"] == OrderStatus.WORKING.value
    assert row["metadata"]["ttl_bars_alive"] == 1
    assert row["metadata"]["ttl_tracking_active"] is True


def test_execution_gateway_persists_ttl_remaining_qty_after_partial_fill(tmp_path) -> None:
    oms = _oms_with_ttl_order(tmp_path)
    gateway = ExecutionGateway(
        adapter=_TtlPartialFillAdapter(),
        broker=SimBroker(initial_equity=10_000.0),
        oms_store=oms,
    )

    assert gateway.clear_ttl_for_fill(_fill(qty=0.4)) is True
    row = oms.get_order("ttl_1")
    oms.close()

    assert row["status"] == OrderStatus.WORKING.value
    assert row["metadata"]["ttl_remaining_qty"] == 0.6
    assert row["metadata"]["ttl_tracking_active"] is True


def test_execution_gateway_marks_ttl_tracking_inactive_after_terminal_fill(tmp_path) -> None:
    oms = _oms_with_ttl_order(tmp_path)
    gateway = ExecutionGateway(
        adapter=_TtlFullFillAdapter(),
        broker=SimBroker(initial_equity=10_000.0),
        oms_store=oms,
    )

    assert gateway.clear_ttl_for_fill(_fill(qty=0.1)) is True
    row = oms.get_order("ttl_1")
    oms.close()

    assert row["status"] == OrderStatus.FILLED.value
    assert row["metadata"]["ttl_remaining_qty"] == 0.0
    assert row["metadata"]["ttl_tracking_active"] is False


def test_execution_gateway_ttl_cancel_failure_does_not_reject_oms_order(tmp_path) -> None:
    report = ExecutionReport(
        report_id="ttl_cancel_failed_1",
        kind=ExecutionReportKind.RESTING,
        timestamp=datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc),
        symbol="BTC",
        side=Side.LONG,
        client_order_id="ttl_1",
        exchange_order_id="101",
        order_status=OrderStatus.WORKING,
        qty=0.1,
        reject_reason="ttl_cancel_rejected",
        metadata={
            "strategy_id": "trend",
            "order_type": OrderType.STOP.value,
            "tag": "entry",
            "ttl_bars": 0,
            "ttl_bars_alive": 1,
            "ttl_cancel_failed": True,
        },
    )
    oms = OmsStore(tmp_path)
    oms.upsert_order(
        client_order_id="ttl_1",
        exchange_order_id="101",
        strategy_id="trend",
        symbol="BTC",
        side=Side.LONG.value,
        order_type=OrderType.STOP.value,
        status=OrderStatus.WORKING.value,
        role="entry",
        metadata={"ttl_bars": 0},
    )
    gateway = ExecutionGateway(
        adapter=_TtlStateAdapter(reports=[report]),
        broker=SimBroker(initial_equity=10_000.0),
        oms_store=oms,
    )

    gateway.expire_ttl_orders_for_bar(_bar())
    row = oms.get_order("ttl_1")
    discrepancies = oms.list_unresolved_discrepancies()
    oms.close()

    assert row["status"] == OrderStatus.WORKING.value
    assert discrepancies[0]["kind"] == "ttl_cancel_failed"
    assert discrepancies[0]["severity"] == "warning"


def _bar() -> Bar:
    return Bar(
        timestamp=datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc),
        symbol="BTC",
        open=99.0,
        high=99.5,
        low=98.5,
        close=99.0,
        volume=1.0,
        timeframe=TimeFrame.M15,
    )


def _fill(*, qty: float) -> Fill:
    return Fill(
        order_id="ttl_1",
        exchange_order_id="101",
        symbol="BTC",
        side=Side.LONG,
        qty=qty,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 16, tzinfo=timezone.utc),
        tag="entry",
    )


def _oms_with_ttl_order(tmp_path) -> OmsStore:
    oms = OmsStore(tmp_path)
    oms.upsert_order(
        client_order_id="ttl_1",
        exchange_order_id="101",
        strategy_id="trend",
        symbol="BTC",
        side=Side.LONG.value,
        order_type=OrderType.STOP.value,
        status=OrderStatus.WORKING.value,
        role="entry",
        metadata={"ttl_bars": 2},
    )
    return oms
