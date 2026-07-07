"""Tests for durable live OMS storage."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crypto_trader.core.execution_gateway import ExecutionGateway
from crypto_trader.core.models import Bar, OrderStatus, OrderType, Side, TimeFrame
from crypto_trader.core.runtime_types import ExecutionReport, ExecutionReportKind
from crypto_trader.live.config import LiveConfig
from crypto_trader.live.engine import LiveEngine
from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter
from crypto_trader.live.lifecycle import LivePositionLedgerEntry
from crypto_trader.live.oms_store import (
    FILL_COORDINATOR_APPLIED_STATUSES,
    FILL_LIFECYCLE_APPLIED_STATUSES,
    FILL_STRATEGY_DISPATCHED_STATUSES,
    FILL_STATUS_COORDINATOR_APPLIED,
    FILL_STATUS_FINALIZED,
    FILL_STATUS_LIFECYCLE_APPLIED,
    FILL_STATUS_PROCESSING_FAILED,
    FILL_STATUS_PROCESSED,
    FILL_STATUS_RECEIVED,
    FILL_STATUS_STRATEGY_DISPATCHED,
    FILL_STATUS_UNRESOLVED,
    OmsStore,
)
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState


def test_oms_store_persists_order_lookup_and_watermark(tmp_path) -> None:
    store = OmsStore(tmp_path)
    store.upsert_order(
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        order_type="MARKET",
        status="WORKING",
        metadata={"risk_R": 1.0},
    )
    store.set_watermark("fills_since", "2026-05-24T12:00:00+00:00")
    store.close()

    reopened = OmsStore(tmp_path)
    by_client = reopened.get_order("client_1")
    by_exchange = reopened.get_order("123")

    assert by_client == by_exchange
    assert by_client["strategy_id"] == "momentum"
    assert by_client["metadata"]["risk_R"] == 1.0
    assert reopened.get_watermark("fills_since") == "2026-05-24T12:00:00+00:00"
    reopened.close()


def test_oms_store_updates_order_metadata_without_replacing_order_fields(tmp_path) -> None:
    store = OmsStore(tmp_path)
    store.upsert_order(
        client_order_id="ttl_1",
        exchange_order_id="101",
        strategy_id="trend",
        symbol="BTC",
        side=Side.LONG.value,
        order_type="STOP",
        status=OrderStatus.WORKING.value,
        role="entry",
        metadata={"ttl_bars": 2, "strategy_id": "trend"},
    )

    updated = store.update_order_metadata(
        "101",
        metadata_updates={"ttl_bars_alive": 1, "ttl_tracking_active": True},
    )
    row = store.get_order("ttl_1")
    store.close()

    assert updated is True
    assert row["strategy_id"] == "trend"
    assert row["status"] == OrderStatus.WORKING.value
    assert row["metadata"] == {
        "strategy_id": "trend",
        "ttl_bars": 2,
        "ttl_bars_alive": 1,
        "ttl_tracking_active": True,
    }


def test_oms_store_records_fills_idempotently(tmp_path) -> None:
    store = OmsStore(tmp_path)
    fill = dict(
        fill_id="fill_1",
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        qty=0.01,
        price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )

    assert store.has_fill("fill_1") is False
    assert store.record_fill(**fill) is True
    assert store.has_fill("fill_1") is True
    assert store.record_fill(**fill) is False
    store.close()


def test_oms_store_tracks_fill_lifecycle_states(tmp_path) -> None:
    store = OmsStore(tmp_path)
    fill = dict(
        fill_id="fill_1",
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        qty=0.01,
        price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
        exchange_fill_id="fill_1",
    )

    store.record_received_fill(**fill)
    assert store.get_fill_status("fill_1") == FILL_STATUS_RECEIVED
    assert store.is_fill_processed("fill_1") is False

    store.mark_fill_unresolved("fill_1", strategy_id="momentum", reason="missing_slot")
    assert store.get_fill_status("fill_1") == FILL_STATUS_UNRESOLVED

    store.record_received_fill(**fill)
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="boom")
    assert store.get_fill_status("fill_1") == FILL_STATUS_PROCESSING_FAILED
    assert store.get_fill("fill_1")["processing_error"] == "boom"

    store.record_received_fill(**fill)
    store.mark_fill_dispatched("fill_1", strategy_id="momentum")
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="ledger down")
    assert store.get_fill_status("fill_1") == FILL_STATUS_STRATEGY_DISPATCHED
    assert store.get_fill("fill_1")["processing_error"] == "ledger down"

    store.record_received_fill(**fill)
    assert store.get_fill_status("fill_1") == FILL_STATUS_STRATEGY_DISPATCHED

    store.mark_fill_processed("fill_1", strategy_id="momentum")
    assert store.get_fill_status("fill_1") == FILL_STATUS_PROCESSED
    assert store.is_fill_processed("fill_1") is True
    store.close()


def test_oms_store_fill_status_updates_are_monotonic(tmp_path) -> None:
    store = OmsStore(tmp_path)
    fill = dict(
        fill_id="fill_1",
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        qty=0.01,
        price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
        exchange_fill_id="fill_1",
    )
    store.record_received_fill(**fill)

    store.mark_fill_strategy_dispatched("fill_1", strategy_id="momentum")
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="boom")
    assert store.get_fill_status("fill_1") == FILL_STATUS_STRATEGY_DISPATCHED

    store.mark_fill_coordinator_applied("fill_1", strategy_id="momentum")
    store.record_received_fill(**fill)
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="boom")
    assert store.get_fill_status("fill_1") == FILL_STATUS_COORDINATOR_APPLIED

    store.mark_fill_lifecycle_applied("fill_1", strategy_id="momentum")
    store.record_received_fill(**fill)
    store.mark_fill_unresolved("fill_1", strategy_id="momentum", reason="missing")
    assert store.get_fill_status("fill_1") == FILL_STATUS_LIFECYCLE_APPLIED

    store.mark_fill_finalized("fill_1", strategy_id="momentum")
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="boom")
    assert store.get_fill_status("fill_1") == FILL_STATUS_FINALIZED

    store.mark_fill_processed("fill_1", strategy_id="momentum")
    store.record_received_fill(**fill)
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="boom")
    assert store.get_fill_status("fill_1") == FILL_STATUS_PROCESSED
    assert store.is_fill_processed("fill_1") is True
    store.close()


def test_oms_store_received_retry_can_reset_only_pre_side_effect_errors(tmp_path) -> None:
    store = OmsStore(tmp_path)
    fill = dict(
        fill_id="fill_1",
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        qty=0.01,
        price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
        exchange_fill_id="fill_1",
    )

    store.record_received_fill(**fill)
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="boom")
    store.record_received_fill(**fill)
    assert store.get_fill_status("fill_1") == FILL_STATUS_RECEIVED
    assert store.get_fill("fill_1")["processing_error"] == ""

    store.mark_fill_unresolved("fill_1", strategy_id="momentum", reason="missing")
    store.record_received_fill(**fill)
    assert store.get_fill_status("fill_1") == FILL_STATUS_RECEIVED
    store.close()


def test_oms_store_legacy_dispatched_is_strategy_dispatched_phase(tmp_path) -> None:
    store = OmsStore(tmp_path)
    fill = dict(
        fill_id="fill_1",
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        qty=0.01,
        price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
        exchange_fill_id="fill_1",
    )
    store.record_received_fill(**fill)
    store._conn.execute("UPDATE fills SET status='DISPATCHED' WHERE fill_id='fill_1'")
    store._conn.commit()

    assert "DISPATCHED" in FILL_STRATEGY_DISPATCHED_STATUSES
    assert "DISPATCHED" not in FILL_COORDINATOR_APPLIED_STATUSES
    assert "DISPATCHED" not in FILL_LIFECYCLE_APPLIED_STATUSES
    assert store.is_fill_processed("fill_1") is False
    store.record_received_fill(**fill)
    assert store.get_fill_status("fill_1") == "DISPATCHED"
    store.mark_fill_processing_failed("fill_1", strategy_id="momentum", error="boom")
    assert store.get_fill_status("fill_1") == "DISPATCHED"
    store.mark_fill_coordinator_applied("fill_1", strategy_id="momentum")
    assert store.get_fill_status("fill_1") == FILL_STATUS_COORDINATOR_APPLIED
    store.close()


def test_oms_store_persists_lifecycle_phase_atomically(tmp_path) -> None:
    store = OmsStore(tmp_path)
    fill = dict(
        fill_id="fill_1",
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        qty=0.01,
        price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
        exchange_fill_id="fill_1",
    )
    entry = LivePositionLedgerEntry(
        strategy_id="momentum",
        symbol="BTC",
        direction=Side.LONG,
        position_instance_id="p1",
        qty=1.0,
        avg_entry=100.0,
        entry_time=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    store.record_received_fill(**fill)
    store.mark_fill_coordinator_applied("fill_1", strategy_id="momentum")

    store.persist_lifecycle_phase("fill_1", [entry], strategy_id="momentum")

    assert store.get_fill_status("fill_1") == FILL_STATUS_LIFECYCLE_APPLIED
    assert len(store.list_lifecycle_entries()) == 1

    with pytest.raises(KeyError):
        store.persist_lifecycle_phase("missing_fill", [], strategy_id="momentum")

    assert store.get_fill_status("fill_1") == FILL_STATUS_LIFECYCLE_APPLIED
    assert len(store.list_lifecycle_entries()) == 1
    store.close()


def test_oms_store_v2_records_snapshots_reports_events_and_discrepancies(tmp_path) -> None:
    store = OmsStore(tmp_path)
    store.upsert_strategy_snapshot("momentum", {"position_meta": {"BTC": {"entry_price": 100}}})
    store.record_execution_report(ExecutionReport(
        report_id="r1",
        kind=ExecutionReportKind.ACCEPTED,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
        symbol="BTC",
        side=Side.LONG,
        client_order_id="c1",
        exchange_order_id="123",
        order_status=OrderStatus.WORKING,
    ))
    store.append_event("decision", datetime(2026, 5, 24, tzinfo=timezone.utc), {"decision_id": "d1"})
    store.record_discrepancy(kind="missing_position", description="missing BTC", symbol="BTC")

    assert store.get_strategy_snapshot("momentum")["position_meta"]["BTC"]["entry_price"] == 100
    assert store.list_events("decision")[0]["payload"]["decision_id"] == "d1"
    assert store.list_unresolved_discrepancies()[0]["kind"] == "missing_position"
    store.close()


def test_oms_store_replaces_lifecycle_entries(tmp_path) -> None:
    store = OmsStore(tmp_path)
    entry = LivePositionLedgerEntry(
        strategy_id="momentum",
        symbol="BTC",
        direction=Side.LONG,
        position_instance_id="p1",
        qty=1.0,
        avg_entry=100.0,
        entry_time=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )

    store.replace_lifecycle_entries([entry])
    assert len(store.list_lifecycle_entries()) == 1
    store.replace_lifecycle_entries([])
    assert store.list_lifecycle_entries() == []
    store.close()


def test_live_engine_rehydrates_broker_maps_from_oms(tmp_path) -> None:
    engine = LiveEngine(LiveConfig(
        wallet_address="0xabc",
        private_key=None,
        state_dir=tmp_path,
    ))
    broker = SimpleNamespace(_orders={}, _local_to_oid={}, _oid_map={})
    manager = PortfolioManager(
        PortfolioConfig(),
        PortfolioState(equity=10_000.0, peak_equity=10_000.0),
    )
    engine._broker = broker
    engine._coordinator = StrategyCoordinator(broker, manager)
    engine._oms.upsert_order(
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="momentum",
        symbol="BTC",
        side="LONG",
        order_type="STOP",
        status="WORKING",
        role="protective_stop",
        metadata={"tag": "protective_stop", "ttl_bars": 2, "ttl_bars_alive": 1},
    )

    engine._rehydrate_oms_orders()

    assert broker._local_to_oid["client_1"] == "123"
    assert broker._oid_map["123"] == "client_1"
    assert broker._orders["client_1"].tag == "protective_stop"
    assert broker._orders["client_1"].ttl_bars == 2
    assert broker._orders["client_1"]._bars_alive == 1
    assert engine._coordinator.get_strategy_for_order("123") == "momentum"
    engine._oms.close()


def test_live_engine_seeds_rehydrated_ttl_with_remaining_budget(tmp_path) -> None:
    engine = LiveEngine(LiveConfig(
        wallet_address="0xabc",
        private_key=None,
        state_dir=tmp_path,
    ))
    broker = MagicMock()
    broker._orders = {}
    broker._local_to_oid = {}
    broker._oid_map = {}
    broker.cancel_order.return_value = True
    manager = PortfolioManager(
        PortfolioConfig(),
        PortfolioState(equity=10_000.0, peak_equity=10_000.0),
    )
    engine._broker = broker
    engine._coordinator = StrategyCoordinator(broker, manager)
    engine._oms.upsert_order(
        client_order_id="client_1",
        exchange_order_id="123",
        strategy_id="trend",
        symbol="BTC",
        side=Side.LONG.value,
        order_type=OrderType.STOP.value,
        status=OrderStatus.WORKING.value,
        role="entry",
        metadata={
            "tag": "entry",
            "strategy_id": "trend",
            "order_type": OrderType.STOP.value,
            "ttl_bars": 2,
            "ttl_bars_alive": 1,
        },
    )
    engine._rehydrate_oms_orders()
    broker.get_open_orders.return_value = [broker._orders["client_1"]]

    gateway = ExecutionGateway(
        adapter=HyperliquidExecutionAdapter(broker, strategy_id="trend"),
        broker=broker,
        oms_store=engine._oms,
    )
    engine._slots = [SimpleNamespace(ctx=SimpleNamespace(broker=gateway))]
    engine._seed_ttl_trackers_from_open_orders()

    expired = gateway.expire_ttl_orders_for_bar(_ttl_bar())
    row = engine._oms.get_order("client_1")

    assert row["metadata"]["ttl_bars_alive"] == 2
    assert expired[0].order_status == OrderStatus.EXPIRED
    broker.cancel_order.assert_called_once_with("client_1")
    engine._oms.close()


def test_live_engine_shutdown_persists_before_closing_oms(tmp_path) -> None:
    engine = LiveEngine(LiveConfig(state_dir=tmp_path))

    asyncio.run(engine.shutdown())

    reopened = OmsStore(tmp_path)
    assert reopened.list_lifecycle_entries() == []
    reopened.close()


def test_admin_allocation_correction_is_audited(tmp_path) -> None:
    store = OmsStore(tmp_path)

    store.record_admin_allocation_correction(
        {
            "position_instance_id": "admin_pos_1",
            "strategy_id": "momentum",
            "symbol": "BTC",
            "allocated_qty": 0.1,
        },
        corrected_by="operator",
        reason="matched exchange residual",
    )
    events = store.list_events("admin_allocation_correction")
    store.close()

    assert len(events) == 1
    assert events[0]["payload"]["position_instance_id"] == "admin_pos_1"
    assert events[0]["payload"]["corrected_by"] == "operator"


def _ttl_bar() -> Bar:
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
