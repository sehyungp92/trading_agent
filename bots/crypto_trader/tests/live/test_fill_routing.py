"""Live fill ownership routing regressions."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crypto_trader.core.engine import MultiTimeFrameBars
from crypto_trader.core.models import Fill, OrderStatus, Side, TimeFrame
from crypto_trader.core.runtime_types import ExecutionReport, ExecutionReportKind
from crypto_trader.live.engine import LiveEngine, _StrategySlot
from crypto_trader.live.oms_store import OmsStore
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.coordinator import StrategyCoordinator
from crypto_trader.portfolio.manager import PortfolioManager
from crypto_trader.portfolio.state import PortfolioState


def test_fill_poll_uses_adapter_reports_for_broker_managed_oca_cancellations(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._last_fill_check = datetime(2026, 5, 24, 11, 59, tzinfo=timezone.utc)
    engine._config = SimpleNamespace(
        state_dir=tmp_path,
        fill_query_overlap_sec=300,
    )
    engine._oms = OmsStore(tmp_path)
    engine._last_assistant_event_at = {}
    engine._emit_assistant_from_canonical_event = MagicMock()
    engine._broker = MagicMock()
    engine._broker.get_fills_since.side_effect = AssertionError("raw broker fill poll bypassed adapter")
    fill_report = ExecutionReport(
        report_id="hl_fill_entry_1",
        kind=ExecutionReportKind.FILL,
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        symbol="BTC",
        side=Side.LONG,
        client_order_id="entry_1",
        exchange_order_id="101",
        fill_id="fill_101",
        order_status=OrderStatus.FILLED,
        filled_qty=0.1,
        fill_price=50_000.0,
        commission=1.0,
        metadata={"tag": "entry", "strategy_id": "momentum"},
    )
    cancel_report = ExecutionReport(
        report_id="hl_oca_cancel_stop_1",
        kind=ExecutionReportKind.CANCELLED,
        timestamp=datetime(2026, 5, 24, 12, 0, 1, tzinfo=timezone.utc),
        symbol="BTC",
        side=Side.SHORT,
        client_order_id="stop_1",
        exchange_order_id="102",
        order_status=OrderStatus.CANCELLED,
        qty=0.1,
        metadata={
            "tag": "protective_stop",
            "strategy_id": "momentum",
            "position_instance_id": "pos_1",
            "reduce_only": True,
            "oca_group": "momentum:BTC:pos_1:exit_oca",
            "cancel_reason": "oca_sibling_filled",
        },
    )
    engine._execution_adapter = SimpleNamespace(
        sync_fills=MagicMock(return_value=[fill_report, cancel_report]),
    )

    def process_fills(fills):
        assert len(fills) == 1
        assert fills[0].order_id == "entry_1"
        assert fills[0].exchange_fill_id == "fill_101"
        return SimpleNamespace(processed=fills, safe_watermark_fills=fills)

    engine._process_fills = MagicMock(side_effect=process_fills)

    processed = engine._poll_and_process_fills()
    row = engine._oms.get_order("stop_1")
    reports = engine._oms._conn.execute(
        "SELECT kind FROM execution_reports WHERE report_id=?",
        ("hl_oca_cancel_stop_1",),
    ).fetchall()
    parity_rows = (tmp_path / "parity_events.jsonl").read_text(encoding="utf-8").splitlines()
    engine._oms.close()

    assert processed[0].order_id == "entry_1"
    engine._execution_adapter.sync_fills.assert_called_once()
    engine._broker.get_fills_since.assert_not_called()
    assert row is not None
    assert row["status"] == OrderStatus.CANCELLED.value
    assert row["metadata"]["cancel_reason"] == "oca_sibling_filled"
    assert [r["kind"] for r in reports] == [ExecutionReportKind.CANCELLED.value]
    assert any("hl_oca_cancel_stop_1" in line for line in parity_rows)


async def _poll_one_fill(
    fill: Fill,
    *,
    coordinator_owner: str | None,
    broker_owner: str | None,
):
    engine = object.__new__(LiveEngine)
    engine._running = True
    engine._last_fill_check = datetime(2026, 4, 25, 8, 59, tzinfo=timezone.utc)
    engine._tracked_positions = {}
    engine._config = SimpleNamespace(fill_poll_interval_sec=0)
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()

    broker = MagicMock()

    def get_fills_since(_since):
        engine._running = False
        return [fill]

    broker.get_fills_since.side_effect = get_fills_since
    broker.get_order_owner.return_value = broker_owner
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = coordinator_owner
    coordinator.on_fill.return_value = coordinator_owner
    engine._coordinator = coordinator

    strategy = SimpleNamespace(symbols=["BTC"], on_fill=MagicMock())
    slot = _StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=MagicMock(),
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )
    engine._slots = [slot]

    with patch("crypto_trader.live.engine.asyncio.sleep", new=AsyncMock()):
        await engine._poll_fills_loop()

    return engine, broker, coordinator, strategy, slot


@pytest.mark.asyncio
async def test_coordinator_owner_wins_and_on_fill_is_called_once() -> None:
    fill = Fill(
        order_id="exchange_or_local_id",
        symbol="BTC",
        side=Side.LONG,
        qty=0.01,
        fill_price=50000.0,
        commission=0.175,
        timestamp=datetime(2026, 4, 25, 9, 0, tzinfo=timezone.utc),
        tag="entry",
    )

    engine, broker, coordinator, strategy, slot = await _poll_one_fill(
        fill,
        coordinator_owner="momentum",
        broker_owner="trend",
    )

    coordinator.on_fill.assert_called_once_with(fill)
    broker.get_order_owner.assert_not_called()
    strategy.on_fill.assert_called_once_with(fill, slot.ctx)
    assert engine._tracked_positions["BTC"]["strategy_id"] == "momentum"
    engine._health.on_error.assert_not_called()


@pytest.mark.asyncio
async def test_broker_owner_is_fallback_when_coordinator_has_no_match() -> None:
    fill = Fill(
        order_id="broker_known_id",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.01,
        fill_price=49000.0,
        commission=0.175,
        timestamp=datetime(2026, 4, 25, 9, 0, tzinfo=timezone.utc),
        tag="protective_stop",
    )

    engine, broker, coordinator, strategy, slot = await _poll_one_fill(
        fill,
        coordinator_owner=None,
        broker_owner="momentum",
    )

    coordinator.on_fill.assert_called_once_with(fill)
    broker.get_order_owner.assert_called_once_with("broker_known_id")
    strategy.on_fill.assert_called_once_with(fill, slot.ctx)
    engine._health.on_error.assert_not_called()


def test_oms_owner_routes_restart_fill_for_filled_order(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._tracked_positions = {}
    engine._config = SimpleNamespace(state_dir=tmp_path)
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._oms.upsert_order(
        client_order_id="client_entry_1",
        exchange_order_id="777",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG.value,
        order_type="MARKET",
        status="FILLED",
        role="entry",
        metadata={"risk_R": 0.75},
    )
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.return_value = None
    engine._lifecycle.snapshot.return_value = []

    broker = MagicMock()
    broker._local_to_oid = {}
    broker.get_order_owner.return_value = None
    engine._broker = broker

    coordinator = MagicMock()
    coordinator.get_strategy_for_order.return_value = None
    coordinator.on_fill.return_value = "momentum"
    engine._coordinator = coordinator

    strategy = SimpleNamespace(symbols=["BTC"], on_fill=MagicMock())
    slot = _StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=MagicMock(),
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )
    engine._slots = [slot]

    fill = Fill(
        order_id="777",
        exchange_order_id="777",
        symbol="BTC",
        side=Side.LONG,
        qty=0.01,
        fill_price=50000.0,
        commission=0.175,
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill777",
    )

    result = engine._process_fills([fill])

    assert len(result.processed) == 1
    assert result.processed[0].order_id == "client_entry_1"
    slot.ctx.broker.clear_ttl_for_fill.assert_called_once_with(result.processed[0])
    strategy.on_fill.assert_called_once_with(result.processed[0], slot.ctx)
    coordinator.register_order.assert_any_call("client_entry_1", "momentum")
    coordinator.register_order.assert_any_call("777", "momentum")
    engine._oms.close()


def test_rehydrated_exchange_owner_still_normalizes_fill_to_client_id(tmp_path) -> None:
    engine = object.__new__(LiveEngine)
    engine._tracked_positions = {}
    engine._config = SimpleNamespace(state_dir=tmp_path)
    engine._health = MagicMock()
    engine._detect_position_closures = MagicMock()
    engine._oms = OmsStore(tmp_path)
    engine._oms.upsert_order(
        client_order_id="client_entry_1",
        exchange_order_id="777",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG.value,
        order_type="MARKET",
        status="FILLED",
        role="entry",
        metadata={"risk_R": 0.75},
    )
    engine._lifecycle = MagicMock()
    engine._lifecycle.apply_fill.return_value = None
    engine._lifecycle.snapshot.return_value = []

    broker = MagicMock()
    broker._orders = {}
    broker._local_to_oid = {}
    broker._oid_map = {}
    broker.get_order_owner.return_value = None
    engine._broker = broker
    manager = PortfolioManager(
        PortfolioConfig(),
        PortfolioState(equity=10_000.0, peak_equity=10_000.0),
    )
    engine._coordinator = StrategyCoordinator(broker, manager)
    engine._rehydrate_oms_orders()
    assert engine._coordinator.get_strategy_for_order("777") == "momentum"

    strategy = SimpleNamespace(symbols=["BTC"], on_fill=MagicMock())
    slot = _StrategySlot(
        strategy_id="momentum",
        strategy=strategy,
        ctx=MagicMock(),
        bars=MultiTimeFrameBars(),
        subscribed_tfs={TimeFrame.M15},
        primary_tf=TimeFrame.M15,
    )
    engine._slots = [slot]
    fill = Fill(
        order_id="777",
        exchange_order_id="777",
        symbol="BTC",
        side=Side.LONG,
        qty=0.01,
        fill_price=50000.0,
        commission=0.175,
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
        tag="entry",
        exchange_fill_id="fill777",
    )

    result = engine._process_fills([fill])

    assert len(result.processed) == 1
    assert result.processed[0].order_id == "client_entry_1"
    strategy.on_fill.assert_called_once_with(result.processed[0], slot.ctx)
    engine._oms.close()
