"""Tests for adapter-neutral execution wrappers."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.broker.sim_execution_adapter import SimExecutionAdapter
from crypto_trader.core.models import Bar, Fill, Order, OrderStatus, OrderType, Position, Side, TimeFrame
from crypto_trader.core.runtime_types import ExecutionReportKind, OrderIntent
from crypto_trader.live.execution_adapter import HyperliquidExecutionAdapter


def _intent(**overrides) -> OrderIntent:
    defaults = dict(
        intent_id="intent_1",
        strategy_id="momentum",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.01,
        decision_id="decision_1",
        client_order_id="client_1",
    )
    defaults.update(overrides)
    return OrderIntent(**defaults)


def test_sim_execution_adapter_submits_intent_as_accepted_report() -> None:
    adapter = SimExecutionAdapter(SimBroker(initial_equity=10_000.0))

    reports = adapter.submit(_intent())

    assert len(reports) == 1
    assert reports[0].kind == ExecutionReportKind.ACCEPTED
    assert reports[0].client_order_id == "client_1"
    assert reports[0].exchange_order_id == "1"
    assert reports[0].order_status == OrderStatus.PENDING


def test_sim_execution_adapter_maps_client_id_to_broker_cancel_id() -> None:
    broker = SimBroker(initial_equity=10_000.0)
    adapter = SimExecutionAdapter(broker)
    adapter.submit(_intent(
        client_order_id="strategy_stop_1",
        order_type=OrderType.LIMIT,
        limit_price=100.0,
    ))

    report = adapter.cancel("strategy_stop_1")[0]

    assert report.kind == ExecutionReportKind.CANCELLED
    assert report.client_order_id == "strategy_stop_1"
    assert report.exchange_order_id == "1"
    assert broker.get_open_orders() == []


def test_sim_execution_adapter_blank_client_id_uses_broker_id() -> None:
    broker = SimBroker(initial_equity=10_000.0)
    adapter = SimExecutionAdapter(broker)

    reports = adapter.submit(_intent(client_order_id=""))

    assert reports[0].client_order_id == "1"
    assert reports[0].exchange_order_id == "1"
    assert broker.get_open_orders()[0].metadata["client_order_id"] == "1"


def test_live_execution_adapter_rejects_unsupported_stop_limit_before_broker() -> None:
    broker = MagicMock()
    adapter = HyperliquidExecutionAdapter(broker)

    reports = adapter.submit(_intent(
        order_type=OrderType.STOP_LIMIT,
        stop_price=100.0,
        limit_price=99.0,
    ))

    assert reports[0].kind == ExecutionReportKind.REJECTED
    assert reports[0].reject_reason == "stop_limit_not_supported_live"
    broker.submit_order.assert_not_called()


def test_live_execution_adapter_accepts_reduce_only_but_rejects_oca_and_bracket() -> None:
    broker = MagicMock()
    broker.submit_order.side_effect = lambda order: order.order_id
    broker._local_to_oid = {"client_1": "101"}
    adapter = HyperliquidExecutionAdapter(broker)

    reports = adapter.submit(_intent(reduce_only=True))

    assert reports[0].kind == ExecutionReportKind.ACCEPTED
    submitted = broker.submit_order.call_args.args[0]
    assert submitted.metadata["reduce_only"] is True

    broker.reset_mock()
    cases = [
        (_intent(oca_group="g1"), "oca_not_supported_live"),
        (_intent(bracket_group="b1"), "bracket_not_supported_live"),
    ]
    for intent, reason in cases:
        reports = adapter.submit(intent)
        assert reports[0].kind == ExecutionReportKind.REJECTED
        assert reports[0].reject_reason == reason
    broker.submit_order.assert_not_called()


def test_live_execution_adapter_accepts_broker_managed_oca_fallback_without_native_claim() -> None:
    broker = MagicMock()
    broker.submit_order.side_effect = lambda order: order.order_id
    broker._local_to_oid = {"client_1": "101"}
    adapter = HyperliquidExecutionAdapter(broker)

    reports = adapter.submit(_intent(
        reduce_only=True,
        oca_group="momentum:BTC:pos_1:exit_oca",
        metadata={
            "tag": "protective_stop",
            "oca_group": "momentum:BTC:pos_1:exit_oca",
            "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
            "exit_only": True,
            "native_oca_required": False,
        },
    ))

    assert HyperliquidExecutionAdapter.capabilities.oca is False
    assert reports[0].kind == ExecutionReportKind.ACCEPTED
    submitted = broker.submit_order.call_args.args[0]
    assert submitted.oca_group == "momentum:BTC:pos_1:exit_oca"
    assert submitted.metadata["oca_group"] == submitted.oca_group


def test_live_execution_adapter_rejects_forged_broker_managed_oca_fallback() -> None:
    broker = MagicMock()
    adapter = HyperliquidExecutionAdapter(broker)

    for intent in [
        _intent(
            oca_group="momentum:BTC:pos_1:exit_oca",
            metadata={
                "tag": "protective_stop",
                "oca_group": "momentum:BTC:pos_1:exit_oca",
                "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
                "exit_only": True,
                "native_oca_required": False,
            },
        ),
        _intent(
            reduce_only=True,
            oca_group="g1",
            metadata={
                "tag": "protective_stop",
                "oca_group": "g1",
                "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
                "exit_only": True,
                "native_oca_required": False,
            },
        ),
        _intent(
            reduce_only=True,
            oca_group="momentum:BTC:pos_1:exit_oca",
            metadata={
                "tag": "entry",
                "oca_group": "momentum:BTC:pos_1:exit_oca",
                "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
                "exit_only": True,
                "native_oca_required": False,
            },
        ),
    ]:
        reports = adapter.submit(intent)
        assert reports[0].kind == ExecutionReportKind.REJECTED
        assert reports[0].reject_reason == "oca_not_supported_live"

    broker.submit_order.assert_not_called()


def test_hyperliquid_oca_probe_documents_no_native_support() -> None:
    probe = HyperliquidExecutionAdapter.probe_oca_capabilities()

    assert probe["native_oca"] is False
    assert probe["broker_managed_fallback"] is True


def test_sim_execution_adapter_accepts_stop_limit_oca_and_ttl_intent() -> None:
    adapter = SimExecutionAdapter(SimBroker(initial_equity=10_000.0))

    reports = adapter.submit(_intent(
        order_type=OrderType.STOP_LIMIT,
        stop_price=100.0,
        limit_price=100.5,
        oca_group="g1",
        ttl_bars=2,
    ))

    assert reports[0].kind == ExecutionReportKind.ACCEPTED
    assert reports[0].order_status == OrderStatus.PENDING


def test_sim_execution_adapter_reports_oca_sibling_cancellation() -> None:
    broker = SimBroker(initial_equity=10_000.0)
    adapter = SimExecutionAdapter(broker)
    ts = datetime(2026, 5, 24, tzinfo=timezone.utc)
    broker._pending_orders = [
        Order(
            order_id="tp_1",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            qty=0.1,
            limit_price=52_000.0,
            tag="tp1",
            oca_group="momentum:BTC:pos_1:exit_oca",
            metadata={"client_order_id": "tp_client", "oca_group": "momentum:BTC:pos_1:exit_oca"},
        ),
        Order(
            order_id="stop_1",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=0.1,
            stop_price=49_000.0,
            tag="protective_stop",
            oca_group="momentum:BTC:pos_1:exit_oca",
            metadata={"client_order_id": "stop_client", "oca_group": "momentum:BTC:pos_1:exit_oca"},
        ),
    ]
    broker._process_oca_cancels([Fill(
        order_id="tp_1",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.1,
        fill_price=52_000.0,
        commission=1.0,
        timestamp=ts,
        tag="tp1",
    )])

    reports = adapter.sync_fills(ts)

    cancelled = [report for report in reports if report.kind == ExecutionReportKind.CANCELLED]
    assert len(cancelled) == 1
    assert cancelled[0].client_order_id == "stop_client"
    assert cancelled[0].metadata["cancel_reason"] == "oca_sibling_filled"


def test_sim_broker_keeps_residual_protective_stop_after_partial_tp() -> None:
    broker = SimBroker(initial_equity=10_000.0)
    group = "momentum:BTC:pos_1:exit_oca"
    ts = datetime(2026, 5, 24, tzinfo=timezone.utc)
    broker._positions["BTC"] = Position(
        symbol="BTC",
        direction=Side.LONG,
        qty=0.1,
        avg_entry=50_000.0,
    )
    stop = Order(
        order_id="stop_1",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.1,
        stop_price=49_000.0,
        tag="protective_stop",
        oca_group=group,
        metadata={
            "oca_group": group,
            "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
            "reduce_only": True,
        },
    )
    broker._pending_orders = [
        Order(
            order_id="tp_1",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            qty=0.1,
            limit_price=52_000.0,
            tag="tp1",
            oca_group=group,
            metadata={
                "oca_group": group,
                "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
                "reduce_only": True,
            },
        ),
        stop,
    ]

    broker._process_oca_cancels([Fill(
        order_id="tp_1",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.1,
        fill_price=52_000.0,
        commission=1.0,
        timestamp=ts,
        tag="tp1",
    )])

    assert stop.status == OrderStatus.PENDING
    assert broker.drain_cancelled_oca_orders() == []


def test_live_adapter_keeps_residual_protective_stop_after_partial_tp() -> None:
    group = "momentum:BTC:pos_1:exit_oca"
    ts = datetime(2026, 5, 24, tzinfo=timezone.utc)
    tp = Order(
        order_id="tp_1",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.LIMIT,
        qty=0.1,
        limit_price=52_000.0,
        tag="tp1",
        oca_group=group,
        metadata={
            "oca_group": group,
            "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
            "reduce_only": True,
        },
    )
    stop = Order(
        order_id="stop_1",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.1,
        stop_price=49_000.0,
        tag="protective_stop",
        oca_group=group,
        metadata={
            "oca_group": group,
            "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
            "reduce_only": True,
        },
    )
    fill = Fill(
        order_id="tp_1",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.1,
        fill_price=52_000.0,
        commission=1.0,
        timestamp=ts,
        tag="tp1",
        raw={"remaining_qty": 0.0},
    )
    broker = MagicMock()
    broker._orders = {"tp_1": tp}
    broker.get_fills_since.return_value = [fill]
    broker.get_position.return_value = Position(
        symbol="BTC",
        direction=Side.LONG,
        qty=0.1,
        avg_entry=50_000.0,
    )
    broker.get_open_orders.return_value = [stop]
    adapter = HyperliquidExecutionAdapter(broker)

    reports = adapter.sync_fills(ts)

    assert [report.kind for report in reports] == [ExecutionReportKind.FILL]
    broker.cancel_order.assert_not_called()


def test_live_adapter_cancels_oca_sibling_after_terminal_exit_fill() -> None:
    group = "momentum:BTC:pos_1:exit_oca"
    ts = datetime(2026, 5, 24, tzinfo=timezone.utc)
    tp = Order(
        order_id="tp_1",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.LIMIT,
        qty=0.1,
        limit_price=52_000.0,
        tag="tp1",
        oca_group=group,
        metadata={
            "oca_group": group,
            "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
            "reduce_only": True,
        },
    )
    stop = Order(
        order_id="stop_1",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.1,
        stop_price=49_000.0,
        tag="protective_stop",
        oca_group=group,
        metadata={
            "client_order_id": "stop_client",
            "oca_group": group,
            "oca_policy": "broker_managed_cancel_siblings_on_terminal_close",
            "reduce_only": True,
        },
    )
    fill = Fill(
        order_id="tp_1",
        symbol="BTC",
        side=Side.SHORT,
        qty=0.1,
        fill_price=52_000.0,
        commission=1.0,
        timestamp=ts,
        tag="tp1",
    )
    broker = MagicMock()
    broker._orders = {"tp_1": tp}
    broker.get_fills_since.return_value = [fill]
    broker.get_position.return_value = None
    broker.get_open_orders.return_value = [stop]
    broker.cancel_order.return_value = True
    adapter = HyperliquidExecutionAdapter(broker)

    reports = adapter.sync_fills(ts)

    cancelled = [report for report in reports if report.kind == ExecutionReportKind.CANCELLED]
    assert len(cancelled) == 1
    assert cancelled[0].client_order_id == "stop_1"
    assert cancelled[0].metadata["cancel_reason"] == "oca_sibling_filled"


def test_live_execution_adapter_sync_fills_maps_fill_report() -> None:
    from crypto_trader.core.models import Fill

    fill = Fill(
        order_id="client_1",
        symbol="BTC",
        side=Side.LONG,
        qty=0.01,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, tzinfo=timezone.utc),
        tag="entry",
    )
    broker = MagicMock()
    broker.get_fills_since.return_value = [fill]
    adapter = HyperliquidExecutionAdapter(broker)

    reports = adapter.sync_fills(datetime(2026, 5, 23, tzinfo=timezone.utc))

    assert reports[0].kind == ExecutionReportKind.FILL
    assert reports[0].filled_qty == 0.01
    assert reports[0].metadata["tag"] == "entry"


def test_live_execution_adapter_expires_ttl_stop_at_last_allowed_bar_close() -> None:
    order = Order(
        order_id="client_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=100.0,
        tag="entry",
        ttl_bars=2,
    )
    broker = MagicMock()
    broker.submit_order.side_effect = lambda submitted: submitted.order_id
    broker.get_open_orders.return_value = [order]
    broker.cancel_order.return_value = True
    broker._local_to_oid = {"client_1": "101"}
    adapter = HyperliquidExecutionAdapter(broker)

    reports = adapter.submit(_intent(
        order_type=OrderType.STOP,
        stop_price=100.0,
        ttl_bars=2,
    ))

    assert reports[0].kind == ExecutionReportKind.ACCEPTED
    bar = Bar(
        timestamp=datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc),
        symbol="BTC",
        open=99.0,
        high=99.5,
        low=98.5,
        close=99.0,
        volume=1.0,
        timeframe=TimeFrame.M15,
    )
    assert adapter.expire_ttl_orders_for_bar(bar) == []
    expired = adapter.expire_ttl_orders_for_bar(bar)

    assert expired[0].kind == ExecutionReportKind.EXPIRED
    assert expired[0].order_status == OrderStatus.EXPIRED
    assert expired[0].client_order_id == "client_1"
    broker.cancel_order.assert_called_once_with("client_1")


def test_live_execution_adapter_clears_stop_ttl_on_fill() -> None:
    broker = MagicMock()
    broker.submit_order.side_effect = lambda submitted: submitted.order_id
    broker.cancel_order.return_value = True
    broker._local_to_oid = {"client_1": "101"}
    broker._orders = {}
    adapter = HyperliquidExecutionAdapter(broker, strategy_id="momentum")

    adapter.submit(_intent(order_type=OrderType.STOP, stop_price=100.0, ttl_bars=2))
    assert len(adapter.active_ttl_orders()) == 1

    cleared = adapter.clear_ttl_for_fill(Fill(
        order_id="101",
        exchange_order_id="101",
        symbol="BTC",
        side=Side.LONG,
        qty=0.01,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 16, tzinfo=timezone.utc),
        tag="entry",
    ))

    assert cleared is True
    assert adapter.active_ttl_orders() == []


def test_live_execution_adapter_partial_non_stop_ttl_fill_keeps_remaining_qty() -> None:
    submitted_orders: dict[str, Order] = {}

    def _store_order(submitted: Order) -> str:
        submitted_orders[submitted.order_id] = submitted
        return submitted.order_id

    broker = MagicMock()
    broker.submit_order.side_effect = _store_order
    broker._local_to_oid = {"client_1": "101"}
    broker._orders = submitted_orders
    adapter = HyperliquidExecutionAdapter(broker, strategy_id="momentum")

    adapter.submit(_intent(order_type=OrderType.LIMIT, limit_price=100.0, qty=1.0, ttl_bars=2))
    adapter.clear_ttl_for_fill(Fill(
        order_id="client_1",
        exchange_order_id="101",
        symbol="BTC",
        side=Side.LONG,
        qty=0.4,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 16, tzinfo=timezone.utc),
        tag="entry",
    ))

    active = adapter.active_ttl_orders()
    assert len(active) == 1
    assert round(active[0]["remaining_qty"], 10) == 0.6
    assert round(broker._orders["client_1"].metadata["ttl_remaining_qty"], 10) == 0.6

    adapter.clear_ttl_for_fill(Fill(
        order_id="client_1",
        exchange_order_id="101",
        symbol="BTC",
        side=Side.LONG,
        qty=0.6,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 17, tzinfo=timezone.utc),
        tag="entry",
    ))

    assert adapter.active_ttl_orders() == []


def test_live_execution_adapter_seed_ttl_defaults_invalid_remaining_qty() -> None:
    order = Order(
        order_id="trend_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.LIMIT,
        qty=0.25,
        limit_price=100.0,
        tag="entry",
        ttl_bars=2,
        metadata={"strategy_id": "trend", "ttl_remaining_qty": None},
    )
    broker = MagicMock()
    broker.get_open_orders.return_value = [order]
    broker._local_to_oid = {"trend_1": "201"}
    broker._orders = {"trend_1": order}
    adapter = HyperliquidExecutionAdapter(broker, strategy_id="trend")

    assert adapter.seed_ttl_orders_from_open_orders() == 1

    active = adapter.active_ttl_orders()
    assert active[0]["remaining_qty"] == 0.25


def test_live_execution_adapter_clear_ttl_uses_broker_exchange_id_map() -> None:
    order = Order(
        order_id="trend_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=100.0,
        tag="entry",
        ttl_bars=2,
        metadata={"strategy_id": "trend"},
    )
    broker = MagicMock()
    broker.get_open_orders.return_value = [order]
    broker._local_to_oid = {}
    broker._oid_map = {"201": "trend_1"}
    broker._orders = {"trend_1": order}
    adapter = HyperliquidExecutionAdapter(broker, strategy_id="trend")
    adapter.seed_ttl_orders_from_open_orders()

    cleared = adapter.clear_ttl_for_fill(Fill(
        order_id="201",
        exchange_order_id="201",
        symbol="BTC",
        side=Side.LONG,
        qty=0.01,
        fill_price=100.0,
        commission=0.01,
        timestamp=datetime(2026, 5, 24, 12, 16, tzinfo=timezone.utc),
        tag="entry",
    ))

    assert cleared is True
    assert adapter.active_ttl_orders() == []


def test_live_execution_adapter_seeds_strategy_scoped_ttl_with_bars_alive() -> None:
    trend_order = Order(
        order_id="trend_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=100.0,
        tag="entry",
        ttl_bars=2,
        metadata={"strategy_id": "trend", "ttl_bars_alive": 1},
        _bars_alive=1,
    )
    momentum_order = Order(
        order_id="momentum_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=100.0,
        tag="entry",
        ttl_bars=2,
        metadata={"strategy_id": "momentum"},
    )
    broker = MagicMock()
    broker.get_open_orders.return_value = [trend_order, momentum_order]
    broker.cancel_order.return_value = True
    broker._local_to_oid = {"trend_1": "201", "momentum_1": "202"}
    broker._orders = {"trend_1": trend_order, "momentum_1": momentum_order}
    adapter = HyperliquidExecutionAdapter(broker, strategy_id="trend")

    assert adapter.seed_ttl_orders_from_open_orders() == 1
    active = adapter.active_ttl_orders()
    assert [state["client_order_id"] for state in active] == ["trend_1"]
    assert active[0]["ttl_bars_alive"] == 1

    bar = Bar(
        timestamp=datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc),
        symbol="BTC",
        open=99.0,
        high=99.5,
        low=98.5,
        close=99.0,
        volume=1.0,
        timeframe=TimeFrame.M15,
    )
    expired = adapter.expire_ttl_orders_for_bar(bar)

    assert expired[0].kind == ExecutionReportKind.EXPIRED
    broker.cancel_order.assert_called_once_with("trend_1")


def test_live_execution_adapter_failed_ttl_cancel_stays_working_and_tracked() -> None:
    order = Order(
        order_id="client_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=100.0,
        tag="entry",
        ttl_bars=0,
        metadata={"strategy_id": "momentum"},
    )
    broker = MagicMock()
    broker.submit_order.side_effect = lambda submitted: submitted.order_id
    broker.cancel_order.return_value = False
    broker._local_to_oid = {"client_1": "101"}
    broker._orders = {"client_1": order}
    adapter = HyperliquidExecutionAdapter(broker, strategy_id="momentum")
    adapter.submit(_intent(order_type=OrderType.STOP, stop_price=100.0, ttl_bars=0))

    reports = adapter.expire_ttl_orders_for_bar(Bar(
        timestamp=datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc),
        symbol="BTC",
        open=99.0,
        high=99.5,
        low=98.5,
        close=99.0,
        volume=1.0,
        timeframe=TimeFrame.M15,
    ))

    assert reports[0].kind == ExecutionReportKind.RESTING
    assert reports[0].order_status == OrderStatus.WORKING
    assert reports[0].metadata["ttl_cancel_failed"] is True
    assert len(adapter.active_ttl_orders()) == 1
