"""Regression tests for live broker local/client order ID safety."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from crypto_trader.core.models import Order, OrderStatus, OrderType, Side


def _make_broker_with_exchange():
    mock_info = MagicMock()
    mock_exchange = MagicMock()

    with patch("hyperliquid.info.Info", return_value=mock_info):
        from crypto_trader.live.broker import HyperliquidBroker

        broker = HyperliquidBroker(
            wallet_address="0x" + "a" * 40,
            private_key=None,
            is_testnet=True,
        )

    broker._info = mock_info
    broker._exchange = mock_exchange
    broker._rate_limit_interval = 0
    mock_info.all_mids.return_value = {"BTC": "50000"}
    return broker, mock_info, mock_exchange


def _resting_response(oid: str) -> dict:
    return {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": oid}}]}},
    }


def _filled_response(oid: str) -> dict:
    return {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"oid": oid}}]}},
    }


def _raw_fill(oid: str) -> dict:
    return {
        "coin": "BTC",
        "px": "50100.0",
        "sz": "0.01",
        "side": "B",
        "time": 1713600000000,
        "fee": "0.175",
        "oid": oid,
    }


def test_blank_market_order_gets_unique_id_and_maps_exchange_fill() -> None:
    broker, info, exchange = _make_broker_with_exchange()
    exchange.order.return_value = _filled_response("123")

    order = Order(
        order_id="",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.01,
        tag="entry",
        metadata={"strategy_id": "momentum"},
    )

    local_id = broker.submit_order(order)

    assert local_id == "hl_momentum_BTC_000001"
    assert order.order_id == local_id
    assert order.metadata["client_order_id"] == local_id
    assert order.status == OrderStatus.FILLED
    assert "" not in broker._orders
    assert "" not in broker._local_to_oid
    assert broker._orders[local_id] is order
    assert broker._local_to_oid[local_id] == "123"
    assert broker._oid_map["123"] == local_id
    assert broker.get_order_owner(local_id) == "momentum"
    assert broker.get_order_owner("123") == "momentum"

    info.user_fills_by_time.return_value = [_raw_fill("123")]
    fills = broker.get_fills_since(datetime(2026, 4, 20, tzinfo=timezone.utc))

    assert len(fills) == 1
    assert fills[0].order_id == local_id
    assert fills[0].tag == "entry"


def test_blank_stop_orders_submitted_sequentially_do_not_collide() -> None:
    broker, _, exchange = _make_broker_with_exchange()
    exchange.order.side_effect = [
        _resting_response("201"),
        _resting_response("202"),
    ]

    first = Order(
        order_id="",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=49000.0,
        tag="protective_stop",
        metadata={"strategy_id": "momentum"},
    )
    second = Order(
        order_id="",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=49500.0,
        tag="trailing_stop",
        metadata={"strategy_id": "momentum"},
    )

    first_id = broker.submit_order(first)
    second_id = broker.submit_order(second)

    assert first_id == "hl_momentum_BTC_000001"
    assert second_id == "hl_momentum_BTC_000002"
    assert first_id != second_id
    assert broker._local_to_oid[first_id] == "201"
    assert broker._local_to_oid[second_id] == "202"


def test_non_empty_strategy_order_id_is_preserved() -> None:
    broker, _, exchange = _make_broker_with_exchange()
    exchange.order.return_value = _resting_response("301")

    order = Order(
        order_id="strategy_order_1",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.LIMIT,
        qty=0.01,
        limit_price=49900.0,
        metadata={"strategy_id": "trend"},
    )

    local_id = broker.submit_order(order)

    assert local_id == "strategy_order_1"
    assert order.order_id == "strategy_order_1"
    assert order.metadata["client_order_id"] == "strategy_order_1"
    assert broker._local_to_oid["strategy_order_1"] == "301"
    assert broker._oid_map["301"] == "strategy_order_1"


def test_cancel_uses_generated_local_id_to_find_exchange_oid() -> None:
    broker, _, exchange = _make_broker_with_exchange()
    exchange.order.return_value = _resting_response("401")
    exchange.cancel.return_value = {"status": "ok"}

    order = Order(
        order_id="",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=49000.0,
        tag="protective_stop",
        metadata={"strategy_id": "breakout"},
    )

    local_id = broker.submit_order(order)

    assert broker.cancel_order(local_id) is True
    exchange.cancel.assert_called_once_with("BTC", 401)
    assert order.status == OrderStatus.CANCELLED


def test_blank_rejected_order_still_gets_id_but_is_not_registered() -> None:
    broker, _, _ = _make_broker_with_exchange()
    broker._exchange = None

    order = Order(
        order_id="",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.01,
        metadata={"strategy_id": "momentum"},
    )

    local_id = broker.submit_order(order)

    assert local_id == "hl_momentum_BTC_000001"
    assert order.status == OrderStatus.REJECTED
    assert broker._orders == {}
    assert broker._local_to_oid == {}
    assert broker._oid_map == {}


def test_blank_zero_qty_and_unsupported_orders_get_ids_but_are_not_registered() -> None:
    broker, _, exchange = _make_broker_with_exchange()

    zero_qty = Order(
        order_id="",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.0,
        metadata={"strategy_id": "momentum"},
    )
    unsupported = Order(
        order_id="",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.STOP_LIMIT,
        qty=0.01,
        limit_price=50100.0,
        stop_price=50000.0,
        metadata={"strategy_id": "momentum"},
    )

    assert broker.submit_order(zero_qty) == "hl_momentum_BTC_000001"
    assert broker.submit_order(unsupported) == "hl_momentum_BTC_000002"
    assert zero_qty.status == OrderStatus.REJECTED
    assert unsupported.status == OrderStatus.REJECTED
    assert broker._orders == {}
    assert broker._local_to_oid == {}
    assert broker._oid_map == {}
    exchange.order.assert_not_called()


def test_exit_stop_submits_reduce_only_to_exchange() -> None:
    broker, _, exchange = _make_broker_with_exchange()
    exchange.order.return_value = _resting_response("501")

    order = Order(
        order_id="stop_1",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=0.01,
        stop_price=49000.0,
        tag="protective_stop",
        metadata={"strategy_id": "momentum", "reduce_only": True},
    )

    broker.submit_order(order)

    assert exchange.order.call_args.kwargs["reduce_only"] is True


def test_momentum_stop_tags_submit_as_stop_loss_triggers() -> None:
    broker, _, exchange = _make_broker_with_exchange()
    exchange.order.side_effect = [_resting_response("601"), _resting_response("602")]

    for tag in ("breakeven_stop", "proof_lock_stop"):
        broker.submit_order(Order(
            order_id=f"{tag}_1",
            symbol="BTC",
            side=Side.SHORT,
            order_type=OrderType.STOP,
            qty=0.01,
            stop_price=49000.0,
            tag=tag,
            metadata={"strategy_id": "momentum", "reduce_only": True},
        ))

    trigger_payloads = [call.args[4]["trigger"] for call in exchange.order.call_args_list]
    assert [payload["tpsl"] for payload in trigger_payloads] == ["sl", "sl"]
