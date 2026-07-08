"""Tests for HyperliquidBroker — uses mocked Info/Exchange."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from crypto_trader.core.models import Fill, Order, OrderStatus, OrderType, Position, Side


def _make_broker(private_key=None, **kwargs):
    """Create a broker with mocked Info (patching the import location)."""
    mock_info = MagicMock()

    with patch("hyperliquid.info.Info", return_value=mock_info):
        from crypto_trader.live.broker import HyperliquidBroker

        broker = HyperliquidBroker(
            wallet_address="0x" + "a" * 40,
            private_key=private_key,
            is_testnet=True,
            **kwargs,
        )

    broker._info = mock_info
    broker._rate_limit_interval = 0  # disable rate limiting in tests
    return broker, mock_info


class TestHyperliquidBroker:
    def test_get_equity(self):
        broker, info = _make_broker()
        info.user_state.return_value = {
            "marginSummary": {"accountValue": "12345.67"},
        }
        assert broker.get_equity() == pytest.approx(12345.67)

    def test_get_position_long(self):
        broker, info = _make_broker()
        info.user_state.return_value = {
            "assetPositions": [{
                "position": {
                    "coin": "BTC",
                    "szi": "0.5",
                    "entryPx": "50000.0",
                    "unrealizedPnl": "100.0",
                    "leverage": {"value": "10"},
                    "liquidationPx": "45000.0",
                },
            }],
        }
        pos = broker.get_position("BTC")
        assert pos is not None
        assert pos.direction == Side.LONG
        assert pos.qty == 0.5
        assert pos.avg_entry == 50000.0
        assert pos.unrealized_pnl == 100.0
        assert pos.leverage == 10.0
        assert pos.liquidation_price == 45000.0

    def test_get_position_short(self):
        broker, info = _make_broker()
        info.user_state.return_value = {
            "assetPositions": [{
                "position": {
                    "coin": "ETH",
                    "szi": "-2.0",
                    "entryPx": "3000.0",
                    "unrealizedPnl": "-50.0",
                    "leverage": {"value": "5"},
                    "liquidationPx": "3500.0",
                },
            }],
        }
        pos = broker.get_position("ETH")
        assert pos is not None
        assert pos.direction == Side.SHORT
        assert pos.qty == 2.0

    def test_get_position_flat(self):
        broker, info = _make_broker()
        info.user_state.return_value = {
            "assetPositions": [{
                "position": {"coin": "BTC", "szi": "0"},
            }],
        }
        assert broker.get_position("BTC") is None

    def test_get_positions_multiple(self):
        broker, info = _make_broker()
        info.user_state.return_value = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "0.1", "entryPx": "50000",
                              "unrealizedPnl": "10", "leverage": {"value": "10"}, "liquidationPx": None}},
                {"position": {"coin": "ETH", "szi": "-1.0", "entryPx": "3000",
                              "unrealizedPnl": "-5", "leverage": {"value": "5"}, "liquidationPx": None}},
                {"position": {"coin": "SOL", "szi": "0", "entryPx": "0",
                              "unrealizedPnl": "0", "leverage": {"value": "1"}, "liquidationPx": None}},
            ],
        }
        positions = broker.get_positions()
        assert len(positions) == 2  # SOL is flat

    def test_get_fills_since(self):
        broker, info = _make_broker()
        info.user_fills_by_time.return_value = [
            {
                "coin": "BTC", "px": "51000.0", "sz": "0.1", "side": "B",
                "time": 1713600000000, "fee": "1.785", "oid": "123",
            },
            {
                "coin": "ETH", "px": "3100.0", "sz": "1.0", "side": "A",
                "time": 1713601000000, "fee": "1.085", "oid": "124",
            },
        ]
        since = datetime(2026, 4, 20, tzinfo=timezone.utc)
        fills = broker.get_fills_since(since)
        assert len(fills) == 2
        assert fills[0].side == Side.LONG
        assert fills[1].side == Side.SHORT
        assert fills[0].qty == 0.1

    def test_get_open_orders(self):
        broker, info = _make_broker()
        info.open_orders.return_value = [
            {"coin": "BTC", "oid": "100", "side": "B", "sz": "0.1", "limitPx": "49000"},
            {"coin": "ETH", "oid": "101", "side": "A", "sz": "2.0", "limitPx": "3200"},
        ]
        orders = broker.get_open_orders()
        assert len(orders) == 2
        assert orders[0].side == Side.LONG
        assert orders[1].side == Side.SHORT

    def test_get_open_orders_filtered(self):
        broker, info = _make_broker()
        info.open_orders.return_value = [
            {"coin": "BTC", "oid": "100", "side": "B", "sz": "0.1", "limitPx": "49000"},
            {"coin": "ETH", "oid": "101", "side": "A", "sz": "2.0", "limitPx": "3200"},
        ]
        orders = broker.get_open_orders("BTC")
        assert len(orders) == 1
        assert orders[0].symbol == "BTC"

    def test_get_open_orders_preserves_tracked_order_metadata(self):
        broker, info = _make_broker()
        tracked = Order(
            order_id="local_1",
            symbol="BTC",
            side=Side.LONG,
            order_type=OrderType.STOP,
            qty=0.1,
            stop_price=49000.0,
            tag="protective_stop",
            ttl_bars=2,
            metadata={"strategy_id": "momentum"},
            _bars_alive=1,
        )
        broker._orders["local_1"] = tracked
        broker._oid_map["100"] = "local_1"
        info.open_orders.return_value = [
            {"coin": "BTC", "oid": "100", "side": "B", "sz": "0.1", "limitPx": "49000"},
        ]

        orders = broker.get_open_orders("BTC")

        assert len(orders) == 1
        assert orders[0].order_id == "local_1"
        assert orders[0].tag == "protective_stop"
        assert orders[0].ttl_bars == 2
        assert orders[0]._bars_alive == 1
        assert orders[0].metadata["strategy_id"] == "momentum"
        assert orders[0].metadata["ttl_bars_alive"] == 1

    def test_submit_order_read_only(self):
        broker, info = _make_broker(private_key=None)
        order = Order(
            order_id="o1", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
        )
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_submit_order_zero_qty(self):
        broker, info = _make_broker()
        order = Order(
            order_id="o1", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.0001,
        )
        broker._lot_sizes["BTC"] = 0.001
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_order_owner_tracking(self):
        broker, info = _make_broker()
        order = Order(
            order_id="local_1", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=0.1,
            metadata={"strategy_id": "momentum"},
        )
        broker._orders["local_1"] = order
        assert broker.get_order_owner("local_1") == "momentum"
        assert broker.get_order_owner("unknown") is None
