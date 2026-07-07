"""Tests for partial fill PnL tracking and force-close at backtest end."""

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import (
    Bar,
    Fill,
    Order,
    OrderType,
    Position,
    Side,
    TimeFrame,
    Trade,
)
from crypto_trader.broker.sim_broker import SimBroker


def _make_bar(symbol="BTC", price=50000.0, ts=None):
    """Create a bar at the given price."""
    if ts is None:
        ts = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    return Bar(
        timestamp=ts,
        symbol=symbol,
        open=price,
        high=price * 1.01,
        low=price * 0.99,
        close=price,
        volume=100.0,
        timeframe=TimeFrame.M15,
    )


class TestPositionPartialFields:
    """Test that Position accumulates partial exit values."""

    def test_position_default_partial_fields(self):
        pos = Position(symbol="BTC", direction=Side.LONG, qty=1.0, avg_entry=50000.0)
        assert pos.partial_exit_pnl == 0.0
        assert pos.partial_exit_commission == 0.0
        assert pos.partial_exit_qty == 0.0

    def test_position_partial_fields_accumulate(self):
        pos = Position(symbol="BTC", direction=Side.LONG, qty=1.0, avg_entry=50000.0)
        pos.partial_exit_pnl += 100.0
        pos.partial_exit_commission += 5.0
        pos.partial_exit_qty += 0.25
        pos.partial_exit_pnl += 200.0
        pos.partial_exit_commission += 3.0
        pos.partial_exit_qty += 0.5
        assert pos.partial_exit_pnl == 300.0
        assert pos.partial_exit_commission == 8.0
        assert pos.partial_exit_qty == pytest.approx(0.75)


class TestSimBrokerPartialFillPnl:
    """Test that SimBroker correctly tracks partial exit PnL."""

    def _make_broker(self, equity=10000.0):
        return SimBroker(
            initial_equity=equity,
            taker_fee_bps=0.0,  # Zero fees to simplify PnL math
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
        )

    def test_partial_close_accumulates_pnl(self):
        """Partial close should accumulate PnL on the Position."""
        broker = self._make_broker()
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # Open long 1.0 BTC at 50000
        entry_order = Order(
            order_id="", symbol="BTC", side=Side.LONG,
            order_type=OrderType.MARKET, qty=1.0,
        )
        broker.submit_order(entry_order)
        bar2 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        # Partially close 0.5 at 52000
        partial_order = Order(
            order_id="", symbol="BTC", side=Side.SHORT,
            order_type=OrderType.MARKET, qty=0.5,
        )
        broker.submit_order(partial_order)
        bar3 = _make_bar(price=52000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar3)

        # Position should still be open with accumulated partial PnL
        pos = broker.get_position("BTC")
        assert pos is not None
        assert pos.qty == pytest.approx(0.5)
        # PnL from partial: 0.5 * (52000 - 50000) = 1000
        assert pos.partial_exit_pnl == pytest.approx(1000.0)
        assert pos.partial_exit_qty == pytest.approx(0.5)

    def test_partial_close_accumulates_commission(self):
        """Partial close should accumulate commission on the Position."""
        broker = SimBroker(
            initial_equity=10000.0,
            taker_fee_bps=10.0,  # 0.1% = easy math
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
        )
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # Open
        entry = Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0)
        broker.submit_order(entry)
        bar2 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        # Partial close 0.5 at 51000
        partial = Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.5)
        broker.submit_order(partial)
        bar3 = _make_bar(price=51000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar3)

        pos = broker.get_position("BTC")
        assert pos is not None
        # Commission on partial close: 0.5 * 51000 * 10/10000 = 25.5
        assert pos.partial_exit_commission == pytest.approx(0.5 * 51000.0 * 10 / 10000)

    def test_trade_pnl_includes_tp_profits(self):
        """Final Trade.pnl should include prior partial close profits."""
        broker = self._make_broker()
        bar0 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar0)

        # Open long 1.0 at 50000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0))
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # TP1: close 0.5 at 52000 → PnL = 0.5*(52000-50000) = 1000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.5))
        bar2 = _make_bar(price=52000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        # Stop: close remaining 0.5 at 49000 → PnL = 0.5*(49000-50000) = -500
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.5))
        bar3 = _make_bar(price=49000.0, ts=datetime(2026, 3, 1, 12, 45, tzinfo=timezone.utc))
        broker.process_bar(bar3)

        assert len(broker._closed_trades) == 1
        trade = broker._closed_trades[0]
        # Total PnL = TP1(1000) + stop(-500) = 500
        assert trade.pnl == pytest.approx(500.0)
        assert trade.qty == pytest.approx(1.0)
        assert trade.exit_price == pytest.approx(50500.0)

    def test_trade_commission_includes_all_fills(self):
        """Trade.commission should include entry + partial exit + final exit."""
        broker = SimBroker(
            initial_equity=100000.0,
            taker_fee_bps=10.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
        )
        bar0 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar0)

        # Open 1.0 at 50000 → commission = 1.0 * 50000 * 10/10000 = 50
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0))
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # TP1: close 0.5 at 51000 → commission = 0.5 * 51000 * 10/10000 = 25.5
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.5))
        bar2 = _make_bar(price=51000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        # Final: close 0.5 at 49000 → commission = 0.5 * 49000 * 10/10000 = 24.5
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.5))
        bar3 = _make_bar(price=49000.0, ts=datetime(2026, 3, 1, 12, 45, tzinfo=timezone.utc))
        broker.process_bar(bar3)

        trade = broker._closed_trades[0]
        expected_commission = 50.0 + 25.5 + 24.5  # entry + TP1 + final
        assert trade.commission == pytest.approx(expected_commission)

    def test_multiple_partial_exits(self):
        """Entry → TP1 → TP2 → stop = all profits captured."""
        broker = self._make_broker(equity=100000.0)
        bar0 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar0)

        # Open 1.0 at 50000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0))
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # TP1: close 0.3 at 52000 → PnL = 0.3 * 2000 = 600
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.3))
        bar2 = _make_bar(price=52000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        # TP2: close 0.3 at 54000 → PnL = 0.3 * 4000 = 1200
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.3))
        bar3 = _make_bar(price=54000.0, ts=datetime(2026, 3, 1, 12, 45, tzinfo=timezone.utc))
        broker.process_bar(bar3)

        # Stop: close 0.4 at 48000 → PnL = 0.4 * (-2000) = -800
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.4))
        bar4 = _make_bar(price=48000.0, ts=datetime(2026, 3, 1, 13, 0, tzinfo=timezone.utc))
        broker.process_bar(bar4)

        trade = broker._closed_trades[0]
        # Total = 600 + 1200 + (-800) = 1000
        assert trade.pnl == pytest.approx(1000.0)
        assert trade.qty == pytest.approx(1.0)
        assert trade.exit_price == pytest.approx(51000.0)

    def test_no_partial_unchanged_behavior(self):
        """Entry → stop (no partials) should work identically to before."""
        broker = self._make_broker()
        bar0 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar0)

        # Open long 1.0 at 50000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0))
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # Close all at 49000 → PnL = 1.0 * (-1000) = -1000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=1.0))
        bar2 = _make_bar(price=49000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        trade = broker._closed_trades[0]
        assert trade.pnl == pytest.approx(-1000.0)
        assert trade.qty == pytest.approx(1.0)
        assert trade.exit_price == pytest.approx(49000.0)

    def test_zero_pnl_partial_does_not_corrupt(self):
        """Partial close at entry price should add zero to accumulator."""
        broker = self._make_broker()
        bar0 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar0)

        # Open 1.0 at 50000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0))
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # Partial close at entry price
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.5))
        bar2 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        pos = broker.get_position("BTC")
        assert pos is not None
        assert pos.partial_exit_pnl == pytest.approx(0.0)


class TestForceClose:
    """Test close_open_positions() for backtest end."""

    def _make_broker(self, equity=10000.0):
        return SimBroker(
            initial_equity=equity,
            taker_fee_bps=0.0,
            maker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
        )

    def test_force_close_creates_trade(self):
        """Force-close should create a Trade with correct PnL."""
        broker = self._make_broker()
        bar0 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar0)

        # Open position
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=0.1))
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # Price moves up
        bar2 = _make_bar(price=55000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        assert len(broker._closed_trades) == 0
        fills = broker.close_open_positions()
        assert len(fills) == 1
        assert len(broker._closed_trades) == 1
        trade = broker._closed_trades[0]
        assert trade.exit_reason == "backtest_end"
        # PnL = 0.1 * (55000 - 50000) = 500
        assert trade.pnl == pytest.approx(500.0)

    def test_force_close_empty_broker(self):
        """Force-close on empty broker should return empty list."""
        broker = self._make_broker()
        fills = broker.close_open_positions()
        assert fills == []
        assert len(broker._closed_trades) == 0

    def test_force_close_with_partial_tp(self):
        """Force-close after partial TP should include TP profit in Trade."""
        broker = self._make_broker(equity=100000.0)
        bar0 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc))
        broker.process_bar(bar0)

        # Open 1.0 at 50000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.LONG, order_type=OrderType.MARKET, qty=1.0))
        bar1 = _make_bar(price=50000.0, ts=datetime(2026, 3, 1, 12, 15, tzinfo=timezone.utc))
        broker.process_bar(bar1)

        # TP1: close 0.5 at 52000 → PnL = 0.5 * 2000 = 1000
        broker.submit_order(Order(order_id="", symbol="BTC", side=Side.SHORT, order_type=OrderType.MARKET, qty=0.5))
        bar2 = _make_bar(price=52000.0, ts=datetime(2026, 3, 1, 12, 30, tzinfo=timezone.utc))
        broker.process_bar(bar2)

        assert len(broker._closed_trades) == 0

        # Force-close remaining 0.5 at last price (52000)
        # PnL from force-close = 0.5 * (52000 - 50000) = 1000
        fills = broker.close_open_positions()
        assert len(fills) == 1
        trade = broker._closed_trades[0]
        # Total = TP1(1000) + final(1000) = 2000
        assert trade.pnl == pytest.approx(2000.0)
        assert trade.qty == pytest.approx(1.0)
        assert trade.exit_price == pytest.approx(52000.0)
        assert trade.exit_reason == "backtest_end"
