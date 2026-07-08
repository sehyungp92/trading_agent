"""Characterize current terminal-mark versus force-close accounting behavior."""

from datetime import datetime, timezone

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.models import Bar, Order, OrderType, Side, TimeFrame


def _bar(ts: datetime, close: float = 105.0) -> Bar:
    return Bar(
        timestamp=ts,
        symbol="BTC",
        open=100.0,
        high=110.0,
        low=95.0,
        close=close,
        volume=1_000.0,
        timeframe=TimeFrame.M15,
    )


def _broker_with_open_position() -> SimBroker:
    broker = SimBroker(initial_equity=10_000.0)
    broker.submit_order(Order(
        order_id="",
        symbol="BTC",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        qty=0.1,
        tag="entry",
    ))
    broker.process_bar(_bar(datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)))
    broker.process_bar(_bar(datetime(2026, 5, 24, 12, 15, tzinfo=timezone.utc), close=108.0))
    assert broker.get_position("BTC") is not None
    return broker


def test_terminal_mark_keeps_position_open_and_records_terminal_mark() -> None:
    broker = _broker_with_open_position()

    marks = broker.mark_open_positions()

    assert len(marks) == 1
    assert marks[0].symbol == "BTC"
    assert broker.get_position("BTC") is not None
    assert broker.closed_trades == []
    assert broker.terminal_marks == marks


def test_force_close_realizes_trade_and_removes_position() -> None:
    broker = _broker_with_open_position()

    fills = broker.close_open_positions()

    assert len(fills) == 1
    assert broker.get_position("BTC") is None
    assert len(broker.closed_trades) == 1
    assert broker.terminal_marks == []
    assert broker.closed_trades[0].exit_reason == "backtest_end"
