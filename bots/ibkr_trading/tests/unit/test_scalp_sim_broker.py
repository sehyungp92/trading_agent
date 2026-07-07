from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backtests.scalp.engine.sim_broker import FillStatus, OrderSide, OrderType, SimBroker, SimOrder


def test_scalp_broker_prevents_same_bar_entry_fill() -> None:
    broker = SimBroker()
    ts = datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc)
    broker.submit_order(
        SimOrder(
            order_id="entry",
            symbol="NQ",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            qty=1,
            stop_price=100.0,
            submit_time=ts,
            earliest_fill_time=ts,
            tag="entry",
        )
    )

    same_bar = broker.process_bar("NQ", ts, 99, 101, 98, 100)
    next_bar = broker.process_bar("NQ", ts + timedelta(minutes=1), 99, 101, 98, 100)

    assert same_bar == []
    assert next_bar[0].status is FillStatus.FILLED


def test_scalp_broker_assumes_stop_first_when_oca_stop_and_target_touch_same_bar() -> None:
    broker = SimBroker()
    ts = datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc)
    broker.submit_order(SimOrder("stop", "NQ", OrderSide.SELL, OrderType.STOP, 1, stop_price=99, submit_time=ts, earliest_fill_time=ts, tag="stop", oca_group="g"))
    broker.submit_order(SimOrder("target", "NQ", OrderSide.SELL, OrderType.LIMIT, 1, limit_price=101, submit_time=ts, earliest_fill_time=ts, tag="target", oca_group="g"))

    fills = broker.process_bar("NQ", ts + timedelta(minutes=1), 100, 102, 98, 100)

    assert len(fills) == 1
    assert fills[0].order.order_id == "stop"

