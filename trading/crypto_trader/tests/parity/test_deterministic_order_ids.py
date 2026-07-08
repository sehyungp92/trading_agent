from __future__ import annotations

from crypto_trader.strategy.breakout.strategy import BreakoutStrategy
from crypto_trader.strategy.trend.strategy import TrendStrategy


def test_trend_management_order_ids_are_deterministic() -> None:
    seed = {
        "fill_id": "fill-1",
        "order_id": "entry-1",
        "timestamp": "2026-05-31T00:00:00+00:00",
        "side": "long",
        "qty": 0.1,
        "stop_price": 99.0,
    }

    first = TrendStrategy._management_order_id("stop", "BTC", seed)
    second = TrendStrategy._management_order_id("stop", "BTC", dict(seed))
    changed = TrendStrategy._management_order_id("stop", "BTC", {**seed, "stop_price": 98.0})

    assert first == second
    assert first != changed
    assert first.startswith("trend_stop_BTC_")


def test_breakout_management_order_ids_are_deterministic() -> None:
    seed = {
        "bar_timestamp": "2026-05-31T00:30:00+00:00",
        "timeframe": "30m",
        "direction": "short",
        "qty": 0.2,
        "stop_price": 101.0,
        "previous_stop_order_id": "brk_stop_BTC_old",
    }

    first = BreakoutStrategy._management_order_id("trail", "ETH", seed)
    second = BreakoutStrategy._management_order_id("trail", "ETH", dict(seed))
    changed = BreakoutStrategy._management_order_id("be", "ETH", seed)

    assert first == second
    assert first != changed
    assert first.startswith("brk_trail_ETH_")
