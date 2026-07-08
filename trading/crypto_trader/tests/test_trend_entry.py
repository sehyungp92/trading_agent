"""Tests for trend entry order generation."""

import pytest

from crypto_trader.core.models import Bar, OrderType, SetupGrade, Side, TimeFrame
from crypto_trader.strategy.trend.config import TrendEntryParams
from crypto_trader.strategy.trend.confirmation import TriggerResult
from crypto_trader.strategy.trend.entry import EntryGenerator
from crypto_trader.strategy.trend.setup import TrendSetupResult
from crypto_trader.strategy.trend.sizing import SizingResult
from datetime import datetime, timezone


def _bar():
    return Bar(
        timestamp=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
        symbol="BTC", open=50000, high=50100, low=49900,
        close=50050, volume=100.0, timeframe=TimeFrame.H1,
    )


def _setup(grade=SetupGrade.B):
    return TrendSetupResult(
        grade=grade, direction=Side.LONG,
        impulse_start=49000, impulse_end=50500,
        impulse_atr_move=2.0, pullback_depth=0.3,
        confluences=("h1_ema_zone", "rsi_pullback"),
        zone_price=50050, room_r=2.5, stop_level=49500,
    )


def _sizing():
    return SizingResult(
        qty=0.1, leverage=5.0, liquidation_price=40000,
        risk_pct_actual=0.005, notional=5000,
        was_reduced=False, reduction_reason=None,
    )


def _trigger():
    return TriggerResult("engulfing", 50050, 0, True)


class TestEntryGenerator:
    def test_aggressive_market_entry(self):
        gen = EntryGenerator(TrendEntryParams(entry_on_close=True))
        order = gen.generate(_bar(), Side.LONG, 0.1, _sizing(), _setup(), _trigger(),
                            "BTC", "test_1")
        assert order is not None
        assert order.order_type == OrderType.MARKET
        assert order.qty == 0.1
        assert order.tag == "entry"
        assert order.metadata["entry_method"] == "aggressive"

    def test_conservative_stop_entry(self):
        gen = EntryGenerator(TrendEntryParams(entry_on_close=False, entry_on_break=True))
        trigger = _trigger()
        order = gen.generate(_bar(), Side.LONG, 0.1, _sizing(), _setup(), trigger,
                            "BTC", "test_2")
        assert order is not None
        assert order.order_type == OrderType.STOP
        assert order.stop_price == trigger.trigger_price
        assert order.metadata["entry_method"] == "conservative"

    def test_conservative_without_trigger_returns_none(self):
        gen = EntryGenerator(TrendEntryParams(entry_on_close=False, entry_on_break=True))
        order = gen.generate(_bar(), Side.LONG, 0.1, _sizing(), _setup(), None,
                            "BTC", "test_3")
        assert order is None

    def test_metadata_populated(self):
        gen = EntryGenerator(TrendEntryParams())
        order = gen.generate(_bar(), Side.LONG, 0.1, _sizing(), _setup(), _trigger(),
                            "BTC", "test_4")
        assert order is not None
        assert order.metadata["setup_grade"] == "B"
        assert "confluences" in order.metadata
        assert order.metadata["room_r"] == 2.5

    def test_hybrid_grade_uses_market_for_a_setups(self):
        gen = EntryGenerator(TrendEntryParams(mode="hybrid_grade"))
        order = gen.generate(_bar(), Side.LONG, 0.1, _sizing(), _setup(SetupGrade.A), _trigger(),
                            "BTC", "test_5")
        assert order is not None
        assert order.order_type == OrderType.MARKET
        assert order.metadata["entry_method"] == "aggressive"

    def test_hybrid_grade_uses_break_for_b_setups(self):
        gen = EntryGenerator(TrendEntryParams(mode="hybrid_grade"))
        trigger = _trigger()
        order = gen.generate(_bar(), Side.LONG, 0.1, _sizing(), _setup(SetupGrade.B), trigger,
                            "BTC", "test_6")
        assert order is not None
        assert order.order_type == OrderType.STOP
        assert order.stop_price == trigger.trigger_price
        assert order.metadata["entry_method"] == "conservative"
