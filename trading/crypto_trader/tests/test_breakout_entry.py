"""Tests for breakout entry order generation."""

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Bar, Order, OrderType, SetupGrade, Side, TimeFrame
from crypto_trader.strategy.breakout.balance import BalanceZone
from crypto_trader.strategy.breakout.config import BreakoutEntryParams
from crypto_trader.strategy.breakout.confirmation import BreakoutConfirmation
from crypto_trader.strategy.breakout.entry import EntryGenerator
from crypto_trader.strategy.breakout.setup import BreakoutSetupResult
from crypto_trader.strategy.breakout.sizing import SizingResult

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _zone(center=100.0, upper=105.0, lower=95.0):
    return BalanceZone(
        center=center,
        upper=upper,
        lower=lower,
        bars_in_zone=10,
        touches=3,
        formation_bar_idx=0,
        volume_contracting=False,
        width_atr=1.0,
    )


def _setup(direction=Side.LONG, grade=SetupGrade.A, is_a_plus=False, zone=None):
    z = zone or _zone()
    return BreakoutSetupResult(
        grade=grade,
        is_a_plus=is_a_plus,
        direction=direction,
        balance_zone=z,
        breakout_price=106.0 if direction == Side.LONG else 94.0,
        lvn_runway_atr=2.0,
        confluences=("h4_alignment", "volume_surge"),
        room_r=2.5,
        volume_mult=1.5,
        body_ratio=0.7,
    )


def _confirmation(model="model1_close"):
    return BreakoutConfirmation(
        model=model,
        trigger_price=106.0,
        bar_index=5,
        volume_confirmed=True,
    )


def _sizing():
    return SizingResult(
        qty=0.5,
        leverage=5.0,
        liquidation_price=85.0,
        risk_pct_actual=0.007,
        notional=5000.0,
        was_reduced=False,
        reduction_reason=None,
    )


def _bar(close=106.0, symbol="BTC"):
    return Bar(
        timestamp=TS,
        symbol=symbol,
        open=104.0,
        high=107.0,
        low=103.0,
        close=close,
        volume=1000.0,
        timeframe=TimeFrame.M30,
    )


class TestEntryGenerator:
    def test_model1_market_order(self):
        gen = EntryGenerator(BreakoutEntryParams())
        order = gen.generate(
            bar=_bar(),
            direction=Side.LONG,
            qty=0.5,
            sizing_result=_sizing(),
            setup=_setup(),
            confirmation=_confirmation("model1_close"),
            symbol="BTC",
            order_id="e1",
        )
        assert order is not None
        assert order.order_type == OrderType.MARKET
        assert order.qty == 0.5

    def test_model2_market_on_close(self):
        cfg = BreakoutEntryParams(model2_entry_on_close=True, model2_entry_on_break=False)
        gen = EntryGenerator(cfg)
        order = gen.generate(
            bar=_bar(),
            direction=Side.LONG,
            qty=0.5,
            sizing_result=_sizing(),
            setup=_setup(),
            confirmation=_confirmation("model2_retest"),
            symbol="BTC",
            order_id="e2",
        )
        assert order is not None
        assert order.order_type == OrderType.MARKET

    def test_model2_stop_on_break(self):
        cfg = BreakoutEntryParams(model2_entry_on_close=False, model2_entry_on_break=True)
        gen = EntryGenerator(cfg)
        setup = _setup(direction=Side.LONG)
        order = gen.generate(
            bar=_bar(),
            direction=Side.LONG,
            qty=0.5,
            sizing_result=_sizing(),
            setup=setup,
            confirmation=_confirmation("model2_retest"),
            symbol="BTC",
            order_id="e3",
        )
        assert order is not None
        assert order.order_type == OrderType.STOP
        assert order.stop_price == setup.balance_zone.upper

    def test_order_tag_is_entry(self):
        gen = EntryGenerator(BreakoutEntryParams())
        order = gen.generate(
            bar=_bar(),
            direction=Side.LONG,
            qty=0.5,
            sizing_result=_sizing(),
            setup=_setup(),
            confirmation=_confirmation("model1_close"),
            symbol="BTC",
            order_id="e4",
        )
        assert order is not None
        assert order.tag == "entry"

    def test_model1_disabled(self):
        cfg = BreakoutEntryParams(model1_entry_on_close=False)
        gen = EntryGenerator(cfg)
        order = gen.generate(
            bar=_bar(),
            direction=Side.LONG,
            qty=0.5,
            sizing_result=_sizing(),
            setup=_setup(),
            confirmation=_confirmation("model1_close"),
            symbol="BTC",
            order_id="e5",
        )
        assert order is None

    def test_metadata_contains_fields(self):
        gen = EntryGenerator(BreakoutEntryParams())
        order = gen.generate(
            bar=_bar(),
            direction=Side.LONG,
            qty=0.5,
            sizing_result=_sizing(),
            setup=_setup(grade=SetupGrade.A),
            confirmation=_confirmation("model1_close"),
            symbol="BTC",
            order_id="e6",
        )
        assert order is not None
        meta = order.metadata
        assert "setup_grade" in meta
        assert meta["setup_grade"] == "A"
        assert "confluences" in meta
        assert "leverage" in meta
        assert meta["leverage"] == 5.0
