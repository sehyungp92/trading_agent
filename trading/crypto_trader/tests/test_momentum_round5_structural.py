"""Focused tests for momentum round-5 structural enablement."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from crypto_trader.broker.sim_broker import SimBroker
from crypto_trader.core.models import Bar, Order, OrderType, SetupGrade, Side, TimeFrame
from crypto_trader.strategy.momentum.config import EntryParams, MomentumConfig
from crypto_trader.strategy.momentum.confirmation import ConfirmationResult
from crypto_trader.strategy.momentum.entry import EntrySignal
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.momentum.setup import SetupResult
from crypto_trader.strategy.momentum.sizing import SizingResult


def _indicators() -> IndicatorSnapshot:
    arr = np.array([100.0])
    return IndicatorSnapshot(
        ema_fast=100.0,
        ema_mid=99.0,
        ema_slow=98.0,
        ema_fast_arr=arr,
        ema_mid_arr=arr,
        ema_slow_arr=arr,
        adx=25.0,
        di_plus=20.0,
        di_minus=15.0,
        adx_rising=True,
        atr=2.0,
        atr_avg=2.0,
        rsi=45.0,
        volume_ma=100.0,
    )


def _setup() -> SetupResult:
    return SetupResult(
        grade=SetupGrade.B,
        zone_price=100.0,
        confluences=("fib_zone", "prior_hl_flip"),
        room_r=2.0,
        projected_r=2.0,
        stop_level=95.0,
        fib_levels={0.5: 100.0},
    )


def _sizing() -> SizingResult:
    return SizingResult(
        qty=1.0,
        leverage=3.0,
        liquidation_price=70.0,
        risk_pct_actual=0.01,
        notional=100.0,
        was_reduced=False,
        reduction_reason=None,
    )


def _confirmation(pattern_type: str, trigger: float = 100.0) -> ConfirmationResult:
    return ConfirmationResult(
        pattern_type=pattern_type,
        trigger_price=trigger,
        bar_index=10,
        volume_confirmed=True,
    )


def _bar(low: float) -> Bar:
    return Bar(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="BTC",
        open=100.0,
        high=101.0,
        low=low,
        close=96.0,
        volume=100.0,
        timeframe=TimeFrame.M15,
    )


def test_momentum_config_roundtrip_includes_new_round5_fields():
    cfg = MomentumConfig()
    assert cfg.confirmation.enforce_volume_on_trigger is False
    assert cfg.entry.mode == "legacy"

    restored = MomentumConfig.from_dict(cfg.to_dict())
    assert restored.confirmation.enforce_volume_on_trigger is False
    assert restored.entry.mode == "legacy"


def test_legacy_break_path_now_uses_real_stop_order_with_ttl():
    signal = EntrySignal(
        EntryParams(
            entry_on_close=False,
            entry_on_break=True,
            max_bars_after_confirmation=2,
            mode="legacy",
        )
    )

    order = signal.generate(
        setup=_setup(),
        confirmation=_confirmation("inside_bar_break", trigger=101.0),
        indicators=_indicators(),
        sizing=_sizing(),
        direction=Side.LONG,
        symbol="BTC",
    )

    assert order is not None
    assert order.order_type == OrderType.STOP
    assert order.stop_price == 101.0
    assert order.ttl_bars == 2
    assert order.metadata["entry_method"] == "break"


def test_confirmation_specific_mode_breaks_only_inside_bar():
    signal = EntrySignal(EntryParams(mode="confirmation_specific", max_bars_after_confirmation=3))

    inside_bar_order = signal.generate(
        setup=_setup(),
        confirmation=_confirmation("inside_bar_break", trigger=102.0),
        indicators=_indicators(),
        sizing=_sizing(),
        direction=Side.LONG,
        symbol="BTC",
    )
    hammer_order = signal.generate(
        setup=_setup(),
        confirmation=_confirmation("hammer", trigger=100.5),
        indicators=_indicators(),
        sizing=_sizing(),
        direction=Side.LONG,
        symbol="BTC",
    )

    assert inside_bar_order is not None
    assert inside_bar_order.order_type == OrderType.STOP
    assert inside_bar_order.metadata["entry_method"] == "break"
    assert hammer_order is not None
    assert hammer_order.order_type == OrderType.MARKET
    assert hammer_order.metadata["entry_method"] == "close"


@pytest.mark.parametrize("tag", ["breakeven_stop", "trailing_stop"])
def test_exit_only_stop_tags_do_not_open_reverse_positions(tag: str):
    broker = SimBroker(
        initial_equity=10_000.0,
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
    )
    broker.submit_order(Order(
        order_id="",
        symbol="BTC",
        side=Side.SHORT,
        order_type=OrderType.STOP,
        qty=1.0,
        stop_price=95.0,
        tag=tag,
    ))

    fills = broker.process_bar(_bar(low=90.0))

    assert fills == []
    assert broker.get_position("BTC") is None
    assert broker.get_open_orders("BTC") == []
