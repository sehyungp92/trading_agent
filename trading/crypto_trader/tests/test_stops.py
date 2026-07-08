"""Tests for stop placement — structural stops with ATR buffer and minimum distance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.momentum.config import StopParams
from crypto_trader.strategy.momentum.stops import StopPlacer
from tests.conftest import make_bar


def _make_bars(close: float, swing_low: float, swing_high: float, count: int = 12) -> list[Bar]:
    """Create bars with a clear swing low/high for structural stop detection."""
    bars = []
    base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(count):
        ts = base_ts + timedelta(minutes=15 * i)
        if i == count // 2:
            # Place the swing point in the middle
            bars.append(make_bar(ts, close, swing_high, swing_low, close))
        else:
            bars.append(make_bar(ts, close, close + 10, close - 10, close))
    return bars


class TestMinStopDistance:
    def test_min_stop_disabled_by_default(self):
        """With min_stop_atr_mult=0, no enforcement happens."""
        params = StopParams(min_stop_atr_mult=0.0)
        placer = StopPlacer(params)
        # Swing low very close to entry
        bars = _make_bars(close=50000.0, swing_low=49990.0, swing_high=50010.0)
        stop = placer.compute(bars, Side.LONG, atr=500.0)
        # Stop should be at structural level minus buffer, no minimum enforcement
        assert stop < 50000.0

    def test_min_stop_enforced_long(self):
        """When structural stop is too close, min_stop_atr_mult widens it."""
        params = StopParams(min_stop_atr_mult=1.0, atr_buffer_mult=0.3)
        placer = StopPlacer(params)
        # Swing low very close to entry → structural stop very tight
        bars = _make_bars(close=50000.0, swing_low=49990.0, swing_high=50010.0)
        atr = 500.0
        stop = placer.compute(bars, Side.LONG, atr=atr)
        # Minimum distance = 1.0 * 500 = 500. Entry = 50000.
        # Stop must be at most 50000 - 500 = 49500
        assert stop <= 50000.0 - atr * 1.0

    def test_min_stop_enforced_short(self):
        """Min stop enforcement works for short direction too."""
        params = StopParams(min_stop_atr_mult=1.0, atr_buffer_mult=0.3)
        placer = StopPlacer(params)
        # Swing high very close to entry → structural stop very tight
        bars = _make_bars(close=50000.0, swing_low=49990.0, swing_high=50010.0)
        atr = 500.0
        stop = placer.compute(bars, Side.SHORT, atr=atr)
        # Minimum distance = 1.0 * 500 = 500. Entry = 50000.
        # Stop must be at least 50000 + 500 = 50500
        assert stop >= 50000.0 + atr * 1.0

    def test_min_stop_no_effect_when_already_wide(self):
        """When structural stop is already wider than minimum, no change."""
        params = StopParams(min_stop_atr_mult=0.5, atr_buffer_mult=0.3)
        placer = StopPlacer(params)
        # Swing low far from entry → structural stop already wide
        bars = _make_bars(close=50000.0, swing_low=49000.0, swing_high=51000.0)
        atr = 500.0
        stop_with_min = placer.compute(bars, Side.LONG, atr=atr)

        params_no_min = StopParams(min_stop_atr_mult=0.0, atr_buffer_mult=0.3)
        placer_no_min = StopPlacer(params_no_min)
        stop_without_min = placer_no_min.compute(bars, Side.LONG, atr=atr)

        # Both should be the same since structural stop is already > 0.5 ATR
        assert stop_with_min == stop_without_min
