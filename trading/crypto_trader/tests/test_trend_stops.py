"""Tests for trend stop placement."""

import pytest
from datetime import datetime, timezone

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.trend.config import TrendStopParams
from crypto_trader.strategy.trend.stops import StopPlacer


def _make_bar(close, high=None, low=None, idx=0):
    return Bar(
        timestamp=datetime(2026, 3, 15, idx, 0, tzinfo=timezone.utc),
        symbol="BTC",
        open=close,
        high=high or close + 50,
        low=low or close - 50,
        close=close,
        volume=100.0,
        timeframe=TimeFrame.H1,
    )


class TestStopPlacer:
    def test_swing_based_long(self):
        """Long stop should be below recent swing low."""
        sp = StopPlacer(TrendStopParams(use_swing=True, use_farther=False))
        bars = [_make_bar(50000 + i * 10, low=49900 + i * 10, idx=i) for i in range(10)]
        stop = sp.compute(bars, Side.LONG, atr=200.0, entry_price=50100)
        assert stop < 49900  # Below lowest bar's low

    def test_atr_based_long(self):
        """ATR stop = entry - atr_mult * atr."""
        sp = StopPlacer(TrendStopParams(use_swing=False, atr_mult=1.5))
        bars = [_make_bar(50000, idx=i) for i in range(5)]
        stop = sp.compute(bars, Side.LONG, atr=200.0, entry_price=50000)
        # Should be approximately 50000 - 1.5 * 200 = 49700
        assert abs(stop - 49700) < 100

    def test_farther_stop_selection(self):
        """use_farther=True takes the more generous (farther) stop."""
        sp = StopPlacer(TrendStopParams(use_swing=True, use_farther=True, atr_mult=1.3))
        # Swing low is closer than ATR stop
        bars = [_make_bar(50000, low=49850, idx=i) for i in range(10)]
        stop = sp.compute(bars, Side.LONG, atr=200.0, entry_price=50000)
        atr_stop = 50000 - 1.3 * 200  # 49740
        # Farther = min(swing_low, atr_stop) for long
        assert stop <= min(49850, atr_stop) + 50  # Allow for buffer

    def test_short_stop_above(self):
        """Short stop should be above entry."""
        sp = StopPlacer(TrendStopParams(atr_mult=1.3))
        bars = [_make_bar(50000, high=50100, idx=i) for i in range(10)]
        stop = sp.compute(bars, Side.SHORT, atr=200.0, entry_price=50000)
        assert stop > 50000

    def test_min_stop_distance_enforced(self):
        """Stop must be at least min_stop_atr * atr away."""
        sp = StopPlacer(TrendStopParams(
            use_swing=False, atr_mult=0.5, min_stop_atr=2.0
        ))
        bars = [_make_bar(50000, idx=i) for i in range(5)]
        stop = sp.compute(bars, Side.LONG, atr=200.0, entry_price=50000)
        distance = abs(50000 - stop)
        assert distance >= 2.0 * 200 - 1  # Allow tiny float error

    def test_buffer_applied(self):
        """Buffer percentage should push stop further away."""
        sp1 = StopPlacer(TrendStopParams(use_swing=False, atr_mult=1.0, buffer_pct=0.0))
        sp2 = StopPlacer(TrendStopParams(use_swing=False, atr_mult=1.0, buffer_pct=0.01))
        bars = [_make_bar(50000, idx=i) for i in range(5)]
        stop1 = sp1.compute(bars, Side.LONG, atr=200.0, entry_price=50000)
        stop2 = sp2.compute(bars, Side.LONG, atr=200.0, entry_price=50000)
        assert stop2 < stop1  # Buffer makes stop farther for long
