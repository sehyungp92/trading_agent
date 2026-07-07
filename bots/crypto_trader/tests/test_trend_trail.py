"""Tests for trend trailing stop."""

import pytest
from datetime import datetime, timezone

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.trend.config import TrendTrailParams
from crypto_trader.strategy.trend.trail import TrailManager


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


def _make_ind(atr=200.0):
    return IndicatorSnapshot(
        ema_fast=49000, ema_fast_arr=None,
        ema_mid=48000, ema_mid_arr=None,
        ema_slow=0, ema_slow_arr=None,
        atr=atr, atr_avg=atr,
        rsi=50.0, adx=25.0,
        di_plus=20.0, di_minus=15.0,
        adx_rising=False, volume_ma=100.0,
    )


class TestTrailManager:
    def test_activation_gate_bars(self):
        """Trail activates when bars >= threshold."""
        tm = TrailManager(TrendTrailParams(trail_activation_bars=3, trail_activation_r=10.0))
        bars = [_make_bar(50500)]
        result = tm.update("BTC", Side.LONG, bars, _make_ind(),
                          current_stop=49500, bars_since_entry=3, current_r=0.5, mfe_r=1.0)
        assert result is not None  # Activated by bars

    def test_activation_gate_r(self):
        """Trail activates when current_r >= threshold (OR logic)."""
        tm = TrailManager(TrendTrailParams(trail_activation_bars=100, trail_activation_r=0.5))
        bars = [_make_bar(50500)]
        result = tm.update("BTC", Side.LONG, bars, _make_ind(),
                          current_stop=49500, bars_since_entry=1, current_r=0.6, mfe_r=0.6)
        assert result is not None  # Activated by R

    def test_not_activated_below_thresholds(self):
        """Trail not active when below both thresholds."""
        tm = TrailManager(TrendTrailParams(trail_activation_bars=5, trail_activation_r=1.0))
        bars = [_make_bar(50500)]
        result = tm.update("BTC", Side.LONG, bars, _make_ind(),
                          current_stop=49500, bars_since_entry=2, current_r=0.3, mfe_r=0.3)
        assert result is None

    def test_r_adaptive_buffer(self):
        """Higher R → tighter buffer."""
        tm = TrailManager(TrendTrailParams(
            trail_buffer_wide=1.0, trail_buffer_tight=0.3, trail_r_ceiling=2.0
        ))
        bars = [_make_bar(50500)]
        ind = _make_ind(atr=200.0)

        # Low R → wider buffer
        stop_low_r = tm.update("BTC", Side.LONG, bars, ind,
                               current_stop=None, bars_since_entry=5,
                               current_r=0.5, mfe_r=0.5)
        # High R → tighter buffer
        stop_high_r = tm.update("BTC", Side.LONG, bars, ind,
                                current_stop=None, bars_since_entry=5,
                                current_r=1.8, mfe_r=1.8)

        assert stop_low_r is not None and stop_high_r is not None
        assert stop_high_r > stop_low_r  # Tighter = closer to price for long

    def test_mfe_can_drive_adaptive_buffer(self):
        """Optional MFE mode tightens the buffer after a large peak gives back."""
        bars = [_make_bar(50500)]
        ind = _make_ind(atr=200.0)
        current_r = 0.2
        mfe_r = 1.8

        current_tm = TrailManager(TrendTrailParams(
            trail_buffer_wide=1.0,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_use_mfe_for_adaptive=False,
        ))
        mfe_tm = TrailManager(TrendTrailParams(
            trail_buffer_wide=1.0,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_use_mfe_for_adaptive=True,
        ))

        current_stop = current_tm.update(
            "BTC", Side.LONG, bars, ind,
            current_stop=None, bars_since_entry=8,
            current_r=current_r, mfe_r=mfe_r,
        )
        mfe_stop = mfe_tm.update(
            "BTC", Side.LONG, bars, ind,
            current_stop=None, bars_since_entry=8,
            current_r=current_r, mfe_r=mfe_r,
        )

        assert current_stop is not None and mfe_stop is not None
        assert mfe_stop > current_stop

    def test_stop_never_retreats_long(self):
        """Long trail stop should only advance, never move down."""
        tm = TrailManager(TrendTrailParams())
        bars = [_make_bar(50500)]
        ind = _make_ind()

        # First update sets stop
        stop1 = tm.update("BTC", Side.LONG, bars, ind,
                          current_stop=49800, bars_since_entry=5, current_r=1.0, mfe_r=1.0)

        if stop1 is not None:
            # Next bar lower → should return None (don't retreat)
            bars2 = [_make_bar(50200)]
            stop2 = tm.update("BTC", Side.LONG, bars2, ind,
                              current_stop=stop1, bars_since_entry=6, current_r=0.8, mfe_r=1.0)
            assert stop2 is None or stop2 > stop1

    def test_stop_never_retreats_short(self):
        """Short trail stop should only advance (decrease), never move up."""
        tm = TrailManager(TrendTrailParams())
        bars = [_make_bar(49500)]
        ind = _make_ind()

        stop1 = tm.update("BTC", Side.SHORT, bars, ind,
                          current_stop=50200, bars_since_entry=5, current_r=1.0, mfe_r=1.0)

        if stop1 is not None:
            bars2 = [_make_bar(49800)]
            stop2 = tm.update("BTC", Side.SHORT, bars2, ind,
                              current_stop=stop1, bars_since_entry=6, current_r=0.8, mfe_r=1.0)
            assert stop2 is None or stop2 < stop1

    def test_structure_trail_disabled_by_default(self):
        """Structure trail should not affect result when disabled."""
        tm = TrailManager(TrendTrailParams(structure_trail_enabled=False))
        bars = [_make_bar(50500, low=50200, idx=i) for i in range(10)]
        result = tm.update("BTC", Side.LONG, bars, _make_ind(),
                          current_stop=None, bars_since_entry=5,
                          current_r=1.0, mfe_r=1.0)
        # Should still produce a result from R-adaptive alone
        assert result is not None

    def test_returns_none_zero_atr(self):
        """Zero ATR → None."""
        tm = TrailManager(TrendTrailParams())
        bars = [_make_bar(50500)]
        result = tm.update("BTC", Side.LONG, bars, _make_ind(atr=0),
                          current_stop=None, bars_since_entry=5,
                          current_r=1.0, mfe_r=1.0)
        assert result is None
