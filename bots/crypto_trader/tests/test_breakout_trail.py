"""Tests for breakout trailing stop."""

from datetime import datetime, timezone

import numpy as np
import pytest

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.breakout.config import BreakoutTrailParams
from crypto_trader.strategy.breakout.trail import TrailManager
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ind(atr=10.0):
    return IndicatorSnapshot(
        ema_fast=100.0,
        ema_mid=99.0,
        ema_slow=98.0,
        ema_fast_arr=None,
        ema_mid_arr=None,
        ema_slow_arr=None,
        adx=25.0,
        di_plus=20.0,
        di_minus=15.0,
        adx_rising=True,
        atr=atr,
        atr_avg=atr,
        rsi=55.0,
        volume_ma=1000.0,
    )


def _bar(close=110.0, high=112.0, low=108.0, symbol="BTC"):
    return Bar(
        timestamp=TS,
        symbol=symbol,
        open=109.0,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        timeframe=TimeFrame.M30,
    )


class TestTrailManager:
    def test_activation_gate_bars(self):
        """Activates after trail_activation_bars."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=3,
            trail_activation_r=999.0,  # Won't trigger via R
        )
        mgr = TrailManager(cfg)
        bar = _bar(close=110.0)
        # Not enough bars
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(), current_stop=90.0,
            bars_since_entry=2, current_r=0.5, mfe_r=1.0,
        )
        assert result is None
        # Enough bars
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(), current_stop=90.0,
            bars_since_entry=3, current_r=0.5, mfe_r=1.0,
        )
        assert result is not None

    def test_activation_gate_r(self):
        """Activates when current_r >= trail_activation_r (OR logic)."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=999,  # Won't trigger via bars
            trail_activation_r=0.5,
        )
        mgr = TrailManager(cfg)
        bar = _bar(close=110.0)
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(), current_stop=90.0,
            bars_since_entry=1, current_r=0.5, mfe_r=1.0,
        )
        assert result is not None

    def test_r_adaptive_formula(self):
        """buffer = wide*(1-r_frac) + tight*r_frac."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=1,
            trail_buffer_wide=2.0,
            trail_buffer_tight=0.5,
            trail_r_ceiling=2.0,
        )
        mgr = TrailManager(cfg)
        atr = 10.0
        bar = _bar(close=120.0)
        # mfe_r=1.5, ceiling=2.0, r_frac=0.75
        # buffer = 2.0*(1-0.75) + 0.5*0.75 = 0.5 + 0.375 = 0.875
        # new_stop = 120 - 10 * 0.875 = 111.25
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(atr=atr), current_stop=90.0,
            bars_since_entry=5, current_r=1.0, mfe_r=1.5,
        )
        assert result == pytest.approx(111.25)

    def test_only_moves_favorably(self):
        """Long stop only moves up; returns None if would move down."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=1,
            trail_buffer_wide=2.0,
            trail_buffer_tight=0.5,
            trail_r_ceiling=2.0,
        )
        mgr = TrailManager(cfg)
        bar = _bar(close=110.0)
        atr = 10.0
        # mfe_r=1.0, ceiling=2.0, r_frac=0.5
        # buffer = 2.0*0.5 + 0.5*0.5 = 1.25
        # new_stop = 110 - 10*1.25 = 97.5
        # current_stop=98 > 97.5 -> should return None
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(atr=atr), current_stop=98.0,
            bars_since_entry=5, current_r=0.5, mfe_r=1.0,
        )
        assert result is None

    def test_trail_activation_uses_mfe_r(self):
        """Trail activates when mfe_r >= threshold even if current_r < threshold."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=999,  # Won't trigger via bars
            trail_activation_r=0.5,
        )
        mgr = TrailManager(cfg)
        bar = _bar(close=110.0)
        # current_r=0.2 (below threshold), mfe_r=0.8 (above threshold)
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(), current_stop=90.0,
            bars_since_entry=1, current_r=0.2, mfe_r=0.8,
        )
        assert result is not None

    def test_trail_no_activation_below_both_thresholds(self):
        """No activation when both mfe_r and bars are below threshold."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=10,
            trail_activation_r=0.5,
        )
        mgr = TrailManager(cfg)
        bar = _bar(close=110.0)
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(), current_stop=90.0,
            bars_since_entry=2, current_r=0.1, mfe_r=0.3,
        )
        assert result is None

    def test_trail_activation_bars_or_gate(self):
        """Trail activates via bars OR-gate even when mfe_r is low."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=3,
            trail_activation_r=999.0,  # Won't trigger via R
        )
        mgr = TrailManager(cfg)
        bar = _bar(close=110.0)
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(), current_stop=90.0,
            bars_since_entry=3, current_r=0.1, mfe_r=0.2,
        )
        assert result is not None

    def test_trail_buffer_uses_mfe_r(self):
        """Buffer calculation uses mfe_r so peak progress tightens the stop."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=1,
            trail_buffer_wide=2.0,
            trail_buffer_tight=0.5,
            trail_r_ceiling=2.0,
        )
        mgr = TrailManager(cfg)
        atr = 10.0
        bar = _bar(close=120.0)
        # mfe_r=2.0, ceiling=2.0, r_frac=1.0
        # buffer = 2.0*(1-1.0) + 0.5*1.0 = 0.5
        # new_stop = 120 - 10 * 0.5 = 115.0
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=[bar],
            m30_ind=_ind(atr=atr), current_stop=90.0,
            bars_since_entry=5, current_r=1.0, mfe_r=2.0,
        )
        assert result == pytest.approx(115.0)

    def test_structure_trail(self):
        """Structure trail uses lookback swing low/high."""
        cfg = BreakoutTrailParams(
            trail_activation_bars=1,
            trail_buffer_wide=5.0,  # Very wide ATR buffer so struct trail is tighter
            trail_buffer_tight=5.0,
            trail_r_ceiling=2.0,
            structure_trail_enabled=True,
            structure_swing_lookback=3,
        )
        mgr = TrailManager(cfg)
        atr = 10.0
        bars = [
            _bar(close=108.0, high=110.0, low=106.0),
            _bar(close=109.0, high=111.0, low=107.0),
            _bar(close=112.0, high=114.0, low=110.0),
        ]
        # R-adaptive: close=112, buffer=5.0*atr=50 -> new_stop = 112-50=62
        # Structure: min(low) of last 3 bars = 106
        # max(r_adaptive=62, struct=106) = 106
        result = mgr.update(
            sym="BTC", direction=Side.LONG, bars=bars,
            m30_ind=_ind(atr=atr), current_stop=90.0,
            bars_since_entry=5, current_r=1.0, mfe_r=1.5,
        )
        assert result == pytest.approx(106.0)
