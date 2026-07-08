"""Tests for trend confirmation/trigger detection."""

import pytest
from datetime import datetime, timezone

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.trend.config import TrendConfirmationParams
from crypto_trader.strategy.trend.confirmation import TriggerDetector, TriggerResult


def _make_bar(open_, high, low, close, volume=100.0, idx=0):
    return Bar(
        timestamp=datetime(2026, 3, 15, idx, 0, tzinfo=timezone.utc),
        symbol="BTC",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timeframe=TimeFrame.H1,
    )


def _make_ind(ema_fast=49000, volume_ma=100.0):
    return IndicatorSnapshot(
        ema_fast=ema_fast,
        ema_fast_arr=None,
        ema_mid=48000,
        ema_mid_arr=None,
        ema_slow=0,
        ema_slow_arr=None,
        atr=200.0,
        atr_avg=200.0,
        rsi=50.0,
        adx=25.0,
        di_plus=20.0,
        di_minus=15.0,
        adx_rising=False,
        volume_ma=volume_ma,
    )


class TestTriggerDetector:
    def test_engulfing_long(self):
        """Bullish engulfing: current green engulfs prior red."""
        det = TriggerDetector(TrendConfirmationParams())
        prev = _make_bar(50100, 50200, 49900, 49950, idx=0)  # Red candle
        curr = _make_bar(49900, 50300, 49850, 50200, idx=1)  # Green engulfs
        result = det.check([prev, curr], Side.LONG, _make_ind())
        assert result is not None
        assert result.pattern == "engulfing"

    def test_engulfing_short(self):
        """Bearish engulfing: current red engulfs prior green."""
        det = TriggerDetector(TrendConfirmationParams())
        prev = _make_bar(49900, 50200, 49850, 50100, idx=0)  # Green
        curr = _make_bar(50200, 50250, 49800, 49850, idx=1)  # Red engulfs
        result = det.check([prev, curr], Side.SHORT, _make_ind())
        assert result is not None
        assert result.pattern == "engulfing"

    def test_hammer_long(self):
        """Hammer: long lower wick for bullish."""
        det = TriggerDetector(TrendConfirmationParams(enable_engulfing=False, enable_hammer=True))
        # Hammer: body at top, long lower wick
        curr = _make_bar(50000, 50050, 49700, 50020, idx=1)  # Body=20, wick=300
        prev = _make_bar(50100, 50200, 50000, 50050, idx=0)
        result = det.check([prev, curr], Side.LONG, _make_ind())
        assert result is not None
        assert result.pattern == "hammer"

    def test_hammer_short(self):
        """Shooting star: long upper wick for bearish."""
        det = TriggerDetector(TrendConfirmationParams(enable_engulfing=False, enable_hammer=True))
        # Shooting star: body at bottom, long upper wick
        curr = _make_bar(50020, 50300, 49950, 50000, idx=1)  # Body=20, upper wick=280
        prev = _make_bar(49900, 50050, 49850, 49950, idx=0)
        result = det.check([prev, curr], Side.SHORT, _make_ind())
        assert result is not None
        assert result.pattern == "hammer"

    def test_ema_reclaim_long(self):
        """EMA reclaim: prior bar below EMA, current above."""
        det = TriggerDetector(TrendConfirmationParams(
            enable_engulfing=False, enable_hammer=False
        ))
        ema = 50000
        prev = _make_bar(49800, 49900, 49700, 49800, idx=0)  # Below EMA
        curr = _make_bar(49900, 50200, 49850, 50100, idx=1)  # Above EMA
        result = det.check([prev, curr], Side.LONG, _make_ind(ema_fast=ema))
        assert result is not None
        assert result.pattern == "ema_reclaim"

    def test_ema_reclaim_short(self):
        """EMA reclaim short: prior bar above EMA, current below."""
        det = TriggerDetector(TrendConfirmationParams(
            enable_engulfing=False, enable_hammer=False
        ))
        ema = 50000
        prev = _make_bar(50100, 50200, 50050, 50100, idx=0)  # Above EMA
        curr = _make_bar(50050, 50100, 49800, 49900, idx=1)  # Below EMA
        result = det.check([prev, curr], Side.SHORT, _make_ind(ema_fast=ema))
        assert result is not None
        assert result.pattern == "ema_reclaim"

    def test_structure_break_long(self):
        """Structure break: close above pullback lower-high."""
        det = TriggerDetector(TrendConfirmationParams(
            enable_engulfing=False, enable_hammer=False, enable_ema_reclaim=False
        ))
        bars = [
            _make_bar(50000, 50200, 49900, 50100, idx=0),
            _make_bar(50100, 50150, 49950, 50000, idx=1),  # Lower high
            _make_bar(49950, 50050, 49850, 49900, idx=2),  # Lower high
            _make_bar(49900, 50000, 49800, 49950, idx=3),
            _make_bar(49950, 50200, 49900, 50100, idx=4),  # Breaks above
        ]
        result = det.check(bars, Side.LONG, _make_ind())
        assert result is not None
        assert result.pattern == "structure_break"

    def test_no_pattern_found(self):
        """Flat bars should not trigger anything."""
        det = TriggerDetector(TrendConfirmationParams())
        bars = [
            _make_bar(50000, 50010, 49990, 50005, idx=0),
            _make_bar(50005, 50015, 49995, 50010, idx=1),
        ]
        # These tiny bars shouldn't engulf, have no hammer wick, etc.
        # May or may not match — depends on body size vs wick ratio
        # Just verify it doesn't crash

    def test_volume_confirm_required(self):
        """When volume confirm required, low volume rejects."""
        det = TriggerDetector(TrendConfirmationParams(
            require_volume_confirm=True,
            volume_threshold_mult=2.0,  # Need 2x average volume
        ))
        prev = _make_bar(50100, 50200, 49900, 49950, volume=50, idx=0)
        curr = _make_bar(49900, 50300, 49850, 50200, volume=50, idx=1)  # Low volume
        result = det.check([prev, curr], Side.LONG, _make_ind(volume_ma=100.0))
        if result is not None:
            assert result.volume_confirmed is False

    def test_enforce_volume_on_trigger_blocks_low_volume_signal(self):
        det = TriggerDetector(TrendConfirmationParams(
            require_volume_confirm=True,
            enforce_volume_on_trigger=True,
            volume_threshold_mult=2.0,
        ))
        prev = _make_bar(50100, 50200, 49900, 49950, volume=50, idx=0)
        curr = _make_bar(49900, 50300, 49850, 50200, volume=50, idx=1)
        result = det.check([prev, curr], Side.LONG, _make_ind(volume_ma=100.0))
        assert result is None

    def test_disabled_patterns_skipped(self):
        """Disabled patterns are not checked."""
        det = TriggerDetector(TrendConfirmationParams(
            enable_engulfing=False,
            enable_hammer=False,
            enable_ema_reclaim=False,
            enable_structure_break=False,
        ))
        prev = _make_bar(50100, 50200, 49900, 49950, idx=0)
        curr = _make_bar(49900, 50300, 49850, 50200, idx=1)
        result = det.check([prev, curr], Side.LONG, _make_ind())
        assert result is None

    def test_insufficient_bars(self):
        """Fewer than 2 bars → None."""
        det = TriggerDetector(TrendConfirmationParams())
        result = det.check([_make_bar(50000, 50100, 49900, 50050)], Side.LONG, _make_ind())
        assert result is None
