"""Tests for breakout context (H4 directional bias) analyzer."""

from __future__ import annotations

import pytest

from crypto_trader.core.models import Side
from crypto_trader.strategy.breakout.config import ContextParams
from crypto_trader.strategy.breakout.context import ContextAnalyzer, ContextBias
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_snapshot(**overrides) -> IndicatorSnapshot:
    """Return an IndicatorSnapshot with reasonable defaults."""
    defaults = dict(
        ema_fast=105.0,
        ema_mid=100.0,
        ema_slow=95.0,
        ema_fast_arr=None,
        ema_mid_arr=None,
        ema_slow_arr=None,
        rsi=50.0,
        adx=25.0,
        di_plus=20.0,
        di_minus=15.0,
        atr=2.0,
        volume_ma=1000.0,
        adx_rising=True,
        atr_avg=2.0,
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContextAnalyzer:
    """ContextAnalyzer.evaluate() tests."""

    def test_none_returns_neutral(self):
        """None input (warmup) returns direction=None, strength='none'."""
        analyzer = ContextAnalyzer(ContextParams())
        bias = analyzer.evaluate(None)
        assert bias.direction is None
        assert bias.strength == "none"

    def test_strong_long(self):
        """Full EMA ordering (fast > mid > slow) + ADX >= strong_min_adx -> LONG strong."""
        analyzer = ContextAnalyzer(ContextParams(strong_min_adx=20.0))
        snap = make_snapshot(ema_fast=110, ema_mid=100, ema_slow=90, adx=25)
        bias = analyzer.evaluate(snap)
        assert bias.direction == Side.LONG
        assert bias.strength == "strong"

    def test_strong_short(self):
        """Full EMA ordering (fast < mid < slow) + ADX >= strong_min_adx -> SHORT strong."""
        analyzer = ContextAnalyzer(ContextParams(strong_min_adx=20.0))
        snap = make_snapshot(ema_fast=90, ema_mid=100, ema_slow=110, adx=25)
        bias = analyzer.evaluate(snap)
        assert bias.direction == Side.SHORT
        assert bias.strength == "strong"

    def test_moderate_long(self):
        """ema_fast > ema_mid (not full ordering) + ADX >= h4_adx_threshold -> LONG moderate."""
        analyzer = ContextAnalyzer(ContextParams(h4_adx_threshold=12.0, strong_min_adx=30.0))
        # fast > mid but mid < slow -- breaks full ordering, so not strong
        snap = make_snapshot(ema_fast=105, ema_mid=100, ema_slow=103, adx=15)
        bias = analyzer.evaluate(snap)
        assert bias.direction == Side.LONG
        assert bias.strength == "moderate"

    def test_moderate_short(self):
        """ema_fast < ema_mid + ADX >= threshold -> SHORT moderate."""
        analyzer = ContextAnalyzer(ContextParams(h4_adx_threshold=12.0, strong_min_adx=30.0))
        # fast < mid but mid > slow -- breaks full ordering
        snap = make_snapshot(ema_fast=95, ema_mid=100, ema_slow=103, adx=15)
        bias = analyzer.evaluate(snap)
        assert bias.direction == Side.SHORT
        assert bias.strength == "moderate"

    def test_no_direction_low_adx(self):
        """EMAs ordered but ADX below both thresholds -> None direction."""
        analyzer = ContextAnalyzer(ContextParams(h4_adx_threshold=20.0, strong_min_adx=30.0))
        snap = make_snapshot(ema_fast=110, ema_mid=100, ema_slow=90, adx=10)
        bias = analyzer.evaluate(snap)
        assert bias.direction is None
        assert bias.strength == "none"

    def test_flat_emas(self):
        """ema_fast == ema_mid -> None direction (no fast vs mid edge)."""
        analyzer = ContextAnalyzer(ContextParams())
        snap = make_snapshot(ema_fast=100, ema_mid=100, ema_slow=95, adx=25)
        bias = analyzer.evaluate(snap)
        assert bias.direction is None
        assert bias.strength == "none"

    def test_custom_thresholds(self):
        """Different ContextParams values produce correct results."""
        # With very low thresholds, even weak ADX should produce strong
        cfg = ContextParams(h4_adx_threshold=5.0, strong_min_adx=10.0)
        analyzer = ContextAnalyzer(cfg)
        snap = make_snapshot(ema_fast=110, ema_mid=100, ema_slow=90, adx=12)
        bias = analyzer.evaluate(snap)
        assert bias.direction == Side.LONG
        assert bias.strength == "strong"

        # With very high thresholds, strong ADX is still not enough
        cfg2 = ContextParams(h4_adx_threshold=50.0, strong_min_adx=60.0)
        analyzer2 = ContextAnalyzer(cfg2)
        snap2 = make_snapshot(ema_fast=110, ema_mid=100, ema_slow=90, adx=40)
        bias2 = analyzer2.evaluate(snap2)
        assert bias2.direction is None
        assert bias2.strength == "none"
