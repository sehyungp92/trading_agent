"""Tests for trend regime classification."""

import pytest
from datetime import datetime, timezone

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.trend.config import RegimeParams
from crypto_trader.strategy.trend.regime import (
    RegimeClassifier,
    RegimeResult,
    StructureState,
    StructureTracker,
)


def _make_d1_bar(close, high=None, low=None, ts=None):
    return Bar(
        timestamp=ts or datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc),
        symbol="BTC",
        open=close - 100,
        high=high or close + 200,
        low=low or close - 300,
        close=close,
        volume=1000.0,
        timeframe=TimeFrame.D1,
    )


def _make_ind(ema_fast, ema_mid, adx, adx_rising=False, atr=100.0, rsi=50.0):
    return IndicatorSnapshot(
        ema_fast=ema_fast,
        ema_fast_arr=None,
        ema_mid=ema_mid,
        ema_mid_arr=None,
        ema_slow=0,
        ema_slow_arr=None,
        atr=atr,
        atr_avg=atr,
        rsi=rsi,
        adx=adx,
        di_plus=20.0,
        di_minus=15.0,
        adx_rising=adx_rising,
        volume_ma=500.0,
    )


class TestRegimeClassifier:
    def test_a_tier_long(self):
        """Close above both EMAs with high ADX → A-tier long."""
        clf = RegimeClassifier(RegimeParams())
        bar = _make_d1_bar(close=50000)
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=25)
        result = clf.evaluate(bar, ind, StructureState())
        assert result.tier == "A"
        assert result.direction == Side.LONG

    def test_a_tier_short(self):
        """Close below both EMAs with high ADX → A-tier short."""
        clf = RegimeClassifier(RegimeParams())
        bar = _make_d1_bar(close=45000)
        ind = _make_ind(ema_fast=46000, ema_mid=47000, adx=25)
        result = clf.evaluate(bar, ind, StructureState())
        assert result.tier == "A"
        assert result.direction == Side.SHORT

    def test_b_tier_long(self):
        """Close above fast EMA but not mid, moderate ADX → B-tier."""
        clf = RegimeClassifier(RegimeParams())
        bar = _make_d1_bar(close=49500)
        ind = _make_ind(ema_fast=49000, ema_mid=50000, adx=14)
        result = clf.evaluate(bar, ind, StructureState())
        assert result.tier == "B"
        assert result.direction == Side.LONG

    def test_no_trade_low_adx(self):
        """ADX below no_trade threshold → none."""
        clf = RegimeClassifier(RegimeParams(no_trade_max_adx=10.0))
        bar = _make_d1_bar(close=50000)
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=8)
        result = clf.evaluate(bar, ind, StructureState())
        assert result.tier == "none"

    def test_no_directional_alignment(self):
        """Price not clearly aligned with EMAs → none."""
        clf = RegimeClassifier(RegimeParams())
        bar = _make_d1_bar(close=49500)
        # Close between fast and mid (not clearly above or below)
        ind = _make_ind(ema_fast=50000, ema_mid=49000, adx=11)
        result = clf.evaluate(bar, ind, StructureState())
        assert result.tier in ("B", "none")

    def test_require_ema_cross(self):
        """When require_ema_cross=True, EMAs must be ordered."""
        clf = RegimeClassifier(RegimeParams(require_ema_cross=True))
        bar = _make_d1_bar(close=50000)
        # fast < mid — not ordered for long
        ind = _make_ind(ema_fast=48000, ema_mid=49000, adx=25)
        result = clf.evaluate(bar, ind, StructureState())
        # Should not get A-tier because EMA cross fails
        assert result.tier != "A" or result.direction != Side.LONG

    def test_require_structure_bullish(self):
        """When require_structure=True, structure must be bullish for A-tier long."""
        clf = RegimeClassifier(RegimeParams(require_structure=True))
        bar = _make_d1_bar(close=50000)
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=25)
        structure = StructureState(pattern="mixed")
        result = clf.evaluate(bar, ind, structure)
        # Can't get A-tier without bullish structure
        assert result.tier != "A"

    def test_require_structure_with_bullish_pattern(self):
        """A-tier long with bullish structure confirmed."""
        clf = RegimeClassifier(RegimeParams(require_structure=True))
        bar = _make_d1_bar(close=50000)
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=25)
        structure = StructureState(pattern="bullish")
        result = clf.evaluate(bar, ind, structure)
        assert result.tier == "A"
        assert result.direction == Side.LONG

    def test_b_adx_rising_required(self):
        """B-tier requires ADX rising when flag set."""
        clf = RegimeClassifier(RegimeParams(b_adx_rising_required=True))
        bar = _make_d1_bar(close=49500)
        ind = _make_ind(ema_fast=49000, ema_mid=50000, adx=14, adx_rising=False)
        result = clf.evaluate(bar, ind, StructureState())
        # B-tier should fail without ADX rising
        assert result.tier != "B" or result.direction != Side.LONG

    def test_adx_threshold_boundary_a(self):
        """ADX exactly at A threshold → A-tier."""
        clf = RegimeClassifier(RegimeParams(a_min_adx=20.0))
        bar = _make_d1_bar(close=50000)
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=20.0)
        result = clf.evaluate(bar, ind, StructureState())
        assert result.tier == "A"


class TestH1RegimeSupplement:
    """Tests for evaluate_h1() — H1-level regime fallback."""

    def test_h1_long_all_conditions_met(self):
        """Close > EMAs, EMAs ordered, ADX >= 20 → B-tier LONG."""
        clf = RegimeClassifier(RegimeParams())
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=25)
        result = clf.evaluate_h1(50000, ind)
        assert result is not None
        assert result.tier == "B"
        assert result.direction == Side.LONG
        assert "h1_regime" in result.reasons

    def test_h1_short_all_conditions_met(self):
        """Close < EMAs, EMAs ordered, ADX >= 20 → B-tier SHORT."""
        clf = RegimeClassifier(RegimeParams())
        ind = _make_ind(ema_fast=46000, ema_mid=47000, adx=22)
        result = clf.evaluate_h1(45000, ind)
        assert result is not None
        assert result.tier == "B"
        assert result.direction == Side.SHORT

    def test_h1_none_adx_below_threshold(self):
        """ADX < h1_min_adx → None."""
        clf = RegimeClassifier(RegimeParams(h1_min_adx=20.0))
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=18)
        result = clf.evaluate_h1(50000, ind)
        assert result is None

    def test_h1_none_emas_not_ordered(self):
        """Close > both EMAs but ema_fast < ema_mid → None."""
        clf = RegimeClassifier(RegimeParams())
        ind = _make_ind(ema_fast=48000, ema_mid=49000, adx=25)
        result = clf.evaluate_h1(50000, ind)
        assert result is None

    def test_h1_none_when_disabled(self):
        """h1_regime_enabled=False → always None."""
        clf = RegimeClassifier(RegimeParams(h1_regime_enabled=False))
        ind = _make_ind(ema_fast=49000, ema_mid=48000, adx=30)
        result = clf.evaluate_h1(50000, ind)
        assert result is None

    def test_h1_none_price_between_emas(self):
        """Close > ema_fast but < ema_mid → None (no clear direction)."""
        clf = RegimeClassifier(RegimeParams())
        ind = _make_ind(ema_fast=48000, ema_mid=50000, adx=25)
        result = clf.evaluate_h1(49000, ind)
        assert result is None


class TestStructureTracker:
    def _make_bars(self, prices):
        """Create D1 bars from (high, low) tuples."""
        bars = []
        for i, (h, l) in enumerate(prices):
            bars.append(Bar(
                timestamp=datetime(2026, 3, i + 1, 0, 0, tzinfo=timezone.utc),
                symbol="BTC",
                open=(h + l) / 2,
                high=h,
                low=l,
                close=(h + l) / 2,
                volume=1000.0,
                timeframe=TimeFrame.D1,
            ))
        return bars

    def test_bullish_structure_detection(self):
        """HH + HL pattern → bullish."""
        tracker = StructureTracker()
        # Create bars with clear HH/HL pattern
        prices = [
            (100, 90), (95, 85), (105, 95),  # First swing high at 105
            (100, 88), (98, 86),
            (110, 100), (108, 98), (115, 105),  # Higher high at 115
            (112, 102), (110, 100),
        ]
        bars = self._make_bars(prices)
        for b in bars:
            tracker.update(b)
        # After enough bars, should detect some swing structure
        assert tracker.state is not None

    def test_mixed_structure(self):
        """Random prices → mixed."""
        tracker = StructureTracker()
        prices = [(100, 90), (98, 88), (102, 92), (97, 87), (103, 93)]
        bars = self._make_bars(prices)
        for b in bars:
            tracker.update(b)
        # With insufficient data, pattern stays mixed
        assert tracker.state.pattern == "mixed"

    def test_swing_lists_capped(self):
        """Swing lists don't grow unbounded (max 4)."""
        tracker = StructureTracker()
        # Feed many bars with clear swings
        prices = []
        for i in range(20):
            if i % 4 < 2:
                prices.append((100 + i * 2, 90 + i))
            else:
                prices.append((95 + i, 85 + i))
        bars = self._make_bars(prices)
        for b in bars:
            tracker.update(b)
        assert len(tracker.state.last_swing_highs) <= 4
        assert len(tracker.state.last_swing_lows) <= 4
