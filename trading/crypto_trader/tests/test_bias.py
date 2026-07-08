"""Tests for multi-timeframe bias detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.momentum.bias import BiasDetector, _detect_structure, _find_swing_points
from crypto_trader.strategy.momentum.config import BiasParams, IndicatorParams
from crypto_trader.strategy.momentum.indicators import compute_indicators
from tests.conftest import make_bar


def _make_trending_bars(
    direction: str,
    count: int = 250,
    tf: TimeFrame = TimeFrame.H4,
    base_price: float = 50000.0,
    step: float = 100.0,
) -> list[Bar]:
    """Create bars with a clear trend direction."""
    bars = []
    for i in range(count):
        if direction == "up":
            c = base_price + i * step
        else:
            c = base_price - i * step
        o = c - step / 2 if direction == "up" else c + step / 2
        h = max(o, c) + step * 0.3
        l = min(o, c) - step * 0.3
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4 * i)
        bars.append(Bar(timestamp=ts, symbol="BTC", open=o, high=h, low=l, close=c, volume=100.0, timeframe=tf))
    return bars


class TestDetectStructure:
    def test_bullish_structure(self):
        # HH/HL pattern
        bars = []
        prices = [100, 110, 105, 115, 108, 120, 112, 125]
        for i, p in enumerate(prices):
            ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            bars.append(make_bar(ts, o=p - 2, h=p + 3, l=p - 3, c=p, tf=TimeFrame.H4))
        result = _detect_structure(bars, 8)
        assert result == "bullish"

    def test_bearish_structure(self):
        # LH/LL pattern
        bars = []
        prices = [125, 118, 120, 112, 115, 108, 110, 100]
        for i, p in enumerate(prices):
            ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            bars.append(make_bar(ts, o=p + 2, h=p + 3, l=p - 3, c=p, tf=TimeFrame.H4))
        result = _detect_structure(bars, 8)
        assert result == "bearish"

    def test_neutral_flat(self):
        bars = []
        for i in range(10):
            ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            bars.append(make_bar(ts, o=100, h=101, l=99, c=100, tf=TimeFrame.H4))
        result = _detect_structure(bars, 10)
        assert result == "neutral"


class TestFindSwingPoints:
    def test_finds_highs_and_lows(self):
        bars = []
        prices = [(100, 105, 95), (110, 115, 105), (100, 105, 95), (115, 120, 110), (105, 110, 100)]
        for i, (c, h, l) in enumerate(prices):
            ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            bars.append(make_bar(ts, o=c, h=h, l=l, c=c, tf=TimeFrame.H4))
        swings = _find_swing_points(bars)
        assert len(swings) > 0
        # Index 1 should be a swing high, index 2 a swing low
        types = [s[2] for s in swings]
        assert "high" in types
        assert "low" in types


class TestBiasDetector:
    def test_strong_bullish_bias(self):
        detector = BiasDetector(BiasParams())
        h4_bars = _make_trending_bars("up", count=250, tf=TimeFrame.H4)
        h1_bars = _make_trending_bars("up", count=250, tf=TimeFrame.H1, step=25.0)

        h4_ind = compute_indicators(h4_bars, IndicatorParams())
        h1_ind = compute_indicators(h1_bars, IndicatorParams())

        result = detector.compute(h4_bars, h1_bars, h4_ind, h1_ind)
        assert result.direction == Side.LONG
        assert result.h4_score >= 2
        assert result.confidence > 0

    def test_strong_bearish_bias(self):
        detector = BiasDetector(BiasParams())
        h4_bars = _make_trending_bars("down", count=250, tf=TimeFrame.H4)
        h1_bars = _make_trending_bars("down", count=250, tf=TimeFrame.H1, step=25.0)

        h4_ind = compute_indicators(h4_bars, IndicatorParams())
        h1_ind = compute_indicators(h1_bars, IndicatorParams())

        result = detector.compute(h4_bars, h1_bars, h4_ind, h1_ind)
        assert result.direction == Side.SHORT
        assert result.h4_score >= 2

    def test_conflicting_timeframes(self):
        detector = BiasDetector(BiasParams())
        h4_bars = _make_trending_bars("up", count=250, tf=TimeFrame.H4)
        h1_bars = _make_trending_bars("down", count=250, tf=TimeFrame.H1, step=25.0)

        h4_ind = compute_indicators(h4_bars, IndicatorParams())
        h1_ind = compute_indicators(h1_bars, IndicatorParams())

        result = detector.compute(h4_bars, h1_bars, h4_ind, h1_ind)
        # Should be None when timeframes disagree
        assert result.direction is None

    def test_insufficient_data(self):
        detector = BiasDetector(BiasParams())
        result = detector.compute([], [], None, None)
        assert result.direction is None
        assert "insufficient data" in result.reasons
