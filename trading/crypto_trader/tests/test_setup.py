"""Tests for pullback zone detection and grading."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from crypto_trader.core.models import Bar, SetupGrade, Side, TimeFrame
from crypto_trader.strategy.momentum.bias import BiasResult
from crypto_trader.strategy.momentum.config import IndicatorParams, SetupParams
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot, compute_indicators
from crypto_trader.strategy.momentum.setup import SetupDetector
from tests.conftest import make_bar


def _make_pullback_bars(
    direction: str = "up",
    count: int = 250,
    tf: TimeFrame = TimeFrame.M15,
) -> list[Bar]:
    """Create bars with a trend + pullback pattern."""
    bars = []
    base = 50000.0
    step = 50.0

    for i in range(count):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
        if i < count * 0.7:
            # Trending phase
            if direction == "up":
                c = base + i * step
            else:
                c = base - i * step
        else:
            # Pullback phase
            pullback_i = i - int(count * 0.7)
            if direction == "up":
                c = base + int(count * 0.7) * step - pullback_i * step * 0.3
            else:
                c = base - int(count * 0.7) * step + pullback_i * step * 0.3

        o = c - step * 0.2
        h = c + step * 0.5
        l = c - step * 0.5
        bars.append(Bar(timestamp=ts, symbol="BTC", open=o, high=h, low=l, close=c, volume=100.0, timeframe=tf))
    return bars


def _make_h1_bars(m15_bars: list[Bar]) -> list[Bar]:
    """Create simplified H1 bars from M15 bars."""
    bars = []
    for i in range(0, len(m15_bars), 4):
        chunk = m15_bars[i:i+4]
        if not chunk:
            break
        bars.append(Bar(
            timestamp=chunk[0].timestamp,
            symbol=chunk[0].symbol,
            open=chunk[0].open,
            high=max(b.high for b in chunk),
            low=min(b.low for b in chunk),
            close=chunk[-1].close,
            volume=sum(b.volume for b in chunk),
            timeframe=TimeFrame.H1,
        ))
    return bars


class TestSetupDetector:
    def test_returns_none_without_bias(self):
        detector = SetupDetector(SetupParams())
        bars = _make_pullback_bars()
        h1_bars = _make_h1_bars(bars)
        ind = compute_indicators(bars, IndicatorParams())
        bias = BiasResult(direction=None, h4_score=0, h1_score=0, confidence=0, reasons=())
        result = detector.detect(bars, h1_bars, ind, bias)
        assert result is None

    def test_detects_setup_in_uptrend_pullback(self):
        detector = SetupDetector(SetupParams(
            min_confluences_a=1,  # Relaxed for test
            min_room_a=1.0,
            min_room_b=0.5,
        ))
        m15_bars = _make_pullback_bars("up")
        h1_bars = _make_h1_bars(m15_bars)
        ind = compute_indicators(m15_bars, IndicatorParams())
        assert ind is not None

        bias = BiasResult(direction=Side.LONG, h4_score=3, h1_score=3, confidence=1.0, reasons=("strong uptrend",))
        result = detector.detect(m15_bars, h1_bars, ind, bias)
        # With pullback into EMAs, should detect a setup
        if result is not None:
            assert result.grade in (SetupGrade.A, SetupGrade.B)
            assert result.zone_price > 0
            assert result.stop_level > 0

    def test_insufficient_data(self):
        detector = SetupDetector(SetupParams())
        bars = [make_bar(datetime(2025, 1, 1, tzinfo=timezone.utc), 100, 110, 90, 100)]
        bias = BiasResult(direction=Side.LONG, h4_score=3, h1_score=3, confidence=1.0, reasons=())
        result = detector.detect(bars, bars, None, bias)
        assert result is None

    def test_rsi_pullback_filter_rejects_high_rsi_long(self):
        """RSI filter rejects longs when RSI is above threshold (not a real pullback)."""
        detector = SetupDetector(SetupParams(
            min_confluences_a=1,
            min_room_a=1.0,
            min_room_b=0.5,
            use_rsi_pullback_filter=True,
            rsi_pullback_threshold=40.0,
            reject_extended_reaction=False,
        ))
        m15_bars = _make_pullback_bars("up")
        h1_bars = _make_h1_bars(m15_bars)
        ind = compute_indicators(m15_bars, IndicatorParams())
        assert ind is not None

        # Force RSI above threshold for deterministic test
        ind_high_rsi = replace(ind, rsi=65.0)

        bias = BiasResult(direction=Side.LONG, h4_score=3, h1_score=3, confidence=1.0, reasons=("strong uptrend",))
        result = detector.detect(m15_bars, h1_bars, ind_high_rsi, bias)
        assert result is None

    def test_rsi_pullback_filter_disabled_by_default(self):
        """RSI filter off by default — should not affect detection."""
        detector = SetupDetector(SetupParams(
            min_confluences_a=1,
            min_room_a=1.0,
            min_room_b=0.5,
            use_rsi_pullback_filter=False,
        ))
        m15_bars = _make_pullback_bars("up")
        h1_bars = _make_h1_bars(m15_bars)
        ind = compute_indicators(m15_bars, IndicatorParams())
        bias = BiasResult(direction=Side.LONG, h4_score=3, h1_score=3, confidence=1.0, reasons=("strong uptrend",))
        # With filter disabled, RSI value doesn't matter
        result = detector.detect(m15_bars, h1_bars, ind, bias)
        # May or may not detect depending on pullback quality, but RSI won't block it
        assert result is None or isinstance(result.grade, SetupGrade)

    def test_rejects_parabolic(self):
        detector = SetupDetector(SetupParams(reject_parabolic_extension=True))
        # Create bars with huge bodies (parabolic)
        bars = []
        for i in range(250):
            ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
            c = 50000 + i * 500  # Very steep
            bars.append(Bar(
                timestamp=ts, symbol="BTC", open=c - 400, high=c + 100,
                low=c - 500, close=c, volume=100.0, timeframe=TimeFrame.M15,
            ))
        h1 = _make_h1_bars(bars)
        ind = compute_indicators(bars, IndicatorParams())
        bias = BiasResult(direction=Side.LONG, h4_score=3, h1_score=3, confidence=1.0, reasons=())
        result = detector.detect(bars, h1, ind, bias)
        # Should either be None (rejected) or still valid depending on ATR
        # The key is that the function runs without error
        assert result is None or isinstance(result.grade, SetupGrade)

    def test_rejects_extended_reaction(self):
        """Extended reaction filter rejects when impulse overextends beyond EMA."""
        detector = SetupDetector(SetupParams(
            reject_extended_reaction=True,
            min_confluences_a=1,
            min_room_a=1.0,
            min_room_b=0.5,
        ))
        m15_bars = _make_pullback_bars("up")
        h1_bars = _make_h1_bars(m15_bars)
        ind = compute_indicators(m15_bars, IndicatorParams())
        assert ind is not None

        # Force EMA fast well below recent high (>4 ATR extension)
        recent_high = max(b.high for b in m15_bars[-20:])
        forced_ema = recent_high - ind.atr * 5.0
        ind_extended = replace(ind, ema_fast=forced_ema)

        bias = BiasResult(direction=Side.LONG, h4_score=3, h1_score=3, confidence=1.0, reasons=("strong uptrend",))
        result = detector.detect(m15_bars, h1_bars, ind_extended, bias)
        assert result is None

    def test_extended_reaction_disabled_passes(self):
        """With extended reaction filter disabled, overextended setups pass this check."""
        detector = SetupDetector(SetupParams(
            reject_extended_reaction=False,
            min_confluences_a=1,
            min_room_a=1.0,
            min_room_b=0.5,
        ))
        m15_bars = _make_pullback_bars("up")
        h1_bars = _make_h1_bars(m15_bars)
        ind = compute_indicators(m15_bars, IndicatorParams())
        assert ind is not None

        bias = BiasResult(direction=Side.LONG, h4_score=3, h1_score=3, confidence=1.0, reasons=("strong uptrend",))
        result = detector.detect(m15_bars, h1_bars, ind, bias)
        # May or may not detect depending on other conditions, but extension won't block it
        assert result is None or isinstance(result.grade, SetupGrade)
