"""Tests for indicator computations against known values."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from crypto_trader.core.models import Bar, TimeFrame
from crypto_trader.strategy.momentum.config import IndicatorParams
from crypto_trader.strategy.momentum.indicators import (
    IndicatorSnapshot,
    _atr,
    _ema,
    _rsi,
    compute_indicators,
)
from tests.conftest import make_bar


def _make_bars(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> list[Bar]:
    """Create bars from close prices with synthetic OHLV."""
    bars = []
    for i, c in enumerate(closes):
        h = highs[i] if highs else c + 10
        l = lows[i] if lows else c - 10
        o = (c + l + h) / 3  # arbitrary open
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(minutes=15 * i)
        bars.append(make_bar(ts, o=o, h=h, l=l, c=c, v=100.0))
    return bars


class TestEMA:
    def test_constant_series(self):
        data = np.full(50, 100.0)
        result = _ema(data, 20)
        assert result[-1] == pytest.approx(100.0, abs=1e-10)

    def test_step_function(self):
        data = np.concatenate([np.full(50, 100.0), np.full(50, 200.0)])
        result = _ema(data, 20)
        # After enough bars, should converge toward 200
        assert result[-1] > 195
        # Middle should be transitioning
        assert 100 < result[60] < 200

    def test_single_value(self):
        data = np.array([42.0])
        result = _ema(data, 5)
        assert result[0] == pytest.approx(42.0)

    def test_length_preserved(self):
        data = np.arange(100, dtype=float)
        result = _ema(data, 10)
        assert len(result) == len(data)


class TestATR:
    def test_constant_bars(self):
        n = 30
        highs = np.full(n, 110.0)
        lows = np.full(n, 90.0)
        closes = np.full(n, 100.0)
        result = _atr(highs, lows, closes, 14)
        # True range = 20 for all bars
        assert result[-1] == pytest.approx(20.0, abs=0.1)

    def test_increasing_range(self):
        n = 30
        highs = np.array([100 + i * 2 for i in range(n)], dtype=float)
        lows = np.array([100 - i * 2 for i in range(n)], dtype=float)
        closes = np.array([100 + i for i in range(n)], dtype=float)
        result = _atr(highs, lows, closes, 14)
        # ATR should increase over time
        valid = result[~np.isnan(result)]
        assert valid[-1] > valid[0]


class TestRSI:
    def test_all_gains(self):
        closes = np.arange(100, 130, dtype=float)
        result = _rsi(closes, 14)
        # With only gains, RSI should be 100
        assert result[-1] == pytest.approx(100.0)

    def test_all_losses(self):
        closes = np.arange(130, 100, -1, dtype=float)
        result = _rsi(closes, 14)
        # With only losses, RSI should be 0
        assert result[-1] == pytest.approx(0.0, abs=0.01)

    def test_neutral(self):
        # Alternating up/down of equal magnitude should give ~50
        closes = np.array([100 + (1 if i % 2 == 0 else -1) for i in range(50)], dtype=float)
        result = _rsi(closes, 14)
        assert 40 < result[-1] < 60


class TestComputeIndicators:
    def test_insufficient_data(self):
        bars = _make_bars([100.0] * 50)
        params = IndicatorParams(ema_slow=200)
        result = compute_indicators(bars, params)
        assert result is None

    def test_sufficient_data(self):
        # 250 bars of trending data
        closes = [100 + i * 0.5 for i in range(250)]
        bars = _make_bars(closes)
        params = IndicatorParams()
        result = compute_indicators(bars, params)
        assert result is not None
        assert isinstance(result, IndicatorSnapshot)

        # EMA fast should be above EMA slow in uptrend
        assert result.ema_fast > result.ema_slow
        # RSI should be elevated in uptrend
        assert result.rsi > 50
        # ATR should be positive
        assert result.atr > 0

    def test_ema_arrays_are_recent(self):
        closes = [100 + i * 0.5 for i in range(250)]
        bars = _make_bars(closes)
        result = compute_indicators(bars, IndicatorParams())
        assert result is not None
        assert len(result.ema_fast_arr) == 10
        assert len(result.ema_mid_arr) == 10
        assert len(result.ema_slow_arr) == 10
