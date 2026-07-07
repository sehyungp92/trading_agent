"""Downturn Dominator indicators -- stateless computations.

Self-contained implementations (not imported from reference modules).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Incremental helpers (O(1) per update instead of O(N) full-array scan)
# ---------------------------------------------------------------------------

class IncrementalEMA:
    """O(1) EMA update. Seeds with SMA over first `period` values."""

    __slots__ = ("period", "alpha", "value", "_count", "_sum")

    def __init__(self, period: int) -> None:
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value = 0.0
        self._count = 0
        self._sum = 0.0

    def update(self, new_val: float) -> float:
        self._count += 1
        if self._count <= self.period:
            self._sum += new_val
            self.value = self._sum / self._count
            if self._count == self.period:
                self.value = self._sum / self.period
        else:
            self.value = self.value * (1 - self.alpha) + new_val * self.alpha
        return self.value


class IncrementalATR:
    """O(1) ATR update using Wilder smoothing."""

    __slots__ = ("period", "alpha", "value", "_count", "_sum", "_prev_close")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.alpha = 1.0 / period
        self.value = 0.0
        self._count = 0
        self._sum = 0.0
        self._prev_close = 0.0

    def update(self, high: float, low: float, close: float) -> float:
        if self._count == 0:
            self._prev_close = close
            self._count = 1
            self.value = high - low
            return self.value
        tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        self._count += 1
        if self._count <= self.period + 1:
            self._sum += tr
            if self._count == self.period + 1:
                self.value = self._sum / self.period
            else:
                self.value = self._sum / (self._count - 1)
        else:
            self.value = self.value * (1 - self.alpha) + tr * self.alpha
        return self.value


# ---------------------------------------------------------------------------
# Basic indicators
# ---------------------------------------------------------------------------

def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    """Compute ATR at the last bar using Wilder smoothing."""
    n = len(closes)
    if n < 2:
        return 0.0
    start = max(1, n - period * 3)  # enough history for smoothing
    tr_vals = np.empty(n - start)
    for i in range(start, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_vals[i - start] = max(hl, hc, lc)
    if len(tr_vals) < period:
        return float(np.mean(tr_vals)) if len(tr_vals) > 0 else 0.0
    atr = float(np.mean(tr_vals[:period]))
    alpha = 1.0 / period
    for i in range(period, len(tr_vals)):
        atr = atr * (1 - alpha) + tr_vals[i] * alpha
    return atr


def compute_ema(closes: np.ndarray, period: int) -> float:
    """Compute EMA at the last bar."""
    n = len(closes)
    if n == 0:
        return 0.0
    if n < period:
        return float(np.mean(closes))
    alpha = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    for i in range(period, n):
        ema = ema * (1 - alpha) + closes[i] * alpha
    return ema


def compute_ema_array(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute full EMA array."""
    n = len(closes)
    out = np.empty(n)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = closes[0]
    for i in range(1, min(period, n)):
        out[i] = out[i - 1] * (1 - alpha) + closes[i] * alpha
    if n >= period:
        out[period - 1] = float(np.mean(closes[:period]))
        for i in range(period, n):
            out[i] = out[i - 1] * (1 - alpha) + closes[i] * alpha
    return out


def compute_sma(closes: np.ndarray, period: int) -> float:
    """Compute SMA at the last bar."""
    if len(closes) < period:
        return float(np.mean(closes)) if len(closes) > 0 else 0.0
    return float(np.mean(closes[-period:]))


def compute_macd_hist(
    closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[float, float, float]:
    """Compute MACD, signal, and histogram at the last bar."""
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = compute_ema_array(closes, fast)
    ema_slow = compute_ema_array(closes, slow)
    macd_line = ema_fast - ema_slow
    # Signal line = EMA of MACD
    sig = np.empty(len(macd_line))
    sig[0] = macd_line[0]
    alpha = 2.0 / (signal + 1)
    for i in range(1, len(macd_line)):
        sig[i] = sig[i - 1] * (1 - alpha) + macd_line[i] * alpha
    hist = macd_line - sig
    return float(macd_line[-1]), float(sig[-1]), float(hist[-1])


def compute_adx(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int,
) -> float:
    """Compute ADX at the last bar using Wilder smoothing."""
    n = len(closes)
    if n < period * 2 + 1:
        return 0.0
    start = max(1, n - period * 4)
    length = n - start
    plus_dm = np.empty(length)
    minus_dm = np.empty(length)
    tr_vals = np.empty(length)
    for i in range(start, n):
        j = i - start
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[j] = up if (up > down and up > 0) else 0.0
        minus_dm[j] = down if (down > up and down > 0) else 0.0
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_vals[j] = max(hl, hc, lc)
    k = min(period, length)
    atr_s = float(np.mean(tr_vals[:k]))
    pdm_s = float(np.mean(plus_dm[:k]))
    mdm_s = float(np.mean(minus_dm[:k]))
    alpha = 1.0 / period
    # Build DX series from smoothed +DI/-DI
    dx_vals: list[float] = []
    for i in range(k, length):
        atr_s = atr_s * (1 - alpha) + tr_vals[i] * alpha
        pdm_s = pdm_s * (1 - alpha) + plus_dm[i] * alpha
        mdm_s = mdm_s * (1 - alpha) + minus_dm[i] * alpha
        if atr_s > 0:
            pdi = 100 * pdm_s / atr_s
            mdi = 100 * mdm_s / atr_s
            di_sum = pdi + mdi
            dx_vals.append(100 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0)
        else:
            dx_vals.append(0.0)
    if not dx_vals:
        return 0.0
    # ADX = Wilder-smoothed DX
    if len(dx_vals) < period:
        return float(np.mean(dx_vals))
    adx = float(np.mean(dx_vals[:period]))
    for i in range(period, len(dx_vals)):
        adx = adx * (1 - alpha) + dx_vals[i] * alpha
    return adx


def compute_adx_suite(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int,
) -> tuple[float, float, float]:
    """Compute (ADX, +DI, -DI) at the last bar using Wilder smoothing.

    Same algorithm as compute_adx() but returns all three values for
    bear conviction scoring.
    """
    n = len(closes)
    if n < period * 2 + 1:
        return 0.0, 0.0, 0.0
    start = max(1, n - period * 4)
    length = n - start
    plus_dm = np.empty(length)
    minus_dm = np.empty(length)
    tr_vals = np.empty(length)
    for i in range(start, n):
        j = i - start
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[j] = up if (up > down and up > 0) else 0.0
        minus_dm[j] = down if (down > up and down > 0) else 0.0
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_vals[j] = max(hl, hc, lc)
    k = min(period, length)
    atr_s = float(np.mean(tr_vals[:k]))
    pdm_s = float(np.mean(plus_dm[:k]))
    mdm_s = float(np.mean(minus_dm[:k]))
    alpha = 1.0 / period
    dx_vals: list[float] = []
    pdi_last = 0.0
    mdi_last = 0.0
    for i in range(k, length):
        atr_s = atr_s * (1 - alpha) + tr_vals[i] * alpha
        pdm_s = pdm_s * (1 - alpha) + plus_dm[i] * alpha
        mdm_s = mdm_s * (1 - alpha) + minus_dm[i] * alpha
        if atr_s > 0:
            pdi_last = 100 * pdm_s / atr_s
            mdi_last = 100 * mdm_s / atr_s
            di_sum = pdi_last + mdi_last
            dx_vals.append(100 * abs(pdi_last - mdi_last) / di_sum if di_sum > 0 else 0.0)
        else:
            dx_vals.append(0.0)
    if not dx_vals:
        return 0.0, 0.0, 0.0
    if len(dx_vals) < period:
        return float(np.mean(dx_vals)), pdi_last, mdi_last
    adx = float(np.mean(dx_vals[:period]))
    for i in range(period, len(dx_vals)):
        adx = adx * (1 - alpha) + dx_vals[i] * alpha
    return adx, pdi_last, mdi_last


def compute_session_vwap(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    volumes: np.ndarray, start_idx: int,
) -> float:
    """Compute session VWAP from start_idx to end of arrays."""
    if start_idx >= len(closes):
        return closes[-1] if len(closes) > 0 else 0.0
    typical = (highs[start_idx:] + lows[start_idx:] + closes[start_idx:]) / 3.0
    vol_slice = volumes[start_idx:]
    total_vol = np.sum(vol_slice)
    if total_vol == 0:
        return float(np.mean(typical))
    return float(np.sum(typical * vol_slice) / total_vol)


def compute_vwap_anchored(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    volumes: np.ndarray, anchor_idx: int,
) -> float:
    """Compute anchored VWAP from anchor_idx."""
    return compute_session_vwap(highs, lows, closes, volumes, anchor_idx)


# ---------------------------------------------------------------------------
# Downturn-specific indicators
# ---------------------------------------------------------------------------

def compute_divergence_magnitude(macd_h1: float, macd_h2: float, atr: float) -> float:
    """Divergence magnitude: |MACD_H2 - MACD_H1| / ATR.  (Spec S4.1)"""
    if atr <= 0:
        return 0.0
    return abs(macd_h2 - macd_h1) / atr


def compute_extension(
    close: float, ema_fast: float, atr: float, mult: float = 1.5,
) -> tuple[bool, bool]:
    """Extension check.  (Spec S5)

    Returns (ext_short, ext_long): whether price is extended above/below mean.
    """
    if atr <= 0:
        return False, False
    ext_short = close > ema_fast + mult * atr
    ext_long = close < ema_fast - mult * atr
    return ext_short, ext_long


def compute_chop_score(atr_pctl_60d: float, vwap_cross_count: int) -> int:
    """Chop score 0-4.  (Spec S6.3)

    Higher = choppier market, less favorable for breakdowns.
    """
    score = 0
    if atr_pctl_60d < 0.25:
        score += 1  # low vol = chop
    if atr_pctl_60d < 0.10:
        score += 1  # very low vol
    if vwap_cross_count >= 6:
        score += 1  # frequent VWAP crosses
    if vwap_cross_count >= 10:
        score += 1
    return min(score, 4)


def compute_trend_strength(ema_fast: float, ema_slow: float, atr: float) -> float:
    """Trend strength = (EMA_fast - EMA_slow) / ATR.  (Spec S2.6)

    Positive = bullish, negative = bearish.
    """
    if atr <= 0:
        return 0.0
    return (ema_fast - ema_slow) / atr


def compute_box_adaptive_length(atr_ratio: float) -> int:
    """Adaptive box length based on ATR ratio.  (Spec S6.1)

    atr_ratio = ATR_fast / ATR_slow.
    """
    if atr_ratio > 1.3:
        return 20   # high vol -> shorter box
    elif atr_ratio < 0.7:
        return 48   # low vol -> longer box
    return 32        # normal


def compute_displacement_metric(close: float, vwap_box: float, atr: float) -> float:
    """Displacement = (VWAP_box - close) / ATR.  (Spec S6.2)

    Positive for short breakdowns (close below VWAP).
    """
    if atr <= 0:
        return 0.0
    return (vwap_box - close) / atr


def compute_momentum_slope_ok(mom15_arr: np.ndarray, t: int, lookback: int = 3) -> bool:
    """Check if 15m momentum slope is bearish.  (Spec S7.4)

    Returns True if momentum is declining over lookback bars.
    """
    if t < lookback or len(mom15_arr) <= t:
        return False
    return bool(mom15_arr[t] < mom15_arr[t - lookback])


def highest(arr: np.ndarray, lookback: int) -> float:
    """Highest value in last `lookback` elements."""
    if len(arr) == 0:
        return 0.0
    start = max(0, len(arr) - lookback)
    return float(np.max(arr[start:]))


def lowest(arr: np.ndarray, lookback: int) -> float:
    """Lowest value in last `lookback` elements."""
    if len(arr) == 0:
        return 0.0
    start = max(0, len(arr) - lookback)
    return float(np.min(arr[start:]))


def percentile_rank(value: float, arr: np.ndarray, window: int) -> float:
    """Percentile rank of value within last `window` elements of arr."""
    if len(arr) == 0:
        return 0.5
    start = max(0, len(arr) - window)
    subset = arr[start:]
    if len(subset) == 0:
        return 0.5
    return float(np.sum(subset <= value) / len(subset))
