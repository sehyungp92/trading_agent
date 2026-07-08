"""AKC-Helix Swing v2.0 — pure indicator computation (no side effects).

All functions accept numpy arrays and return scalars or dataclasses.
Re-implements EMA/ATR locally to avoid cross-strategy coupling.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from .config import (
    ADX_PERIOD,
    ATR_DAILY_PERIOD,
    DAILY_EMA_FAST,
    DAILY_EMA_SLOW,
    EMA_4H_FAST,
    EMA_4H_SLOW,
    EXTREME_VOL_PCT,
    HIGH_VOL_PCT,
    LOW_VOL_PCT,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    VOLFACTOR_BASE_PERIOD,
    VOLFACTOR_MAX,
    VOLFACTOR_MIN,
)
from .models import DailyState, Pivot, PivotKind, Regime


# ---------------------------------------------------------------------------
# Primitive indicators
# ---------------------------------------------------------------------------

def ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with SMA seed."""
    out = np.empty_like(arr, dtype=float)
    k = 2.0 / (period + 1)
    seed_len = min(period, len(arr))
    out[0] = float(np.mean(arr[:seed_len]))
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""
    n = len(highs)
    tr = np.empty(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    out = np.empty(n, dtype=float)
    out[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
    return out


# ---------------------------------------------------------------------------
# ADX — Wilder's Average Directional Index (v2.0)
# ---------------------------------------------------------------------------

def compute_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = ADX_PERIOD,
) -> float:
    """Wilder's Average Directional Index on daily bars. Returns current ADX value."""
    n = len(highs)
    if n < period + 1:
        return 0.0

    # True Range, +DM, -DM
    tr = np.empty(n, dtype=float)
    plus_dm = np.empty(n, dtype=float)
    minus_dm = np.empty(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing
    alpha = 1.0 / period
    atr_w = np.empty(n, dtype=float)
    plus_di_smooth = np.empty(n, dtype=float)
    minus_di_smooth = np.empty(n, dtype=float)

    atr_w[0] = tr[0]
    plus_di_smooth[0] = plus_dm[0]
    minus_di_smooth[0] = minus_dm[0]

    for i in range(1, n):
        atr_w[i] = atr_w[i - 1] * (1 - alpha) + tr[i] * alpha
        plus_di_smooth[i] = plus_di_smooth[i - 1] * (1 - alpha) + plus_dm[i] * alpha
        minus_di_smooth[i] = minus_di_smooth[i - 1] * (1 - alpha) + minus_dm[i] * alpha

    # DI values and DX
    dx = np.empty(n, dtype=float)
    for i in range(n):
        if atr_w[i] > 0:
            pdi = 100.0 * plus_di_smooth[i] / atr_w[i]
            mdi = 100.0 * minus_di_smooth[i] / atr_w[i]
        else:
            pdi = mdi = 0.0
        s = pdi + mdi
        dx[i] = 100.0 * abs(pdi - mdi) / s if s > 0 else 0.0

    # ADX = Wilder smoothing of DX
    adx_arr = np.empty(n, dtype=float)
    adx_arr[0] = dx[0]
    for i in range(1, n):
        adx_arr[i] = adx_arr[i - 1] * (1 - alpha) + dx[i] * alpha

    return float(adx_arr[-1])


# ---------------------------------------------------------------------------
# 4H Regime (v2.0)
# ---------------------------------------------------------------------------

def compute_regime_4h(
    close_4h: float,
    ema_fast_4h: float,
    ema_slow_4h: float,
) -> Regime:
    """Same logic as daily regime but on 4H bars."""
    return compute_regime(close_4h, ema_fast_4h, ema_slow_4h)


# ---------------------------------------------------------------------------
# MACD (spec s3)
# ---------------------------------------------------------------------------

def macd(
    closes: np.ndarray,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD → (line, signal_line, histogram)."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    line = ema_fast - ema_slow
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


# ---------------------------------------------------------------------------
# Pivot detection (spec s2) — 5-bar non-repainting
# ---------------------------------------------------------------------------

def confirmed_pivot(
    highs: np.ndarray,
    lows: np.ndarray,
    t_idx: int,
    macd_line: np.ndarray,
    macd_hist: np.ndarray,
    atr_arr: np.ndarray,
    bar_times: list[datetime],
) -> Optional[Pivot]:
    """Check for a confirmed pivot at t_idx-2 using 5-bar window [t-4..t].

    Pivot high at t-2 if High[t-2] == max(High[t-4..t]).
    Pivot low at t-2 if Low[t-2] == min(Low[t-4..t]).
    Returns the pivot (anchored at t-2) or None.
    """
    if t_idx < 4:
        return None

    center = t_idx - 2
    window_start = t_idx - 4

    # Check for pivot high
    window_highs = highs[window_start:t_idx + 1]
    if highs[center] == float(np.max(window_highs)):
        return Pivot(
            ts=bar_times[center],
            kind=PivotKind.HIGH,
            price=float(highs[center]),
            macd_line=float(macd_line[center]),
            macd_hist=float(macd_hist[center]),
            atr_tf=float(atr_arr[center]),
            bar_index=center,
        )

    # Check for pivot low
    window_lows = lows[window_start:t_idx + 1]
    if lows[center] == float(np.min(window_lows)):
        return Pivot(
            ts=bar_times[center],
            kind=PivotKind.LOW,
            price=float(lows[center]),
            macd_line=float(macd_line[center]),
            macd_hist=float(macd_hist[center]),
            atr_tf=float(atr_arr[center]),
            bar_index=center,
        )

    return None


def scan_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    macd_line: np.ndarray,
    macd_hist: np.ndarray,
    atr_arr: np.ndarray,
    bar_times: list[datetime],
    start_idx: int = 4,
) -> list[Pivot]:
    """Scan full array for confirmed pivots from start_idx onward."""
    pivots: list[Pivot] = []
    for t in range(start_idx, len(highs)):
        p = confirmed_pivot(highs, lows, t, macd_line, macd_hist, atr_arr, bar_times)
        if p is not None:
            pivots.append(p)
    return pivots


# ---------------------------------------------------------------------------
# Buffer (spec s2)
# ---------------------------------------------------------------------------

def calc_buffer(tick_size: float, atr_tf: float, is_etf: bool) -> float:
    """Buffer = max(tick_size or 0.01, 0.05 * ATR_TF)."""
    floor = tick_size if tick_size > 0 else 0.01
    return max(floor, 0.05 * atr_tf)


# ---------------------------------------------------------------------------
# Regime (spec s5)
# ---------------------------------------------------------------------------

def compute_regime(close: float, ema_fast: float, ema_slow: float) -> Regime:
    """BULL if close > EMA_fast > EMA_slow, BEAR if close < EMA_fast < EMA_slow, else CHOP."""
    if close > ema_fast > ema_slow:
        return Regime.BULL
    elif close < ema_fast < ema_slow:
        return Regime.BEAR
    return Regime.CHOP


def compute_trend_strength(ema_fast: float, ema_slow: float, atr_d: float) -> float:
    """Trend strength = abs(EMA_fast - EMA_slow) / ATRd."""
    if atr_d <= 0:
        return 0.0
    return abs(ema_fast - ema_slow) / atr_d


# ---------------------------------------------------------------------------
# VolFactor (spec s6)
# ---------------------------------------------------------------------------

def compute_vol_factor(atr_today: float, atr_base: float, vol_pct: float) -> float:
    """VolFactor = clamp(ATR_base / ATR_today, MIN, MAX) with low-vol cap."""
    if atr_today <= 0:
        return 1.0
    raw = atr_base / atr_today
    clamped = max(VOLFACTOR_MIN, min(VOLFACTOR_MAX, raw))
    # Low-vol environment: cap at 1.0 to prevent over-sizing
    if vol_pct < LOW_VOL_PCT:
        clamped = min(clamped, 1.0)
    return clamped


def percentile_rank(value: float, array: np.ndarray) -> float:
    """Percent of values in array that are <= value (0-100 scale)."""
    if len(array) == 0:
        return 50.0
    return float(np.sum(array <= value) / len(array) * 100.0)


# ---------------------------------------------------------------------------
# Composite daily state
# ---------------------------------------------------------------------------

def compute_daily_state(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    prev: Optional[DailyState],
    daily_bar_date: Optional[str] = None,
) -> DailyState:
    """Compute all daily indicators and return a new DailyState."""
    ema_f = ema(closes, DAILY_EMA_FAST)
    ema_s = ema(closes, DAILY_EMA_SLOW)
    atr_arr = atr(highs, lows, closes, ATR_DAILY_PERIOD)

    cur_close = float(closes[-1])
    cur_ema_f = float(ema_f[-1])
    cur_ema_s = float(ema_s[-1])
    cur_atr = float(atr_arr[-1])

    # Regime
    regime = compute_regime(cur_close, cur_ema_f, cur_ema_s)

    # Trend strength
    ts_now = compute_trend_strength(cur_ema_f, cur_ema_s, cur_atr)
    # 3 bars ago
    if len(ema_f) >= 4 and len(ema_s) >= 4 and len(atr_arr) >= 4:
        ts_3d = compute_trend_strength(
            float(ema_f[-4]), float(ema_s[-4]), float(atr_arr[-4])
        )
    else:
        ts_3d = ts_now

    # ATR base (rolling 60-day median of ATR for VolFactor — spec s6)
    if len(atr_arr) >= VOLFACTOR_BASE_PERIOD:
        atr_base = float(np.median(atr_arr[-VOLFACTOR_BASE_PERIOD:]))
    else:
        atr_base = float(np.median(atr_arr))

    # Volatility percentile (where current ATR sits in 60-day range)
    lookback = min(VOLFACTOR_BASE_PERIOD, len(atr_arr))
    vol_pct = percentile_rank(cur_atr, atr_arr[-lookback:])

    # VolFactor
    vf = compute_vol_factor(cur_atr, atr_base, vol_pct)

    # Extreme vol flag
    extreme = vol_pct >= EXTREME_VOL_PCT

    # ADX (v2.0)
    adx_val = compute_adx(highs, lows, closes, ADX_PERIOD)

    return DailyState(
        ema_fast=cur_ema_f,
        ema_slow=cur_ema_s,
        atr_d=cur_atr,
        regime=regime,
        trend_strength=ts_now,
        trend_strength_3d_ago=ts_3d,
        vol_pct=vol_pct,
        atr_base=atr_base,
        vol_factor=vf,
        extreme_vol=extreme,
        close=cur_close,
        last_bar_date=daily_bar_date,
        adx=adx_val,
    )
