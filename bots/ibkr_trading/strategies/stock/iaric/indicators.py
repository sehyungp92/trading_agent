"""Pure-numpy vectorized indicators for the IARIC pullback-buy engine.

All functions take full-length arrays and return same-length arrays.
NaN-padded during warmup periods. No external TA library dependencies.

Performance: Uses pandas Cython-backed ewm/rolling for O(N) C-speed on
recursive (Wilder EMA) and windowed (rolling std/max) computations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers -- pandas-accelerated EMA primitives
# ---------------------------------------------------------------------------

def _wilder_ema(values: np.ndarray, period: int, *, seed_start: int = 0) -> np.ndarray:
    """Wilder EMA via pandas ewm (Cython). ~20-50x faster than Python loop.

    seed_start: index of the first valid element whose value seeds the EMA.
    The seed value should already be in ``values[seed_start]``.  Prior values
    are replaced with NaN so ewm ignores them.
    """
    n = len(values)
    if n <= seed_start:
        return np.full(n, np.nan, dtype=np.float64)
    buf = np.full(n, np.nan, dtype=np.float64)
    buf[seed_start:] = values[seed_start:]
    alpha = 1.0 / period
    result = pd.Series(buf).ewm(alpha=alpha, adjust=False, ignore_na=True).mean().to_numpy(dtype=np.float64, copy=True)
    result[:seed_start] = np.nan
    return result


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------

def rolling_sma(closes: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average via cumsum trick. O(N), NaN-padded warmup."""
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 1:
        return out
    cs = np.cumsum(closes)
    out[period - 1] = cs[period - 1] / period
    out[period:] = (cs[period:] - cs[:-period]) / period
    return out


def sma_slope_positive(sma: np.ndarray, lookback: int) -> np.ndarray:
    """Bool array: SMA[i] > SMA[i - lookback]. False during warmup."""
    n = len(sma)
    out = np.zeros(n, dtype=np.bool_)
    if lookback < 1 or n <= lookback:
        return out
    valid = ~np.isnan(sma[lookback:]) & ~np.isnan(sma[:-lookback])
    out[lookback:] = valid & (sma[lookback:] > sma[:-lookback])
    return out


def rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI using pandas ewm (Cython-backed).

    Standard Wilder method:
      1. First avg_gain/avg_loss = simple mean of first `period` changes
      2. Subsequent values use EMA: prev * (period-1)/period + current/period
    """
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1 or period < 1:
        return out

    deltas = np.diff(closes)  # length n-1
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple mean of first `period` values
    seed_gain = np.mean(gains[:period])
    seed_loss = np.mean(losses[:period])

    # Build seeded arrays for ewm
    g_buf = np.full(len(deltas), np.nan, dtype=np.float64)
    l_buf = np.full(len(deltas), np.nan, dtype=np.float64)
    g_buf[period - 1] = seed_gain
    g_buf[period:] = gains[period:]
    l_buf[period - 1] = seed_loss
    l_buf[period:] = losses[period:]

    alpha = 1.0 / period
    avg_g = pd.Series(g_buf).ewm(alpha=alpha, adjust=False, ignore_na=True).mean().values
    avg_l = pd.Series(l_buf).ewm(alpha=alpha, adjust=False, ignore_na=True).mean().values

    # RSI = 100 - 100/(1+RS) where RS = avg_gain/avg_loss
    valid = avg_l > 0
    rs = np.where(valid, avg_g / np.where(valid, avg_l, 1.0), 0.0)
    rsi_vals = np.where(valid, 100.0 - 100.0 / (1.0 + rs), 100.0)
    # Map back: deltas[i] corresponds to closes[i+1], so rsi output index = i+1
    out[period:] = rsi_vals[period - 1:]
    return out


def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Wilder ATR using pandas ewm (Cython-backed)."""
    n = len(highs)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1 or period < 1:
        return out

    # True range (vectorized)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    hl = highs[1:] - lows[1:]
    hc = np.abs(highs[1:] - closes[:-1])
    lc = np.abs(lows[1:] - closes[:-1])
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)

    # Seed = simple mean of first `period` TRs (index 1..period)
    seed = np.mean(tr[1 : period + 1])
    buf = np.full(n, np.nan, dtype=np.float64)
    buf[period] = seed
    buf[period + 1:] = tr[period + 1:]
    out[period:] = _wilder_ema(buf, period, seed_start=period)[period:]
    return out


def consecutive_down_days(closes: np.ndarray) -> np.ndarray:
    """Int array counting consecutive down-close streaks.

    cdd[i] = number of consecutive days where close[j] < close[j-1]
    ending at day i (inclusive). cdd[0] = 0.
    """
    n = len(closes)
    out = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if closes[i] < closes[i - 1]:
            out[i] = out[i - 1] + 1
        else:
            out[i] = 0
    return out


# ---------------------------------------------------------------------------
# V2 Indicators
# ---------------------------------------------------------------------------


def pullback_depth(
    highs: np.ndarray,
    closes: np.ndarray,
    atr_arr: np.ndarray,
    lookback: int = 10,
) -> np.ndarray:
    """ATR-normalized drop from recent high using pandas rolling max (C-speed).

    depth[i] = (max(highs[i-lb:i]) - closes[i]) / atr[i].
    Uses highs for the peak window, excludes current bar from peak.
    """
    n = len(highs)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < lookback + 1 or lookback < 1:
        return out
    # Rolling max of highs with window=lookback, shifted by 1 to exclude current bar
    rolling_peak = pd.Series(highs).rolling(window=lookback).max().values
    # rolling_peak[i] = max(highs[i-lookback+1:i+1]), shift by 1 for exclusion
    # We want max(highs[i-lookback:i]) = rolling_peak[i-1]
    peak = np.full(n, np.nan, dtype=np.float64)
    peak[lookback:] = rolling_peak[lookback - 1 : n - 1]
    valid = ~np.isnan(atr_arr) & (atr_arr > 0) & ~np.isnan(peak)
    out[valid] = (peak[valid] - closes[valid]) / atr_arr[valid]
    return out


def rate_of_change(closes: np.ndarray, period: int = 5) -> np.ndarray:
    """ROC[i] = (close[i] - close[i-period]) / close[i-period] * 100."""
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period or period < 1:
        return out
    prev = closes[:-period]
    safe_prev = np.where(prev != 0, prev, np.nan)
    out[period:] = (closes[period:] - prev) / safe_prev * 100.0
    return out


def bollinger_pctb(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> np.ndarray:
    """%B via pandas rolling std (Cython-backed). O(N) vs O(N*period)."""
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 1:
        return out
    s = pd.Series(closes)
    sma = s.rolling(window=period).mean().values
    std = s.rolling(window=period).std(ddof=0).values
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower
    valid = band_width > 0
    out[valid] = (closes[valid] - lower[valid]) / band_width[valid]
    return out


def volume_climax_ratio(
    volumes: np.ndarray,
    sma_period: int = 20,
) -> np.ndarray:
    """VCR[i] = volumes[i] / SMA(volumes, period)[i]."""
    n = len(volumes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < sma_period or sma_period < 1:
        return out
    vol_sma = rolling_sma(volumes.astype(np.float64), sma_period)
    valid = ~np.isnan(vol_sma) & (vol_sma > 0)
    out[valid] = volumes[valid] / vol_sma[valid]
    return out


def relative_strength_ratio(
    closes: np.ndarray,
    benchmark_closes: np.ndarray | None,
    period: int = 20,
) -> np.ndarray:
    """RS[i] = (closes[i]/closes[i-period]) / (bench[i]/bench[i-period]).

    Fully vectorized -- no Python loop.
    """
    if benchmark_closes is None:
        return np.full(len(closes), np.nan, dtype=np.float64)
    n = min(len(closes), len(benchmark_closes))
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period or period < 1:
        return out
    c_prev = closes[:n - period]
    b_prev = benchmark_closes[:n - period]
    c_curr = closes[period:n]
    b_curr = benchmark_closes[period:n]
    valid = (c_prev > 0) & (b_prev > 0) & (b_curr > 0)
    stock_ret = np.where(valid, c_curr / np.where(c_prev > 0, c_prev, 1.0), np.nan)
    bench_ret = np.where(valid, b_curr / np.where(b_prev > 0, b_prev, 1.0), np.nan)
    safe_bench = np.where(bench_ret > 0, bench_ret, np.nan)
    out[period:] = stock_ret / safe_bench
    return out


def adx_suite(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder's ADX using pandas ewm for smoothing (Cython-backed)."""
    n = len(highs)
    alpha = 1.0 / period

    # True range + directional movement (vectorized)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    hl = highs[1:] - lows[1:]
    hc = np.abs(highs[1:] - closes[:-1])
    lc = np.abs(lows[1:] - closes[:-1])
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)

    hi_diff = np.empty(n, dtype=np.float64)
    lo_diff = np.empty(n, dtype=np.float64)
    hi_diff[0] = 0.0
    lo_diff[0] = 0.0
    hi_diff[1:] = highs[1:] - highs[:-1]
    lo_diff[1:] = lows[:-1] - lows[1:]

    plus_dm = np.where((hi_diff > lo_diff) & (hi_diff > 0), hi_diff, 0.0)
    minus_dm = np.where((lo_diff > hi_diff) & (lo_diff > 0), lo_diff, 0.0)

    # Wilder EMA smoothing via pandas ewm
    s_tr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    s_plus = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values
    s_minus = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values

    safe_tr = np.where(s_tr > 0, s_tr, 1.0)
    plus_di = np.where(s_tr > 0, 100.0 * s_plus / safe_tr, 0.0)
    minus_di = np.where(s_tr > 0, 100.0 * s_minus / safe_tr, 0.0)

    di_sum = plus_di + minus_di
    safe_di_sum = np.where(di_sum > 0, di_sum, 1.0)
    dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / safe_di_sum, 0.0)

    adx_arr = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    return adx_arr, plus_di, minus_di


def ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average via pandas ewm (Cython-backed)."""
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 1:
        return out
    # Seed with SMA of first `period` values, then EMA from there
    seed = np.mean(closes[:period])
    buf = np.full(n, np.nan, dtype=np.float64)
    buf[period - 1] = seed
    buf[period:] = closes[period:]
    mult = 2.0 / (period + 1)
    out[period - 1:] = pd.Series(buf).ewm(alpha=mult, adjust=False, ignore_na=True).mean().values[period - 1:]
    return out
