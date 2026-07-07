"""NQ Dominant Trend Capture v2.0 — pure indicator computation."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed Average True Range."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def sma(series: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average via cumulative sum."""
    n = len(series)
    out = np.full(n, np.nan)
    if n < period:
        return out
    cs = np.cumsum(series)
    out[period - 1] = cs[period - 1] / period
    out[period:] = (cs[period:] - cs[:-period]) / period
    return out


def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    n = len(series)
    out = np.full(n, np.nan)
    if n < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = np.mean(series[:period])
    for i in range(period, n):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def macd_hist(series: np.ndarray, fast: int = 8, slow: int = 21,
              signal: int = 5) -> np.ndarray:
    """MACD histogram = MACD_line - signal_line."""
    n = len(series)
    ema_f = ema(series, fast)
    ema_s = ema(series, slow)
    macd_line = ema_f - ema_s
    sig_line = np.full(n, np.nan)
    start = slow - 1
    if n < start + signal:
        return np.full(n, np.nan)
    sig_line[start + signal - 1] = np.nanmean(macd_line[start:start + signal])
    k = 2.0 / (signal + 1)
    for i in range(start + signal, n):
        if np.isnan(macd_line[i]):
            continue
        sig_line[i] = macd_line[i] * k + sig_line[i - 1] * (1 - k)
    return macd_line - sig_line


def highest(series: np.ndarray, period: int) -> np.ndarray:
    """Rolling highest value."""
    n = len(series)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.max(series[max(0, i - period + 1):i + 1])
    return out


def lowest(series: np.ndarray, period: int) -> np.ndarray:
    """Rolling lowest value."""
    n = len(series)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.min(series[max(0, i - period + 1):i + 1])
    return out


# ---------------------------------------------------------------------------
# ADX (Section 9)
# ---------------------------------------------------------------------------

def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ADX, +DI, -DI. Returns (adx_arr, plus_di, minus_di)."""
    n = len(closes)
    adx_out = np.full(n, np.nan)
    pdi_out = np.full(n, np.nan)
    mdi_out = np.full(n, np.nan)
    if n < period + 1:
        return adx_out, pdi_out, mdi_out

    # True range, +DM, -DM
    tr = np.empty(n)
    plus_dm = np.empty(n)
    minus_dm = np.empty(n)
    tr[0] = highs[0] - lows[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing
    atr_s = np.mean(tr[1:period + 1])
    pdm_s = np.mean(plus_dm[1:period + 1])
    mdm_s = np.mean(minus_dm[1:period + 1])

    pdi_arr = np.full(n, np.nan)
    mdi_arr = np.full(n, np.nan)
    dx_arr = np.full(n, np.nan)

    idx = period
    pdi_arr[idx] = 100.0 * pdm_s / atr_s if atr_s > 0 else 0.0
    mdi_arr[idx] = 100.0 * mdm_s / atr_s if atr_s > 0 else 0.0
    s = pdi_arr[idx] + mdi_arr[idx]
    dx_arr[idx] = 100.0 * abs(pdi_arr[idx] - mdi_arr[idx]) / s if s > 0 else 0.0

    for i in range(period + 1, n):
        atr_s = (atr_s * (period - 1) + tr[i]) / period
        pdm_s = (pdm_s * (period - 1) + plus_dm[i]) / period
        mdm_s = (mdm_s * (period - 1) + minus_dm[i]) / period
        pdi_arr[i] = 100.0 * pdm_s / atr_s if atr_s > 0 else 0.0
        mdi_arr[i] = 100.0 * mdm_s / atr_s if atr_s > 0 else 0.0
        s = pdi_arr[i] + mdi_arr[i]
        dx_arr[i] = 100.0 * abs(pdi_arr[i] - mdi_arr[i]) / s if s > 0 else 0.0

    # ADX = Wilder-smoothed DX
    first_adx_idx = period + period
    if first_adx_idx >= n:
        return adx_out, pdi_arr, mdi_arr
    adx_out[first_adx_idx] = np.nanmean(dx_arr[period:first_adx_idx + 1])
    for i in range(first_adx_idx + 1, n):
        adx_out[i] = (adx_out[i - 1] * (period - 1) + dx_arr[i]) / period

    return adx_out, pdi_arr, mdi_arr


# ---------------------------------------------------------------------------
# VWAP (Section 5)
# ---------------------------------------------------------------------------

def session_vwap(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    volumes: np.ndarray, start_idx: int,
) -> np.ndarray:
    """Cumulative VWAP from start_idx onward."""
    n = len(closes)
    out = np.full(n, np.nan)
    tp = (highs + lows + closes) / 3.0
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(start_idx, n):
        v = volumes[i] if volumes[i] > 0 else 1.0
        cum_tpv += tp[i] * v
        cum_vol += v
        out[i] = cum_tpv / cum_vol if cum_vol > 0 else tp[i]
    return out


# ---------------------------------------------------------------------------
# Displacement (Section 11.2)
# ---------------------------------------------------------------------------

def displacement_metric(close_30m: float, vwap_box: float, atr14_30m: float) -> float:
    """DispMetric = |close - vwap_box| / ATR14_30m."""
    if atr14_30m <= 0:
        return 0.0
    return abs(close_30m - vwap_box) / atr14_30m


def rolling_quantile_past_only(data: list[float], q: float) -> float:
    """Percentile threshold from past-only data. q in [0,1]."""
    if not data:
        return 0.0
    arr = np.array(data)
    return float(np.quantile(arr, q))


# ---------------------------------------------------------------------------
# Squeeze metric (for scorecard)
# ---------------------------------------------------------------------------

def squeeze_metric(box_width: float, atr14_30m: float) -> float:
    """Normalized squeeze = box_width / ATR14_30m (smaller = tighter)."""
    if atr14_30m <= 0:
        return 0.0
    return box_width / atr14_30m


# ---------------------------------------------------------------------------
# RVOL (Section 11.3)
# ---------------------------------------------------------------------------

def compute_rvol(volume: float, median_volume: float) -> float:
    """Relative volume vs median same-slot."""
    if median_volume <= 0:
        return 1.0
    return volume / median_volume


# ---------------------------------------------------------------------------
# Percentile rank (past-only)
# ---------------------------------------------------------------------------

def percentile_rank(value: float, series: np.ndarray) -> float:
    """Percentile rank of value within series (0–100)."""
    valid = series[~np.isnan(series)]
    if len(valid) == 0:
        return 50.0
    return float(np.sum(valid < value) / len(valid) * 100)


# ---------------------------------------------------------------------------
# VWAP cross count (for chop, Section 10)
# ---------------------------------------------------------------------------

def vwap_cross_count(closes: np.ndarray, vwap: np.ndarray, lookback: int) -> int:
    """Count VWAP crosses in last `lookback` bars."""
    start = max(1, len(closes) - lookback)
    count = 0
    for i in range(start, len(closes)):
        if np.isnan(vwap[i]) or np.isnan(vwap[i - 1]):
            continue
        above_now = closes[i] > vwap[i]
        above_prev = closes[i - 1] > vwap[i - 1]
        if above_now != above_prev:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Chandelier trail (Section 17.5)
# ---------------------------------------------------------------------------

def chandelier_long(highs_1h: np.ndarray, atr14_1h: np.ndarray, lookback: int, mult: float) -> float:
    """Chandelier stop for long: highest_high(lookback) - mult * ATR14_1H."""
    if len(highs_1h) < lookback or np.isnan(atr14_1h[-1]):
        return float("-inf")
    hh = float(np.max(highs_1h[-lookback:]))
    return hh - mult * float(atr14_1h[-1])


def chandelier_short(lows_1h: np.ndarray, atr14_1h: np.ndarray, lookback: int, mult: float) -> float:
    """Chandelier stop for short: lowest_low(lookback) + mult * ATR14_1H."""
    if len(lows_1h) < lookback or np.isnan(atr14_1h[-1]):
        return float("inf")
    ll = float(np.min(lows_1h[-lookback:]))
    return ll + mult * float(atr14_1h[-1])


# ---------------------------------------------------------------------------
# Incremental (O(1)-per-bar) helpers for backtest engine
# ---------------------------------------------------------------------------

class IncrementalATR:
    """O(1)-per-bar ATR using Wilder smoothing.

    Pre-allocate with total bar count n; call update(t, H, L, C) each bar.
    Results stored in self.values[t].
    """
    __slots__ = ('period', 'values', '_prev_close', '_count', '_tr_sum')

    def __init__(self, n: int, period: int = 14):
        self.period = period
        self.values = np.full(n, np.nan)
        self._prev_close = np.nan
        self._count = 0
        self._tr_sum = 0.0

    def update(self, t: int, high: float, low: float, close: float) -> None:
        """Feed one bar. O(1)."""
        if self._count == 0:
            tr = high - low
        else:
            pc = self._prev_close
            hl = high - low
            hc = abs(high - pc)
            lc = abs(low - pc)
            tr = hl if (hl >= hc and hl >= lc) else (hc if hc >= lc else lc)
        self._prev_close = close
        self._count += 1
        if self._count < self.period:
            self._tr_sum += tr
        elif self._count == self.period:
            self._tr_sum += tr
            self.values[t] = self._tr_sum / self.period
        else:
            self.values[t] = (self.values[t - 1] * (self.period - 1) + tr) / self.period
