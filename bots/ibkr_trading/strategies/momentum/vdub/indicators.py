"""Vdubus NQ v4.0 — deterministic indicator functions (pure numpy)."""
from __future__ import annotations

import numpy as np
from .models import PivotPoint


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Wilder-smoothed ATR."""
    n = len(highs)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]),
                   np.abs(lows[1:] - closes[:-1])))
    if n < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def sma(series: np.ndarray, period: int) -> np.ndarray:
    n = len(series)
    out = np.full(n, np.nan)
    if n < period:
        return out
    cs = np.concatenate([[0.0], np.nancumsum(series)])
    out[period - 1:] = (cs[period:] - cs[:-period]) / period
    return out


def ema(series: np.ndarray, period: int) -> np.ndarray:
    n = len(series)
    out = np.full(n, np.nan)
    if n < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = np.nanmean(series[:period])
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
    # Signal line = EMA of MACD from first valid index
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


def highest(series: np.ndarray, lookback: int) -> np.ndarray:
    n = len(series)
    out = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        out[i] = np.nanmax(series[i - lookback + 1:i + 1])
    return out


def lowest(series: np.ndarray, lookback: int) -> np.ndarray:
    n = len(series)
    out = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        out[i] = np.nanmin(series[i - lookback + 1:i + 1])
    return out


def session_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 volumes: np.ndarray, start_idx: int) -> np.ndarray:
    """Cumulative VWAP from start_idx onward."""
    n = len(highs)
    out = np.full(n, np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(max(0, start_idx), n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        v = max(volumes[i], 1.0)
        cum_tpv += tp * v
        cum_vol += v
        out[i] = cum_tpv / cum_vol
    return out


def anchored_vwap_value(highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray, volumes: np.ndarray,
                        anchor_idx: int) -> float:
    """Current VWAP-A value from anchor_idx to end."""
    if anchor_idx < 0 or anchor_idx >= len(highs):
        return np.nan
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(anchor_idx, len(highs)):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        v = max(volumes[i], 1.0)
        cum_tpv += tp * v
        cum_vol += v
    return cum_tpv / cum_vol if cum_vol > 0 else np.nan


def anchored_vwap_series(highs: np.ndarray, lows: np.ndarray,
                         closes: np.ndarray, volumes: np.ndarray,
                         anchor_idx: int) -> np.ndarray:
    """Running VWAP-A series from anchor_idx to end."""
    n = len(highs)
    out = np.full(n, np.nan)
    if anchor_idx < 0 or anchor_idx >= n:
        return out
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(anchor_idx, n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        v = max(volumes[i], 1.0)
        cum_tpv += tp * v
        cum_vol += v
        out[i] = cum_tpv / cum_vol
    return out


def confirmed_pivots(highs: np.ndarray, lows: np.ndarray,
                     nconfirm: int) -> list[PivotPoint]:
    """Non-repainting confirmed pivots.

    A pivot high at i: confirmed when nconfirm subsequent bars all have lower highs.
    A pivot low at i: confirmed when nconfirm subsequent bars all have higher lows.
    """
    n = len(highs)
    pivots: list[PivotPoint] = []
    for i in range(1, n - nconfirm):
        # Pivot high
        if highs[i] > highs[i - 1]:
            is_high = True
            for j in range(1, nconfirm + 1):
                if highs[i + j] >= highs[i]:
                    is_high = False
                    break
            if is_high:
                pivots.append(PivotPoint(
                    idx=i, price=float(highs[i]), ptype="high",
                    confirmed_at=i + nconfirm))
        # Pivot low
        if lows[i] < lows[i - 1]:
            is_low = True
            for j in range(1, nconfirm + 1):
                if lows[i + j] <= lows[i]:
                    is_low = False
                    break
            if is_low:
                pivots.append(PivotPoint(
                    idx=i, price=float(lows[i]), ptype="low",
                    confirmed_at=i + nconfirm))
    return sorted(pivots, key=lambda p: p.idx)


def percentile_rank(value: float, series: np.ndarray,
                    lookback: int = 252) -> float:
    valid = series[-lookback:]
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return 50.0
    return float(np.sum(valid < value) / len(valid) * 100)


def median_val(series: np.ndarray, lookback: int = 252) -> float:
    valid = series[-lookback:]
    valid = valid[~np.isnan(valid)]
    return float(np.nanmedian(valid)) if len(valid) > 0 else np.nan


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


class IncrementalMACD:
    """O(1)-per-bar MACD histogram (fast EMA - slow EMA, smoothed by signal EMA).

    Pre-allocate with total bar count n; call update(t, close) each bar.
    Results stored in self.values[t].
    """
    __slots__ = ('values',
                 '_fast_k', '_slow_k', '_sig_k',
                 '_fast_val', '_slow_val', '_sig_val',
                 '_fast_sum', '_slow_sum',
                 '_fast_n', '_slow_n',
                 '_fast_period', '_slow_period', '_sig_period',
                 '_seed', '_sig_ready')

    def __init__(self, n: int, fast: int = 8, slow: int = 21, signal: int = 5):
        self.values = np.full(n, np.nan)
        self._fast_period = fast
        self._slow_period = slow
        self._sig_period = signal
        self._fast_k = 2.0 / (fast + 1)
        self._slow_k = 2.0 / (slow + 1)
        self._sig_k = 2.0 / (signal + 1)
        self._fast_val = 0.0
        self._slow_val = 0.0
        self._sig_val = 0.0
        self._fast_sum = 0.0
        self._slow_sum = 0.0
        self._fast_n = 0
        self._slow_n = 0
        self._seed: list[float] = []
        self._sig_ready = False

    def update(self, t: int, close: float) -> None:
        """Feed one bar. O(1)."""
        # Fast EMA
        self._fast_n += 1
        if self._fast_n < self._fast_period:
            self._fast_sum += close
        elif self._fast_n == self._fast_period:
            self._fast_sum += close
            self._fast_val = self._fast_sum / self._fast_period
        else:
            self._fast_val = close * self._fast_k + self._fast_val * (1.0 - self._fast_k)

        # Slow EMA
        self._slow_n += 1
        if self._slow_n < self._slow_period:
            self._slow_sum += close
            return  # Can't compute MACD line yet
        elif self._slow_n == self._slow_period:
            self._slow_sum += close
            self._slow_val = self._slow_sum / self._slow_period
        else:
            self._slow_val = close * self._slow_k + self._slow_val * (1.0 - self._slow_k)

        macd = self._fast_val - self._slow_val

        # Signal EMA
        if not self._sig_ready:
            self._seed.append(macd)
            if len(self._seed) == self._sig_period:
                self._sig_val = sum(self._seed) / self._sig_period
                self._sig_ready = True
                self.values[t] = macd - self._sig_val
        else:
            self._sig_val = macd * self._sig_k + self._sig_val * (1.0 - self._sig_k)
            self.values[t] = macd - self._sig_val
