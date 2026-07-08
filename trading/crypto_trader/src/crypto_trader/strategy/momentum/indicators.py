"""Indicator computations — both batch (numpy) and incremental (O(1) per bar)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from crypto_trader.core.models import Bar
from crypto_trader.strategy.momentum.config import IndicatorParams


@dataclass(frozen=True)
class IndicatorSnapshot:
    ema_fast: float
    ema_mid: float
    ema_slow: float
    ema_fast_arr: np.ndarray
    ema_mid_arr: np.ndarray
    ema_slow_arr: np.ndarray
    adx: float
    di_plus: float
    di_minus: float
    adx_rising: bool
    atr: float
    atr_avg: float
    rsi: float
    volume_ma: float


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average over the full array."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Average True Range using Wilder smoothing."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr_arr = np.empty(n)
    atr_arr[:period] = np.nan
    atr_arr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period
    return atr_arr


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, DI+, DI- using Wilder smoothing. Returns (adx_arr, di_plus_arr, di_minus_arr)."""
    n = len(highs)
    if n < period + 1:
        nans = np.full(n, np.nan)
        return nans, nans, nans

    # Directional movement
    up_move = np.diff(highs)
    down_move = -np.diff(lows)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True range on diff-aligned array (len n-1)
    tr = np.empty(n - 1)
    for i in range(n - 1):
        tr[i] = max(
            highs[i + 1] - lows[i + 1],
            abs(highs[i + 1] - closes[i]),
            abs(lows[i + 1] - closes[i]),
        )

    # Wilder smooth
    def _wilder_smooth(data: np.ndarray, p: int) -> np.ndarray:
        out = np.empty(len(data))
        out[:p] = np.nan
        out[p - 1] = np.sum(data[:p])
        for i in range(p, len(data)):
            out[i] = out[i - 1] - out[i - 1] / p + data[i]
        return out

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus_raw = np.where(smoothed_tr > 0, 100.0 * smoothed_plus / smoothed_tr, 0.0)
        di_minus_raw = np.where(smoothed_tr > 0, 100.0 * smoothed_minus / smoothed_tr, 0.0)

        di_sum = di_plus_raw + di_minus_raw
        dx = np.where(
            di_sum > 0,
            100.0 * np.abs(di_plus_raw - di_minus_raw) / di_sum,
            0.0,
        )

    # ADX = Wilder smooth of DX
    adx_arr = np.full(len(dx), np.nan)
    start = period - 1 + period  # need period DX values starting from index period-1
    if start <= len(dx):
        adx_arr[start - 1] = np.mean(dx[period - 1:start])
        for i in range(start, len(dx)):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    # Pad back to length n (prepend one NaN for the diff offset)
    pad = lambda a: np.concatenate(([np.nan], a))
    return pad(adx_arr), pad(di_plus_raw), pad(di_minus_raw)


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI using Wilder smoothing."""
    n = len(closes)
    rsi_arr = np.full(n, np.nan)
    if n < period + 1:
        return rsi_arr

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_arr[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_arr[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_arr[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_arr[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi_arr


def compute_indicators(bars: list[Bar], params: IndicatorParams) -> IndicatorSnapshot | None:
    """Compute all indicators from a list of bars. Returns None if insufficient data."""
    if len(bars) < params.ema_slow + 1:
        return None

    # Cap computation window — recursive indicators (EMA/ADX/ATR) have exponential
    # decay; bars beyond 3× the longest period have < 2% weight.  Without this cap,
    # recomputing on the full history every bar is O(N²) over the backtest.
    max_window = params.ema_slow * 3
    if len(bars) > max_window:
        bars = bars[-max_window:]

    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume for b in bars])

    ema_fast_arr = _ema(closes, params.ema_fast)
    ema_mid_arr = _ema(closes, params.ema_mid)
    ema_slow_arr = _ema(closes, params.ema_slow)

    adx_arr, di_plus_arr, di_minus_arr = _adx(highs, lows, closes, params.adx_period)
    atr_arr = _atr(highs, lows, closes, params.atr_period)
    rsi_arr = _rsi(closes, params.rsi_period)

    # Volume MA
    if len(volumes) >= params.volume_ma_period:
        volume_ma = float(np.mean(volumes[-params.volume_ma_period:]))
    else:
        volume_ma = float(np.mean(volumes))

    # ATR average (for expansion/compression detection)
    valid_atr = atr_arr[~np.isnan(atr_arr)]
    if len(valid_atr) >= params.atr_avg_period:
        atr_avg = float(np.mean(valid_atr[-params.atr_avg_period:]))
    else:
        atr_avg = float(np.mean(valid_atr)) if len(valid_atr) > 0 else 0.0

    # ADX rising: current > 2 bars ago
    adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
    adx_prev = float(adx_arr[-3]) if len(adx_arr) >= 3 and not np.isnan(adx_arr[-3]) else adx_val
    adx_rising = adx_val > adx_prev

    # Recent EMA arrays for slope detection (last 10 bars)
    tail = 10

    return IndicatorSnapshot(
        ema_fast=float(ema_fast_arr[-1]),
        ema_mid=float(ema_mid_arr[-1]),
        ema_slow=float(ema_slow_arr[-1]),
        ema_fast_arr=ema_fast_arr[-tail:],
        ema_mid_arr=ema_mid_arr[-tail:],
        ema_slow_arr=ema_slow_arr[-tail:],
        adx=adx_val,
        di_plus=float(di_plus_arr[-1]) if not np.isnan(di_plus_arr[-1]) else 0.0,
        di_minus=float(di_minus_arr[-1]) if not np.isnan(di_minus_arr[-1]) else 0.0,
        adx_rising=adx_rising,
        atr=float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0,
        atr_avg=atr_avg,
        rsi=float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0,
        volume_ma=volume_ma,
    )


class IncrementalIndicators:
    """O(1) per-bar indicator computation using running Wilder/EMA state.

    Drop-in replacement for repeated compute_indicators() calls.
    During warmup (first ema_slow bars), update() returns None.
    After warmup, each call is O(1) instead of O(window_size).
    """

    __slots__ = (
        "_p", "_n", "_af", "_am", "_as",
        "_ef", "_em", "_es", "_ef_buf", "_em_buf", "_es_buf",
        "_ph", "_pl", "_pc",
        "_atr", "_atr_ready", "_tr_warmup", "_atr_hist",
        "_smo_tr", "_smo_pdm", "_smo_mdm", "_di_p", "_di_m",
        "_adx", "_adx_ready", "_di_ready",
        "_adx_hist",
        "_dm_w_tr", "_dm_w_pdm", "_dm_w_mdm", "_dx_w",
        "_avg_gain", "_avg_loss", "_rsi", "_rsi_ready",
        "_rsi_w_g", "_rsi_w_l",
        "_vbuf", "_min_bars",
    )

    def __init__(self, params: IndicatorParams) -> None:
        self._p = params
        self._n = 0

        # EMA alphas
        self._af = 2.0 / (params.ema_fast + 1)
        self._am = 2.0 / (params.ema_mid + 1)
        self._as = 2.0 / (params.ema_slow + 1)

        # Running EMA values
        self._ef = self._em = self._es = 0.0

        # Tail buffers (last 10 for slope detection)
        self._ef_buf: deque[float] = deque(maxlen=10)
        self._em_buf: deque[float] = deque(maxlen=10)
        self._es_buf: deque[float] = deque(maxlen=10)

        # Previous bar
        self._ph = self._pl = self._pc = 0.0

        # ATR (Wilder smoothing)
        self._atr = 0.0
        self._atr_ready = False
        self._tr_warmup: list[float] = []
        self._atr_hist: deque[float] = deque(maxlen=params.atr_avg_period)

        # ADX (Wilder smoothing of +DM, -DM, TR, then DX)
        self._smo_tr = self._smo_pdm = self._smo_mdm = 0.0
        self._di_p = self._di_m = self._adx = 0.0
        self._adx_ready = False
        self._di_ready = False
        self._adx_hist: deque[float] = deque(maxlen=3)
        self._dm_w_tr: list[float] = []
        self._dm_w_pdm: list[float] = []
        self._dm_w_mdm: list[float] = []
        self._dx_w: list[float] = []

        # RSI (Wilder smoothing)
        self._avg_gain = self._avg_loss = 0.0
        self._rsi = 50.0
        self._rsi_ready = False
        self._rsi_w_g: list[float] = []
        self._rsi_w_l: list[float] = []

        # Volume MA
        self._vbuf: deque[float] = deque(maxlen=params.volume_ma_period)

        # Minimum bars before valid snapshot
        self._min_bars = params.ema_slow + 1

    def update(self, bar: Bar) -> IndicatorSnapshot | None:
        """Process one bar, return snapshot or None during warmup."""
        self._n += 1
        c, h, l, v = bar.close, bar.high, bar.low, bar.volume

        # ── EMA ──
        if self._n == 1:
            self._ef = self._em = self._es = c
        else:
            self._ef = self._af * c + (1.0 - self._af) * self._ef
            self._em = self._am * c + (1.0 - self._am) * self._em
            self._es = self._as * c + (1.0 - self._as) * self._es

        self._ef_buf.append(self._ef)
        self._em_buf.append(self._em)
        self._es_buf.append(self._es)

        # ── Volume ──
        self._vbuf.append(v)

        # ── ATR / ADX / RSI (need previous bar) ──
        if self._n >= 2:
            tr = max(h - l, abs(h - self._pc), abs(l - self._pc))
            self._step_atr(tr)
            self._step_adx(h, l, tr)
            self._step_rsi(c - self._pc)

        self._ph, self._pl, self._pc = h, l, c

        # ── Warmup gate ──
        if self._n < self._min_bars:
            return None

        # ── Build snapshot ──
        atr_val = self._atr if self._atr_ready else 0.0
        atr_avg = (sum(self._atr_hist) / len(self._atr_hist)) if self._atr_hist else 0.0

        adx_val = self._adx if self._adx_ready else 0.0
        self._adx_hist.append(adx_val)
        adx_prev = self._adx_hist[0] if len(self._adx_hist) >= 3 else adx_val

        return IndicatorSnapshot(
            ema_fast=self._ef,
            ema_mid=self._em,
            ema_slow=self._es,
            ema_fast_arr=np.array(self._ef_buf),
            ema_mid_arr=np.array(self._em_buf),
            ema_slow_arr=np.array(self._es_buf),
            adx=adx_val,
            di_plus=self._di_p if self._di_ready else 0.0,
            di_minus=self._di_m if self._di_ready else 0.0,
            adx_rising=adx_val > adx_prev,
            atr=atr_val,
            atr_avg=atr_avg,
            rsi=self._rsi if self._rsi_ready else 50.0,
            volume_ma=sum(self._vbuf) / len(self._vbuf),
        )

    # -- Internal Wilder/EMA stepping -----------------------------------

    def _step_atr(self, tr: float) -> None:
        p = self._p.atr_period
        if not self._atr_ready:
            self._tr_warmup.append(tr)
            if len(self._tr_warmup) == p:
                self._atr = sum(self._tr_warmup) / p
                self._atr_ready = True
                self._atr_hist.append(self._atr)
                self._tr_warmup = []
        else:
            self._atr = (self._atr * (p - 1) + tr) / p
            self._atr_hist.append(self._atr)

    def _step_adx(self, h: float, l: float, tr: float) -> None:
        p = self._p.adx_period
        up = h - self._ph
        dn = self._pl - l
        pdm = up if (up > dn and up > 0) else 0.0
        mdm = dn if (dn > up and dn > 0) else 0.0

        if not self._di_ready:
            self._dm_w_tr.append(tr)
            self._dm_w_pdm.append(pdm)
            self._dm_w_mdm.append(mdm)
            if len(self._dm_w_tr) == p:
                self._smo_tr = sum(self._dm_w_tr)
                self._smo_pdm = sum(self._dm_w_pdm)
                self._smo_mdm = sum(self._dm_w_mdm)
                self._di_ready = True
                self._dm_w_tr = []
                self._dm_w_pdm = []
                self._dm_w_mdm = []
                self._di_dx_step(p)
        else:
            self._smo_tr = self._smo_tr - self._smo_tr / p + tr
            self._smo_pdm = self._smo_pdm - self._smo_pdm / p + pdm
            self._smo_mdm = self._smo_mdm - self._smo_mdm / p + mdm
            self._di_dx_step(p)

    def _di_dx_step(self, period: int) -> None:
        if self._smo_tr > 0:
            self._di_p = 100.0 * self._smo_pdm / self._smo_tr
            self._di_m = 100.0 * self._smo_mdm / self._smo_tr
        else:
            self._di_p = self._di_m = 0.0

        di_sum = self._di_p + self._di_m
        dx = 100.0 * abs(self._di_p - self._di_m) / di_sum if di_sum > 0 else 0.0

        if not self._adx_ready:
            self._dx_w.append(dx)
            if len(self._dx_w) == period:
                self._adx = sum(self._dx_w) / period
                self._adx_ready = True
                self._dx_w = []
        else:
            self._adx = (self._adx * (period - 1) + dx) / period

    def _step_rsi(self, delta: float) -> None:
        p = self._p.rsi_period
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0

        if not self._rsi_ready:
            self._rsi_w_g.append(gain)
            self._rsi_w_l.append(loss)
            if len(self._rsi_w_g) == p:
                self._avg_gain = sum(self._rsi_w_g) / p
                self._avg_loss = sum(self._rsi_w_l) / p
                self._rsi_ready = True
                self._rsi_w_g = []
                self._rsi_w_l = []
                self._rsi = self._compute_rsi()
        else:
            self._avg_gain = (self._avg_gain * (p - 1) + gain) / p
            self._avg_loss = (self._avg_loss * (p - 1) + loss) / p
            self._rsi = self._compute_rsi()

    def _compute_rsi(self) -> float:
        if self._avg_loss == 0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - 100.0 / (1.0 + rs)
