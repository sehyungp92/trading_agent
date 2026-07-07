"""ETRS vFinal — pure indicator computation (no side effects).

All functions accept numpy arrays and return scalars or dataclasses.
"""
from __future__ import annotations

import numpy as np

from .config import (
    ADX_MIN_STRUCT,
    ADX_STRONG,
    ADX_STRONG_SLOPE_FLOOR,
    CONFIRM_DAYS_NORMAL,
    DI_MIN,
    FAST_CONFIRM_ADX,
    FAST_CONFIRM_SCORE,
    PULLBACK_LOOKBACK,
    PULLBACK_TOUCH_TOLERANCE_ATR,
    PULLBACK_TOUCH_TOLERANCE_PCT,
    SEP_MIN,
    SymbolConfig,
)
from .models import DailyState, Direction, HourlyState, Regime


# ---------------------------------------------------------------------------
# Primitive indicators
# ---------------------------------------------------------------------------

def ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average over *arr* with given *period*.

    Uses the standard multiplier ``2 / (period + 1)`` and seeds the first
    value with the SMA of the first *period* elements when enough data is
    available.
    """
    out = np.empty_like(arr, dtype=float)
    k = 2.0 / (period + 1)
    # Seed with SMA(period) when enough data is available
    seed_len = min(period, len(arr))
    out[0] = float(np.mean(arr[:seed_len]))
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def ema_last(arr: np.ndarray, period: int) -> float:
    """Return only the last EMA value (avoids full array allocation).

    Mathematically identical to ``ema(arr, period)[-1]``.
    """
    k = 2.0 / (period + 1)
    one_minus_k = 1.0 - k
    seed_len = min(period, len(arr))
    prev = float(np.mean(arr[:seed_len]))
    for i in range(1, len(arr)):
        prev = float(arr[i]) * k + prev * one_minus_k
    return prev


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
    # Wilder smoothing
    out = np.empty(n, dtype=float)
    out[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
    return out


def atr_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int) -> float:
    """Return only the last ATR value (avoids full array allocation).

    Mathematically identical to ``atr(highs, lows, closes, period)[-1]``.
    """
    n = len(highs)
    prev_close = highs[0] - lows[0]  # tr[0] seed = first range
    out_prev = prev_close
    alpha = 1.0 / period
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n):
        tr_i = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        out_prev = out_prev * one_minus_alpha + tr_i * alpha
    return float(out_prev)


def adx_suite(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder's ADX → ``(adx, plus_di, minus_di)`` arrays."""
    n = len(highs)
    alpha = 1.0 / period

    # True Range
    tr = np.empty(n, dtype=float)
    plus_dm = np.empty(n, dtype=float)
    minus_dm = np.empty(n, dtype=float)

    tr[0] = highs[0] - lows[0]
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    for i in range(1, n):
        hi_diff = highs[i] - highs[i - 1]
        lo_diff = lows[i - 1] - lows[i]
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        plus_dm[i] = hi_diff if (hi_diff > lo_diff and hi_diff > 0) else 0.0
        minus_dm[i] = lo_diff if (lo_diff > hi_diff and lo_diff > 0) else 0.0

    # Wilder-smoothed TR, +DM, -DM
    s_tr = np.empty(n, dtype=float)
    s_plus = np.empty(n, dtype=float)
    s_minus = np.empty(n, dtype=float)
    s_tr[0] = tr[0]
    s_plus[0] = plus_dm[0]
    s_minus[0] = minus_dm[0]
    for i in range(1, n):
        s_tr[i] = s_tr[i - 1] * (1 - alpha) + tr[i] * alpha
        s_plus[i] = s_plus[i - 1] * (1 - alpha) + plus_dm[i] * alpha
        s_minus[i] = s_minus[i - 1] * (1 - alpha) + minus_dm[i] * alpha

    # +DI, -DI -- use safe division to suppress numpy warnings
    safe_tr = np.where(s_tr > 0, s_tr, 1.0)
    plus_di = np.where(s_tr > 0, 100.0 * s_plus / safe_tr, 0.0)
    minus_di = np.where(s_tr > 0, 100.0 * s_minus / safe_tr, 0.0)

    # DX → ADX (Wilder smooth of DX)
    di_sum = plus_di + minus_di
    safe_di_sum = np.where(di_sum > 0, di_sum, 1.0)
    dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / safe_di_sum, 0.0)

    adx_arr = np.empty(n, dtype=float)
    adx_arr[0] = dx[0]
    for i in range(1, n):
        adx_arr[i] = adx_arr[i - 1] * (1 - alpha) + dx[i] * alpha

    return adx_arr, plus_di, minus_di


# ---------------------------------------------------------------------------
# Composite daily state
# ---------------------------------------------------------------------------

def compute_daily_state(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    prev: DailyState | None,
    cfg: SymbolConfig,
    daily_bar_date: str | None = None,
) -> DailyState:
    """Compute all daily indicators and return a new ``DailyState``.

    *closes*, *highs*, *lows* should contain enough history for the longest
    look-back (55 bars minimum).  The last element is "today".
    """
    ema_f = ema(closes, cfg.daily_ema_fast)
    ema_s = ema(closes, cfg.daily_ema_slow)
    atr_arr = atr(highs, lows, closes, cfg.atr_daily_period)
    adx_arr, pdi, mdi = adx_suite(highs, lows, closes, cfg.adx_period)

    cur_close = float(closes[-1])
    cur_ema_f = float(ema_f[-1])
    cur_ema_s = float(ema_s[-1])
    cur_atr = float(atr_arr[-1])
    cur_adx = float(adx_arr[-1])
    cur_pdi = float(pdi[-1])
    cur_mdi = float(mdi[-1])

    # ADX slope over last 3 bars
    if len(adx_arr) >= 4:
        adx_slope_3 = float(adx_arr[-1] - adx_arr[-4])
    else:
        adx_slope_3 = 0.0

    # EMA fast slope over 5 bars (spec Section 2.1)
    if len(ema_f) >= 6:
        ema_fast_slope_5 = float(ema_f[-1] - ema_f[-6])
    else:
        ema_fast_slope_5 = 0.0

    # Regime hysteresis (per-symbol thresholds, spec Section 2.2)
    was_on = prev.regime_on if prev else False
    if cur_adx >= cfg.adx_on:
        regime_on = True
    elif cur_adx < cfg.adx_off:
        regime_on = False
    else:
        regime_on = was_on

    # Regime classification
    if regime_on and cur_adx >= ADX_STRONG and adx_slope_3 > ADX_STRONG_SLOPE_FLOOR:
        regime = Regime.STRONG_TREND
    elif regime_on:
        regime = Regime.TREND
    else:
        regime = Regime.RANGE

    # EMA separation %
    ema_sep_pct = abs(cur_ema_f - cur_ema_s) / cur_close * 100 if cur_close > 0 else 0.0

    # DI diff
    di_diff = abs(cur_pdi - cur_mdi)

    # Score 0-100
    score = (
        min(max((cur_adx - cfg.adx_on) * 2, 0), 30)
        + min(ema_sep_pct * 10, 30)
        + min(di_diff, 40)
    )

    # Raw bias from EMA structure only (DI no longer a hard gate)
    if cur_ema_f > cur_ema_s and cur_close > cur_ema_f:
        raw_bias = Direction.LONG
    elif cur_ema_f < cur_ema_s and cur_close < cur_ema_f:
        raw_bias = Direction.SHORT
    else:
        raw_bias = Direction.FLAT

    # DI modifies score; blocks only in weak regimes (ADX < adx_off)
    if raw_bias == Direction.LONG:
        if cur_pdi > cur_mdi:
            score += 5
        elif cur_adx < cfg.adx_off:
            raw_bias = Direction.FLAT
        else:
            score -= 5
    elif raw_bias == Direction.SHORT:
        if cur_mdi > cur_pdi:
            score += 5
        elif cur_adx < cfg.adx_off:
            raw_bias = Direction.FLAT
        else:
            score -= 5

    # Trend carry: if raw_bias flickered FLAT but yesterday was confirmed,
    # carry forward unless EMA structure breaks or price deeply violates.
    if raw_bias == Direction.FLAT and prev is not None and prev.trend_dir != Direction.FLAT:
        prev_sign = 1 if prev.ema_fast > prev.ema_slow else -1
        curr_sign = 1 if cur_ema_f > cur_ema_s else -1
        ema_crossed = prev_sign != curr_sign

        if prev.trend_dir == Direction.LONG:
            deep_violation = cur_close < cur_ema_f - 0.5 * cur_atr
        else:
            deep_violation = cur_close > cur_ema_f + 0.5 * cur_atr

        if not ema_crossed and not deep_violation:
            raw_bias = prev.trend_dir  # carry forward

    # Confirmed trend direction: hold_count >= 2 consecutive same raw bias + regime ON
    # Only increment hold_count when a new daily bar appears (spec S1.5).
    prev_raw = prev.raw_bias if prev else Direction.FLAT
    prev_hold = prev.hold_count if prev else 0
    prev_bar_date = prev.last_daily_bar_date if prev else None

    is_new_daily_bar = daily_bar_date is not None and daily_bar_date != prev_bar_date

    if raw_bias != Direction.FLAT and raw_bias == prev_raw:
        hold_count = (prev_hold + 1) if is_new_daily_bar else prev_hold
    elif raw_bias != Direction.FLAT:
        hold_count = 1
    else:
        hold_count = 0

    if regime_on and raw_bias != Direction.FLAT:
        if regime == Regime.STRONG_TREND and hold_count >= 0:
            trend_dir = raw_bias                     # instant confirm in STRONG_TREND
        elif hold_count >= CONFIRM_DAYS_NORMAL:
            trend_dir = raw_bias                     # standard N-day confirm
        elif hold_count >= 1 and score >= FAST_CONFIRM_SCORE and cur_adx >= FAST_CONFIRM_ADX:
            trend_dir = raw_bias                     # fast 1-day confirm (spec S2.6)
        elif hold_count >= 1 and di_diff >= DI_MIN and ema_sep_pct >= SEP_MIN and cur_adx >= ADX_MIN_STRUCT:
            trend_dir = raw_bias                     # Path C confirmation (spec S2.6)
        else:
            trend_dir = Direction.FLAT
    else:
        trend_dir = Direction.FLAT

    # HH/LL over last 20 daily bars (for chandelier trailing)
    dc_period = cfg.atr_daily_period  # 20
    if len(highs) >= dc_period:
        hh_20d = float(np.max(highs[-dc_period:]))
        ll_20d = float(np.min(lows[-dc_period:]))
    else:
        hh_20d = float(np.max(highs))
        ll_20d = float(np.min(lows))

    return DailyState(
        ema_fast=cur_ema_f,
        ema_slow=cur_ema_s,
        adx=cur_adx,
        plus_di=cur_pdi,
        minus_di=cur_mdi,
        atr20=cur_atr,
        regime=regime,
        trend_dir=trend_dir,
        score=score,
        ema_sep_pct=ema_sep_pct,
        di_diff=di_diff,
        adx_slope_3=adx_slope_3,
        raw_bias=raw_bias,
        raw_bias_prev=prev_raw,
        hold_count=hold_count,
        regime_on=regime_on,
        ema_fast_slope_5=ema_fast_slope_5,
        hh_20d=hh_20d,
        ll_20d=ll_20d,
        last_daily_bar_date=daily_bar_date,
    )


# ---------------------------------------------------------------------------
# Composite hourly state
# ---------------------------------------------------------------------------

def compute_hourly_state(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    daily: DailyState,
    cfg: SymbolConfig,
    bar_time: object | None = None,
    opens: np.ndarray | None = None,
) -> HourlyState:
    """Compute hourly indicators for the most recent bar.

    *closes*, *highs*, *lows* should contain at least ``max(48, 50, 20)``
    hourly bars.  The last element is "this bar".
    """
    # EMA pull — adaptive to regime (need full array for pullback touch check)
    pull_period = (
        cfg.ema_pull_strong
        if daily.regime == Regime.STRONG_TREND
        else cfg.ema_pull_normal
    )
    ema_pull_arr = ema(closes, pull_period)
    # EMA mom and ATR: only last value used — skip full array allocation
    cur_ema_mom = ema_last(closes, cfg.ema_mom_period)
    cur_atrh = atr_last(highs, lows, closes, cfg.atr_hourly_period)

    # Donchian channel over last donchian_period bars (excluding current bar)
    dc_lookback = cfg.donchian_period
    if len(highs) > dc_lookback:
        dc_highs = highs[-(dc_lookback + 1):-1]
        dc_lows = lows[-(dc_lookback + 1):-1]
    else:
        dc_highs = highs[:-1] if len(highs) > 1 else highs
        dc_lows = lows[:-1] if len(lows) > 1 else lows

    donchian_high = float(np.max(dc_highs)) if len(dc_highs) > 0 else float(highs[-1])
    donchian_low = float(np.min(dc_lows)) if len(dc_lows) > 0 else float(lows[-1])

    # Prior bar high/low
    if len(highs) >= 2:
        prior_high = float(highs[-2])
        prior_low = float(lows[-2])
    else:
        prior_high = float(highs[-1])
        prior_low = float(lows[-1])

    cur_close = float(closes[-1])

    # Distance to daily EMA_fast in ATR units
    if daily.atr20 > 0:
        dist_atr = abs(cur_close - daily.ema_fast) / daily.atr20
    else:
        dist_atr = 0.0

    # Multi-bar pullback touch: check if low/high touched ema_pull
    # within the last PULLBACK_LOOKBACK bars (including current)
    # Tolerance allows near-touches (within fraction of ATR_hourly)
    lb = min(PULLBACK_LOOKBACK, len(lows))
    touch_tol = max(PULLBACK_TOUCH_TOLERANCE_ATR * cur_atrh, PULLBACK_TOUCH_TOLERANCE_PCT * cur_close)
    recent_pull_touch_long = bool(np.any(lows[-lb:] <= ema_pull_arr[-lb:] + touch_tol))
    recent_pull_touch_short = bool(np.any(highs[-lb:] >= ema_pull_arr[-lb:] - touch_tol))

    return HourlyState(
        time=bar_time,
        open=float(opens[-1]) if opens is not None else float(closes[-1]),
        high=float(highs[-1]),
        low=float(lows[-1]),
        close=cur_close,
        ema_mom=cur_ema_mom,
        ema_pull=float(ema_pull_arr[-1]),
        atrh=cur_atrh,
        donchian_high=donchian_high,
        donchian_low=donchian_low,
        prior_high=prior_high,
        prior_low=prior_low,
        dist_atr=dist_atr,
        recent_pull_touch_long=recent_pull_touch_long,
        recent_pull_touch_short=recent_pull_touch_short,
    )
