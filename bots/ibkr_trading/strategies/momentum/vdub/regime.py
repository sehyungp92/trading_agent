"""Vdubus NQ v4.0 — regime classification, vol state, direction permissions."""
from __future__ import annotations

import numpy as np

from . import config as C
from .indicators import atr, sma, ema, percentile_rank, median_val
from .models import Direction, DayCounters, RegimeState, VolState


# ---------------------------------------------------------------------------
# Persisted trend (2-close confirmation)
# ---------------------------------------------------------------------------

def update_persisted_trend(
    current_persisted: int, last_raw: int, raw: int, streak: int,
    persist_bars: int = C.TREND_PERSIST_BARS,
) -> tuple[int, int, int]:
    """Returns (new_persisted, new_raw, new_streak)."""
    if raw == last_raw:
        streak += 1
    else:
        streak = 1
    if streak >= persist_bars and raw != current_persisted:
        return raw, raw, streak
    return current_persisted, raw, streak


# ---------------------------------------------------------------------------
# Daily trend (ES SMA200)
# ---------------------------------------------------------------------------

def compute_daily_trend(
    es_closes: np.ndarray, regime: RegimeState,
) -> None:
    """Update regime.daily_trend in-place using SMA200 persistence."""
    sma200 = sma(es_closes, C.DAILY_SMA_PERIOD)
    if np.isnan(sma200[-1]):
        return
    raw = 1 if es_closes[-1] > sma200[-1] else -1
    old_trend = regime.daily_trend
    regime.daily_trend, regime.last_daily_raw, regime.daily_raw_streak = (
        update_persisted_trend(
            regime.daily_trend, regime.last_daily_raw, raw,
            regime.daily_raw_streak, C.TREND_PERSIST_BARS))
    regime.flip_just_happened = (regime.daily_trend != 0 and
                                 regime.daily_trend != old_trend and
                                 old_trend != 0)
    if regime.flip_just_happened:
        regime.daily_trend_prev = old_trend


# ---------------------------------------------------------------------------
# Vol state (ES daily)
# ---------------------------------------------------------------------------

def compute_vol_state(es_highs: np.ndarray, es_lows: np.ndarray,
                      es_closes: np.ndarray) -> VolState:
    """Shock / High / Normal from ES daily ATR."""
    atrv = atr(es_highs, es_lows, es_closes, C.VOL_ATR_PERIOD)
    if np.isnan(atrv[-1]):
        return VolState.NORMAL
    pctl = percentile_rank(atrv[-1], atrv, C.VOL_LOOKBACK)
    med = median_val(atrv, C.VOL_LOOKBACK)
    if pctl > C.SHOCK_PCTL and atrv[-1] > C.SHOCK_MED_MULT * med:
        return VolState.SHOCK
    if pctl > C.HIGH_PCTL:
        return VolState.HIGH
    return VolState.NORMAL


# ---------------------------------------------------------------------------
# 1H tactical trend (NQ EMA50)
# ---------------------------------------------------------------------------

def compute_1h_trend(nq_closes_1h: np.ndarray, regime: RegimeState) -> None:
    """Update regime.trend_1h in-place."""
    ema50 = ema(nq_closes_1h, C.HOURLY_EMA_PERIOD)
    if np.isnan(ema50[-1]):
        return
    raw = 1 if nq_closes_1h[-1] > ema50[-1] else -1
    regime.trend_1h, regime.last_hourly_raw, regime.hourly_raw_streak = (
        update_persisted_trend(
            regime.trend_1h, regime.last_hourly_raw, raw,
            regime.hourly_raw_streak, C.HOURLY_PERSIST_BARS))


# ---------------------------------------------------------------------------
# Direction permission (Section 4)
# ---------------------------------------------------------------------------

def compute_choppiness(highs_1h: np.ndarray, lows_1h: np.ndarray,
                       closes_1h: np.ndarray, period: int = 20) -> float:
    """Choppiness Index (0-100). Higher = choppier / range-bound.

    CI = 100 * log10(sum_of_true_ranges / (highest_high - lowest_low)) / log10(period)
    Values > 61.8 suggest choppy market; < 38.2 suggest trending.
    """
    if len(closes_1h) < period + 1:
        return 50.0  # neutral default
    h = highs_1h[-period:]
    l = lows_1h[-period:]
    c = closes_1h[-(period + 1):-1]  # previous closes for TR calc

    # True range for each bar
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    sum_tr = float(np.nansum(tr))
    hh = float(np.nanmax(h))
    ll = float(np.nanmin(l))
    rng = hh - ll
    if rng <= 0 or sum_tr <= 0:
        return 50.0

    ci = 100.0 * np.log10(sum_tr / rng) / np.log10(period)
    return float(np.clip(ci, 0, 100))


def direction_allowed(regime: RegimeState, direction: Direction) -> bool:
    """Hard gate: DailyTrend must match, Shock must be false."""
    if regime.vol_state == VolState.SHOCK:
        return False
    if direction == Direction.LONG:
        return regime.daily_trend == 1
    if direction == Direction.SHORT:
        return regime.daily_trend == -1
    return False


def flip_entry_eligible(
    regime: RegimeState, counters: DayCounters, direction: Direction,
) -> bool:
    """Flip entry exception (Section 4.2)."""
    if not regime.flip_just_happened:
        return False
    if regime.vol_state == VolState.SHOCK:
        return False
    if direction == Direction.LONG:
        return regime.daily_trend == 1 and not counters.flip_entry_used_long
    if direction == Direction.SHORT:
        return regime.daily_trend == -1 and not counters.flip_entry_used_short
    return False
