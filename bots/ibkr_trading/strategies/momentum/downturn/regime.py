"""Downturn Dominator regime classification -- composite regime + vol state."""
from __future__ import annotations

import numpy as np

from .bt_models import (
    CompositeRegime,
    Regime4H,
    VolState,
)


def classify_4h_regime(
    adx_4h: float,
    slope_4h: float,
    slope_threshold_4h: float = 0.0,
    adx_trending_threshold: float = 25.0,
    adx_range_threshold: float = 15.0,
) -> Regime4H:
    """Classify 4H regime.  (Spec S2.3)

    Args:
        adx_4h: ADX value on 4H timeframe
        slope_4h: EMA50 slope on 4H (positive=up, negative=down)
        slope_threshold_4h: Minimum absolute slope for trending
        adx_trending_threshold: ADX value above which regime is trending
        adx_range_threshold: ADX value below which regime is range
    """
    if adx_4h >= adx_trending_threshold and abs(slope_4h) > slope_threshold_4h:
        return Regime4H.TRENDING
    elif adx_4h <= adx_range_threshold:
        return Regime4H.RANGE
    return Regime4H.TRANSITIONAL


def classify_daily_trend(
    close: float,
    sma200: float,
    prev_trend: int,
    consec_count: int,
) -> tuple[int, int]:
    """Classify daily trend with 2-bar persistence.  (Spec S2.4)

    Returns (trend, new_consec_count) where trend is +1 (bull) or -1 (bear).
    """
    raw = 1 if close >= sma200 else -1
    if raw != prev_trend:
        # Count consecutive bars in new direction
        new_count = consec_count + 1 if consec_count > 0 and raw != prev_trend else 1
        if new_count >= 2:
            return raw, 0
        return prev_trend, new_count
    return raw, 0


def compute_composite_regime(
    regime_4h: Regime4H,
    daily_trend: int,
    slope_direction: int,
    short_trend: int = 0,
) -> CompositeRegime:
    """Compute composite regime from 4H regime + daily trend.  (Spec S2.5)

    Args:
        regime_4h: 4H regime classification
        daily_trend: +1 (bull) or -1 (bear)
        slope_direction: +1 (EMA slope up) or -1 (EMA slope down)
        short_trend: -1 if close < short SMA (alternative bear signal)
    """
    # Aligned Bear: 4H trending + daily bear + slope bearish
    if regime_4h == Regime4H.TRENDING and daily_trend == -1 and slope_direction == -1:
        return CompositeRegime.ALIGNED_BEAR

    # Emerging Bear: 4H transitional + daily bear (or slope turning)
    if daily_trend == -1 and regime_4h == Regime4H.TRANSITIONAL:
        return CompositeRegime.EMERGING_BEAR

    # Short-SMA emerging bear: above SMA200 but below short SMA (bull-market correction)
    if short_trend == -1 and daily_trend >= 0:
        return CompositeRegime.EMERGING_BEAR

    # Counter: 4H trending + daily bull (counter-trend short)
    if regime_4h == Regime4H.TRENDING and daily_trend == 1:
        return CompositeRegime.COUNTER

    # Range: 4H range regardless of daily
    if regime_4h == Regime4H.RANGE:
        return CompositeRegime.RANGE

    return CompositeRegime.NEUTRAL


def classify_intraday_regime(
    hourly_ema: float,
    hourly_close: float,
    four_hour_adx: float,
    four_hour_slope: float,
    adx_trending_threshold: float = 25.0,
    adx_range_threshold: float = 15.0,
) -> CompositeRegime:
    """Intraday-only regime classification (no daily bars needed).

    Uses 1H EMA as bear/bull proxy and 4H ADX for trend strength.
    Bypasses daily-bar alignment lag.
    """
    bearish_1h = hourly_close < hourly_ema
    slope_bearish = four_hour_slope < 0

    if four_hour_adx >= adx_trending_threshold and bearish_1h and slope_bearish:
        return CompositeRegime.ALIGNED_BEAR

    if bearish_1h and four_hour_adx >= adx_range_threshold:
        return CompositeRegime.EMERGING_BEAR

    if four_hour_adx >= adx_trending_threshold and not bearish_1h:
        return CompositeRegime.COUNTER

    if four_hour_adx < adx_range_threshold:
        return CompositeRegime.RANGE

    return CompositeRegime.NEUTRAL


def multi_tf_regime_vote(
    hourly_bearish: bool,
    four_hour_bearish: bool,
    daily_bearish: bool,
) -> CompositeRegime:
    """2-of-3 majority vote across timeframes."""
    bear_votes = sum([hourly_bearish, four_hour_bearish, daily_bearish])
    if bear_votes >= 3:
        return CompositeRegime.ALIGNED_BEAR
    if bear_votes >= 2:
        return CompositeRegime.EMERGING_BEAR
    if bear_votes == 1:
        return CompositeRegime.NEUTRAL
    return CompositeRegime.COUNTER


def compute_vol_state(
    atr_pct: float,
    atr_v: float,
    atr_med: float,
) -> VolState:
    """Classify volatility state.  (Spec S2.1)

    Args:
        atr_pct: Percentile rank of current ATR in 60-day window
        atr_v: Current ATR value
        atr_med: 60-day median ATR
    """
    # Shock: extreme vol spike
    shock_pctl = 0.95
    shock_mult = 2.5
    if atr_pct >= shock_pctl or (atr_med > 0 and atr_v >= shock_mult * atr_med):
        return VolState.SHOCK

    # High: elevated but manageable
    high_pctl = 0.80
    if atr_pct >= high_pctl:
        return VolState.HIGH

    return VolState.NORMAL


def compute_vol_factor(
    atr_base: float,
    atr_today: float,
    vol_pct: float,
) -> float:
    """Compute volatility adjustment factor, clamped [0.40, 1.50].  (Spec S9.1)

    Used to adjust position sizing based on current vs baseline volatility.
    """
    if atr_today <= 0:
        return 1.0
    raw = atr_base / atr_today if atr_base > 0 else 1.0
    # Blend with percentile-based adjustment
    pct_adj = 1.0
    if vol_pct > 0.80:
        pct_adj = 0.70
    elif vol_pct > 0.60:
        pct_adj = 0.85
    factor = raw * 0.6 + pct_adj * 0.4
    return max(0.40, min(1.50, factor))


def compute_strong_bear(trend_strength: float, alignment_score: float) -> bool:
    """Determine strong bear flag.  (Spec S2.6)

    Args:
        trend_strength: (EMA_fast - EMA_slow) / ATR (negative = bearish)
        alignment_score: composite regime alignment measure
    """
    return trend_strength < -1.5 and alignment_score > 0.7


def regime_sizing_mult(
    composite_regime: CompositeRegime,
    param_overrides: dict[str, float] | None = None,
) -> float:
    """Regime-based sizing multiplier.  (Spec S2.5 table)"""
    po = param_overrides or {}
    defaults = {
        CompositeRegime.ALIGNED_BEAR: ("regime_mult_aligned", 1.0),
        CompositeRegime.EMERGING_BEAR: ("regime_mult_emerging", 0.75),
        CompositeRegime.NEUTRAL: ("regime_mult_neutral", 0.50),
        CompositeRegime.COUNTER: ("regime_mult_counter", 0.25),
        CompositeRegime.RANGE: ("regime_mult_range", 0.40),
    }
    key, default = defaults.get(composite_regime, ("regime_mult_neutral", 0.50))
    return po.get(key, default)


# ---------------------------------------------------------------------------
# Fast-crash override paths E/F/G
# ---------------------------------------------------------------------------

def check_fast_crash_override(
    daily_closes: np.ndarray,
    ema_fast: float,
    atr_current: float,
    atr_baseline: float,
    param_overrides: dict[str, float] | None = None,
) -> bool:
    """Check if real-time price action warrants EMERGING_BEAR override.

    Three independent paths (any one triggers):
      Path E (Flash Crash): daily_return < crash_daily_threshold AND atr_ratio > crash_atr_ratio
      Path F (Mild Drop):   daily_return < crash_mild_threshold AND close < EMA_fast
      Path G (Grinding):    N-day cumulative return < crash_cumulative_threshold AND close < EMA_fast
    """
    if len(daily_closes) < 2:
        return False

    po = param_overrides or {}
    crash_daily = po.get("crash_daily_threshold", -0.025)
    crash_mild = po.get("crash_mild_threshold", -0.015)
    crash_cum = po.get("crash_cumulative_threshold", -0.03)
    crash_cum_period = int(po.get("crash_cumulative_period", 5))
    crash_atr_ratio = po.get("crash_atr_ratio", 1.5)

    close = daily_closes[-1]
    prev_close = daily_closes[-2]
    daily_return = (close - prev_close) / prev_close if prev_close != 0 else 0.0

    # Path E: Flash crash -- large single-day drop with elevated volatility
    if daily_return < crash_daily:
        atr_ratio = atr_current / atr_baseline if atr_baseline > 0 else 1.0
        if atr_ratio > crash_atr_ratio:
            return True

    # Path F: Mild drop -- moderate drop with price below fast EMA
    if daily_return < crash_mild and close < ema_fast:
        return True

    # Path G: Grinding decline -- cumulative N-day decline
    if len(daily_closes) > crash_cum_period:
        anchor = daily_closes[-(crash_cum_period + 1)]
        cum_return = (close - anchor) / anchor if anchor != 0 else 0.0
        if cum_return < crash_cum and close < ema_fast:
            return True

    return False


# ---------------------------------------------------------------------------
# Bear conviction scoring quality gate
# ---------------------------------------------------------------------------

def compute_bear_conviction(
    adx: float,
    plus_di: float,
    minus_di: float,
    ema_fast: float,
    ema_slow: float,
    close: float,
    prev_ema_fast: float = 0.0,
) -> float:
    """Compute bear conviction score 0-100.

    Components:
      ADX magnitude:        0-30  (ADX 10->0, ADX 40->30, linear)
      -DI dominance:        0-25  ((-DI minus +DI), 0->0, 20->25)
      EMA separation:       0-20  (fast/slow gap %, 0->0, 2.0->20)
      Price below both EMAs: +15
      Fast EMA slope neg:    +10
    """
    score = 0.0

    # ADX magnitude: linear from 10->0 to 40->30
    adx_score = max(0.0, min(30.0, (adx - 10.0) * 30.0 / 30.0))
    score += adx_score

    # -DI dominance: (-DI - +DI), clamped 0-20, scaled to 0-25
    di_gap = max(0.0, minus_di - plus_di)
    di_score = min(25.0, di_gap * 25.0 / 20.0)
    score += di_score

    # EMA separation: |fast - slow| / slow as percentage
    if ema_slow > 0:
        sep_pct = abs(ema_fast - ema_slow) / ema_slow * 100.0
        # Only count if bearish (fast below slow)
        if ema_fast < ema_slow:
            ema_score = min(20.0, sep_pct * 20.0 / 2.0)
            score += ema_score

    # Price below both EMAs
    if close < ema_fast and close < ema_slow:
        score += 15.0

    # Fast EMA slope negative
    if prev_ema_fast > 0 and ema_fast < prev_ema_fast:
        score += 10.0

    return min(100.0, score)


# ---------------------------------------------------------------------------
# ADX hysteresis + bear structure override paths B/C
# ---------------------------------------------------------------------------

def compute_regime_on(
    adx: float,
    prev_regime_on: bool,
    adx_on: float = 25.0,
    adx_off: float = 15.0,
) -> bool:
    """ADX hysteresis: turns ON at adx_on, stays ON until adx < adx_off."""
    if prev_regime_on:
        return adx >= adx_off
    return adx >= adx_on


def check_bear_structure_override(
    adx: float,
    plus_di: float,
    minus_di: float,
    close: float,
    ema_fast: float,
    ema_slow: float,
    regime_on: bool,
    bear_conviction: float,
    param_overrides: dict[str, float] | None = None,
) -> bool:
    """Check if gradual bear structure warrants EMERGING_BEAR override.

    Three checks (any triggers):
      BEAR_FORMING: regime_on + 2-of-3 bear conditions
      Path B: regime_on + bear_conviction >= threshold + ADX >= 20
      Path C: regime_on + DI_gap >= 8 + EMA_sep >= 0.15% + ADX >= 18
    """
    if not regime_on:
        return False

    po = param_overrides or {}
    bear_min_conditions = int(po.get("bear_structure_min_conditions", 2))
    path_b_conviction = po.get("bear_structure_path_b_conviction", 50.0)
    path_b_adx = po.get("bear_structure_path_b_adx", 20.0)
    path_c_di_gap = po.get("bear_structure_path_c_di_gap", 8.0)
    path_c_ema_sep = po.get("bear_structure_path_c_ema_sep", 0.15)
    path_c_adx = po.get("bear_structure_path_c_adx", 18.0)

    # Count bear conditions for regime classification
    bear_conds = sum([
        close < ema_fast,
        ema_fast < ema_slow,
        minus_di > plus_di,
    ])

    # BEAR_FORMING: partial conditions met
    if bear_conds >= bear_min_conditions:
        return True

    # Path B: conviction-fast confirmation
    if bear_conviction >= path_b_conviction and adx >= path_b_adx:
        return True

    # Path C: structural evidence
    di_gap = minus_di - plus_di
    ema_sep_pct = abs(ema_fast - ema_slow) / ema_slow * 100 if ema_slow > 0 else 0.0
    if di_gap >= path_c_di_gap and ema_sep_pct >= path_c_ema_sep and adx >= path_c_adx:
        return True

    return False


# ---------------------------------------------------------------------------
# Real-time drawdown override (R6 -- rolling-high drawdown detection)
# ---------------------------------------------------------------------------

def check_drawdown_override(
    daily_closes: np.ndarray,
    lookback: int = 20,
    threshold: float = 0.03,
) -> bool:
    """Check if current close is >threshold below lookback-day rolling high.

    Unlike correction_regime_override (pre-computed windows), this fires from
    real-time price action the moment the drawdown criterion is met.
    """
    if len(daily_closes) < lookback:
        return False
    rolling_high = float(np.max(daily_closes[-lookback:]))
    if rolling_high <= 0:
        return False
    drawdown = (rolling_high - daily_closes[-1]) / rolling_high
    return drawdown >= threshold
