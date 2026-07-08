"""NQ Dominant Trend Capture v2.0 — breakout qualification, entry triggers, scorecard."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import config as C
from .indicators import displacement_metric, macd_hist, rolling_quantile_past_only, compute_rvol, percentile_rank
from .models import (
    BoxEngineState, BoxState, BreakoutEngineState, BreakoutState,
    ChopMode, CompositeRegime, Direction, EntrySubtype,
    RegimeState, SessionEngineState,
)


# ---------------------------------------------------------------------------
# Regime classification (Section 9)
# ---------------------------------------------------------------------------

def classify_4h(
    ema50_4h: np.ndarray, atr14_4h: np.ndarray, adx14_4h: np.ndarray,
) -> tuple[str, Direction, float, float]:
    """Classify 4H regime. Returns (regime_str, trend_dir, slope, adx_val)."""
    if len(ema50_4h) < 4 or np.isnan(ema50_4h[-1]) or np.isnan(adx14_4h[-1]):
        return "TRANSITIONAL", Direction.FLAT, 0.0, 0.0

    slope = float(ema50_4h[-1] - ema50_4h[-4])
    adx_val = float(adx14_4h[-1])
    threshold = C.K_SLOPE * float(atr14_4h[-1]) if not np.isnan(atr14_4h[-1]) else 0.0

    if adx_val > C.ADX_TRENDING and abs(slope) > threshold:
        trend_dir = Direction.LONG if slope > 0 else Direction.SHORT
        return "TRENDING", trend_dir, slope, adx_val

    sma_atr = np.nanmean(atr14_4h[-50:]) if len(atr14_4h) >= 50 else np.nanmean(atr14_4h)
    if adx_val < C.ADX_RANGE and sma_atr > 0 and float(atr14_4h[-1]) / sma_atr < 1.0:
        return "RANGE", Direction.FLAT, slope, adx_val

    return "TRANSITIONAL", Direction.FLAT, slope, adx_val


def classify_daily_support(
    ema50_d: np.ndarray, atr14_d: np.ndarray, trade_dir: Direction,
) -> tuple[bool, bool]:
    """Returns (daily_supports, daily_opposes) for the given trade direction."""
    if len(ema50_d) < 4 or np.isnan(ema50_d[-1]) or np.isnan(atr14_d[-1]):
        return False, False

    slope_d = float(ema50_d[-1] - ema50_d[-4])
    threshold = C.K_SLOPE * float(atr14_d[-1])

    if trade_dir == Direction.LONG:
        supports = slope_d > threshold
        opposes = slope_d < -threshold
    else:
        supports = slope_d < -threshold
        opposes = slope_d > threshold
    return supports, opposes


def compute_composite_regime(
    regime_4h: str, trend_dir_4h: Direction, trade_dir: Direction,
    daily_supports: bool, daily_opposes: bool,
) -> CompositeRegime:
    """Map to composite regime (Section 9.3)."""
    aligned_4h = (regime_4h == "TRENDING" and trend_dir_4h == trade_dir)
    counter_4h = (regime_4h == "TRENDING" and trend_dir_4h != Direction.FLAT and trend_dir_4h != trade_dir)

    if aligned_4h and (daily_supports or not daily_opposes):
        return CompositeRegime.ALIGNED
    if regime_4h == "TRANSITIONAL" or (daily_opposes and aligned_4h):
        return CompositeRegime.NEUTRAL
    if counter_4h and (daily_supports or not daily_opposes):
        return CompositeRegime.CAUTION
    if regime_4h == "RANGE":
        return CompositeRegime.RANGE
    if counter_4h and daily_opposes:
        return CompositeRegime.COUNTER
    return CompositeRegime.NEUTRAL


def regime_hard_block(regime_4h: str, trend_dir_4h: Direction, trade_dir: Direction, daily_opposes: bool) -> bool:
    """Section 9.4: hard block only if 4H Counter AND daily strongly opposes."""
    counter_4h = (regime_4h == "TRENDING" and trend_dir_4h != Direction.FLAT and trend_dir_4h != trade_dir)
    return counter_4h and daily_opposes


# ---------------------------------------------------------------------------
# 15m Slope Filter (Phase 1.1)
# ---------------------------------------------------------------------------

def slope_supports_breakout(
    closes_15m: np.ndarray,
    direction: Direction,
    fast: int = C.MACD_FAST,
    slow: int = C.MACD_SLOW,
    signal: int = C.MACD_SIGNAL,
    lookback: int = C.SLOPE_LOOKBACK,
) -> bool:
    """Check if 15m MACD histogram slope supports the breakout direction.

    Returns True if momentum is aligned with breakout (continuation),
    False if momentum opposes breakout (reversal — the higher-edge setup).
    """
    if len(closes_15m) < slow + signal + lookback:
        return False
    hist = macd_hist(closes_15m, fast, slow, signal)
    # 3-bar slope of histogram
    if np.isnan(hist[-1]) or np.isnan(hist[-1 - lookback]):
        return False
    slope = float(hist[-1] - hist[-1 - lookback])
    if direction == Direction.LONG:
        return slope > 0
    else:
        return slope < 0


# ---------------------------------------------------------------------------
# CHOP score (Section 10)
# ---------------------------------------------------------------------------

def compute_chop_score(atr_pctl_60d: float, vwap_cross_cnt: int) -> int:
    """Graduated chop score 0–4."""
    score = 0
    if atr_pctl_60d > C.CHOP_ATR_PCTL_1:
        score += 1
    if atr_pctl_60d > C.CHOP_ATR_PCTL_2:
        score += 1
    if vwap_cross_cnt >= C.CHOP_VWAP_CROSS_1:
        score += 1
    if vwap_cross_cnt >= C.CHOP_VWAP_CROSS_2:
        score += 1
    return score


def chop_mode(chop_score: int) -> ChopMode:
    if chop_score >= 2:
        return ChopMode.DEGRADED
    return ChopMode.NORMAL


# ---------------------------------------------------------------------------
# Breakout qualification (Section 11)
# ---------------------------------------------------------------------------

def breakout_structural(close_30m: float, box_high: float, box_low: float) -> Optional[Direction]:
    """Check structural breakout. Returns direction or None."""
    if close_30m > box_high:
        return Direction.LONG
    if close_30m < box_low:
        return Direction.SHORT
    return None


def displacement_pass(
    close_30m: float, vwap_box: float, atr14_30m: float,
    disp_hist: list[float], q_disp: float = C.Q_DISP,
    atr_expanding: bool = False,
    squeeze_good: bool = False,
    regime_aligned: bool = False,
) -> tuple[float, float, bool]:
    """Check displacement threshold. Returns (disp_metric, threshold, passed).

    Context-adaptive: lower threshold when squeeze is tight or regime is aligned.
    """
    disp = displacement_metric(close_30m, vwap_box, atr14_30m)
    # Context-adaptive threshold selection
    if squeeze_good:
        effective_q = C.Q_DISP_TIGHT_BOX
    elif regime_aligned:
        effective_q = C.Q_DISP_ALIGNED
    else:
        effective_q = q_disp
    effective_q = effective_q - 0.05 if atr_expanding else effective_q
    threshold = rolling_quantile_past_only(disp_hist, effective_q)
    return disp, threshold, disp >= threshold


def breakout_quality_reject(
    bar_high: float, bar_low: float, bar_open: float, bar_close: float,
    atr14_30m: float, rvol: float, direction: Direction,
) -> tuple[bool, bool]:
    """Section 11.3: returns (rejected, body_decisive)."""
    rng = bar_high - bar_low
    body = abs(bar_close - bar_open)
    body_ratio = body / rng if rng > 0 else 1.0

    if direction == Direction.LONG:
        adverse_wick = bar_high - max(bar_open, bar_close)
    else:
        adverse_wick = min(bar_open, bar_close) - bar_low
    wick_ratio = adverse_wick / rng if rng > 0 else 0.0

    rejected = (
        rng > C.BREAKOUT_REJECT_RANGE_MULT * atr14_30m and
        (body_ratio < C.BREAKOUT_REJECT_BODY_RATIO or wick_ratio > C.BREAKOUT_REJECT_WICK_RATIO) and
        rvol > C.BREAKOUT_REJECT_RVOL
    )
    body_decisive = body_ratio >= 0.50
    return rejected, body_decisive


# ---------------------------------------------------------------------------
# Evidence scorecard (Section 13)
# ---------------------------------------------------------------------------

def compute_score(
    rvol: float,
    two_outside: bool,
    atr_rising: bool,
    squeeze_good: bool,
    squeeze_loose: bool,
    regime_4h: str,
    trend_dir_4h: Direction,
    trade_dir: Direction,
    daily_supports: bool,
    body_decisive: bool,
) -> float:
    """Evidence score (Section 13). Returns total score."""
    score = 1.0   # displacement baseline
    if rvol > C.RVOL_SCORE_THRESH:
        score += 1.0
    if two_outside and atr_rising:
        score += 1.0
    if squeeze_good:
        score += 1.0
    if squeeze_loose:
        score -= 1.0
    # 4H regime alignment bonus removed (Phase 3.2: misleading after regime flattening)
    if daily_supports:
        score += 1.0
    if body_decisive:
        score += 0.5
    return score


def score_threshold(mode: ChopMode) -> float:
    if mode == ChopMode.DEGRADED:
        return C.SCORE_DEGRADED
    return C.SCORE_NORMAL


def contextual_score_filter_pass(
    *,
    score: float,
    box_width: float,
    rvol: float,
) -> tuple[bool, str]:
    """Return whether score has enough box/RVOL context to be tradable."""
    if (
        C.WEAK_SCORE_BAND_FILTER_ENABLED
        and C.WEAK_SCORE_BAND_LOW <= score < C.WEAK_SCORE_BAND_HIGH
        and (
            box_width > C.WEAK_SCORE_BAND_MAX_BOX_WIDTH
            or rvol < C.WEAK_SCORE_BAND_MIN_RVOL
        )
    ):
        return False, "weak_score_context"
    if (
        C.WIDE_BOX_SCORE_FILTER_ENABLED
        and box_width >= C.WIDE_BOX_MIN_WIDTH
        and (score < C.WIDE_BOX_MIN_SCORE or rvol < C.WIDE_BOX_MIN_RVOL)
    ):
        return False, "wide_box_context"
    return True, ""


def b_entry_regime_allowed(composite_regime: CompositeRegime | str) -> bool:
    """Shared B-entry regime permission used by live and backtest."""
    value = getattr(composite_regime, "value", composite_regime)
    return (
        (value == CompositeRegime.ALIGNED.value and C.B_ALLOW_ALIGNED)
        or (value == CompositeRegime.RANGE.value and C.B_ALLOW_RANGE)
        or (value == CompositeRegime.NEUTRAL.value and C.B_ALLOW_NEUTRAL)
        or (value == CompositeRegime.CAUTION.value and C.B_ALLOW_CAUTION)
    )


def a_entry_context_allowed(*, score: float, box_width: float) -> tuple[bool, str]:
    """Shared A-entry context gate used by live and backtest."""
    max_box = getattr(C, "A_MAX_BOX_WIDTH", 0.0)
    if max_box > 0 and box_width > max_box:
        return False, "a_box_width"

    min_score = getattr(C, "A_MIN_SCORE", 0.0)
    if min_score > 0 and score < min_score:
        return False, "a_min_score"

    if (
        getattr(C, "A_BLOCK_WEAK_SCORE_BAND", False)
        and getattr(C, "A_WEAK_SCORE_BAND_LOW", 2.5) <= score < getattr(C, "A_WEAK_SCORE_BAND_HIGH", 3.0)
    ):
        return False, "a_weak_score_band"

    return True, ""



# ---------------------------------------------------------------------------
# Dirty-wick reclaim (Section G, fix #6)
# ---------------------------------------------------------------------------

def dirty_wick_reclaim_check(
    close_30m: float,
    dirty_wick_extreme: float,
    dirty_direction: str,
) -> bool:
    """Check if price reclaims past the dirty wick extreme.

    For a failed long breakout, the wick extreme is the high reached during
    the dirty trigger.  Reclaim requires close above that extreme.
    """
    if dirty_direction == "long":
        return close_30m > dirty_wick_extreme
    elif dirty_direction == "short":
        return close_30m < dirty_wick_extreme
    return False


# ---------------------------------------------------------------------------
# Breakout state management (Section 12)
# ---------------------------------------------------------------------------

def activate_breakout(
    breakout: BreakoutEngineState,
    direction: Direction,
    atr_pctl_30m: float,
    bar_ts,
    box_high: float, box_low: float, box_width: float,
    box_bars_active: int,
    breakout_bar_high: float = 0.0,
    breakout_bar_low: float = 0.0,
) -> None:
    """Set breakout active with computed expiry."""
    breakout.active = True
    breakout.direction = direction
    breakout.breakout_bar_ts = bar_ts
    breakout.bars_since_breakout = 0
    breakout.continuation_mode = False
    breakout.consec_inside_count = 0
    breakout.mm_reached = False
    breakout.breakout_bar_high = breakout_bar_high
    breakout.breakout_bar_low = breakout_bar_low

    # Expiry (Section 12)
    raw = round(8 * (atr_pctl_30m / 50))
    breakout.expiry_bars = max(6, min(16, int(raw)))
    breakout.hard_expiry_bars = breakout.expiry_bars + C.HARD_EXPIRY_EXTENSION

    # Measured move (Section 17.1)
    duration_factor = max(C.MM_DURATION_MIN, min(C.MM_DURATION_MAX, math.sqrt(box_bars_active / 20)))
    if direction == Direction.LONG:
        breakout.mm_level = box_high + C.MM_WIDTH_MULT * box_width * duration_factor
    else:
        breakout.mm_level = box_low - C.MM_WIDTH_MULT * box_width * duration_factor


def update_breakout_state(
    breakout: BreakoutEngineState,
    close_30m: float,
    atr14_30m: float,
    box_high: float, box_low: float,
    regime_hard_blocked: bool = False,
) -> None:
    """Update expiry mult, continuation check, invalidation check."""
    if not breakout.active:
        return

    # Expiry clock pauses while continuation mode is active (Section 12.1)
    if not breakout.continuation_mode:
        breakout.bars_since_breakout += 1

    # Continuation check (Section 12.1)
    if not breakout.continuation_mode:
        if breakout.mm_reached:
            breakout.continuation_mode = True
        elif atr14_30m > 0:
            if breakout.direction == Direction.LONG:
                r_proxy = (close_30m - box_high) / atr14_30m
            else:
                r_proxy = (box_low - close_30m) / atr14_30m
            if r_proxy >= C.CONTINUATION_R_PROXY:
                breakout.continuation_mode = True

    # Measured move reached
    if not breakout.mm_reached:
        if breakout.direction == Direction.LONG and close_30m >= breakout.mm_level:
            breakout.mm_reached = True
        elif breakout.direction == Direction.SHORT and close_30m <= breakout.mm_level:
            breakout.mm_reached = True

    # Invalidation (Section 12.2)
    # 1) Regime hard block
    if regime_hard_blocked:
        breakout.active = False
        return
    # 2) Consecutive inside bars
    inside = (box_low <= close_30m <= box_high)
    if inside:
        breakout.consec_inside_count += 1
    else:
        breakout.consec_inside_count = 0
    if breakout.consec_inside_count >= C.INVALIDATION_CONSEC_INSIDE:
        breakout.active = False
    # 3) Hard expiry without continuation
    elif breakout.bars_since_breakout > breakout.hard_expiry_bars and not breakout.continuation_mode:
        breakout.active = False


# ---------------------------------------------------------------------------
# Entry triggers (5m) — Section 16
# ---------------------------------------------------------------------------

def entry_a_trigger(
    close_5m: float, low_5m: float, high_5m: float,
    vwap_session: float, breakout_high_30m: float, breakout_low_30m: float,
    box_high: float, atr14_30m: float,
    direction: Direction,
) -> tuple[float, float]:
    """Compute A1 (limit) and A2 (stop) prices. Returns (a1_price, a2_price)."""
    tick = C.NQ_SPECS[C.DEFAULT_SYMBOL]["tick"]
    if direction == Direction.LONG:
        a1 = vwap_session + C.A1_OFFSET_TICKS * tick
        a2 = breakout_high_30m + C.A2_BUFFER_TICKS * tick
    else:
        a1 = vwap_session - C.A1_OFFSET_TICKS * tick
        a2 = breakout_low_30m - C.A2_BUFFER_TICKS * tick
    return a1, a2


def entry_a_cancel_check(
    close_5m: float, box_high: float, box_low: float,
    atr14_30m: float, direction: Direction,
) -> bool:
    """Return True if A orders should be cancelled (Section 16.2)."""
    if direction == Direction.LONG:
        return close_5m < box_high - C.A_CANCEL_DEPTH_ATR * atr14_30m
    return close_5m > box_low + C.A_CANCEL_DEPTH_ATR * atr14_30m


def entry_b_trigger(
    low_5m: float, high_5m: float, close_5m: float,
    vwap_session: float, atr14_30m: float, direction: Direction,
) -> bool:
    """Check B sweep+reclaim trigger on 5m bar (Section 16.3)."""
    sweep_depth = C.B_SWEEP_DEPTH_ATR * atr14_30m
    if direction == Direction.LONG:
        return low_5m < vwap_session - sweep_depth and close_5m > vwap_session
    return high_5m > vwap_session + sweep_depth and close_5m < vwap_session


def entry_c_hold_check(
    closes_5m: np.ndarray, lows_5m: np.ndarray, highs_5m: np.ndarray,
    vwap_session: float, direction: Direction,
    atr14_30m: float = 0.0,
) -> tuple[bool, float]:
    """Check C hold signal (C_HOLD_BARS consecutive bars). Returns (triggered, hold_ref).

    Allows minor wick penetration of VWAP up to 0.05 * ATR tolerance.
    """
    if len(closes_5m) < C.C_HOLD_BARS:
        return False, 0.0

    tolerance = C.C_ENTRY_OFFSET_ATR * atr14_30m if atr14_30m > 0 else 0.0

    if C.C_HOLD_BARS == 1:
        bar = len(closes_5m) - 1
        if direction == Direction.LONG:
            ok = closes_5m[bar] > vwap_session and lows_5m[bar] >= vwap_session - tolerance
            hold_ref = float(lows_5m[bar])
        else:
            ok = closes_5m[bar] < vwap_session and highs_5m[bar] <= vwap_session + tolerance
            hold_ref = float(highs_5m[bar])
        return ok, hold_ref

    bar1 = len(closes_5m) - 2
    bar2 = len(closes_5m) - 1

    if direction == Direction.LONG:
        ok = (
            closes_5m[bar1] > vwap_session and lows_5m[bar1] >= vwap_session - tolerance and
            closes_5m[bar2] > vwap_session and lows_5m[bar2] >= vwap_session - tolerance
        )
        hold_ref = min(lows_5m[bar1], lows_5m[bar2])
    else:
        ok = (
            closes_5m[bar1] < vwap_session and highs_5m[bar1] <= vwap_session + tolerance and
            closes_5m[bar2] < vwap_session and highs_5m[bar2] <= vwap_session + tolerance
        )
        hold_ref = max(highs_5m[bar1], highs_5m[bar2])
    return ok, float(hold_ref)


def entry_c_continuation_pause(
    highs_5m: np.ndarray, lows_5m: np.ndarray, atr14_5m: float,
) -> bool:
    """Check tight pause constraint for C_continuation (Section 16.4)."""
    if len(highs_5m) < 2 or atr14_5m <= 0:
        return False
    rng1 = highs_5m[-2] - lows_5m[-2]
    rng2 = highs_5m[-1] - lows_5m[-1]
    return max(rng1, rng2) <= C.C_CONT_PAUSE_ATR_MULT * atr14_5m
