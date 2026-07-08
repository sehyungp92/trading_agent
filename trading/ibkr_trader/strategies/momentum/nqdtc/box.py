"""NQ Dominant Trend Capture v2.0 — adaptive 30m box state machine."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from . import config as C
from .indicators import atr, squeeze_metric
from .models import BoxEngineState, BoxState, BreakoutEngineState, Direction, VWAPAccumulator

# Volume arrays needed for VWAP backfill
_VWAP_BACKFILL_ENABLED = True

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive L (Section 8.1)
# ---------------------------------------------------------------------------

def compute_adaptive_L(
    atr14_30m: float, atr50_30m: float,
    last_bucket: str, streak: int,
) -> tuple[Optional[int], str, int]:
    """Return (new_L_or_None, bucket, streak)."""
    if atr50_30m <= 0:
        return None, last_bucket, streak
    ratio = atr14_30m / atr50_30m
    if ratio < 0.7:
        bucket = "LOW"
    elif ratio > 1.2:
        bucket = "HIGH"
    else:
        bucket = "MID"

    if bucket == last_bucket:
        streak += 1
    else:
        last_bucket, streak = bucket, 1

    if streak >= C.BOX_BUCKET_HYSTERESIS:
        L = C.ADAPTIVE_L[bucket]
        return L, last_bucket, streak
    return None, last_bucket, streak


# ---------------------------------------------------------------------------
# Candidate evaluation (Section 8.3–8.5)
# ---------------------------------------------------------------------------

def evaluate_box_candidate(
    highs_30m: np.ndarray, lows_30m: np.ndarray, closes_30m: np.ndarray,
    L: int, atr14_30m: float,
) -> tuple[bool, float, float, float, float]:
    """Check if box candidate qualifies. Returns (ok, hi, lo, width, mid)."""
    n = len(closes_30m)
    if n < L:
        return False, 0, 0, 0, 0

    hi = float(np.max(highs_30m[-L:]))
    lo = float(np.min(lows_30m[-L:]))
    width = hi - lo
    mid = (hi + lo) / 2.0

    # Containment (Section 8.3)
    recent = closes_30m[-L:]
    contained = np.sum((recent >= lo) & (recent <= hi))
    containment = contained / L
    if containment < C.CONTAINMENT_MIN:
        return False, hi, lo, width, mid

    # Violations
    violations = np.sum((recent > hi) | (recent < lo))
    if violations > C.VIOL_MAX:
        return False, hi, lo, width, mid

    # Box height min (Section 8.4)
    min_atr = C.BOX_HEIGHT_MIN_ATR_SHORT if L == C.ADAPTIVE_L["LOW"] else C.BOX_HEIGHT_MIN_ATR_LONG
    if atr14_30m > 0 and width < min_atr * atr14_30m:
        return False, hi, lo, width, mid

    return True, hi, lo, width, mid


# ---------------------------------------------------------------------------
# Box state machine update (called on each 30m close)
# ---------------------------------------------------------------------------

def update_box_state(
    box: BoxEngineState,
    breakout: BreakoutEngineState,
    highs_30m: np.ndarray,
    lows_30m: np.ndarray,
    closes_30m: np.ndarray,
    atr14_30m: float,
    atr50_30m: float,
    bar_ts: datetime,
    vwap_box: VWAPAccumulator,
    squeeze_hist: list[float],
    volumes_30m: Optional[np.ndarray] = None,
) -> None:
    """Full box lifecycle update."""
    # 1) Update adaptive L
    new_L, box.last_bucket, box.bucket_streak = compute_adaptive_L(
        atr14_30m, atr50_30m, box.last_bucket, box.bucket_streak,
    )
    if new_L is not None:
        box.L = new_L

    # 2) Squeeze metric: computed and appended by engine after qualification (past-only)

    # 3) State transitions
    if box.state == BoxState.INACTIVE:
        ok, hi, lo, width, mid = evaluate_box_candidate(
            highs_30m, lows_30m, closes_30m, box.L, atr14_30m,
        )
        if ok:
            box.state = BoxState.ACTIVE
            box.box_high = hi
            box.box_low = lo
            box.box_width = width
            box.box_mid = mid
            box.box_anchor_ts = bar_ts
            box.box_bars_active = 0
            box.L_used = box.L
            # Reset box-anchored VWAP and backfill with bars from the box period (fix #21)
            vwap_box.reset(bar_ts)
            n = len(closes_30m)
            if _VWAP_BACKFILL_ENABLED and volumes_30m is not None and len(volumes_30m) == n:
                backfill_start = max(0, n - box.L)
                for i in range(backfill_start, n):
                    vwap_box.update(
                        float(highs_30m[i]), float(lows_30m[i]),
                        float(closes_30m[i]), float(volumes_30m[i]),
                    )
            logger.info("Box ACTIVE: H=%.2f L=%.2f W=%.2f L=%d", hi, lo, width, box.L)

    elif box.state == BoxState.ACTIVE:
        box.box_bars_active += 1
        # Check DIRTY trigger
        if _dirty_triggered(box, breakout, closes_30m, highs_30m, lows_30m):
            box.state = BoxState.DIRTY
            box.dirty_start_idx = len(closes_30m) - 1
            box.dirty_high = box.box_high
            box.dirty_low = box.box_low
            box.dirty_direction = "long" if breakout.direction == Direction.LONG else "short"
            # Record wick extreme for dirty-wick reclaim (Section G, fix #6)
            t = len(closes_30m) - 1
            if breakout.direction == Direction.LONG:
                box.dirty_wick_extreme = float(highs_30m[t])
            else:
                box.dirty_wick_extreme = float(lows_30m[t])
            logger.info("Box → DIRTY (direction=%s, wick_extreme=%.2f)", box.dirty_direction, box.dirty_wick_extreme)

    elif box.state == BoxState.DIRTY:
        box.box_bars_active += 1
        # Timeout: if DIRTY persists > DIRTY_TIMEOUT_MULT * L bars, retire box
        dirty_age = (len(closes_30m) - 1 - box.dirty_start_idx) if box.dirty_start_idx >= 0 else 0
        if dirty_age > C.DIRTY_TIMEOUT_MULT * box.L_used:
            retire_box(box)
            return
        # Check DIRTY reset
        if _dirty_reset_ok(box, highs_30m, lows_30m, closes_30m, atr14_30m, squeeze_hist):
            box.state = BoxState.ACTIVE
            # Re-freeze bounds
            ok, hi, lo, width, mid = evaluate_box_candidate(
                highs_30m, lows_30m, closes_30m, box.L, atr14_30m,
            )
            if ok:
                box.box_high = hi
                box.box_low = lo
                box.box_width = width
                box.box_mid = mid
                box.box_anchor_ts = bar_ts
                box.box_bars_active = 0
                vwap_box.reset(bar_ts)
            logger.info("DIRTY → ACTIVE (reset)")
            box.dirty_start_idx = -1


def retire_box(box: BoxEngineState) -> None:
    """Retire current box and reset to INACTIVE."""
    box.state = BoxState.INACTIVE
    box.box_high = 0.0
    box.box_low = 0.0
    box.box_width = 0.0
    box.box_mid = 0.0
    box.box_anchor_ts = None
    box.box_bars_active = 0
    box.dirty_start_idx = -1
    logger.info("Box retired → INACTIVE")


# ---------------------------------------------------------------------------
# DIRTY detection (Section G)
# ---------------------------------------------------------------------------

def _dirty_triggered(
    box: BoxEngineState,
    breakout: BreakoutEngineState,
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
) -> bool:
    """Breakout attempt then deep return within M_BREAK bars."""
    if not breakout.active:
        return False
    if breakout.bars_since_breakout > C.M_BREAK:
        return False

    t = len(closes) - 1
    if breakout.direction == Direction.LONG:
        depth = box.box_high - C.DIRTY_DEPTH_FRAC * box.box_width
        return closes[t] <= box.box_high and lows[t] <= depth
    else:
        depth = box.box_low + C.DIRTY_DEPTH_FRAC * box.box_width
        return closes[t] >= box.box_low and highs[t] >= depth


def _dirty_reset_ok(
    box: BoxEngineState,
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    atr14_30m: float, squeeze_hist: list[float],
) -> bool:
    """Check if DIRTY state can be reset (Section G)."""
    if not squeeze_hist:
        return False
    # squeeze_good = current squeeze below 20th percentile of history
    sq_arr = np.array(squeeze_hist)
    sq_threshold = float(np.quantile(sq_arr, C.SQUEEZE_GOOD_QUANTILE))
    n = len(closes)
    if n < box.L:
        return False
    hi = float(np.max(highs[-box.L:]))
    lo = float(np.min(lows[-box.L:]))
    current_sq = squeeze_metric(hi - lo, atr14_30m) if atr14_30m > 0 else 999
    squeeze_good = current_sq <= sq_threshold

    if not squeeze_good:
        return False

    # Shift or duration condition
    t_idx = n - 1
    dirty_duration = t_idx - box.dirty_start_idx if box.dirty_start_idx >= 0 else 0
    box_shifted = (
        abs(hi - box.dirty_high) >= C.DIRTY_RESET_SHIFT_ATR * atr14_30m and
        abs(lo - box.dirty_low) >= C.DIRTY_RESET_SHIFT_ATR * atr14_30m
    )
    return box_shifted or dirty_duration >= C.DIRTY_RESET_DURATION_FRAC * box.L_used
