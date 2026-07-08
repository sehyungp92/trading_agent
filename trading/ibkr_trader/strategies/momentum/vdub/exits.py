"""Vdubus NQ v4.0 — position management: partials, trailing, exits, decision gate."""
from __future__ import annotations

import numpy as np

from . import config as C
from .models import Direction, PositionState, PositionStage


# ---------------------------------------------------------------------------
# Early Kill — fast-dying trades (Section 16.0)
# ---------------------------------------------------------------------------

def check_early_kill(pos: PositionState, current_price: float) -> bool:
    """Two-stage early invalidation for fast-dying trades.

    Stage 1 (warning): set warning flag when unrealized R < threshold
                        AND peak MFE hasn't reached floor.
    Stage 2 (confirm): exit only if warning was set on previous bar
                        AND condition still holds (confirmed failure).
    """
    if pos.partial_done:
        return False
    if pos.bars_since_entry > C.EARLY_KILL_BARS:
        return False
    if pos.r_points <= 0:
        return False
    if pos.direction == Direction.LONG:
        unreal_r = (current_price - pos.entry_price) / pos.r_points
    else:
        unreal_r = (pos.entry_price - current_price) / pos.r_points

    failure = unreal_r < C.EARLY_KILL_R and pos.peak_mfe_r < C.EARLY_KILL_MFE_FLOOR

    if not failure:
        # Condition cleared — reset warning
        pos.early_warning_bar = -1
        return False

    # Stage 2: if warning was set on a previous bar, confirm exit
    if pos.early_warning_bar >= 0 and pos.bars_since_entry > pos.early_warning_bar:
        return True

    # Stage 1: set warning on this bar
    if pos.early_warning_bar < 0:
        pos.early_warning_bar = pos.bars_since_entry

    return False


# ---------------------------------------------------------------------------
# +1R Free-Ride (Section 16.1)
# ---------------------------------------------------------------------------

def check_partial(pos: PositionState, current_price: float) -> int:
    """Returns qty to close for +1R partial, or 0."""
    if pos.partial_done or pos.r_points <= 0:
        return 0
    if pos.direction == Direction.LONG:
        unreal_pts = current_price - pos.entry_price
    else:
        unreal_pts = pos.entry_price - current_price
    unreal_r = unreal_pts / pos.r_points
    if unreal_r < 1.0:
        return 0
    if pos.qty_entry >= 2:
        return max(1, int(pos.qty_entry * C.PARTIAL_PCT))
    return 0  # 1-lot: just move stop to BE, no partial


def breakeven_stop(pos: PositionState) -> float:
    """Return breakeven stop price."""
    return pos.entry_price


# ---------------------------------------------------------------------------
# ACTIVE_FREE stale exit + profit lock (Section 16.1b)
# ---------------------------------------------------------------------------

def check_free_ride_stale(pos: PositionState, current_price: float) -> bool:
    """Exit if ACTIVE_FREE, >= FREE_STALE_BARS_15M bars since partial,
    and unrealized R < FREE_STALE_R_THRESHOLD."""
    if pos.stage != PositionStage.ACTIVE_FREE:
        return False
    if pos.bars_since_partial < C.FREE_STALE_BARS_15M:
        return False
    if pos.r_points <= 0:
        return False
    if pos.direction == Direction.LONG:
        unreal_r = (current_price - pos.entry_price) / pos.r_points
    else:
        unreal_r = (pos.entry_price - current_price) / pos.r_points
    return unreal_r < C.FREE_STALE_R_THRESHOLD


def compute_free_profit_lock(pos: PositionState, current_price: float) -> float:
    """Tighten stop to lock +FREE_PROFIT_LOCK_R once peak_r_since_free >= 0.50R.
    Returns new stop candidate (only tightens, never loosens)."""
    if pos.stage != PositionStage.ACTIVE_FREE:
        return pos.stop_price
    if pos.peak_r_since_free < 0.50:
        return pos.stop_price
    if pos.r_points <= 0:
        return pos.stop_price

    if pos.direction == Direction.LONG:
        lock_price = pos.entry_price + C.FREE_PROFIT_LOCK_R * pos.r_points
        return max(pos.stop_price, lock_price)
    else:
        lock_price = pos.entry_price - C.FREE_PROFIT_LOCK_R * pos.r_points
        return min(pos.stop_price, lock_price)


# ---------------------------------------------------------------------------
# Max position duration (hard cap)
# ---------------------------------------------------------------------------

def check_max_duration(pos: PositionState) -> bool:
    """Return True if position exceeds max duration hard cap."""
    if pos.stage == PositionStage.ACTIVE_FREE:
        return pos.bars_since_entry >= C.MAX_POSITION_BARS_FREE
    return pos.bars_since_entry >= C.MAX_POSITION_BARS_15M


# ---------------------------------------------------------------------------
# Intraday trailing (Section 16.2)
# ---------------------------------------------------------------------------

def compute_intraday_trail(
    pos: PositionState, highs_15m: np.ndarray, lows_15m: np.ndarray,
    atr15: float, current_price: float, tighten_factor: float = 1.0,
    stage: PositionStage | None = None,
) -> float:
    """Compute trailing stop (tighten-only). Returns new stop candidate."""
    if pos.r_points <= 0 or pos.qty_open <= 0:
        return pos.stop_price

    if pos.direction == Direction.LONG:
        unreal_pts = current_price - pos.entry_price
    else:
        unreal_pts = pos.entry_price - current_price
    r_now = unreal_pts / pos.r_points

    # Use tighter base trail for ACTIVE_FREE positions
    effective_stage = stage if stage is not None else pos.stage
    base = C.TRAIL_MULT_POST_PARTIAL if effective_stage == PositionStage.ACTIVE_FREE else C.TRAIL_MULT_BASE

    # Staged trail: wide early, tighten gradually after +1.5R
    if r_now < 1.5:
        trail_mult = base
    elif r_now < 2.5:
        trail_mult = max(C.TRAIL_MULT_MIN,
                         base - ((r_now - 1.5) / C.TRAIL_MULT_R_DIV))
    else:
        trail_mult = C.TRAIL_MULT_MIN
    trail_mult *= tighten_factor
    lb = min(C.TRAIL_LOOKBACK_15M, len(highs_15m))

    if pos.direction == Direction.LONG:
        hh = float(np.nanmax(highs_15m[-lb:]))
        candidate = hh - trail_mult * atr15
        return max(pos.stop_price, candidate)
    else:
        ll = float(np.nanmin(lows_15m[-lb:]))
        candidate = ll + trail_mult * atr15
        return min(pos.stop_price, candidate)


# ---------------------------------------------------------------------------
# VWAP failure exit (Section 16.3) — pre-+1R only
# ---------------------------------------------------------------------------

def check_vwap_failure(
    pos: PositionState, closes_15m: np.ndarray, vwap_used: float,
) -> bool:
    """Return True if VWAP failure exit should trigger."""
    if pos.partial_done:
        return False
    n = len(closes_15m)
    if n < C.VWAP_FAIL_CONSEC:
        return False
    recent = closes_15m[-C.VWAP_FAIL_CONSEC:]
    if pos.direction == Direction.LONG:
        return bool(np.all(recent < vwap_used))
    return bool(np.all(recent > vwap_used))


# ---------------------------------------------------------------------------
# MFE Ratchet Floor (v4.2) — progressive profit lock
# ---------------------------------------------------------------------------

def compute_mfe_ratchet_floor(pos: PositionState) -> float:
    """Return the ratchet floor stop price based on peak MFE tiers.

    Applies MFE_RATCHET_TIERS: once peak_mfe_r crosses a threshold,
    lock a minimum stop at the corresponding floor R.
    The floor is a minimum — the trailing stop can be higher.
    Returns 0.0 if no tier has been hit (no floor active).
    """
    if pos.r_points <= 0:
        return 0.0
    floor_r = 0.0
    for mfe_thresh, lock_r in C.MFE_RATCHET_TIERS:
        if pos.peak_mfe_r >= mfe_thresh:
            floor_r = lock_r
    if floor_r <= 0.0:
        return 0.0
    if pos.direction == Direction.LONG:
        return pos.entry_price + floor_r * pos.r_points
    return pos.entry_price - floor_r * pos.r_points


def compute_mfe_rescue_stop(pos: PositionState, current_price: float) -> float:
    """Protect proven-but-stalling trades before they decay into slow deaths."""
    if pos.partial_done or pos.r_points <= 0:
        return pos.stop_price
    if pos.bars_since_entry < C.MFE_RESCUE_AFTER_BARS:
        return pos.stop_price
    if pos.peak_mfe_r < C.MFE_RESCUE_MIN_R:
        return pos.stop_price

    if pos.direction == Direction.LONG:
        current_r = (current_price - pos.entry_price) / pos.r_points
        if current_r > C.MFE_RESCUE_TRIGGER_R:
            return pos.stop_price
        rescue = pos.entry_price + C.MFE_RESCUE_LOCK_R * pos.r_points
        return max(pos.stop_price, rescue)

    current_r = (pos.entry_price - current_price) / pos.r_points
    if current_r > C.MFE_RESCUE_TRIGGER_R:
        return pos.stop_price
    rescue = pos.entry_price - C.MFE_RESCUE_LOCK_R * pos.r_points
    return min(pos.stop_price, rescue)


def compute_close_mfe_ratchet(pos: PositionState) -> float:
    """CLOSE-specific MFE ratchet for skip-partial entries.

    More aggressive than the global ratchet — starts at 1.0R MFE (not 0.75R)
    and locks +0.40R (vs +0.15R), replacing the original 0.33R partial floor
    with a tighter stop-based equivalent on the full position.
    Returns 0.0 if no tier active.
    """
    if pos.r_points <= 0:
        return 0.0
    floor_r = 0.0
    for mfe_thresh, lock_r in C.CLOSE_MFE_RATCHET_TIERS:
        if pos.peak_mfe_r >= mfe_thresh:
            floor_r = lock_r
    if floor_r <= 0.0:
        return 0.0
    if pos.direction == Direction.LONG:
        return pos.entry_price + floor_r * pos.r_points
    return pos.entry_price - floor_r * pos.r_points


# ---------------------------------------------------------------------------
# Stale exit (Section 16.4) — pre-+1R only
# ---------------------------------------------------------------------------

def check_stale_exit(pos: PositionState, current_price: float, sub_window: str = "CORE") -> bool:
    if pos.partial_done:
        return False
    stale_bars = C.STALE_BARS_BY_WINDOW.get(sub_window, C.STALE_BARS_15M)
    if pos.bars_since_entry < stale_bars:
        return False
    if pos.r_points <= 0:
        return False
    if pos.direction == Direction.LONG:
        unreal_r = (current_price - pos.entry_price) / pos.r_points
    else:
        unreal_r = (pos.entry_price - current_price) / pos.r_points
    return unreal_r < C.STALE_R


# ---------------------------------------------------------------------------
# 15:50 Decision Gate — "Earn the Hold" (Section 17)
# ---------------------------------------------------------------------------

def decision_gate(
    pos: PositionState, is_friday: bool,
    current_price: float, slope_ok_dir: bool, trend_1h_aligned: bool,
) -> tuple[str, float]:
    """Returns ('HOLD'|'FLATTEN', new_stop).
    new_stop only meaningful when HOLD."""
    if pos.r_points <= 0:
        return "FLATTEN", pos.stop_price

    if pos.direction == Direction.LONG:
        unreal_r = (current_price - pos.entry_price) / pos.r_points
    else:
        unreal_r = (pos.entry_price - current_price) / pos.r_points

    if is_friday:
        if unreal_r >= C.HOLD_FRIDAY_R:
            # Tighten to lock at least +0.5R
            if pos.direction == Direction.LONG:
                lock = pos.entry_price + C.WEEKEND_LOCK_R * pos.r_points
            else:
                lock = pos.entry_price - C.WEEKEND_LOCK_R * pos.r_points
            new_stop = _tighten(pos, lock)
            return "HOLD", new_stop
        return "FLATTEN", pos.stop_price

    # Weekday
    if unreal_r >= C.HOLD_WEEKDAY_R:
        return "HOLD", pos.stop_price
    if (unreal_r >= C.HOLD_WEEKDAY_BORDER_R and
            slope_ok_dir and trend_1h_aligned):
        return "HOLD", pos.stop_price
    return "FLATTEN", pos.stop_price


def _tighten(pos: PositionState, candidate: float) -> float:
    """Tighten stop only (never loosen)."""
    if pos.direction == Direction.LONG:
        return max(pos.stop_price, candidate)
    return min(pos.stop_price, candidate)


# ---------------------------------------------------------------------------
# Overnight trail (Section 18.1)
# ---------------------------------------------------------------------------

def compute_overnight_trail(
    pos: PositionState, highs_1h: np.ndarray, lows_1h: np.ndarray,
    atr1h: float,
) -> float:
    """Compute overnight trailing stop."""
    lb = min(C.OVERNIGHT_1H_LOOKBACK, len(highs_1h))
    if lb == 0:
        return pos.stop_price

    if pos.direction == Direction.LONG:
        ll = float(np.nanmin(lows_1h[-lb:]))
        candidate = ll - C.OVERNIGHT_ATR_MULT * atr1h
        if pos.partial_done:
            # BE earned: allow widening but never below entry
            candidate = max(candidate, pos.entry_price)
            return candidate
        return max(pos.stop_price, candidate)  # pre-BE: tighten-only
    else:
        hh = float(np.nanmax(highs_1h[-lb:]))
        candidate = hh + C.OVERNIGHT_ATR_MULT * atr1h
        if pos.partial_done:
            # BE earned: allow widening but never above entry
            candidate = min(candidate, pos.entry_price)
            return candidate
        return min(pos.stop_price, candidate)  # pre-BE: tighten-only


# ---------------------------------------------------------------------------
# VWAP-A failure exit (Section 18.3) — multi-session, profitable only
# ---------------------------------------------------------------------------

def check_vwap_a_failure(
    pos: PositionState, close_1h: float, vwap_a_val: float,
    current_price: float, atr1h: float = 0.0,
) -> bool:
    """Return True if VWAP-A failure exit should trigger.

    Multi-session (session_count >= 2): original behavior.
    First-session (session_count == 1): allowed for ACTIVE_FREE positions
    when 1H close crosses VWAP-A by >= VWAP_A_FAIL_FIRST_SESSION_MARGIN_ATR.
    """
    if pos.session_count < C.VWAP_A_FAIL_MIN_SESSIONS:
        return False
    if pos.r_points <= 0:
        return False

    # First-session: require ACTIVE_FREE and margin
    if pos.session_count == 1:
        if pos.stage != PositionStage.ACTIVE_FREE:
            return False
        margin = C.VWAP_A_FAIL_FIRST_SESSION_MARGIN_ATR * atr1h if atr1h > 0 else 0
        if pos.direction == Direction.LONG:
            r_now = (current_price - pos.entry_price) / pos.r_points
            return r_now > 0 and close_1h < vwap_a_val - margin
        else:
            r_now = (pos.entry_price - current_price) / pos.r_points
            return r_now > 0 and close_1h > vwap_a_val + margin

    # Multi-session: original behavior
    if pos.direction == Direction.LONG:
        r_now = (current_price - pos.entry_price) / pos.r_points
        return r_now > 0 and close_1h < vwap_a_val
    else:
        r_now = (pos.entry_price - current_price) / pos.r_points
        return r_now > 0 and close_1h > vwap_a_val


# ---------------------------------------------------------------------------
# Shock mid-position (Section 4.3)
# ---------------------------------------------------------------------------

def shock_stop_tighten(pos: PositionState) -> float:
    """If Shock, tighten stop to at least breakeven."""
    if pos.direction == Direction.LONG:
        return max(pos.stop_price, pos.entry_price)
    return min(pos.stop_price, pos.entry_price)


# ---------------------------------------------------------------------------
# Late Trail (v4.4) -- independent late-activation trailing stop
# ---------------------------------------------------------------------------

def compute_late_trail_stop(
    pos: PositionState, highs_15m: np.ndarray, lows_15m: np.ndarray,
    atr15: float, current_price: float, sub_window: str = "CORE",
) -> float:
    """Late-activation trailing stop (tighten-only, wide multiplier).

    Independent of plus_1r_partial. Activated once peak_mfe_r crosses
    LATE_TRAIL_ACTIVATE_R. Uses staged tightening: wide until TIGHTEN_R,
    then progressively tighter toward MULT_MIN.
    """
    if pos.r_points <= 0 or pos.qty_open <= 0:
        return pos.stop_price

    if pos.direction == Direction.LONG:
        unreal_pts = current_price - pos.entry_price
    else:
        unreal_pts = pos.entry_price - current_price
    r_now = unreal_pts / pos.r_points

    # Staged: wide until TIGHTEN_R, then progressively tighter
    if r_now < C.LATE_TRAIL_TIGHTEN_R:
        trail_mult = C.LATE_TRAIL_MULT
    else:
        trail_mult = max(
            C.LATE_TRAIL_MULT_MIN,
            C.LATE_TRAIL_MULT - (r_now - C.LATE_TRAIL_TIGHTEN_R) / C.LATE_TRAIL_TIGHTEN_DIVISOR,
        )

    trail_mult *= C.LATE_TRAIL_WINDOW_MULT.get(sub_window, 1.0)
    lb = min(C.LATE_TRAIL_LOOKBACK, len(highs_15m))

    if pos.direction == Direction.LONG:
        hh = float(np.nanmax(highs_15m[-lb:]))
        return max(pos.stop_price, hh - trail_mult * atr15)
    else:
        ll = float(np.nanmin(lows_15m[-lb:]))
        return min(pos.stop_price, ll + trail_mult * atr15)
