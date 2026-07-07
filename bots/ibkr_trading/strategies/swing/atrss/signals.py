"""ETRS vFinal — signal detection (stateless pure functions)."""
from __future__ import annotations

from datetime import datetime, timedelta

from .config import (
    ADDON_A_R,
    ADDON_B_R,
    BREAKOUT_DIRECT_ENTRY,
    BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE,
    BREAKOUT_RETRACE_ENTRY_FRAC,
    BREAKOUT_RETRACE_LIMIT_FRAC,
    COOLDOWN_HOURS,
    MOMENTUM_TOLERANCE_ATR,
    PULLBACK_MOMENTUM_FILTER_ENABLED,
    RECOVERY_TOLERANCE_ATR,
    RECOVERY_TOLERANCE_ATR_STRONG,
    RECOVERY_TOLERANCE_ATR_TREND,
    SCORE_REVERSE_MIN,
    VOUCHER_VALID_HOURS,
)
from .models import (
    DailyState,
    Direction,
    HourlyState,
    PositionBook,
    ReentryState,
    Regime,
)


# ---------------------------------------------------------------------------
# Prerequisite: momentum-state filter
# ---------------------------------------------------------------------------

def momentum_ok(h: HourlyState, direction: int) -> bool:
    """Close must be on the correct side of EMA_mom (with tolerance)."""
    tol = MOMENTUM_TOLERANCE_ATR * h.atrh
    if direction == Direction.LONG:
        return h.close > h.ema_mom - tol
    elif direction == Direction.SHORT:
        return h.close < h.ema_mom + tol
    return False


# ---------------------------------------------------------------------------
# Short safety filter (spec Section 4)
# ---------------------------------------------------------------------------

def short_safety_ok(d: DailyState) -> bool:
    """Short entries require EMA_fast slope over 5 bars <= 0."""
    return d.ema_fast_slope_5 <= 0


def short_symbol_gate(symbol: str, d: DailyState, h: HourlyState) -> bool:
    """Per-symbol additional gates for short entries.

    QQQ: require close < EMA_slow (below higher-TF structure).
    GLD: require DI agrees + ADX >= 22.
    USO: require ADX >= 22 + DI agrees.
    """
    if symbol == "QQQ":
        return True  # Test: no per-symbol gate
    elif symbol == "GLD":
        return d.minus_di > d.plus_di and d.adx >= 22
    elif symbol == "USO":
        return d.adx >= 22 and d.minus_di > d.plus_di
    # Default: no extra gate
    return True


def compute_entry_quality(h: HourlyState, d: DailyState, direction: int) -> float:
    """Entry quality score (0-7). Higher = better setup."""
    score = 0.0
    # ADX (0-2)
    if d.adx >= 25:
        score += 2.0
    elif d.adx >= 20:
        score += 1.0
    # DI alignment (0-2)
    if direction == Direction.LONG:
        if d.plus_di > d.minus_di:
            score += 2.0
    else:
        if d.minus_di > d.plus_di:
            score += 2.0
    # EMA separation (0-1)
    if d.ema_sep_pct >= 0.25:
        score += 1.0
    elif d.ema_sep_pct >= 0.15:
        score += 0.5
    # Touch distance (0-1): closer to EMA pull = better
    if h.atrh > 0:
        td = abs(h.close - h.ema_pull) / h.atrh
        if td <= 0.05:
            score += 1.0
        elif td <= 0.10:
            score += 0.5
    # Momentum (0-1)
    if h.atrh > 0:
        if direction == Direction.LONG:
            md = (h.close - h.ema_mom) / h.atrh
        else:
            md = (h.ema_mom - h.close) / h.atrh
        if md >= 0.05:
            score += 1.0
    return score


# ---------------------------------------------------------------------------
# Candidate A: Pullback-to-Value
# ---------------------------------------------------------------------------

def pullback_signal(h: HourlyState, d: DailyState) -> Direction:
    """Return the pullback direction, or FLAT if no signal.

    LONG: trend LONG, regime TREND/STRONG, low <= ema_pull (within lookback),
          close > ema_pull (with regime-adaptive recovery tolerance).
          Directional candle (close > open) grants extra tolerance.
    SHORT: symmetric + short safety filter.
    Momentum filter is NOT applied here (handled at post-signal level for breakouts only).
    """
    if d.regime == Regime.STRONG_TREND:
        tol = RECOVERY_TOLERANCE_ATR_STRONG
    elif d.regime == Regime.TREND:
        tol = RECOVERY_TOLERANCE_ATR_TREND
    else:
        return Direction.FLAT

    if d.trend_dir == Direction.LONG:
        if (
            h.recent_pull_touch_long
            and h.close > h.ema_pull - tol * h.atrh
        ):
            if PULLBACK_MOMENTUM_FILTER_ENABLED and not momentum_ok(h, Direction.LONG):
                return Direction.FLAT
            return Direction.LONG
    elif d.trend_dir == Direction.SHORT:
        if (
            h.recent_pull_touch_short
            and h.close < h.ema_pull + tol * h.atrh
            and short_safety_ok(d)
        ):
            if PULLBACK_MOMENTUM_FILTER_ENABLED and not momentum_ok(h, Direction.SHORT):
                return Direction.FLAT
            return Direction.SHORT
    return Direction.FLAT


# ---------------------------------------------------------------------------
# Candidate B: Strong-Trend Breakout (arm-then-pullback, spec S7.2-7.3)
# ---------------------------------------------------------------------------

def check_breakout_arm(h: HourlyState, d: DailyState) -> Direction:
    """Check whether a breakout arm event occurred this bar.

    STRONG_TREND only; *high* (not close) beyond Donchian, momentum OK.
    SHORT also requires short safety filter.
    Returns arm direction or FLAT.
    """
    if d.regime != Regime.STRONG_TREND:
        return Direction.FLAT

    if d.trend_dir == Direction.LONG:
        if h.high > h.donchian_high and momentum_ok(h, Direction.LONG):
            return Direction.LONG
    elif d.trend_dir == Direction.SHORT:
        if h.low < h.donchian_low and momentum_ok(h, Direction.SHORT) and short_safety_ok(d):
            return Direction.SHORT

    return Direction.FLAT


def breakout_pullback_signal(
    h: HourlyState, d: DailyState, armed_dir: Direction,
    arm_high: float = 0.0, arm_low: float = 0.0,
) -> Direction:
    """Check for pullback trigger while breakout is armed (retracement method).

    Uses 30-50% retracement of the arm bar's range as the pullback zone.
    LONG: bar dips into retrace zone and closes above midpoint with bullish bar.
    SHORT: symmetric + short safety filter.
    """
    breakout_range = arm_high - arm_low
    if breakout_range <= 0:
        return Direction.FLAT

    if BREAKOUT_DIRECT_ENTRY:
        if armed_dir == Direction.LONG and d.trend_dir == Direction.LONG and momentum_ok(h, Direction.LONG):
            return Direction.LONG
        if (
            armed_dir == Direction.SHORT
            and d.trend_dir == Direction.SHORT
            and momentum_ok(h, Direction.SHORT)
            and short_safety_ok(d)
        ):
            return Direction.SHORT
        return Direction.FLAT

    if armed_dir == Direction.LONG:
        retrace_entry = arm_high - BREAKOUT_RETRACE_ENTRY_FRAC * breakout_range
        retrace_limit = arm_high - BREAKOUT_RETRACE_LIMIT_FRAC * breakout_range
        if (
            h.low <= retrace_entry
            and h.close > retrace_limit
            and (not BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE or h.close > h.open)
        ):
            return Direction.LONG
    elif armed_dir == Direction.SHORT:
        retrace_entry = arm_low + BREAKOUT_RETRACE_ENTRY_FRAC * breakout_range
        retrace_limit = arm_low + BREAKOUT_RETRACE_LIMIT_FRAC * breakout_range
        if (
            h.high >= retrace_entry
            and h.close < retrace_limit
            and (not BREAKOUT_REQUIRE_DIRECTIONAL_CANDLE or h.close < h.open)
            and short_safety_ok(d)
        ):
            return Direction.SHORT

    return Direction.FLAT


# ---------------------------------------------------------------------------
# Candidate C: Stop-and-Reverse
# ---------------------------------------------------------------------------

def reverse_entry_ok(h: HourlyState, d: DailyState) -> bool:
    """Check whether a reverse entry is eligible given current state.

    Requires regime ON, score >= 60, momentum-state confirms new direction.
    SHORT also requires short safety filter (spec Section 4).
    """
    if not d.regime_on:
        return False
    if d.score < SCORE_REVERSE_MIN:
        return False
    if d.trend_dir == Direction.LONG:
        return momentum_ok(h, Direction.LONG)
    elif d.trend_dir == Direction.SHORT:
        return momentum_ok(h, Direction.SHORT) and short_safety_ok(d)
    return False


# ---------------------------------------------------------------------------
# Pyramiding eligibility
# ---------------------------------------------------------------------------

def addon_a_eligible(
    pos: PositionBook, h: HourlyState, d: DailyState,
    current_r: float = 0.0,
) -> bool:
    """Add-on A: wait for pullback after +1.5R MFE.

    Requires BE triggered, MFE >= 1.5R, current R pulled back to 0.75-1.0R,
    bullish bar confirmation, and momentum alignment.
    """
    if pos.addon_a_done:
        return False
    if pos.mfe < ADDON_A_R:
        return False
    if not pos.be_triggered:
        return False
    if not (0.75 <= current_r <= 1.0):
        return False
    # Bullish bar confirmation
    if pos.direction == Direction.LONG:
        if h.close <= h.open:
            return False
    elif pos.direction == Direction.SHORT:
        if h.close >= h.open:
            return False
    return momentum_ok(h, pos.direction)


def addon_b_eligible(pos: PositionBook, h: HourlyState, d: DailyState) -> bool:
    """Add-on B ("continuation add" at +2R, STRONG_TREND only).

    MFE >= 2R, STRONG_TREND, fresh pullback signal matches direction.
    """
    if pos.addon_b_done:
        return False
    if pos.mfe < ADDON_B_R:
        return False
    if d.regime != Regime.STRONG_TREND:
        return False
    pb_dir = pullback_signal(h, d)
    return pb_dir == pos.direction


# ---------------------------------------------------------------------------
# Re-entry cooldown / reset gate
# ---------------------------------------------------------------------------

def _count_rth_hours(start: datetime, end: datetime) -> int:
    """Count RTH hours elapsed between *start* and *end*.

    RTH is defined as 09:30-16:00 ET (NYSE hours).
    Each RTH day has 6.5 hours; we count whole hour boundaries that
    fall within RTH.

    H6 fix: Clamp start to RTH open so the first partial hour (09:xx)
    isn't skipped due to the previous logic rounding down to 09:00 and
    failing the >= 09:30 check.
    """
    if end <= start:
        return 0

    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    rth_open_minutes = 9 * 60 + 30   # 09:30
    rth_close_minutes = 16 * 60       # 16:00

    count = 0
    # Clamp start to RTH open if it falls before 09:30 on that day
    start_et = start.astimezone(et)
    start_day_open = start_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if start_et < start_day_open:
        start_et = start_day_open

    # Start from next hour boundary after (clamped) start
    current = start_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    end_et = end.astimezone(et)

    while current <= end_et:
        day = current.weekday()  # 0=Mon, 6=Sun
        if day < 5:  # weekday
            # The interval [current-1h, current] overlaps RTH if:
            # - current > 09:30 (so the interval starts at or after 09:30 territory)
            # - current-1h < 16:00 (so the interval hasn't passed close)
            cur_minutes = current.hour * 60 + current.minute
            interval_start = max(
                (current - timedelta(hours=1)).hour * 60 + (current - timedelta(hours=1)).minute,
                rth_open_minutes,
            )
            interval_end = min(cur_minutes, rth_close_minutes)
            if interval_end > interval_start and cur_minutes > rth_open_minutes:
                count += 1
        current += timedelta(hours=1)

    return count


def same_direction_reentry_allowed(
    reentry: ReentryState,
    direction: Direction,
    now: datetime,
    regime: Regime,
    trend_dir: Direction = Direction.FLAT,
) -> bool:
    """Check whether a same-direction re-entry is permitted.

    Tiered re-entry logic:
    1. Quality exit (MFE >= 0.3R or TP exit) → immediate re-entry, no gates.
    2. STRONG_TREND → cooldown only, no reset gate.
    3. Stall exit (FLATTEN_STALL etc.) → cooldown only, no reset gate.
    4. Otherwise → full cooldown + reset gate (close must cross EMA_pull).
    """
    # No previous exit → allowed
    if reentry.last_exit_time is None:
        return True

    # Opposite direction → always allowed
    if reentry.last_exit_dir != direction:
        return True

    # Quality gate: prior trade showed some development (MFE >= 0.3R or TP exit)
    # → waive both cooldown and reset gate (it's a continuation, not revenge)
    quality_exit = (
        reentry.last_exit_mfe >= 0.3
        or reentry.last_exit_reason in ("TP1", "TP2")
    )
    if quality_exit:
        return True

    # STRONG_TREND: waive reset gate (momentum is its own confirmation)
    if regime == Regime.STRONG_TREND:
        # Still require time cooldown but not reset
        cooldown = COOLDOWN_HOURS.get(regime.value, 24)
        rth_elapsed = _count_rth_hours(reentry.last_exit_time, now)
        return rth_elapsed >= cooldown

    # Stall exits: waive reset gate (stall = trade didn't develop, not a
    # directional signal; re-entry after cooldown is appropriate)
    stall_exit = reentry.last_exit_reason in (
        "FLATTEN_STALL", "FLATTEN_MID_STALL", "EARLY_STALL_PARTIAL",
    )
    if stall_exit:
        cooldown = COOLDOWN_HOURS.get(regime.value, 24)
        rth_elapsed = _count_rth_hours(reentry.last_exit_time, now)
        return rth_elapsed >= cooldown

    # Reset gate (required for non-quality, non-stall exits outside STRONG_TREND)
    if direction == Direction.LONG and not reentry.reset_seen_long:
        return False
    if direction == Direction.SHORT and not reentry.reset_seen_short:
        return False

    # Check for valid voucher (bypasses time cooldown per spec Section 4)
    has_voucher = _has_valid_voucher(reentry, direction, now, trend_dir)
    if has_voucher:
        return True

    # Time gate — RTH hours (spec Section 8.1)
    cooldown = COOLDOWN_HOURS.get(regime.value, 24)
    rth_elapsed = _count_rth_hours(reentry.last_exit_time, now)
    if rth_elapsed < cooldown:
        return False

    return True


def _has_valid_voucher(
    reentry: ReentryState,
    direction: Direction,
    now: datetime,
    trend_dir: Direction = Direction.FLAT,
) -> bool:
    """Check if a valid (unexpired, matching-direction) voucher exists."""
    if reentry.voucher_granted_time is None:
        return False
    if (now - reentry.voucher_granted_time) > timedelta(hours=VOUCHER_VALID_HOURS):
        return False
    # Spec §3.3: confirmed bias must match voucher direction (FLAT never matches)
    if trend_dir != direction:
        return False
    if direction == Direction.LONG:
        return reentry.voucher_long
    elif direction == Direction.SHORT:
        return reentry.voucher_short
    return False


def consume_voucher(reentry: ReentryState, direction: Direction) -> None:
    """Consume a voucher after use (one-time use per spec Section 4.2)."""
    if direction == Direction.LONG:
        reentry.voucher_long = False
    elif direction == Direction.SHORT:
        reentry.voucher_short = False
