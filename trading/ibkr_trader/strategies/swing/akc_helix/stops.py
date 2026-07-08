"""AKC-Helix Swing v2.0 — stop calculation and lifecycle."""
from __future__ import annotations

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from .config import (
    BE_ATR1H_OFFSET,
    R_BE,
    STOP_1H_HIGHVOL,
    STOP_1H_STD,
    STOP_4H_MULT,
    TRAIL_BASE,
    TRAIL_CHOP_PENALTY,
    TRAIL_FLIP_PENALTY,
    TRAIL_MAX,
    TRAIL_MIN,
    TRAIL_MOM_BONUS,
    TRAIL_R_DIV,
    HIGH_VOL_PCT,
)
from .models import Direction


# ---------------------------------------------------------------------------
# Stop0 (spec s10)
# ---------------------------------------------------------------------------

def compute_stop0_4h(
    direction: Direction,
    pivot2_price: float,
    atr_4h: float,
    tick_size: float,
) -> float:
    """4H stop: L2 - 0.75*ATR4H (long) or H2 + 0.75*ATR4H (short)."""
    if direction == Direction.LONG:
        raw = pivot2_price - STOP_4H_MULT * atr_4h
        return round_to_tick(raw, tick_size, "down")
    else:
        raw = pivot2_price + STOP_4H_MULT * atr_4h
        return round_to_tick(raw, tick_size, "up")


def compute_stop0_1h(
    direction: Direction,
    pivot2_price: float,
    atr_1h: float,
    vol_pct: float,
    tick_size: float,
) -> float:
    """1H stop: 0.50 or 0.75 ATR1H mult depending on vol environment."""
    mult = STOP_1H_HIGHVOL if vol_pct > HIGH_VOL_PCT else STOP_1H_STD
    if direction == Direction.LONG:
        raw = pivot2_price - mult * atr_1h
        return round_to_tick(raw, tick_size, "down")
    else:
        raw = pivot2_price + mult * atr_1h
        return round_to_tick(raw, tick_size, "up")


# ---------------------------------------------------------------------------
# BE stop (spec s13.2)
# ---------------------------------------------------------------------------

def compute_be_stop(
    direction: Direction,
    avg_entry: float,
    atr_1h: float,
    tick_size: float,
) -> float:
    """Buffered BE: AvgEntry minus 0.15*ATR1H cushion (spec s13.2)."""
    offset = BE_ATR1H_OFFSET * atr_1h
    if direction == Direction.LONG:
        return round_to_tick(avg_entry - offset, tick_size, "down")
    else:
        return round_to_tick(avg_entry + offset, tick_size, "up")


# ---------------------------------------------------------------------------
# Ratchet stop (spec s13.3)
# ---------------------------------------------------------------------------

def compute_ratchet_stop(
    direction: Direction,
    avg_entry: float,
    r_price: float,
    tick_size: float,
) -> float:
    """Ratchet: lock ~+1R on remainder (spec s13.3).

    At +2.5R partial, ratchet to avg_entry + 1.0*r_price (long).
    """
    if direction == Direction.LONG:
        raw = avg_entry + 1.0 * r_price
        return round_to_tick(raw, tick_size, "up")
    else:
        raw = avg_entry - 1.0 * r_price
        return round_to_tick(raw, tick_size, "down")


# ---------------------------------------------------------------------------
# Right-then-stopped leakage guard
# ---------------------------------------------------------------------------

def should_arm_rts_guard(
    *,
    max_mfe_r: float,
    current_r: float,
    bars_held: int,
    fading_bars: int,
    trail_active: bool,
    min_mfe_r: float,
    min_giveback_r: float,
    min_bars: int,
    fade_bars: int,
    max_mfe_r_limit: float,
) -> bool:
    """Return True when a small/mid MFE winner is decaying after entry.

    The guard is intentionally stateful and disabled unless min_mfe_r is set.
    It only reacts after a trade has shown positive MFE, then given back enough
    R with optional momentum-fade confirmation.
    """
    if min_mfe_r <= 0.0:
        return False
    if trail_active:
        return False
    if bars_held < max(0, min_bars):
        return False
    if max_mfe_r < min_mfe_r:
        return False
    if max_mfe_r_limit > 0.0 and max_mfe_r > max_mfe_r_limit:
        return False
    if fading_bars < max(0, fade_bars):
        return False
    return (max_mfe_r - current_r) >= max(0.0, min_giveback_r)


def compute_rts_guard_stop(
    *,
    direction: Direction,
    avg_entry: float,
    r_price: float,
    current_price: float,
    tick_size: float,
    floor_r: float,
) -> float | None:
    """Compute a non-marketable protective stop at a configured R floor."""
    if r_price <= 0.0 or tick_size <= 0.0:
        return None
    if direction == Direction.LONG:
        stop = round_to_tick(avg_entry + floor_r * r_price, tick_size, "down")
        if stop >= current_price:
            return None
        return stop
    stop = round_to_tick(avg_entry - floor_r * r_price, tick_size, "up")
    if stop <= current_price:
        return None
    return stop


def should_flatten_rts_failure(
    *,
    max_mfe_r: float,
    current_r: float,
    bars_held: int,
    fading_bars: int,
    trail_active: bool,
    min_mfe_r: float,
    min_giveback_r: float,
    min_bars: int,
    fade_bars: int,
    max_mfe_r_limit: float,
    flatten_r: float,
) -> bool:
    """Return True when an armed right-then-stopped guard has already failed."""
    if flatten_r <= -900.0:
        return False
    if current_r > flatten_r:
        return False
    return should_arm_rts_guard(
        max_mfe_r=max_mfe_r,
        current_r=current_r,
        bars_held=bars_held,
        fading_bars=fading_bars,
        trail_active=trail_active,
        min_mfe_r=min_mfe_r,
        min_giveback_r=min_giveback_r,
        min_bars=min_bars,
        fade_bars=fade_bars,
        max_mfe_r_limit=max_mfe_r_limit,
    )


# ---------------------------------------------------------------------------
# Momentum check (spec s14.2)
# ---------------------------------------------------------------------------

def is_momentum_strong(
    macd_now: float,
    macd_5ago: float,
    hist_now: float,
    direction: Direction = Direction.LONG,
) -> bool:
    """Momentum is strong if MACD trending in trade direction and histogram confirms.

    Spec s14.2:
    Long:  macd_1h[t] > macd_1h[t-5] AND hist_1h[t] > 0
    Short: macd_1h[t] < macd_1h[t-5] AND hist_1h[t] < 0
    """
    if direction == Direction.LONG:
        return macd_now > macd_5ago and hist_now > 0
    else:
        return macd_now < macd_5ago and hist_now < 0


# ---------------------------------------------------------------------------
# Trailing multiplier (spec s14.1-14.4)
# ---------------------------------------------------------------------------

def compute_trailing_mult(
    r_state: float,
    momentum_strong: bool,
    regime_deteriorated: bool,
    regime_flipped: bool,
    mult_bonus: float,
    mom_cont_penalty: float = 0.0,
) -> float:
    """Adaptive trailing multiplier (spec s14.1).

    Base = max(2.0, 4.0 - R_state/5.0).
    Bonuses/penalties for momentum, regime deterioration, flip.
    Clamped to [TRAIL_MIN=2.0, TRAIL_MAX=4.0].
    """
    base = TRAIL_BASE - r_state / TRAIL_R_DIV
    if momentum_strong:
        base += TRAIL_MOM_BONUS
    if regime_deteriorated:
        base -= TRAIL_CHOP_PENALTY
    if regime_flipped:
        base -= TRAIL_FLIP_PENALTY
    base += mult_bonus
    base -= mom_cont_penalty
    return max(TRAIL_MIN, min(TRAIL_MAX, base))


# ---------------------------------------------------------------------------
# Chandelier trailing stop (spec s14.3)
# ---------------------------------------------------------------------------

def compute_chandelier_stop(
    direction: Direction,
    highs: list[float],
    lows: list[float],
    lookback: int,
    atr_1h: float,
    mult: float,
    tick_size: float,
) -> float:
    """Chandelier stop based on highest high / lowest low over lookback bars.

    LONG:  HH(lookback) - mult * ATR1H
    SHORT: LL(lookback) + mult * ATR1H
    """
    if direction == Direction.LONG:
        if highs:
            window = highs[-lookback:] if len(highs) >= lookback else highs
            hh = max(window)
        else:
            return 0.0
        raw = hh - mult * atr_1h
        return round_to_tick(raw, tick_size, "down")
    else:
        if lows:
            window = lows[-lookback:] if len(lows) >= lookback else lows
            ll = min(window)
        else:
            return 0.0
        raw = ll + mult * atr_1h
        return round_to_tick(raw, tick_size, "up")
