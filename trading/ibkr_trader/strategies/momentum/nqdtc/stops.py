"""NQ Dominant Trend Capture v2.0 — stop placement and exit management."""
from __future__ import annotations

import math

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from . import config as C
from .models import Direction, EntrySubtype, ExitTier, PositionState, TPLevel


# ---------------------------------------------------------------------------
# Initial stop by entry subtype (Section 16.5)
# ---------------------------------------------------------------------------

def compute_initial_stop(
    subtype: EntrySubtype,
    direction: Direction,
    entry_price: float,
    box_high: float,
    box_low: float,
    box_mid: float,
    atr14_30m: float,
    hold_ref: float = 0.0,
    tick_size: float = 0.25,
) -> float:
    """Compute initial stop based on entry subtype."""
    if subtype in (EntrySubtype.A_LATCH, EntrySubtype.A_RETEST, EntrySubtype.MARKET_FALLBACK):
        # Structural midpoint: box_mid ± A_STOP_ATR_MULT*ATR14_30m
        if direction == Direction.LONG:
            raw = box_mid - C.A_STOP_ATR_MULT * atr14_30m
        else:
            raw = box_mid + C.A_STOP_ATR_MULT * atr14_30m

    elif subtype == EntrySubtype.C_STANDARD:
        # R-efficient structural: box_edge ± 0.60*ATR14_30m
        if direction == Direction.LONG:
            raw = box_high - 0.60 * atr14_30m
        else:
            raw = box_low + 0.60 * atr14_30m

    elif subtype == EntrySubtype.C_CONTINUATION:
        if C.C_CONT_STOP_USE_BOX_EDGE:
            # Structural stop like C_standard: box_edge - 0.60*ATR14_30m
            if direction == Direction.LONG:
                raw = box_high - 0.60 * atr14_30m
            else:
                raw = box_low + 0.60 * atr14_30m
        else:
            # Legacy hold-based: hold_ref ± 0.80*ATR14_30m
            if direction == Direction.LONG:
                raw = hold_ref - 0.80 * atr14_30m
            else:
                raw = hold_ref + 0.80 * atr14_30m

    elif subtype == EntrySubtype.B_SWEEP:
        # B_sweep structural stop: box_mid ± B_STOP_ATR_MULT*ATR14_30m
        if direction == Direction.LONG:
            raw = box_mid - C.B_STOP_ATR_MULT * atr14_30m
        else:
            raw = box_mid + C.B_STOP_ATR_MULT * atr14_30m
    else:
        raw = entry_price

    # Phase 3.1: cap stop distance at MAX_STOP_ATR_MULT * ATR14_30m
    if atr14_30m > 0:
        max_dist = C.MAX_STOP_ATR_MULT * atr14_30m
        if direction == Direction.LONG:
            min_stop = entry_price - max_dist
            raw = max(raw, min_stop)
        else:
            max_stop = entry_price + max_dist
            raw = min(raw, max_stop)

    if direction == Direction.LONG:
        return round_to_tick(raw, tick_size, "down")
    return round_to_tick(raw, tick_size, "up")


# ---------------------------------------------------------------------------
# Profit-funded BE (Section 17.4)
# ---------------------------------------------------------------------------

def compute_be_stop(
    direction: Direction,
    entry_price: float,
    atr14_5m: float,
    tick_size: float = 0.25,
) -> float:
    """BE + buffer after TP1 fill."""
    buffer = max(C.BE_BUFFER_ATR_5M * atr14_5m, 2 * tick_size)
    if direction == Direction.LONG:
        return round_to_tick(entry_price + buffer, tick_size, "up")
    return round_to_tick(entry_price - buffer, tick_size, "down")


# ---------------------------------------------------------------------------
# TP level computation (Section 17.3)
# ---------------------------------------------------------------------------

def compute_tp_levels(
    direction: Direction,
    entry_price: float,
    r_points: float,
    exit_tier: ExitTier,
    total_qty: int,
    tick_size: float = 0.25,
) -> list[TPLevel]:
    """Build TP levels from exit tier schedule."""
    schedule = C.EXIT_TIERS.get(exit_tier.value, C.EXIT_TIERS["Neutral"])
    levels: list[TPLevel] = []
    remaining = total_qty

    for r_target, pct in schedule:
        qty = max(1, round(total_qty * pct))
        qty = min(qty, remaining)
        if qty <= 0:
            continue
        if direction == Direction.LONG:
            price = round_to_tick(entry_price + r_target * r_points, tick_size, "up")
        else:
            price = round_to_tick(entry_price - r_target * r_points, tick_size, "down")
        levels.append(TPLevel(r_target=r_target, pct=pct, qty=qty))
        remaining -= qty

    return levels


# ---------------------------------------------------------------------------
# Exit tier determination (Section 17.2)
# ---------------------------------------------------------------------------

def determine_exit_tier(composite_regime: str, quality_mult: float) -> ExitTier:
    """Determine exit tier, frozen at entry."""
    base_map = {
        "Aligned": ExitTier.ALIGNED,
        "Neutral": ExitTier.NEUTRAL,
        "Caution": ExitTier.CAUTION,
        "Range": ExitTier.CAUTION,
        "Counter": ExitTier.CAUTION,
    }
    tier = base_map.get(composite_regime, ExitTier.NEUTRAL)

    # Downgrade by quality
    tiers = [ExitTier.ALIGNED, ExitTier.NEUTRAL, ExitTier.CAUTION]
    idx = tiers.index(tier) if tier in tiers else 1
    if quality_mult < 0.25:
        return ExitTier.CAUTION
    if 0.25 <= quality_mult < 0.50:
        idx = min(idx + 1, 2)
        return tiers[idx]
    return tier


# ---------------------------------------------------------------------------
# Chandelier trail tier selection (Section 17.5)
# ---------------------------------------------------------------------------

def chandelier_params(open_trade_r: float, mm_reached: bool) -> tuple[int, float]:
    """Return (lookback, mult) for chandelier trailing based on open R."""
    for min_r, max_r, mm_req, lookback, mult in C.CHANDELIER_TIERS:
        if min_r <= open_trade_r < max_r:
            if mm_req and not mm_reached:
                continue
            return lookback, mult
    # Default fallback
    return 12, 3.0


# ---------------------------------------------------------------------------
# Stale exit check (Section 17.6)
# ---------------------------------------------------------------------------

def stale_exit_check(
    bars_since_entry_30m: int,
    open_r: float,
    mode: str,
    bridge_extra_bars: int = 0,
    tp1_filled: bool = False,
) -> bool:
    """Return True if position is stale and should be exited.

    bridge_extra_bars: additional bars from overnight bridge extension (Section 17.6).
    tp1_filled: if True, never stale-exit (trade reached TP1).
    """
    if tp1_filled:
        return False
    if mode in ("DEGRADED", "RANGE"):
        threshold_bars = C.STALE_BARS_DEGRADED
    else:
        threshold_bars = C.STALE_BARS_NORMAL
    threshold_bars += bridge_extra_bars
    return bars_since_entry_30m >= threshold_bars and open_r < C.STALE_R_THRESHOLD


# ---------------------------------------------------------------------------
# TP1-only cap for DEGRADED / RANGE (Section 17.7, fix #3)
# ---------------------------------------------------------------------------

def should_cap_tp1_only(chop_mode: str, regime_4h: str, mode: str | None = None) -> bool:
    """Return True if exits should be capped at TP1 only.

    Section 17.7: RANGE and DEGRADED cap at TP1 only + shortened stale timer.
    """
    cap_mode = mode or C.TP1_ONLY_CAP_MODE
    if cap_mode == "off":
        return False
    if cap_mode == "degraded_only":
        return chop_mode == "DEGRADED"
    if cap_mode == "range_only":
        return regime_4h == "RANGE"
    return chop_mode == "DEGRADED" or regime_4h == "RANGE"


def compute_mfe_ratcheted_stop(
    direction: Direction,
    entry_price: float,
    initial_r_points: float,
    peak_r_initial: float,
    tick_size: float = 0.25,
) -> float | None:
    """Return a stop that locks fixed R at configured MFE tiers, if enabled."""
    if not C.MFE_RATCHET_TIERS_ENABLED or initial_r_points <= 0:
        return None
    lock_r = 0.0
    tiers = (
        (C.MFE_RATCHET_T1_R, C.MFE_RATCHET_T1_LOCK_R),
        (C.MFE_RATCHET_T2_R, C.MFE_RATCHET_T2_LOCK_R),
        (C.MFE_RATCHET_T3_R, C.MFE_RATCHET_T3_LOCK_R),
    )
    for trigger_r, tier_lock_r in tiers:
        if peak_r_initial >= trigger_r:
            lock_r = max(lock_r, tier_lock_r)
    if lock_r <= 0:
        return None
    if direction == Direction.LONG:
        raw = entry_price + lock_r * initial_r_points
        return round_to_tick(raw, tick_size, "down")
    raw = entry_price - lock_r * initial_r_points
    return round_to_tick(raw, tick_size, "up")


# ---------------------------------------------------------------------------
# Overnight bridge (Section 17.6, fix #9)
# ---------------------------------------------------------------------------

def overnight_bridge_eligible(
    close: float,
    box_high: float,
    box_low: float,
    direction: Direction,
    regime_4h: str,
    trend_dir_4h: Direction,
) -> bool:
    """Check if position qualifies for overnight stale-timer bridge.

    At RTH close, if price holds breakout side AND 4H TRENDING supports
    direction, extend stale timer to next RTH open + 4 bars.
    """
    if regime_4h != "TRENDING":
        return False
    if trend_dir_4h != direction:
        return False
    # Price must hold breakout side
    if direction == Direction.LONG:
        return close > box_high
    else:
        return close < box_low
