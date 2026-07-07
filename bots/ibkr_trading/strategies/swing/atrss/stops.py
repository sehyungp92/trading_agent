"""ETRS vFinal — stop calculation and lifecycle."""
from __future__ import annotations

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from .config import BE_ATR_OFFSET, PROFIT_FLOOR, PROFIT_FLOOR_SHORT
from .models import DailyState, Direction, HourlyState


# ---------------------------------------------------------------------------
# Initial stop: hybrid ATR + structure-aware
# ---------------------------------------------------------------------------

def compute_initial_stop(
    direction: int,
    entry: float,
    signal_bar: HourlyState,
    daily_atr: float,
    hourly_atr: float,
    d_mult: float,
    h_mult: float,
    tick_size: float,
) -> float:
    """Wider of ATR stop vs signal-candle structure +-1 tick.

    LONG: stop below entry; SHORT: stop above entry.
    """
    stop_dist_atr = max(d_mult * daily_atr, h_mult * hourly_atr)

    if direction == Direction.LONG:
        stop_atr = entry - stop_dist_atr
        stop_struct = signal_bar.low - tick_size  # signal low - 1 tick
        raw = min(stop_atr, stop_struct)           # wider = lower for long
        return round_to_tick(raw, tick_size, "down")
    else:
        stop_atr = entry + stop_dist_atr
        stop_struct = signal_bar.high + tick_size  # signal high + 1 tick
        raw = max(stop_atr, stop_struct)           # wider = higher for short
        return round_to_tick(raw, tick_size, "up")


# ---------------------------------------------------------------------------
# Break-even stop
# ---------------------------------------------------------------------------

def compute_be_stop(
    direction: int,
    entry_price: float,
    daily_atr: float,
    tick_size: float,
) -> float:
    """BE + 0.1 * daily_ATR20."""
    offset = BE_ATR_OFFSET * daily_atr
    if direction == Direction.LONG:
        return round_to_tick(entry_price + offset, tick_size, "up")
    else:
        return round_to_tick(entry_price - offset, tick_size, "down")


# ---------------------------------------------------------------------------
# Chandelier trailing stop
# ---------------------------------------------------------------------------

def compute_chandelier_stop(
    direction: int,
    daily_state: DailyState,
    chand_mult: float,
    tick_size: float,
) -> float:
    """Trailing stop based on 20-day HH/LL and daily ATR20.

    LONG:  HH(20d) - chand_mult * DailyATR20
    SHORT: LL(20d) + chand_mult * DailyATR20
    """
    if direction == Direction.LONG:
        raw = daily_state.hh_20d - chand_mult * daily_state.atr20
        return round_to_tick(raw, tick_size, "down")
    else:
        raw = daily_state.ll_20d + chand_mult * daily_state.atr20
        return round_to_tick(raw, tick_size, "up")


# ---------------------------------------------------------------------------
# Profit floor (spec Section 10.4)
# ---------------------------------------------------------------------------

def apply_profit_floor(
    direction: int,
    entry_price: float,
    risk_per_unit: float,
    mfe_r: float,
    current_stop: float,
    tick_size: float,
) -> float:
    """Enforce minimum profit lock based on MFE thresholds.

    Uses PROFIT_FLOOR_SHORT for shorts (tighter locks) and PROFIT_FLOOR for longs.
    Iterates from highest threshold down; first match wins.
    """
    floor = PROFIT_FLOOR_SHORT if direction == Direction.SHORT else PROFIT_FLOOR
    for mfe_threshold in sorted(floor, reverse=True):
        if mfe_r >= mfe_threshold:
            min_profit_r = floor[mfe_threshold]
            if direction == Direction.LONG:
                min_stop = entry_price + min_profit_r * risk_per_unit
                current_stop = max(current_stop, round_to_tick(min_stop, tick_size, "up"))
            else:
                min_stop = entry_price - min_profit_r * risk_per_unit
                current_stop = min(current_stop, round_to_tick(min_stop, tick_size, "down"))
            break
    return current_stop
