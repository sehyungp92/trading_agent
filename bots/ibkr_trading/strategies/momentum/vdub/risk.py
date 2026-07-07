"""Vdubus NQ v4.0 — stops, sizing, risk gates, viability, execution prices."""
from __future__ import annotations

from typing import Optional

from . import config as C
from .models import (
    DayCounters, Direction, PivotPoint, PositionState,
    SessionWindow, SubWindow, VolState,
)


# ---------------------------------------------------------------------------
# Initial stop (Section 13)
# ---------------------------------------------------------------------------

def compute_initial_stop(
    entry_price: float, direction: Direction,
    pivots_1h: list[PivotPoint], atr1h: float, atr15: float,
) -> float:
    """Structure-first stop with ATR guardrail + clamp."""
    if direction == Direction.LONG:
        pivot = _last_pivot_of_type(pivots_1h, "low")
        structure_stop = (pivot.price - C.STRUCTURE_STOP_ATR_MULT * atr1h
                          if pivot else entry_price - C.ATR_STOP_MULT * atr15)
        atr_stop = entry_price - C.ATR_STOP_MULT * atr15
        stop = min(structure_stop, atr_stop)  # wider (farther from entry)
    else:
        pivot = _last_pivot_of_type(pivots_1h, "high")
        structure_stop = (pivot.price + C.STRUCTURE_STOP_ATR_MULT * atr1h
                          if pivot else entry_price + C.ATR_STOP_MULT * atr15)
        atr_stop = entry_price + C.ATR_STOP_MULT * atr15
        stop = max(structure_stop, atr_stop)  # wider

    return _clamp_stop(entry_price, stop, direction)


def _last_pivot_of_type(pivots: list[PivotPoint], ptype: str) -> Optional[PivotPoint]:
    for p in reversed(pivots):
        if p.ptype == ptype:
            return p
    return None


def _clamp_stop(entry: float, stop: float, direction: Direction) -> float:
    dist = abs(entry - stop)
    if dist < C.MIN_STOP_POINTS:
        dist = float(C.MIN_STOP_POINTS)
    elif dist > C.MAX_STOP_POINTS:
        dist = float(C.MAX_STOP_POINTS)
    tick = C.NQ_SPEC["tick"]
    if direction == Direction.LONG:
        stop = entry - dist
    else:
        stop = entry + dist
    return round(stop / tick) * tick


# ---------------------------------------------------------------------------
# Position sizing (Section 12)
# ---------------------------------------------------------------------------

def compute_unit_risk(nav: float, vol_state: VolState) -> float:
    vf = C.VOL_FACTOR.get(vol_state.value, 1.0)
    return nav * C.BASE_RISK_PCT * vf


def compute_effective_risk(
    unit_risk: float, class_mult: float, session_mult: float,
) -> float:
    return unit_risk * class_mult * session_mult


def compute_qty(effective_risk: float, r_points: float) -> int:
    if r_points <= 0:
        return 0
    return int(effective_risk // (r_points * C.NQ_SPEC["point_value"]))


def compute_addon_risk(effective_risk: float) -> float:
    return effective_risk * C.PYRAMID_ADD_RISK_MULT


def compute_wr_multiplier(recent_results: list[bool], window: int = 20) -> float:
    """Scale position size by rolling win rate over last `window` trades.

    Returns a multiplier in [0.3, 1.0]:
      WR >= 50%: 1.0x
      WR 40-50%: 0.7x
      WR 30-40%: 0.5x
      WR < 30%:  0.3x
    """
    if len(recent_results) < 5:
        return 1.0  # not enough data yet
    last_n = recent_results[-window:]
    wr = sum(last_n) / len(last_n)
    if wr >= 0.50:
        return 1.0
    if wr >= 0.40:
        return 0.7
    if wr >= 0.30:
        return 0.5
    return 0.3


# ---------------------------------------------------------------------------
# Risk gates (Section 12.5–12.7)
# ---------------------------------------------------------------------------

def pass_risk_gates(
    counters: DayCounters, direction: Direction,
    open_risk_usd: float, new_risk_usd: float, unit_risk: float,
) -> tuple[bool, str]:
    """Returns (ok, denial_reason)."""
    # Daily breaker
    if counters.breaker_hit:
        return False, "daily_breaker"
    if unit_risk > 0 and counters.daily_realized_pnl <= C.DAILY_BREAKER_MULT * unit_risk:
        counters.breaker_hit = True
        return False, "daily_breaker"

    # Direction caps
    if direction == Direction.LONG and counters.long_fills >= C.MAX_LONGS_PER_DAY:
        return False, "long_cap"
    if direction == Direction.SHORT and counters.short_fills >= C.MAX_SHORTS_PER_DAY:
        return False, "short_cap"

    # Heat cap
    if open_risk_usd + new_risk_usd > C.HEAT_CAP_MULT * unit_risk:
        return False, "heat_cap"

    return True, ""


# ---------------------------------------------------------------------------
# Pyramiding check (Section 12.8)
# ---------------------------------------------------------------------------

def pyramid_eligible(
    pos: PositionState, direction: Direction,
    new_entry_price: float, counters: DayCounters,
) -> bool:
    """Check if add-on is allowed."""
    if pos.direction != direction:
        return False
    if pos.stage == "ACTIVE_RISK":
        return False
    # Existing must be >= +1R
    unreal_r = _unrealized_r(pos)
    if unreal_r < 1.0:
        return False
    # Entry must be above stop (long) / below stop (short) — no chasing
    if direction == Direction.LONG:
        if new_entry_price <= pos.stop_price:
            return False
    else:
        if new_entry_price >= pos.stop_price:
            return False
    # Not already used
    if direction == Direction.LONG and counters.addon_used_long:
        return False
    if direction == Direction.SHORT and counters.addon_used_short:
        return False
    return True


def _unrealized_r(pos: PositionState) -> float:
    if pos.r_points <= 0:
        return 0.0
    # Use highest/lowest as proxy for current price
    if pos.direction == Direction.LONG:
        return (pos.highest_since_entry - pos.entry_price) / pos.r_points
    return (pos.entry_price - pos.lowest_since_entry) / pos.r_points


# ---------------------------------------------------------------------------
# Viability (Section 14)
# ---------------------------------------------------------------------------

def pass_viability(
    qty: int, r_points: float, sub_window: SubWindow,
) -> tuple[bool, str]:
    """Cost/risk check. Returns (ok, reason)."""
    window_key = sub_window.value
    slip_ticks = C.SLIP_TICKS_BY_WINDOW.get(window_key, 1)
    tick_val = C.NQ_SPEC["tick_value"]

    slip_cost = slip_ticks * tick_val * qty
    fees_cost = C.RT_COMM_FEES * qty
    total_cost = slip_cost + fees_cost
    risk_usd = r_points * C.NQ_SPEC["point_value"] * qty
    if risk_usd <= 0:
        return False, "zero_risk"
    if total_cost / risk_usd > C.COST_RISK_MAX:
        return False, "cost_risk"
    return True, ""


# ---------------------------------------------------------------------------
# Entry order prices (Section 15)
# ---------------------------------------------------------------------------

def compute_entry_prices(
    trigger_bar_high: float, trigger_bar_low: float,
    atr15_ticks: float, direction: Direction,
) -> tuple[float, float]:
    """Returns (stop_entry, limit_entry) for stop-limit order."""
    tick = C.NQ_SPEC["tick"]
    offset = int(round(C.OFFSET_TICKS_ATR_FRAC * atr15_ticks))
    offset = max(C.OFFSET_TICKS_MIN, min(C.OFFSET_TICKS_MAX, offset))

    if direction == Direction.LONG:
        stop_entry = trigger_bar_high + C.BUFFER_TICKS * tick
        limit_entry = stop_entry + offset * tick
    else:
        stop_entry = trigger_bar_low - C.BUFFER_TICKS * tick
        limit_entry = stop_entry - offset * tick

    return round(stop_entry / tick) * tick, round(limit_entry / tick) * tick
