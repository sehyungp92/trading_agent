"""Vdubus NQ v4.0 — entry signal detection (Type A, Type B, Predator, momentum)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import config as C
from .models import Direction, PivotPoint, SubWindow


# ---------------------------------------------------------------------------
# Momentum confirmation (Section 7)
# ---------------------------------------------------------------------------

def slope_ok(mom15: np.ndarray) -> tuple[bool, bool]:
    """Returns (long_ok, short_ok) from decelerating-slope logic."""
    n = len(mom15)
    if n < C.MOM_N + C.SLOPE_LB + 1:
        return False, False

    t = -1
    slope = float(mom15[t] - mom15[t - C.SLOPE_LB])
    slope_prev = float(mom15[t - 1] - mom15[t - 1 - C.SLOPE_LB])

    lookback = mom15[-1 - C.MOM_N:-1]
    valid = lookback[~np.isnan(lookback)]
    if len(valid) == 0:
        return False, False
    mn, mx = float(np.min(valid)), float(np.max(valid))
    rng = mx - mn
    floor = mn + C.FLOOR_PCT * rng
    ceiling = mx - C.FLOOR_PCT * rng

    long_ok = (slope > 0) or (slope > slope_prev and float(mom15[t]) > floor)
    short_ok = (slope < 0) or (slope < slope_prev and float(mom15[t]) < ceiling)
    return long_ok, short_ok


# ---------------------------------------------------------------------------
# VWAP touch + selection (Section 9.1 / 2.5)
# ---------------------------------------------------------------------------

def _find_last_touch_idx(
    lows: np.ndarray, highs: np.ndarray,
    vwap_vals: np.ndarray, direction: Direction, lookback: int,
) -> int:
    """Return index of most recent bar (within lookback) that touched VWAP.
    Returns -1 if no touch found."""
    n = len(lows)
    for i in range(n - 1, max(n - 1 - lookback, -1), -1):
        if np.isnan(vwap_vals[i]):
            continue
        if direction == Direction.LONG and lows[i] <= vwap_vals[i]:
            return i
        if direction == Direction.SHORT and highs[i] >= vwap_vals[i]:
            return i
    return -1


def choose_vwap_used(
    lows_15m: np.ndarray, highs_15m: np.ndarray,
    svwap: np.ndarray, vwap_a: np.ndarray,
    direction: Direction, lookback: int = C.TOUCH_LOOKBACK_15M,
) -> tuple[Optional[float], str]:
    """Select VWAP_used (most recently touched). Returns (value, source)."""
    touch_s = _find_last_touch_idx(lows_15m, highs_15m, svwap, direction, lookback)
    touch_a = _find_last_touch_idx(lows_15m, highs_15m, vwap_a, direction, lookback)

    if touch_s < 0 and touch_a < 0:
        return None, ""
    if touch_s >= touch_a:
        return float(svwap[-1]) if not np.isnan(svwap[-1]) else None, "session"
    val = float(vwap_a[-1]) if len(vwap_a) > 0 and not np.isnan(vwap_a[-1]) else None
    return val, "anchor"


# ---------------------------------------------------------------------------
# Type A — Trend Pullback Reclaim (Section 9.1)
# ---------------------------------------------------------------------------

def type_a_check(
    closes_15m: np.ndarray, lows_15m: np.ndarray, highs_15m: np.ndarray,
    svwap: np.ndarray, vwap_a: np.ndarray, atr15_val: float,
    direction: Direction, sub_window: SubWindow,
) -> Optional[dict]:
    """Check Type A conditions. Returns signal dict or None."""
    vwap_used, source = choose_vwap_used(
        lows_15m, highs_15m, svwap, vwap_a, direction)
    if vwap_used is None:
        return None

    close = float(closes_15m[-1])
    b_cap = C.VWAP_CAP_CORE if sub_window == SubWindow.CORE else C.VWAP_CAP_OPEN_EVE

    if direction == Direction.LONG:
        if close <= vwap_used:
            return None
        if close > vwap_used + b_cap * atr15_val:
            return None
    else:
        if close >= vwap_used:
            return None
        if close < vwap_used - b_cap * atr15_val:
            return None

    return {"type": "A", "dir": direction, "vwap_used": vwap_used, "source": source}


# ---------------------------------------------------------------------------
# Type B — Breakout Retest (Section 9.2)
# ---------------------------------------------------------------------------

def _highest_pivot_high(pivots: list[PivotPoint], min_idx: int) -> Optional[float]:
    """Highest confirmed pivot high with idx >= min_idx."""
    highs = [p.price for p in pivots if p.ptype == "high" and p.idx >= min_idx]
    return max(highs) if highs else None


def _lowest_pivot_low(pivots: list[PivotPoint], min_idx: int) -> Optional[float]:
    """Lowest confirmed pivot low with idx >= min_idx."""
    lows = [p.price for p in pivots if p.ptype == "low" and p.idx >= min_idx]
    return min(lows) if lows else None


def type_b_check(
    closes_15m: np.ndarray, lows_15m: np.ndarray, highs_15m: np.ndarray,
    pivots_1h: list[PivotPoint], n_1h_bars: int,
    atr15_val: float, direction: Direction,
) -> Optional[dict]:
    """Check Type B conditions. Returns signal dict or None.

    Breakout detection uses highs (longs) / lows (shorts) — a wick beyond
    pivot is sufficient. Retest checks whether the *current bar* is touching
    the break level (price near break_level +/- tolerance and close back
    beyond it).
    """
    n = len(closes_15m)
    if n < C.BREAKOUT_LOOKBACK_15M:
        return None

    # Break level from 1H pivots
    min_1h_idx = max(0, n_1h_bars - C.BREAK_LOOKBACK_1H)
    if direction == Direction.LONG:
        break_level = _highest_pivot_high(pivots_1h, min_1h_idx)
    else:
        break_level = _lowest_pivot_low(pivots_1h, min_1h_idx)
    if break_level is None:
        return None

    # Breakout: 15m high/low beyond break_level within last BREAKOUT_LOOKBACK bars
    # (use highs for longs, lows for shorts — wick beyond pivot is sufficient)
    breakout_found = False
    start = max(0, n - 1 - C.BREAKOUT_LOOKBACK_15M)
    for i in range(n - 2, start - 1, -1):  # exclude current bar
        if direction == Direction.LONG and highs_15m[i] > break_level:
            breakout_found = True
            break
        if direction == Direction.SHORT and lows_15m[i] < break_level:
            breakout_found = True
            break
    if not breakout_found:
        return None

    # Retest: check if *current bar* is retesting the break level
    # (price comes back near break_level within tolerance, close resolves beyond it)
    tol = C.RETEST_TOL_ATR * atr15_val
    close = float(closes_15m[-1])
    if direction == Direction.LONG:
        # Low touches break level zone AND close is above break level
        if not (lows_15m[-1] <= break_level + tol and close > break_level):
            return None
    else:
        # High touches break level zone AND close is below break level
        if not (highs_15m[-1] >= break_level - tol and close < break_level):
            return None

    # Extension sanity
    if direction == Direction.LONG:
        if close > break_level + C.EXTENSION_SKIP_ATR * atr15_val:
            return None
    else:
        if close < break_level - C.EXTENSION_SKIP_ATR * atr15_val:
            return None

    return {"type": "B", "dir": direction, "break_level": break_level}


# ---------------------------------------------------------------------------
# Type C -- Continuation reclaim (v4.5 research)
# ---------------------------------------------------------------------------

def _latest_vwap_reference(
    svwap: np.ndarray, vwap_a: np.ndarray, close: float,
) -> tuple[Optional[float], str]:
    """Return the nearest current VWAP reference if one is available."""
    refs: list[tuple[float, str]] = []
    if len(svwap) > 0 and not np.isnan(svwap[-1]):
        refs.append((float(svwap[-1]), "session"))
    if len(vwap_a) > 0 and not np.isnan(vwap_a[-1]):
        refs.append((float(vwap_a[-1]), "anchor"))
    if not refs:
        return None, ""
    # The caller only has the completed close; choosing the closest current
    # reference avoids injecting future touch knowledge into the entry.
    return min(refs, key=lambda item: abs(close - item[0]))


def type_c_continuation_check(
    closes_15m: np.ndarray, lows_15m: np.ndarray, highs_15m: np.ndarray,
    svwap: np.ndarray, vwap_a: np.ndarray, atr15_val: float,
    direction: Direction, sub_window: SubWindow,
) -> Optional[dict]:
    """Completed-bar continuation signal for the shadow-positive no_signal bucket.

    The signal intentionally requires a current-bar close through the prior
    completed range and a strong close location. Orders are still staged by the
    backtest/live adapters after this completed bar, preserving fill causality.
    """
    n = len(closes_15m)
    if atr15_val <= 0 or n < C.TYPE_C_LOOKBACK_15M + 1:
        return None
    if sub_window.value not in C.TYPE_C_ALLOWED_WINDOWS:
        return None

    lookback = min(C.TYPE_C_LOOKBACK_15M, n - 1)
    prior_high = float(np.nanmax(highs_15m[-lookback - 1:-1]))
    prior_low = float(np.nanmin(lows_15m[-lookback - 1:-1]))
    close = float(closes_15m[-1])
    high = float(highs_15m[-1])
    low = float(lows_15m[-1])
    bar_range = high - low
    if bar_range <= 0 or bar_range > C.TYPE_C_MAX_BAR_ATR * atr15_val:
        return None

    vwap_ref, source = _latest_vwap_reference(svwap, vwap_a, close)
    if C.TYPE_C_REQUIRE_VWAP_SIDE and vwap_ref is None:
        return None
    if vwap_ref is not None and abs(close - vwap_ref) > C.TYPE_C_MAX_VWAP_DIST_ATR * atr15_val:
        return None

    buffer = C.TYPE_C_BREAK_BUFFER_ATR * atr15_val
    if direction == Direction.LONG:
        close_frac = (close - low) / bar_range
        if close <= prior_high + buffer or close_frac < C.TYPE_C_MIN_CLOSE_FRAC:
            return None
        if C.TYPE_C_REQUIRE_VWAP_SIDE and close <= float(vwap_ref):
            return None
        break_level = prior_high
    else:
        close_frac = (high - close) / bar_range
        if close >= prior_low - buffer or close_frac < C.TYPE_C_MIN_CLOSE_FRAC:
            return None
        if C.TYPE_C_REQUIRE_VWAP_SIDE and close >= float(vwap_ref):
            return None
        break_level = prior_low

    return {
        "type": "C",
        "dir": direction,
        "break_level": break_level,
        "vwap_used": vwap_ref or 0.0,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Predator overlay (Section 8)
# ---------------------------------------------------------------------------

def predator_present(
    pivots_1h: list[PivotPoint],
    bars_1h_high: np.ndarray, bars_1h_low: np.ndarray,
    mom15: np.ndarray, bars_15m_per_1h: int,
    direction: Direction,
) -> bool:
    """Check Predator divergence on last two confirmed 1H pivots."""
    if direction == Direction.LONG:
        lows = [p for p in pivots_1h if p.ptype == "low"]
        if len(lows) < 2:
            return False
        p1, p2 = lows[-2], lows[-1]
        # Higher low in structure
        if p2.price <= p1.price:
            return False
        # Bearish divergence in momentum (lower momentum at higher price)
        m1 = _mom15_at_1h_pivot(p1.idx, bars_15m_per_1h, mom15)
        m2 = _mom15_at_1h_pivot(p2.idx, bars_15m_per_1h, mom15)
        return m2 < m1
    else:
        highs = [p for p in pivots_1h if p.ptype == "high"]
        if len(highs) < 2:
            return False
        p1, p2 = highs[-2], highs[-1]
        if p2.price >= p1.price:
            return False
        m1 = _mom15_at_1h_pivot(p1.idx, bars_15m_per_1h, mom15)
        m2 = _mom15_at_1h_pivot(p2.idx, bars_15m_per_1h, mom15)
        return m2 > m1


def _mom15_at_1h_pivot(pivot_1h_idx: int, bars_15m_per_1h: int,
                       mom15: np.ndarray) -> float:
    """Sample Mom15 at the 15m bar closest to 1H pivot close."""
    idx_15m = pivot_1h_idx * bars_15m_per_1h + (bars_15m_per_1h - 1)
    idx_15m = min(idx_15m, len(mom15) - 1)
    if idx_15m < 0 or np.isnan(mom15[idx_15m]):
        return 0.0
    return float(mom15[idx_15m])
