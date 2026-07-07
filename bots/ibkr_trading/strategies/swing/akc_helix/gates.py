"""AKC-Helix Swing — entry gates and eligibility checks.

Pure gate functions. Each returns bool or (bool, str).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

from .config import (
    BASKET_4H_SECOND_MULT,
    BASKET_SYMBOLS,
    DISABLE_CIRCUIT_BREAKER,
    EXTREME_VOL_CAP_R,
    EXTREME_VOL_PCT,
    INSTRUMENT_CAP_R,
    NEWS_WINDOWS,
    PORTFOLIO_CAP_R,
    SymbolConfig,
)
from .models import (
    CircuitBreakerState,
    DailyState,
    Direction,
    Regime,
    SetupClass,
    SetupInstance,
    SetupState,
)


# ---------------------------------------------------------------------------
# Entry window (spec s1.1)
# ---------------------------------------------------------------------------

def is_entry_window_open(now_et: datetime, cfg: SymbolConfig) -> bool:
    """Check if current ET time is within the symbol's entry window."""
    start = _parse_time(cfg.entry_window_start_et)
    end = _parse_time(cfg.entry_window_end_et)
    current = now_et.time()
    if start <= end:
        return start <= current <= end
    # Wraps midnight (e.g., futures 03:00-16:30 doesn't wrap, but guard anyway)
    return current >= start or current <= end


def _parse_time(s: str) -> time:
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


# ---------------------------------------------------------------------------
# News guard (spec s4)
# ---------------------------------------------------------------------------

def is_news_blocked(
    now_et: datetime,
    symbol: str,
    calendar: list[tuple[str, datetime]],
) -> bool:
    """Check if any news event blocks entry for this symbol.

    Calendar = list of (event_type, event_datetime_et).
    Also enforces conservative CPI/NFP day gate (spec s4):
    on CPI/NFP days, no new entries until 10:00 ET.
    """
    for event_type, event_dt in calendar:
        windows = NEWS_WINDOWS.get(event_type)
        if windows is None:
            continue
        # CL_INVENTORY only blocks CL/MCL/USO
        if event_type == "CL_INVENTORY" and symbol not in ("CL", "MCL", "USO"):
            continue
        # CRYPTO_EVENT only blocks crypto symbols
        if event_type == "CRYPTO_EVENT" and symbol not in ("BT", "MBT"):
            continue
        before_min, after_min = windows
        window_start = event_dt + timedelta(minutes=before_min)
        window_end = event_dt + timedelta(minutes=after_min)
        if window_start <= now_et <= window_end:
            return True
        # Conservative CPI/NFP day gate (spec s4): no new entries until 10:00 ET
        if event_type in ("CPI", "NFP") and event_dt.date() == now_et.date():
            if now_et.hour < 10:
                return True
    return False


# ---------------------------------------------------------------------------
# Spread gate (spec s7.1)
# ---------------------------------------------------------------------------

def spread_gate_ok(
    spread_ticks: float,
    max_ticks: int,
    setup: SetupInstance,
    spread_dollars: float = 0.0,
    max_spread_dollars: float = 0.0,
    spread_bps: float = 0.0,
    max_spread_bps: float = 0.0,
    is_etf: bool = False,
) -> bool:
    """Spread gate per spec s7.1.

    ETFs: BOTH spread <= SpreadMax_$ AND spread_bps <= SpreadMax_bps must pass.
    Futures: spread_ticks <= max_ticks.
    Re-check grace: 2 consecutive 1H bars (spec s7.1).
    """
    if is_etf:
        ok = (spread_dollars <= max_spread_dollars and
              spread_bps <= max_spread_bps)
    else:
        ok = spread_ticks <= max_ticks

    if ok:
        setup.spread_fail_count = 0
        return True
    setup.spread_fail_count += 1
    # Allow recheck for up to 2 consecutive 1H bars (spec s7.1)
    return setup.spread_fail_count <= 2


# ---------------------------------------------------------------------------
# Min stop gate (spec s7.2)
# ---------------------------------------------------------------------------

def min_stop_gate_ok(
    entry: float,
    stop0: float,
    point_value: float,
    atr_1h: float,
    floor: float,
) -> bool:
    """Stop distance in dollars must meet max(floor, 0.30*ATR1H*pv) -- spec s7.2."""
    stop_dist_dollars = abs(entry - stop0) * point_value
    min_stop = max(floor, 0.30 * atr_1h * point_value)
    return stop_dist_dollars >= min_stop


# ---------------------------------------------------------------------------
# Corridor cap (spec s10.2 adaptive)
# ---------------------------------------------------------------------------

def corridor_cap_ok(
    entry: float,
    stop0: float,
    direction: Direction,
    daily: DailyState,
    cfg: SymbolConfig,
) -> bool:
    """Check entry-to-stop distance against corridor cap (spec s10.2).

    Uses trend-aligned logic: LONG+BULL or SHORT+BEAR gets wider cap.
    """
    entry_to_stop = abs(entry - stop0)
    if daily.regime == Regime.CHOP:
        cap = cfg.corridor_cap_chop * daily.atr_d
    else:
        # Trend-aligned gets wider corridor
        aligned = (
            (direction == Direction.LONG and daily.regime == Regime.BULL)
            or (direction == Direction.SHORT and daily.regime == Regime.BEAR)
        )
        if aligned:
            cap = cfg.corridor_cap_trend * daily.atr_d
        else:
            cap = cfg.corridor_cap_other * daily.atr_d

    if cap <= 0:
        return True
    return entry_to_stop <= cap


def corridor_inverted(
    min_stop_floor: float,
    corridor_cap: float,
    point_value: float,
) -> bool:
    """Disable 4H setups if min_stop_floor > corridor_cap (in dollar terms)."""
    if point_value <= 0:
        return False
    cap_dollars = corridor_cap * point_value
    return min_stop_floor > cap_dollars


# ---------------------------------------------------------------------------
# Heat caps (spec s8.3)
# ---------------------------------------------------------------------------

def heat_cap_ok(
    portfolio_r: float,
    pending_r: float,
    candidate_r: float,
    instrument_r: float,
    extreme_vol: bool,
) -> bool:
    """Check portfolio and instrument heat caps.

    portfolio_r + pending_r + candidate_r <= PORTFOLIO_CAP_R
    instrument_r + candidate_r <= INSTRUMENT_CAP_R
    If extreme_vol: portfolio cap reduced to EXTREME_VOL_CAP_R
    """
    port_cap = EXTREME_VOL_CAP_R if extreme_vol else PORTFOLIO_CAP_R
    if portfolio_r + pending_r + candidate_r > port_cap:
        return False
    if instrument_r + candidate_r > INSTRUMENT_CAP_R:
        return False
    return True


def apply_basket_adjustment(
    symbol: str,
    setup_class: SetupClass,
    mult: float,
    portfolio_active: dict[str, SetupInstance],
) -> float:
    """Apply risk-on basket rule (spec s8.3).

    Basket: QQQ and its futures/crypto peers NQ/MNQ + BT/MBT.
    1H-class: if any basket peer already has a 1H setup armed/active,
        block (return 0.0) -- only one 1H-class at a time.
    4H-class: if any basket peer already has a 4H setup,
        second gets 0.60x multiplier.
    """
    if symbol not in BASKET_SYMBOLS:
        return mult

    # All basket symbols are peers of each other
    peers = BASKET_SYMBOLS - {symbol}

    is_1h_class = setup_class in (SetupClass.CLASS_B, SetupClass.CLASS_D)
    is_4h_class = setup_class in (SetupClass.CLASS_A, SetupClass.CLASS_C)

    for sid, setup in portfolio_active.items():
        if setup.symbol not in peers:
            continue
        if setup.state not in (SetupState.ARMED, SetupState.TRIGGERED, SetupState.FILLED, SetupState.ACTIVE):
            continue

        # 1H mutual exclusion: if peer has 1H setup, block this one
        if is_1h_class and setup.origin_tf == "1H":
            return 0.0

        # 4H reduction: if peer has 4H setup, reduce this one
        if is_4h_class and setup.origin_tf == "4H":
            return mult * BASKET_4H_SECOND_MULT

    return mult


# ---------------------------------------------------------------------------
# Extreme vol gate (spec s6)
# ---------------------------------------------------------------------------

def extreme_vol_gate(setup_class: SetupClass, vol_pct: float) -> bool:
    """Disable 1H-origin setups (Class B and D) if vol_pct > 95th percentile (spec s6)."""
    if vol_pct > EXTREME_VOL_PCT:
        if setup_class in (SetupClass.CLASS_B, SetupClass.CLASS_D):
            return False
    return True


# ---------------------------------------------------------------------------
# Circuit breaker (spec s8.4)
# ---------------------------------------------------------------------------

def circuit_breaker_ok(cb: CircuitBreakerState, now: datetime) -> bool:
    """Check if circuit breaker allows new entries."""
    if DISABLE_CIRCUIT_BREAKER:
        return True
    if cb.paused_until and now < cb.paused_until:
        return False
    return True


# ---------------------------------------------------------------------------
# Full eligibility check (spec s11.1)
# ---------------------------------------------------------------------------

def full_eligibility_check(
    setup: SetupInstance,
    now_et: datetime,
    daily: DailyState,
    cfg: SymbolConfig,
    spread_ticks: float,
    portfolio_r: float,
    pending_r: float,
    instrument_r: float,
    cb: CircuitBreakerState,
    calendar: list[tuple[str, datetime]],
    portfolio_active: dict[str, SetupInstance],
    point_value: float,
    atr_1h: float,
    spread_dollars: float = 0.0,
    spread_bps: float = 0.0,
) -> tuple[bool, str]:
    """Run all gates in order per spec s11.1. Returns (ok, reason)."""
    # 1. Entry window
    if not is_entry_window_open(now_et, cfg):
        return False, "outside_entry_window"

    # 2. News guard
    if is_news_blocked(now_et, setup.symbol, calendar):
        return False, "news_blocked"

    # 3. Circuit breaker
    if not circuit_breaker_ok(cb, now_et):
        return False, "circuit_breaker"

    # 4. Extreme vol gate (1H setups)
    if not extreme_vol_gate(setup.setup_class, daily.vol_pct):
        return False, "extreme_vol"

    # 5. Spread gate (spec s7.1: dual $ + bps for ETFs)
    if not spread_gate_ok(
        spread_ticks, cfg.max_spread_ticks, setup,
        spread_dollars=spread_dollars,
        max_spread_dollars=cfg.max_spread_dollars,
        spread_bps=spread_bps,
        max_spread_bps=cfg.max_spread_bps,
        is_etf=cfg.is_etf,
    ):
        return False, "spread_too_wide"

    # 9. Min stop gate
    if not min_stop_gate_ok(
        setup.bos_level, setup.stop0, point_value, atr_1h,
        cfg.min_stop_floor_dollars,
    ):
        return False, "min_stop_floor"

    # 10. Corridor cap
    if not corridor_cap_ok(
        setup.bos_level, setup.stop0, setup.direction, daily, cfg,
    ):
        return False, "corridor_cap"

    # 11. Compute candidate R
    entry_to_stop = abs(setup.bos_level - setup.stop0)
    if entry_to_stop > 0 and setup.unit1_risk_dollars > 0:
        candidate_r = (entry_to_stop * point_value * setup.qty_planned) / setup.unit1_risk_dollars
    else:
        candidate_r = 0.0

    # 12. Heat cap
    if not heat_cap_ok(
        portfolio_r, pending_r, candidate_r, instrument_r, daily.extreme_vol,
    ):
        return False, "heat_cap"

    # 13. Basket adjustment (modifies size mult in place)
    setup.setup_size_mult = apply_basket_adjustment(
        setup.symbol, setup.setup_class, setup.setup_size_mult, portfolio_active,
    )
    if setup.setup_size_mult <= 0:
        return False, "basket_disabled"

    return True, ""
