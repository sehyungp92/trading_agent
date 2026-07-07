"""ATRSS v4.5 state dataclasses and enums."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    RANGE = "RANGE"
    TREND = "TREND"
    STRONG_TREND = "STRONG_TREND"


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class CandidateType(str, Enum):
    PULLBACK = "PULLBACK"
    BREAKOUT = "BREAKOUT"
    REVERSE = "REVERSE"
    ADDON_A = "ADDON_A"
    ADDON_B = "ADDON_B"


class LegType(str, Enum):
    BASE = "BASE"
    ADDON_A = "ADDON_A"
    ADDON_B = "ADDON_B"


# ---------------------------------------------------------------------------
# Daily state — regime, bias, score computed once per daily bar
# ---------------------------------------------------------------------------

@dataclass
class DailyState:
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    atr20: float = 0.0
    regime: Regime = Regime.RANGE
    trend_dir: Direction = Direction.FLAT
    score: float = 0.0
    ema_sep_pct: float = 0.0
    di_diff: float = 0.0
    adx_slope_3: float = 0.0
    raw_bias: Direction = Direction.FLAT
    raw_bias_prev: Direction = Direction.FLAT
    hold_count: int = 0              # consecutive days of same raw bias
    regime_on: bool = False          # hysteresis flag
    ema_fast_slope_5: float = 0.0   # EMA_fast[-1] - EMA_fast[-6] (spec S2.1)
    hh_20d: float = 0.0             # Highest high over 20 daily bars
    ll_20d: float = 0.0             # Lowest low over 20 daily bars
    last_daily_bar_date: Optional[str] = None  # ISO date of most recent daily bar


# ---------------------------------------------------------------------------
# Hourly state — indicators per hourly bar
# ---------------------------------------------------------------------------

@dataclass
class HourlyState:
    time: Optional[datetime] = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    ema_mom: float = 0.0
    ema_pull: float = 0.0
    atrh: float = 0.0
    donchian_high: float = 0.0
    donchian_low: float = 0.0
    prior_high: float = 0.0
    prior_low: float = 0.0
    dist_atr: float = 0.0           # distance-to-EMA in ATR units
    recent_pull_touch_long: bool = False   # low <= ema_pull within lookback
    recent_pull_touch_short: bool = False  # high >= ema_pull within lookback


# ---------------------------------------------------------------------------
# Entry candidate
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    symbol: str = ""
    type: CandidateType = CandidateType.PULLBACK
    direction: Direction = Direction.FLAT
    trigger_price: float = 0.0
    initial_stop: float = 0.0
    qty: int = 0
    signal_bar: Optional[HourlyState] = None
    time: Optional[datetime] = None
    rank_score: float = 0.0
    atrh: float = 0.0              # hourly ATR at signal time
    tick_size: float = 0.0         # contract tick size


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass
class PositionLeg:
    leg_type: LegType = LegType.BASE
    qty: int = 0
    entry_price: float = 0.0
    initial_stop: float = 0.0
    fill_time: Optional[datetime] = None
    entry_commission: float = 0.0
    oms_order_id: str = ""
    trade_id: str = ""


@dataclass
class PositionBook:
    symbol: str = ""
    direction: Direction = Direction.FLAT
    legs: list[PositionLeg] = field(default_factory=list)
    current_stop: float = 0.0
    mfe: float = 0.0                # in R units
    mfe_price: float = 0.0
    mae: float = 0.0                # in R units (adverse excursion)
    mae_price: float = 0.0
    entry_time: Optional[datetime] = None
    be_triggered: bool = False
    addon_a_done: bool = False
    addon_a_pending_id: str = ""  # C2: tracks pending addon A order to prevent re-trigger
    addon_b_done: bool = False
    tp1_done: bool = False
    tp2_done: bool = False
    stop_oms_order_id: str = ""
    stop_pending: bool = False  # C3: True while protective stop is being placed
    bars_held: int = 0
    early_partial_done: bool = False

    @property
    def total_qty(self) -> int:
        return sum(leg.qty for leg in self.legs)

    @property
    def base_leg(self) -> Optional[PositionLeg]:
        for leg in self.legs:
            if leg.leg_type == LegType.BASE:
                return leg
        return None

    @property
    def avg_entry(self) -> float:
        total_qty = self.total_qty
        if total_qty == 0:
            return 0.0
        return sum(leg.entry_price * leg.qty for leg in self.legs) / total_qty

    @property
    def base_risk_per_unit(self) -> float:
        """Risk per contract based on base leg entry vs initial stop."""
        base = self.base_leg
        if base is None:
            return 0.0
        return abs(base.entry_price - base.initial_stop)


# ---------------------------------------------------------------------------
# Re-entry cooldown / reset tracking
# ---------------------------------------------------------------------------

@dataclass
class ReentryState:
    last_exit_time: Optional[datetime] = None
    last_exit_dir: Direction = Direction.FLAT
    reset_seen_long: bool = True
    reset_seen_short: bool = True
    # Voucher system (spec Section 4)
    voucher_long: bool = False
    voucher_short: bool = False
    voucher_granted_time: Optional[datetime] = None
    # Quality-based re-entry gate
    last_exit_mfe: float = 0.0
    last_exit_reason: str = ""


# ---------------------------------------------------------------------------
# Breakout arm tracking (spec Section 7.2)
# ---------------------------------------------------------------------------

@dataclass
class BreakoutArmState:
    breakout_armed_dir: Direction = Direction.FLAT
    breakout_armed_until: Optional[datetime] = None
    breakout_arm_high: float = 0.0
    breakout_arm_low: float = 0.0


# ---------------------------------------------------------------------------
# Halt / limit state tracking (spec Section 12)
# ---------------------------------------------------------------------------

@dataclass
class HaltState:
    is_halted: bool = False
    halt_detected_at: Optional[datetime] = None
    queued_stop_updates: list[tuple[str, float]] = field(default_factory=list)
    pre_halt_order_ids: list[str] = field(default_factory=list)
    unprotected: bool = False        # True when stop order rejected during halt (spec S10.2)
