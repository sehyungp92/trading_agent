"""Vdubus NQ v4.0 — state dataclasses and enums."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class VolState(str, Enum):
    NORMAL = "Normal"
    HIGH = "High"
    SHOCK = "Shock"


class SessionWindow(str, Enum):
    RTH = "RTH"
    EVENING = "EVENING"
    BLOCKED = "BLOCKED"


class SubWindow(str, Enum):
    OPEN = "OPEN"
    CORE = "CORE"
    CLOSE = "CLOSE"
    EVENING = "EVENING"


class EntryType(str, Enum):
    TYPE_A = "A"
    TYPE_B = "B"
    TYPE_C = "C"


class PositionStage(str, Enum):
    ACTIVE_RISK = "ACTIVE_RISK"
    ACTIVE_FREE = "ACTIVE_FREE"
    SWING_HOLD = "SWING_HOLD"


@dataclass
class PivotPoint:
    idx: int
    price: float
    ptype: str              # "high" or "low"
    confirmed_at: int


@dataclass
class RegimeState:
    daily_trend: int = 0
    daily_trend_prev: int = 0
    vol_state: VolState = VolState.NORMAL
    trend_1h: int = 0
    # persistence tracking
    daily_raw_streak: int = 0
    hourly_raw_streak: int = 0
    last_daily_raw: int = 0
    last_hourly_raw: int = 0
    flip_just_happened: bool = False
    choppiness: float = 50.0


@dataclass
class DayCounters:
    long_fills: int = 0
    short_fills: int = 0
    daily_realized_pnl: float = 0.0
    breaker_hit: bool = False
    flip_entry_used_long: bool = False
    flip_entry_used_short: bool = False
    trade_date: Optional[str] = None
    addon_used_long: bool = False
    addon_used_short: bool = False

    def reset(self) -> None:
        self.long_fills = 0
        self.short_fills = 0
        self.daily_realized_pnl = 0.0
        self.breaker_hit = False
        self.flip_entry_used_long = False
        self.flip_entry_used_short = False
        self.addon_used_long = False
        self.addon_used_short = False


@dataclass
class PositionState:
    trade_id: str = ""
    direction: Direction = Direction.FLAT
    entry_price: float = 0.0
    stop_price: float = 0.0
    qty_entry: int = 0
    qty_open: int = 0
    r_points: float = 0.0
    stage: PositionStage = PositionStage.ACTIVE_RISK
    partial_done: bool = False
    entry_time: Optional[datetime] = None
    bars_since_entry: int = 0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    vwap_used_at_entry: float = 0.0
    is_addon: bool = False
    stop_oms_order_id: str = ""
    entry_type: EntryType = EntryType.TYPE_A
    entry_session: SessionWindow = SessionWindow.RTH
    is_flip_entry: bool = False
    class_mult: float = 1.0
    vwap_fail_count: int = 0
    session_count: int = 1  # how many RTH sessions held
    bars_since_partial: int = 0     # bars elapsed since +1R partial
    peak_r_since_free: float = 0.0  # highest unrealized R since ACTIVE_FREE
    peak_mfe_r: float = 0.0        # max favorable R excursion (for early kill)
    peak_mae_r: float = 0.0        # worst adverse excursion in R-multiples (always >= 0)
    early_warning_bar: int = -1    # bar when early kill warning first triggered (-1 = none)
    late_trail_active: bool = False    # True once peak_mfe_r crossed LATE_TRAIL_ACTIVATE_R
    late_trail_be_done: bool = False   # True once stop moved to BE via late trail
    session_transitions_log: list = field(default_factory=list)


@dataclass
class WorkingEntry:
    oms_order_id: str = ""
    entry_type: EntryType = EntryType.TYPE_A
    direction: Direction = Direction.FLAT
    stop_entry: float = 0.0
    limit_entry: float = 0.0
    qty: int = 0
    submitted_bar_idx: int = 0
    ttl_bars: int = 3
    initial_stop: float = 0.0
    fallback_allowed: bool = True
    triggered: bool = False
    triggered_bar_idx: int = -1
    vwap_used: float = 0.0
    class_mult: float = 1.0
    session: SessionWindow = SessionWindow.RTH
    is_flip: bool = False
    is_addon: bool = False
    filter_decisions: list[dict] | None = None
    signal_id: str = ""
    bar_id: str = ""
    exchange_timestamp: Optional[datetime] = None


@dataclass
class EventBlockState:
    blocked: bool = False
    block_end_ts: Optional[datetime] = None
    cooldown_remaining: int = 0
    pre_event_atr15: float = 0.0
    event_type: str = ""
    rearmed: bool = True
    post_event_bars: int = 0
    max_post_bars: int = 0
