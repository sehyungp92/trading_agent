"""AKC-Helix Swing v2.0 — enums and dataclasses."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CHOP = "CHOP"


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class SetupClass(str, Enum):
    CLASS_A = "A"     # 4H hidden divergence continuation
    CLASS_B = "B"     # 1H hidden divergence continuation (frequency)
    CLASS_C = "C"     # 4H classic divergence reversal (gated)
    CLASS_D = "D"     # 1H no-div momentum continuation (trend-only)


class SetupState(str, Enum):
    NEW = "NEW"
    ARMED = "ARMED"
    TRIGGERED = "TRIGGERED"
    FILLED = "FILLED"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"      # flatten/partial submitted, awaiting fill
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


class PivotKind(str, Enum):
    HIGH = "H"
    LOW = "L"


class LegType(str, Enum):
    UNIT1 = "UNIT1"
    ADD = "ADD"


# ---------------------------------------------------------------------------
# Pivot
# ---------------------------------------------------------------------------

@dataclass
class Pivot:
    ts: datetime
    kind: PivotKind
    price: float
    macd_line: float
    macd_hist: float
    atr_tf: float
    bar_index: int


# ---------------------------------------------------------------------------
# PivotStore — rolling bounded list per symbol per TF
# ---------------------------------------------------------------------------

@dataclass
class PivotStore:
    highs: list[Pivot] = field(default_factory=list)
    lows: list[Pivot] = field(default_factory=list)
    max_size: int = 50

    def add(self, pivot: Pivot) -> None:
        if pivot.kind == PivotKind.HIGH:
            self.highs.append(pivot)
            if len(self.highs) > self.max_size:
                self.highs = self.highs[-self.max_size:]
        else:
            self.lows.append(pivot)
            if len(self.lows) > self.max_size:
                self.lows = self.lows[-self.max_size:]

    def last_high(self) -> Optional[Pivot]:
        return self.highs[-1] if self.highs else None

    def last_low(self) -> Optional[Pivot]:
        return self.lows[-1] if self.lows else None

    def highs_between(self, ts_start: datetime, ts_end: datetime) -> list[Pivot]:
        return [p for p in self.highs if ts_start <= p.ts <= ts_end]

    def lows_between(self, ts_start: datetime, ts_end: datetime) -> list[Pivot]:
        return [p for p in self.lows if ts_start <= p.ts <= ts_end]


# ---------------------------------------------------------------------------
# DailyState
# ---------------------------------------------------------------------------

@dataclass
class DailyState:
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    atr_d: float = 0.0
    regime: Regime = Regime.CHOP
    trend_strength: float = 0.0
    trend_strength_3d_ago: float = 0.0
    vol_pct: float = 50.0
    atr_base: float = 0.0
    vol_factor: float = 1.0
    extreme_vol: bool = False
    close: float = 0.0
    last_bar_date: Optional[str] = None
    adx: float = 0.0


# ---------------------------------------------------------------------------
# TFState — per-symbol per-timeframe (1H / 4H)
# ---------------------------------------------------------------------------

@dataclass
class TFState:
    tf_label: str = "1H"              # "1H" or "4H"
    atr: float = 0.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    macd_line_history: list[float] = field(default_factory=list)
    macd_hist_history: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)   # for chandelier
    lows: list[float] = field(default_factory=list)
    close: float = 0.0
    bar_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# SetupInstance — the central tracking object for full lifecycle
# ---------------------------------------------------------------------------

@dataclass
class SetupInstance:
    # Identity
    setup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    setup_class: SetupClass = SetupClass.CLASS_A
    direction: Direction = Direction.FLAT
    origin_tf: str = "4H"
    state: SetupState = SetupState.NEW
    created_ts: Optional[datetime] = None
    armed_ts: Optional[datetime] = None

    # Structure
    pivot_1: Optional[Pivot] = None
    pivot_2: Optional[Pivot] = None
    bos_pivot: Optional[Pivot] = None
    bos_level: float = 0.0
    stop0: float = 0.0
    buffer: float = 0.0
    adx_at_entry: float = 0.0
    div_mag_norm: float = 0.0
    regime_4h_at_entry: Optional[str] = None

    # Risk
    unit1_risk_dollars: float = 0.0
    base_unit1_risk_dollars: float = 0.0
    target_initial_risk_dollars: float = 0.0
    actual_initial_risk_dollars: float = 0.0
    risk_utilization: float = 0.0
    setup_size_mult: float = 1.0
    vol_factor_at_placement: float = 1.0
    offset_ticks_at_placement: int = 0
    qty_planned: int = 0

    # OCA
    oca_group: str = ""
    primary_order_id: str = ""
    catchup_order_id: str = ""
    rescue_order_id: str = ""
    stop_order_id: str = ""

    # TTL
    expiry_ts: Optional[datetime] = None
    catchup_expiry_ts: Optional[datetime] = None
    rescue_expiry_ts: Optional[datetime] = None
    triggered_ts: Optional[datetime] = None

    # Spread
    spread_fail_count: int = 0

    # Fill
    fill_price: float = 0.0
    avg_entry_price: float = 0.0
    fill_qty: int = 0
    fill_ts: Optional[datetime] = None
    r_price: float = 0.0             # dollar risk per unit for R calc

    # Management
    trail_active: bool = False
    partial_2p5_done: bool = False
    partial_5_done: bool = False
    trailing_mult_bonus: float = 0.0
    add_allowed: bool = True
    add_done: bool = False
    teleport_fill: bool = False        # slippage exceeded limits (spec s11.2)
    add_min_r_override: float = 0.0    # teleport penalty: add delayed to +2R

    # Regime at entry (for tracking deterioration transitions)
    regime_at_entry: Optional[str] = None

    # Stale
    bars_held_1h: int = 0
    bars_held_4h: int = 0

    # Trailing profit delay (change #9): bars position has been at +1R
    bars_at_r1: int = 0
    # Momentum tracking: bars with negative AND declining histogram
    bars_neg_fading_hist: int = 0
    # Stalled winner tracking: peak MFE in R and bar count when achieved
    mfe_r_peak: float = 0.0
    mae_r_trough: float = 0.0
    bar_of_max_mfe: int = 0

    # R
    realized_pnl: float = 0.0
    qty_open: int = 0
    current_stop: float = 0.0
    stop_source: str = "INITIAL"  # tracks last stop update source for exit reason

    # Recording
    trade_id: str = ""

    # Gate decision telemetry (populated at arm time)
    gate_decisions: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# CircuitBreakerState
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreakerState:
    weekly_realized_r: float = 0.0
    daily_realized_r: float = 0.0
    daily_bucket: Optional[str] = None
    weekly_bucket: Optional[str] = None
    consecutive_stops: int = 0
    halved_until: Optional[datetime] = None
    paused_until: Optional[datetime] = None
