"""Downturn Dominator models -- enums, state dataclasses, trade/signal/result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, Enum
from typing import Any

import numpy as np


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EngineTag(str, Enum):
    REVERSAL = "reversal"
    BREAKDOWN = "breakdown"
    FADE = "fade"


class CompositeRegime(str, Enum):
    ALIGNED_BEAR = "aligned_bear"
    EMERGING_BEAR = "emerging_bear"
    NEUTRAL = "neutral"
    COUNTER = "counter"
    RANGE = "range"


class VolState(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    SHOCK = "shock"


class Regime4H(str, Enum):
    TRENDING = "trending"
    RANGE = "range"
    TRANSITIONAL = "transitional"


# ---------------------------------------------------------------------------
# Engine state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReversalState:
    """4H pivot pair tracking for reversal divergence detection."""
    h1_price: float = 0.0
    h1_idx: int = -1
    h2_price: float = 0.0
    h2_idx: int = -1
    l_between: float = 0.0
    macd_at_h1: float = 0.0
    macd_at_h2: float = 0.0
    divergence_arm_active: bool = False
    disabled: bool = False  # set when strong_bear or Shock


@dataclass
class BreakdownBoxState:
    """Frozen box bounds for 30m breakdown detection."""
    range_high: float = 0.0
    range_low: float = 0.0
    age: int = 0
    containment_ratio: float = 0.0
    violations: int = 0
    active: bool = False
    vwap_box: float = 0.0
    displacement_history: list[float] = field(default_factory=list)
    expiry_countdown: int = 0
    adaptive_L: int = 32


@dataclass
class FadeState:
    """VWAP session tracking for fade rejection detection."""
    vwap_session: float = 0.0
    vwap_anchored: float = 0.0
    vwap_used: float = 0.0
    touch_bars: list[bool] = field(default_factory=list)  # last 8 bars of 15m
    consecutive_above_vwap: int = 0


@dataclass
class DownturnRegimeCtx:
    """Composite regime + volatility state context."""
    regime_4h: Regime4H = Regime4H.RANGE
    daily_trend: int = 0  # +1/-1 with 2-bar persistence
    daily_trend_consec: int = 0
    composite_regime: CompositeRegime = CompositeRegime.NEUTRAL
    vol_state: VolState = VolState.NORMAL
    vol_factor: float = 1.0
    trend_strength: float = 0.0
    strong_bear: bool = False
    short_trend: int = 0  # -1 if close < short SMA (e.g. SMA50)
    extension_short: bool = False
    extension_long: bool = False


# ---------------------------------------------------------------------------
# Correction window
# ---------------------------------------------------------------------------

@dataclass
class CorrectionWindow:
    """A period where NQ dropped >3% from 20-day rolling high."""
    start_date: datetime
    end_date: datetime
    peak_to_trough_pct: float


# ---------------------------------------------------------------------------
# Signal events
# ---------------------------------------------------------------------------

@dataclass
class ReversalSignal:
    """Reversal short signal details."""
    h1_price: float
    h2_price: float
    divergence_mag: float
    class_mult: float  # 0.65 (predator) vs 0.40 (standard)
    predator_present: bool = False


@dataclass
class BreakdownSignal:
    """Breakdown short signal details."""
    box_high: float
    box_low: float
    displacement_metric: float
    box_age: int


@dataclass
class FadeSignal:
    """Fade VWAP rejection signal details."""
    vwap_used: float
    rejection_close: float
    class_mult: float  # 1.0 (predator) vs 0.70 (standard)
    predator_present: bool = False


# ---------------------------------------------------------------------------
# Trade + signal event records
# ---------------------------------------------------------------------------

@dataclass
class DownturnSignalEvent:
    """Record of a detected signal (entered or not)."""
    engine_tag: EngineTag
    direction: Direction
    signal_class: str  # e.g. "classic_divergence", "vwap_rejection", "box_breakdown"
    regime_at_signal: CompositeRegime
    gates_passed: list[str] = field(default_factory=list)
    gates_blocked: list[str] = field(default_factory=list)
    timestamp: datetime | None = None
    entered: bool = False


@dataclass
class DownturnTradeRecord:
    """Complete trade record for analysis."""
    # Core (same fields as NQDTCTradeRecord)
    symbol: str = ""
    direction: Direction = Direction.SHORT
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    qty: int = 0
    pnl: float = 0.0
    r_multiple: float = 0.0
    stop0: float = 0.0
    commission: float = 0.0
    entry_type: str = ""  # "stop_market", "stop_limit"
    exit_type: str = ""   # "tp1", "tp2", "tp3", "chandelier", "stale", "climax", "catastrophic", "stop"
    hold_bars: int = 0
    hold_bars_5m: int = 0  # 5-minute bar count for hold duration
    mfe: float = 0.0      # max favorable excursion in R
    mae: float = 0.0      # max adverse excursion in R

    # Downturn-specific
    engine_tag: EngineTag = EngineTag.BREAKDOWN
    composite_regime_at_entry: CompositeRegime = CompositeRegime.NEUTRAL
    vol_state_at_entry: VolState = VolState.NORMAL
    in_correction_window: bool = False
    predator_present: bool = False
    signal_class: str = ""


# ---------------------------------------------------------------------------
# Per-engine counters
# ---------------------------------------------------------------------------

@dataclass
class EngineCounters:
    """Signal/entry/fill counters per engine."""
    signals_detected: int = 0
    entries_placed: int = 0
    entries_filled: int = 0
    gates_blocked: int = 0


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------

@dataclass
class DownturnResult:
    """Complete backtest result."""
    symbol: str
    trades: list[DownturnTradeRecord] = field(default_factory=list)
    signal_events: list[DownturnSignalEvent] = field(default_factory=list)
    decision_stream: list[dict[str, Any]] = field(default_factory=list)
    trade_outcomes: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    total_commission: float = 0.0
    correction_windows: list[CorrectionWindow] = field(default_factory=list)

    # Per-engine counters
    reversal_counters: EngineCounters = field(default_factory=EngineCounters)
    breakdown_counters: EngineCounters = field(default_factory=EngineCounters)
    fade_counters: EngineCounters = field(default_factory=EngineCounters)
