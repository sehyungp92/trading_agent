"""NQ Dominant Trend Capture v2.0 — state dataclasses and enums."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Session(str, Enum):
    ETH = "ETH"
    RTH = "RTH"


class BoxState(str, Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    DIRTY = "DIRTY"


class BreakoutState(str, Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    CONTINUATION = "CONTINUATION"
    INVALID = "INVALID"


class Regime4H(str, Enum):
    TRENDING = "TRENDING"
    TRANSITIONAL = "TRANSITIONAL"
    RANGE = "RANGE"


class CompositeRegime(str, Enum):
    ALIGNED = "Aligned"
    NEUTRAL = "Neutral"
    CAUTION = "Caution"
    RANGE = "Range"
    COUNTER = "Counter"


class ChopMode(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    HALT = "HALT"


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class EntrySubtype(str, Enum):
    A_RETEST = "A_retest"
    A_LATCH = "A_latch"
    B_SWEEP = "B_sweep"
    C_STANDARD = "C_standard"
    C_CONTINUATION = "C_continuation"
    MARKET_FALLBACK = "MARKET_fallback"


class ExitTier(str, Enum):
    ALIGNED = "Aligned"
    NEUTRAL = "Neutral"
    CAUTION = "Caution"


# ---------------------------------------------------------------------------
# Box engine state (per session: ETH/RTH)
# ---------------------------------------------------------------------------

@dataclass
class BoxEngineState:
    # Adaptive L
    L: int = 18         # session-scaled MID default (was 26, optimized via sweep)
    L_used: int = 18
    last_bucket: str = "MID"
    bucket_streak: int = 0

    # Box bounds (frozen on activation)
    state: BoxState = BoxState.INACTIVE
    box_high: float = 0.0
    box_low: float = 0.0
    box_mid: float = 0.0
    box_width: float = 0.0
    box_anchor_ts: Optional[datetime] = None
    box_bars_active: int = 0

    # DIRTY
    dirty_start_idx: int = -1
    dirty_high: float = 0.0
    dirty_low: float = 0.0
    dirty_direction: str = ""
    dirty_wick_extreme: float = 0.0


# ---------------------------------------------------------------------------
# Breakout state
# ---------------------------------------------------------------------------

@dataclass
class BreakoutEngineState:
    active: bool = False
    direction: Direction = Direction.FLAT
    breakout_bar_ts: Optional[datetime] = None
    bars_since_breakout: int = 0
    expiry_bars: int = 10
    hard_expiry_bars: int = 18
    continuation_mode: bool = False
    consec_inside_count: int = 0
    mm_reached: bool = False
    mm_level: float = 0.0
    continuation_fills: int = 0  # count of continuation entries filled for this breakout
    last_trade_peak_r: float = 0.0  # peak MFE R of most recent closed trade in this breakout
    # Frozen breakout bar high/low for A2 entry placement
    breakout_bar_high: float = 0.0
    breakout_bar_low: float = 0.0


# ---------------------------------------------------------------------------
# VWAP accumulators
# ---------------------------------------------------------------------------

@dataclass
class VWAPAccumulator:
    cum_tpv: float = 0.0
    cum_vol: float = 0.0
    anchor_ts: Optional[datetime] = None

    @property
    def value(self) -> float:
        if self.cum_vol <= 0:
            return 0.0
        return self.cum_tpv / self.cum_vol

    def update(self, high: float, low: float, close: float, volume: float) -> float:
        tp = (high + low + close) / 3.0
        v = volume if volume > 0 else 1.0
        self.cum_tpv += tp * v
        self.cum_vol += v
        return self.value

    def reset(self, ts: Optional[datetime] = None) -> None:
        self.cum_tpv = 0.0
        self.cum_vol = 0.0
        self.anchor_ts = ts


# ---------------------------------------------------------------------------
# Displacement / squeeze rolling buffers
# ---------------------------------------------------------------------------

@dataclass
class RollingBuffer:
    maxlen: int = 2880  # ~60 calendar days of 30m bars (48/day)
    data: list[float] = field(default_factory=list)

    def append(self, val: float) -> None:
        self.data.append(val)
        if len(self.data) > self.maxlen:
            self.data = self.data[-self.maxlen:]


# ---------------------------------------------------------------------------
# Session engine state (per ETH/RTH)
# ---------------------------------------------------------------------------

@dataclass
class SessionEngineState:
    session: Session = Session.RTH
    vwap_session: VWAPAccumulator = field(default_factory=VWAPAccumulator)
    vwap_box: VWAPAccumulator = field(default_factory=VWAPAccumulator)

    box: BoxEngineState = field(default_factory=BoxEngineState)
    breakout: BreakoutEngineState = field(default_factory=BreakoutEngineState)

    disp_hist: RollingBuffer = field(default_factory=RollingBuffer)
    squeeze_hist: RollingBuffer = field(default_factory=RollingBuffer)

    chop_score: int = 0
    mode: ChopMode = ChopMode.NORMAL

    # ATR cache (30m)
    atr14_30m: float = 0.0
    atr50_30m: float = 0.0

    # Score + qualification cache
    last_score: float = 0.0
    last_disp_metric: float = 0.0
    last_disp_threshold: float = 0.0
    last_rvol: float = 0.0

    # Re-entry tracking (Section 18.1)
    reentry_allowed: bool = True
    reentry_used: bool = False
    last_stopout_r: float = 0.0
    last_stopout_ts: Optional[datetime] = None

    # 30m bar tracking for proper detection (fix #14)
    last_30m_bar_count: int = 0

    # Trend cycling (Section 18.2)
    last_profitable_exit_dir: Direction = Direction.FLAT


# ---------------------------------------------------------------------------
# Regime state (global, 4H + Daily)
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    regime_4h: Regime4H = Regime4H.TRANSITIONAL
    trend_dir_4h: Direction = Direction.FLAT
    daily_supports: bool = False
    daily_opposes: bool = False
    composite: CompositeRegime = CompositeRegime.NEUTRAL

    # Cached values
    slope_4h: float = 0.0
    adx_4h: float = 0.0
    slope_d: float = 0.0

    # ES SMA200 daily trend (cross-strategy signal from Vdubus regime)
    es_daily_trend: int = 0            # +1 = above SMA200, -1 = below, 0 = unknown
    last_es_daily_raw: int = 0         # raw (unconfirmed) reading
    es_daily_raw_streak: int = 0       # consecutive bars with same raw reading


# ---------------------------------------------------------------------------
# Position state (global, one position rule)
# ---------------------------------------------------------------------------

@dataclass
class TPLevel:
    r_target: float = 0.0
    pct: float = 0.0
    qty: int = 0
    filled: bool = False
    oms_order_id: str = ""


@dataclass
class PositionState:
    open: bool = False
    symbol: str = "NQ"
    direction: Direction = Direction.FLAT
    entry_subtype: EntrySubtype = EntrySubtype.A_RETEST
    entry_price: float = 0.0
    stop_price: float = 0.0
    initial_stop_price: float = 0.0  # frozen at entry, never migrated
    qty: int = 0
    qty_open: int = 0
    R_dollars: float = 0.0
    risk_pct: float = 0.0
    quality_mult: float = 1.0
    final_risk_pct: float = 0.0
    exit_tier: ExitTier = ExitTier.NEUTRAL
    profit_funded: bool = False
    tp_levels: list[TPLevel] = field(default_factory=list)
    runner_active: bool = False
    chandelier_trail: float = 0.0
    mm_level: float = 0.0
    mm_reached: bool = False
    stop_oms_order_id: str = ""
    bars_since_entry_30m: int = 0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    peak_mfe_r: float = 0.0        # best favorable excursion in R-multiples
    peak_mae_r: float = 0.0        # worst adverse excursion in R-multiples (always >= 0)
    hold_ref: float = 0.0     # for C entries
    box_high_at_entry: float = 0.0
    box_low_at_entry: float = 0.0
    box_mid_at_entry: float = 0.0
    entry_session: Session = Session.RTH
    # TP1-only cap for DEGRADED/RANGE (fix #3)
    tp1_only_cap: bool = False
    # Overnight bridge extension (fix #9)
    stale_bridge_extended: bool = False
    stale_bridge_extra_bars: int = 0
    bars_since_tp1: int = -1  # -1 = TP1 not yet hit; 0+ = bars since TP1 fill
    peak_r_initial: float = 0.0   # peak R using initial_stop_price as denominator
    early_be_triggered: bool = False  # True once pre-TP1 BE has been applied
    stop_source: str = "INITIAL"  # tracks last stop update source for exit reason


# ---------------------------------------------------------------------------
# Working orders
# ---------------------------------------------------------------------------

@dataclass
class WorkingOrder:
    oms_order_id: str = ""
    subtype: EntrySubtype = EntrySubtype.A_RETEST
    direction: Direction = Direction.FLAT
    price: float = 0.0
    qty: int = 0
    submitted_bar_idx: int = 0
    ttl_bars: int = 3
    oca_group: str = ""
    is_limit: bool = False
    rescue_attempted: bool = False
    # Carry quality_mult for exit tier at fill (fix #5)
    quality_mult: float = 1.0
    # Stop price for risk (needed at fill for position setup)
    stop_for_risk: float = 0.0
    # Expected fill price for slippage calculation (planned_entry, not stop_for_risk)
    expected_fill_price: float = 0.0
    # Normalized displacement at entry for diagnostic decomposition
    disp_norm: float = 0.0


# ---------------------------------------------------------------------------
# Daily risk state
# ---------------------------------------------------------------------------

@dataclass
class NewsEvent:
    event_type: str = ""
    event_time_utc: Optional[datetime] = None


@dataclass
class DailyRiskState:
    realized_pnl_R: float = 0.0
    halted: bool = False
    trade_date: Optional[str] = None

    # Weekly / monthly for circuit breakers
    weekly_realized_R: float = 0.0
    monthly_realized_R: float = 0.0
    weekly_halted: bool = False
    monthly_halted: bool = False

    # Rolling daily PnL ledger: list of (date_str, pnl_R) for last 20 days
    daily_pnl_ledger: list = field(default_factory=list)
