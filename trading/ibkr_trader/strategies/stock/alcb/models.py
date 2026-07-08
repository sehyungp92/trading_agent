"""Typed models for the ALCB v1 strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


@dataclass(slots=True)
class ResearchDailyBar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    event_tag: str = ""

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def cpr(self) -> float:
        width = max(self.high - self.low, 1e-9)
        return (self.close - self.low) / width


@dataclass(slots=True)
class Bar:
    symbol: str
    start_time: datetime
    end_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def cpr(self) -> float:
        width = max(self.high - self.low, 1e-9)
        return (self.close - self.low) / width


@dataclass(slots=True)
class MarketResearch:
    price_ok: bool
    breadth_pct_above_20dma: float
    vix_percentile_1y: float
    hy_spread_5d_bps_change: float
    market_wide_institutional_selling: bool = False


@dataclass(slots=True)
class SectorResearch:
    name: str
    flow_trend_20d: float
    breadth_20d: float
    participation: float


@dataclass(slots=True)
class ResearchSymbol:
    symbol: str
    exchange: str
    primary_exchange: str
    currency: str
    tick_size: float
    point_value: float
    sector: str
    price: float
    adv20_usd: float
    median_spread_pct: float
    earnings_within_sessions: int | None
    blacklist_flag: bool
    halted_flag: bool
    severe_news_flag: bool
    etf_flag: bool = False
    adr_flag: bool = False
    preferred_flag: bool = False
    otc_flag: bool = False
    hard_to_borrow_flag: bool = False
    biotech_flag: bool = False
    flow_proxy_history: list[float] = field(default_factory=list)
    daily_bars: list[ResearchDailyBar] = field(default_factory=list)
    bars_30m: list[Bar] = field(default_factory=list)
    sector_return_20d: float = 0.0
    sector_return_60d: float = 0.0
    intraday_atr_seed: float = 0.0
    average_30m_volume: float = 0.0
    median_30m_volume: float = 0.0
    expected_5m_volume: float = 0.0

    @property
    def trend_price(self) -> float:
        if self.daily_bars:
            return self.daily_bars[-1].close
        return self.price


@dataclass(slots=True)
class HeldPositionResearch:
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    size: int
    stop: float
    initial_r: float
    setup_tag: str = ""
    carry_eligible_flag: bool = False


@dataclass(slots=True)
class ResearchSnapshot:
    trade_date: date
    market: MarketResearch
    sectors: dict[str, SectorResearch]
    symbols: dict[str, ResearchSymbol]
    held_positions: list[HeldPositionResearch] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class RegimeSnapshot:
    score: float
    tier: str
    risk_multiplier: float
    price_ok: bool
    breadth_ok: bool
    vol_ok: bool
    credit_ok: bool
    market_regime: str = "TRANSITIONAL"


class CampaignState(str, Enum):
    INACTIVE = "INACTIVE"
    COMPRESSION = "COMPRESSION"
    BREAKOUT = "BREAKOUT"
    POSITION_OPEN = "POSITION_OPEN"
    CONTINUATION = "CONTINUATION"
    DIRTY = "DIRTY"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(str, Enum):
    # Legacy (compression breakout)
    A_AVWAP_RETEST = "A_AVWAP_RETEST"
    B_SWEEP_RECLAIM = "B_SWEEP_RECLAIM"
    C_CONTINUATION = "C_CONTINUATION"
    D_DIRECT_BREAKOUT = "D_DIRECT_BREAKOUT"
    # Momentum continuation (T1)
    OR_BREAKOUT = "OR_BREAKOUT"
    PDH_BREAKOUT = "PDH_BREAKOUT"
    COMBINED_BREAKOUT = "COMBINED_BREAKOUT"


class Regime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    TRANSITIONAL = "TRANSITIONAL"
    CHOP = "CHOP"


class CompressionTier(str, Enum):
    GOOD = "GOOD"
    NEUTRAL = "NEUTRAL"
    LOOSE = "LOOSE"


class MomentumTradeClass(str, Enum):
    MOMENTUM_CONTINUATION = "MOMENTUM_CONTINUATION"
    GRINDING_HIGHER = "GRINDING_HIGHER"
    STALLING = "STALLING"
    FAILED = "FAILED"


@dataclass(slots=True)
class MomentumSetup:
    symbol: str
    or_high: float
    or_low: float
    or_volume: float
    prior_day_high: float
    prior_day_low: float
    prior_day_close: float
    breakout_level: float
    entry_type: str
    rvol_at_entry: float
    momentum_score: int
    score_detail: dict
    avwap_at_entry: float


@dataclass(slots=True)
class Box:
    start_date: str
    end_date: str
    L_used: int
    high: float
    low: float
    mid: float
    height: float
    containment: float
    squeeze_metric: float
    tier: CompressionTier


@dataclass(slots=True)
class BreakoutQualification:
    direction: Direction
    breakout_date: str
    structural_pass: bool
    displacement_pass: bool
    disp_value: float
    disp_threshold: float
    breakout_rejected: bool
    rvol_d: float
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PositionPlan:
    symbol: str
    direction: Direction
    entry_type: EntryType
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    quantity: int
    risk_per_share: float
    risk_dollars: float
    quality_mult: float
    regime_mult: float
    corr_mult: float


@dataclass(slots=True)
class Campaign:
    symbol: str
    state: CampaignState = CampaignState.INACTIVE
    campaign_id: int = 0
    box_version: int = 0
    box: Box | None = None
    avwap_anchor_ts: str | None = None
    breakout: BreakoutQualification | None = None
    dirty_since: str | None = None
    add_count: int = 0
    campaign_risk_used: float = 0.0
    profit_funded: bool = False
    position_open: bool = False
    continuation_enabled: bool = False
    reentry_block_same_direction: bool = False
    reentry_block_opposite_enhanced: bool = False
    last_entry_type: EntryType | None = None


@dataclass(slots=True)
class CandidateItem:
    symbol: str
    exchange: str
    primary_exchange: str
    currency: str
    tick_size: float
    point_value: float
    sector: str
    adv20_usd: float
    median_spread_pct: float
    selection_score: int
    selection_detail: dict[str, int]
    stock_regime: str
    market_regime: str
    sector_regime: str
    daily_trend_sign: int
    relative_strength_percentile: float
    accumulation_score: float
    ttm_squeeze_bonus: int
    average_30m_volume: float
    median_30m_volume: float
    tradable_flag: bool
    direction_bias: str
    price: float
    earnings_risk_flag: bool
    campaign: Campaign = field(default_factory=lambda: Campaign(symbol=""))
    daily_bars: list[ResearchDailyBar] = field(default_factory=list)
    bars_30m: list[Bar] = field(default_factory=list)


@dataclass(slots=True)
class CandidateArtifact:
    trade_date: date
    generated_at: datetime
    regime: RegimeSnapshot
    items: list[CandidateItem]
    tradable: list[CandidateItem]
    overflow: list[CandidateItem]
    long_candidates: list[CandidateItem] = field(default_factory=list)
    short_candidates: list[CandidateItem] = field(default_factory=list)
    market_wide_institutional_selling: bool = False

    @property
    def by_symbol(self) -> dict[str, CandidateItem]:
        return {item.symbol: item for item in self.items}

    def tradable_symbols(self) -> list[str]:
        return [item.symbol for item in self.tradable]


@dataclass(slots=True)
class QuoteSnapshot:
    ts: datetime
    bid: float
    ask: float
    last: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    cumulative_volume: float = 0.0
    cumulative_value: float = 0.0
    vwap: float | None = None
    is_halted: bool = False
    spread_pct: float = 0.0


@dataclass(slots=True)
class VWAPLedger:
    cum_pv: float = 0.0
    cum_vol: float = 0.0
    value: float | None = None
    reset_key: str = ""

    def update(self, bar: Bar, reset_key: str | None = None) -> None:
        if reset_key is not None and self.reset_key and self.reset_key != reset_key:
            self.cum_pv = 0.0
            self.cum_vol = 0.0
            self.value = None
        if reset_key is not None:
            self.reset_key = reset_key
        self.cum_pv += bar.typical_price * bar.volume
        self.cum_vol += bar.volume
        if self.cum_vol > 0:
            self.value = self.cum_pv / self.cum_vol


@dataclass(slots=True)
class AVWAPLedger:
    anchor_ts: datetime | None = None
    cum_pv: float = 0.0
    cum_vol: float = 0.0
    value: float | None = None

    def bootstrap(self, value: float, anchor_ts: datetime | None = None) -> None:
        self.anchor_ts = anchor_ts
        self.cum_pv = value
        self.cum_vol = 1.0
        self.value = value

    def update(self, bar: Bar) -> None:
        self.cum_pv += bar.typical_price * bar.volume
        self.cum_vol += bar.volume
        if self.cum_vol > 0:
            self.value = self.cum_pv / self.cum_vol


@dataclass(slots=True)
class PendingOrderState:
    oms_order_id: str
    submitted_at: datetime
    role: str
    requested_qty: int
    filled_qty: int = 0
    limit_price: float | None = None
    stop_price: float | None = None
    direction: Direction | None = None
    entry_type: EntryType | None = None
    entry_price: float | None = None
    planned_stop_price: float | None = None
    planned_tp1_price: float | None = None
    planned_tp2_price: float | None = None
    risk_per_share: float | None = None
    risk_dollars: float | None = None
    cancel_requested: bool = False


@dataclass(slots=True)
class PositionState:
    direction: Direction
    entry_price: float
    qty_entry: int
    qty_open: int
    final_stop: float
    current_stop: float
    entry_time: datetime
    initial_risk_per_share: float
    max_favorable_price: float
    max_adverse_price: float
    tp1_price: float
    tp2_price: float
    partial_taken: bool = False
    tp2_taken: bool = False
    profit_funded: bool = False
    stop_order_id: str = ""
    stop_submitted_at: datetime | None = None
    tp1_order_id: str = ""
    tp2_order_id: str = ""
    trade_id: str = ""
    realized_pnl_usd: float = 0.0
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    setup_tag: str = "UNCLASSIFIED"
    stale_warning_emitted: bool = False
    opened_trade_date: date | None = None
    exit_oca_group: str = ""

    @property
    def total_initial_risk_usd(self) -> float:
        return self.initial_risk_per_share * self.qty_entry

    def unrealized_r(self, last_price: float) -> float:
        if self.initial_risk_per_share <= 0:
            return 0.0
        if self.direction == Direction.LONG:
            return (last_price - self.entry_price) / self.initial_risk_per_share
        return (self.entry_price - last_price) / self.initial_risk_per_share


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    last_price: float | None = None
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0
    session_vwap: float | None = None
    weekly_vwap: float | None = None
    avwap_live: float | None = None
    last_quote: QuoteSnapshot | None = None
    last_1m_bar: Bar | None = None
    last_30m_bar: Bar | None = None
    last_4h_bar: Bar | None = None
    minute_bars: list[Bar] = field(default_factory=list)
    bars_30m: list[Bar] = field(default_factory=list)
    bars_4h: list[Bar] = field(default_factory=list)
    daily_bars: list[ResearchDailyBar] = field(default_factory=list)


@dataclass(slots=True)
class SymbolRuntimeState:
    symbol: str
    campaign: Campaign
    intraday_score: int = 0
    intraday_detail: dict[str, int] = field(default_factory=dict)
    mode: str = "NORMAL"
    last_transition_reason: str = ""
    last_30m_bar_time: datetime | None = None
    entry_order: PendingOrderState | None = None
    stop_order: PendingOrderState | None = None
    exit_order: PendingOrderState | None = None
    tp1_order: PendingOrderState | None = None
    tp2_order: PendingOrderState | None = None
    position: PositionState | None = None
    pending_hard_exit: bool = False
    pending_add: bool = False
    last_signal_factors: dict[str, Any] = field(default_factory=dict)
    last_market_regime: str = ""
    last_stock_regime: str = ""


@dataclass(slots=True)
class IntradayStateSnapshot:
    trade_date: date
    saved_at: datetime
    symbols: list[SymbolRuntimeState]
    markets: list[MarketSnapshot]
    last_decision_code: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class T2PositionState:
    """Live position state for the T2 momentum engine."""

    symbol: str
    direction: Direction
    entry_price: float
    stop_price: float
    current_stop: float
    quantity: int
    qty_original: int
    risk_per_share: float
    entry_time: datetime
    entry_type: str  # OR_BREAKOUT / COMBINED_BREAKOUT / PDH_BREAKOUT
    sector: str
    regime_tier: str
    momentum_score: int
    avwap_at_entry: float
    or_high: float
    or_low: float
    hold_bars: int = 0
    mfe_r: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    partial_taken: bool = False
    partial_qty_exited: int = 0
    realized_partial_pnl: float = 0.0
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    carry_days: int = 0
    setup_tag: str = ""
    stop_order_id: str = ""
    trade_id: str = ""
    fr_trailing_active: bool = False  # True if any trailing mechanism ratcheted the stop
    trade_class: str = ""  # MomentumTradeClass value (set at exit)

    def unrealized_r(self, price: float) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        if self.direction == Direction.LONG:
            return (price - self.entry_price) / self.risk_per_share
        return (self.entry_price - price) / self.risk_per_share

    def update_mfe_mae(self, high: float, low: float) -> None:
        if self.direction == Direction.LONG:
            self.max_favorable = max(self.max_favorable, high)
            self.max_adverse = min(self.max_adverse, low)
        else:
            self.max_favorable = min(self.max_favorable, low)
            self.max_adverse = max(self.max_adverse, high)
        if self.risk_per_share > 0:
            if self.direction == Direction.LONG:
                self.mfe_r = max(self.mfe_r, (self.max_favorable - self.entry_price) / self.risk_per_share)
            else:
                self.mfe_r = max(self.mfe_r, (self.entry_price - self.max_favorable) / self.risk_per_share)


@dataclass(slots=True)
class PortfolioState:
    account_equity: float
    base_risk_fraction: float
    open_positions: dict[str, PositionState] = field(default_factory=dict)
    pending_entry_risk: dict[str, float] = field(default_factory=dict)
    total_pnl_pct: float = 0.0
    halt_new_entries: bool = False
    flatten_all: bool = False

    def open_risk_dollars(self) -> float:
        total = 0.0
        for position in self.open_positions.values():
            total += position.qty_open * position.initial_risk_per_share
        return total

    def directional_risk_dollars(self, direction: Direction) -> float:
        total = 0.0
        for position in self.open_positions.values():
            if position.direction != direction:
                continue
            total += position.qty_open * position.initial_risk_per_share
        return total

    def pending_entry_risk_dollars(self) -> float:
        return sum(self.pending_entry_risk.values())

    def occupied_slots(self) -> int:
        pending_symbols = {symbol for symbol, risk in self.pending_entry_risk.items() if risk > 0}
        return len(set(self.open_positions) | pending_symbols)

    def sector_position_count(self, symbol_to_sector: dict[str, str], sector: str) -> int:
        return sum(1 for symbol in self.open_positions if symbol_to_sector.get(symbol) == sector)

    def sector_open_risk(self, symbol_to_sector: dict[str, str], sector: str) -> float:
        total = 0.0
        for symbol, position in self.open_positions.items():
            if symbol_to_sector.get(symbol) != sector:
                continue
            total += position.qty_open * position.initial_risk_per_share
        return total

    def correlation_heat_penalty(self, symbol: str, direction: Direction) -> float:
        same_direction = sum(1 for position in self.open_positions.values() if position.direction == direction)
        if same_direction <= 1:
            return 1.0
        return 1.0 + (0.1 * min(same_direction - 1, 3))
