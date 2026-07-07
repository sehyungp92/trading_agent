"""Typed models for the IARIC v1 strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import fmean
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
    flow_proxy_history: list[float] = field(default_factory=list)
    daily_bars: list[ResearchDailyBar] = field(default_factory=list)
    sector_return_20d: float = 0.0
    sector_return_60d: float = 0.0
    intraday_atr_seed: float = 0.0
    average_30m_volume: float = 0.0
    expected_5m_volume: float = 0.0

    @property
    def trend_price(self) -> float:
        if self.daily_bars:
            return self.daily_bars[-1].close
        return self.price

    @property
    def sma20(self) -> float:
        bars = self.daily_bars[-20:]
        if not bars:
            return self.trend_price
        return fmean(bar.close for bar in bars)

    @property
    def sma50(self) -> float:
        bars = self.daily_bars[-50:]
        if not bars:
            return self.sma20
        return fmean(bar.close for bar in bars)

    @property
    def sma50_slope(self) -> float:
        if len(self.daily_bars) < 51:
            return 0.0
        prev = fmean(bar.close for bar in self.daily_bars[-51:-1])
        curr = self.sma50
        return curr - prev

    @property
    def stock_return_20d(self) -> float:
        if len(self.daily_bars) < 21 or self.daily_bars[-21].close <= 0:
            return 0.0
        return (self.daily_bars[-1].close - self.daily_bars[-21].close) / self.daily_bars[-21].close

    @property
    def stock_return_60d(self) -> float:
        if len(self.daily_bars) < 61 or self.daily_bars[-61].close <= 0:
            return 0.0
        return (self.daily_bars[-1].close - self.daily_bars[-61].close) / self.daily_bars[-61].close

    @property
    def daily_atr_estimate(self) -> float:
        sample = self.daily_bars[-15:]
        if len(sample) < 2:
            return max(self.intraday_atr_seed, self.tick_size)
        true_ranges: list[float] = []
        prev_close = sample[0].close
        for bar in sample[1:]:
            true_ranges.append(max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close)))
            prev_close = bar.close
        return fmean(true_ranges) if true_ranges else max(self.intraday_atr_seed, self.tick_size)


@dataclass(slots=True)
class HeldPositionResearch:
    symbol: str
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


@dataclass(slots=True)
class HeldPositionDirective:
    symbol: str
    entry_time: datetime
    entry_price: float
    size: int
    stop: float
    initial_r: float
    setup_tag: str
    time_stop_deadline: datetime | None
    carry_eligible_flag: bool
    flow_reversal_flag: bool


@dataclass(slots=True)
class WatchlistItem:
    symbol: str
    exchange: str
    primary_exchange: str
    currency: str
    tick_size: float
    point_value: float
    sector: str
    regime_score: float
    regime_tier: str
    regime_risk_multiplier: float
    sector_score: float
    sector_rank_weight: float
    sponsorship_score: float
    sponsorship_state: str
    persistence: float
    intensity_z: float
    accel_z: float
    rs_percentile: float
    leader_pass: bool
    trend_pass: bool
    trend_strength: float
    earnings_risk_flag: bool
    blacklist_flag: bool
    anchor_date: date
    anchor_type: str
    acceptance_pass: bool
    avwap_ref: float
    avwap_band_lower: float
    avwap_band_upper: float
    daily_atr_estimate: float
    intraday_atr_seed: float
    daily_rank: float
    tradable_flag: bool
    conviction_bucket: str
    conviction_multiplier: float
    recommended_risk_r: float
    average_30m_volume: float = 0.0
    expected_5m_volume: float = 0.0
    entry_gap_pct: float = 0.0
    flow_proxy_gate_pass: bool = True   # True = flow positive (or unavailable), safe default
    overflow_rank: int | None = None
    # Pullback V2 fields
    daily_signal_score: float = 0.0
    trigger_types: list[str] = field(default_factory=list)
    trigger_tier: str = "STANDARD"       # PREMIUM/STANDARD/REDUCED/MINIMUM
    trend_tier: str = "STRONG"           # STRONG/SECULAR/EXCLUDED
    rescue_flow_candidate: bool = False
    sizing_mult: float = 1.0
    cdd_value: int = 0                    # consecutive down days at selection time
    ema10_daily: float = 0.0              # latest daily EMA(10) for exit chain
    rsi14_daily: float = 0.0              # latest daily RSI(14) for exit chain


@dataclass(slots=True)
class WatchlistArtifact:
    trade_date: date
    generated_at: datetime
    regime: RegimeSnapshot
    items: list[WatchlistItem]
    tradable: list[WatchlistItem]
    overflow: list[WatchlistItem]
    market_wide_institutional_selling: bool = False
    held_positions: list[HeldPositionDirective] = field(default_factory=list)

    @property
    def by_symbol(self) -> dict[str, WatchlistItem]:
        return {item.symbol: item for item in self.items}

    def tradable_symbols(self) -> list[str]:
        return [item.symbol for item in self.tradable]


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
    typical_price: float = field(init=False)

    def __post_init__(self) -> None:
        self.typical_price = (self.high + self.low + self.close) / 3.0

    @property
    def cpr(self) -> float:
        width = max(self.high - self.low, 1e-9)
        return (self.close - self.low) / width


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

    def update(self, bar: Bar) -> None:
        self.cum_pv += bar.typical_price * bar.volume
        self.cum_vol += bar.volume
        if self.cum_vol > 0:
            self.value = self.cum_pv / self.cum_vol


@dataclass(slots=True)
class AVWAPLedger:
    cum_pv: float
    cum_vol: float
    value: float

    @classmethod
    def bootstrap(cls, avwap: float, volume: float = 1.0) -> "AVWAPLedger":
        base_volume = max(volume, 1.0)
        return cls(cum_pv=avwap * base_volume, cum_vol=base_volume, value=avwap)

    def update(self, bar: Bar) -> None:
        self.cum_pv += bar.typical_price * bar.volume
        self.cum_vol += bar.volume
        self.value = self.cum_pv / max(self.cum_vol, 1e-9)


@dataclass(slots=True)
class PendingOrderState:
    oms_order_id: str
    submitted_at: datetime
    role: str
    requested_qty: int
    limit_price: float | None = None
    stop_price: float | None = None
    cancel_requested: bool = False


@dataclass(slots=True)
class PositionState:
    entry_price: float
    qty_entry: int
    qty_open: int
    final_stop: float
    current_stop: float
    entry_time: datetime
    initial_risk_per_share: float
    max_favorable_price: float
    max_adverse_price: float
    partial_taken: bool = False
    stop_order_id: str = ""
    trade_id: str = ""
    realized_pnl_usd: float = 0.0
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    setup_tag: str = "UNCLASSIFIED"
    time_stop_deadline: datetime | None = None

    @property
    def total_initial_risk_usd(self) -> float:
        return self.initial_risk_per_share * self.qty_entry


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    last_price: float | None = None
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0
    session_high: float | None = None
    session_low: float | None = None
    session_vwap: float | None = None
    avwap_live: float | None = None
    last_quote: QuoteSnapshot | None = None
    last_1m_bar: Bar | None = None
    last_5m_bar: Bar | None = None
    last_30m_bar: Bar | None = None
    minute_bars: deque[Bar] = field(default_factory=lambda: deque(maxlen=390))
    bars_5m: deque[Bar] = field(default_factory=lambda: deque(maxlen=120))
    bars_30m: deque[Bar] = field(default_factory=lambda: deque(maxlen=40))
    tick_pressure_window: deque[tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=512))
    # Incremental VWAP accumulators (avoids O(N) re-sum every bar)
    _cum_pv: float = 0.0
    _cum_vol: float = 0.0

    @property
    def minutes_since_hod(self) -> int:
        if self.session_high is None:
            return 0
        for index, bar in enumerate(reversed(self.minute_bars)):
            if bar.high >= self.session_high:
                return index
        return len(self.minute_bars)

    @property
    def drop_from_hod(self) -> float:
        if self.session_high is None or self.last_price is None or self.session_high <= 0:
            return 0.0
        return max(0.0, (self.session_high - self.last_price) / self.session_high)


@dataclass(slots=True)
class SymbolIntradayState:
    symbol: str
    tier: str = "COLD"
    fsm_state: str = "IDLE"
    in_position: bool = False
    position_qty: int = 0
    avg_price: float | None = None
    setup_type: str | None = None
    setup_low: float | None = None
    reclaim_level: float | None = None
    stop_level: float | None = None
    setup_time: datetime | None = None
    invalidated_at: datetime | None = None
    acceptance_count: int = 0
    required_acceptance_count: int = 0
    location_grade: str | None = None
    session_vwap: float | None = None
    avwap_live: float | None = None
    sponsorship_signal: str = "NEUTRAL"
    micropressure_signal: str = "NEUTRAL"
    micropressure_mode: str = "PROXY"
    flowproxy_signal: str = "UNAVAILABLE"
    confidence: str | None = None
    last_1m_bar_time: datetime | None = None
    last_5m_bar_time: datetime | None = None
    active_order_id: str | None = None
    time_stop_deadline: datetime | None = None
    setup_tag: str | None = None
    expected_volume_pct: float = 0.0
    average_30m_volume: float = 0.0
    last_transition_reason: str = ""
    entry_order: PendingOrderState | None = None
    position: PositionState | None = None
    exit_order: PendingOrderState | None = None
    pending_hard_exit: bool = False


@dataclass(slots=True)
class PBSymbolState:
    """Per-symbol intraday state for the pullback hybrid engine."""
    symbol: str
    stage: str = "WATCHING"  # WATCHING|FLUSH_LOCKED|RECLAIMING|READY|IN_POSITION|INVALIDATED
    route_family: str = ""   # OPENING_RECLAIM|DELAYED_CONFIRM|VWAP_BOUNCE|AFTERNOON_RETEST|OPEN_SCORED_ENTRY
    intraday_setup_type: str = ""
    setup_low: float = 0.0
    reclaim_level: float = 0.0
    stop_level: float = 0.0
    acceptance_count: int = 0
    required_acceptance: int = 1
    intraday_score: float = 0.0
    score_components: dict = field(default_factory=dict)
    bars_seen_today: int = 0
    session_low: float = 0.0
    session_high: float = 0.0
    in_position: bool = False
    position: PositionState | None = None
    entry_order: PendingOrderState | None = None
    exit_order: PendingOrderState | None = None
    pending_hard_exit: bool = False
    # Daily context from watchlist
    daily_signal_score: float = 0.0
    trigger_types: list[str] = field(default_factory=list)
    trigger_tier: str = "STANDARD"
    trend_tier: str = "STRONG"
    rescue_flow_candidate: bool = False
    sizing_mult: float = 1.0
    daily_atr: float = 0.0
    entry_atr: float = 0.0
    last_1m_bar_time: datetime | None = None
    last_5m_bar_time: datetime | None = None
    active_order_id: str | None = None
    last_transition_reason: str = ""
    # V2 position tracking
    mfe_stage: int = 0
    breakeven_activated: bool = False
    trail_active: bool = False
    hold_bars: int = 0
    risk_per_share: float = 0.0
    v2_partial_taken: bool = False
    carry_decision_path: str = ""
    # Bars below VWAP counter for VWAP fail exit
    consecutive_bars_below_vwap: int = 0
    cdd_value: int = 0
    # Daily indicator snapshots for exit chain
    ema10_daily: float = 0.0
    rsi14_daily: float = 0.0
    # Intraday tracking (research parity)
    stopped_out_today: bool = False
    flush_bar_idx: int = 0
    ready_bar_idx: int = -1
    target_entry_price: float = 0.0
    improvement_expires: int = 0
    invalid_reason: str = ""
    invalid_reset_bar: int = 0
    ready_cpr: float = 0.0
    ready_volume_ratio: float = 0.0
    ready_timestamp: datetime | None = None
    accepted_bar_idx: int = -1
    accepted_timestamp: datetime | None = None
    accepted_entry_price: float = 0.0
    accepted_entry_trigger: str = ""
    accepted_route_family: str = ""
    accepted_score: float = 0.0
    accepted_session_atr: float = 0.0
    accepted_score_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class IntradayStateSnapshot:
    trade_date: date
    saved_at: datetime
    symbols: list[SymbolIntradayState]
    last_decision_code: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioState:
    account_equity: float
    base_risk_fraction: float
    open_positions: dict[str, PositionState] = field(default_factory=dict)
    pending_entry_risk: dict[str, float] = field(default_factory=dict)
    regime_allows_no_new_entries: bool = False

    def open_risk_dollars(self) -> float:
        total = 0.0
        for position in self.open_positions.values():
            total += position.qty_open * max(position.entry_price - position.current_stop, 0.0)
        return total

    def sector_position_count(self, symbol_to_sector: dict[str, str], sector: str) -> int:
        return sum(1 for symbol in self.open_positions if symbol_to_sector.get(symbol) == sector)

    def sector_open_risk(self, symbol_to_sector: dict[str, str], sector: str) -> float:
        total = 0.0
        for symbol, position in self.open_positions.items():
            if symbol_to_sector.get(symbol) != sector:
                continue
            total += position.qty_open * max(position.entry_price - position.current_stop, 0.0)
        return total


@dataclass(slots=True)
class TierChange:
    symbol: str
    from_tier: str
    to_tier: str
    reason: str
    at: datetime
