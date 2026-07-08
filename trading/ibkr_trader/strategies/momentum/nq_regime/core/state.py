from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

from strategies.momentum.nq_regime.config import Grade, IBType, ModuleId, TradeSide
from strategies.momentum.nq_regime.core.levels import IBLevels, KeyLevels
from strategies.momentum.nq_regime.core.session import SessionPhase
from strategies.momentum.nq_regime.modules.base import SetupCandidate


@dataclass(frozen=True, slots=True)
class BarData:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    vwap: float | None = None

    @property
    def range_pts(self) -> float:
        return max(0.0, self.high - self.low)

    @property
    def body_pts(self) -> float:
        return abs(self.close - self.open)

    @property
    def close_location_long(self) -> float:
        if self.high <= self.low:
            return 0.5
        return (self.close - self.low) / (self.high - self.low)

    @property
    def close_location_short(self) -> float:
        if self.high <= self.low:
            return 0.5
        return (self.high - self.close) / (self.high - self.low)


@dataclass(frozen=True, slots=True)
class BarEvent:
    ts: datetime
    bar_5m: BarData
    bar_15m_closed: BarData | None = None
    is_new_15m: bool = False
    daily_context: KeyLevels | None = None
    live_context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExpansionModuleState:
    active_break_side: TradeSide = TradeSide.FLAT
    active_break_ts: datetime | None = None
    active_break_bar_index: int = -1
    active_break_level: float = 0.0
    active_break_high: float = 0.0
    active_break_low: float = 0.0
    active_break_midpoint: float = 0.0
    active_break_fvg_low: float = 0.0
    active_break_fvg_high: float = 0.0
    active_break_score: int = 0
    failed_break_count: int = 0
    retest_expiry_bar: int = -1
    pullback_side: TradeSide = TradeSide.FLAT
    pullback_ts: datetime | None = None
    pullback_bar_index: int = -1
    pullback_reference: float = 0.0
    pullback_extreme: float = 0.0
    pullback_trigger: float = 0.0


@dataclass(slots=True)
class ReversionModuleState:
    last_sweep_level: float = 0.0
    last_sweep_side: TradeSide = TradeSide.FLAT
    last_sweep_ts: datetime | None = None
    last_sweep_bar_index: int = -1
    last_sweep_extreme: float = 0.0
    sweeps_seen: int = 0


@dataclass(slots=True)
class SecondWindModuleState:
    squeeze_start_ts: datetime | None = None
    squeeze_duration: int = 0
    fired_ts: datetime | None = None
    bias: TradeSide = TradeSide.FLAT


@dataclass(slots=True)
class RegimeCoreState:
    phase: SessionPhase = SessionPhase.PRE_MARKET
    regime: Any = None
    regime_scores: Any = None
    active_module: ModuleId = ModuleId.NONE
    active_session_date: str = ""

    ib_levels: IBLevels = field(default_factory=IBLevels)
    ib_high_working: float = 0.0
    ib_low_working: float = 0.0
    ib_locked: bool = False
    ib_type: IBType = IBType.UNCLASSIFIED
    levels: KeyLevels | None = None

    position_side: TradeSide = TradeSide.FLAT
    entry_price: float = 0.0
    entry_time: datetime | None = None
    entry_bar_index: int = -1
    stop_price: float = 0.0
    qty: int = 0
    qty_open: int = 0
    entry_module: ModuleId = ModuleId.NONE
    setup_grade: Grade = Grade.INVALID
    setup_score: int = 0
    initial_risk_points: float = 0.0
    planned_targets: tuple[float, ...] = ()
    partial_taken: int = 0
    stop_at_be: bool = False
    active_trade_id: str | None = None

    working_entry_order_id: str | None = None
    working_stop_order_id: str | None = None
    working_target_order_ids: tuple[str, ...] = ()
    order_to_role: dict[str, str] = field(default_factory=dict)
    order_to_candidate: dict[str, str] = field(default_factory=dict)
    pending_candidates: dict[str, SetupCandidate] = field(default_factory=dict)
    pending_cancel_reason: str | None = None
    last_submitted_signal_id: str | None = None

    expansion_state: ExpansionModuleState = field(default_factory=ExpansionModuleState)
    reversion_state: ReversionModuleState = field(default_factory=ReversionModuleState)
    second_wind_state: SecondWindModuleState = field(default_factory=SecondWindModuleState)

    daily_trades: int = 0
    daily_full_risk_trades: int = 0
    daily_losses: int = 0
    daily_realized_r: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_locked_out: bool = False

    bars_5m: list[BarData] = field(default_factory=list)
    bars_15m: list[BarData] = field(default_factory=list)
    indicators: Any = None
    routing_log: list[Any] = field(default_factory=list)

    bar_index: int = 0
    last_bar_ts: datetime | None = None
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)


def clone_core_state(state: RegimeCoreState) -> RegimeCoreState:
    """Clone mutable state containers without deep-copying immutable history."""
    return replace(
        state,
        order_to_role=dict(state.order_to_role),
        order_to_candidate=dict(state.order_to_candidate),
        pending_candidates=dict(state.pending_candidates),
        expansion_state=replace(state.expansion_state),
        reversion_state=replace(state.reversion_state),
        second_wind_state=replace(state.second_wind_state),
        bars_5m=list(state.bars_5m),
        bars_15m=list(state.bars_15m),
        routing_log=list(state.routing_log),
        last_decision_details=dict(state.last_decision_details),
        planned_targets=tuple(state.planned_targets),
        working_target_order_ids=tuple(state.working_target_order_ids),
    )


@dataclass(slots=True)
class FillEvent:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    fill_time: datetime
    symbol: str = ""
    commission: float = 0.0
    order_role: Literal["entry", "stop", "target_1", "target_2", "partial", "flatten", "unknown"] = "unknown"
    exit_type: str = ""
    fill_id: str = ""
    intent_id: str = ""
    risk_decision_ref: str = ""
    portfolio_decision_ref: str = ""


@dataclass(slots=True)
class OrderUpdateEvent:
    oms_order_id: str
    status: str
    timestamp: datetime
    symbol: str = ""
    order_role: Literal["entry", "stop", "target_1", "target_2", "partial", "flatten", "unknown"] = "unknown"
    reason: str = ""
