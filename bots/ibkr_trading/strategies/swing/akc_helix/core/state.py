from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.swing.akc_helix.models import (
    CircuitBreakerState,
    DailyState,
    Direction,
    PivotStore,
    Regime,
    SetupInstance,
    TFState,
)


@dataclass(slots=True)
class AKCHelixCoreState:
    daily_states: dict[str, DailyState] = field(default_factory=dict)
    tf_states: dict[str, dict[str, TFState]] = field(default_factory=dict)
    pivots: dict[str, dict[str, PivotStore]] = field(default_factory=dict)
    regime_4h: dict[str, Regime] = field(default_factory=dict)
    div_mag_history: dict[str, list[float]] = field(default_factory=dict)
    active_setups: dict[str, SetupInstance] = field(default_factory=dict)
    pending_setups: dict[str, SetupInstance] = field(default_factory=dict)
    queued_setups: dict[str, SetupInstance] = field(default_factory=dict)
    circuit_breakers: dict[str, CircuitBreakerState] = field(default_factory=dict)
    order_to_setup: dict[str, str] = field(default_factory=dict)
    oca_counter: int = 0
    last_b_long_l2_ts: dict[str, datetime | None] = field(default_factory=dict)
    last_b_short_h2_ts: dict[str, datetime | None] = field(default_factory=dict)
    last_d_long_l2_ts: dict[str, datetime | None] = field(default_factory=dict)
    last_d_short_h2_ts: dict[str, datetime | None] = field(default_factory=dict)
    regime_streaks: dict[str, int] = field(default_factory=dict)
    prev_regimes: dict[str, Regime | None] = field(default_factory=dict)
    risk_halted: bool = False
    risk_halt_reason: str = ""
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class AKCHelixBarInput:
    symbol: str = ""
    timeframe: str = ""
    bar_ts: datetime | None = None
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AKCHelixEntryRequest:
    client_order_id: str
    setup: SetupInstance
    order_type: Literal["STOP", "STOP_LIMIT", "MARKET"] = "STOP_LIMIT"
    tif: str = "GTC"
    order_role: Literal["entry", "catchup", "rescue", "add"] = "entry"
    limit_price: float | None = None
    qty: int | None = None


@dataclass(slots=True)
class AKCHelixStopUpdateRequest:
    setup_id: str
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class AKCHelixPartialExitRequest:
    client_order_id: str
    setup_id: str
    symbol: str
    qty: int
    reason: str
    order_type: Literal["MARKET"] = "MARKET"
    tif: str = "GTC"


@dataclass(slots=True)
class AKCHelixFlattenRequest:
    setup_id: str
    symbol: str
    reason: str


@dataclass(slots=True)
class AKCHelixOrderUpdate:
    oms_order_id: str
    status: str = ""
    symbol: str = ""
    timestamp: datetime | None = None
    order_role: Literal["entry", "catchup", "rescue", "add", "partial", "stop", "flatten", "unknown"] = "unknown"
    timeframe: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AKCHelixFill:
    oms_order_id: str
    fill_price: float = 0.0
    fill_qty: int = 0
    point_value: float = 1.0
    symbol: str = ""
    fill_time: datetime | None = None
    commission: float = 0.0
    order_role: Literal["entry", "catchup", "rescue", "add", "partial", "stop", "flatten", "unknown"] = "unknown"
    exit_type: str = ""
    fill_id: str = ""
    intent_id: str = ""
    risk_decision_ref: str = ""
    portfolio_decision_ref: str = ""
    runtime_payload: dict[str, Any] = field(default_factory=dict)
    timeframe: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)
