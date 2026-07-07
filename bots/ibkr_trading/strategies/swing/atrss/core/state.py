from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.swing.atrss.models import (
    BreakoutArmState,
    Candidate,
    CandidateType,
    DailyState,
    Direction,
    HaltState,
    HourlyState,
    LegType,
    PositionLeg,
    PositionBook,
    ReentryState,
)


@dataclass(slots=True)
class ATRSSCoreState:
    daily_states: dict[str, DailyState] = field(default_factory=dict)
    hourly_states: dict[str, HourlyState] = field(default_factory=dict)
    positions: dict[str, PositionBook] = field(default_factory=dict)
    reentry_states: dict[str, ReentryState] = field(default_factory=dict)
    pending_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    prev_trend_dirs: dict[str, Direction] = field(default_factory=dict)
    halt_states: dict[str, HaltState] = field(default_factory=dict)
    pending_reverses: list[Candidate] = field(default_factory=list)
    pending_flattens: dict[str, dict[str, Any]] = field(default_factory=dict)
    reopen_at: dict[str, datetime] = field(default_factory=dict)
    breakout_arm_states: dict[str, BreakoutArmState] = field(default_factory=dict)
    risk_halted: bool = False
    risk_halt_reason: str = ""
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class ATRSSBarInput:
    symbol: str = ""
    timeframe: str = ""
    bar_ts: datetime | None = None
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ATRSSEntryRequest:
    client_order_id: str
    symbol: str
    candidate: Candidate
    limit_price: float
    tif: str = "GTC"
    order_type: Literal["STOP_LIMIT"] = "STOP_LIMIT"


@dataclass(slots=True)
class ATRSSAddOnARequest:
    client_order_id: str
    symbol: str
    direction: Direction
    qty: int
    entry_price: float
    stop_price: float
    tif: str = "GTC"
    order_type: Literal["MARKET"] = "MARKET"


@dataclass(slots=True)
class ATRSSStopUpdateRequest:
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class ATRSSPartialExitRequest:
    client_order_id: str
    symbol: str
    qty: int
    reason: str
    tif: str = "GTC"
    order_type: Literal["MARKET"] = "MARKET"


@dataclass(slots=True)
class ATRSSFlattenRequest:
    symbol: str
    reason: str


@dataclass(slots=True)
class ATRSSOrderUpdate:
    oms_order_id: str
    status: str = ""
    symbol: str = ""
    timestamp: datetime | None = None
    order_role: Literal["entry", "add_on", "partial", "stop", "flatten", "unknown"] = "unknown"
    timeframe: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ATRSSFill:
    oms_order_id: str
    fill_price: float = 0.0
    fill_qty: int = 0
    symbol: str = ""
    fill_time: datetime | None = None
    commission: float = 0.0
    exit_type: str = ""
    fill_id: str = ""
    intent_id: str = ""
    risk_decision_ref: str = ""
    portfolio_decision_ref: str = ""
    runtime_payload: dict[str, Any] = field(default_factory=dict)
    timeframe: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)
