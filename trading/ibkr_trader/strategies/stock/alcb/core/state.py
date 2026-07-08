from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.stock.alcb.models import PositionPlan, T2PositionState


@dataclass(slots=True)
class ALCBCoreState:
    positions: dict[str, T2PositionState] = field(default_factory=dict)
    or_data: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    or_built: dict[str, bool] = field(default_factory=dict)
    order_index: dict[str, tuple[str, str]] = field(default_factory=dict)
    pending_entries: dict[str, str] = field(default_factory=dict)
    pending_exits: dict[str, str] = field(default_factory=dict)
    pending_plans: dict[str, PositionPlan] = field(default_factory=dict)
    entry_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    exit_reasons: dict[str, str] = field(default_factory=dict)
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class ALCBEntryRequest:
    client_order_id: str
    symbol: str
    plan: PositionPlan
    meta: dict[str, Any] = field(default_factory=dict)
    tif: str = "DAY"
    order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"] = "STOP_LIMIT"


@dataclass(slots=True)
class ALCBStopUpdateRequest:
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class ALCBPartialExitRequest:
    client_order_id: str
    symbol: str
    qty: int
    reason: str = "PARTIAL"
    tif: str = "DAY"
    order_type: Literal["MARKET"] = "MARKET"


@dataclass(slots=True)
class ALCBFlattenRequest:
    symbol: str
    reason: str


@dataclass(slots=True)
class ALCBEntryFillContext:
    trade_id: str = ""
    emergency_stop: float | None = None


@dataclass(slots=True)
class ALCBOrderUpdate:
    oms_order_id: str
    status: str
    timestamp: datetime | None = None
    symbol: str = ""
    order_role: Literal["entry", "exit", "partial", "stop", "flatten", "unknown"] = "unknown"
    reason: str = ""
    accepted_entry: ALCBEntryRequest | None = None


@dataclass(slots=True)
class ALCBFill:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    fill_time: datetime | None = None
    commission: float = 0.0
    exit_type: str | None = None
    entry_context: ALCBEntryFillContext | None = None
