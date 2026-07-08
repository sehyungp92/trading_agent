from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.stock.iaric.models import IntradayStateSnapshot

IARICCoreState = IntradayStateSnapshot


@dataclass(slots=True)
class IARICBarInput:
    symbol: str = ""
    timeframe: str = ""
    bar_ts: datetime | None = None
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IARICEntryRequest:
    client_order_id: str
    symbol: str
    route: str
    qty: int
    limit_price: float
    stop_price: float
    tif: str = "DAY"
    order_type: Literal["LIMIT"] = "LIMIT"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IARICStopUpdateRequest:
    symbol: str
    stop_price: float
    qty: int
    reason: str


@dataclass(slots=True)
class IARICPartialExitRequest:
    client_order_id: str
    symbol: str
    qty: int
    reason: str = "TP"


@dataclass(slots=True)
class IARICFlattenRequest:
    symbol: str
    reason: str
    qty: int = 0


@dataclass(slots=True)
class IARICOrderUpdate:
    oms_order_id: str
    status: str = ""
    timestamp: datetime | None = None
    symbol: str = ""
    order_role: Literal["ENTRY", "TP", "EXIT", "STOP", "UNKNOWN"] = "UNKNOWN"
    reason: str = ""
    timeframe: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IARICFill:
    oms_order_id: str
    fill_price: float = 0.0
    fill_qty: int = 0
    fill_time: datetime | None = None
    commission: float = 0.0
    symbol: str = ""
    order_role: Literal["ENTRY", "TP", "EXIT", "STOP", "UNKNOWN"] = "UNKNOWN"
    exit_type: str = ""
    timeframe: str = ""
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IARICEntryAcceptance:
    accepted_bar_idx: int
    accepted_timestamp: datetime
    accepted_entry_price: float
    entry_trigger: str
    route_family: str
    score: float
    session_atr: float
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class IARICRouteStep:
    prior_stage: str
    stage: str
    reason: str = ""
    score: float = 0.0
    entry_feasible: bool = False
    acceptance: IARICEntryAcceptance | None = None
