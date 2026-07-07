from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ActionOrderType = Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
ActionSide = Literal["BUY", "SELL"]


@dataclass(slots=True, frozen=True)
class ActionContext:
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = ""
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmitEntry:
    client_order_id: str
    symbol: str
    side: ActionSide
    qty: int
    order_type: ActionOrderType
    tif: str = "DAY"
    price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = ""
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmitExit:
    client_order_id: str
    symbol: str
    side: ActionSide
    qty: int
    order_type: ActionOrderType
    tif: str = "DAY"
    price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = ""
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmitAddOnEntry:
    client_order_id: str
    symbol: str
    side: ActionSide
    qty: int
    order_type: ActionOrderType
    tif: str = "DAY"
    price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = "add_on_entry"
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmitProtectiveStop:
    client_order_id: str
    symbol: str
    side: ActionSide
    qty: int
    stop_price: float
    tif: str = "GTC"
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = "protective_stop"
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ReplaceProtectiveStop:
    symbol: str
    target_order_id: str
    side: ActionSide
    stop_price: float
    qty: int
    reason: str
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = "protective_stop"
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CancelAction:
    symbol: str
    target_order_id: str
    reason: str
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = ""
    route: str = ""
    session: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FlattenPosition:
    symbol: str
    reason: str
    side: ActionSide | None = None
    qty: int = 0
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = "flatten"
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmitProfitTarget:
    client_order_id: str
    symbol: str
    side: ActionSide
    qty: int
    limit_price: float
    tif: str = "GTC"
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = "profit_target"
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmitPartialExit:
    client_order_id: str
    symbol: str
    side: ActionSide
    qty: int
    order_type: ActionOrderType = "MARKET"
    tif: str = "DAY"
    price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = "partial_exit"
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SubmitMarketExit:
    client_order_id: str
    symbol: str
    side: ActionSide
    qty: int
    tif: str = "DAY"
    parent_order_id: str = ""
    oca_group: str = ""
    role: str = "market_exit"
    route: str = ""
    session: str = ""
    risk_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


NeutralOrderAction = (
    SubmitEntry
    | SubmitExit
    | SubmitAddOnEntry
    | SubmitProtectiveStop
    | ReplaceProtectiveStop
    | SubmitProfitTarget
    | SubmitPartialExit
    | SubmitMarketExit
)

NeutralAction = NeutralOrderAction | CancelAction | FlattenPosition
