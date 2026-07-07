from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from strategies.momentum.nqdtc.models import (
    Direction,
    EntrySubtype,
    ExitTier,
    PositionState,
    Session,
    TPLevel,
    WorkingOrder,
)


@dataclass(slots=True)
class NQDTCCoreState:
    symbol: str = ""
    position: PositionState = field(default_factory=PositionState)
    working_orders: list[WorkingOrder] = field(default_factory=list)
    bar_count_5m: int = 0
    last_decision_code: str = "IDLE"
    last_decision_details: dict = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class NQDTCEntryRequest:
    client_order_id: str
    symbol: str
    subtype: EntrySubtype
    direction: Direction
    qty: int
    stop_for_risk: float
    tif: str = "DAY"
    order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"] = "STOP_LIMIT"
    price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    oca_group: str = ""
    is_limit: bool = False
    quality_mult: float = 1.0
    submitted_bar_idx: int = 0
    ttl_bars: int = 6


@dataclass(slots=True)
class NQDTCSimpleRequest:
    reason: str
    price: float | None = None
    qty: int = 0


@dataclass(slots=True)
class NQDTCOrderUpdate:
    oms_order_id: str
    status: str
    timestamp: datetime | None = None
    order_role: Literal["entry", "stop", "flatten", "unknown"] = "unknown"
    accepted_entry: NQDTCEntryRequest | None = None


@dataclass(slots=True)
class NQDTCEntryFillContext:
    exit_tier: ExitTier
    tp_levels: list[TPLevel]
    mm_level: float
    mm_reached: bool
    box_high_at_entry: float
    box_low_at_entry: float
    box_mid_at_entry: float
    entry_session: Session
    tp1_only_cap: bool
    r_dollars: float = 0.0


@dataclass(slots=True)
class NQDTCFill:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    fill_time: datetime | None = None
    entry_context: NQDTCEntryFillContext | None = None
    exit_type: str | None = None
