from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from strategies.momentum.downturn.models import (
    ActivePosition,
    CompositeRegime,
    EngineTag,
    VolState,
    WorkingEntry,
)


@dataclass(slots=True)
class DownturnCoreState:
    symbol: str = ""
    position: ActivePosition | None = None
    working_entries: list[WorkingEntry] = field(default_factory=list)
    bar_count_5m: int = 0
    bars_since_last_entry: int = 999
    last_decision_code: str = "IDLE"
    last_decision_details: dict = field(default_factory=dict)
    last_bar_ts: datetime | None = None


@dataclass(slots=True)
class DownturnEntryRequest:
    client_order_id: str
    symbol: str
    engine_tag: EngineTag
    signal_class: str
    qty: int
    entry_price: float
    stop0: float
    tif: str = "DAY"
    order_type: Literal["STOP", "STOP_LIMIT"] = "STOP_LIMIT"
    side: Literal["SELL"] = "SELL"
    price: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    submitted_bar_idx: int = 0
    ttl_bars: int = 72
    composite_regime: CompositeRegime = CompositeRegime.NEUTRAL
    vol_state: VolState = VolState.NORMAL
    in_correction: bool = False
    predator: bool = False
    tp_schedule: list[tuple[float, float]] = field(default_factory=list)
    signal_strength: float = 0.5


@dataclass(slots=True)
class DownturnOrderUpdate:
    oms_order_id: str
    status: str
    timestamp: datetime | None = None
    order_role: Literal["entry", "stop", "flatten", "unknown"] = "unknown"
    accepted_entry: DownturnEntryRequest | None = None


@dataclass(slots=True)
class DownturnFill:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    commission: float = 0.0
    fill_time: datetime | None = None
    exit_type: str | None = None


@dataclass(slots=True)
class DownturnStopUpdateRequest:
    stop_price: float
    qty: int
    reason: str
