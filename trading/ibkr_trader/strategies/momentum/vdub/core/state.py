from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from strategies.momentum.vdub.models import (
    DayCounters,
    Direction,
    EntryType,
    EventBlockState,
    PositionState,
    PositionStage,
    RegimeState,
    SessionWindow,
    WorkingEntry,
)


@dataclass(slots=True)
class VdubCoreState:
    regime: RegimeState = field(default_factory=RegimeState)
    counters: DayCounters = field(default_factory=DayCounters)
    positions: list[PositionState] = field(default_factory=list)
    working_entries: dict[str, WorkingEntry] = field(default_factory=dict)
    event_state: EventBlockState = field(default_factory=EventBlockState)
    bar_idx: int = 0
    last_reset_date: str = ""
    recent_wins: list[bool] = field(default_factory=list)
    last_flatten_oms_id: str | None = None
    last_decision_code: str = "IDLE"
    last_decision_details: dict[str, Any] = field(default_factory=dict)
    last_bar_ts: datetime | None = None


# ── Request / event dataclasses ──────────────────────────────────


@dataclass(slots=True)
class VdubEntrySubmitted:
    """Emitted after engine submits an entry order to OMS."""
    working_entry: WorkingEntry
    oms_order_id: str
    bar_idx: int = 0


@dataclass(slots=True)
class VdubStopUpdateRequest:
    """Request to trail a protective stop."""
    pos_id: str
    new_stop: float
    reason: str


@dataclass(slots=True)
class VdubFlattenRequest:
    """Request to flatten a position."""
    pos_id: str
    reason: str


@dataclass(slots=True)
class VdubPartialExitDone:
    """Notification that engine executed a partial exit (qty already reduced)."""
    pos_id: str
    qty_closed: int
    new_qty: int


@dataclass(slots=True)
class VdubOrderUpdate:
    """OMS order status change."""
    oms_order_id: str
    status: str
    timestamp: datetime | None = None
    order_role: Literal["entry", "stop", "exit", "flatten", "unknown"] = "unknown"
    # For accepted stops: link to position
    pos_id: str = ""


@dataclass(slots=True)
class VdubEntryFillContext:
    """Attached to a fill when it matches a working entry."""
    working_entry: WorkingEntry


@dataclass(slots=True)
class VdubFill:
    """Fill event from OMS."""
    oms_order_id: str
    fill_price: float
    fill_qty: int
    fill_time: datetime | None = None
    point_value: float = 2.0
    commission: float = 0.0
    entry_context: VdubEntryFillContext | None = None
