"""Order models and enums."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .instrument import Instrument


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderRole(Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    STOP = "STOP"
    TP = "TP"


class OrderStatus(Enum):
    CREATED = "CREATED"
    RISK_APPROVED = "RISK_APPROVED"
    QUEUED = "QUEUED"
    ROUTED = "ROUTED"
    ACKED = "ACKED"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REPLACE_REQUESTED = "REPLACE_REQUESTED"
    REPLACED = "REPLACED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    DONE = "DONE"


TERMINAL_STATUSES = {
    OrderStatus.CANCELLED,
    OrderStatus.FILLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
}


@dataclass(frozen=True)
class BrokerRef:
    """Broker-agnostic order reference for cancel/replace operations."""
    broker_order_id: int
    perm_id: int = 0
    con_id: int = 0


@dataclass
class EntryPolicy:
    """Required for ENTRY orders."""

    ttl_bars: Optional[int] = None
    ttl_seconds: Optional[int] = None
    max_reprices: int = 0
    teleport_ticks: Optional[int] = None
    reprice_band_ticks: Optional[int] = None


@dataclass
class RiskContext:
    """Required for ENTRY orders — used by risk gateway."""

    stop_for_risk: float  # protective stop price
    planned_entry_price: float  # expected fill price
    risk_budget_tag: str = ""  # e.g. "MR", "Trend"
    risk_dollars: float = 0.0  # computed: qty * |entry - stop| * point_value
    portfolio_size_mult: float = 1.0  # cross-strategy sizing adjustment
    # H7: persisted instead of transient dynamic attr (matches retry_count fix below).
    # gateway.py attaches strat_cfg.unit_risk_dollars after approval; without this
    # field, repository.__dict__ persistence + RiskContext(**rc_data) reload raises.
    unit_risk_dollars: float = 0.0
    intent_id: str = ""
    risk_decision_ref: str = ""
    portfolio_decision_ref: str = ""
    gateway_decision_context: dict = field(default_factory=dict)
    trace_id: str = ""
    signal_id: str = ""
    bar_id: str = ""
    exchange_timestamp: Optional[datetime] = None
    lineage_context: dict = field(default_factory=dict)


@dataclass
class OMSOrder:
    # IDs
    oms_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = ""  # idempotency key from strategy

    # Ownership
    strategy_id: str = ""
    account_id: str = ""

    # Instrument
    instrument: Optional["Instrument"] = None

    # Economics
    side: OrderSide = OrderSide.BUY
    qty: int = 0
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: str = "DAY"

    # Role
    role: OrderRole = OrderRole.ENTRY

    # Policy (ENTRY only)
    entry_policy: Optional[EntryPolicy] = None

    # Risk context (ENTRY only)
    risk_context: Optional[RiskContext] = None

    # Broker mapping
    broker: str = "IBKR"
    broker_order_id: Optional[int] = None
    perm_id: Optional[int] = None
    broker_order_ref: Optional[object] = None  # C4: broker ref from adapter ack
    reject_reason: str = ""  # C4: rejection reason from broker

    # OCA
    oca_group: str = ""
    oca_type: int = 0

    # Status
    status: OrderStatus = OrderStatus.CREATED
    created_at: Optional[datetime] = None
    queued_at: Optional[datetime] = None
    queue_priority: Optional[int] = None
    queue_reason: str = ""
    queue_attempt: int = 0
    queue_expires_at: Optional[datetime] = None
    queue_claimed_by: str = ""
    queue_claimed_at: Optional[datetime] = None
    queue_claim_expires_at: Optional[datetime] = None
    dequeued_at: Optional[datetime] = None
    queue_denial_reason: str = ""
    submitted_at: Optional[datetime] = None
    acked_at: Optional[datetime] = None
    last_update_at: Optional[datetime] = None

    # Fill tracking
    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    avg_fill_price: float = 0.0

    # Reprice tracking
    reprice_count: int = 0

    # Retry tracking (H5: persisted instead of transient dynamic attr)
    retry_count: int = 0
