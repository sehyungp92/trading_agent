"""Typed dataclasses for IBKR-specific representations."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class BrokerOrderStatus(Enum):
    PENDING_SUBMIT = "PendingSubmit"
    PENDING_CANCEL = "PendingCancel"
    PRE_SUBMITTED = "PreSubmitted"
    SUBMITTED = "Submitted"
    CANCELLED = "Cancelled"
    FILLED = "Filled"
    INACTIVE = "Inactive"


class RejectCategory(Enum):
    PERMISSIONS = "REJECT_PERMISSIONS"
    INVALID_PRICE = "REJECT_INVALID_PRICE"
    RISK = "REJECT_RISK"
    PACING = "REJECT_PACING"
    CONTRACT = "REJECT_CONTRACT"
    DUPLICATE = "REJECT_DUPLICATE"
    UNKNOWN = "REJECT_UNKNOWN"
    TRANSIENT = "TRANSIENT_DISCONNECT"


@dataclass(frozen=True)
class IBContractSpec:
    con_id: int
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    multiplier: float
    tick_size: float
    trading_class: str
    primary_exchange: str = ""
    last_trade_date: str = ""  # YYYYMMDD for futures, blank for stocks


@dataclass(frozen=True)
class BrokerOrderRef:
    broker_order_id: int
    perm_id: int
    con_id: int


@dataclass(frozen=True)
class ExecutionReport:
    exec_id: str
    broker_order_id: int
    perm_id: int
    symbol: str
    side: str  # "BOT" or "SLD"
    qty: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    exchange: str = ""


@dataclass(frozen=True)
class OrderStatusEvent:
    broker_order_id: int
    perm_id: int
    status: BrokerOrderStatus
    filled_qty: float
    remaining_qty: float
    avg_fill_price: float
    last_fill_price: float = 0.0
    order_ref: str = ""
    account: str = ""
    client_id: int | None = None


@dataclass(frozen=True)
class PositionSnapshot:
    account: str
    con_id: int
    symbol: str
    qty: float
    avg_cost: float


@dataclass(frozen=True)
class AccountSnapshot:
    account: str
    net_liquidation: float
    available_funds: float
    buying_power: float
    timestamp: datetime
