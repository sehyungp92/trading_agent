"""OMS event models."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OMSEventType(Enum):
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_RISK_APPROVED = "ORDER_RISK_APPROVED"
    ORDER_QUEUED = "ORDER_QUEUED"
    ORDER_ROUTED = "ORDER_ROUTED"
    ORDER_ACKED = "ORDER_ACKED"
    ORDER_WORKING = "ORDER_WORKING"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    # Status event emitted when an order reaches FILLED. This is not a fill
    # payload; consumers that mutate trade state must wait for FILL.
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    FILL = "FILL"
    POSITION_UPDATE = "POSITION_UPDATE"
    RISK_HALT = "RISK_HALT"
    RISK_DENIAL = "RISK_DENIAL"
    RISK_DECISION = "RISK_DECISION"
    RECONCILIATION_ALERT = "RECONCILIATION_ALERT"
    COORDINATION = "COORDINATION"


@dataclass
class OMSEvent:
    event_type: OMSEventType
    timestamp: datetime
    strategy_id: str
    oms_order_id: Optional[str] = None
    payload: Optional[dict] = None
