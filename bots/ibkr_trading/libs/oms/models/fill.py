"""Fill model."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Fill:
    fill_id: str  # internal unique
    oms_order_id: str
    broker_fill_id: str  # IB execId — dedupe key
    price: float
    qty: float
    timestamp: datetime
    fees: float = 0.0
