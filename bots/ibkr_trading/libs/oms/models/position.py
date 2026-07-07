"""Position model."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    account_id: str
    instrument_symbol: str
    strategy_id: str
    net_qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    open_risk_dollars: float = 0.0
    open_risk_R: float = 0.0
    last_update_at: Optional[datetime] = None
