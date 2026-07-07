from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import EntryType, SetupTier, TradeDirection


@dataclass(frozen=True, slots=True)
class PriceBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body_percent(self) -> float:
        rng = self.high - self.low
        if rng <= 0:
            return 0.0
        return abs(self.close - self.open) / rng


@dataclass(frozen=True, slots=True)
class Po3Context:
    daily_bias: TradeDirection = TradeDirection.FLAT
    h4_bias: TradeDirection = TradeDirection.FLAT
    h1_target: float = 0.0


@dataclass(slots=True)
class Po3Setup:
    setup_id: str
    symbol: str
    direction: TradeDirection
    tier: SetupTier
    entry_type: EntryType
    signal_time: datetime
    score: float
    entry_price: float
    stop_price: float
    target_price: float
    qty: int
    rr: float
    qty_open: int = 0
    avg_entry: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TradeRecord:
    symbol: str
    side: str
    qty: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    pnl_dollars: float
    r_multiple: float
    exit_reason: str = ""
    setup_id: str = ""
    tier: str = ""
    entry_type: str = ""
