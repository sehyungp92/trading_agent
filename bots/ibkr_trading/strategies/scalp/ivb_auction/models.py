from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import EntryTrigger, IvbModule, TradeDirection


@dataclass(frozen=True, slots=True)
class ScalpTick:
    ts: datetime
    price: float
    size: float
    bid: float | None = None
    ask: float | None = None
    side: int = 0


@dataclass(frozen=True, slots=True)
class FootprintBarData:
    start_ts: datetime
    end_ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    @property
    def delta(self) -> float:
        return self.ask_volume - self.bid_volume

    @property
    def absorption_score(self) -> float:
        return abs(self.delta)


@dataclass(frozen=True, slots=True)
class IvbSignalScore:
    total: float
    size_multiplier: float = 1.0
    footprint_available: bool = False
    available_components: tuple[str, ...] = ()


@dataclass(slots=True)
class IvbSetup:
    setup_id: str
    symbol: str
    module: IvbModule
    direction: TradeDirection
    trigger: EntryTrigger
    signal_time: datetime
    score: float
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    qty: int
    size_multiplier: float
    rr_to_tp1: float
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
    module: str = ""
    trigger: str = ""
