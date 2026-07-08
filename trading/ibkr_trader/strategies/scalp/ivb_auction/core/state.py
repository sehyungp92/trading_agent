from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp._shared.session import ScalpSessionBlock
from strategies.scalp.ivb_auction.config import EntryTrigger, IvbModule, TradeDirection
from strategies.scalp.ivb_auction.models import FootprintBarData, IvbSetup


@dataclass(frozen=True, slots=True)
class IvbBarInput:
    symbol: str
    bar_ts: datetime
    bar_ohlcv: tuple[float, float, float, float, float]
    session_block: ScalpSessionBlock
    ivb_levels: IVBLevels
    breakout_direction: TradeDirection = TradeDirection.FLAT
    breakout_accepted: bool = False
    module: IvbModule | None = None
    trigger: EntryTrigger | None = None
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    qty: int = 0
    rr_to_tp1: float = 0.0
    signal_score: float = 0.0
    size_multiplier: float = 1.0
    footprint_state: FootprintBarData | None = None
    decision_code: str = ""
    decision_details: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IvbFill:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    symbol: str
    fill_time: datetime
    commission: float = 0.0
    order_role: str = ""


@dataclass(frozen=True, slots=True)
class IvbFlattenRequest:
    setup_id: str
    symbol: str
    reason: str


@dataclass(slots=True)
class IvbAuctionCoreState:
    active_setups: dict[str, IvbSetup] = field(default_factory=dict)
    order_to_setup: dict[str, str] = field(default_factory=dict)
    order_kind: dict[str, str] = field(default_factory=dict)
    positions: dict[str, IvbSetup] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
