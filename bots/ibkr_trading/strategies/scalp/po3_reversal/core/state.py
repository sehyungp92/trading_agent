from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from strategies.scalp.po3_reversal.config import EntryType, SetupTier, TradeDirection
from strategies.scalp.po3_reversal.liquidity import SweepResult
from strategies.scalp.po3_reversal.models import Po3Context, Po3Setup
from strategies.scalp.po3_reversal.smt import SmtResult


@dataclass(frozen=True, slots=True)
class Po3BarInput:
    symbol: str
    bar_ts: datetime
    bar_ohlcv: tuple[float, float, float, float, float]
    context: Po3Context
    sweep: SweepResult | None = None
    smt: SmtResult | None = None
    ifvg: object | None = None
    signal_score: float = 0.0
    signal_threshold: float = 0.0
    tier: SetupTier = SetupTier.NONE
    direction: TradeDirection = TradeDirection.FLAT
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    qty: int = 0
    rr: float = 0.0
    risk_approved: bool = False
    decision_code: str = ""
    decision_details: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Po3Fill:
    oms_order_id: str
    fill_price: float
    fill_qty: int
    symbol: str
    fill_time: datetime
    commission: float = 0.0
    order_role: str = ""


@dataclass(frozen=True, slots=True)
class Po3FlattenRequest:
    setup_id: str
    symbol: str
    reason: str


@dataclass(slots=True)
class Po3ReversalCoreState:
    active_setup: Po3Setup | None = None
    position: Po3Setup | None = None
    order_to_setup: dict[str, str] = field(default_factory=dict)
    order_kind: dict[str, str] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
