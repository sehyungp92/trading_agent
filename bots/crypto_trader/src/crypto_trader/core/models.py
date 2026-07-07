"""Core domain models for the trading system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TimeFrame(Enum):
    """Supported bar timeframes."""
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        _map = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        return _map[self.value]

    @classmethod
    def from_interval(cls, s: str) -> TimeFrame:
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown interval: {s!r}")


class Side(Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class SetupGrade(Enum):
    """Quality grade for trade setups."""
    A = "A"
    B = "B"
    C = "C"


class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "PENDING"
    WORKING = "WORKING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bar:
    """Single OHLCV bar."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame


# ---------------------------------------------------------------------------
# Orders / Fills
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """An order submitted to a broker."""
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    qty: float
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""
    oca_group: str | None = None
    time_in_force: str = "GTC"
    submit_time: datetime | None = None
    ttl_bars: int | None = None
    status: OrderStatus = OrderStatus.PENDING
    metadata: dict = field(default_factory=dict)
    _bars_alive: int = 0


@dataclass(frozen=True, slots=True)
class Fill:
    """A single execution (fill) of an order."""
    order_id: str
    symbol: str
    side: Side
    qty: float
    fill_price: float
    commission: float
    timestamp: datetime
    tag: str
    exchange_order_id: str = ""
    exchange_fill_id: str = ""
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Positions / Trades
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """An open position in a symbol."""
    symbol: str
    direction: Side
    qty: float
    avg_entry: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    partial_exit_pnl: float = 0.0
    partial_exit_commission: float = 0.0
    partial_exit_qty: float = 0.0
    open_time: datetime | None = None
    leverage: float = 1.0
    liquidation_price: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Trade:
    """A completed round-trip trade."""
    trade_id: str
    symbol: str
    direction: Side
    entry_price: float
    exit_price: float
    qty: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    r_multiple: float | None
    commission: float
    bars_held: int
    setup_grade: SetupGrade | None
    exit_reason: str
    confluences_used: list[str] | None
    confirmation_type: str | None
    entry_method: str | None
    funding_paid: float
    mae_r: float | None
    mfe_r: float | None
    realized_r_multiple: float | None = None
    signal_variant: str | None = None

    @property
    def net_pnl(self) -> float:
        """PnL net of commissions — use for trade-level win/loss, PF, etc."""
        return self.pnl - (self.commission or 0.0)

    @property
    def economic_r_multiple(self) -> float | None:
        """Prefer realized/net R when available, else fall back to geometric R."""
        if self.realized_r_multiple is not None:
            return self.realized_r_multiple
        return self.r_multiple


@dataclass
class TerminalMark:
    """Net-liquidation mark for a still-open position at backtest end."""
    symbol: str
    direction: Side
    qty: float
    timestamp: datetime
    entry_price: float
    mark_price_raw: float
    mark_price_net_liquidation: float
    unrealized_pnl_net: float
    unrealized_r_at_mark: float | None
    setup_grade: SetupGrade | None = None
    confluences_used: list[str] | None = None
    confirmation_type: str | None = None
    entry_method: str | None = None
    leverage: float | None = None
    liquidation_price: float | None = None
    metadata: dict = field(default_factory=dict)
