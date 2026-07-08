"""Shared enums and dataclasses for the stock backtesting framework.

All strategy engines import Direction and TradeRecord from here,
keeping sim_broker strategy-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass
class TradeRecord:
    """Completed trade record for post-hoc analysis."""

    strategy: str
    symbol: str
    direction: Direction
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    r_multiple: float
    risk_per_share: float
    commission: float
    slippage: float
    entry_type: str = ""
    exit_reason: str = ""
    sector: str = ""
    regime_tier: str = ""
    hold_bars: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    metadata: dict = field(default_factory=dict)
    signal_time: datetime | None = None
    signal_bar_index: int = -1
    fill_time: datetime | None = None
    fill_bar_index: int = -1
    reentry_sequence: int = 0

    @property
    def pnl_net(self) -> float:
        # pnl already includes slippage (computed from slipped fill prices),
        # so only subtract commission to avoid double-counting.
        return self.pnl - self.commission

    @property
    def hold_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600

    @property
    def is_winner(self) -> bool:
        return self.pnl_net > 0
