from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from backtests.stock.models import Direction, TradeRecord

from ..phase_candidates import STRATEGY_ORDER


class PortfolioActionType(str, Enum):
    SUBMIT_ENTRY = "SubmitEntry"
    BLOCK_ENTRY = "BlockEntry"
    SUBMIT_EXIT = "SubmitExit"


@dataclass(frozen=True)
class PortfolioAction:
    action_type: PortfolioActionType
    timestamp: datetime
    strategy_id: str
    symbol: str
    reason: str = ""
    risk_dollars: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionEvent:
    timestamp: datetime
    strategy_id: str
    symbol: str
    decision_code: str
    reason: str
    state_snapshot_ref: str
    actions_emitted: tuple[PortfolioAction, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeOutcome:
    strategy_id: str
    symbol: str
    entry_time: datetime
    decision_time: datetime
    fill_time: datetime
    exit_time: datetime
    gross_pnl: float
    commission: float
    net_pnl: float
    r_multiple: float
    risk_dollars: float
    exit_reason: str
    route: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioPosition:
    strategy: str
    symbol: str
    sector: str
    direction: Direction
    entry_time: datetime
    decision_time: datetime
    fill_time: datetime
    exit_time: datetime
    risk_dollars: float
    pnl: float
    r_multiple: float
    quality: float
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    price_scale: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""
    entry_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayCandidate:
    strategy: str
    trade: TradeRecord
    risk_dollars: float
    pnl: float
    r_multiple: float
    heat_r: float
    quality: float
    size_mult: float
    portfolio_size_mult: float = 1.0


@dataclass(frozen=True)
class BlockedCandidate:
    strategy: str
    symbol: str
    sector: str
    entry_time: datetime
    r_multiple: float
    reason: str
    quality: float
    heat_r: float


@dataclass
class PortfolioCoreState:
    equity: float
    peak_equity: float
    reference_risk_pct: float
    active_positions: list[PortfolioPosition] = field(default_factory=list)
    accepted_positions: list[PortfolioPosition] = field(default_factory=list)
    blocked_candidates: list[BlockedCandidate] = field(default_factory=list)
    equity_points: list[float] = field(default_factory=list)
    equity_times: list[datetime] = field(default_factory=list)
    daily_realized_r: dict[str, float] = field(default_factory=dict)
    weekly_realized_r: dict[str, float] = field(default_factory=dict)
    strategy_recent: dict[str, deque[float]] = field(default_factory=dict)
    risk_by_strategy: dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    decision_seq: int = 0

    @classmethod
    def initial(cls, *, initial_equity: float, reference_risk_pct: float, lookback_trades: int) -> "PortfolioCoreState":
        return cls(
            equity=float(initial_equity),
            peak_equity=float(initial_equity),
            reference_risk_pct=float(reference_risk_pct),
            equity_points=[float(initial_equity)],
            strategy_recent={strategy: deque(maxlen=int(lookback_trades)) for strategy in STRATEGY_ORDER},
            risk_by_strategy={strategy: 0.0 for strategy in STRATEGY_ORDER},
        )


@dataclass(frozen=True)
class PortfolioReplayResult:
    metrics: dict[str, float]
    state: PortfolioCoreState
    decisions: tuple[DecisionEvent, ...]
    actions: tuple[PortfolioAction, ...]
    trade_outcomes: tuple[TradeOutcome, ...]
    replay_architecture: str = "stock_portfolio_core_live_rule_adapter"
