"""Unified risk state models for the monorepo scaffold."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class StrategyRiskState:
    strategy_id: str
    trade_date: date
    daily_realized_pnl: float = 0.0
    daily_realized_R: float = 0.0
    open_risk_dollars: float = 0.0
    open_risk_R: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    weekly_realized_pnl: float = 0.0
    weekly_realized_R: float = 0.0
    strategy_daily_pnl: float = 0.0


@dataclass
class PortfolioRiskState:
    trade_date: date
    daily_realized_pnl: float = 0.0
    daily_realized_R: float = 0.0
    open_risk_dollars: float = 0.0
    open_risk_R: float = 0.0
    pending_entry_risk_R: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    weekly_realized_pnl: float = 0.0
    weekly_realized_R: float = 0.0
    week_start_date: date | None = None
    strategy_daily_pnl: dict[str, float] = field(default_factory=dict)

