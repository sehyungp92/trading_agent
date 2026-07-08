"""Schema row types for OMS persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class RiskDailyStrategyRow:
    """Daily risk metrics per strategy."""

    trade_date: date
    strategy_id: str
    family_id: str = "unknown"
    daily_realized_r: Decimal = Decimal("0")
    daily_realized_usd: Optional[Decimal] = None
    open_risk_r: Decimal = Decimal("0")
    filled_entries: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None
    last_update_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class RiskDailyPortfolioRow:
    """Daily risk metrics for entire portfolio (per-family)."""

    trade_date: date
    family_id: str = "unknown"
    daily_realized_r: Decimal = Decimal("0")
    daily_realized_usd: Optional[Decimal] = None
    portfolio_open_risk_r: Decimal = Decimal("0")
    halted: bool = False
    halt_reason: Optional[str] = None
    last_update_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class TradeRow:
    """Completed trade record."""

    trade_id: str
    strategy_id: str
    instrument_symbol: str
    direction: str  # LONG/SHORT
    quantity: int
    entry_ts: datetime
    entry_price: Decimal
    exit_ts: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    realized_r: Optional[Decimal] = None
    realized_usd: Optional[Decimal] = None
    exit_reason: Optional[str] = None
    setup_tag: Optional[str] = None
    entry_type: Optional[str] = None
    notes: Optional[str] = None
    meta_json: str = "{}"
    account_id: str = "default"


@dataclass(frozen=True)
class TradeMarksRow:
    """MAE/MFE metrics for a trade."""

    trade_id: str
    duration_seconds: Optional[int] = None
    duration_bars: Optional[int] = None
    mae_r: Optional[Decimal] = None
    mfe_r: Optional[Decimal] = None
    mae_usd: Optional[Decimal] = None
    mfe_usd: Optional[Decimal] = None
    max_adverse_price: Optional[Decimal] = None
    max_favorable_price: Optional[Decimal] = None


@dataclass(frozen=True)
class StrategyStateRow:
    """Strategy instance health state."""

    strategy_id: str
    mode: str = "RUNNING"  # RUNNING, STAND_DOWN, HALTED
    last_heartbeat_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    instance_id: str = "primary"
    stand_down_reason: Optional[str] = None
    last_decision_code: Optional[str] = None
    last_decision_details_json: str = "{}"
    last_error_ts: Optional[datetime] = None
    last_error: Optional[str] = None
    last_seen_bar_ts: Optional[datetime] = None
    heat_r: Decimal = Decimal("0")
    daily_pnl_r: Decimal = Decimal("0")


@dataclass(frozen=True)
class AdapterStateRow:
    """Broker adapter connection state."""

    adapter_id: str
    broker: str = "IBKR"
    connected: bool = False
    last_heartbeat_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_disconnect_ts: Optional[datetime] = None
    disconnect_count_24h: int = 0
    last_error_code: Optional[str] = None
    last_error_message: Optional[str] = None
