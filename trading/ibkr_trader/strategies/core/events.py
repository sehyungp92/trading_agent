from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class DecisionEvent:
    code: str
    ts: datetime
    symbol: str
    timeframe: str
    details: dict[str, Any] = field(default_factory=dict)
    strategy_id: str = ""
    state_ref: str = ""
    emitted_actions: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = "decision_event_v1"
    event_type: str = "decision_event"
    bot_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    strategy_version: str = ""
    config_version: str = ""
    portfolio_config_version: str = ""
    risk_config_version: str = ""
    allocation_version: str = ""
    strategy_registry_version: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    code_sha: str = ""
    trace_id: str = ""
    bar_id: str = ""
    decision_kind: str = ""
    sequence: int = 0


@dataclass(slots=True, frozen=True)
class TradeOutcome:
    strategy_id: str
    symbol: str
    direction: int
    entry_ts: datetime
    exit_ts: datetime | None
    qty: int
    entry_price: float
    exit_price: float | None = None
    initial_stop: float | None = None
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    realized: bool = True
    exit_reason: str = ""
    source_label: str = ""
    decision_ts: datetime | None = None
    fill_ts: datetime | None = None
    route: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
