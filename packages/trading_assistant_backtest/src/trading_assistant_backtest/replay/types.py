"""Normalized replay result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: date
    end: date


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    window: WindowSpec
    trade_count: int = 0
    net_return: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    objective_score: float = 0.0
    trades: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
