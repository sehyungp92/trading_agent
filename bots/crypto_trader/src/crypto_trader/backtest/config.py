"""Backtest configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class BacktestConfig:
    symbols: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    initial_equity: float = 10_000.0
    taker_fee_bps: float = 3.5
    maker_fee_bps: float = 1.0
    slippage_bps: float = 2.0
    spread_bps: float = 2.0
    train_pct: float = 0.70
    apply_funding: bool = True
    warmup_days: int = 0  # extra days loaded before start_date for indicator warmup
