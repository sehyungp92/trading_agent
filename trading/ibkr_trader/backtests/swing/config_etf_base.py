"""Shared backtest configuration for ETF swing strategies."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ETFSlippageConfig:
    slip_ticks_normal: int = 1
    slip_ticks_illiquid: int = 2
    illiquid_hours: tuple[int, ...] = (16, 17)  # 12:00-13:59 ET in UTC-ish fallback
    commission_per_share: float = 0.005
    commission_min_order: float = 0.35
    spread_bps: float = 1.0
    tick_size: float = 0.01
    halt_zero_range_bars: int = 2
    halt_extra_slip_ticks: int = 3

    @property
    def commission_per_contract(self) -> float:
        return self.commission_per_share

