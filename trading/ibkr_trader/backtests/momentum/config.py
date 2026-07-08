"""Shared backtest configuration dataclasses."""
from __future__ import annotations

import math
from dataclasses import dataclass


def round_to_tick(price: float, tick_size: float, direction: str = "nearest") -> float:
    """Round price to valid tick increment."""
    if direction == "up":
        return math.ceil(price / tick_size) * tick_size
    elif direction == "down":
        return math.floor(price / tick_size) * tick_size
    return round(price / tick_size) * tick_size


@dataclass(frozen=True)
class SlippageConfig:
    """Execution simulation parameters."""

    slip_ticks_normal: int = 1
    slip_ticks_illiquid: int = 2
    illiquid_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 22, 23)  # UTC
    commission_per_contract: float = 0.62  # IBKR micros
    use_stop_limit: bool = True
    use_stop_market: bool = False         # J2 variant: stop-market fills (optimistic)
    halt_zero_range_bars: int = 2         # consecutive zero-range bars → halt
    halt_extra_slip_ticks: int = 3        # additional slippage on post-halt reopen
    spread_bps: float = 0.0              # spread-based slippage (bps of price); 0 = disabled
