"""Downturn Dominator backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.momentum.config import SlippageConfig
from strategies.momentum.downturn.config import DownturnAblationFlags  # noqa: F401


@dataclass
class DownturnBacktestConfig:
    """Top-level Downturn Dominator backtest configuration."""

    symbols: list[str] = field(default_factory=lambda: ["NQ"])
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 10_000.0
    slippage: SlippageConfig = field(
        default_factory=lambda: SlippageConfig(
            commission_per_contract=0.62,
            slip_ticks_normal=1,
            slip_ticks_illiquid=2,
        ),
    )
    flags: DownturnAblationFlags = field(default_factory=DownturnAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: Path("backtests/momentum/data/raw"))
    track_signals: bool = True
    skip_parity_output: bool = False  # skip decision_stream + trade_outcomes normalization (optimization mode)
    warmup_days: int = 60

    # Instrument (MNQ -- uses NQ price data, trades Micro E-mini)
    tick_size: float = 0.50
    point_value: float = 2.0
    max_contracts: int = 0  # 0 = unlimited; >0 = hard cap on position qty
    max_notional_leverage: float = 20.0  # cap qty to equity * leverage / notional (20x = realistic MNQ)
    max_dd_abort: float = 0.0  # >0 enables early termination when portfolio DD exceeds threshold
