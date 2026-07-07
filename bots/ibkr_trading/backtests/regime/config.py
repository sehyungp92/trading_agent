"""Configuration for regime backtesting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RegimeBacktestConfig:
    initial_equity: float = 100_000.0
    rebalance_cost_bps: float = 5.0
    growth_feature: str = "GROWTH"
    inflation_feature: str = "INFLATION"
    data_dir: Path = Path("backtests/regime/data/raw")
