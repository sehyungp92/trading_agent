"""Portfolio backtest configuration.

Wraps PortfolioConfig presets with backtest-specific fields
(which strategies to run, fixed qty overrides, date filters).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from libs.oms.config.portfolio_config import (
    PortfolioConfig,
    make_10k_config,
    make_10k_optimized_config,
    make_10k_v3_config,
    make_10k_v4_config,
    make_10k_v5_config,
    make_10k_v6_config,
    make_100k_config,
    make_100k_halfsized_config,
    make_100k_optimized_config,
    make_100k_v3_config,
    make_100k_v3_max_config,
)


PRESETS: dict[str, callable] = {
    "10k": make_10k_config,
    "10k_opt": make_10k_optimized_config,
    "10k_v3": make_10k_v3_config,
    "10k_v4": make_10k_v4_config,
    "10k_v5": make_10k_v5_config,
    "10k_v6": make_10k_v6_config,
    "100k": make_100k_config,
    "100k_opt": make_100k_optimized_config,
    "100k_v3": make_100k_v3_config,
    "100k_v3_max": make_100k_v3_max_config,
    "100k_half": make_100k_halfsized_config,
}


@dataclass
class PortfolioBacktestConfig:
    """Portfolio-level backtest configuration."""

    portfolio: PortfolioConfig = field(default_factory=make_10k_v6_config)
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))

    # Which strategies to include
    run_nqdtc: bool = True
    run_vdubus: bool = True

    # Date filters (applied to all strategies)
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Weekly portfolio stop (in R-multiples, 0 = disabled)
    portfolio_weekly_stop_R: float = 12.0

    # MNQ instrument (shared across all strategies)
    point_value: float = 2.0
    tick_size: float = 0.25
    commission_per_side: float = 0.62
