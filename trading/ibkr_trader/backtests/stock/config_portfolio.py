"""Portfolio backtest configuration for stock family.

Wraps ALCB + IARIC into a unified backtest with stock family portfolio rules
(8R directional cap, symbol collision half_size).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backtests.stock.config import SlippageConfig
from backtests.stock.config_alcb import ALCBBacktestConfig
from backtests.stock.config_iaric import IARICBacktestConfig


@dataclass
class PortfolioBacktestConfig:
    """Portfolio-level backtest configuration for stock family."""

    data_dir: Path = field(default_factory=lambda: Path("backtests/stock/data/raw"))

    # Which strategies to include
    run_alcb: bool = True
    run_iaric: bool = True

    # Date filters (applied to all strategies)
    start_date: str = "2024-01-01"
    end_date: str = "2026-03-01"
    initial_equity: float = 10_000.0
    tier: int = 1

    # Stock family portfolio rules
    family_directional_cap_r: float = 8.0     # max 8R same-direction aggregate
    symbol_collision_half_size: bool = True    # half size if sibling holds same ticker
    combined_heat_cap_r: float = 10.0         # combined heat cap across strategies
    base_risk_fraction: float = 0.005         # 50bps — used to convert dollar risk → R-units

    # Per-strategy configs (constructed from shared fields if not overridden)
    alcb_config: ALCBBacktestConfig | None = None
    iaric_config: IARICBacktestConfig | None = None

    slippage: SlippageConfig = field(default_factory=SlippageConfig)

    verbose: bool = False

    def build_strategy_configs(self) -> tuple[ALCBBacktestConfig, IARICBacktestConfig]:
        """Build per-strategy configs from shared portfolio settings."""
        alcb = self.alcb_config or ALCBBacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=self.initial_equity,
            tier=self.tier,
            data_dir=self.data_dir,
            slippage=self.slippage,
            verbose=self.verbose,
        )
        iaric = self.iaric_config or IARICBacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=self.initial_equity,
            tier=self.tier,
            data_dir=self.data_dir,
            slippage=self.slippage,
            verbose=self.verbose,
        )
        return alcb, iaric
