"""IARIC-specific backtest configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backtests.stock.config import SlippageConfig, UniverseConfig


@dataclass(frozen=True)
class IARICAblationFlags:
    """Toggle individual IARIC components for attribution analysis."""

    use_regime_gate: bool = True
    use_sector_limit: bool = True
    use_sponsorship_filter: bool = True
    use_conviction_scaling: bool = True
    use_flow_reversal_exit: bool = True
    use_time_stop: bool = True
    use_partial_take: bool = True
    use_avwap_breakdown_exit: bool = True
    use_carry_logic: bool = True


@dataclass
class IARICBacktestConfig:
    """IARIC backtest configuration."""

    start_date: str = "2024-03-22"
    end_date: str = "2026-03-01"
    initial_equity: float = 10_000.0
    tier: int = 1                              # 1 = daily bars, 2 = 5m bars
    data_dir: Path = field(default_factory=lambda: Path("backtests/stock/data/raw"))
    warmup_days: int = 250                     # bars needed before first trade
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    ablation: IARICAblationFlags = field(default_factory=IARICAblationFlags)

    # Portfolio constraints (IARIC-specific defaults)
    max_positions_tier_a: int = 8
    max_positions_tier_b: int = 5
    max_per_sector: int = 5
    sector_risk_cap_pct: float = 35.0

    # Strategy param overrides (keys match StrategySettings fields)
    param_overrides: dict = field(default_factory=dict)

    # Logging
    verbose: bool = False
    log_trades: bool = True
