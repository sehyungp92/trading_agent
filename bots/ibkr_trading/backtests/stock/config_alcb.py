"""ALCB-specific backtest configuration — Intraday Momentum Continuation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backtests.stock.config import SlippageConfig, UniverseConfig


@dataclass(frozen=True)
class ALCBAblationFlags:
    """Toggle individual ALCB momentum components for attribution analysis."""

    # Core
    use_regime_gate: bool = True
    use_sector_limit: bool = True
    use_heat_cap: bool = True
    use_long_only: bool = True

    # Entry filters
    use_rvol_filter: bool = True
    use_cpr_filter: bool = True
    use_avwap_filter: bool = True
    use_momentum_score_gate: bool = True
    use_prior_day_high_breakout: bool = True
    use_combined_breakout: bool = True

    # Exit
    use_flow_reversal_exit: bool = True
    use_carry_logic: bool = True
    use_partial_takes: bool = True
    use_time_stop: bool = False
    use_time_based_quick_exit: bool = False

    # Phase 8 gates
    use_combined_quality_gate: bool = False
    use_avwap_distance_cap: bool = False
    use_or_width_min: bool = False
    use_breakout_distance_cap: bool = False

    # Phase 9 gates
    use_quick_exit_stage1: bool = False
    use_or_quality_gate: bool = False
    use_qe_no_recycle: bool = False

    # Phase 10 (P14): Engine-level alpha recovery
    use_mfe_conviction_exit: bool = False
    use_adaptive_trail: bool = False

    # P15 extension: US_ORB-inspired gates and management
    use_orb_quality_gate: bool = False
    use_orb_gap_policy: bool = False
    use_orb_entry_range_gate: bool = False
    use_orb_retracement_trail: bool = False


@dataclass
class ALCBBacktestConfig:
    """ALCB backtest configuration."""

    start_date: str = "2024-01-01"
    end_date: str = "2026-03-01"
    initial_equity: float = 10_000.0
    tier: int = 1
    data_dir: Path = field(default_factory=lambda: Path("backtests/stock/data/raw"))
    warmup_days: int = 250
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    ablation: ALCBAblationFlags = field(default_factory=ALCBAblationFlags)

    # Portfolio constraints
    max_positions: int = 8
    max_per_sector: int = 3
    heat_cap_r: float = 7.0

    # Strategy param overrides (keys match StrategySettings fields)
    param_overrides: dict = field(default_factory=dict)

    # Logging
    verbose: bool = False
    log_trades: bool = True
