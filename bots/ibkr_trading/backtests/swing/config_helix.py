"""Helix-specific backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.swing.config import SlippageConfig


def _default_helix_symbols() -> list[str]:
    try:
        from strategies.swing.akc_helix.config import SYMBOLS
        return list(SYMBOLS)
    except ImportError:
        return ["QQQ", "GLD"]


@dataclass
class HelixAblationFlags:
    """Toggle each Helix-specific filter/condition for ablation testing.

    Baseline: all False (nothing disabled).  Set one to True to disable it.
    """

    disable_class_a: bool = False
    disable_class_b: bool = False
    disable_class_c: bool = False
    disable_class_d: bool = False
    disable_add_ons: bool = False
    disable_partial_2p5r: bool = False
    disable_partial_5r: bool = False
    disable_chandelier_trailing: bool = False
    disable_circuit_breaker: bool = False
    disable_spread_gate: bool = False
    disable_corridor_cap: bool = False
    disable_basket_rule: bool = False
    disable_extreme_vol_gate: bool = False


@dataclass
class HelixBacktestConfig:
    """Top-level Helix backtest configuration."""

    symbols: list[str] = field(default_factory=_default_helix_symbols)
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 100_000.0
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    flags: HelixAblationFlags = field(default_factory=HelixAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    track_shadows: bool = True
    warmup_daily: int = 60
    warmup_hourly: int = 55
    warmup_4h: int = 50
    fixed_qty: int | None = None
    enforce_initial_risk_cap: bool = True
    initial_risk_cap_buffer: float = 0.0
