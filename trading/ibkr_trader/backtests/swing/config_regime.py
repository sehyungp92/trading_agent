"""Regime-following strategy configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.swing.config import SlippageConfig


def _default_atrss_symbols() -> list[str]:
    try:
        from strategies.swing.atrss.config import SYMBOLS
        return list(SYMBOLS)
    except ImportError:
        return ["QQQ", "GLD"]


@dataclass
class RegimeConfig:
    """Configuration for regime-following strategy.

    Investigation-optimal defaults from structural analysis:
    - chand_mult=1.5 (Inv 4: tight trailing beats 2.2-3.2x baseline)
    - be_trigger_r=1.0 (move to BE at +1.0R)
    - time_exit_hours=100 (Inv 1: TimeExit@80-120h is best exit rule)
    - shorts_enabled=False (Inv 2: shorts destroy value)
    """

    symbols: list[str] = field(default_factory=_default_atrss_symbols)
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 100_000.0
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    warmup_daily: int = 60
    warmup_hourly: int = 55
    fixed_qty: int | None = None

    # Regime-specific parameters (investigation-optimal defaults)
    chand_mult: float = 1.5
    be_trigger_r: float = 1.0
    chandelier_trigger_r: float = 1.0
    regime_downgrade_exit: bool = True
    time_exit_hours: int = 100
    time_exit_min_r: float = 1.0
    shorts_enabled: bool = False
    param_overrides: dict[str, float] = field(default_factory=dict)
