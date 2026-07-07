from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.momentum.config import SlippageConfig


@dataclass
class NqRegimeAblationFlags:
    enable_structural_expansion: bool = True
    enable_liquidity_reversion: bool = True
    enable_second_wind: bool = True
    disable_live_context_shadow: bool = False


@dataclass
class NqRegimeBacktestConfig:
    analysis_symbol: str = "NQ"
    trade_symbol: str = "MNQ"
    symbols: list[str] = field(default_factory=lambda: ["MNQ"])
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 10_000.0
    data_dir: Path = field(default_factory=lambda: Path("backtests/momentum/data/raw"))
    flags: NqRegimeAblationFlags = field(default_factory=NqRegimeAblationFlags)
    param_overrides: dict[str, float | int | bool] = field(default_factory=dict)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    fixed_qty: int | None = None
    max_contracts: int | None = None
    replay_mode: str = "bar_only"
    bar_timestamp_mode: str = "start"
    track_decisions: bool = True
    track_trades: bool = True

    def __post_init__(self) -> None:
        self.analysis_symbol = self.analysis_symbol.upper()
        self.trade_symbol = self.trade_symbol.upper()
        self.symbols = [self.trade_symbol]
        if self.bar_timestamp_mode not in {"start", "close"}:
            raise ValueError("bar_timestamp_mode must be 'start' or 'close'")
