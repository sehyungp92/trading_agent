from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class IvbAblationFlags:
    disable_a2_module: bool = False
    disable_absorption_gate: bool = False
    disable_delta_gate: bool = False
    disable_time_filter: bool = False
    disable_ivb_range_filter: bool = False
    disable_chase_rejection: bool = False
    disable_target_gate: bool = False
    disable_depth_shadow: bool = False


@dataclass
class IvbAuctionBacktestConfig:
    analysis_symbol: str = "NQ"
    trade_symbol: str = "MNQ"
    symbols: list[str] = field(default_factory=lambda: ["MNQ"])
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 10_000.0
    data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    flags: IvbAblationFlags = field(default_factory=IvbAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    replay_mode: str = "bar_with_footprint"
    warmup_daily: int = 20
    fixed_qty: int | None = None
    track_shadows: bool = True
    track_signals: bool = True

    def __post_init__(self) -> None:
        self.analysis_symbol = self.analysis_symbol.upper()
        self.trade_symbol = self.trade_symbol.upper()
        self.symbols = [self.trade_symbol]
