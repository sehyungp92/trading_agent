from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Po3ReversalAblationFlags:
    disable_smt: bool = False
    disable_liquidity_sweep: bool = False
    disable_ifvg: bool = False
    disable_spread_gate: bool = False
    disable_volatility_gate: bool = False
    disable_b_tier: bool = False
    disable_time_stop: bool = False
    disable_break_even: bool = False


@dataclass
class Po3ReversalBacktestConfig:
    analysis_symbol: str = "NQ"
    trade_symbol: str = "MNQ"
    symbols: list[str] = field(default_factory=lambda: ["MNQ"])
    confirmation_symbol: str = "ES"
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 10_000.0
    data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    flags: Po3ReversalAblationFlags = field(default_factory=Po3ReversalAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    warmup_daily: int = 60
    warmup_h4: int = 30
    warmup_h1: int = 30
    fixed_qty: int | None = None
    track_signals: bool = True
    track_shadows: bool = True

    def __post_init__(self) -> None:
        self.analysis_symbol = self.analysis_symbol.upper()
        self.trade_symbol = self.trade_symbol.upper()
        self.confirmation_symbol = self.confirmation_symbol.upper()
        self.symbols = [self.trade_symbol]
