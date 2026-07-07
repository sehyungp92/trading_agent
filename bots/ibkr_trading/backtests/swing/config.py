"""Backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

def _default_atrss_symbols() -> list[str]:
    try:
        from strategies.swing.atrss.config import SYMBOLS
        return list(SYMBOLS)
    except ImportError:
        return ["QQQ", "GLD"]


@dataclass(frozen=True)
class SlippageConfig:
    """Execution simulation parameters."""

    slip_ticks_normal: int = 1
    slip_ticks_illiquid: int = 2
    illiquid_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 22, 23)  # UTC
    commission_per_contract: float = 0.62  # IBKR micros / full-size futures
    commission_per_share_etf: float = 0.0035  # IBKR tiered US equities (min $0.35/order)
    commission_min_etf_order: float = 0.35
    use_stop_limit: bool = True
    use_stop_market: bool = False         # J2 variant: stop-market fills (optimistic)
    halt_zero_range_bars: int = 2         # consecutive zero-range bars → halt
    halt_extra_slip_ticks: int = 3        # additional slippage on post-halt reopen
    spread_bps: float = 0.0              # spread-based slippage (bps of price); 0 = disabled
    overlay_slip_bps: float = 5.0         # overlay ETF execution slippage (bps of price)


@dataclass
class AblationFlags:
    """Toggle each filter/condition for ablation testing.

    Baseline: all True.  Set one to False to measure its contribution.
    """

    momentum_filter: bool = True          # 1
    conviction_gating: bool = True        # 2
    fast_confirm: bool = True             # 3
    reset_requirement: bool = True        # 4
    voucher_system: bool = True           # 5
    cooldown: bool = True                 # 6
    slippage_abort: bool = True           # 7
    prior_high_confirm: bool = True       # 8
    short_safety: bool = True             # 9
    time_decay: bool = True              # 10
    addon_a: bool = True                 # 11
    addon_b: bool = True                 # 12
    breakout_entries: bool = True        # 13
    hysteresis_gap: bool = True          # 14
    stall_exit: bool = True             # 15
    early_stall_exit: bool = True      # 16
    quality_gate: bool = False          # 17 (disabled: score doesn't predict trade quality)


@dataclass
class BacktestConfig:
    """Top-level backtest configuration."""

    symbols: list[str] = field(default_factory=_default_atrss_symbols)
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 100_000.0
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    flags: AblationFlags = field(default_factory=AblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    track_shadows: bool = True
    warmup_daily: int = 60
    warmup_hourly: int = 55
    fixed_qty: int | None = 10
