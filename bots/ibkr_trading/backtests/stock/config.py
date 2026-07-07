"""Shared backtest configuration dataclasses for stock strategies."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageConfig:
    """Execution simulation parameters for US equities."""

    commission_per_share: float = 0.005       # IBKR tiered for US stocks
    slip_bps_normal: float = 5.0              # 5bps typical S&P 500
    slip_bps_illiquid: float = 15.0           # wider for thin names
    spread_bps_default: float = 10.0          # median spread estimate
    halt_zero_range_bars: int = 2             # consecutive zero-range bars -> halt
    halt_extra_slip_bps: float = 25.0         # additional slippage on post-halt reopen


@dataclass(frozen=True)
class UniverseConfig:
    """Universe selection parameters."""

    use_sp500_base: bool = True               # SP500 base universe
    use_backtested_intraday_universe: bool = True  # stock replay/backtests use the focused intraday cohort
    include_scanner_supplement: bool = False   # scanner supplement (live only)
    min_adv20_usd: float = 5_000_000.0        # minimum 20-day ADV in USD
    min_price: float = 5.0                    # minimum stock price
    max_price: float = 1_000.0                # maximum stock price
    default_spread_pct: float = 0.10          # hardcoded spread estimate
    default_earnings_sessions: int = 99       # never block (no calendar)
