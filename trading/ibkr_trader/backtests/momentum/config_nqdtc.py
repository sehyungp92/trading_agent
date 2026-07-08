"""NQDTC v2.0 backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.momentum.config import SlippageConfig


@dataclass
class NQDTCAblationFlags:
    """Toggle each gate/mechanism for NQDTC ablation testing.

    Baseline: all ``True`` (enabled).
    Set one to ``False`` to measure its contribution.
    """

    # Qualification gates
    displacement_threshold: bool = True
    score_threshold: bool = True
    breakout_quality_reject: bool = True

    # Mechanism toggles
    dirty_mechanism: bool = True
    continuation_mode: bool = True
    chop_halt: bool = True
    chop_degraded: bool = True

    # Risk gates
    news_blackout: bool = True
    friction_gate: bool = True
    micro_guard: bool = True
    daily_stop: bool = True
    weekly_stop: bool = True
    monthly_stop: bool = True
    drawdown_throttle: bool = True         # DD-based sizing reduction (no daily cap — uses existing daily_stop)

    # Entry subtypes
    entry_a_retest: bool = True
    entry_a_latch: bool = True
    entry_b_sweep: bool = True
    entry_c_standard: bool = True
    entry_c_continuation: bool = False  # nqdtc_v4 step 1: 5 trades, 40% WR, -0.341 avg R

    # Session gating
    rth_entries: bool = True  # enable RTH entries for increased frequency

    # Exit / position management
    tiered_exits: bool = True
    chandelier_trailing: bool = True
    stale_exit: bool = True
    profit_funded_be: bool = True
    overnight_bridge: bool = True
    max_loss_cap: bool = False              # Force exit at -3R (superseded by min_stop_distance)
    min_stop_distance: float = 3.0          # Reject entries with stop < 3 pts (pathological stop filter)
    early_chandelier: bool = True           # Activate chandelier immediately after TP1
    max_stop_width: bool = False            # Reject entries with stop distance > 200 pts (kills A_retest edge)
    loss_streak_cooldown: bool = True       # Skip 1 entry after 3 consecutive losses
    block_05_et: bool = True                # Block entries during 05:00 ET hour
    block_04_et: bool = True                # Block entries during 04:00 ET hour
    block_06_et: bool = True                # Block entries during 06:00 ET hour (P5: pre-European-open, WR=39%)
    block_09_et: bool = True                # Block entries during 09:00 ET hour (RTH open whipsaw)
    block_12_et: bool = True                # Block entries during 12:00 ET hour (17% WR, outlier-dependent)
    block_thursday: bool = False             # Tested: -54 trades, -$11.6k, Sharpe -0.03
    early_be: bool = False                   # Tested: 0.8R Sharpe->2.23, 1.0R->2.16 (both worse)
    es_daily_trend: bool = False             # ES SMA200 directional sizing: reduce size when opposing ES trend
    block_eth_shorts: bool = False           # Block ETH short entries (37% WR, +0.176R)


@dataclass
class NQDTCBacktestConfig:
    """Top-level NQDTC v2.0 backtest configuration."""

    symbols: list[str] = field(default_factory=lambda: ["NQ"])
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 100_000.0
    slippage: SlippageConfig = field(
        default_factory=lambda: SlippageConfig(
            commission_per_contract=0.62,  # MNQ per side
        ),
    )
    flags: NQDTCAblationFlags = field(default_factory=NQDTCAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    track_signals: bool = True
    track_shadows: bool = True
    fixed_qty: int | None = None
    news_calendar_path: Path | None = None

    # Warmup periods (in bars of each timeframe)
    warmup_daily: int = 60
    warmup_30m: int = 100
    warmup_1h: int = 55
    warmup_4h: int = 55
    warmup_5m: int = 200

    # MNQ instrument defaults (comparable basis with other strategies)
    tick_size: float = 0.25
    point_value: float = 2.0

    # Performance optimization flags (for auto-optimization pipeline)
    scoring_mode: bool = False       # Skip post-run normalizations
    max_dd_abort: float = 0.0       # >0 enables early termination on drawdown
