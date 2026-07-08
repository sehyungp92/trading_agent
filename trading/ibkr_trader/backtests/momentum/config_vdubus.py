"""VdubusNQ v4.0 backtest configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backtests.momentum.config import SlippageConfig


@dataclass
class VdubusAblationFlags:
    """Toggle each gate/mechanism for VdubusNQ ablation testing.

    Baseline: all ``True`` (enabled).
    Set one to ``False`` to measure its contribution.
    """

    # Regime / volatility gates
    daily_trend_gate: bool = True
    shock_block: bool = True
    hourly_alignment: bool = True

    # Entry structure
    type_a_enabled: bool = True
    type_b_enabled: bool = True
    vwap_cap_gate: bool = True
    extension_sanity: bool = True
    touch_lookback_gate: bool = True

    # Confirmation
    slope_gate: bool = True
    predator_overlay: bool = True
    momentum_floor: bool = True

    # Risk / execution
    min_max_stop: bool = True
    ttl_cancel: bool = True
    teleport_skip: bool = True
    fallback_market: bool = True
    viability_filter: bool = True
    direction_caps: bool = True
    heat_cap: bool = True

    # Exits / hold
    vwap_failure_exit: bool = True
    stale_exit: bool = True
    plus_1r_partial: bool = True
    decision_gate: bool = True
    overnight_widening: bool = True
    vwap_a_failure: bool = True
    friday_override: bool = True

    # Phase 1 improvements
    free_ride_stale: bool = True          # ACTIVE_FREE time-based exit (1.1)
    free_profit_lock: bool = True         # ACTIVE_FREE profit lock (1.1)
    max_duration: bool = True             # hard max position duration (1.3)
    post_partial_trail_tighten: bool = True  # tighter trail after partial (1.2)

    # Risk management
    drawdown_throttle: bool = True         # DD-based sizing reduction + daily loss cap

    # Event safety
    event_blocking: bool = True

    # v4.0 improvements
    expanded_dead_zone: bool = True        # wider midday dead zone (10:45-15:00)
    choppiness_gate: bool = True           # reduce entries in choppy regimes
    early_kill: bool = True                # kill fast-dying trades early
    vwap_fail_evening: bool = False        # skip VWAP_FAIL for evening entries (stale VWAP)

    # v4.1 improvements
    dow_sizing: bool = False               # day-of-week sizing reduction
    entry_quality_gate: bool = False       # EQS pre-filter on entries
    evening_vwap_cap: bool = False         # separate tighter VWAP cap for evening

    # v4.2 improvements
    block_20h_hour: bool = True            # block entries during 20:00 ET hour (evening)
    adaptive_stale: bool = False           # per-window stale timer (CLOSE=12, EVENING=5) — REVERTED: -0.12 Sharpe, -$5.7k
    close_skip_partial: bool = True        # CLOSE entries skip +1R partial + 0.85R ultra-tight ratchet
    mfe_ratchet: bool = False              # progressive MFE floor stops — REVERTED: neutral (+$324, -0.01 Sharpe)
    bar_quality_gate: bool = False         # filter spike bars, weak closes, single-bar reclaims — REVERTED: Sharpe 0.64, trades 91, catastrophic

    # v4.3 improvements
    stale_mfe_exempt: bool = False         # exempt from stale exit if peak MFE > threshold

    # v4.4 improvements
    late_trail: bool = False               # independent late-activation trail (no partial)

    # v4.5 research paths (disabled by default; shared-signal compatible)
    hourly_bypass_quality: bool = False    # allow select non-1H-aligned entries through quality gate
    slope_bypass_quality: bool = False     # allow select slope rejects through quality gate
    type_c_enabled: bool = False           # continuation entry for no-signal shadow alpha
    mfe_rescue_stop: bool = False          # protect trades that show early MFE then stall


@dataclass
class VdubusBacktestConfig:
    """Top-level VdubusNQ v4.0 backtest configuration."""

    symbols: list[str] = field(default_factory=lambda: ["NQ"])
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_equity: float = 100_000.0
    slippage: SlippageConfig = field(
        default_factory=lambda: SlippageConfig(
            commission_per_contract=0.62,  # MNQ per side
        ),
    )
    flags: VdubusAblationFlags = field(default_factory=VdubusAblationFlags)
    param_overrides: dict[str, float] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    track_signals: bool = True
    track_shadows: bool = True
    fixed_qty: int | None = None
    news_calendar_path: Path | None = None

    # Warmup periods (in bars of each timeframe)
    warmup_daily_es: int = 260  # SMA200 needs 200+ bars
    warmup_1h: int = 55
    warmup_15m: int = 200
    warmup_5m: int = 600

    # MNQ instrument defaults (comparable basis with other strategies)
    tick_size: float = 0.25
    point_value: float = 2.0
