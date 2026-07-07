"""Unified multi-strategy backtest configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
from backtests.swing.config_etf_base import ETFSlippageConfig
from backtests.swing.config_helix import HelixAblationFlags, HelixBacktestConfig
from backtests.swing.config_tpc import TPCBacktestConfig


@dataclass
class StrategySlot:
    """Per-strategy portfolio-level parameters (mirrors main_multi.py)."""

    strategy_id: str
    priority: int            # 0 = highest
    unit_risk_pct: float     # fraction of NAV per 1R
    max_heat_R: float        # per-strategy heat ceiling in R
    daily_stop_R: float      # per-strategy daily stop in R (positive value)
    max_working_orders: int = 4


# Default strategy slots — P1 heat-unlock optimized (Mar 14).
# Raised heat_cap 2.0→3.0 unlocked 682 blocked entries, +54% total PnL,
# Sharpe 1.24→1.52, ratio 72:28→58:42 overlay:active.
# ATRSS(0): highest expectancy, always gets first fill.
# Helix(1): second priority after ATRSS.
ATRSS_SLOT = StrategySlot(
    strategy_id="ATRSS", priority=0,
    unit_risk_pct=0.018, max_heat_R=1.50, daily_stop_R=2.0,
    max_working_orders=4,
)
HELIX_SLOT = StrategySlot(
    strategy_id="AKC_HELIX", priority=1,
    unit_risk_pct=0.008, max_heat_R=1.20, daily_stop_R=2.5,
    max_working_orders=4,
)
TPC_SLOT = StrategySlot(
    strategy_id="TPC", priority=2,
    unit_risk_pct=0.005, max_heat_R=1.00, daily_stop_R=2.0,
    max_working_orders=3,
)
@dataclass
class UnifiedBacktestConfig:
    """Top-level config for the active unified swing portfolio backtest."""

    initial_equity: float = 10_000.0
    data_dir: Path = field(default_factory=lambda: Path("backtest/data/raw"))
    start_date: str | None = None  # "YYYY-MM-DD" — trim data before this date
    end_date: str | None = None    # "YYYY-MM-DD" — trim data after this date
    slippage: SlippageConfig = field(default_factory=SlippageConfig)

    # Symbol lists per strategy (defaults match production)
    atrss_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    helix_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    tpc_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])

    # Portfolio-level risk rules
    heat_cap_R: float = 3.0
    portfolio_daily_stop_R: float = 4.0
    portfolio_constraints_enabled: bool = True
    dynamic_risk_enabled: bool = False
    drawdown_risk_tiers: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: (
            (0.06, 1.00),
            (0.09, 0.70),
            (0.12, 0.40),
            (0.15, 0.00),
        )
    )
    atrss: StrategySlot = field(default_factory=lambda: ATRSS_SLOT)
    helix: StrategySlot = field(default_factory=lambda: HELIX_SLOT)
    tpc: StrategySlot = field(default_factory=lambda: TPC_SLOT)

    # Cross-strategy coordination
    enable_atrss_helix_tighten: bool = True
    enable_atrss_helix_size_boost: bool = True

    # Warmup periods
    warmup_daily: int = 60
    warmup_hourly: int = 55
    warmup_4h: int = 50
    warmup_15m: int = 2_000

    # Position sizing — fixed_qty=10 for ETFs (matches individual runners)
    fixed_qty: int | None = None

    # Idle-capital overlay (EMA crossover on daily closes)
    overlay_enabled: bool = True
    overlay_mode: str = "ema"            # "ema" (legacy) or "multi" (EMA+RSI+MACD)
    overlay_ema_fast: int = 13           # fast EMA period
    overlay_ema_slow: int = 48           # slow EMA period
    overlay_symbols: list[str] = field(default_factory=lambda: ["QQQ", "GLD"])
    overlay_ema_overrides: dict[str, tuple[int, int]] = field(default_factory=lambda: {"QQQ": (10, 21), "GLD": (13, 21)})
    # Maps symbol -> (fast_period, slow_period). Symbols not in dict use overlay_ema_fast/slow defaults.
    overlay_max_pct: float = 0.85        # max fraction of equity for overlay
    overlay_weights: dict[str, float] | None = None
    # Per-symbol relative weights for overlay capital allocation.
    # None = equal-weight (current behavior). Renormalized across bullish symbols only.

    # Multi-overlay RSI parameters
    overlay_rsi_period: int = 14
    overlay_rsi_overbought: float = 70.0
    overlay_rsi_bull_min: float = 40.0
    overlay_rsi_overrides: dict[str, dict] = field(default_factory=dict)
    # Per-symbol RSI overrides: {"QQQ": {"period": 10, "overbought": 75}} etc.

    # Multi-overlay MACD parameters
    overlay_macd_fast: int = 12
    overlay_macd_slow: int = 26
    overlay_macd_signal: int = 9
    overlay_macd_overrides: dict[str, dict] = field(default_factory=dict)
    # Per-symbol MACD overrides: {"QQQ": {"fast": 8, "slow": 21, "signal": 7}} etc.

    # Multi-overlay scoring and sizing
    overlay_entry_score_min: float = 0.6   # minimum composite score to enter
    overlay_exit_score_max: float = 0.3    # exit below this (asymmetric hysteresis)
    overlay_adaptive_sizing: bool = True
    overlay_min_alloc_pct: float = 0.30    # min allocation pct when adaptive
    overlay_max_alloc_pct: float = 1.00    # max allocation pct when adaptive

    # Multi-overlay score component weights (EMA, RSI, MACD) — must sum to 1.0
    overlay_score_weights: tuple[float, float, float] = (0.40, 0.30, 0.30)
    # EMA spread normalization: spread/norm capped at 1.0. Lower = more sensitive
    overlay_ema_spread_norm: float = 0.01
    # MACD histogram score mapping: (pos_rising, pos_falling, neg_falling, neg_rising)
    overlay_macd_scores: tuple[float, float, float, float] = (1.0, 0.6, 0.0, 0.3)

    # Simulate live OMS R normalization bug: use per-strategy URD for
    # portfolio R sums instead of consistent portfolio base. This makes the
    # backtest match live's more conservative behavior for impact measurement.
    simulate_live_r_normalization: bool = False
    reserve_idle_higher_priority: bool = True

    # Per-strategy per-symbol risk multipliers.
    # Maps "STRATEGY_ID:SYMBOL" -> multiplier that scales base_risk_pct.
    # e.g. {"ATRSS:QQQ": 0.8, "ATRSS:GLD": 1.2}. Missing entries default to 1.0.
    symbol_risk_multipliers: dict[str, float] = field(default_factory=dict)

    # Per-strategy ablation flags (baseline = all enabled)
    atrss_flags: AblationFlags = field(default_factory=lambda: AblationFlags(stall_exit=False))
    helix_flags: HelixAblationFlags = field(default_factory=HelixAblationFlags)

    # Per-strategy engine param overrides (applied by build_*_config methods)
    # These allow greedy optimizers to route strategy-specific params through
    # the unified config without modifying per-strategy config classes.
    atrss_param_overrides: dict = field(default_factory=dict)
    helix_param_overrides: dict = field(default_factory=dict)
    tpc_param_overrides: dict = field(default_factory=dict)

    def build_atrss_config(self) -> BacktestConfig:
        slippage = self.slippage
        if self.fixed_qty is not None:
            slippage = SlippageConfig(commission_per_contract=1.00)
        elif slippage.commission_per_contract == SlippageConfig().commission_per_contract:
            slippage = replace(slippage, commission_per_contract=1.00)
        cfg = BacktestConfig(
            symbols=self.atrss_symbols,
            initial_equity=self.initial_equity,
            slippage=slippage,
            flags=self.atrss_flags,
            data_dir=self.data_dir,
            track_shadows=False,
            warmup_daily=self.warmup_daily,
            warmup_hourly=self.warmup_hourly,
            fixed_qty=self.fixed_qty,
        )
        if self.atrss_param_overrides:
            merged = {**cfg.param_overrides, **self.atrss_param_overrides}
            cfg = replace(cfg, param_overrides=merged)
        return cfg

    def build_helix_config(self) -> HelixBacktestConfig:
        slippage = self.slippage
        if self.fixed_qty is not None:
            slippage = SlippageConfig(commission_per_contract=1.00)
        cfg = HelixBacktestConfig(
            symbols=self.helix_symbols,
            initial_equity=self.initial_equity,
            slippage=slippage,
            flags=self.helix_flags,
            data_dir=self.data_dir,
            track_shadows=False,
            warmup_daily=self.warmup_daily,
            warmup_hourly=self.warmup_hourly,
            warmup_4h=self.warmup_4h,
            fixed_qty=self.fixed_qty,
        )
        if self.helix_param_overrides:
            merged = {**cfg.param_overrides, **self.helix_param_overrides}
            cfg = replace(cfg, param_overrides=merged)
        return cfg

    def _etf_slippage(self) -> ETFSlippageConfig:
        return ETFSlippageConfig()

    def build_tpc_config(self) -> TPCBacktestConfig:
        cfg = TPCBacktestConfig(
            symbols=tuple(self.tpc_symbols),
            initial_equity=self.initial_equity,
            data_dir=self.data_dir,
            warmup_15m=self.warmup_15m,
            slippage=self._etf_slippage(),
        )
        return cfg.with_overrides(self.tpc_param_overrides) if self.tpc_param_overrides else cfg

# ---------------------------------------------------------------------------
# Preset factory functions — each returns a fully configured UnifiedBacktestConfig
# with risk-based sizing (fixed_qty=None).
# ---------------------------------------------------------------------------

def _slot(sid: str, priority: int, urp: float, mh: float, ds: float,
          mwo: int = 4) -> StrategySlot:
    """Shorthand for building a StrategySlot."""
    return StrategySlot(
        strategy_id=sid, priority=priority,
        unit_risk_pct=urp, max_heat_R=mh, daily_stop_R=ds,
        max_working_orders=mwo,
    )


def make_baseline(equity: float) -> UnifiedBacktestConfig:
    """Pre-optimized production parameters: ATRSS(0) > Helix(1)."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.005, 0.85, 2.5),
        overlay_ema_overrides={},
    )


def make_a1_atrss_tilt(equity: float) -> UnifiedBacktestConfig:
    """A1: Increase ATRSS allocation (best expectancy), reduce others."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.015, 1.2, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.003, 0.5, 2.5),
        overlay_ema_overrides={},
    )


def make_a2_equal_alloc(equity: float) -> UnifiedBacktestConfig:
    """A2: Equal allocation — all strategies get same risk budget."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.0075, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.0075, 1.0, 2.5),
        overlay_ema_overrides={},
    )


def make_b1_tight_heat(equity: float) -> UnifiedBacktestConfig:
    """B1: Tight heat — reduce portfolio heat cap to force selective entry."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.0,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 0.6, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.005, 0.5, 2.5),
        overlay_ema_overrides={},
    )


def make_b2_expanded_heat(equity: float) -> UnifiedBacktestConfig:
    """B2: Expanded heat — allow more concurrent positions."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=2.0,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.2, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.005, 1.0, 2.5),
        overlay_ema_overrides={},
    )


def make_c1_old_priority(equity: float) -> UnifiedBacktestConfig:
    """C1: Legacy Helix priority preset for comparison."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.005, 0.85, 2.5),
        overlay_ema_overrides={},
    )


def make_d1_no_coordination(equity: float) -> UnifiedBacktestConfig:
    """D1: Disable tighten + boost to measure coordination net impact."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.005, 0.85, 2.5),
        enable_atrss_helix_tighten=False,
        enable_atrss_helix_size_boost=False,
        overlay_ema_overrides={},
    )


def make_e1_tighter_daily_stops(equity: float) -> UnifiedBacktestConfig:
    """E1: Tighter daily stops — conservative, better for small accounts."""
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=1.5,
        portfolio_daily_stop_R=2.5,
        atrss=_slot("ATRSS", 0, 0.01, 1.0, 1.5),
        helix=_slot("AKC_HELIX", 1, 0.005, 0.85, 2.0),
        overlay_ema_overrides={},
    )


def make_optimized_v1(equity: float) -> UnifiedBacktestConfig:
    """Optimized v1: per-asset EMAs (QQQ 10/21, GLD 13/21) + ATRSS risk boost to 1.2%.

    Pinned to pre-P1 defaults for backward compatibility.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=2.0,
        portfolio_daily_stop_R=3.0,
        atrss=_slot("ATRSS", 0, 0.012, 1.0, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.005, 0.85, 2.5),
    )


def make_multi_overlay(equity: float) -> UnifiedBacktestConfig:
    """Multi-overlay: EMA+RSI+MACD scoring with adaptive sizing on QQQ/GLD (optimized).

    Uses the optimized_v1 active strategy parameters with the enhanced
    multi-indicator overlay for idle capital deployment.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        overlay_enabled=True,
        overlay_mode="multi",
        overlay_symbols=["QQQ", "GLD"],
        overlay_ema_overrides={"QQQ": (10, 21), "GLD": (13, 21)},
        overlay_weights={"QQQ": 0.50, "GLD": 0.50},
        overlay_rsi_period=14,
        overlay_rsi_overbought=70.0,
        overlay_rsi_bull_min=40.0,
        overlay_macd_fast=12,
        overlay_macd_slow=26,
        overlay_macd_signal=9,
        overlay_entry_score_min=0.6,
        overlay_exit_score_max=0.3,
        overlay_adaptive_sizing=True,
        overlay_min_alloc_pct=0.30,
        overlay_max_alloc_pct=1.00,
        overlay_score_weights=(0.40, 0.30, 0.30),
        overlay_macd_scores=(1.0, 0.6, 0.0, 0.3),
    )


def make_live_parity(equity: float) -> UnifiedBacktestConfig:
    """Live parity: greedy v4 optimized defaults (Mar 25).

    Uses default StrategySlot values:
      ATRSS(0)  URD 1.8%  max_heat 1.50R  daily_stop 2.0R
      Helix(1)  URD 0.8%  max_heat 1.20R  daily_stop 2.5R
      heat_cap_R=3.0, portfolio_daily_stop_R=4.0
    Greedy v4 mutation retained here: ATRSS stall_exit disabled.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        atrss_flags=AblationFlags(stall_exit=False),  # greedy v4
    )


def make_live_r_simulation(equity: float) -> UnifiedBacktestConfig:
    """Simulate live OMS R normalization bug for impact measurement.

    Uses current live_parity defaults with per-strategy URD for portfolio
    R sums instead of consistent portfolio base.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        simulate_live_r_normalization=True,
    )


def make_p1_heat_unlock(equity: float) -> UnifiedBacktestConfig:
    """P1: Heat unlock — now the default configuration (Mar 14).

    Winner of P1-P4 optimization sweep: +54% PnL, Sharpe 1.24→1.52,
    ratio 72:28→58:42, 952→270 blocked entries. Kept as a named preset
    for backward compatibility — identical to live_parity/defaults.
    """
    return UnifiedBacktestConfig(initial_equity=equity)


def make_p2_aggressive_active(equity: float) -> UnifiedBacktestConfig:
    """P2: Maximum active push — highest risk on proven strategies.

    Pushes active allocation to the limit with aggressive sizing on
    high-expectancy strategies and moderate Helix increase.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=3.5,
        portfolio_daily_stop_R=4.5,
        atrss=_slot("ATRSS", 0, 0.020, 1.8, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.006, 1.0, 2.5),
        symbol_risk_multipliers={
            "ATRSS:QQQ": 1.3,
            "ATRSS:GLD": 1.3,
        },
    )


def make_p3_balanced_split(equity: float) -> UnifiedBacktestConfig:
    """P3: Approach 50:50 from both sides — moderate active boost + overlay reduction.

    Reduces overlay_max_pct to constrain overlay capital while moderately
    increasing active risk. May reduce total PnL — included as a data point.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=2.5,
        portfolio_daily_stop_R=3.5,
        atrss=_slot("ATRSS", 0, 0.015, 1.2, 2.0),
        helix=_slot("AKC_HELIX", 1, 0.007, 1.0, 2.5),
        overlay_max_pct=0.65,
    )


def make_p4_multiplier_boost(equity: float) -> UnifiedBacktestConfig:
    """P4: Symbol risk multipliers + heat unlock — selective high-edge boosts.

    Uses symbol_risk_multipliers to selectively scale risk on proven
    strategy:symbol pairs instead of blanket increases.
    """
    return UnifiedBacktestConfig(
        initial_equity=equity,
        heat_cap_R=3.0,
        portfolio_daily_stop_R=4.0,
        atrss=_slot("ATRSS", 0, 0.015, 1.5, 2.0),
        symbol_risk_multipliers={
            "ATRSS:QQQ": 1.5,
            "ATRSS:GLD": 1.3,
            "AKC_HELIX:GLD": 1.2,
        },
    )


PRESETS: dict[str, Callable[[float], UnifiedBacktestConfig]] = {
    "live_parity": make_live_parity,
    "live_r_simulation": make_live_r_simulation,
    "baseline": make_baseline,
    "a1_atrss_tilt": make_a1_atrss_tilt,
    "a2_equal_alloc": make_a2_equal_alloc,
    "b1_tight_heat": make_b1_tight_heat,
    "b2_expanded_heat": make_b2_expanded_heat,
    "c1_old_priority": make_c1_old_priority,
    "d1_no_coordination": make_d1_no_coordination,
    "e1_tighter_daily_stops": make_e1_tighter_daily_stops,
    "optimized_v1": make_optimized_v1,
    "multi_overlay": make_multi_overlay,
    "p1_heat_unlock": make_p1_heat_unlock,
    "p2_aggressive_active": make_p2_aggressive_active,
    "p3_balanced_split": make_p3_balanced_split,
    "p4_multiplier_boost": make_p4_multiplier_boost,
}
