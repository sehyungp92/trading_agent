"""Downturn Dominator v1 -- R7c production configuration.

Exports consumed by the coordinator and engine:
  - STRATEGY_ID, BASE_RISK_PCT, DAILY_STOP_R, HEAT_CAP_R, PORTFOLIO_DAILY_STOP_R
  - MAX_LEVERAGE_MULT (per-strategy leverage ceiling for engine)
  - build_instruments() -> dict[str, Instrument]
  - R7C_FLAGS: DownturnAblationFlags   (backtest dataclass, R7c values)
  - R7C_PARAM_OVERRIDES: dict[str, float]

Authoritative source:
  the centralized round baseline under
  `backtests/output/momentum/downturn/round_4/optimized_config.json`
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry


@dataclass
class DownturnAblationFlags:
    """Toggle each gate/mechanism for ablation testing.

    Baseline: all ``True`` (enabled) except deferred features.
    """

    # Regime & Volatility
    use_composite_regime: bool = True
    use_volatility_states: bool = True
    use_shock_block: bool = True
    use_strong_bear_bonus: bool = True

    # Engine enables
    reversal_engine: bool = True
    breakdown_engine: bool = True
    fade_engine: bool = True

    # Reversal gates
    reversal_divergence_gate: bool = True
    reversal_trend_weakness_gate: bool = True  # 2-of-3 gate
    reversal_extension_gate: bool = True
    reversal_corridor_cap: bool = True

    # Breakdown gates
    breakdown_containment_gate: bool = True
    breakdown_displacement_gate: bool = True
    breakdown_chop_filter: bool = True
    breakdown_spike_reject: bool = True

    # Fade gates
    fade_vwap_rejection: bool = True
    fade_momentum_confirm: bool = True
    fade_bear_regime_required: bool = True
    fade_cap_gate: bool = True

    # Entry common
    use_entry_windows: bool = True
    use_dead_zones: bool = True
    use_news_blackout: bool = True

    # Exit / Position management
    tiered_exits: bool = True
    chandelier_trailing: bool = True
    stale_exit: bool = True
    climax_exit: bool = True
    vwap_failure_exit: bool = True  # Fade pre-+1R

    # Risk
    daily_circuit_breaker: bool = True
    directional_entry_caps: bool = True
    friction_gate: bool = True

    # Regime filters
    block_counter_regime: bool = False    # Block all entries during COUNTER regime
    correction_sizing_bonus: bool = False  # Boost sizing during correction windows
    non_correction_penalty: bool = False   # Reduce sizing outside correction windows

    # Structural gap fixes
    correction_regime_override: bool = False  # Override NEUTRAL/RANGE -> EMERGING_BEAR during correction windows
    short_sma_trend: bool = False             # Use short SMA (e.g. 50) as alternative bear signal
    allow_reversal_strong_bear: bool = False  # Allow reversal signals during strong bear
    stale_to_tp: bool = False                 # Tag profitable stale exits as tp0 instead of stale

    # Exit enhancements
    profit_floor_trail: bool = False  # Lock fraction of profit after threshold R
    multi_tier_profit_floor: bool = False  # 5-tier MFE-based profit floor ratchet
    regime_adaptive_chandelier: bool = False  # Regime multiplier on chandelier trailing

    # Fast-crash override (paths E/F/G)
    fast_crash_override: bool = False  # Override regime to EMERGING_BEAR on real-time price drops
    conviction_scoring: bool = False   # Bear conviction quality gate (0-100) for overrides

    # Bear structure override (paths B/C + BEAR_FORMING)
    bear_structure_override: bool = False  # ADX hysteresis + gradual bear detection

    # Correction-specific gates
    allow_reversal_in_correction: bool = False  # Exempt reversal from block_counter_regime in corrections

    # Scale-out
    scale_out_enabled: bool = False  # Partial profit lock at target R

    # R6 -- Adaptive exits
    adaptive_profit_floor: bool = False  # MFE-tiered lock_pct (higher capture from big winners)

    # R6 -- Coverage expansion
    drawdown_regime_override: bool = False  # Real-time rolling-high drawdown -> EMERGING_BEAR override
    progressive_sma: bool = True

    # R6 -- Signal expansion
    momentum_signal: bool = False  # ROC-based fade alternative (no VWAP rejection needed)

    # R6 -- Hold period
    min_hold_period: bool = False  # Skip exits for first N bars after entry (except catastrophic)

    # R8 -- Intraday regime proxy (Category A)
    four_hour_only_regime: bool = False       # Regime from 4H ADX + 4H EMA only (drop daily)
    intraday_regime_proxy: bool = False       # 1H EMA as bear/bull proxy
    intraday_regime_ema_period: int = 100     # EMA period for intraday proxy (50 or 100)
    multi_tf_regime_vote: bool = False        # 2-of-3 vote: 1H + 4H + daily
    regime_proxy_atr_expansion: bool = False  # 1H ATR expansion as stress signal
    correction_intraday_detect: bool = False  # Intraday cumulative decline for corrections

    # R8 -- Reversal engine revival (Category B)
    reversal_min_gate_count: int = 2          # 2-of-3 -> 1-of-3 when set to 1
    reversal_no_extension_gate: bool = False  # Remove extension gate
    reversal_hourly_pivots: bool = False      # Use 1H pivots instead of 4H
    reversal_wider_corridor: float = 0.0      # Override corridor_cap_mult (0=default)

    # R8 -- Entry filters (Category C)
    correction_only_mode: bool = False        # Block ALL entries outside correction windows
    correction_only_fade: bool = False        # Block fade entries outside corrections
    vol_percentile_gate: float = 0.0          # ATR percentile minimum at entry (0=disabled)
    regime_confidence_gate: float = 0.0       # Conviction score minimum at entry (0=disabled)

    # R8 -- Exit improvements (Category D)
    wider_initial_stop_mult: float = 0.0      # Scale initial stop by this mult (0=default)
    atr_scaled_initial_stop: bool = False     # Scale initial stop by ATR percentile
    partial_at_breakeven: float = 0.0         # Take this fraction off at BE trigger (0=disabled)
    time_stop_widening: bool = False          # Widen chandelier after N bars without progress
    time_stop_widening_bars: int = 48         # Bar threshold for widening

    # Deferred (optimize later)
    earn_the_hold: bool = False
    adds_enabled: bool = False
    breakdown_reentry: bool = False
    fade_reentry: bool = False

# ---------------------------------------------------------------------------
# Strategy identity
# ---------------------------------------------------------------------------
STRATEGY_ID = "DownturnDominator_v1"

# ---------------------------------------------------------------------------
# Contract specifications (MNQ primary, NQ for reference)
# ---------------------------------------------------------------------------
NQ_SPECS = {
    "NQ":  {"tick": 0.25, "tick_value": 5.00, "point_value": 20.00},
    "MNQ": {"tick": 0.25, "tick_value": 0.50, "point_value":  2.00},
}
DEFAULT_SYMBOL = "MNQ"

# ---------------------------------------------------------------------------
# Risk -- coordinator-level exports
# ---------------------------------------------------------------------------
BASE_RISK_PCT = 0.004
DAILY_STOP_R = 2.0
HEAT_CAP_R = 3.5               # max simultaneous heat
PORTFOLIO_DAILY_STOP_R = 2.75
MAX_LEVERAGE_MULT = 20.0       # R7c research-validated leverage ceiling for MNQ

# ---------------------------------------------------------------------------
# R7c ablation flags (16 changes from defaults)
# ---------------------------------------------------------------------------
R7C_FLAGS = DownturnAblationFlags(
    # --- Disable (default True -> False) ---
    breakdown_engine=False,         # 0 trades; never fires
    tiered_exits=False,             # 0% TP hit rate
    use_volatility_states=False,    # interferes with regime overrides
    # --- Enable (default False -> True) ---
    profit_floor_trail=True,
    adaptive_profit_floor=True,
    regime_adaptive_chandelier=True,
    min_hold_period=True,
    correction_regime_override=True,
    fast_crash_override=True,
    conviction_scoring=True,
    bear_structure_override=True,
    drawdown_regime_override=True,
    block_counter_regime=True,
    short_sma_trend=True,
    correction_sizing_bonus=True,
    momentum_signal=True,
)

# ---------------------------------------------------------------------------
# R7c parameter overrides (25 overrides)
# ---------------------------------------------------------------------------
R7C_PARAM_OVERRIDES: dict[str, float] = {
    # Profit floor
    "profit_floor_r_threshold": 1.5,
    "profit_floor_lock_pct": 0.60,
    # Adaptive profit floor
    "adaptive_lock_t1": 1.5,
    "adaptive_lock_bonus_1": 0.15,
    # Chandelier
    "chandelier_lookback": 24,
    "chandelier_mult_floor": 2.2,
    "chandelier_mult_ceiling": 4.5,
    # Breakeven
    "be_trigger_r": 0.9,
    "be_stop_buffer_mult": 0.08,
    # Min hold
    "min_hold_bars": 12,
    # Fast crash
    "crash_daily_threshold": -0.018,
    # Conviction
    "conviction_threshold": 35,
    # VWAP caps
    "vwap_cap_core": 0.45,
    "vwap_cap_extended": 0.77,
    # Divergence
    "divergence_mag_threshold": 0.1,
    # Indicators
    "ema_fast_period": 20,
    "sma200_period": 250,
    "short_sma_period": 40,
    "adx_trending_threshold": 22,
    "adx_range_threshold": 18,
    # Sizing
    "base_risk_pct": 0.0064,
    "regime_mult_emerging": 1.8,
    "correction_sizing_mult": 1.2,
    # Circuit breaker
    "circuit_breaker_threshold": -2400,
    # Momentum
    "momentum_cooldown_bars": 24,
    "drawdown_lookback": 10,
    "regime_mult_counter": 0.2,
    "tp1_r_aligned": 2.4,
    "tp1_r_emerging": 1.8,
    "momentum_roc_threshold": -0.003,
    "progressive_sma_min": 100,
    "friction_min_atr_pctl": 0.05,
    "entry_buffer_ticks": 1,
}


# ---------------------------------------------------------------------------
# Instrument builder (coordinator pattern)
# ---------------------------------------------------------------------------
def build_instruments() -> dict[str, Instrument]:
    """Register NQ / MNQ instruments with the OMS."""
    instruments: dict[str, Instrument] = {}
    for sym, spec in NQ_SPECS.items():
        inst = Instrument(
            symbol=sym,
            root=sym,
            venue="CME",
            tick_size=spec["tick"],
            tick_value=spec["tick_value"],
            multiplier=spec["point_value"],
            contract_expiry="",
            sec_type="FUT",
            trading_class=sym,
        )
        InstrumentRegistry.register(inst)
        instruments[sym] = inst
    return instruments
