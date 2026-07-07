"""Static configuration for the ALCB v1 strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from libs.oms.models.instrument import Instrument

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

STRATEGY_ID = "ALCB_v1"
STRATEGY_TYPE = "strategy_alcb"
PROXY_SYMBOLS = ("SPY", "QQQ", "IWM")

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Semiconductors": "SMH",
}


@dataclass(frozen=True)
class ScannerSettings:
    scan_codes: tuple[str, ...] = ("TOP_PERC_GAIN", "TOP_PERC_LOSE", "HOT_BY_VOLUME")
    instrument: str = "STK"
    location_code: str = "STK.US.MAJOR"
    rows_per_scan: int = 75
    above_price: float = 15.0
    above_volume: int = 250_000
    stock_type_filter: str = "CORP"


@dataclass(frozen=True)
class StrategySettings:
    premarket_start: time = time(7, 0)
    post_close_scan: time = time(16, 10)
    market_open: time = time(9, 30)
    first_30m_close: time = time(10, 0)
    entry_end: time = time(15, 30)
    close_block_start: time = time(15, 50)
    forced_flatten: time = time(15, 58)

    hot_max: int = 40
    active_monitoring_target: int = 35
    warm_poll_interval_s: int = 30
    cold_poll_interval_s: int = 120

    min_price: float = 10.0
    min_adv_usd: float = 10_000_000.0
    max_median_spread_pct: float = 0.0050
    max_friction_to_risk: float = 0.10
    earnings_block_days: int = 3
    allow_adrs: bool = False
    allow_biotech: bool = False

    atr_ratio_low: float = 0.75
    atr_ratio_high: float = 1.20
    box_length_low: int = 8
    box_length_mid: int = 12
    box_length_high: int = 18
    hysteresis_days: int = 3
    min_containment: float = 0.80
    max_squeeze_metric: float = 1.10
    squeeze_lookback: int = 60
    sq_good_quantile: float = 0.30
    sq_loose_quantile: float = 0.65

    base_q_disp: float = 0.70
    atr_expansion_q_disp_adj: float = -0.05
    breakout_reject_range_atr: float = 2.0
    breakout_reject_min_body_ratio: float = 0.25
    breakout_reject_max_wick_ratio: float = 0.55
    breakout_reject_min_rvol_d: float = 2.0
    # Opt 4: Breakout Volume Confirmation
    breakout_min_rvol_d: float = 1.0
    breakout_low_vol_disp_premium: float = 0.15

    # Opt 5: Enhanced Accumulation/Distribution Scoring
    accum_vol_trend_bonus: float = 0.15

    # Opt 6: Time-of-Day Entry Quality Adjustment
    early_entry_end: time = time(11, 30)
    late_entry_start: time = time(14, 0)
    early_entry_rvol_bonus_min: float = 1.5

    normal_intraday_score_min: int = 2
    degraded_intraday_score_min: int = 3
    intraday_rvol_min: float = 1.2
    # Opt 2: Graduated RVOL + Displacement Bonus
    intraday_rvol_strong: float = 2.0
    evidence_disp_bonus_threshold: float = 1.30

    base_risk_fraction: float = 0.00702
    volatile_base_risk_fraction: float = 0.0035
    daily_stop_r: float = 2.35
    heat_cap_r: float = 4.0
    portfolio_daily_stop_r: float = 3.5
    max_portfolio_heat_fraction: float = 0.03
    final_risk_min_mult: float = 0.20
    final_risk_max_mult: float = 1.00
    atr_stop_mult_std: float = 1.0
    atr_stop_mult_volatile: float = 1.2
    max_participation_30m: float = 0.01
    thin_participation_30m: float = 0.005

    max_positions: int = 6
    max_positions_per_sector: int = 3
    max_adds: int = 2
    max_campaign_risk_mult: float = 1.5
    corr_threshold: float = 0.70
    corr_lookback: int = 60

    # Opt 3: Scaled quality_mult + Quality-Scaled TP2
    evidence_score_top_tier: int = 6
    quality_mult_top_score: float = 1.10
    high_conviction_quality_mult_min: float = 1.0

    tp1_aligned_r: float = 1.25
    tp2_aligned_r: float = 2.5
    tp2_aligned_r_high_conviction: float = 3.0
    tp1_neutral_r: float = 1.0
    tp2_neutral_r: float = 2.0
    tp2_neutral_r_high_conviction: float = 2.5
    tp1_fraction: float = 0.30
    tp2_fraction: float = 0.30
    runner_fraction: float = 0.40
    breakeven_buffer_atr: float = 0.10

    # Opt 1: Adaptive Runner Trailing Stop
    runner_structure_atr_base: float = 1.0
    runner_structure_atr_strong: float = 1.5
    runner_adx_strong_threshold: float = 30.0
    runner_atr_expansion_threshold: float = 1.15
    runner_profit_ratchet_r_base: float = 2.0
    runner_profit_ratchet_r_strong: float = 3.0

    stale_warn_days: int = 8
    stale_exit_days: int = 10
    stale_exit_runner_r_threshold: float = 0.0
    continuation_box_mult: float = 1.5
    continuation_r_mult: float = 2.0

    dirty_break_fail_days: int = 3
    dirty_reset_days: int = 5

    # --- Momentum continuation (T1) ---
    opening_range_bars: int = 6             # 6 x 5m = 30 min
    entry_window_start: time = time(10, 0)
    entry_window_end: time = time(12, 30)
    rvol_threshold: float = 2.0
    cpr_threshold: float = 0.6
    momentum_score_min: int = 2
    momentum_size_mult_score_3: float = 1.00
    momentum_size_mult_score_4: float = 1.15
    momentum_size_mult_score_5: float = 1.05
    momentum_size_mult_score_6: float = 1.20
    momentum_size_mult_score_7_plus: float = 1.25
    adx_threshold: float = 20.0
    stop_atr_multiple: float = 0.8
    use_or_low_stop: bool = True
    partial_r_trigger: float = 1.25
    partial_fraction: float = 0.33
    move_stop_to_be: bool = True
    use_partial_takes: bool = False           # P14: partials disabled
    eod_flatten_time: time = time(15, 55)
    carry_min_r: float = 0.5
    carry_min_cpr: float = 0.6
    carry_regime_required: tuple[str, ...] = ("A", "B")
    max_carry_days: int = 2
    regime_mult_a: float = 1.0
    regime_mult_b: float = 0.7
    regime_mult_c: float = 0.6
    block_combined_regime_b: bool = True           # Block COMBINED_BREAKOUT in Tier B
    # Flow reversal tuning
    flow_reversal_min_hold_bars: int = 12      # grace period before checking
    flow_reversal_require_below_entry: bool = False  # also require close < entry

    # --- Diagnostic-driven experiment params (Phase 6) ---
    rvol_max: float = 5.0                        # Max RVOL at entry; reject above
    fr_mfe_grace_r: float = 0.20                 # Skip FR if position MFE (R) exceeds this (0=disabled)
    thursday_sizing_mult: float = 1.0            # Sizing multiplier for Thursday (1.0=identity)
    tuesday_sizing_mult: float = 1.0             # Sizing multiplier for Tuesday (1.0=identity)
    fr_trailing_activate_r: float = 0.0          # Activate trailing stop at this MFE R (0=disabled)
    fr_trailing_distance_r: float = 0.3          # Trail distance in R once activated
    close_stop_be_after_r: float = 0.0           # Move stop to breakeven after this MFE R (0=disabled)
    entry_window_end_early: time = time(15, 30)  # Override entry_window_end if tighter

    # --- Phase 7: Quality & Robustness ---
    min_daily_atr_usd: float = 0.0             # Min daily ATR in $ at entry (0=disabled)
    min_selection_score: int = 0                # Min CandidateItem.selection_score (0=disabled)
    min_rs_percentile: float = 0.0             # Min relative_strength_percentile (0=disabled)
    late_entry_cutoff: time = time(11, 0)      # Time after which late_entry_score_min applies
    late_entry_score_min: int = 0              # Min momentum score for late entries (0=disabled)
    late_avwap_cap_pct: float = 0.0            # Max AVWAP premium for late entries (0=disabled)
    late_entry_size_mult: float = 1.0          # Size multiplier for late entries
    bar9_score_min: int = 0                    # Extra momentum score floor for bar 9 / 10:10 entries
    bar9_rvol_min: float = 0.0                 # Extra RVOL floor for bar 9 entries
    bar9_avwap_cap_pct: float = 0.0            # Max AVWAP premium for bar 9 entries (0=disabled)
    bar9_size_mult: float = 1.0                # Size multiplier for bar 9 entries

    # --- Phase 8: Exit & Signal Diagnostics ---
    # A. Time-based quick exit (cut short-hold losers)
    quick_exit_max_bars: int = 0               # Exit if < min_r after N bars (0=disabled)
    quick_exit_min_r: float = 0.2              # Min R to survive quick exit check

    # B. COMBINED_BREAKOUT quality gate (separate thresholds)
    combined_breakout_score_min: int = 5       # Min momentum score for COMBINED entries (0=use global)
    combined_breakout_min_rvol: float = 2.5    # Min RVOL for COMBINED entries (0=use global)

    # C. Signal filters
    avwap_distance_cap_pct: float = 0.0        # Max % above AVWAP at entry (0=disabled)
    or_width_min_pct: float = 0.0015           # Min OR width as % of price (0=disabled)
    or_width_max_pct: float = 0.0              # Max OR width as % of price (0=disabled)
    breakout_distance_cap_r: float = 1.0

    # D. Sector-weighted sizing
    sector_mult_financials: float = 0.65       # Sizing mult for Financials
    sector_mult_communication: float = 0.8     # Sizing mult for Communication Services
    sector_mult_industrials: float = 0.5       # Sizing mult for Industrials
    sector_mult_consumer_disc: float = 1.2     # Sizing mult for Consumer Discretionary
    sector_mult_healthcare: float = 1.0       # Sizing mult for Healthcare

    # E. FR conditional gating (only trigger FR under specific conditions)
    fr_max_hold_bars: int = 48                 # Disable FR after this many bars (0=disabled, distinct from min_hold)
    fr_cpr_threshold: float = 0.3              # Only allow FR when CPR < this (0=disabled)

    # --- Phase 9: Quick Exit Refinement & OR Quality Gate ---
    qe_stage1_bars: int = 10
    qe_stage1_min_r: float = -0.5             # Stage 1 R threshold (exit if below)
    or_breakout_score_min: int = 0             # Min momentum score for OR_BREAKOUT (0=disabled; P14 ablation OFF)
    or_breakout_min_rvol: float = 0.0          # Min RVOL for OR_BREAKOUT (0=use global)
    pdh_breakout_score_min: int = 0            # Min momentum score for PDH entries (0=disabled)
    pdh_breakout_min_rvol: float = 0.0         # Min RVOL for PDH entries (0=use global)
    pdh_entry_window_end: time = time(15, 30)  # Extra PDH-specific entry cutoff
    pdh_avwap_cap_pct: float = 0.005           # Max AVWAP premium for PDH entries (0=disabled)
    pdh_size_mult: float = 0.75                # Size multiplier for PDH entries

    # --- Phase 10: MFE Conviction Exit ---
    mfe_conviction_check_bars: int = 16        # Bar at which to check MFE (0=disabled)
    mfe_conviction_min_r: float = 0.20         # Min MFE in R to survive check
    mfe_conviction_floor_r: float = -0.15      # Compound mode: also require current R < this (0=MFE-only)

    # --- Phase 10: Adaptive Trailing Stop ---
    adaptive_trail_start_bars: int = 25        # Bar at which mid-phase trail begins (0=disabled)
    adaptive_trail_tighten_bars: int = 25      # Bar at which late-phase tight trail begins
    adaptive_trail_mid_activate_r: float = 0.20  # MFE activation for mid phase
    adaptive_trail_mid_distance_r: float = 0.40  # Trail distance in mid phase (wider)
    adaptive_trail_late_activate_r: float = 0.22
    adaptive_trail_late_distance_r: float = 0.12

    # --- Phase 10: COMBINED-Specific Entry Filters ---
    combined_avwap_cap_pct: float = 0.003     # Max AVWAP distance for COMBINED entries (0=disabled)
    combined_breakout_cap_r: float = 0.0      # Max breakout distance for COMBINED entries in R (0=disabled)
    or_breakout_cap_r: float = 0.0            # Max OR breakout distance in R (0=disabled)
    pdh_breakout_cap_r: float = 0.0           # Max PDH breakout distance in R (0=disabled)

    # --- Optimizer-safe conditional entry controls ---
    # All keys use completed signal-bar metadata and preserve next-bar execution.
    block_entry_bars: tuple[int, ...] = ()
    entry_bar_size_mults: dict = field(default_factory=dict)
    entry_type_bar_blocklist: tuple[str, ...] = ()
    entry_type_bar_size_mults: dict = field(default_factory=dict)
    entry_score_blocklist: tuple[str, ...] = ("COMBINED_BREAKOUT:5",)
    entry_score_size_mults: dict = field(default_factory=lambda: {"OR_BREAKOUT:5": 0.75, "COMBINED_BREAKOUT:7": 1.15, "PDH_BREAKOUT:6": 0.5})
    entry_detail_blocklist: tuple[str, ...] = ()
    entry_detail_size_mults: dict = field(default_factory=lambda: {"OR_BREAKOUT:5:!bar_vol_surge": 0.55})
    sector_entry_blocklist: tuple[str, ...] = ()
    sector_entry_size_mults: dict = field(default_factory=dict)

    # --- Round 3: causal early-failure stop tightening ---
    failure_stop_bars: int = 10
    failure_stop_mfe_max_r: float = 0.2
    failure_stop_current_r_max: float = 0.0   # Only tighten if current R is no better than this
    failure_stop_to_r: float = -0.25          # Stop level in R from entry after the completed-bar check
    failure_stop_close_buffer_pct: float = 0.0005 # Keep updated stop beyond current close for next-bar validity

    # --- Round 3 extension: causal trade-maturation confirmation ---
    maturation_stop_bars: int = 0             # Tighten stop after N completed hold bars when maturation evidence fails
    maturation_stop_min_failed_checks: int = 1 # Number of enabled maturation checks that must fail before tightening
    maturation_stop_min_current_r: float = -999.0 # Minimum current R at check (-999=disabled)
    maturation_stop_min_mfe_r: float = 0.0    # Minimum early MFE in R (0=disabled)
    maturation_stop_max_mae_r: float = 0.0    # Max early adverse excursion in R (0=disabled)
    maturation_stop_min_rvol_ratio: float = 0.0 # Current RVOL / signal RVOL floor (0=disabled)
    maturation_stop_min_rvol: float = 0.0     # Current-bar RVOL floor (0=disabled)
    maturation_stop_require_above_breakout: bool = False # Require completed bar to hold breakout level
    maturation_stop_require_above_avwap: bool = False # Require completed bar to hold session AVWAP
    maturation_stop_level_buffer_pct: float = 0.0 # Buffer for above-breakout/AVWAP checks
    maturation_stop_to_r: float = -0.10       # Stop level in R after a failed maturation check
    maturation_stop_close_buffer_pct: float = 0.0005 # Keep updated stop beyond current close for next-bar validity

    # --- Round 3 extension: delayed entry confirmation ---
    entry_confirmation_bars: int = 0          # Observe N completed bars after signal, then fill next bar if confirmed
    entry_confirmation_min_current_r: float = -999.0 # Min R from signal close to confirmation close (-999=disabled)
    entry_confirmation_min_mfe_r: float = 0.0 # Min favorable excursion during confirmation window (0=disabled)
    entry_confirmation_max_mae_r: float = 0.0 # Max adverse excursion during confirmation window (0=disabled)
    entry_confirmation_min_rvol_ratio: float = 0.0 # Confirmation RVOL / signal RVOL floor (0=disabled)
    entry_confirmation_min_rvol: float = 0.0  # Confirmation-bar RVOL floor (0=disabled)
    entry_confirmation_require_above_breakout: bool = False
    entry_confirmation_require_above_avwap: bool = False
    entry_confirmation_level_buffer_pct: float = 0.0
    entry_confirmation_size_mult: float = 1.0 # Optional size uplift after confirmed maturation

    # --- Phase 11: Reclaim / retest entry experiments ---
    reclaim_entry_mode: str = "off"           # off, or, or_avwap, or_pdh, or_pdh_avwap
    reclaim_lookback_bars: int = 24           # Prior bars allowed to establish first breakout
    reclaim_touch_tolerance_pct: float = 0.001 # Reference touch tolerance for retest/reclaim
    reclaim_min_rvol: float = 2.0             # Reclaim-specific RVOL floor
    reclaim_cpr_threshold: float = 0.55       # Reclaim-specific close-location floor
    reclaim_max_avwap_premium_pct: float = 0.0075 # Avoid reclaim entries too extended above AVWAP

    # --- P15 extension: US_ORB-inspired entry quality and acceptance ---
    orb_quality_score_min: float = 0.0        # Min ORB-style composite quality score (0=disabled)
    orb_quality_size_floor: float = 0.0       # Quality sizing floor at min score (0=disabled)
    orb_quality_top_score: float = 85.0       # Score that earns top quality sizing
    orb_quality_top_mult: float = 1.15        # Max quality sizing multiplier
    orb_gap_policy_mode: str = "off"          # off, size, filter
    orb_gap_block_pct: float = 0.12           # Hard block large gaps when policy enabled
    orb_gap_down_block_pct: float = -0.05     # Hard block large down gaps when policy enabled
    orb_gap_caution_pct: float = 0.08         # Caution gap threshold
    orb_gap_tight_pct: float = 0.05           # Mild gap threshold
    orb_gap_caution_mult: float = 0.65        # Size multiplier for caution gaps
    orb_gap_tight_mult: float = 0.80          # Size multiplier for mild gaps
    orb_entry_range_cap_r: float = 1.1
    orb_time_decay_start: time = time(10, 30) # Start late-entry RVOL/size decay
    orb_late_rvol_add_per_30m: float = 0.0    # Additive RVOL floor per 30m after start
    orb_late_size_decay_per_30m: float = 0.0  # Multiplicative size decay per 30m after start
    orb_late_size_floor: float = 0.50         # Minimum late-entry size multiplier
    orb_structure_stop_mode: str = "default"  # default, reclaim, all
    orb_structure_stop_buffer_pct: float = 0.0015 # Buffer below retest/support level
    orb_structure_min_risk_pct: float = 0.60  # Avoid unrealistically tiny retest stops

    # --- P15 extension: US_ORB-inspired scratch / retracement exits ---
    orb_retracement_trail_start_bars: int = 0 # Enable retracement trail after N bars (0=disabled)
    orb_retracement_trail_tighten_bars: int = 0
    orb_retracement_trail_min_mfe_r: float = 0.40
    orb_retracement_trail_early: float = 0.45 # Preserve this fraction of MFE before tighten
    orb_retracement_trail_late: float = 0.72  # Preserve this fraction of MFE after tighten

    # --- Position Sizing: Buying Power ---
    intraday_leverage: float = 2.0             # Max leverage (2.0 = Reg T, 4.0 = PDT intraday)

    selection_long_count: int = 20
    selection_short_count: int = 20
    universe_cap: int = 650

    diagnostics_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "diagnostics"
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "cache"
    )
    research_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "research"
    )
    artifact_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "artifacts"
    )
    state_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_alcb" / "state"
    )
    blacklist_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "blacklist.txt"
    )
    scanner: ScannerSettings = field(default_factory=ScannerSettings)


def build_proxy_instruments() -> list[Instrument]:
    primary = {"SPY": "ARCA", "QQQ": "NASDAQ", "IWM": "ARCA"}
    return [
        Instrument(
            symbol=symbol,
            root=symbol,
            venue="SMART",
            primary_exchange=primary[symbol],
            sec_type="STK",
            tick_size=0.01,
            tick_value=0.01,
            multiplier=1.0,
            point_value=1.0,
            currency="USD",
        )
        for symbol in PROXY_SYMBOLS
    ]
