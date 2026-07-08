"""Static configuration for the IARIC v1 strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from libs.oms.models.instrument import Instrument

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
STRATEGY_ID = "IARIC_v1"
PROXY_SYMBOLS = ("SPY", "QQQ", "IWM")


@dataclass(frozen=True)
class StrategySettings:
    premarket_start: time = time(7, 0)
    premarket_end: time = time(8, 30)
    market_open: time = time(9, 30)
    open_block_end: time = time(9, 35)
    entry_end: time = time(15, 0)
    close_block_start: time = time(15, 45)
    forced_flatten: time = time(15, 58)

    hot_max: int = 40
    active_monitoring_target: int = 20
    warm_poll_interval_s: int = 30
    cold_poll_interval_s: int = 90
    monitoring_refresh_minutes: int = 60

    min_price: float = 10.0
    min_adv_usd: float = 20_000_000.0
    max_median_spread_pct: float = 0.0050
    avwap_band_pct: float = 0.0050
    avwap_acceptance_band_pct: float = 0.0100
    anchor_lookback_sessions: int = 40

    tier_a_min: float = 0.65
    tier_b_min: float = 0.40
    regime_full_mult_score: float = 0.825
    breadth_threshold_pct: float = 55.0
    vix_percentile_threshold: float = 80.0
    hy_spread_5d_bps_threshold: float = 15.0

    panic_flush_drop_pct: float = 0.03
    drift_exhaustion_drop_pct: float = 0.02
    warm_drop_pct: float = 0.007
    hot_drop_pct: float = 0.015
    panic_flush_minutes: int = 15
    drift_exhaustion_minutes: int = 60
    setup_stale_minutes: int = 45
    invalidation_cooldown_minutes: int = 15

    acceptance_base_closes: int = 2
    partial_r_multiple: float = 1.5
    partial_exit_fraction: float = 0.50
    partial_r_min: float = 1.2
    partial_r_max: float = 2.0
    partial_frac_min: float = 0.33
    partial_frac_max: float = 0.60
    time_stop_minutes: int = 45
    avwap_breakdown_pct: float = 0.007
    avwap_breakdown_volume_mult: float = 1.5

    base_risk_fraction: float = 0.00864
    intraday_leverage: float = 2.0             # Max leverage (2.0 = Reg T, 4.0 = PDT intraday)
    daily_stop_r: float = 2.75
    heat_cap_r: float = 5.4
    portfolio_daily_stop_r: float = 3.5
    sector_risk_cap_pct: float = 0.35
    max_positions_tier_a: int = 8
    max_positions_tier_b: int = 5
    max_positions_per_sector: int = 4
    minimum_remaining_size_pct: float = 0.30

    confidence_green_mult: float = 1.0
    confidence_yellow_mult: float = 0.65
    stale_penalty: float = 0.85

    # Backtest-sweepable parameters (defaults match hardcoded originals)
    stop_risk_cap_pct: float = 0.02          # max risk offset as fraction of price
    max_carry_days: int = 7                   # max overnight carry duration
    min_carry_r: float = 0.5                  # minimum R-multiple to qualify for carry
    flow_reversal_lookback: int = 1           # consecutive negative flow days to trigger exit
    min_conviction_multiplier: float = 0.25   # minimum conviction to enter (0 = any positive)

    # T2 intraday engine tuning (legacy FSM engine)
    t2_avwap_band_mult: float = 1.0        # multiplier on avwap_band_pct for T2 FSM band check

    # T2 v2: Entry triggers
    t2_vwap_pullback_atr_mult: float = 0.5   # VWAP pullback distance (ATR units)
    t2_vwap_reclaim_vol_pct: float = 0.80    # min reclaim bar volume vs expected
    t2_orb_cpr_min: float = 0.60             # min CPR on ORB breakout bar
    t2_afternoon_vol_mult: float = 1.20      # PM volume acceleration threshold
    t2_fallback_entry_bar: int = 6           # bar index for fallback entry (10:00 = bar 6)
    t2_max_entries_per_symbol: int = 1       # max entries per symbol per day (excl. PM re-entry)
    t2_pm_reentry: bool = True               # allow afternoon re-entry for stopped-out symbols

    # T2 v2: Adaptive stop
    t2_initial_atr_mult: float = 1.5         # Phase 1 stop width (ATR units)
    t2_breakeven_r: float = 0.5              # MFE to move stop to breakeven (Phase 2)
    t2_profit_trail_atr: float = 1.0         # Phase 3 trailing distance (ATR)
    t2_tight_trail_atr: float = 0.5          # Phase 4 tight trailing (ATR)
    t2_tight_trail_r: float = 2.0            # MFE to engage Phase 4

    # T2 v2: Staleness & partial
    t2_staleness_hours: float = 5.0          # hours before staleness timeout (daily-ATR scale)
    t2_staleness_min_r: float = 0.1          # min MFE R to avoid staleness exit
    t2_partial_r: float = 1.5               # R-mult for partial take
    t2_partial_frac: float = 0.33            # fraction to exit on partial

    # T2 v2: EOD carry scoring
    t2_carry_threshold: float = 65.0         # min carry score for full carry
    t2_carry_partial_threshold: float = 45.0 # min carry score for partial carry

    # T2 v2: Day-of-week sizing
    t2_tuesday_mult: float = 0.50           # sizing multiplier for Tuesdays
    t2_friday_mult: float = 1.25            # sizing multiplier for Fridays

    # T2 v2: Structural improvements (Phase 5)
    t2_open_entry: bool = False              # enter at bar 1 (near open), like T1
    t2_open_entry_stop_atr: float = 1.5      # stop for open entries (daily ATR mult)
    t2_open_entry_size_mult: float = 1.0     # sizing multiplier for open entries
    t2_default_carry_profitable: bool = False # carry all profitable positions by default
    t2_default_carry_min_r: float = 0.0      # min unrealized R to default-carry
    t2_default_carry_stop_atr: float = 1.0   # protective overnight stop (ATR mult)
    t2_entry_strength_gate: float = 0.0      # min entry strength score (0 = no gate)
    t2_entry_strength_sizing: bool = False   # multiply sizing by entry_strength
    t2_orb_window_bars: int = 1              # ORB trigger window (1 = bar 6 only)
    t2_orb_require_bullish: bool = True      # require close > open on ORB bar
    t2_vwap_pullback_multibar: bool = False  # multi-bar VWAP pullback detection
    t2_staleness_action: str = "CLOSE"       # CLOSE | TIGHTEN | SKIP
    t2_staleness_tighten_atr: float = 0.5    # tighten distance when action=TIGHTEN
    t2_avwap_breakdown_action: str = "CLOSE" # CLOSE | TIGHTEN
    t2_regime_b_sizing_mult: float = 1.0     # sizing multiplier for Tier B entries
    t2_breakeven_use_closes: bool = False     # use highest_close instead of mfe_r for BE
    t2_fallback_require_above_vwap: bool = False  # VWAP gate for fallback entries
    t2_fallback_momentum_bars: int = 0       # consecutive green bars required (0 = none)

    # Structural alpha amplification (defaults = current behavior)
    carry_only_entry: bool = True            # only enter if carry conditions possible
    strong_only_entry: bool = False          # only enter STRONG sponsorship
    entry_order_by_conviction: bool = False  # sort tradable by conviction desc before entering
    use_close_stop: bool = True              # exit at close if underwater (replaces hard stop)
    intraday_flow_check: bool = False        # check flow reversal on same-day positions
    regime_b_carry_mult: float = 0.6         # >0 enables Regime B carry at this size mult
    carry_top_quartile: bool = False         # require close in top 25% of daily range to carry

    # T1 entry flow gate: require positive flow proxy at entry (default OFF = current behavior)
    t1_entry_flow_gate: bool = True          # gate entries on positive flow proxy
    t1_entry_flow_lookback: int = 1          # number of recent flow values to check

    # T1 intraday partial takes (default OFF = current behavior)
    t1_partial_takes: bool = False           # enable intraday partial profit-taking
    t1_partial_r_trigger: float = 0.5        # R-multiple at which to take partial
    t1_partial_fraction: float = 0.25        # fraction of position to exit

    # T1 gap-down entry filter (default 0 = no filter = current behavior)
    t1_gap_down_skip_pct: float = 0.0        # skip entry if gap-down exceeds this (0 = disabled)

    # Carry close-pct minimum (default 0.75 = current hardcoded value)
    carry_close_pct_min: float = 0.75        # min close-in-range pct for carry eligibility

    # Carry trailing stop (default 0 = disabled = current behavior)
    carry_trail_activate_days: int = 0       # days before trailing activates (0 = disabled)
    carry_trail_atr_mult: float = 1.5        # trail distance in ATR (0 = breakeven)

    # T1 day-of-week sizing (default 1.0 = no adjustment = current behavior)
    dow_tuesday_mult: float = 1.0            # sizing multiplier for Tuesdays
    dow_friday_mult: float = 1.0             # sizing multiplier for Fridays

    # T1 Phase 4: Close-stop graduation (0.0 = current behavior)
    t1_close_stop_buffer_r: float = 0.0      # CLOSE_STOP only if C < entry - buffer * risk_per_share

    # T1 Phase 4: Regime A filtering (False/1.0 = current behavior)
    t1_regime_a_skip: bool = False            # skip ALL Tier A entries
    t1_regime_a_size_mult: float = 1.0        # sizing mult for Tier A (0.5 = half size)

    # T1 Phase 4: Carry MFE gate (0.0 = disabled)
    t1_carry_mfe_gate_r: float = 0.0         # min intraday MFE (R-units) to qualify for carry

    # ── Phase 5: MFE Shield (suppress CLOSE_STOP for trades that showed momentum) ──
    t1_mfe_shield_enabled: bool = False          # enable MFE-based close-stop suppression
    t1_mfe_shield_threshold_r: float = 0.5       # min intraday MFE (R) to earn shield
    t1_mfe_shield_close_floor_r: float = -0.5    # min unrealized R to carry when shielded (neg = allow underwater)

    # ── Phase 5: Multi-level partial takes ──
    t1_partial_level2_enabled: bool = False      # enable 2nd partial take level
    t1_partial_level2_r_trigger: float = 1.0     # R-trigger for 2nd partial
    t1_partial_level2_fraction: float = 0.33     # fraction of REMAINING position
    t1_partial_level3_enabled: bool = False      # enable 3rd partial take level
    t1_partial_level3_r_trigger: float = 1.5     # R-trigger for 3rd partial
    t1_partial_level3_fraction: float = 0.50     # fraction of REMAINING position

    # ── Phase 5: Conviction dampening ──
    t1_conviction_dampen: float = 1.0            # 1.0 = full conviction, 0.0 = flat sizing
    t1_conviction_floor: float = 0.0             # min effective conviction (0 = no floor)
    t1_conviction_cap: float = 99.0              # max effective conviction (99 = no cap)

    # ── Phase 6: Feature-Based Entry Gates ──
    t1_prev_cpr_min: float = 0.0                 # min previous day (C-L)/(H-L); 0 = disabled
    t1_persistence_min: float = 0.0              # min flow persistence (0-1); 0 = disabled
    t1_rs_percentile_min: float = 0.0            # min RS percentile (0-100); 0 = disabled
    t1_require_trend_pass: bool = False           # require price > SMA50 + positive slope
    t1_require_leader_pass: bool = False          # require top RS in sector
    t1_anchor_quality_gate: bool = False          # only IMPULSE_DAY/BREAKOUT anchors

    # ── Phase 6: Runner Protection ──
    t1_partial_keep_runner: bool = False          # skip partial if it would fully exit position

    # ── Phase 6: Day-of-Week Intelligence ──
    t1_wed_skip: bool = False                    # skip all Wednesday entries
    t1_thu_skip: bool = False                    # skip all Thursday entries
    dow_wednesday_mult: float = 1.0              # sizing multiplier for Wednesdays

    # ── Tier 3: Pullback-Buy V2 ──
    pb_v2_enabled: bool = True                     # master switch; False = exact legacy behavior

    # V2 Trend Filter
    pb_v2_allow_secular: bool = True               # allow SECULAR trend tier (below SMA50, above SMA200)
    pb_v2_secular_sizing_mult: float = 0.65        # sizing discount for SECULAR tier

    # V2 Range Filters (widened)
    pb_v2_gap_min_pct: float = -15.0
    pb_v2_gap_max_pct: float = 2.0
    pb_v2_sma_dist_min_pct: float = -10.0
    pb_v2_sma_dist_max_pct: float = 25.0

    # V2 Triggers
    pb_v2_rsi2_thresh: float = 15.0               # Trigger A: relaxed from 10
    pb_v2_rsi5_thresh: float = 30.0               # Trigger B
    pb_v2_cdd_min_for_rsi5: int = 2               # Trigger B: min CDD
    pb_v2_depth_thresh: float = 1.5               # Trigger C: ATR pullback depth
    pb_v2_bb_pctb_thresh: float = 0.05            # Trigger D
    pb_v2_vol_climax_thresh: float = 2.0          # Trigger E
    pb_v2_rs_ratio_thresh: float = 1.02           # Trigger F
    pb_v2_roc_thresh: float = -3.0                # Trigger F: ROC(5) threshold
    pb_v2_gap_fill_thresh: float = -2.0           # Trigger G: gap-down % at open

    # V2 Scoring
    pb_v2_signal_floor: float = 66.0
    pb_v2_signal_floor_tier_b: float = 0.0         # Tier B signal floor override (0=use global)
    pb_v2_sizing_premium: float = 1.00            # score >= 75
    pb_v2_sizing_standard: float = 0.80           # score 60-74
    pb_v2_sizing_reduced: float = 0.55            # score 45-59
    pb_v2_sizing_minimum: float = 0.35            # score 30-44
    pb_v2_candidate_persistence_bonus: float = 4.0

    # V2 Entry Routes
    pb_v2_open_scored_enabled: bool = True
    pb_v2_open_scored_min_score: float = 45.0
    pb_v2_open_scored_max_slots: int = 4
    pb_v2_delayed_confirm_min_close_pct: float = 0.40
    pb_v2_delayed_confirm_vol_ratio: float = 0.50
    pb_v2_vwap_bounce_enabled: bool = False
    pb_v2_vwap_bounce_after_bar: int = 12
    pb_v2_vwap_bounce_vol_ratio: float = 0.60
    pb_v2_afternoon_retest_enabled: bool = False
    pb_v2_afternoon_retest_after_bar: int = 48
    pb_v2_afternoon_retest_min_score: float = 50.0
    pb_v2_afternoon_retest_sizing_mult: float = 0.80

    # V2 Route access control (allow rescue candidates into intraday routes)
    pb_v2_open_scored_rank_pct_max: float = 100.0       # max rank percentile for OPEN_SCORED in V2 mode (100 = no filter)
    pb_v2_vwap_bounce_allow_rescue: bool = False         # allow rescue-flow candidates into VWAP_BOUNCE route
    pb_v2_afternoon_retest_allow_rescue: bool = False    # allow rescue-flow candidates into AFTERNOON_RETEST route
    pb_v2_delayed_confirm_allow_rescue: bool = False    # allow rescue-flow candidates into DELAYED_CONFIRM route

    # V2 Exits
    pb_v2_mfe_stage1_trigger: float = 0.50
    pb_v2_mfe_stage1_stop_r: float = -0.10
    pb_v2_mfe_stage2_trigger: float = 0.60
    pb_v2_mfe_stage3_trigger: float = 1.25
    pb_v2_mfe_stage3_trail_atr: float = 0.75
    pb_v2_partial_profit_trigger_r: float = 0.1
    pb_v2_partial_profit_remainder_stop_r: float = 0.7
    pb_v2_ema_reversion_exit: bool = True
    pb_v2_ema_reversion_min_r: float = 0.03
    pb_v2_rsi_exit_open_scored: float = 60.0
    pb_v2_rsi_exit_delayed: float = 55.0
    pb_v2_rsi_exit_vwap_bounce: float = 55.0
    pb_v2_rsi_exit_afternoon: float = 50.0
    pb_v2_vwap_fail_bars: int = 3
    pb_v2_vwap_fail_close_pct: float = 0.35
    pb_v2_stale_bars: int = 4
    pb_v2_stale_mfe_thresh: float = 0.08
    pb_v2_stale_tighten_pct: float = 0.30

    # V2 Carry (inverted -- default is carry, flatten only when conditions met)
    pb_v2_flatten_loss_r: float = -0.50
    pb_v2_flatten_regime_c_min_r: float = 0.50
    pb_v2_carry_profit_lock_r: float = 0.75        # overnight stop ratchet: lock profit above this close_r threshold
    pb_v2_carry_overnight_stop_atr: float = 1.0
    pb_v2_flow_grace_days: int = 2                 # skip flow reversal check during first N hold days (pullback entries have inherently negative flow)

    # ── Tier 3: Pullback-Buy Mean-Reversion (legacy) ──
    pb_rsi_period: int = 2                       # RSI lookback (2, 3, 5)
    pb_rsi_entry: float = 10.0                   # RSI < this triggers entry
    pb_rsi_exit: float = 70.0                    # RSI > this triggers mean-reversion exit
    pb_cdd_min: int = 3                          # consecutive down days trigger (0=disabled)
    pb_ma_zone_entry: bool = False               # price between SMA20 and SMA50 as trigger
    pb_trend_sma: int = 50                       # trend filter SMA period
    pb_trend_slope_lookback: int = 10            # bars to measure SMA slope
    pb_atr_period: int = 14                      # ATR period for stop
    pb_atr_stop_mult: float = 1.0                # stop = entry - mult * ATR (V4R1; live uses pb_stop_daily_atr_cap=1.0)
    pb_use_close_stop: bool = False              # exit at close instead of stop price when stop is breached
    pb_max_hold_days: int = 2                    # time stop in days
    pb_profit_target_r: float = 0.0              # profit target R-mult (0=disabled)
    pb_flow_gate: bool = True                    # require positive flow proxy
    pb_max_positions: int = 9
    pb_regime_gate: str = "C_only_skip"          # "C_only_skip" | "B_and_above" | "any"
    pb_carry_enabled: bool = True                # allow overnight carry
    pb_carry_min_r: float = 0.25                 # min R to qualify for carry
    pb_gap_min_pct: float = -99.0                # minimum acceptable gap percent at entry
    pb_gap_max_pct: float = 99.0                 # maximum acceptable gap percent at entry
    pb_gap_up_size_mult: float = 0.60            # phase-5 portfolio overlay parity
    pb_sma_dist_min_pct: float = 0.0             # minimum distance above trend SMA at entry
    pb_sma_dist_max_pct: float = 99.0            # maximum distance above trend SMA at entry
    pb_cdd_max: int = 6
    pb_entry_rank_min: int = 1                   # minimum ranked candidate to admit
    pb_entry_rank_max: int = 999                 # maximum ranked candidate to admit
    pb_entry_rank_pct_min: float = 0.0           # minimum candidate rank percentile to admit
    pb_entry_rank_pct_max: float = 100.0         # maximum candidate rank percentile to admit
    pb_min_candidates_day: int = 8               # scaled to effective trade universe when running 5m-covered pullback tests
    pb_tuesday_mult: float = 1.0                 # Tuesday sizing multiplier
    pb_wednesday_mult: float = 1.0               # Wednesday sizing multiplier
    pb_thursday_mult: float = 0.75               # Thursday sizing multiplier (risk sweep optimized)
    pb_friday_mult: float = 1.0                  # Friday sizing multiplier
    pb_carry_close_pct_min: float = 0.0          # minimum close-in-range pct to carry
    pb_carry_mfe_gate_r: float = 0.0             # minimum intraday MFE R to carry
    pb_carry_min_daily_signal_score: float = 0.0 # minimum daily signal score to carry
    pb_flow_reversal_lookback: int = 1           # negative flow days needed for carry exit
    pb_execution_mode: str = "intraday_hybrid"   # daily | intraday_hybrid
    pb_intraday_bar_minutes: int = 5             # 5m replay for hybrid pullback
    pb_intraday_entry_start: time = time(9, 35)  # no open-chasing by default
    pb_intraday_entry_end: time = time(15, 0)
    pb_intraday_force_exit: time = time(15, 55)
    pb_intraday_priority_reserve_slots: int = 1  # reserve capacity for 5m-refined names before daily fallback
    pb_opening_range_bars: int = 6               # 30 minutes
    pb_flush_window_bars: int = 18               # first 90 minutes
    pb_flush_min_atr: float = 0.20               # minimum flush depth in daily ATR
    pb_flush_cpr_max: float = 0.45               # flush bar should close weak
    pb_reclaim_offset_atr: float = 0.10          # reclaim buffer over flush bar
    pb_ready_acceptance_bars: int = 1            # bullish confirmation bars
    pb_ready_min_cpr: float = 0.50               # ready bar close strength
    pb_ready_min_volume_ratio: float = 0.70      # ready bar vol vs expected
    pb_ready_vwap_buffer_atr: float = 0.00       # extra VWAP reclaim buffer
    pb_entry_score_min: float = 50.0             # core entry threshold (0-100)
    pb_entry_score_family: str = "meanrev_sweetspot_v1"  # route_momentum_v1 | meanrev_sweetspot_v1
    pb_entry_strength_sizing: bool = True        # scale size modestly by intraday score
    pb_improvement_window_bars: int = 2          # better-price wait after READY
    pb_improvement_discount_pct: float = 0.0015  # discount target for READY entry
    pb_delayed_confirm_after_bar: int = 5
    pb_delayed_confirm_min_close_pct: float = 0.50
    pb_delayed_confirm_score_min: float = 52.0
    pb_pm_reentry: bool = False                  # opt-in only; default quality is weak in current baseline
    pb_pm_reentry_after_bar: int = 48            # 13:30 ET on 5m bars
    pb_max_reentries_per_day: int = 1
    pb_rescue_flow_enabled: bool = False         # allow narrow rescue lane for flow rejects
    pb_rescue_max_per_day: int = 1
    pb_rescue_min_score: float = 72.0            # stricter than core threshold
    pb_stop_session_atr_mult: float = 0.60       # structural stop padding in session ATR
    pb_stop_daily_atr_cap: float = 1.00          # cap structural stop width in daily ATR
    pb_stale_exit_bars: int = 0                  # <=0 disables stale exit
    pb_stale_exit_min_r: float = 0.10
    pb_partial_r: float = 1.10                   # first scale-out trigger
    pb_partial_frac: float = 0.33
    pb_mfe_protect_trigger_r: float = 0.0        # move stop after trade earns this MFE
    pb_mfe_protect_stop_r: float = 0.0           # protected stop in R above entry
    pb_breakeven_r: float = 0.75                 # move stop to BE after genuine progress
    pb_trail_activate_r: float = 1.25
    pb_trail_atr_mult: float = 0.75
    pb_vwap_fail_lookback_bars: int = 3          # lower-high + VWAP failure exit
    pb_vwap_fail_cpr_max: float = -1.0           # <0 disables VWAP-fail exit
    pb_carry_score_threshold: float = 50.0       # fallback carry score
    pb_carry_score_fallback: bool = True
    pb_daily_signal_family: str = "meanrev_sweetspot_v1"  # balanced_v1 | trend_guard | meanrev_v1 | hybrid_alpha_v1 | meanrev_sweetspot_v1 | quality_hybrid_v1 | sponsor_rs_hybrid_v1 | meanrev_plus_v1
    pb_flow_policy: str = "soft_penalty_rescue"  # hard_reject | soft_penalty | soft_penalty_rescue
    pb_min_candidates_day_hard_gate: bool = False
    pb_backtest_intraday_universe_only: bool = True
    pb_daily_signal_min_score: float = 54.0
    pb_daily_rescue_min_score: float = 52.0
    pb_rescue_size_mult: float = 0.65
    pb_signal_rank_gate_mode: str = "score_rank"  # score_rank | percentile_only
    pb_open_scored_enabled: bool = True
    pb_open_scored_rank_pct_max: float = 100.0
    pb_open_scored_min_score: float = 45.0
    pb_open_scored_max_share: float = 0.25
    pb_open_scored_missing_5m_allow: bool = True
    pb_open_scored_fill_timing: str = "next_5m_open"  # next_5m_open | same_open
    pb_open_scored_carry_min_r: float = 0.00
    pb_open_scored_carry_close_pct_min: float = 0.0
    pb_open_scored_carry_mfe_gate_r: float = 0.0
    pb_open_scored_carry_min_daily_signal_score: float = 0.0
    pb_open_scored_carry_score_threshold: float = 50.0
    pb_open_scored_carry_score_fallback_enabled: bool = True
    pb_open_scored_max_hold_days: int = 2
    pb_open_scored_rsi_exit: float = 62.0
    pb_open_scored_flow_reversal_lookback: int = 2
    pb_open_scored_quick_exit_loss_r: float = 0.0
    pb_open_scored_stale_exit_bars: int = 0
    pb_open_scored_stale_exit_min_r: float = 0.05
    pb_open_scored_vwap_fail_lookback_bars: int = 4
    pb_open_scored_vwap_fail_cpr_max: float = -1.0
    pb_open_scored_partial_r: float = 1.20
    pb_open_scored_mfe_protect_trigger_r: float = 0.0
    pb_open_scored_mfe_protect_stop_r: float = 0.0
    pb_open_scored_breakeven_r: float = 0.85
    pb_open_scored_trail_activate_r: float = 1.35
    pb_delayed_confirm_enabled: bool = True
    pb_delayed_confirm_min_daily_signal_score: float = 35.0
    pb_delayed_confirm_carry_min_r: float = 0.10
    pb_delayed_confirm_carry_close_pct_min: float = 0.62
    pb_delayed_confirm_carry_mfe_gate_r: float = 0.20
    pb_delayed_confirm_carry_min_daily_signal_score: float = 0.0
    pb_delayed_confirm_carry_score_threshold: float = 50.0
    pb_delayed_confirm_carry_score_fallback_enabled: bool = True
    pb_delayed_confirm_max_hold_days: int = 4
    pb_delayed_confirm_rsi_exit: float = 60.0
    pb_delayed_confirm_flow_reversal_lookback: int = 2
    pb_delayed_confirm_quick_exit_loss_r: float = 0.0
    pb_delayed_confirm_stale_exit_bars: int = 0
    pb_delayed_confirm_stale_exit_min_r: float = 0.08
    pb_delayed_confirm_vwap_fail_lookback_bars: int = 3
    pb_delayed_confirm_vwap_fail_cpr_max: float = -1.0
    pb_delayed_confirm_partial_r: float = 1.05
    pb_delayed_confirm_mfe_protect_trigger_r: float = 0.0
    pb_delayed_confirm_mfe_protect_stop_r: float = 0.0
    pb_delayed_confirm_breakeven_r: float = 0.70
    pb_delayed_confirm_trail_activate_r: float = 1.20
    pb_opening_reclaim_enabled: bool = False
    pb_opening_reclaim_min_daily_signal_score: float = 50.0
    pb_opening_reclaim_carry_min_r: float = 0.18
    pb_opening_reclaim_carry_close_pct_min: float = 0.68
    pb_opening_reclaim_carry_mfe_gate_r: float = 0.30
    pb_opening_reclaim_carry_min_daily_signal_score: float = 0.0
    pb_opening_reclaim_carry_score_threshold: float = 50.0
    pb_opening_reclaim_carry_score_fallback_enabled: bool = True
    pb_opening_reclaim_max_hold_days: int = 3
    pb_opening_reclaim_rsi_exit: float = 58.0
    pb_opening_reclaim_flow_reversal_lookback: int = 1
    pb_opening_reclaim_quick_exit_loss_r: float = 0.0
    pb_opening_reclaim_stale_exit_bars: int = 0
    pb_opening_reclaim_stale_exit_min_r: float = 0.10
    pb_opening_reclaim_vwap_fail_lookback_bars: int = 3
    pb_opening_reclaim_vwap_fail_cpr_max: float = -1.0
    pb_opening_reclaim_partial_r: float = 1.00
    pb_opening_reclaim_mfe_protect_trigger_r: float = 0.0
    pb_opening_reclaim_mfe_protect_stop_r: float = 0.0
    pb_opening_reclaim_breakeven_r: float = 0.65
    pb_opening_reclaim_trail_activate_r: float = 1.10

    # Hybrid daily-alpha mode: disable intraday exits, use daily-level decisions only
    t2_daily_alpha_mode: bool = False       # skip TIME_STOP, AVWAP_BD, PARTIAL intraday
    t2_bar_minutes: int = 5                 # bar granularity: 5 (default) or 30

    # ── V3 Enhancement Flags (all OFF = T1 FSM behavior + EOD flatten) ──

    # Overnight carry (combines FSM entries with carry alpha)
    v3_carry_enabled: bool = False               # Enable overnight carry
    v3_carry_score_fallback: bool = False         # Use carry score when binary check fails
    v3_carry_score_threshold: float = 65.0        # Min score for fallback carry

    # Adaptive trailing stop (above structural stop only, no breakeven trap)
    v3_adaptive_trail: bool = False               # Enable trailing
    v3_trail_activation_r: float = 1.0            # R-multiple to start trailing
    v3_trail_atr_mult: float = 1.5               # Trail distance in session ATR

    # Entry price improvement window (after FSM fires READY_TO_ENTER)
    v3_entry_improvement: bool = False            # Enable improvement window
    v3_improvement_window_bars: int = 3           # Bars to wait for better price
    v3_improvement_discount_pct: float = 0.003    # Min improvement to take

    # PM re-entry for stopped-out FSM setups
    v3_pm_reentry: bool = False
    v3_pm_reentry_after_bar: int = 48             # 13:30 ET

    # Staleness tighten (tighten stop, don't exit)
    v3_staleness_tighten: bool = False
    v3_staleness_hours: float = 5.0
    v3_staleness_tighten_atr: float = 0.5

    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "cache"
    )
    blacklist_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "blacklist.txt"
    )
    diagnostics_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "diagnostics"
    )
    research_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "research"
    )
    artifact_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "artifacts"
    )
    state_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "strategy_iaric" / "state"
    )

    @property
    def timing_sizing(self) -> tuple[tuple[time, time, float], ...]:
        return (
            (time(9, 35), time(10, 30), 1.00),
            (time(10, 30), time(12, 0), 0.85),
            (time(12, 0), time(13, 30), 0.70),
            (time(13, 30), time(14, 30), 0.90),
            (time(14, 30), time(15, 0), 0.75),
        )


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
