from __future__ import annotations

from typing import Any


DAILY_BASE_MUTATIONS: dict[str, Any] = {
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_entry_rank_min": 1,
    "param_overrides.pb_entry_rank_max": 999,
    "param_overrides.pb_entry_rank_pct_min": 0.0,
    "param_overrides.pb_entry_rank_pct_max": 100.0,
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_min_candidates_day_hard_gate": False,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",
    "param_overrides.max_positions_per_sector": 5,
    "param_overrides.pb_wednesday_mult": 1.0,
    "param_overrides.pb_thursday_mult": 1.0,
    "param_overrides.pb_friday_mult": 1.0,
    "param_overrides.pb_atr_stop_mult": 1.0,
}


BASE_MUTATIONS: dict[str, Any] = {
    # Daily signal
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_entry_rank_min": 1,
    "param_overrides.pb_entry_rank_max": 999,
    "param_overrides.pb_entry_rank_pct_min": 20.0,
    "param_overrides.pb_entry_rank_pct_max": 50.0,
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",
    "param_overrides.pb_min_candidates_day_hard_gate": False,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_cdd_max": 5,
    "param_overrides.pb_sma_dist_max_pct": 10.0,
    "param_overrides.max_positions_per_sector": 2,
    # Sizing
    "param_overrides.pb_wednesday_mult": 1.0,
    "param_overrides.pb_thursday_mult": 0.5,
    "param_overrides.pb_friday_mult": 1.0,
    # Stop
    "param_overrides.pb_atr_stop_mult": 1.25,
    # Hybrid execution
    "param_overrides.pb_execution_mode": "intraday_hybrid",
    "param_overrides.pb_carry_enabled": True,
    "param_overrides.pb_carry_score_threshold": 50.0,
    "param_overrides.pb_entry_score_min": 50.0,
    "param_overrides.pb_entry_score_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_ready_min_cpr": 0.50,
    "param_overrides.pb_ready_min_volume_ratio": 0.70,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_delayed_confirm_after_bar": 6,
    "param_overrides.pb_delayed_confirm_score_min": 42.0,
    "param_overrides.pb_delayed_confirm_min_daily_signal_score": 35.0,
    "param_overrides.pb_rescue_flow_enabled": False,
    # Disabled routes (P3 may enable)
    "param_overrides.pb_opening_reclaim_enabled": False,
    "param_overrides.pb_open_scored_enabled": False,
    # Disabled protection mechanisms (P1 may enable)
    "param_overrides.pb_delayed_confirm_quick_exit_loss_r": 0.0,
    "param_overrides.pb_delayed_confirm_stale_exit_bars": 0,
    "param_overrides.pb_delayed_confirm_vwap_fail_cpr_max": -1.0,
    "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.0,
    "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.0,
}


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Protected Widening + MFE Protection", ["avg_r", "managed_exit_share", "eod_flatten_inverse", "total_trades", "expected_total_r"]),
    2: ("Signal Aperture + Recalibration", ["avg_r", "signal_score_edge", "total_trades", "expected_total_r", "profit_factor"]),
    3: ("Route Diversification + Capacity Expansion", ["avg_r", "route_diversity", "expected_total_r", "total_trades"]),
    4: ("Carry Calibration + Rescue", ["avg_r", "carry_avg_r", "carry_trade_share", "managed_exit_share", "total_trades"]),
    5: ("Robustness Overlays + Frequency Validation", ["avg_r", "profit_factor", "sharpe", "expected_total_r", "total_trades"]),
}


PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: EOD Flatten Mitigation + MFE Protection --
    1: [
        # MFE protection variants (primary lever)
        (
            "protect_040_after_060",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.60,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.40,
            },
        ),
        (
            "protect_030_after_050",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.30,
            },
        ),
        (
            "protect_020_after_040",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.40,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.20,
            },
        ),
        (
            "protect_be_after_035",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.35,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.00,
            },
        ),
        # Earlier breakeven (current default 0.70R)
        (
            "breakeven_050",
            {
                "param_overrides.pb_delayed_confirm_breakeven_r": 0.50,
            },
        ),
        (
            "breakeven_040",
            {
                "param_overrides.pb_delayed_confirm_breakeven_r": 0.40,
            },
        ),
        # Earlier trail activation (current 1.20R)
        (
            "trail_activate_090",
            {
                "param_overrides.pb_delayed_confirm_trail_activate_r": 0.90,
            },
        ),
        (
            "trail_activate_100",
            {
                "param_overrides.pb_delayed_confirm_trail_activate_r": 1.00,
            },
        ),
        # Quick exit for losers that never get going
        (
            "quick_exit_050_stale8",
            {
                "param_overrides.pb_delayed_confirm_quick_exit_loss_r": 0.50,
                "param_overrides.pb_delayed_confirm_stale_exit_bars": 8,
                "param_overrides.pb_delayed_confirm_stale_exit_min_r": 0.08,
            },
        ),
        (
            "quick_exit_070_stale12",
            {
                "param_overrides.pb_delayed_confirm_quick_exit_loss_r": 0.70,
                "param_overrides.pb_delayed_confirm_stale_exit_bars": 12,
                "param_overrides.pb_delayed_confirm_stale_exit_min_r": 0.05,
            },
        ),
        # VWAP failure exit
        (
            "vwap_fail_3bar_035",
            {
                "param_overrides.pb_delayed_confirm_vwap_fail_lookback_bars": 3,
                "param_overrides.pb_delayed_confirm_vwap_fail_cpr_max": 0.35,
            },
        ),
        # Lower partial take threshold (current 1.05R)
        (
            "partial_085",
            {
                "param_overrides.pb_delayed_confirm_partial_r": 0.85,
            },
        ),
        (
            "partial_075",
            {
                "param_overrides.pb_delayed_confirm_partial_r": 0.75,
            },
        ),
        # Protected widening (MFE protection + wider entry aperture)
        (
            "protect_030_rank55",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.30,
                "param_overrides.pb_entry_rank_pct_max": 55.0,
            },
        ),
        (
            "protect_030_rank70",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.30,
                "param_overrides.pb_entry_rank_pct_max": 70.0,
            },
        ),
        (
            "protect_030_sector3",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.30,
                "param_overrides.max_positions_per_sector": 3,
            },
        ),
        (
            "protect_030_floor50",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.30,
                "param_overrides.pb_daily_signal_min_score": 50.0,
            },
        ),
        (
            "protect_040_ready_relax",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.60,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.40,
                "param_overrides.pb_ready_min_cpr": 0.40,
                "param_overrides.pb_ready_min_volume_ratio": 0.55,
            },
        ),
        (
            "protect_030_combo_wide",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.30,
                "param_overrides.pb_daily_signal_min_score": 51.0,
                "param_overrides.pb_entry_rank_pct_max": 65.0,
                "param_overrides.max_positions_per_sector": 3,
            },
        ),
        (
            "protect_030_regime_any",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.30,
                "param_overrides.pb_regime_gate": "any",
            },
        ),
    ],
    # -- Phase 2: Signal Aperture + Recalibration --
    2: [
        # Raise signal floor to skip Q2 dead zone
        (
            "floor_59",
            {
                "param_overrides.pb_daily_signal_min_score": 59.0,
            },
        ),
        (
            "floor_64",
            {
                "param_overrides.pb_daily_signal_min_score": 64.0,
            },
        ),
        # Alternative signal families
        (
            "hybrid_alpha_floor52",
            {
                "param_overrides.pb_daily_signal_family": "hybrid_alpha_v1",
                "param_overrides.pb_daily_signal_min_score": 52.0,
                "param_overrides.pb_daily_rescue_min_score": 56.0,
                "param_overrides.pb_rescue_size_mult": 0.50,
            },
        ),
        (
            "quality_hybrid_floor50",
            {
                "param_overrides.pb_daily_signal_family": "quality_hybrid_v1",
                "param_overrides.pb_daily_signal_min_score": 50.0,
                "param_overrides.pb_daily_rescue_min_score": 58.0,
                "param_overrides.pb_rescue_size_mult": 0.50,
            },
        ),
        (
            "meanrev_plus_floor56",
            {
                "param_overrides.pb_daily_signal_family": "meanrev_plus_v1",
                "param_overrides.pb_daily_signal_min_score": 56.0,
                "param_overrides.pb_entry_rank_pct_max": 100.0,
            },
        ),
        (
            "sponsor_rs_floor50",
            {
                "param_overrides.pb_daily_signal_family": "sponsor_rs_hybrid_v1",
                "param_overrides.pb_daily_signal_min_score": 50.0,
                "param_overrides.pb_signal_rank_gate_mode": "percentile_only",
                "param_overrides.pb_entry_rank_pct_max": 90.0,
            },
        ),
        # CDD sweet spot enforcement (diagnostics: 4-5 days = +0.469 avg R)
        (
            "cdd_min3_max5",
            {
                "param_overrides.pb_cdd_min": 3,
                "param_overrides.pb_cdd_max": 5,
            },
        ),
        (
            "cdd_min2_max6",
            {
                "param_overrides.pb_cdd_min": 2,
                "param_overrides.pb_cdd_max": 6,
            },
        ),
        # Gap exploitation (negative gap = best alpha)
        (
            "gap_tight",
            {
                "param_overrides.pb_gap_max_pct": 0.50,
            },
        ),
        # Hard flow reject
        (
            "hard_flow_reject",
            {
                "param_overrides.pb_flow_policy": "hard_reject",
            },
        ),
        # Intraday confirm score floor
        (
            "confirm_score_min_46",
            {
                "param_overrides.pb_delayed_confirm_score_min": 46.0,
            },
        ),
        # Combo: floor + CDD sweet spot
        (
            "combo_signal_focus",
            {
                "param_overrides.pb_daily_signal_min_score": 59.0,
                "param_overrides.pb_cdd_min": 3,
                "param_overrides.pb_gap_max_pct": 1.00,
            },
        ),
        # Signal aperture relaxation
        (
            "floor_53",
            {
                "param_overrides.pb_daily_signal_min_score": 53.0,
            },
        ),
        (
            "floor_50",
            {
                "param_overrides.pb_daily_signal_min_score": 50.0,
            },
        ),
        (
            "rank_band_15_60",
            {
                "param_overrides.pb_entry_rank_pct_min": 15.0,
                "param_overrides.pb_entry_rank_pct_max": 60.0,
            },
        ),
        (
            "rank_band_0_70",
            {
                "param_overrides.pb_entry_rank_pct_min": 0.0,
                "param_overrides.pb_entry_rank_pct_max": 70.0,
            },
        ),
        (
            "confirm_score_38",
            {
                "param_overrides.pb_delayed_confirm_score_min": 38.0,
            },
        ),
        (
            "floor_52_rescue",
            {
                "param_overrides.pb_daily_signal_min_score": 52.0,
                "param_overrides.pb_rescue_flow_enabled": True,
            },
        ),
        # Aggressive structural signal changes
        (
            "rsi_entry_15",
            {
                "param_overrides.pb_rsi_entry": 15.0,
            },
        ),
        (
            "rsi_entry_20",
            {
                "param_overrides.pb_rsi_entry": 20.0,
            },
        ),
        (
            "rsi_period_3_entry_15",
            {
                "param_overrides.pb_rsi_period": 3,
                "param_overrides.pb_rsi_entry": 15.0,
            },
        ),
        (
            "combo_freq_max_p2",
            {
                "param_overrides.pb_daily_signal_min_score": 50.0,
                "param_overrides.pb_entry_rank_pct_min": 0.0,
                "param_overrides.pb_entry_rank_pct_max": 80.0,
                "param_overrides.pb_delayed_confirm_score_min": 38.0,
                "param_overrides.pb_rsi_entry": 15.0,
            },
        ),
    ],
    # -- Phase 3: Route Diversification + Capacity Expansion --
    3: [
        # Open-scored entry
        (
            "open_scored_probe",
            {
                "param_overrides.pb_open_scored_enabled": True,
                "param_overrides.pb_open_scored_min_score": 68.0,
                "param_overrides.pb_open_scored_rank_pct_max": 15.0,
                "param_overrides.pb_open_scored_max_share": 0.15,
                "param_overrides.pb_open_scored_missing_5m_allow": False,
            },
        ),
        (
            "open_scored_balanced",
            {
                "param_overrides.pb_open_scored_enabled": True,
                "param_overrides.pb_open_scored_min_score": 62.0,
                "param_overrides.pb_open_scored_rank_pct_max": 25.0,
                "param_overrides.pb_open_scored_max_share": 0.25,
                "param_overrides.pb_open_scored_missing_5m_allow": True,
            },
        ),
        # Opening reclaim
        (
            "reclaim_selective",
            {
                "param_overrides.pb_opening_reclaim_enabled": True,
                "param_overrides.pb_opening_reclaim_min_daily_signal_score": 58.0,
                "param_overrides.pb_ready_acceptance_bars": 1,
                "param_overrides.pb_flush_window_bars": 3,
                "param_overrides.pb_flush_cpr_max": 0.35,
                "param_overrides.pb_reclaim_offset_atr": 0.20,
            },
        ),
        (
            "reclaim_balanced",
            {
                "param_overrides.pb_opening_reclaim_enabled": True,
                "param_overrides.pb_opening_reclaim_min_daily_signal_score": 50.0,
                "param_overrides.pb_ready_acceptance_bars": 2,
                "param_overrides.pb_flush_window_bars": 4,
                "param_overrides.pb_reclaim_offset_atr": 0.15,
            },
        ),
        # Earlier / later delayed confirm
        (
            "delayed_after_bar4",
            {
                "param_overrides.pb_delayed_confirm_after_bar": 4,
            },
        ),
        (
            "delayed_after_bar8",
            {
                "param_overrides.pb_delayed_confirm_after_bar": 8,
            },
        ),
        # PM re-entry for stopped-out names
        (
            "pm_reentry_48",
            {
                "param_overrides.pb_pm_reentry": True,
                "param_overrides.pb_pm_reentry_after_bar": 48,
                "param_overrides.pb_max_reentries_per_day": 1,
            },
        ),
        # Priority reserve slots + open scored
        (
            "reserve_1_open_scored",
            {
                "param_overrides.pb_open_scored_enabled": True,
                "param_overrides.pb_open_scored_min_score": 62.0,
                "param_overrides.pb_open_scored_rank_pct_max": 25.0,
                "param_overrides.pb_open_scored_max_share": 0.20,
                "param_overrides.pb_intraday_priority_reserve_slots": 1,
            },
        ),
        # Combo: dual route enablement
        (
            "combo_dual_route",
            {
                "param_overrides.pb_open_scored_enabled": True,
                "param_overrides.pb_open_scored_min_score": 65.0,
                "param_overrides.pb_open_scored_rank_pct_max": 20.0,
                "param_overrides.pb_open_scored_max_share": 0.20,
                "param_overrides.pb_opening_reclaim_enabled": True,
                "param_overrides.pb_opening_reclaim_min_daily_signal_score": 55.0,
                "param_overrides.pb_intraday_priority_reserve_slots": 1,
            },
        ),
        # Relaxed delayed confirm daily signal floor
        (
            "delayed_daily_40",
            {
                "param_overrides.pb_delayed_confirm_min_daily_signal_score": 40.0,
            },
        ),
        # Capacity expansion
        (
            "sector_cap_3",
            {
                "param_overrides.max_positions_per_sector": 3,
            },
        ),
        (
            "sector_cap_4",
            {
                "param_overrides.max_positions_per_sector": 4,
            },
        ),
        (
            "positions_10",
            {
                "param_overrides.pb_max_positions": 10,
            },
        ),
        (
            "sma_dist_12_cdd6",
            {
                "param_overrides.pb_sma_dist_max_pct": 12.0,
                "param_overrides.pb_cdd_max": 6,
            },
        ),
        (
            "ready_relax_045_060",
            {
                "param_overrides.pb_ready_min_cpr": 0.45,
                "param_overrides.pb_ready_min_volume_ratio": 0.60,
            },
        ),
        # Aggressive structural capacity
        (
            "sma_dist_20_cdd8",
            {
                "param_overrides.pb_sma_dist_max_pct": 20.0,
                "param_overrides.pb_cdd_max": 8,
            },
        ),
        (
            "sector4_positions12",
            {
                "param_overrides.max_positions_per_sector": 4,
                "param_overrides.pb_max_positions": 12,
            },
        ),
        (
            "combo_capacity_routes",
            {
                "param_overrides.max_positions_per_sector": 3,
                "param_overrides.pb_max_positions": 10,
                "param_overrides.pb_open_scored_enabled": True,
                "param_overrides.pb_open_scored_min_score": 62.0,
                "param_overrides.pb_open_scored_rank_pct_max": 25.0,
                "param_overrides.pb_open_scored_max_share": 0.25,
                "param_overrides.pb_open_scored_missing_5m_allow": True,
            },
        ),
        (
            "delayed_after_bar3_ready_relax",
            {
                "param_overrides.pb_delayed_confirm_after_bar": 3,
                "param_overrides.pb_ready_min_cpr": 0.40,
                "param_overrides.pb_ready_min_volume_ratio": 0.50,
            },
        ),
    ],
    # -- Phase 4: Carry Calibration + Rescue --
    4: [
        # Flow reversal lookback (baseline default 2)
        (
            "flowrev_3_delayed",
            {
                "param_overrides.pb_delayed_confirm_flow_reversal_lookback": 3,
            },
        ),
        (
            "flowrev_1_delayed",
            {
                "param_overrides.pb_delayed_confirm_flow_reversal_lookback": 1,
            },
        ),
        # Carry score threshold tuning (baseline default 50.0)
        (
            "carry_score_55",
            {
                "param_overrides.pb_delayed_confirm_carry_score_threshold": 55.0,
            },
        ),
        (
            "carry_score_60",
            {
                "param_overrides.pb_delayed_confirm_carry_score_threshold": 60.0,
            },
        ),
        # Carry MFE gate (baseline default 0.20)
        (
            "carry_mfe_gate_025",
            {
                "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.25,
            },
        ),
        # Carry close-pct minimum (baseline default 0.62)
        (
            "carry_close_070",
            {
                "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.70,
            },
        ),
        # Max hold days variants (baseline default 4)
        (
            "carry_maxhold_3",
            {
                "param_overrides.pb_delayed_confirm_max_hold_days": 3,
            },
        ),
        # New route carry configs (if P3 enabled them)
        (
            "carry_open_high_quality",
            {
                "param_overrides.pb_open_scored_carry_min_r": 0.00,
                "param_overrides.pb_open_scored_carry_close_pct_min": 0.70,
                "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.15,
                "param_overrides.pb_open_scored_carry_min_daily_signal_score": 65.0,
                "param_overrides.pb_open_scored_carry_score_threshold": 64.0,
                "param_overrides.pb_open_scored_max_hold_days": 2,
            },
        ),
        (
            "carry_reclaim_short_leash",
            {
                "param_overrides.pb_opening_reclaim_carry_min_r": 0.15,
                "param_overrides.pb_opening_reclaim_carry_close_pct_min": 0.70,
                "param_overrides.pb_opening_reclaim_carry_mfe_gate_r": 0.20,
                "param_overrides.pb_opening_reclaim_carry_min_daily_signal_score": 62.0,
                "param_overrides.pb_opening_reclaim_carry_score_threshold": 66.0,
                "param_overrides.pb_opening_reclaim_carry_score_fallback_enabled": True,
                "param_overrides.pb_opening_reclaim_max_hold_days": 2,
            },
        ),
        # Frontier combo from diagnostics best exit frontier
        (
            "carry_frontier_combo",
            {
                "param_overrides.pb_delayed_confirm_carry_min_r": 0.10,
                "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.65,
                "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.25,
                "param_overrides.pb_delayed_confirm_carry_min_daily_signal_score": 58.0,
                "param_overrides.pb_delayed_confirm_carry_score_threshold": 58.0,
                "param_overrides.pb_delayed_confirm_carry_score_fallback_enabled": True,
                "param_overrides.pb_delayed_confirm_max_hold_days": 3,
            },
        ),
        # Rescue + frequency expansion
        (
            "rescue_flow_enabled",
            {
                "param_overrides.pb_rescue_flow_enabled": True,
            },
        ),
        (
            "carry_maxhold_5_loose",
            {
                "param_overrides.pb_delayed_confirm_max_hold_days": 5,
                "param_overrides.pb_delayed_confirm_carry_min_r": 0.15,
                "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.58,
            },
        ),
        (
            "thursday_full_risk",
            {
                "param_overrides.pb_thursday_mult": 1.0,
            },
        ),
        # Aggressive structural
        (
            "pm_reentry_rescue",
            {
                "param_overrides.pb_pm_reentry": True,
                "param_overrides.pb_pm_reentry_after_bar": 48,
                "param_overrides.pb_rescue_flow_enabled": True,
                "param_overrides.pb_rescue_max_per_day": 2,
            },
        ),
        (
            "full_week_carry5",
            {
                "param_overrides.pb_thursday_mult": 1.0,
                "param_overrides.pb_friday_mult": 1.0,
                "param_overrides.pb_delayed_confirm_max_hold_days": 5,
            },
        ),
    ],
    # -- Phase 5: Robustness Overlays + Frequency Validation --
    5: [
        # Thursday/Friday sizing
        ("thu_full_risk", {"param_overrides.pb_thursday_mult": 1.0}),
        (
            "thu_075_fri_085",
            {
                "param_overrides.pb_thursday_mult": 0.75,
                "param_overrides.pb_friday_mult": 0.85,
            },
        ),
        ("fri_075", {"param_overrides.pb_friday_mult": 0.75}),
        # Sector caps
        ("sector_cap_3", {"param_overrides.max_positions_per_sector": 3}),
        ("sector_cap_4", {"param_overrides.max_positions_per_sector": 4}),
        # ATR stop mult (baseline 1.25)
        ("atr_stop_150", {"param_overrides.pb_atr_stop_mult": 1.50}),
        ("atr_stop_100", {"param_overrides.pb_atr_stop_mult": 1.00}),
        # RSI exit threshold (currently 60 for delayed_confirm)
        ("rsi_exit_55", {"param_overrides.pb_delayed_confirm_rsi_exit": 55.0}),
        ("rsi_exit_65", {"param_overrides.pb_delayed_confirm_rsi_exit": 65.0}),
        # Min candidates per day
        ("min_candidates_6", {"param_overrides.pb_min_candidates_day": 6}),
        ("min_candidates_10", {"param_overrides.pb_min_candidates_day": 10}),
        # Reserve slots
        ("reserve_2", {"param_overrides.pb_intraday_priority_reserve_slots": 2}),
        # Frequency validation
        ("sector_cap_5", {"param_overrides.max_positions_per_sector": 5}),
        ("atr_stop_200", {"param_overrides.pb_atr_stop_mult": 2.00}),
        ("profit_target_150", {"param_overrides.pb_profit_target_r": 1.50}),
        (
            "combo_robustness_freq",
            {
                "param_overrides.max_positions_per_sector": 4,
                "param_overrides.pb_max_positions": 10,
                "param_overrides.pb_sma_dist_max_pct": 15.0,
                "param_overrides.pb_cdd_max": 7,
                "param_overrides.pb_thursday_mult": 1.0,
                "param_overrides.pb_atr_stop_mult": 1.50,
            },
        ),
    ],
}


AGGRESSIVE_ONLY_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        (
            "protect_015_after_025",
            {
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.25,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.15,
            },
        ),
        (
            "combo_aggressive_protect",
            {
                "param_overrides.pb_delayed_confirm_breakeven_r": 0.45,
                "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.50,
                "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.25,
                "param_overrides.pb_delayed_confirm_trail_activate_r": 0.95,
                "param_overrides.pb_delayed_confirm_partial_r": 0.85,
            },
        ),
    ],
    2: [
        (
            "quality_hybrid_floor48_rescue52",
            {
                "param_overrides.pb_daily_signal_family": "quality_hybrid_v1",
                "param_overrides.pb_daily_signal_min_score": 48.0,
                "param_overrides.pb_daily_rescue_min_score": 52.0,
                "param_overrides.pb_rescue_size_mult": 0.75,
                "param_overrides.pb_entry_rank_pct_max": 100.0,
            },
        ),
        (
            "quality_hard_flow_floor48",
            {
                "param_overrides.pb_daily_signal_family": "quality_hybrid_v1",
                "param_overrides.pb_flow_policy": "hard_reject",
                "param_overrides.pb_daily_signal_min_score": 48.0,
                "param_overrides.pb_daily_rescue_min_score": 60.0,
            },
        ),
    ],
    3: [
        (
            "open_scored_wide",
            {
                "param_overrides.pb_open_scored_enabled": True,
                "param_overrides.pb_open_scored_min_score": 58.0,
                "param_overrides.pb_open_scored_rank_pct_max": 35.0,
                "param_overrides.pb_open_scored_max_share": 0.35,
                "param_overrides.pb_open_scored_missing_5m_allow": True,
            },
        ),
        (
            "reclaim_fast_pm",
            {
                "param_overrides.pb_opening_reclaim_enabled": True,
                "param_overrides.pb_opening_reclaim_min_daily_signal_score": 48.0,
                "param_overrides.pb_ready_acceptance_bars": 1,
                "param_overrides.pb_pm_reentry": True,
                "param_overrides.pb_pm_reentry_after_bar": 48,
            },
        ),
    ],
    5: [
        (
            "weekday_full_risk",
            {
                "param_overrides.pb_thursday_mult": 1.0,
                "param_overrides.pb_friday_mult": 1.0,
            },
        ),
        (
            "pm_reentry_48_x2",
            {
                "param_overrides.pb_pm_reentry": True,
                "param_overrides.pb_pm_reentry_after_bar": 48,
                "param_overrides.pb_max_reentries_per_day": 2,
            },
        ),
        ("sector_cap_6", {"param_overrides.max_positions_per_sector": 6}),
    ],
}


def get_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> list[tuple[str, dict[str, Any]]]:
    experiments = list(PHASE_CANDIDATES.get(phase, []))
    if str(profile or "mainline").lower() == "aggressive":
        experiments.extend(AGGRESSIVE_ONLY_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
        )
    }


# ==========================================================================
# R5 -- Exploit Unexplored Parameter Space
# ==========================================================================
R5_BASE_MUTATIONS: dict[str, Any] = {
    # Daily signal
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_entry_rank_min": 1,
    "param_overrides.pb_entry_rank_max": 999,
    "param_overrides.pb_entry_rank_pct_min": 20.0,
    "param_overrides.pb_entry_rank_pct_max": 50.0,
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",
    "param_overrides.pb_min_candidates_day_hard_gate": False,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_cdd_max": 5,
    "param_overrides.pb_sma_dist_max_pct": 10.0,
    "param_overrides.max_positions_per_sector": 2,
    # Sizing
    "param_overrides.pb_wednesday_mult": 1.0,
    "param_overrides.pb_thursday_mult": 0.5,
    "param_overrides.pb_friday_mult": 1.0,
    # Stop
    "param_overrides.pb_atr_stop_mult": 1.25,
    # Hybrid execution
    "param_overrides.pb_execution_mode": "intraday_hybrid",
    "param_overrides.pb_carry_enabled": True,
    "param_overrides.pb_carry_score_threshold": 50.0,
    "param_overrides.pb_entry_score_min": 50.0,
    "param_overrides.pb_entry_score_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_ready_min_cpr": 0.50,
    "param_overrides.pb_ready_min_volume_ratio": 0.70,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_delayed_confirm_after_bar": 6,
    "param_overrides.pb_delayed_confirm_score_min": 42.0,
    "param_overrides.pb_delayed_confirm_min_daily_signal_score": 35.0,
    "param_overrides.pb_rescue_flow_enabled": False,
    "param_overrides.pb_opening_reclaim_enabled": True,
    "param_overrides.pb_open_scored_enabled": True,
    # Disabled protection mechanisms (Phase 3 may enable)
    "param_overrides.pb_delayed_confirm_quick_exit_loss_r": 0.0,
    "param_overrides.pb_delayed_confirm_stale_exit_bars": 0,
    "param_overrides.pb_delayed_confirm_vwap_fail_cpr_max": -1.0,
    "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.0,
    "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.0,
}


R5_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Signal Model Selection", ["signal_score_edge", "total_trades", "expected_total_r"]),
    2: ("Signal & Entry Threshold Calibration", ["avg_r", "total_trades", "expected_total_r", "profit_factor"]),
    3: ("FSM & Entry Mechanics", ["total_trades", "expected_total_r", "avg_r", "inv_dd"]),
    4: ("Carry & Exit Enhancement", ["carry_avg_r", "carry_trade_share", "expected_total_r"]),
    5: ("Robustness & Scaling", ["sharpe", "expected_total_r", "total_trades", "inv_dd"]),
}


R5_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Signal Model Selection (9 candidates) --
    1: [
        ("family_balanced_v1", {"param_overrides.pb_daily_signal_family": "balanced_v1"}),
        ("family_trend_guard", {"param_overrides.pb_daily_signal_family": "trend_guard"}),
        ("family_meanrev_v1", {"param_overrides.pb_daily_signal_family": "meanrev_v1"}),
        ("family_hybrid_alpha_v1", {"param_overrides.pb_daily_signal_family": "hybrid_alpha_v1"}),
        ("family_quality_hybrid_v1", {"param_overrides.pb_daily_signal_family": "quality_hybrid_v1"}),
        ("family_sponsor_rs_hybrid_v1", {"param_overrides.pb_daily_signal_family": "sponsor_rs_hybrid_v1"}),
        ("family_meanrev_plus_v1", {"param_overrides.pb_daily_signal_family": "meanrev_plus_v1"}),
        ("sweetspot_lower_floor_48", {"param_overrides.pb_daily_signal_min_score": 48.0}),
        ("sweetspot_higher_floor_60", {"param_overrides.pb_daily_signal_min_score": 60.0}),
    ],
    # -- Phase 2: Threshold Calibration (12 candidates) --
    2: [
        ("confirm_score_41", {"param_overrides.pb_delayed_confirm_score_min": 41.0}),
        ("confirm_score_51", {"param_overrides.pb_delayed_confirm_score_min": 51.0}),
        ("confirm_score_56", {"param_overrides.pb_delayed_confirm_score_min": 56.0}),
        ("entry_score_42", {"param_overrides.pb_entry_score_min": 42.0}),
        ("entry_score_55", {"param_overrides.pb_entry_score_min": 55.0}),
        ("signal_floor_48", {"param_overrides.pb_daily_signal_min_score": 48.0}),
        ("signal_floor_59", {"param_overrides.pb_daily_signal_min_score": 59.0}),
        ("signal_floor_64", {"param_overrides.pb_daily_signal_min_score": 64.0}),
        ("rescue_enabled_68", {"param_overrides.pb_rescue_flow_enabled": True, "param_overrides.pb_rescue_min_score": 68.0}),
        ("rescue_relaxed_60", {"param_overrides.pb_rescue_flow_enabled": True, "param_overrides.pb_rescue_min_score": 60.0}),
        ("rank_pct_max_65", {"param_overrides.pb_entry_rank_pct_max": 65.0}),
        ("rank_pct_max_80", {"param_overrides.pb_entry_rank_pct_max": 80.0}),
    ],
    # -- Phase 3: FSM & Entry Mechanics (14 candidates) --
    3: [
        ("flush_window_6", {"param_overrides.pb_flush_window_bars": 6}),
        ("flush_window_12", {"param_overrides.pb_flush_window_bars": 12}),
        ("flush_cpr_025", {"param_overrides.pb_flush_cpr_max": 0.25}),
        ("flush_cpr_055", {"param_overrides.pb_flush_cpr_max": 0.55}),
        ("ready_cpr_040", {"param_overrides.pb_ready_min_cpr": 0.40}),
        ("ready_cpr_035", {"param_overrides.pb_ready_min_cpr": 0.35}),
        ("ready_vol_040", {"param_overrides.pb_ready_min_volume_ratio": 0.40}),
        ("ready_vol_085", {"param_overrides.pb_ready_min_volume_ratio": 0.85}),
        ("reclaim_offset_005", {"param_overrides.pb_reclaim_offset_atr": 0.05}),
        ("reclaim_offset_020", {"param_overrides.pb_reclaim_offset_atr": 0.20}),
        ("acceptance_bars_2", {"param_overrides.pb_ready_acceptance_bars": 2}),
        ("delayed_after_bar_1", {"param_overrides.pb_delayed_confirm_after_bar": 1}),
        ("delayed_after_bar_6", {"param_overrides.pb_delayed_confirm_after_bar": 6}),
        ("or_signal_floor_50", {"param_overrides.pb_opening_reclaim_min_daily_signal_score": 50.0}),
    ],
    # -- Phase 4: Carry & Exit (11 candidates) --
    4: [
        ("carry_min_r_015", {"param_overrides.pb_carry_min_r": 0.15}),
        ("carry_min_r_010", {"param_overrides.pb_carry_min_r": 0.10}),
        ("carry_close_pct_040", {"param_overrides.pb_carry_close_pct_min": 0.40}),
        ("carry_close_pct_055", {"param_overrides.pb_carry_close_pct_min": 0.55}),
        ("dc_carry_close_pct_050", {"param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.50}),
        ("dc_carry_min_r_005", {"param_overrides.pb_delayed_confirm_carry_min_r": 0.05}),
        ("dc_carry_hold_5", {"param_overrides.pb_delayed_confirm_max_hold_days": 5}),
        ("or_carry_hold_4", {"param_overrides.pb_opening_reclaim_max_hold_days": 4}),
        ("rsi_exit_65", {"param_overrides.pb_rsi_exit": 65.0}),
        ("rsi_exit_75", {"param_overrides.pb_rsi_exit": 75.0}),
        ("dc_rsi_exit_55", {"param_overrides.pb_delayed_confirm_rsi_exit": 55.0}),
    ],
    # -- Phase 5: Robustness & Scaling (11 candidates) --
    5: [
        ("atr_stop_1.2", {"param_overrides.pb_atr_stop_mult": 1.2}),
        ("atr_stop_0.8", {"param_overrides.pb_atr_stop_mult": 0.8}),
        ("max_pos_10", {"param_overrides.pb_max_positions": 10}),
        ("max_pos_6", {"param_overrides.pb_max_positions": 6}),
        ("sector_cap_4", {"param_overrides.max_positions_per_sector": 4}),
        ("sector_cap_3", {"param_overrides.max_positions_per_sector": 3}),
        ("thursday_075", {"param_overrides.pb_thursday_mult": 0.75}),
        ("tuesday_115", {"param_overrides.pb_tuesday_mult": 1.15}),
        ("breakeven_060", {"param_overrides.pb_breakeven_r": 0.60}),
        ("trail_activate_100", {"param_overrides.pb_trail_activate_r": 1.00}),
        ("partial_080", {"param_overrides.pb_partial_r": 0.80}),
    ],
}


R5_AGGRESSIVE_ONLY_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        (
            "family_balanced_lower_floor_48",
            {
                "param_overrides.pb_daily_signal_family": "balanced_v1",
                "param_overrides.pb_daily_signal_min_score": 48.0,
            },
        ),
        (
            "family_hybrid_alpha_lower_floor_48",
            {
                "param_overrides.pb_daily_signal_family": "hybrid_alpha_v1",
                "param_overrides.pb_daily_signal_min_score": 48.0,
            },
        ),
    ],
    2: [
        (
            "rescue_wide_combo",
            {
                "param_overrides.pb_rescue_flow_enabled": True,
                "param_overrides.pb_rescue_min_score": 60.0,
                "param_overrides.pb_entry_rank_pct_max": 80.0,
            },
        ),
        (
            "threshold_relaxation_combo",
            {
                "param_overrides.pb_delayed_confirm_score_min": 38.0,
                "param_overrides.pb_entry_score_min": 42.0,
                "param_overrides.pb_daily_signal_min_score": 48.0,
            },
        ),
    ],
    3: [
        (
            "ready_relax_combo",
            {
                "param_overrides.pb_ready_min_cpr": 0.35,
                "param_overrides.pb_ready_min_volume_ratio": 0.40,
                "param_overrides.pb_flush_cpr_max": 0.55,
            },
        ),
        (
            "early_delayed_relax_combo",
            {
                "param_overrides.pb_delayed_confirm_after_bar": 1,
                "param_overrides.pb_ready_min_cpr": 0.40,
                "param_overrides.pb_flush_window_bars": 6,
            },
        ),
    ],
    4: [
        (
            "carry_relaxation_combo",
            {
                "param_overrides.pb_carry_min_r": 0.10,
                "param_overrides.pb_carry_close_pct_min": 0.40,
                "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.50,
            },
        ),
        (
            "exit_widening_combo",
            {
                "param_overrides.pb_rsi_exit": 75.0,
                "param_overrides.pb_delayed_confirm_rsi_exit": 55.0,
                "param_overrides.pb_delayed_confirm_max_hold_days": 5,
            },
        ),
    ],
    5: [
        (
            "capacity_expansion_combo",
            {
                "param_overrides.pb_max_positions": 10,
                "param_overrides.max_positions_per_sector": 4,
                "param_overrides.pb_thursday_mult": 0.75,
            },
        ),
        (
            "stop_trail_combo",
            {
                "param_overrides.pb_atr_stop_mult": 0.8,
                "param_overrides.pb_trail_activate_r": 1.00,
                "param_overrides.pb_breakeven_r": 0.60,
            },
        ),
    ],
}


def get_r5_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> list[tuple[str, dict[str, Any]]]:
    experiments = list(R5_PHASE_CANDIDATES.get(phase, []))
    if str(profile or "mainline").lower() == "aggressive":
        experiments.extend(R5_AGGRESSIVE_ONLY_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_r5_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_r5_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
        )
    }


# ==========================================================================
# V2R1 -- Phased Auto-Optimization for V2 Hybrid Engine
# Baseline: 815 trades, PF 1.37, avg_r +0.054, DD 3.3%, Sharpe 1.80
# ==========================================================================
V2R1_BASE_MUTATIONS: dict[str, Any] = {
    # Engine mode
    "param_overrides.pb_v2_enabled": True,
    "param_overrides.pb_execution_mode": "intraday_hybrid",

    # Daily signal
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",

    # V2 signal floor
    "param_overrides.pb_v2_signal_floor": 75.0,

    # Trend filter
    "param_overrides.pb_v2_allow_secular": True,
    "param_overrides.pb_v2_secular_sizing_mult": 0.65,

    # Entry routes
    "param_overrides.pb_v2_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_max_slots": 4,
    "param_overrides.pb_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_rank_pct_max": 100.0,
    "param_overrides.pb_v2_open_scored_min_score": 45.0,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_v2_vwap_bounce_enabled": True,
    "param_overrides.pb_v2_afternoon_retest_enabled": True,

    # Stops & protection
    "param_overrides.pb_atr_stop_mult": 1.5,
    "param_overrides.pb_v2_mfe_stage1_trigger": 0.50,
    "param_overrides.pb_v2_mfe_stage1_stop_r": -0.20,
    "param_overrides.pb_v2_mfe_stage2_trigger": 0.75,
    "param_overrides.pb_v2_mfe_stage3_trigger": 1.25,
    "param_overrides.pb_v2_mfe_stage3_trail_atr": 0.75,
    "param_overrides.pb_v2_partial_profit_trigger_r": 1.50,

    # Exits
    "param_overrides.pb_v2_ema_reversion_exit": True,
    "param_overrides.pb_v2_ema_reversion_min_r": 0.10,
    "param_overrides.pb_v2_rsi_exit_open_scored": 60.0,

    # Carry
    "param_overrides.pb_v2_carry_overnight_stop_atr": 1.0,
    "param_overrides.pb_v2_flatten_loss_r": -0.50,
    "param_overrides.pb_v2_flow_grace_days": 2,
    "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
    "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.10,
    "param_overrides.pb_open_scored_flow_reversal_lookback": 2,
    "param_overrides.pb_max_hold_days": 5,
    "param_overrides.pb_max_positions": 8,
    "param_overrides.max_positions_per_sector": 5,
}


V2R1_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Stop, MFE Protection & Exit Priority", ["avg_r", "expected_total_r", "profit_factor", "protection_candidate_inverse", "stop_hit_inverse"]),
    2: ("Signal & Scoring", ["signal_score_edge", "expected_total_r", "avg_r", "total_trades", "crowded_day_discrimination"]),
    3: ("Route Diversification", ["route_diversity", "expected_total_r", "total_trades", "route_score_monotonicity"]),
    4: ("Carry & Overnight", ["carry_avg_r", "expected_total_r", "avg_r", "eod_flatten_inverse"]),
    5: ("Robustness & Sizing", ["sharpe", "expected_total_r", "avg_r", "inv_dd"]),
}


V2R1_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Stop, MFE Protection & Exit Priority (41 candidates) --
    1: [
        # EMA reversion min R -- baseline 0.10
        ("ema_min_005", {"param_overrides.pb_v2_ema_reversion_min_r": 0.05}),
        ("ema_min_015", {"param_overrides.pb_v2_ema_reversion_min_r": 0.15}),
        ("ema_min_020", {"param_overrides.pb_v2_ema_reversion_min_r": 0.20}),
        ("ema_min_000", {"param_overrides.pb_v2_ema_reversion_min_r": 0.00}),
        # EMA exit ablation
        ("ema_exit_off", {"param_overrides.pb_v2_ema_reversion_exit": False}),
        # MFE Stage 1 trigger -- baseline 0.50R
        ("mfe1_020", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.20}),
        ("mfe1_025", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.25}),
        ("mfe1_030", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.30}),
        ("mfe1_040", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.40}),
        # MFE Stage 1 stop level -- baseline -0.20R
        ("mfe1_stop_030", {"param_overrides.pb_v2_mfe_stage1_stop_r": -0.30}),
        ("mfe1_stop_050", {"param_overrides.pb_v2_mfe_stage1_stop_r": -0.50}),
        ("mfe1_stop_010", {"param_overrides.pb_v2_mfe_stage1_stop_r": -0.10}),
        # MFE Stage 2 breakeven trigger -- baseline 0.75R
        ("mfe2_030", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.30}),
        ("mfe2_040", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.40}),
        ("mfe2_050", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.50}),
        ("mfe2_060", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.60}),
        # MFE Stage 3 trail activation -- baseline 1.25R
        ("mfe3_050", {"param_overrides.pb_v2_mfe_stage3_trigger": 0.50}),
        ("mfe3_075", {"param_overrides.pb_v2_mfe_stage3_trigger": 0.75}),
        ("mfe3_100", {"param_overrides.pb_v2_mfe_stage3_trigger": 1.00}),
        # Trail distance -- baseline 0.75 ATR
        ("trail_050", {"param_overrides.pb_v2_mfe_stage3_trail_atr": 0.50}),
        ("trail_100", {"param_overrides.pb_v2_mfe_stage3_trail_atr": 1.00}),
        # Partial profit trigger -- baseline 1.50R
        ("partial_075", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.75}),
        ("partial_100", {"param_overrides.pb_v2_partial_profit_trigger_r": 1.00}),
        ("partial_125", {"param_overrides.pb_v2_partial_profit_trigger_r": 1.25}),
        # ATR stop multiplier -- baseline 1.50
        ("atr_stop_100", {"param_overrides.pb_atr_stop_mult": 1.00}),
        ("atr_stop_125", {"param_overrides.pb_atr_stop_mult": 1.25}),
        ("atr_stop_175", {"param_overrides.pb_atr_stop_mult": 1.75}),
        ("atr_stop_200", {"param_overrides.pb_atr_stop_mult": 2.00}),
        # Gap stop (overnight) -- baseline 1.0 ATR
        ("gap_stop_125", {"param_overrides.pb_v2_carry_overnight_stop_atr": 1.25}),
        ("gap_stop_150", {"param_overrides.pb_v2_carry_overnight_stop_atr": 1.50}),
        ("gap_stop_200", {"param_overrides.pb_v2_carry_overnight_stop_atr": 2.00}),
        # VWAP fail exit
        ("vwap_fail_bars_2", {"param_overrides.pb_v2_vwap_fail_bars": 2}),
        ("vwap_fail_bars_5", {"param_overrides.pb_v2_vwap_fail_bars": 5}),
        ("vwap_fail_close_025", {"param_overrides.pb_v2_vwap_fail_close_pct": 0.25}),
        # Stale position management
        ("stale_bars_6", {"param_overrides.pb_v2_stale_bars": 6}),
        ("stale_mfe_005", {"param_overrides.pb_v2_stale_mfe_thresh": 0.05}),
        # Combo candidates
        ("protect_early", {
            "param_overrides.pb_v2_mfe_stage1_trigger": 0.25,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.40,
            "param_overrides.pb_v2_mfe_stage3_trigger": 0.75,
        }),
        ("protect_aggressive", {
            "param_overrides.pb_v2_mfe_stage1_trigger": 0.20,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.30,
            "param_overrides.pb_v2_mfe_stage3_trigger": 0.50,
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.75,
        }),
        ("gap_and_protect", {
            "param_overrides.pb_v2_carry_overnight_stop_atr": 1.50,
            "param_overrides.pb_v2_mfe_stage1_trigger": 0.30,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.50,
        }),
        ("ema_low_mfe_low", {
            "param_overrides.pb_v2_ema_reversion_min_r": 0.05,
            "param_overrides.pb_v2_mfe_stage1_trigger": 0.25,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.40,
        }),
        ("ema_high_mfe_low", {
            "param_overrides.pb_v2_ema_reversion_min_r": 0.20,
            "param_overrides.pb_v2_mfe_stage1_trigger": 0.25,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.40,
            "param_overrides.pb_v2_mfe_stage3_trigger": 0.75,
        }),
    ],

    # -- Phase 2: Signal & Scoring (31 candidates) --
    2: [
        # Signal floor -- baseline 75.0
        ("floor_65", {"param_overrides.pb_v2_signal_floor": 65.0}),
        ("floor_70", {"param_overrides.pb_v2_signal_floor": 70.0}),
        ("floor_80", {"param_overrides.pb_v2_signal_floor": 80.0}),
        ("floor_85", {"param_overrides.pb_v2_signal_floor": 85.0}),
        # Daily signal min score -- baseline 54.0
        ("daily_min_48", {"param_overrides.pb_daily_signal_min_score": 48.0}),
        ("daily_min_50", {"param_overrides.pb_daily_signal_min_score": 50.0}),
        ("daily_min_58", {"param_overrides.pb_daily_signal_min_score": 58.0}),
        ("daily_min_62", {"param_overrides.pb_daily_signal_min_score": 62.0}),
        # Flow rescue policy
        ("rescue_off", {"param_overrides.pb_flow_policy": "hard_reject"}),
        ("rescue_soft_only", {"param_overrides.pb_flow_policy": "soft_penalty"}),
        ("rescue_tight_60", {"param_overrides.pb_daily_rescue_min_score": 60.0}),
        ("rescue_tight_65", {"param_overrides.pb_daily_rescue_min_score": 65.0}),
        ("rescue_tight_70", {"param_overrides.pb_daily_rescue_min_score": 70.0}),
        # Rescue sizing -- baseline 0.65
        ("rescue_size_040", {"param_overrides.pb_rescue_size_mult": 0.40}),
        ("rescue_size_050", {"param_overrides.pb_rescue_size_mult": 0.50}),
        # Daily signal family
        ("family_balanced", {"param_overrides.pb_daily_signal_family": "balanced_v1"}),
        ("family_hybrid_alpha", {"param_overrides.pb_daily_signal_family": "hybrid_alpha_v1"}),
        ("family_quality_hybrid", {"param_overrides.pb_daily_signal_family": "quality_hybrid_v1"}),
        ("family_meanrev_plus", {"param_overrides.pb_daily_signal_family": "meanrev_plus_v1"}),
        # V2 rank filter -- requires code fix 2A
        ("v2_rank_pct_30", {"param_overrides.pb_v2_open_scored_rank_pct_max": 30.0}),
        ("v2_rank_pct_50", {"param_overrides.pb_v2_open_scored_rank_pct_max": 50.0}),
        ("v2_rank_pct_75", {"param_overrides.pb_v2_open_scored_rank_pct_max": 75.0}),
        # V2 open scored min score -- baseline 45.0
        ("v2_open_min_55", {"param_overrides.pb_v2_open_scored_min_score": 55.0}),
        ("v2_open_min_65", {"param_overrides.pb_v2_open_scored_min_score": 65.0}),
        ("v2_open_min_35", {"param_overrides.pb_v2_open_scored_min_score": 35.0}),
        # SMA distance filter
        ("sma_dist_max_15", {"param_overrides.pb_v2_sma_dist_max_pct": 15.0}),
        ("sma_dist_max_12", {"param_overrides.pb_v2_sma_dist_max_pct": 12.0}),
        # Gap filter
        ("gap_max_3", {"param_overrides.pb_v2_gap_max_pct": 3.0}),
        ("gap_min_8", {"param_overrides.pb_v2_gap_min_pct": -8.0}),
        # Signal + rescue combos
        ("tight_signal", {
            "param_overrides.pb_v2_signal_floor": 80.0,
            "param_overrides.pb_flow_policy": "hard_reject",
            "param_overrides.pb_daily_signal_min_score": 58.0,
        }),
        ("balanced_signal", {
            "param_overrides.pb_v2_signal_floor": 70.0,
            "param_overrides.pb_daily_rescue_min_score": 65.0,
            "param_overrides.pb_daily_signal_min_score": 50.0,
            "param_overrides.pb_v2_sma_dist_max_pct": 15.0,
        }),
    ],

    # -- Phase 3: Route Diversification (22 candidates) --
    3: [
        # DELAYED_CONFIRM relaxation -- currently 14 trades, +0.196R
        ("delayed_close_030", {"param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.30}),
        ("delayed_close_035", {"param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.35}),
        ("delayed_vol_030", {"param_overrides.pb_v2_delayed_confirm_vol_ratio": 0.30}),
        ("delayed_vol_040", {"param_overrides.pb_v2_delayed_confirm_vol_ratio": 0.40}),
        ("delayed_score_30", {"param_overrides.pb_delayed_confirm_score_min": 30.0}),
        ("delayed_after_3", {"param_overrides.pb_delayed_confirm_after_bar": 3}),
        ("delayed_after_1", {"param_overrides.pb_delayed_confirm_after_bar": 1}),
        # VWAP_BOUNCE -- requires code fix 2B
        ("vwap_bounce_rescue", {"param_overrides.pb_v2_vwap_bounce_allow_rescue": True}),
        ("vwap_bounce_after_6", {
            "param_overrides.pb_v2_vwap_bounce_after_bar": 6,
            "param_overrides.pb_v2_vwap_bounce_allow_rescue": True,
        }),
        ("vwap_bounce_vol_040", {
            "param_overrides.pb_v2_vwap_bounce_vol_ratio": 0.40,
            "param_overrides.pb_v2_vwap_bounce_allow_rescue": True,
        }),
        # AFTERNOON_RETEST -- requires code fix 2B
        ("afternoon_rescue", {"param_overrides.pb_v2_afternoon_retest_allow_rescue": True}),
        ("afternoon_after_36", {
            "param_overrides.pb_v2_afternoon_retest_after_bar": 36,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
        }),
        ("afternoon_score_30", {
            "param_overrides.pb_v2_afternoon_retest_min_score": 30.0,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
        }),
        ("afternoon_sizing_100", {
            "param_overrides.pb_v2_afternoon_retest_sizing_mult": 1.0,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
        }),
        # OPENING_RECLAIM -- currently disabled
        ("reclaim_enabled", {"param_overrides.pb_opening_reclaim_enabled": True}),
        ("reclaim_signal_40", {
            "param_overrides.pb_opening_reclaim_enabled": True,
            "param_overrides.pb_opening_reclaim_min_daily_signal_score": 40.0,
        }),
        ("reclaim_signal_50", {
            "param_overrides.pb_opening_reclaim_enabled": True,
            "param_overrides.pb_opening_reclaim_min_daily_signal_score": 50.0,
        }),
        # Open scored max slots -- baseline 4
        ("open_slots_3", {"param_overrides.pb_v2_open_scored_max_slots": 3}),
        ("open_slots_5", {"param_overrides.pb_v2_open_scored_max_slots": 5}),
        ("open_slots_2", {"param_overrides.pb_v2_open_scored_max_slots": 2}),
        # Route combos
        ("route_diversity", {
            "param_overrides.pb_v2_open_scored_max_slots": 3,
            "param_overrides.pb_opening_reclaim_enabled": True,
            "param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.35,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
        }),
        ("route_rescue_unlock", {
            "param_overrides.pb_v2_open_scored_max_slots": 3,
            "param_overrides.pb_v2_vwap_bounce_allow_rescue": True,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
            "param_overrides.pb_delayed_confirm_after_bar": 3,
        }),
    ],

    # -- Phase 4: Carry & Overnight (22 candidates) --
    4: [
        # Flatten loss threshold -- baseline -0.50R
        ("flatten_loss_030", {"param_overrides.pb_v2_flatten_loss_r": -0.30}),
        ("flatten_loss_075", {"param_overrides.pb_v2_flatten_loss_r": -0.75}),
        ("flatten_loss_100", {"param_overrides.pb_v2_flatten_loss_r": -1.00}),
        # Max hold days -- baseline 5
        ("hold_3", {
            "param_overrides.pb_max_hold_days": 3,
            "param_overrides.pb_open_scored_max_hold_days": 3,
        }),
        ("hold_4", {
            "param_overrides.pb_max_hold_days": 4,
            "param_overrides.pb_open_scored_max_hold_days": 4,
        }),
        ("hold_7", {
            "param_overrides.pb_max_hold_days": 7,
            "param_overrides.pb_open_scored_max_hold_days": 7,
        }),
        ("hold_2", {
            "param_overrides.pb_max_hold_days": 2,
            "param_overrides.pb_open_scored_max_hold_days": 2,
        }),
        # Flow grace days -- baseline 2
        ("flow_grace_1", {"param_overrides.pb_v2_flow_grace_days": 1}),
        ("flow_grace_3", {"param_overrides.pb_v2_flow_grace_days": 3}),
        ("flow_grace_0", {"param_overrides.pb_v2_flow_grace_days": 0}),
        # Carry close_pct gate -- baseline 0.55
        ("carry_close_040", {"param_overrides.pb_open_scored_carry_close_pct_min": 0.40}),
        ("carry_close_065", {"param_overrides.pb_open_scored_carry_close_pct_min": 0.65}),
        ("carry_close_075", {"param_overrides.pb_open_scored_carry_close_pct_min": 0.75}),
        # Carry MFE gate -- baseline 0.10
        ("carry_mfe_000", {"param_overrides.pb_open_scored_carry_mfe_gate_r": 0.00}),
        ("carry_mfe_020", {"param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20}),
        ("carry_mfe_030", {"param_overrides.pb_open_scored_carry_mfe_gate_r": 0.30}),
        # RSI exit thresholds
        ("rsi_exit_50", {"param_overrides.pb_v2_rsi_exit_open_scored": 50.0}),
        ("rsi_exit_55", {"param_overrides.pb_v2_rsi_exit_open_scored": 55.0}),
        ("rsi_exit_70", {"param_overrides.pb_v2_rsi_exit_open_scored": 70.0}),
        # Carry combos
        ("strict_carry", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.65,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_v2_flatten_loss_r": -0.30,
        }),
        ("lenient_carry", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.40,
            "param_overrides.pb_v2_flatten_loss_r": -0.75,
            "param_overrides.pb_max_hold_days": 7,
            "param_overrides.pb_open_scored_max_hold_days": 7,
            "param_overrides.pb_v2_flow_grace_days": 3,
        }),
        ("no_carry", {
            "param_overrides.pb_max_hold_days": 1,
            "param_overrides.pb_open_scored_max_hold_days": 1,
        }),
    ],

    # -- Phase 5: Robustness & Sizing (15 candidates) --
    5: [
        # Day-of-week sizing
        ("dow_thu_080", {"param_overrides.pb_thursday_mult": 0.80}),
        ("dow_thu_060", {"param_overrides.pb_thursday_mult": 0.60}),
        # Position limits
        ("max_pos_6", {"param_overrides.pb_max_positions": 6}),
        ("max_pos_10", {"param_overrides.pb_max_positions": 10}),
        ("max_pos_12", {"param_overrides.pb_max_positions": 12}),
        # Sector cap
        ("sector_max_3", {"param_overrides.max_positions_per_sector": 3}),
        ("sector_max_4", {"param_overrides.max_positions_per_sector": 4}),
        ("sector_max_2", {"param_overrides.max_positions_per_sector": 2}),
        # Secular trend sizing -- baseline 0.65
        ("secular_050", {"param_overrides.pb_v2_secular_sizing_mult": 0.50}),
        ("secular_080", {"param_overrides.pb_v2_secular_sizing_mult": 0.80}),
        ("secular_off", {"param_overrides.pb_v2_allow_secular": False}),
        # V2 sizing tiers
        ("sizing_flat", {
            "param_overrides.pb_v2_sizing_premium": 1.0,
            "param_overrides.pb_v2_sizing_standard": 1.0,
            "param_overrides.pb_v2_sizing_reduced": 1.0,
            "param_overrides.pb_v2_sizing_minimum": 1.0,
        }),
        ("sizing_steep", {
            "param_overrides.pb_v2_sizing_premium": 1.0,
            "param_overrides.pb_v2_sizing_standard": 0.65,
            "param_overrides.pb_v2_sizing_reduced": 0.40,
            "param_overrides.pb_v2_sizing_minimum": 0.20,
        }),
        # Min candidates day
        ("min_cand_5", {"param_overrides.pb_min_candidates_day": 5}),
        ("min_cand_12", {"param_overrides.pb_min_candidates_day": 12}),
    ],
}


def get_v2r1_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> list[tuple[str, dict[str, Any]]]:
    del profile  # V2R1 uses a single candidate set
    experiments = list(V2R1_PHASE_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v2r1_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v2r1_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
        )
    }


# ==========================================================================
# V2R2 -- Alpha-Maximizing Phased Optimization (Post-Phase-3 Restart)
# Baseline: 810 trades, PF 1.55, avg_r +0.052, DD 1.9%, Sharpe 2.30
# 3 phases, 55 candidates, immutable alpha-focused scoring
# ==========================================================================
V2R2_BASE_MUTATIONS: dict[str, Any] = {
    # Engine mode
    "param_overrides.pb_v2_enabled": True,
    "param_overrides.pb_execution_mode": "intraday_hybrid",

    # Daily signal
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",

    # V2 signal floor
    "param_overrides.pb_v2_signal_floor": 75.0,

    # Trend filter
    "param_overrides.pb_v2_allow_secular": True,
    "param_overrides.pb_v2_secular_sizing_mult": 0.65,

    # Entry routes
    "param_overrides.pb_v2_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_max_slots": 4,
    "param_overrides.pb_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_rank_pct_max": 100.0,
    "param_overrides.pb_v2_open_scored_min_score": 45.0,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_v2_vwap_bounce_enabled": True,
    "param_overrides.pb_v2_afternoon_retest_enabled": True,

    # Phase 1 accepted: Stop/MFE improvements
    "param_overrides.pb_atr_stop_mult": 2.00,
    "param_overrides.pb_v2_partial_profit_trigger_r": 0.75,
    "param_overrides.pb_v2_ema_reversion_min_r": 0.05,
    "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
    "param_overrides.pb_v2_stale_bars": 6,
    "param_overrides.pb_v2_mfe_stage1_stop_r": -0.10,
    "param_overrides.pb_v2_mfe_stage2_trigger": 0.60,

    # Phase 3 accepted: Route activation
    "param_overrides.pb_opening_reclaim_enabled": True,

    # Unchanged MFE/exit params
    "param_overrides.pb_v2_mfe_stage1_trigger": 0.50,
    "param_overrides.pb_v2_mfe_stage3_trigger": 1.25,
    "param_overrides.pb_v2_mfe_stage3_trail_atr": 0.75,
    "param_overrides.pb_v2_ema_reversion_exit": True,
    "param_overrides.pb_v2_rsi_exit_open_scored": 60.0,

    # Carry (Phase 1 will optimize these)
    "param_overrides.pb_v2_carry_overnight_stop_atr": 1.0,
    "param_overrides.pb_v2_flatten_loss_r": -0.50,
    "param_overrides.pb_v2_flow_grace_days": 2,
    "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
    "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.10,
    "param_overrides.pb_open_scored_flow_reversal_lookback": 2,
    "param_overrides.pb_max_hold_days": 5,
    "param_overrides.pb_max_positions": 8,
    "param_overrides.max_positions_per_sector": 5,
}


V2R2_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Smart Carry & Exit", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    2: ("Signal Quality & Frequency", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    3: ("Route Expansion & Capture", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
}


V2R2_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Smart Carry & Exit Optimization (20 candidates) --
    1: [
        # Carry close_pct gate -- baseline 0.55
        ("carry_close_060", {"param_overrides.pb_open_scored_carry_close_pct_min": 0.60}),
        ("carry_close_065", {"param_overrides.pb_open_scored_carry_close_pct_min": 0.65}),
        ("carry_close_070", {"param_overrides.pb_open_scored_carry_close_pct_min": 0.70}),
        ("carry_close_075", {"param_overrides.pb_open_scored_carry_close_pct_min": 0.75}),
        # Carry MFE gate -- baseline 0.10R
        ("carry_mfe_015", {"param_overrides.pb_open_scored_carry_mfe_gate_r": 0.15}),
        ("carry_mfe_020", {"param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20}),
        ("carry_mfe_025", {"param_overrides.pb_open_scored_carry_mfe_gate_r": 0.25}),
        # Overnight gap stop -- baseline 1.0 ATR
        ("gap_stop_125", {"param_overrides.pb_v2_carry_overnight_stop_atr": 1.25}),
        ("gap_stop_150", {"param_overrides.pb_v2_carry_overnight_stop_atr": 1.50}),
        # Flatten loss threshold -- baseline -0.50R
        ("flatten_loss_025", {"param_overrides.pb_v2_flatten_loss_r": -0.25}),
        ("flatten_loss_035", {"param_overrides.pb_v2_flatten_loss_r": -0.35}),
        # Max hold days -- baseline 5
        ("hold_3", {
            "param_overrides.pb_max_hold_days": 3,
            "param_overrides.pb_open_scored_max_hold_days": 3,
        }),
        ("hold_2", {
            "param_overrides.pb_max_hold_days": 2,
            "param_overrides.pb_open_scored_max_hold_days": 2,
        }),
        # Flow grace days -- baseline 2
        ("flow_grace_1", {"param_overrides.pb_v2_flow_grace_days": 1}),
        ("flow_grace_0", {"param_overrides.pb_v2_flow_grace_days": 0}),
        # EMA min_r interaction -- baseline 0.05
        ("ema_min_000", {"param_overrides.pb_v2_ema_reversion_min_r": 0.00}),
        ("ema_min_003", {"param_overrides.pb_v2_ema_reversion_min_r": 0.03}),
        # Structural combos
        ("strict_carry", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.65,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_v2_flatten_loss_r": -0.35,
        }),
        ("tight_carry", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.70,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.25,
            "param_overrides.pb_max_hold_days": 3,
            "param_overrides.pb_open_scored_max_hold_days": 3,
            "param_overrides.pb_v2_flow_grace_days": 1,
        }),
        ("ema_carry_combo", {
            "param_overrides.pb_v2_ema_reversion_min_r": 0.00,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.65,
            "param_overrides.pb_v2_carry_overnight_stop_atr": 1.50,
        }),
    ],

    # -- Phase 2: Signal Quality & Frequency (21 candidates) --
    2: [
        # Signal floor -- baseline 75.0
        ("floor_65", {"param_overrides.pb_v2_signal_floor": 65.0}),
        ("floor_70", {"param_overrides.pb_v2_signal_floor": 70.0}),
        ("floor_80", {"param_overrides.pb_v2_signal_floor": 80.0}),
        # Rescue policy -- 76% of trades enter via rescue
        ("rescue_off", {"param_overrides.pb_flow_policy": "hard_reject"}),
        ("rescue_tight_60", {"param_overrides.pb_daily_rescue_min_score": 60.0}),
        ("rescue_tight_65", {"param_overrides.pb_daily_rescue_min_score": 65.0}),
        ("rescue_size_040", {"param_overrides.pb_rescue_size_mult": 0.40}),
        # Signal family
        ("family_balanced", {"param_overrides.pb_daily_signal_family": "balanced_v1"}),
        ("family_meanrev_plus", {"param_overrides.pb_daily_signal_family": "meanrev_plus_v1"}),
        ("family_quality_hybrid", {"param_overrides.pb_daily_signal_family": "quality_hybrid_v1"}),
        # V2 rank filter
        ("v2_rank_pct_50", {"param_overrides.pb_v2_open_scored_rank_pct_max": 50.0}),
        ("v2_rank_pct_75", {"param_overrides.pb_v2_open_scored_rank_pct_max": 75.0}),
        # V2 open scored min score -- baseline 45.0
        ("v2_open_min_35", {"param_overrides.pb_v2_open_scored_min_score": 35.0}),
        ("v2_open_min_55", {"param_overrides.pb_v2_open_scored_min_score": 55.0}),
        # SMA distance filter
        ("sma_dist_max_15", {"param_overrides.pb_v2_sma_dist_max_pct": 15.0}),
        # Daily min score & breadth
        ("daily_min_50", {"param_overrides.pb_daily_signal_min_score": 50.0}),
        ("daily_min_58", {"param_overrides.pb_daily_signal_min_score": 58.0}),
        ("min_cand_5", {"param_overrides.pb_min_candidates_day": 5}),
        # Structural combos
        ("quality_focused", {
            "param_overrides.pb_v2_signal_floor": 80.0,
            "param_overrides.pb_daily_rescue_min_score": 65.0,
            "param_overrides.pb_v2_sma_dist_max_pct": 15.0,
        }),
        ("wide_funnel", {
            "param_overrides.pb_v2_signal_floor": 65.0,
            "param_overrides.pb_daily_signal_min_score": 50.0,
            "param_overrides.pb_v2_open_scored_min_score": 35.0,
            "param_overrides.pb_min_candidates_day": 5,
        }),
        ("rescue_quality", {
            "param_overrides.pb_daily_rescue_min_score": 60.0,
            "param_overrides.pb_rescue_size_mult": 0.40,
            "param_overrides.pb_v2_signal_floor": 70.0,
        }),
    ],

    # -- Phase 3: Route Expansion & Profit Capture (14 candidates) --
    3: [
        # DELAYED_CONFIRM expansion -- +0.196R avg, 78.6% WR
        ("delayed_close_030", {"param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.30}),
        ("delayed_close_035", {"param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.35}),
        ("delayed_after_3", {"param_overrides.pb_delayed_confirm_after_bar": 3}),
        # VWAP_BOUNCE activation
        ("vwap_bounce_rescue", {"param_overrides.pb_v2_vwap_bounce_allow_rescue": True}),
        ("vwap_bounce_combo", {
            "param_overrides.pb_v2_vwap_bounce_allow_rescue": True,
            "param_overrides.pb_v2_vwap_bounce_after_bar": 6,
        }),
        # AFTERNOON_RETEST activation
        ("afternoon_rescue", {"param_overrides.pb_v2_afternoon_retest_allow_rescue": True}),
        # Open scored slots -- baseline 4
        ("open_slots_5", {"param_overrides.pb_v2_open_scored_max_slots": 5}),
        ("open_slots_3", {"param_overrides.pb_v2_open_scored_max_slots": 3}),
        # RSI exit -- baseline 60
        ("rsi_exit_50", {"param_overrides.pb_v2_rsi_exit_open_scored": 50.0}),
        ("rsi_exit_55", {"param_overrides.pb_v2_rsi_exit_open_scored": 55.0}),
        # Position limits -- baseline 8
        ("max_pos_10", {"param_overrides.pb_max_positions": 10}),
        ("max_pos_12", {"param_overrides.pb_max_positions": 12}),
        # Structural combos
        ("route_unlock", {
            "param_overrides.pb_v2_open_scored_max_slots": 3,
            "param_overrides.pb_v2_vwap_bounce_allow_rescue": True,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
            "param_overrides.pb_delayed_confirm_after_bar": 3,
        }),
        ("capture_max", {
            "param_overrides.pb_v2_open_scored_max_slots": 5,
            "param_overrides.pb_max_positions": 12,
            "param_overrides.pb_v2_rsi_exit_open_scored": 55.0,
        }),
    ],
}


def get_v2r2_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> list[tuple[str, dict[str, Any]]]:
    del profile  # V2R2 uses a single candidate set
    experiments = list(V2R2_PHASE_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v2r2_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v2r2_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
        )
    }


# ==========================================================================
# V2R3 -- Structural Engine Fixes + Re-optimization
# Baseline: V2R2 final (832 trades, PF 1.55, avg_r +0.055, DD 2.1%)
# 2 phases: carry quality tuning (12), route expansion (10)
# Key change: V2 carry quality gate now active, DELAYED_CONFIRM rescue
# ==========================================================================
V2R3_BASE_MUTATIONS: dict[str, Any] = {
    # Inherit entire V2R2 base
    **V2R2_BASE_MUTATIONS,
    # V2R2 accepted overrides (hold_2, ema_min_003, max_pos_10)
    "param_overrides.pb_max_hold_days": 2,
    "param_overrides.pb_open_scored_max_hold_days": 2,
    "param_overrides.pb_v2_ema_reversion_min_r": 0.03,
    "param_overrides.pb_max_positions": 10,
    # Carry gate pass-through (unconditional carry is net positive; gate tested negative)
    "param_overrides.pb_open_scored_carry_close_pct_min": 0.0,
    "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.0,
    "param_overrides.pb_carry_close_pct_min": 0.0,
    "param_overrides.pb_carry_mfe_gate_r": 0.0,
}


V2R3_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Route Expansion", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
}


V2R3_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Route Expansion (10 candidates) --
    1: [
        # DELAYED_CONFIRM rescue unlock
        ("delayed_rescue", {"param_overrides.pb_v2_delayed_confirm_allow_rescue": True}),
        ("delayed_rescue_bar6", {
            "param_overrides.pb_v2_delayed_confirm_allow_rescue": True,
            "param_overrides.pb_delayed_confirm_after_bar": 6,
        }),
        ("delayed_rescue_bar4", {
            "param_overrides.pb_v2_delayed_confirm_allow_rescue": True,
            "param_overrides.pb_delayed_confirm_after_bar": 4,
        }),
        # Earlier delayed confirm arm (no rescue)
        ("delayed_bar6", {"param_overrides.pb_delayed_confirm_after_bar": 6}),
        ("delayed_bar4", {"param_overrides.pb_delayed_confirm_after_bar": 4}),
        # Re-test rescue routes with restored unconditional carry baseline
        ("vwap_bounce_rescue", {"param_overrides.pb_v2_vwap_bounce_allow_rescue": True}),
        ("afternoon_rescue", {"param_overrides.pb_v2_afternoon_retest_allow_rescue": True}),
        # All routes rescue
        ("all_routes_rescue", {
            "param_overrides.pb_v2_delayed_confirm_allow_rescue": True,
            "param_overrides.pb_v2_vwap_bounce_allow_rescue": True,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
        }),
        # Full unlock combo
        ("route_unlock", {
            "param_overrides.pb_v2_delayed_confirm_allow_rescue": True,
            "param_overrides.pb_delayed_confirm_after_bar": 4,
            "param_overrides.pb_v2_vwap_bounce_allow_rescue": True,
            "param_overrides.pb_v2_afternoon_retest_allow_rescue": True,
        }),
        # Flatten loss tighter (tested ~neutral in carry-gate run, re-test on clean baseline)
        ("flatten_loss_035", {"param_overrides.pb_v2_flatten_loss_r": -0.35}),
    ],
}


def get_v2r3_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> list[tuple[str, dict[str, Any]]]:
    del profile  # V2R3 uses a single candidate set
    experiments = list(V2R3_PHASE_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v2r3_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v2r3_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
        )
    }


# ==========================================================================
# V2R4 -- Overnight Profit Lock + Intraday Protection
# Baseline: V2R3 final (832 trades, PF 1.58, avg_r +0.054, DD 2.1%)
# ==========================================================================
V2R4_BASE_MUTATIONS: dict[str, Any] = {
    **V2R3_BASE_MUTATIONS,
}


V2R4_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Overnight Profit Lock", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    2: ("Intraday Protection + Capacity", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
}


V2R4_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Overnight Profit Lock (11 candidates) --
    1: [
        # Pure profit lock variants (parameterize the hardcoded 0.75 threshold)
        ("lock_000", {"param_overrides.pb_v2_carry_profit_lock_r": 0.00}),
        ("lock_010", {"param_overrides.pb_v2_carry_profit_lock_r": 0.10}),
        ("lock_015", {"param_overrides.pb_v2_carry_profit_lock_r": 0.15}),
        ("lock_020", {"param_overrides.pb_v2_carry_profit_lock_r": 0.20}),
        ("lock_030", {"param_overrides.pb_v2_carry_profit_lock_r": 0.30}),
        ("lock_040", {"param_overrides.pb_v2_carry_profit_lock_r": 0.40}),
        ("lock_050", {"param_overrides.pb_v2_carry_profit_lock_r": 0.50}),
        ("lock_060", {"param_overrides.pb_v2_carry_profit_lock_r": 0.60}),
        # Profit lock + gap stop ATR combos (0.75/0.50 ATR never tested in V2R2)
        ("lock_020_gap075", {
            "param_overrides.pb_v2_carry_profit_lock_r": 0.20,
            "param_overrides.pb_v2_carry_overnight_stop_atr": 0.75,
        }),
        ("lock_030_gap075", {
            "param_overrides.pb_v2_carry_profit_lock_r": 0.30,
            "param_overrides.pb_v2_carry_overnight_stop_atr": 0.75,
        }),
        ("lock_020_gap050", {
            "param_overrides.pb_v2_carry_profit_lock_r": 0.20,
            "param_overrides.pb_v2_carry_overnight_stop_atr": 0.50,
        }),
    ],
    # -- Phase 2: Intraday Protection + Capacity (10 candidates) --
    2: [
        # Individual MFE stage knobs (V2R2 tested combos, not individuals)
        ("mfe1_035", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.35}),
        ("mfe1_045", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.45}),
        ("mfe2_055", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.55}),
        ("mfe3_090", {"param_overrides.pb_v2_mfe_stage3_trigger": 0.90}),
        ("mfe3_110", {"param_overrides.pb_v2_mfe_stage3_trigger": 1.10}),
        # Partial profit trigger (never tested below 0.75)
        ("partial_050", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.50}),
        ("partial_060", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.60}),
        # Capacity + delayed confirm timing
        ("delayed_bar3", {"param_overrides.pb_delayed_confirm_after_bar": 3}),
        ("delayed_bar5", {"param_overrides.pb_delayed_confirm_after_bar": 5}),
        ("slots_5_bar5", {
            "param_overrides.pb_max_positions": 5,
            "param_overrides.pb_delayed_confirm_after_bar": 5,
        }),
    ],
}


def get_v2r4_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> list[tuple[str, dict[str, Any]]]:
    del profile  # V2R4 uses a single candidate set
    experiments = list(V2R4_PHASE_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v2r4_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v2r4_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
        )
    }


# ==========================================================================
# V3R1 -- Ablation-First Integration (Multiphase + V2R4 + Tier B)
# Baseline: multiphase final + tier B (634 trades, PF 1.42, avg_r +0.082, 51.89R)
# Phase 1 tests REMOVING subsystems; Phases 2-4 add V2R4 enhancements
# ==========================================================================
V3R1_BASE_MUTATIONS: dict[str, Any] = {
    # Multiphase r4_hybrid cumulative mutations (64 params)
    "param_overrides.max_positions_per_sector": 2,
    "param_overrides.pb_atr_stop_mult": 1.0,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_carry_enabled": True,
    "param_overrides.pb_carry_score_threshold": 50.0,
    "param_overrides.pb_cdd_max": 5,
    "param_overrides.pb_cdd_min": 3,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_delayed_confirm_after_bar": 3,
    "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.65,
    "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.25,
    "param_overrides.pb_delayed_confirm_carry_min_daily_signal_score": 58.0,
    "param_overrides.pb_delayed_confirm_carry_min_r": 0.1,
    "param_overrides.pb_delayed_confirm_carry_score_fallback_enabled": True,
    "param_overrides.pb_delayed_confirm_carry_score_threshold": 58.0,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_delayed_confirm_max_hold_days": 3,
    "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.15,
    "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.25,
    "param_overrides.pb_delayed_confirm_min_daily_signal_score": 35.0,
    "param_overrides.pb_delayed_confirm_quick_exit_loss_r": 0.0,
    "param_overrides.pb_delayed_confirm_score_min": 46.0,
    "param_overrides.pb_delayed_confirm_stale_exit_bars": 0,
    "param_overrides.pb_delayed_confirm_trail_activate_r": 0.9,
    "param_overrides.pb_delayed_confirm_vwap_fail_cpr_max": -1.0,
    "param_overrides.pb_entry_rank_max": 999,
    "param_overrides.pb_entry_rank_min": 1,
    "param_overrides.pb_entry_rank_pct_max": 50.0,
    "param_overrides.pb_entry_rank_pct_min": 20.0,
    "param_overrides.pb_entry_score_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_entry_score_min": 50.0,
    "param_overrides.pb_execution_mode": "intraday_hybrid",
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_flush_cpr_max": 0.35,
    "param_overrides.pb_flush_window_bars": 3,
    "param_overrides.pb_friday_mult": 1.0,
    "param_overrides.pb_intraday_priority_reserve_slots": 1,
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_min_candidates_day_hard_gate": False,
    "param_overrides.pb_open_scored_enabled": True,
    "param_overrides.pb_open_scored_max_share": 0.2,
    "param_overrides.pb_open_scored_min_score": 65.0,
    "param_overrides.pb_open_scored_rank_pct_max": 20.0,
    "param_overrides.pb_opening_reclaim_carry_close_pct_min": 0.7,
    "param_overrides.pb_opening_reclaim_carry_mfe_gate_r": 0.2,
    "param_overrides.pb_opening_reclaim_carry_min_daily_signal_score": 62.0,
    "param_overrides.pb_opening_reclaim_carry_min_r": 0.15,
    "param_overrides.pb_opening_reclaim_carry_score_fallback_enabled": True,
    "param_overrides.pb_opening_reclaim_carry_score_threshold": 66.0,
    "param_overrides.pb_opening_reclaim_enabled": True,
    "param_overrides.pb_opening_reclaim_max_hold_days": 2,
    "param_overrides.pb_opening_reclaim_min_daily_signal_score": 58.0,
    "param_overrides.pb_ready_acceptance_bars": 1,
    "param_overrides.pb_ready_min_cpr": 0.45,
    "param_overrides.pb_ready_min_volume_ratio": 0.6,
    "param_overrides.pb_reclaim_offset_atr": 0.2,
    "param_overrides.pb_rescue_flow_enabled": False,
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",
    "param_overrides.pb_sma_dist_max_pct": 10.0,
    "param_overrides.pb_thursday_mult": 0.5,
    "param_overrides.pb_wednesday_mult": 1.0,
    # Tier B sweep winner
    "param_overrides.t2_regime_b_sizing_mult": 0.7,
}


V3R1_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Ablation -- Structural Pruning", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    2: ("V2R4 Exit Mechanics", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    3: ("Carry + Signal Refinement", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    4: ("Tier B + Capacity", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
}


V3R1_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: Ablation -- Structural Pruning (13 candidates) --
    # Test if removing multiphase subsystems IMPROVES the combined score.
    # Each candidate reverts a param group to engine defaults.
    1: [
        # Route ablation
        ("remove_dc_route", {"param_overrides.pb_delayed_confirm_enabled": False}),
        ("remove_or_route", {"param_overrides.pb_opening_reclaim_enabled": False}),
        ("remove_os_route", {"param_overrides.pb_v2_open_scored_enabled": False}),
        ("remove_carry", {"param_overrides.pb_carry_enabled": False}),
        # Gate/filter ablation
        ("remove_thursday", {"param_overrides.pb_thursday_mult": 1.0}),
        ("remove_rescue_penalty", {"param_overrides.pb_rescue_size_mult": 1.0}),
        ("remove_cdd_upper", {"param_overrides.pb_cdd_max": 999}),
        ("remove_cdd_lower", {"param_overrides.pb_cdd_min": 0}),
        ("loosen_entry_score", {"param_overrides.pb_entry_score_min": 40.0}),
        ("loosen_rank_pct", {"param_overrides.pb_entry_rank_pct_max": 80.0}),
        ("remove_sma_gate", {"param_overrides.pb_sma_dist_max_pct": 99.0}),
        # Structural
        ("widen_sector", {"param_overrides.max_positions_per_sector": 5}),
        ("remove_tier_b", {"param_overrides.t2_regime_b_sizing_mult": 1.0}),
    ],
    # -- Phase 2: V2R4 Exit Mechanics (14 candidates) --
    2: [
        # MFE staged protection (V2R4 core)
        ("mfe1_trigger_050", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.5}),
        ("mfe1_stop_neg010", {"param_overrides.pb_v2_mfe_stage1_stop_r": -0.1}),
        ("mfe2_trigger_060", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.6}),
        ("mfe3_trigger_125", {"param_overrides.pb_v2_mfe_stage3_trigger": 1.25}),
        ("mfe3_trail_075", {"param_overrides.pb_v2_mfe_stage3_trail_atr": 0.75}),
        # Stale detection
        ("stale_bars_6", {"param_overrides.pb_v2_stale_bars": 6}),
        ("stale_mfe_005", {"param_overrides.pb_v2_stale_mfe_thresh": 0.05}),
        # RSI exits
        ("rsi_exit_os_60", {"param_overrides.pb_v2_rsi_exit_open_scored": 60.0}),
        # Flatten loss
        ("flatten_loss_neg050", {"param_overrides.pb_v2_flatten_loss_r": -0.5}),
        # EMA reversion
        ("ema_rev_min_003", {"param_overrides.pb_v2_ema_reversion_min_r": 0.03}),
        # Profit lock
        ("partial_050", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.5}),
        # Flow grace
        ("flow_grace_2", {"param_overrides.pb_v2_flow_grace_days": 2}),
        # Secular sizing
        ("secular_065", {"param_overrides.pb_v2_secular_sizing_mult": 0.65}),
        ("secular_080", {"param_overrides.pb_v2_secular_sizing_mult": 0.80}),
    ],
    # -- Phase 3: Carry + Signal Refinement (12 candidates) --
    3: [
        ("atr_stop_150", {"param_overrides.pb_atr_stop_mult": 1.5}),
        ("atr_stop_200", {"param_overrides.pb_atr_stop_mult": 2.0}),
        ("signal_floor_72", {"param_overrides.pb_v2_signal_floor": 72.0}),
        ("signal_floor_78", {"param_overrides.pb_v2_signal_floor": 78.0}),
        ("profit_lock_010", {"param_overrides.pb_v2_carry_profit_lock_r": 0.10}),
        ("profit_lock_020", {"param_overrides.pb_v2_carry_profit_lock_r": 0.20}),
        ("profit_lock_030", {"param_overrides.pb_v2_carry_profit_lock_r": 0.30}),
        ("carry_close_060", {"param_overrides.pb_carry_close_pct_min": 0.60}),
        ("carry_mfe_010", {"param_overrides.pb_carry_mfe_gate_r": 0.10}),
        ("overnight_stop_075", {"param_overrides.pb_v2_carry_overnight_stop_atr": 0.75}),
        ("overnight_stop_125", {"param_overrides.pb_v2_carry_overnight_stop_atr": 1.25}),
        ("hold_days_3", {"param_overrides.pb_max_hold_days": 3}),
    ],
    # -- Phase 4: Tier B + Capacity (14 candidates) --
    4: [
        # Tier B sizing (0.7 already in base; test alternatives)
        ("sizing_b_050", {"param_overrides.t2_regime_b_sizing_mult": 0.5}),
        ("sizing_b_060", {"param_overrides.t2_regime_b_sizing_mult": 0.6}),
        ("sizing_b_080", {"param_overrides.t2_regime_b_sizing_mult": 0.8}),
        # Tier B signal floor
        ("floor_b_78", {"param_overrides.pb_v2_signal_floor_tier_b": 78.0}),
        # Regime B carry
        ("carry_b_030", {"param_overrides.regime_b_carry_mult": 0.3}),
        ("carry_b_000", {"param_overrides.regime_b_carry_mult": 0.0}),
        # Tier B position limits
        ("max_pos_b_4", {"param_overrides.max_positions_tier_b": 4}),
        ("max_pos_b_7", {"param_overrides.max_positions_tier_b": 7}),
        # Capacity
        ("max_pos_8", {"param_overrides.pb_max_positions": 8}),
        ("max_pos_12", {"param_overrides.pb_max_positions": 12}),
        ("min_cand_6", {"param_overrides.pb_min_candidates_day": 6}),
        ("min_cand_10", {"param_overrides.pb_min_candidates_day": 10}),
        # Rescue / day-of-week
        ("rescue_050", {"param_overrides.pb_rescue_size_mult": 0.50}),
        ("friday_080", {"param_overrides.pb_friday_mult": 0.8}),
    ],
}


def get_v3r1_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> list[tuple[str, dict[str, Any]]]:
    del profile  # V3R1 uses a single candidate set
    experiments = list(V3R1_PHASE_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v3r1_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v3r1_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
        )
    }


# ==========================================================================
# V4R1 -- Comprehensive Auto-Optimization (V2R4 base + all lineages)
# Baseline: V2R4 cumulative (46 keys) -- 634 trades, PF 1.42, avg_r +0.082, 51.89R
# 5 phases: ablation, tier B grid, multiphase adoption, exit tuning, signal sweep
# ==========================================================================
V4R1_BASE_MUTATIONS: dict[str, Any] = {
    # V2R4 cumulative mutations (46 keys) -- verbatim from output_v2r4/phase_state.json
    "param_overrides.pb_v2_enabled": True,
    "param_overrides.pb_execution_mode": "intraday_hybrid",
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",
    "param_overrides.pb_v2_signal_floor": 75.0,
    "param_overrides.pb_v2_allow_secular": True,
    "param_overrides.pb_v2_secular_sizing_mult": 0.65,
    "param_overrides.pb_v2_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_max_slots": 4,
    "param_overrides.pb_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_rank_pct_max": 100.0,
    "param_overrides.pb_v2_open_scored_min_score": 45.0,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_v2_vwap_bounce_enabled": True,
    "param_overrides.pb_v2_afternoon_retest_enabled": True,
    "param_overrides.pb_atr_stop_mult": 2.0,
    "param_overrides.pb_v2_partial_profit_trigger_r": 0.5,
    "param_overrides.pb_v2_ema_reversion_min_r": 0.03,
    "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
    "param_overrides.pb_v2_stale_bars": 6,
    "param_overrides.pb_v2_mfe_stage1_stop_r": -0.1,
    "param_overrides.pb_v2_mfe_stage2_trigger": 0.6,
    "param_overrides.pb_opening_reclaim_enabled": True,
    "param_overrides.pb_v2_mfe_stage1_trigger": 0.5,
    "param_overrides.pb_v2_mfe_stage3_trigger": 1.25,
    "param_overrides.pb_v2_mfe_stage3_trail_atr": 0.75,
    "param_overrides.pb_v2_ema_reversion_exit": True,
    "param_overrides.pb_v2_rsi_exit_open_scored": 60.0,
    "param_overrides.pb_v2_carry_overnight_stop_atr": 1.0,
    "param_overrides.pb_v2_flatten_loss_r": -0.5,
    "param_overrides.pb_v2_flow_grace_days": 2,
    "param_overrides.pb_open_scored_carry_close_pct_min": 0.0,
    "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.0,
    "param_overrides.pb_open_scored_flow_reversal_lookback": 2,
    "param_overrides.pb_max_hold_days": 2,
    "param_overrides.pb_max_positions": 10,
    "param_overrides.max_positions_per_sector": 5,
    "param_overrides.pb_open_scored_max_hold_days": 2,
    "param_overrides.pb_carry_close_pct_min": 0.0,
    "param_overrides.pb_carry_mfe_gate_r": 0.0,
}


V4R1_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("V2R4 Structural Ablation", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    2: ("Tier B Full Grid", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    3: ("Multiphase Feature Adoption", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    4: ("Exit Mechanics Tuning", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "inv_dd"]),
    5: ("Signal Threshold Sweep", ["expected_total_r", "avg_r", "total_trades", "profit_factor", "sharpe"]),
}


V4R1_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    # -- Phase 1: V2R4 Structural Ablation (15 candidates) --
    # Each candidate REMOVES a feature. If score improves, the feature was hurting.
    1: [
        ("ablate_delayed_confirm", {"param_overrides.pb_delayed_confirm_enabled": False}),
        ("ablate_opening_reclaim", {"param_overrides.pb_opening_reclaim_enabled": False}),
        ("ablate_open_scored", {"param_overrides.pb_v2_open_scored_enabled": False, "param_overrides.pb_open_scored_enabled": False}),
        ("ablate_afternoon_retest", {"param_overrides.pb_v2_afternoon_retest_enabled": False}),
        ("ablate_vwap_bounce", {"param_overrides.pb_v2_vwap_bounce_enabled": False}),
        ("ablate_carry_overnight", {"param_overrides.pb_carry_close_pct_min": 999.0, "param_overrides.pb_carry_mfe_gate_r": 999.0}),
        ("ablate_secular_tier", {"param_overrides.pb_v2_allow_secular": False}),
        ("ablate_ema_reversion_exit", {"param_overrides.pb_v2_ema_reversion_exit": False}),
        ("ablate_flow_policy", {"param_overrides.pb_flow_policy": "strict"}),
        ("ablate_rescue_sizing", {"param_overrides.pb_rescue_size_mult": 1.0}),
        ("ablate_hold_limit", {"param_overrides.pb_max_hold_days": 999}),
        ("reduce_max_pos_6", {"param_overrides.pb_max_positions": 6}),
        ("reduce_max_pos_8", {"param_overrides.pb_max_positions": 8}),
        ("reduce_sector_3", {"param_overrides.max_positions_per_sector": 3}),
        ("widen_atr_stop_3", {"param_overrides.pb_atr_stop_mult": 3.0}),
    ],
    # -- Phase 2: Tier B Full Grid (11 candidates) --
    # Re-test ALL tier B options on the (potentially pruned) V2R4 base.
    2: [
        ("tier_b_max_pos_3", {"param_overrides.max_positions_tier_b": 3}),
        ("tier_b_max_pos_4", {"param_overrides.max_positions_tier_b": 4}),
        ("tier_b_max_pos_6", {"param_overrides.max_positions_tier_b": 6}),
        ("tier_b_floor_78", {"param_overrides.pb_v2_signal_floor_tier_b": 78.0}),
        ("tier_b_floor_80", {"param_overrides.pb_v2_signal_floor_tier_b": 80.0}),
        ("tier_b_carry_0", {"param_overrides.regime_b_carry_mult": 0.0}),
        ("tier_b_carry_03", {"param_overrides.regime_b_carry_mult": 0.3}),
        ("tier_b_sizing_05", {"param_overrides.t2_regime_b_sizing_mult": 0.5}),
        ("tier_b_sizing_07", {"param_overrides.t2_regime_b_sizing_mult": 0.7}),
        ("tier_b_no_delayed", {"param_overrides.pb_delayed_confirm_enabled": False}),
        ("tier_b_best_combo", {"param_overrides.max_positions_tier_b": 6, "param_overrides.t2_regime_b_sizing_mult": 0.7}),
    ],
    # -- Phase 3: Multiphase Feature Adoption (19 candidates) --
    # Test multiphase-unique features on the pruned V2R4 + tier B base.
    3: [
        ("mp_entry_rank_gating", {
            "param_overrides.pb_entry_rank_min": 1,
            "param_overrides.pb_entry_rank_max": 999,
            "param_overrides.pb_entry_rank_pct_min": 20.0,
            "param_overrides.pb_entry_rank_pct_max": 50.0,
        }),
        ("mp_cdd_bounds", {
            "param_overrides.pb_cdd_min": 3,
            "param_overrides.pb_cdd_max": 5,
            "param_overrides.pb_min_candidates_day_hard_gate": False,
        }),
        ("mp_sma_dist_cap", {"param_overrides.pb_sma_dist_max_pct": 10.0}),
        ("mp_sector_tight", {"param_overrides.max_positions_per_sector": 2}),
        ("mp_thursday_discount", {
            "param_overrides.pb_wednesday_mult": 1.0,
            "param_overrides.pb_thursday_mult": 0.5,
            "param_overrides.pb_friday_mult": 1.0,
        }),
        ("mp_atr_stop_tight", {"param_overrides.pb_atr_stop_mult": 1.0}),
        ("mp_carry_enable", {
            "param_overrides.pb_carry_enabled": True,
            "param_overrides.pb_carry_score_threshold": 50.0,
        }),
        ("mp_entry_score_gate", {
            "param_overrides.pb_entry_score_min": 50.0,
            "param_overrides.pb_entry_score_family": "meanrev_sweetspot_v1",
        }),
        ("mp_ready_state", {
            "param_overrides.pb_ready_min_cpr": 0.45,
            "param_overrides.pb_ready_min_volume_ratio": 0.6,
            "param_overrides.pb_ready_acceptance_bars": 1,
        }),
        ("mp_dc_timing_exit", {
            "param_overrides.pb_delayed_confirm_after_bar": 3,
            "param_overrides.pb_delayed_confirm_score_min": 46.0,
            "param_overrides.pb_delayed_confirm_min_daily_signal_score": 35.0,
            "param_overrides.pb_delayed_confirm_quick_exit_loss_r": 0.0,
            "param_overrides.pb_delayed_confirm_stale_exit_bars": 0,
            "param_overrides.pb_delayed_confirm_vwap_fail_cpr_max": -1.0,
        }),
        ("mp_dc_mfe_protect", {
            "param_overrides.pb_delayed_confirm_mfe_protect_trigger_r": 0.25,
            "param_overrides.pb_delayed_confirm_mfe_protect_stop_r": 0.15,
            "param_overrides.pb_delayed_confirm_trail_activate_r": 0.9,
        }),
        ("mp_dc_carry", {
            "param_overrides.pb_delayed_confirm_carry_min_r": 0.1,
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.65,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.25,
            "param_overrides.pb_delayed_confirm_carry_min_daily_signal_score": 58.0,
            "param_overrides.pb_delayed_confirm_carry_score_threshold": 58.0,
            "param_overrides.pb_delayed_confirm_carry_score_fallback_enabled": True,
        }),
        ("mp_dc_hold_3d", {"param_overrides.pb_delayed_confirm_max_hold_days": 3}),
        ("mp_os_tight_gate", {
            "param_overrides.pb_open_scored_min_score": 65.0,
            "param_overrides.pb_open_scored_rank_pct_max": 20.0,
            "param_overrides.pb_open_scored_max_share": 0.2,
        }),
        ("mp_or_carry_bundle", {
            "param_overrides.pb_opening_reclaim_min_daily_signal_score": 58.0,
            "param_overrides.pb_opening_reclaim_carry_min_r": 0.15,
            "param_overrides.pb_opening_reclaim_carry_close_pct_min": 0.7,
            "param_overrides.pb_opening_reclaim_carry_mfe_gate_r": 0.2,
            "param_overrides.pb_opening_reclaim_carry_min_daily_signal_score": 62.0,
            "param_overrides.pb_opening_reclaim_carry_score_threshold": 66.0,
            "param_overrides.pb_opening_reclaim_carry_score_fallback_enabled": True,
            "param_overrides.pb_opening_reclaim_max_hold_days": 2,
        }),
        ("mp_flush_mechanics", {
            "param_overrides.pb_flush_window_bars": 3,
            "param_overrides.pb_flush_cpr_max": 0.35,
        }),
        ("mp_reclaim_offset", {"param_overrides.pb_reclaim_offset_atr": 0.2}),
        ("mp_priority_reserve", {"param_overrides.pb_intraday_priority_reserve_slots": 1}),
        ("mp_no_rescue_flow", {"param_overrides.pb_rescue_flow_enabled": False}),
    ],
    # -- Phase 4: Exit Mechanics Tuning (20 candidates) --
    4: [
        # MFE staged protection
        ("mfe_s1_trigger_03", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.3}),
        ("mfe_s1_trigger_07", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.7}),
        ("mfe_s1_stop_0", {"param_overrides.pb_v2_mfe_stage1_stop_r": 0.0}),
        ("mfe_s2_trigger_05", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.5}),
        ("mfe_s2_trigger_08", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.8}),
        ("mfe_s3_trigger_10", {"param_overrides.pb_v2_mfe_stage3_trigger": 1.0}),
        ("mfe_s3_trigger_15", {"param_overrides.pb_v2_mfe_stage3_trigger": 1.5}),
        ("mfe_s3_trail_05", {"param_overrides.pb_v2_mfe_stage3_trail_atr": 0.5}),
        ("mfe_s3_trail_10", {"param_overrides.pb_v2_mfe_stage3_trail_atr": 1.0}),
        # Flatten and stale
        ("flatten_loss_03", {"param_overrides.pb_v2_flatten_loss_r": -0.3}),
        ("stale_bars_4", {"param_overrides.pb_v2_stale_bars": 4}),
        ("stale_bars_8", {"param_overrides.pb_v2_stale_bars": 8}),
        # Partial profit trigger (base 0.5)
        ("partial_profit_03", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.3}),
        ("partial_profit_07", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.7}),
        # RSI exit for open scored (base 60.0)
        ("rsi_exit_os_50", {"param_overrides.pb_v2_rsi_exit_open_scored": 50.0}),
        ("rsi_exit_os_70", {"param_overrides.pb_v2_rsi_exit_open_scored": 70.0}),
        # Overnight carry stop ATR (base 1.0)
        ("overnight_stop_075", {"param_overrides.pb_v2_carry_overnight_stop_atr": 0.75}),
        ("overnight_stop_125", {"param_overrides.pb_v2_carry_overnight_stop_atr": 1.25}),
        # Stale MFE threshold (base 0.05)
        ("stale_mfe_003", {"param_overrides.pb_v2_stale_mfe_thresh": 0.03}),
        ("stale_mfe_008", {"param_overrides.pb_v2_stale_mfe_thresh": 0.08}),
    ],
    # -- Phase 5: Signal Threshold Sweep (10 candidates) --
    5: [
        ("signal_floor_72", {"param_overrides.pb_v2_signal_floor": 72.0}),
        ("signal_floor_78", {"param_overrides.pb_v2_signal_floor": 78.0}),
        ("daily_signal_50", {"param_overrides.pb_daily_signal_min_score": 50.0}),
        ("daily_signal_58", {"param_overrides.pb_daily_signal_min_score": 58.0}),
        ("rescue_score_48", {"param_overrides.pb_daily_rescue_min_score": 48.0}),
        ("rescue_score_56", {"param_overrides.pb_daily_rescue_min_score": 56.0}),
        ("flow_grace_1", {"param_overrides.pb_v2_flow_grace_days": 1}),
        ("flow_grace_3", {"param_overrides.pb_v2_flow_grace_days": 3}),
        ("ema_rev_min_0", {"param_overrides.pb_v2_ema_reversion_min_r": 0.0}),
        ("ema_rev_min_01", {"param_overrides.pb_v2_ema_reversion_min_r": 0.10}),
    ],
}

# Names of Phase 3 candidates that depend on delayed_confirm being enabled
_V4R1_DC_DEPENDENT_CANDIDATES = {"mp_dc_timing_exit", "mp_dc_mfe_protect", "mp_dc_carry", "mp_dc_hold_3d"}
# Names of Phase 3 candidates that depend on opening_reclaim being enabled
_V4R1_OR_DEPENDENT_CANDIDATES = {"mp_or_carry_bundle"}
# Names of Phase 3 candidates that depend on open_scored being enabled
_V4R1_OS_DEPENDENT_CANDIDATES = {"mp_os_tight_gate"}


def get_v4r1_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
    accepted_mutations: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    del profile  # V4R1 uses a single candidate set
    experiments = list(V4R1_PHASE_CANDIDATES.get(phase, []))

    # Phase 3: skip DC/OR/OS candidates if Phase 1 ablated those routes
    if phase == 3 and accepted_mutations:
        dc_ablated = accepted_mutations.get("param_overrides.pb_delayed_confirm_enabled") is False
        or_ablated = accepted_mutations.get("param_overrides.pb_opening_reclaim_enabled") is False
        os_ablated = accepted_mutations.get("param_overrides.pb_v2_open_scored_enabled") is False
        skip_names: set[str] = set()
        if dc_ablated:
            skip_names |= _V4R1_DC_DEPENDENT_CANDIDATES
        if or_ablated:
            skip_names |= _V4R1_OR_DEPENDENT_CANDIDATES
        if os_ablated:
            skip_names |= _V4R1_OS_DEPENDENT_CANDIDATES
        if skip_names:
            experiments = [(name, muts) for name, muts in experiments if name not in skip_names]

    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v4r1_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
    accepted_mutations: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v4r1_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
            accepted_mutations=accepted_mutations,
        )
    }


# ---------------------------------------------------------------------------
# V5R1 -- Alpha Extraction and Capacity Expansion from Round-1 optimum
#
# Baseline: backtests/output/stock/iaric/round_1/optimized_config.json
# The candidate set deliberately stays in the existing config surface so this
# round optimizes strategy behavior without changing execution or timing logic.
# ---------------------------------------------------------------------------
V5R1_BASE_MUTATIONS: dict[str, Any] = {
    "param_overrides.pb_v2_enabled": True,
    "param_overrides.pb_execution_mode": "intraday_hybrid",
    "param_overrides.pb_daily_signal_family": "meanrev_sweetspot_v1",
    "param_overrides.pb_daily_signal_min_score": 54.0,
    "param_overrides.pb_daily_rescue_min_score": 52.0,
    "param_overrides.pb_flow_policy": "soft_penalty_rescue",
    "param_overrides.pb_rescue_size_mult": 0.65,
    "param_overrides.pb_backtest_intraday_universe_only": True,
    "param_overrides.pb_min_candidates_day": 8,
    "param_overrides.pb_signal_rank_gate_mode": "score_rank",
    "param_overrides.pb_v2_signal_floor": 75.0,
    "param_overrides.pb_v2_allow_secular": True,
    "param_overrides.pb_v2_secular_sizing_mult": 0.65,
    "param_overrides.pb_v2_open_scored_enabled": True,
    "param_overrides.pb_v2_open_scored_max_slots": 4,
    "param_overrides.pb_open_scored_enabled": True,
    "param_overrides.pb_open_scored_fill_timing": "next_5m_open",
    "param_overrides.pb_v2_open_scored_rank_pct_max": 100.0,
    "param_overrides.pb_v2_open_scored_min_score": 45.0,
    "param_overrides.pb_delayed_confirm_enabled": True,
    "param_overrides.pb_v2_vwap_bounce_enabled": True,
    "param_overrides.pb_v2_afternoon_retest_enabled": True,
    "param_overrides.pb_atr_stop_mult": 1.0,
    "param_overrides.pb_v2_partial_profit_trigger_r": 0.3,
    "param_overrides.pb_v2_ema_reversion_min_r": 0.03,
    "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
    "param_overrides.pb_v2_stale_bars": 6,
    "param_overrides.pb_v2_mfe_stage1_stop_r": -0.1,
    "param_overrides.pb_v2_mfe_stage2_trigger": 0.6,
    "param_overrides.pb_opening_reclaim_enabled": False,
    "param_overrides.pb_v2_mfe_stage1_trigger": 0.5,
    "param_overrides.pb_v2_mfe_stage3_trigger": 1.25,
    "param_overrides.pb_v2_mfe_stage3_trail_atr": 0.75,
    "param_overrides.pb_v2_ema_reversion_exit": True,
    "param_overrides.pb_v2_rsi_exit_open_scored": 60.0,
    "param_overrides.pb_v2_carry_overnight_stop_atr": 1.0,
    "param_overrides.pb_v2_flatten_loss_r": -0.5,
    "param_overrides.pb_v2_flow_grace_days": 2,
    "param_overrides.pb_open_scored_carry_close_pct_min": 0.0,
    "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.0,
    "param_overrides.pb_open_scored_flow_reversal_lookback": 2,
    "param_overrides.pb_max_hold_days": 2,
    "param_overrides.pb_max_positions": 10,
    "param_overrides.max_positions_per_sector": 2,
    "param_overrides.pb_open_scored_max_hold_days": 2,
    "param_overrides.pb_carry_close_pct_min": 0.0,
    "param_overrides.pb_carry_mfe_gate_r": 0.0,
}


V5R1_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Signal Alpha Recovery and Rejection",
        ["expected_total_r", "avg_r", "total_trades", "profit_factor", "alpha_discrimination"],
    ),
    2: (
        "Entry Route Quality and Timing",
        ["expected_total_r", "avg_r", "total_trades", "profit_factor", "alpha_discrimination"],
    ),
    3: (
        "Capacity and Frequency Expansion",
        ["expected_total_r", "total_trades", "avg_r", "profit_factor", "inv_dd"],
    ),
    4: (
        "Carry, Management, and Exit Capture",
        ["expected_total_r", "avg_r", "profit_factor", "sharpe", "inv_dd"],
    ),
    5: (
        "Robust Interaction Bundles",
        ["expected_total_r", "total_trades", "avg_r", "profit_factor", "alpha_discrimination"],
    ),
}


V5R1_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        ("signal_floor_72", {"param_overrides.pb_v2_signal_floor": 72.0}),
        ("signal_floor_78", {"param_overrides.pb_v2_signal_floor": 78.0}),
        ("signal_floor_80", {"param_overrides.pb_v2_signal_floor": 80.0}),
        ("signal_floor_72_gap_max_1", {
            "param_overrides.pb_v2_signal_floor": 72.0,
            "param_overrides.pb_v2_gap_max_pct": 1.0,
        }),
        ("gap_max_0", {"param_overrides.pb_v2_gap_max_pct": 0.0}),
        ("gap_max_1", {"param_overrides.pb_v2_gap_max_pct": 1.0}),
        ("gap_max_2", {"param_overrides.pb_v2_gap_max_pct": 2.0}),
        ("gap_min_neg8", {"param_overrides.pb_v2_gap_min_pct": -8.0}),
        ("gap_down_focus", {
            "param_overrides.pb_v2_gap_min_pct": -8.0,
            "param_overrides.pb_v2_gap_max_pct": 0.5,
        }),
        ("sma_dist_max_12", {"param_overrides.pb_v2_sma_dist_max_pct": 12.0}),
        ("sma_dist_max_15", {"param_overrides.pb_v2_sma_dist_max_pct": 15.0}),
        ("cdd_max_4", {"param_overrides.pb_cdd_max": 4}),
        ("cdd_max_6", {"param_overrides.pb_cdd_max": 6}),
        ("rsi2_strict_12", {"param_overrides.pb_v2_rsi2_thresh": 12.0}),
        ("rsi2_broad_20_rsi5_35", {
            "param_overrides.pb_v2_rsi2_thresh": 20.0,
            "param_overrides.pb_v2_rsi5_thresh": 35.0,
        }),
        ("rs_ratio_relax_100", {"param_overrides.pb_v2_rs_ratio_thresh": 1.00}),
        ("secular_size_050", {"param_overrides.pb_v2_secular_sizing_mult": 0.50}),
    ],
    2: [
        ("vwap_bounce_off", {"param_overrides.pb_v2_vwap_bounce_enabled": False}),
        ("afternoon_retest_off", {"param_overrides.pb_v2_afternoon_retest_enabled": False}),
        ("tail_routes_off", {
            "param_overrides.pb_v2_vwap_bounce_enabled": False,
            "param_overrides.pb_v2_afternoon_retest_enabled": False,
        }),
        ("delayed_confirm_off", {"param_overrides.pb_delayed_confirm_enabled": False}),
        ("delayed_bar_5", {"param_overrides.pb_delayed_confirm_after_bar": 5}),
        ("delayed_bar_7", {"param_overrides.pb_delayed_confirm_after_bar": 7}),
        ("delayed_tighter_quality", {
            "param_overrides.pb_delayed_confirm_after_bar": 5,
            "param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.55,
            "param_overrides.pb_v2_delayed_confirm_vol_ratio": 0.70,
            "param_overrides.pb_delayed_confirm_score_min": 52.0,
        }),
        ("delayed_later_quality", {
            "param_overrides.pb_delayed_confirm_after_bar": 7,
            "param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.55,
            "param_overrides.pb_delayed_confirm_score_min": 52.0,
        }),
        ("delayed_rescue_allowed", {"param_overrides.pb_v2_delayed_confirm_allow_rescue": True}),
        ("open_scored_min_50", {"param_overrides.pb_v2_open_scored_min_score": 50.0}),
        ("open_scored_min_55", {"param_overrides.pb_v2_open_scored_min_score": 55.0}),
        ("open_scored_rank_90", {"param_overrides.pb_v2_open_scored_rank_pct_max": 90.0}),
        ("open_scored_rank_75", {"param_overrides.pb_v2_open_scored_rank_pct_max": 75.0}),
        ("missing_5m_disallow", {"param_overrides.pb_open_scored_missing_5m_allow": False}),
        ("opening_reclaim_high_quality", {
            "param_overrides.pb_opening_reclaim_enabled": True,
            "param_overrides.pb_opening_reclaim_min_daily_signal_score": 62.0,
            "param_overrides.pb_flush_window_bars": 6,
            "param_overrides.pb_flush_cpr_max": 0.35,
            "param_overrides.pb_ready_min_cpr": 0.55,
            "param_overrides.pb_ready_min_volume_ratio": 0.80,
        }),
    ],
    3: [
        ("sector_cap_3", {"param_overrides.max_positions_per_sector": 3}),
        ("sector_cap_4", {"param_overrides.max_positions_per_sector": 4}),
        ("max_pos_11", {"param_overrides.pb_max_positions": 11}),
        ("max_pos_12", {"param_overrides.pb_max_positions": 12}),
        ("sector3_pos12", {
            "param_overrides.max_positions_per_sector": 3,
            "param_overrides.pb_max_positions": 12,
        }),
        ("tier_b_cap_6", {"param_overrides.max_positions_tier_b": 6}),
        ("tier_b_cap_7", {"param_overrides.max_positions_tier_b": 7}),
        ("tier_b_floor_72", {"param_overrides.pb_v2_signal_floor_tier_b": 72.0}),
        ("tier_b_floor_78", {"param_overrides.pb_v2_signal_floor_tier_b": 78.0}),
        ("tier_b_size_060", {"param_overrides.t2_regime_b_sizing_mult": 0.60}),
        ("tier_b_size_080", {"param_overrides.t2_regime_b_sizing_mult": 0.80}),
        ("leverage_250", {"param_overrides.intraday_leverage": 2.5}),
        ("leverage_300", {"param_overrides.intraday_leverage": 3.0}),
        ("capacity_quality_bundle", {
            "param_overrides.max_positions_per_sector": 3,
            "param_overrides.pb_max_positions": 12,
            "param_overrides.pb_v2_open_scored_min_score": 50.0,
        }),
    ],
    4: [
        ("v2_carry_off_gates", {
            "param_overrides.pb_carry_close_pct_min": 999.0,
            "param_overrides.pb_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_open_scored_carry_close_pct_min": 999.0,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 999.0,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_opening_reclaim_carry_close_pct_min": 999.0,
            "param_overrides.pb_opening_reclaim_carry_mfe_gate_r": 999.0,
        }),
        ("carry_quality_055_020", {
            "param_overrides.pb_carry_close_pct_min": 0.55,
            "param_overrides.pb_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.55,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.20,
        }),
        ("carry_quality_070_030", {
            "param_overrides.pb_carry_close_pct_min": 0.70,
            "param_overrides.pb_carry_mfe_gate_r": 0.30,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.70,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.30,
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.70,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.30,
        }),
        ("profit_lock_050", {"param_overrides.pb_v2_carry_profit_lock_r": 0.50}),
        ("profit_lock_100", {"param_overrides.pb_v2_carry_profit_lock_r": 1.00}),
        ("partial_profit_020", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.20}),
        ("partial_profit_040", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.40}),
        ("partial_profit_050", {"param_overrides.pb_v2_partial_profit_trigger_r": 0.50}),
        ("partial_remainder_030", {"param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.30}),
        ("partial_remainder_070", {"param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.70}),
        ("mfe_s1_trigger_040", {"param_overrides.pb_v2_mfe_stage1_trigger": 0.40}),
        ("mfe_s1_stop_000", {"param_overrides.pb_v2_mfe_stage1_stop_r": 0.0}),
        ("mfe_s2_trigger_050", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.50}),
        ("mfe_s2_trigger_080", {"param_overrides.pb_v2_mfe_stage2_trigger": 0.80}),
        ("mfe_s3_trigger_100", {"param_overrides.pb_v2_mfe_stage3_trigger": 1.00}),
        ("mfe_s3_trail_050", {"param_overrides.pb_v2_mfe_stage3_trail_atr": 0.50}),
        ("stale_bars_4_mfe_008", {
            "param_overrides.pb_v2_stale_bars": 4,
            "param_overrides.pb_v2_stale_mfe_thresh": 0.08,
        }),
        ("ema_min_000", {"param_overrides.pb_v2_ema_reversion_min_r": 0.0}),
        ("ema_min_008", {"param_overrides.pb_v2_ema_reversion_min_r": 0.08}),
        ("rsi_exit_os_55", {"param_overrides.pb_v2_rsi_exit_open_scored": 55.0}),
        ("rsi_exit_os_65", {"param_overrides.pb_v2_rsi_exit_open_scored": 65.0}),
    ],
    5: [
        ("combo_gap_capacity", {
            "param_overrides.pb_v2_gap_max_pct": 1.0,
            "param_overrides.max_positions_per_sector": 3,
            "param_overrides.pb_max_positions": 12,
        }),
        ("combo_floor72_gap_capacity", {
            "param_overrides.pb_v2_signal_floor": 72.0,
            "param_overrides.pb_v2_gap_max_pct": 1.0,
            "param_overrides.max_positions_per_sector": 3,
            "param_overrides.pb_max_positions": 12,
        }),
        ("combo_route_quarantine", {
            "param_overrides.pb_v2_vwap_bounce_enabled": False,
            "param_overrides.pb_v2_afternoon_retest_enabled": False,
            "param_overrides.pb_delayed_confirm_after_bar": 5,
            "param_overrides.pb_delayed_confirm_score_min": 52.0,
        }),
        ("combo_carry_off_exit_capture", {
            "param_overrides.pb_carry_close_pct_min": 999.0,
            "param_overrides.pb_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_open_scored_carry_close_pct_min": 999.0,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 999.0,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.20,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.50,
        }),
        ("combo_carry_quality_profit_lock", {
            "param_overrides.pb_carry_close_pct_min": 0.55,
            "param_overrides.pb_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_v2_carry_profit_lock_r": 0.50,
        }),
        ("combo_exit_capture_fast", {
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.20,
            "param_overrides.pb_v2_partial_profit_remainder_stop_r": 0.70,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.50,
            "param_overrides.pb_v2_mfe_stage3_trail_atr": 0.50,
        }),
        ("combo_signal_strict_exit_fast", {
            "param_overrides.pb_v2_signal_floor": 78.0,
            "param_overrides.pb_v2_gap_max_pct": 1.0,
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.20,
            "param_overrides.pb_v2_mfe_stage2_trigger": 0.50,
        }),
        ("combo_no_secular_capacity", {
            "param_overrides.pb_v2_allow_secular": False,
            "param_overrides.max_positions_per_sector": 3,
            "param_overrides.pb_max_positions": 12,
        }),
        ("combo_missing5m_strict_open", {
            "param_overrides.pb_open_scored_missing_5m_allow": False,
            "param_overrides.pb_v2_open_scored_min_score": 50.0,
            "param_overrides.pb_v2_open_scored_rank_pct_max": 90.0,
        }),
        ("combo_quality_final_defensive", {
            "param_overrides.pb_v2_signal_floor": 78.0,
            "param_overrides.pb_v2_vwap_bounce_enabled": False,
            "param_overrides.pb_v2_afternoon_retest_enabled": False,
            "param_overrides.pb_v2_stale_bars": 4,
            "param_overrides.pb_v2_stale_mfe_thresh": 0.08,
        }),
    ],
}


_V5R1_DC_DEPENDENT_CANDIDATES = {
    "delayed_bar_5",
    "delayed_bar_7",
    "delayed_tighter_quality",
    "delayed_later_quality",
    "delayed_rescue_allowed",
}


def get_v5r1_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
    accepted_mutations: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    del profile
    experiments = list(V5R1_PHASE_CANDIDATES.get(phase, []))
    if phase == 2 and accepted_mutations:
        if accepted_mutations.get("param_overrides.pb_delayed_confirm_enabled") is False:
            experiments = [
                (name, mutations)
                for name, mutations in experiments
                if name not in _V5R1_DC_DEPENDENT_CANDIDATES
            ]

    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v5r1_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
    accepted_mutations: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v5r1_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
            accepted_mutations=accepted_mutations,
        )
    }


V5R2_BASE_MUTATIONS = dict(V5R1_BASE_MUTATIONS)


V5R2_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Carry Drag Suppression",
        ["net_profit", "expected_total_r", "profit_factor", "sharpe", "residual_alpha_quality"],
    ),
    2: (
        "Delayed Confirm Extraction",
        ["net_profit", "expected_total_r", "profit_factor", "total_trades", "residual_alpha_quality"],
    ),
    3: (
        "Selection and Capacity Validation",
        ["net_profit", "expected_total_r", "total_trades", "profit_factor", "inv_dd"],
    ),
    4: (
        "Robust Interaction Bundles",
        ["net_profit", "expected_total_r", "profit_factor", "sharpe", "inv_dd"],
    ),
}


V5R2_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        ("open_carry_quality_055_020", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
        }),
        ("open_carry_quality_070_030", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.70,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.30,
        }),
        ("open_carry_off", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 999.0,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 999.0,
        }),
        ("all_route_carry_quality_055_020", {
            "param_overrides.pb_carry_close_pct_min": 0.55,
            "param_overrides.pb_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.55,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.20,
        }),
        ("all_route_carry_off", {
            "param_overrides.pb_carry_close_pct_min": 999.0,
            "param_overrides.pb_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_open_scored_carry_close_pct_min": 999.0,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 999.0,
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 999.0,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 999.0,
        }),
        ("profit_lock_050", {"param_overrides.pb_v2_carry_profit_lock_r": 0.50}),
        ("flow_grace_1", {"param_overrides.pb_v2_flow_grace_days": 1}),
    ],
    2: [
        ("delayed_score_47", {"param_overrides.pb_delayed_confirm_score_min": 47.0}),
        ("delayed_score_57", {"param_overrides.pb_delayed_confirm_score_min": 57.0}),
        ("delayed_bar_7", {"param_overrides.pb_delayed_confirm_after_bar": 7}),
        ("delayed_bar7_score47", {
            "param_overrides.pb_delayed_confirm_after_bar": 7,
            "param_overrides.pb_delayed_confirm_score_min": 47.0,
        }),
        ("delayed_quality_055_070", {
            "param_overrides.pb_v2_delayed_confirm_min_close_pct": 0.55,
            "param_overrides.pb_v2_delayed_confirm_vol_ratio": 0.70,
        }),
        ("delayed_carry_relaxed_045_010", {
            "param_overrides.pb_delayed_confirm_carry_close_pct_min": 0.45,
            "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 0.10,
        }),
    ],
    3: [
        ("ready_quality_065_120", {
            "param_overrides.pb_ready_min_cpr": 0.65,
            "param_overrides.pb_ready_min_volume_ratio": 1.20,
        }),
        ("ready_cpr_065", {"param_overrides.pb_ready_min_cpr": 0.65}),
        ("gap_max_1", {"param_overrides.pb_v2_gap_max_pct": 1.0}),
        ("gap_down_focus", {
            "param_overrides.pb_v2_gap_min_pct": -8.0,
            "param_overrides.pb_v2_gap_max_pct": 0.5,
        }),
        ("max_pos_11", {"param_overrides.pb_max_positions": 11}),
        ("max_pos_12", {"param_overrides.pb_max_positions": 12}),
    ],
    4: [
        ("carry_quality_delayed47", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_delayed_confirm_score_min": 47.0,
        }),
        ("carry_quality_delayed_bar7", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_delayed_confirm_after_bar": 7,
        }),
        ("carry_quality_capacity11", {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
            "param_overrides.pb_max_positions": 11,
        }),
        ("defensive_gap_carry_quality", {
            "param_overrides.pb_v2_gap_max_pct": 1.0,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.55,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
        }),
    ],
}


def get_v5r2_phase_candidates(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
    accepted_mutations: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    del profile, accepted_mutations
    experiments = list(V5R2_PHASE_CANDIDATES.get(phase, []))
    if suggested_experiments:
        existing = {name for name, _ in experiments}
        experiments.extend(
            (name, mutations)
            for name, mutations in suggested_experiments
            if name not in existing
        )
    return experiments


def get_v5r2_phase_candidate_lookup(
    phase: int,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    profile: str = "mainline",
    accepted_mutations: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        name: dict(mutations)
        for name, mutations in get_v5r2_phase_candidates(
            phase,
            suggested_experiments=suggested_experiments,
            profile=profile,
            accepted_mutations=accepted_mutations,
        )
    }

