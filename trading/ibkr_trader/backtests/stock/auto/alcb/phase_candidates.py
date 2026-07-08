"""ALCB phased candidates for round-3 residual alpha extraction.

Round 3 starts from the round-2 optimized configuration.  The candidates here
target broad completed-bar signal components, causal early-failure protection,
profit retention, and quality-preserved frequency recovery.  Deliberately avoid
sector/day/symbol micro-cohorts: those are too sample-specific for this dataset.
"""
from __future__ import annotations

from typing import Any


BASE_MUTATIONS: dict[str, Any] = {
    "ablation.use_adaptive_trail": True,
    "ablation.use_combined_quality_gate": True,
    "ablation.use_mfe_conviction_exit": True,
    "ablation.use_or_width_min": True,
    "ablation.use_partial_takes": False,
    "param_overrides.adaptive_trail_late_activate_r": 0.25,
    "param_overrides.adaptive_trail_late_distance_r": 0.20,
    "param_overrides.adaptive_trail_start_bars": 25,
    "param_overrides.adaptive_trail_tighten_bars": 25,
    "param_overrides.block_combined_regime_b": True,
    "param_overrides.carry_min_cpr": 0.6,
    "param_overrides.carry_min_r": 0.5,
    "param_overrides.combined_avwap_cap_pct": 0.003,
    "param_overrides.combined_breakout_min_rvol": 2.5,
    "param_overrides.combined_breakout_score_min": 5,
    "param_overrides.entry_window_end": "12:30:00",
    "param_overrides.flow_reversal_min_hold_bars": 12,
    "param_overrides.fr_cpr_threshold": 0.3,
    "param_overrides.fr_mfe_grace_r": 0.20,
    "param_overrides.fr_trailing_activate_r": 0.0,
    "param_overrides.mfe_conviction_check_bars": 16,
    "param_overrides.mfe_conviction_floor_r": -0.15,
    "param_overrides.mfe_conviction_min_r": 0.20,
    "param_overrides.opening_range_bars": 6,
    "param_overrides.or_width_min_pct": 0.0015,
    "param_overrides.pdh_avwap_cap_pct": 0.005,
    "param_overrides.pdh_size_mult": 0.75,
    "param_overrides.regime_mult_b": 0.7,
    "param_overrides.rvol_threshold": 2.0,
}


def sanitize_round2_seed(mutations: dict[str, Any]) -> dict[str, Any]:
    """Preserve the previous optimized config exactly as the round-2 seed."""
    return dict(mutations)


SMALL_SAMPLE_OVERFIT_MUTATION_KEYS: frozenset[str] = frozenset({
    "param_overrides.sector_entry_blocklist",
    "param_overrides.sector_entry_size_mults",
    "param_overrides.monday_sizing_mult",
    "param_overrides.tuesday_sizing_mult",
    "param_overrides.wednesday_sizing_mult",
    "param_overrides.thursday_sizing_mult",
    "param_overrides.friday_sizing_mult",
})


def is_small_sample_overfit_candidate(name: str, mutations: dict[str, Any]) -> bool:
    """Reject candidate families that are too cohort-specific for this sample."""
    del name
    return any(key in SMALL_SAMPLE_OVERFIT_MUTATION_KEYS for key in mutations)


PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Completed-bar score component discrimination",
        [
            "expected_total_r",
            "trades_per_month",
            "expectancy_dollar",
            "profit_factor",
            "signal_quality",
            "score_monotonicity",
            "sizing_alignment",
        ],
    ),
    2: (
        "Early-failure protective stop tightening",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "profit_protection",
            "short_hold_24_drag_inverse",
            "mfe_capture_efficiency",
            "trades_per_month",
        ],
    ),
    3: (
        "Exit management and profit retention",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "profit_protection",
            "flow_mfe_exit_inverse",
            "mfe_capture_efficiency",
            "long_hold_capture",
        ],
    ),
    4: (
        "Quality-preserved frequency recovery",
        [
            "trades_per_month",
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "late_entry_quality",
            "timing_quality",
            "signal_quality",
        ],
    ),
    5: (
        "Entry geometry retest and extension control",
        [
            "expected_total_r",
            "trades_per_month",
            "profit_factor",
            "timing_quality",
            "extended_avwap_inverse",
            "rvol_selectivity",
            "inv_dd",
        ],
    ),
    6: (
        "Round-3 structural synthesis",
        [
            "expected_total_r",
            "trades_per_month",
            "net_profit",
            "profit_factor",
            "expectancy_dollar",
            "signal_quality",
            "timing_quality",
            "sizing_alignment",
            "profit_protection",
        ],
    ),
    7: (
        "Causal early trade-maturation protection",
        [
            "expected_total_r",
            "net_profit",
            "profit_factor",
            "profit_protection",
            "short_hold_24_drag_inverse",
            "mfe_capture_efficiency",
            "trades_per_month",
        ],
    ),
    8: (
        "Delayed entry confirmation and maturation synthesis",
        [
            "expected_total_r",
            "trades_per_month",
            "net_profit",
            "expectancy_dollar",
            "profit_factor",
            "signal_quality",
            "timing_quality",
            "profit_protection",
        ],
    ),
}


PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        (
            "r3_score5_or_no_volume_size55",
            {
                "param_overrides.entry_detail_size_mults": {
                    "OR_BREAKOUT:5:!bar_vol_surge": 0.55,
                },
            },
        ),
        (
            "r3_score5_no_adx_size70",
            {
                "param_overrides.entry_detail_size_mults": {
                    "*:5:!adx_trending": 0.70,
                },
            },
        ),
        (
            "r3_score5_no_cpr_size65",
            {
                "param_overrides.entry_detail_size_mults": {
                    "*:5:!strong_cpr": 0.65,
                },
            },
        ),
        (
            "r3_score5_no_volume_or_adx_size",
            {
                "param_overrides.entry_detail_size_mults": {
                    "OR_BREAKOUT:5:!bar_vol_surge": 0.65,
                    "*:5:!adx_trending": 0.80,
                },
            },
        ),
        (
            "r3_score4_6_up_score5_down",
            {
                "param_overrides.entry_score_size_mults": {
                    "OR_BREAKOUT:4": 1.10,
                    "OR_BREAKOUT:5": 0.65,
                    "OR_BREAKOUT:6": 1.08,
                    "COMBINED_BREAKOUT:7": 1.10,
                    "PDH_BREAKOUT:6": 0.65,
                },
            },
        ),
    ],
    2: [
        (
            "r3_failstop8_mfe015_cur000_to_m030",
            {
                "param_overrides.failure_stop_bars": 8,
                "param_overrides.failure_stop_mfe_max_r": 0.15,
                "param_overrides.failure_stop_current_r_max": 0.00,
                "param_overrides.failure_stop_to_r": -0.30,
            },
        ),
        (
            "r3_failstop10_mfe020_cur000_to_m025",
            {
                "param_overrides.failure_stop_bars": 10,
                "param_overrides.failure_stop_mfe_max_r": 0.20,
                "param_overrides.failure_stop_current_r_max": 0.00,
                "param_overrides.failure_stop_to_r": -0.25,
            },
        ),
        (
            "r3_failstop12_mfe020_cur010_to_m020",
            {
                "param_overrides.failure_stop_bars": 12,
                "param_overrides.failure_stop_mfe_max_r": 0.20,
                "param_overrides.failure_stop_current_r_max": 0.10,
                "param_overrides.failure_stop_to_r": -0.20,
            },
        ),
        (
            "r3_failstop16_mfe025_cur000_to_m010",
            {
                "param_overrides.failure_stop_bars": 16,
                "param_overrides.failure_stop_mfe_max_r": 0.25,
                "param_overrides.failure_stop_current_r_max": 0.00,
                "param_overrides.failure_stop_to_r": -0.10,
            },
        ),
        (
            "r3_failstop10_mfe015_cur_m010_to_m040",
            {
                "param_overrides.failure_stop_bars": 10,
                "param_overrides.failure_stop_mfe_max_r": 0.15,
                "param_overrides.failure_stop_current_r_max": -0.10,
                "param_overrides.failure_stop_to_r": -0.40,
            },
        ),
    ],
    3: [
        (
            "r3_mfe14_min025_floor_m010",
            {
                "ablation.use_mfe_conviction_exit": True,
                "param_overrides.mfe_conviction_check_bars": 14,
                "param_overrides.mfe_conviction_min_r": 0.25,
                "param_overrides.mfe_conviction_floor_r": -0.10,
            },
        ),
        (
            "r3_mfe18_min030_floor_m005",
            {
                "ablation.use_mfe_conviction_exit": True,
                "param_overrides.mfe_conviction_check_bars": 18,
                "param_overrides.mfe_conviction_min_r": 0.30,
                "param_overrides.mfe_conviction_floor_r": -0.05,
            },
        ),
        (
            "r3_adaptive_start22_late015",
            {
                "ablation.use_adaptive_trail": True,
                "param_overrides.adaptive_trail_start_bars": 22,
                "param_overrides.adaptive_trail_tighten_bars": 25,
                "param_overrides.adaptive_trail_late_activate_r": 0.25,
                "param_overrides.adaptive_trail_late_distance_r": 0.15,
            },
        ),
        (
            "r3_adaptive_late012",
            {
                "ablation.use_adaptive_trail": True,
                "param_overrides.adaptive_trail_late_activate_r": 0.22,
                "param_overrides.adaptive_trail_late_distance_r": 0.12,
            },
        ),
        (
            "r3_fr_grace030_hold14",
            {
                "param_overrides.flow_reversal_min_hold_bars": 14,
                "param_overrides.fr_mfe_grace_r": 0.30,
                "param_overrides.fr_cpr_threshold": 0.25,
            },
        ),
    ],
    4: [
        (
            "r3_late1300_score6_avwap0045_decay",
            {
                "param_overrides.entry_window_end": "13:00:00",
                "param_overrides.late_entry_cutoff": "11:30:00",
                "param_overrides.late_entry_score_min": 6,
                "param_overrides.late_avwap_cap_pct": 0.0045,
                "param_overrides.orb_late_rvol_add_per_30m": 0.15,
                "param_overrides.orb_late_size_decay_per_30m": 0.10,
                "param_overrides.orb_late_size_floor": 0.60,
            },
        ),
        (
            "r3_late1330_score6_avwap0040_decay",
            {
                "param_overrides.entry_window_end": "13:30:00",
                "param_overrides.late_entry_cutoff": "11:30:00",
                "param_overrides.late_entry_score_min": 6,
                "param_overrides.late_avwap_cap_pct": 0.0040,
                "param_overrides.orb_late_rvol_add_per_30m": 0.20,
                "param_overrides.orb_late_size_decay_per_30m": 0.12,
                "param_overrides.orb_late_size_floor": 0.55,
            },
        ),
        (
            "r3_rvol190_quality62_cap100",
            {
                "param_overrides.rvol_threshold": 1.90,
                "ablation.use_orb_quality_gate": True,
                "param_overrides.orb_quality_score_min": 62.0,
                "ablation.use_breakout_distance_cap": True,
                "param_overrides.breakout_distance_cap_r": 1.00,
            },
        ),
        (
            "r3_selection30_quality62",
            {
                "param_overrides.selection_long_count": 30,
                "param_overrides.universe_cap": 800,
                "ablation.use_orb_quality_gate": True,
                "param_overrides.orb_quality_score_min": 62.0,
                "param_overrides.rvol_threshold": 2.0,
            },
        ),
        (
            "r3_reclaim_or_avwap_strict",
            {
                "param_overrides.reclaim_entry_mode": "or_avwap",
                "param_overrides.reclaim_min_rvol": 2.50,
                "param_overrides.reclaim_cpr_threshold": 0.62,
                "param_overrides.reclaim_max_avwap_premium_pct": 0.0045,
                "param_overrides.orb_structure_stop_mode": "reclaim",
            },
        ),
    ],
    5: [
        (
            "r3_entry_range100_breakout095",
            {
                "ablation.use_orb_entry_range_gate": True,
                "param_overrides.orb_entry_range_cap_r": 1.00,
                "ablation.use_breakout_distance_cap": True,
                "param_overrides.breakout_distance_cap_r": 0.95,
            },
        ),
        (
            "r3_or_width002_max009",
            {
                "ablation.use_or_width_min": True,
                "param_overrides.or_width_min_pct": 0.0020,
                "param_overrides.or_width_max_pct": 0.0090,
            },
        ),
        (
            "r3_avwap006_breakout100",
            {
                "ablation.use_avwap_distance_cap": True,
                "param_overrides.avwap_distance_cap_pct": 0.0060,
                "ablation.use_breakout_distance_cap": True,
                "param_overrides.breakout_distance_cap_r": 1.00,
            },
        ),
        (
            "r3_quality60_range110",
            {
                "ablation.use_orb_quality_gate": True,
                "param_overrides.orb_quality_score_min": 60.0,
                "ablation.use_orb_entry_range_gate": True,
                "param_overrides.orb_entry_range_cap_r": 1.10,
            },
        ),
    ],
    6: [
        (
            "r3_score_failstop_balanced",
            {
                "param_overrides.entry_detail_size_mults": {
                    "OR_BREAKOUT:5:!bar_vol_surge": 0.65,
                    "*:5:!adx_trending": 0.85,
                },
                "param_overrides.entry_score_size_mults": {
                    "OR_BREAKOUT:5": 0.70,
                    "COMBINED_BREAKOUT:7": 1.15,
                    "PDH_BREAKOUT:6": 0.65,
                },
                "param_overrides.failure_stop_bars": 10,
                "param_overrides.failure_stop_mfe_max_r": 0.20,
                "param_overrides.failure_stop_current_r_max": 0.00,
                "param_overrides.failure_stop_to_r": -0.25,
            },
        ),
        (
            "r3_archived_value_tune",
            {
                "param_overrides.adaptive_trail_late_activate_r": 0.22,
                "param_overrides.adaptive_trail_late_distance_r": 0.12,
                "param_overrides.entry_detail_size_mults": {
                    "OR_BREAKOUT:5:!bar_vol_surge": 0.55,
                },
                "param_overrides.entry_score_size_mults": {
                    "OR_BREAKOUT:5": 0.75,
                    "COMBINED_BREAKOUT:7": 1.15,
                    "PDH_BREAKOUT:6": 0.50,
                },
            },
        ),
        (
            "r3_failstop_latefreq_combo",
            {
                "param_overrides.failure_stop_bars": 12,
                "param_overrides.failure_stop_mfe_max_r": 0.20,
                "param_overrides.failure_stop_current_r_max": 0.10,
                "param_overrides.failure_stop_to_r": -0.20,
                "param_overrides.entry_window_end": "13:00:00",
                "param_overrides.late_entry_cutoff": "11:30:00",
                "param_overrides.late_entry_score_min": 6,
                "param_overrides.late_avwap_cap_pct": 0.0045,
                "param_overrides.orb_late_rvol_add_per_30m": 0.15,
                "param_overrides.orb_late_size_decay_per_30m": 0.10,
                "param_overrides.orb_late_size_floor": 0.60,
            },
        ),
        (
            "r3_discrim_geometry_combo",
            {
                "param_overrides.entry_detail_size_mults": {
                    "*:5:!strong_cpr": 0.70,
                    "*:5:!adx_trending": 0.85,
                },
                "param_overrides.entry_score_size_mults": {
                    "OR_BREAKOUT:4": 1.08,
                    "OR_BREAKOUT:5": 0.70,
                    "OR_BREAKOUT:6": 1.05,
                    "COMBINED_BREAKOUT:7": 1.12,
                    "PDH_BREAKOUT:6": 0.65,
                },
                "ablation.use_orb_entry_range_gate": True,
                "param_overrides.orb_entry_range_cap_r": 1.00,
                "ablation.use_breakout_distance_cap": True,
                "param_overrides.breakout_distance_cap_r": 0.95,
            },
        ),
        (
            "r3_profit_retention_combo",
            {
                "param_overrides.failure_stop_bars": 10,
                "param_overrides.failure_stop_mfe_max_r": 0.15,
                "param_overrides.failure_stop_current_r_max": -0.10,
                "param_overrides.failure_stop_to_r": -0.40,
                "ablation.use_adaptive_trail": True,
                "param_overrides.adaptive_trail_start_bars": 22,
                "param_overrides.adaptive_trail_tighten_bars": 25,
                "param_overrides.adaptive_trail_late_activate_r": 0.25,
                "param_overrides.adaptive_trail_late_distance_r": 0.15,
            },
        ),
    ],
    7: [
        (
            "r3_mature2_struct_avwap_stop_m015",
            {
                "param_overrides.maturation_stop_bars": 2,
                "param_overrides.maturation_stop_min_failed_checks": 2,
                "param_overrides.maturation_stop_min_current_r": -0.05,
                "param_overrides.maturation_stop_min_mfe_r": 0.05,
                "param_overrides.maturation_stop_min_rvol_ratio": 0.25,
                "param_overrides.maturation_stop_require_above_breakout": True,
                "param_overrides.maturation_stop_require_above_avwap": True,
                "param_overrides.maturation_stop_to_r": -0.15,
            },
        ),
        (
            "r3_mature3_struct_path_stop_m010",
            {
                "param_overrides.maturation_stop_bars": 3,
                "param_overrides.maturation_stop_min_failed_checks": 2,
                "param_overrides.maturation_stop_min_current_r": 0.00,
                "param_overrides.maturation_stop_min_mfe_r": 0.10,
                "param_overrides.maturation_stop_max_mae_r": 0.80,
                "param_overrides.maturation_stop_require_above_breakout": True,
                "param_overrides.maturation_stop_to_r": -0.10,
            },
        ),
        (
            "r3_mature3_volume_path_stop_m010",
            {
                "param_overrides.maturation_stop_bars": 3,
                "param_overrides.maturation_stop_min_failed_checks": 2,
                "param_overrides.maturation_stop_min_current_r": -0.05,
                "param_overrides.maturation_stop_min_mfe_r": 0.10,
                "param_overrides.maturation_stop_max_mae_r": 0.75,
                "param_overrides.maturation_stop_min_rvol_ratio": 0.40,
                "param_overrides.maturation_stop_to_r": -0.10,
            },
        ),
        (
            "r3_mature4_strict_hold_stop_flat",
            {
                "param_overrides.maturation_stop_bars": 4,
                "param_overrides.maturation_stop_min_failed_checks": 2,
                "param_overrides.maturation_stop_min_current_r": 0.00,
                "param_overrides.maturation_stop_min_mfe_r": 0.15,
                "param_overrides.maturation_stop_max_mae_r": 0.70,
                "param_overrides.maturation_stop_min_rvol_ratio": 0.35,
                "param_overrides.maturation_stop_require_above_breakout": True,
                "param_overrides.maturation_stop_require_above_avwap": True,
                "param_overrides.maturation_stop_to_r": 0.00,
            },
        ),
        (
            "r3_mature2_soft_path_stop_m020",
            {
                "param_overrides.maturation_stop_bars": 2,
                "param_overrides.maturation_stop_min_failed_checks": 1,
                "param_overrides.maturation_stop_min_current_r": -0.15,
                "param_overrides.maturation_stop_min_mfe_r": 0.05,
                "param_overrides.maturation_stop_max_mae_r": 0.90,
                "param_overrides.maturation_stop_to_r": -0.20,
            },
        ),
    ],
    8: [
        (
            "r3_confirm1_struct_avwap_size105",
            {
                "param_overrides.entry_confirmation_bars": 1,
                "param_overrides.entry_confirmation_min_current_r": -0.05,
                "param_overrides.entry_confirmation_min_mfe_r": 0.05,
                "param_overrides.entry_confirmation_max_mae_r": 0.90,
                "param_overrides.entry_confirmation_min_rvol_ratio": 0.25,
                "param_overrides.entry_confirmation_require_above_breakout": True,
                "param_overrides.entry_confirmation_require_above_avwap": True,
                "param_overrides.entry_confirmation_size_mult": 1.05,
            },
        ),
        (
            "r3_confirm2_struct_path_size110",
            {
                "param_overrides.entry_confirmation_bars": 2,
                "param_overrides.entry_confirmation_min_current_r": 0.00,
                "param_overrides.entry_confirmation_min_mfe_r": 0.10,
                "param_overrides.entry_confirmation_max_mae_r": 0.80,
                "param_overrides.entry_confirmation_min_rvol_ratio": 0.25,
                "param_overrides.entry_confirmation_require_above_breakout": True,
                "param_overrides.entry_confirmation_require_above_avwap": True,
                "param_overrides.entry_confirmation_size_mult": 1.10,
            },
        ),
        (
            "r3_confirm2_volume_persist_size110",
            {
                "param_overrides.entry_confirmation_bars": 2,
                "param_overrides.entry_confirmation_min_current_r": -0.02,
                "param_overrides.entry_confirmation_min_mfe_r": 0.10,
                "param_overrides.entry_confirmation_max_mae_r": 0.80,
                "param_overrides.entry_confirmation_min_rvol_ratio": 0.40,
                "param_overrides.entry_confirmation_require_above_breakout": True,
                "param_overrides.entry_confirmation_size_mult": 1.10,
            },
        ),
        (
            "r3_confirm1_mature_stop_combo",
            {
                "param_overrides.entry_confirmation_bars": 1,
                "param_overrides.entry_confirmation_min_current_r": -0.05,
                "param_overrides.entry_confirmation_min_mfe_r": 0.05,
                "param_overrides.entry_confirmation_max_mae_r": 0.90,
                "param_overrides.entry_confirmation_min_rvol_ratio": 0.25,
                "param_overrides.entry_confirmation_require_above_breakout": True,
                "param_overrides.entry_confirmation_require_above_avwap": True,
                "param_overrides.entry_confirmation_size_mult": 1.05,
                "param_overrides.maturation_stop_bars": 3,
                "param_overrides.maturation_stop_min_failed_checks": 2,
                "param_overrides.maturation_stop_min_current_r": -0.05,
                "param_overrides.maturation_stop_min_mfe_r": 0.10,
                "param_overrides.maturation_stop_max_mae_r": 0.80,
                "param_overrides.maturation_stop_require_above_breakout": True,
                "param_overrides.maturation_stop_to_r": -0.10,
            },
        ),
    ],
}


def get_phase_candidates(
    phase: int,
    *,
    experiment_filter: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    experiments = [
        (name, mutations)
        for name, mutations in PHASE_CANDIDATES.get(phase, [])
        if not is_small_sample_overfit_candidate(name, mutations)
    ]
    if experiment_filter:
        experiments = [
            (name, mutations)
            for name, mutations in experiments
            if name in experiment_filter
        ]
    return experiments
