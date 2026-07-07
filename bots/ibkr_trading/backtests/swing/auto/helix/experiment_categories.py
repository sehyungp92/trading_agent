"""Helix experiment definitions for phased Helix optimization.

Each experiment is a (name, mutations_dict) tuple.
Categories: CLASS_PRUNING, ENTRY_QUALITY, CLASS_D_QUALITY,
            STRUCTURAL_ABLATION, STOP_PLACEMENT, TRAILING,
            INLINE_TRAILING, LEAKAGE_GUARD, STALE_BAIL, PARTIALS_BE,
            VOLATILITY, ADDON, ADDON_EXPANSION, CIRCUIT_BREAKER
"""
from __future__ import annotations


def _class_pruning_experiments() -> list[tuple[str, dict]]:
    """5 CLASS_PRUNING experiments (Phase 1)."""
    return [
        ("prune_no_class_a", {"flags.disable_class_a": True}),
        ("prune_no_class_c", {"flags.disable_class_c": True}),
        ("prune_no_4h", {"flags.disable_class_a": True, "flags.disable_class_c": True}),
        ("prune_d_only", {"flags.disable_class_a": True, "flags.disable_class_b": True, "flags.disable_class_c": True}),
        ("prune_no_class_b", {"flags.disable_class_b": True}),
    ]


def _entry_quality_experiments() -> list[tuple[str, dict]]:
    """14 ENTRY_QUALITY experiments (Phase 1)."""
    return [
        ("entry_b_adx_20", {"param_overrides.CLASS_B_MIN_ADX": 20}),
        ("entry_b_adx_24", {"param_overrides.CLASS_B_MIN_ADX": 24}),
        ("entry_b_adx_25", {"param_overrides.CLASS_B_MIN_ADX": 25}),
        ("entry_b_adx_30", {"param_overrides.CLASS_B_MIN_ADX": 30}),
        ("entry_pivot_sep_12", {"param_overrides.CLASS_B_MIN_PIVOT_SEP_BARS": 12}),
        ("entry_pivot_sep_15", {"param_overrides.CLASS_B_MIN_PIVOT_SEP_BARS": 15}),
        ("entry_div_mag_floor_06", {"param_overrides.DIV_MAG_FLOOR": 0.06}),
        ("entry_div_mag_floor_10", {"param_overrides.DIV_MAG_FLOOR": 0.10}),
        ("entry_div_mag_pctile_40", {"param_overrides.DIV_MAG_PERCENTILE": 40}),
        ("entry_div_mag_pctile_50", {"param_overrides.DIV_MAG_PERCENTILE": 50}),
        ("entry_d_mom_5", {"param_overrides.CLASS_D_MOM_LOOKBACK": 5}),
        ("entry_d_mom_6", {"param_overrides.CLASS_D_MOM_LOOKBACK": 6}),
        ("entry_adx_cap_40", {"param_overrides.ADX_UPPER_GATE": 40}),
        ("entry_adx_cap_45", {"param_overrides.ADX_UPPER_GATE": 45}),
        ("entry_b_mom_lookback_4", {"param_overrides.CLASS_B_MOM_LOOKBACK": 4}),
        ("entry_b_mom_lookback_5", {"param_overrides.CLASS_B_MOM_LOOKBACK": 5}),
    ]


def _class_d_quality_experiments() -> list[tuple[str, dict]]:
    """Class D momentum filters targeted at durable side-level alpha."""
    return [
        ("entry_d_restore_streak_1", {"param_overrides.CLASS_D_REGIME_STREAK_MIN": 1}),
        ("entry_d_restore_streak_0", {"param_overrides.CLASS_D_REGIME_STREAK_MIN": 0}),
        ("entry_d_min_adx_18", {"param_overrides.CLASS_D_MIN_ADX": 18.0}),
        ("entry_d_min_adx_22", {"param_overrides.CLASS_D_MIN_ADX": 22.0}),
        ("entry_d_short_adx_16", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0}),
        ("entry_d_short_adx_20", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0}),
        ("entry_d_short_adx_24", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 24.0}),
        ("entry_d_short_adx_28", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 28.0}),
        ("entry_d_short_adx_0", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0}),
        ("entry_d_hist_sign", {"param_overrides.CLASS_D_HIST_SIGN_GATE": True}),
        ("entry_d_regime_streak_2", {"param_overrides.CLASS_D_REGIME_STREAK_MIN": 2}),
        ("entry_d_regime_streak_3", {"param_overrides.CLASS_D_REGIME_STREAK_MIN": 3}),
        (
            "entry_d_short_adx24_hist",
            {
                "param_overrides.CLASS_D_SHORT_MIN_ADX": 24.0,
                "param_overrides.CLASS_D_HIST_SIGN_GATE": True,
            },
        ),
        (
            "entry_d_short_adx24_streak2",
            {
                "param_overrides.CLASS_D_SHORT_MIN_ADX": 24.0,
                "param_overrides.CLASS_D_REGIME_STREAK_MIN": 2,
            },
        ),
    ]


def _class_d_entry_discriminator_experiments() -> list[tuple[str, dict]]:
    """Pre-entry Class D filters aimed at fast failed breakouts, not tail clipping."""
    return [
        ("d_sep_4", {"param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4}),
        ("d_sep_6", {"param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 6}),
        ("d_p2_age_20", {"param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 20}),
        ("d_p2_age_24", {"param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 24}),
        ("d_p2_age_30", {"param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 30}),
        ("d_daily_ext_300", {"param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00}),
        ("d_daily_ext_350", {"param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.50}),
        ("d_sep4_age20", {
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 20,
        }),
        ("d_sep4_dailyext300", {
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("d_short0_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("d_short0_sep4_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
    ]


def _structural_ablation_experiments() -> list[tuple[str, dict]]:
    """6 STRUCTURAL_ABLATION experiments (Phase 1 + Phase 3)."""
    return [
        # Phase 1
        ("struct_no_corridor", {"flags.disable_corridor_cap": True}),
        # Phase 3
        ("struct_no_extreme_vol", {"flags.disable_extreme_vol_gate": True}),
        ("struct_no_spread_gate", {"flags.disable_spread_gate": True}),
        ("struct_no_chandelier", {"flags.disable_chandelier_trailing": True}),
        ("struct_no_basket", {"flags.disable_basket_rule": True}),
        ("partial_disable_all", {"flags.disable_partial_2p5r": True, "flags.disable_partial_5r": True}),
    ]


def _stop_placement_experiments() -> list[tuple[str, dict]]:
    """10 STOP_PLACEMENT experiments (Phase 2)."""
    return [
        ("stop_1h_std_060", {"param_overrides.STOP_1H_STD": 0.60}),
        ("stop_1h_std_070", {"param_overrides.STOP_1H_STD": 0.70}),
        ("stop_1h_highvol_090", {"param_overrides.STOP_1H_HIGHVOL": 0.90}),
        ("stop_1h_highvol_100", {"param_overrides.STOP_1H_HIGHVOL": 1.00}),
        ("stop_4h_mult_090", {"param_overrides.STOP_4H_MULT": 0.90}),
        ("stop_4h_mult_100", {"param_overrides.STOP_4H_MULT": 1.00}),
        ("stop_be_offset_010", {"param_overrides.BE_ATR1H_OFFSET": 0.10}),
        ("stop_be_offset_020", {"param_overrides.BE_ATR1H_OFFSET": 0.20}),
        ("stop_emergency_neg15", {"param_overrides.EMERGENCY_STOP_R": -1.5}),
        ("stop_emergency_neg25", {"param_overrides.EMERGENCY_STOP_R": -2.5}),
    ]


def _trailing_experiments() -> list[tuple[str, dict]]:
    """12 TRAILING experiments (Phase 2) -- configurable via strategy_2.stops."""
    return [
        ("trail_base_30", {"param_overrides.TRAIL_BASE": 3.0}),
        ("trail_base_35", {"param_overrides.TRAIL_BASE": 3.5}),
        ("trail_base_50", {"param_overrides.TRAIL_BASE": 5.0}),
        ("trail_min_15", {"param_overrides.TRAIL_MIN": 1.5}),
        ("trail_min_25", {"param_overrides.TRAIL_MIN": 2.5}),
        ("trail_r_div_30", {"param_overrides.TRAIL_R_DIV": 3.0}),
        ("trail_r_div_40", {"param_overrides.TRAIL_R_DIV": 4.0}),
        ("trail_r_div_70", {"param_overrides.TRAIL_R_DIV": 7.0}),
        ("trail_mom_bonus_025", {"param_overrides.TRAIL_MOM_BONUS": 0.25}),
        ("trail_mom_bonus_075", {"param_overrides.TRAIL_MOM_BONUS": 0.75}),
        ("trail_delay_2", {"param_overrides.TRAIL_PROFIT_DELAY_BARS": 2}),
        ("trail_delay_6", {"param_overrides.TRAIL_PROFIT_DELAY_BARS": 6}),
    ]


def _inline_trailing_experiments() -> list[tuple[str, dict]]:
    """12 INLINE_TRAILING experiments (Phase 2) -- requires engine prereq 0A."""
    return [
        ("fade_penalty_050", {"param_overrides.TRAIL_FADE_PENALTY": 0.50}),
        ("fade_penalty_100", {"param_overrides.TRAIL_FADE_PENALTY": 1.00}),
        ("fade_floor_10", {"param_overrides.TRAIL_FADE_FLOOR": 1.0}),
        ("fade_floor_20", {"param_overrides.TRAIL_FADE_FLOOR": 2.0}),
        ("fade_min_r_05", {"param_overrides.TRAIL_FADE_MIN_R": 0.5}),
        ("fade_min_r_15", {"param_overrides.TRAIL_FADE_MIN_R": 1.5}),
        ("timedecay_onset_15", {"param_overrides.TRAIL_TIMEDECAY_ONSET": 15}),
        ("timedecay_onset_25", {"param_overrides.TRAIL_TIMEDECAY_ONSET": 25}),
        ("timedecay_rate_008", {"param_overrides.TRAIL_TIMEDECAY_RATE": 0.08}),
        ("stall_onset_6", {"param_overrides.TRAIL_STALL_ONSET": 6}),
        ("stall_onset_12", {"param_overrides.TRAIL_STALL_ONSET": 12}),
        ("stall_rate_012", {"param_overrides.TRAIL_STALL_RATE": 0.12}),
    ]


def _class_specific_trailing_experiments() -> list[tuple[str, dict]]:
    """Class-level trailing shapes focused on Class D leakage without clipping Class B."""
    return [
        ("class_d_trail_base_30", {"param_overrides.TRAIL_BASE_CLASS_D": 3.0}),
        ("class_d_trail_base_35", {"param_overrides.TRAIL_BASE_CLASS_D": 3.5}),
        ("class_d_trail_rdiv_60", {"param_overrides.TRAIL_R_DIV_CLASS_D": 6.0}),
        ("class_d_trail_rdiv_75", {"param_overrides.TRAIL_R_DIV_CLASS_D": 7.5}),
        ("class_d_stall_onset_4", {"param_overrides.TRAIL_STALL_ONSET_CLASS_D": 4}),
        ("class_d_stall_onset_6", {"param_overrides.TRAIL_STALL_ONSET_CLASS_D": 6}),
        (
            "class_d_tighter_fade",
            {
                "param_overrides.TRAIL_STALL_ONSET_CLASS_D": 4,
                "param_overrides.TRAIL_FADE_PENALTY_CLASS_D": 1.25,
                "param_overrides.TRAIL_FADE_MIN_R_CLASS_D": 0.75,
            },
        ),
    ]


def _leakage_guard_experiments() -> list[tuple[str, dict]]:
    """Existing RTS guard parameterizations for small/mid-MFE giveback leaks."""
    return [
        (
            "rts_guard_mfe050_be",
            {
                "param_overrides.RTS_GUARD_MFE_R": 0.50,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.35,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.0,
            },
        ),
        (
            "rts_guard_mfe075_be",
            {
                "param_overrides.RTS_GUARD_MFE_R": 0.75,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.45,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.0,
            },
        ),
        (
            "rts_guard_mfe100_be",
            {
                "param_overrides.RTS_GUARD_MFE_R": 1.00,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.50,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.0,
            },
        ),
        (
            "rts_guard_mfe075_neg010",
            {
                "param_overrides.RTS_GUARD_MFE_R": 0.75,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.45,
                "param_overrides.RTS_GUARD_FLOOR_R": -0.10,
            },
        ),
        (
            "rts_guard_mfe075_fade2",
            {
                "param_overrides.RTS_GUARD_MFE_R": 0.75,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.45,
                "param_overrides.RTS_GUARD_FADE_BARS": 2,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.0,
            },
        ),
        (
            "rts_fail_mfe100_neg025",
            {
                "param_overrides.RTS_GUARD_MFE_R": 1.00,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.65,
                "param_overrides.RTS_FAIL_FLATTEN_R": -0.25,
            },
        ),
        (
            "rts_guard_mfe025_gb005_floor005",
            {
                "param_overrides.RTS_GUARD_MFE_R": 0.25,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.05,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.05,
            },
        ),
        (
            "rts_guard_mfe025_gb005_floor005_minbars10",
            {
                "param_overrides.RTS_GUARD_MFE_R": 0.25,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.05,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.05,
                "param_overrides.RTS_GUARD_MIN_BARS": 10,
            },
        ),
        (
            "rts_guard_mfe030_gb010_floor005",
            {
                "param_overrides.RTS_GUARD_MFE_R": 0.30,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.10,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.05,
            },
        ),
        (
            "freq_d_short0_sep4_daily_rts025",
            {
                "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
                "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
                "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
                "param_overrides.RTS_GUARD_MFE_R": 0.25,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.05,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.05,
            },
        ),
        (
            "freq_d_short0_sep4_daily_rts030",
            {
                "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
                "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
                "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
                "param_overrides.RTS_GUARD_MFE_R": 0.30,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.10,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.05,
            },
        ),
        (
            "freq_d_short0_sep4_classc_rts035",
            {
                "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
                "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
                "flags.disable_class_c": False,
                "param_overrides.RTS_GUARD_MFE_R": 0.35,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.15,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.10,
            },
        ),
    ]


def _stale_bail_experiments() -> list[tuple[str, dict]]:
    """16 STALE_BAIL experiments (Phase 2)."""
    return [
        ("stale_1h_20", {"param_overrides.STALE_1H_BARS": 20}),
        ("stale_1h_25", {"param_overrides.STALE_1H_BARS": 25}),
        ("stale_4h_10", {"param_overrides.STALE_4H_BARS": 10}),
        ("stale_r_thresh_03", {"param_overrides.STALE_R_THRESH": 0.3}),
        ("stale_r_thresh_075", {"param_overrides.STALE_R_THRESH": 0.75}),
        ("stale_floor_neg015", {"param_overrides.STALE_FLATTEN_R_FLOOR": -0.15}),
        ("stale_floor_0", {"param_overrides.STALE_FLATTEN_R_FLOOR": 0.0}),
        ("ttl_1h_4", {"param_overrides.TTL_1H_HOURS": 4}),
        ("ttl_1h_8", {"param_overrides.TTL_1H_HOURS": 8}),
        ("bail_b_bars_6", {"param_overrides.CLASS_B_BAIL_BARS": 6}),
        ("bail_b_bars_10", {"param_overrides.CLASS_B_BAIL_BARS": 10}),
        ("bail_b_r_neg035", {"param_overrides.CLASS_B_BAIL_R_THRESH": -0.35}),
        ("bail_d_bars_6", {"param_overrides.CLASS_D_BAIL_BARS": 6}),
        ("bail_d_bars_8", {"param_overrides.CLASS_D_BAIL_BARS": 8}),
        ("bail_d_bars_10", {"param_overrides.CLASS_D_BAIL_BARS": 10}),
        ("bail_d_r_neg030", {"param_overrides.CLASS_D_BAIL_R_THRESH": -0.3}),
    ]


def _partials_be_experiments() -> list[tuple[str, dict]]:
    """10 PARTIALS_BE experiments (Phase 2)."""
    return [
        ("be_1h_050", {"param_overrides.R_BE_1H": 0.50}),
        ("be_1h_100", {"param_overrides.R_BE_1H": 1.00}),
        ("be_4h_075", {"param_overrides.R_BE": 0.75}),
        ("be_4h_125", {"param_overrides.R_BE": 1.25}),
        ("partial_2p5_r_20", {"param_overrides.R_PARTIAL_2P5": 2.0}),
        ("partial_2p5_r_30", {"param_overrides.R_PARTIAL_2P5": 3.0}),
        ("partial_2p5_frac_040", {"param_overrides.PARTIAL_2P5_FRAC": 0.40}),
        ("partial_5_r_40", {"param_overrides.R_PARTIAL_5": 4.0}),
        ("partial_5_r_60", {"param_overrides.R_PARTIAL_5": 6.0}),
        ("partial_disable_2p5", {"flags.disable_partial_2p5r": True}),
    ]


def _volatility_experiments() -> list[tuple[str, dict]]:
    """10 VOLATILITY experiments (Phase 3)."""
    return [
        ("vol_vf_min_030", {"param_overrides.VOLFACTOR_MIN": 0.30}),
        ("vol_vf_min_050", {"param_overrides.VOLFACTOR_MIN": 0.50}),
        ("vol_vf_max_13", {"param_overrides.VOLFACTOR_MAX": 1.3}),
        ("vol_vf_max_20", {"param_overrides.VOLFACTOR_MAX": 2.0}),
        ("vol_extreme_90", {"param_overrides.EXTREME_VOL_PCT": 90}),
        ("vol_extreme_97", {"param_overrides.EXTREME_VOL_PCT": 97}),
        ("vol_high_75", {"param_overrides.HIGH_VOL_PCT": 75}),
        ("vol_high_90", {"param_overrides.HIGH_VOL_PCT": 90}),
        ("ema_4h_fast_15", {"param_overrides.EMA_4H_FAST": 15}),
        ("ema_4h_fast_25", {"param_overrides.EMA_4H_FAST": 25}),
    ]


def _addon_experiments() -> list[tuple[str, dict]]:
    """9 ADDON experiments (Phase 3)."""
    return [
        ("addon_1h_r_10", {"param_overrides.ADD_1H_R": 1.0}),
        ("addon_1h_r_20", {"param_overrides.ADD_1H_R": 2.0}),
        ("addon_4h_r_040", {"param_overrides.ADD_4H_R": 0.40}),
        ("addon_4h_r_080", {"param_overrides.ADD_4H_R": 0.80}),
        ("addon_risk_frac_030", {"param_overrides.ADD_RISK_FRAC": 0.30}),
        ("addon_risk_frac_070", {"param_overrides.ADD_RISK_FRAC": 0.70}),
        ("addon_max_bars_25", {"param_overrides.ADD_MAX_BARS": 25}),
        ("addon_max_bars_45", {"param_overrides.ADD_MAX_BARS": 45}),
        ("addon_disable", {"flags.disable_add_ons": True}),
    ]


def _addon_expansion_experiments() -> list[tuple[str, dict]]:
    """Local add-on expansion around the latest optimized add-on posture."""
    return [
        ("addon_1h_r_055", {"param_overrides.ADD_1H_R": 0.55}),
        ("addon_1h_r_075", {"param_overrides.ADD_1H_R": 0.75}),
        ("addon_risk_frac_150", {"param_overrides.ADD_RISK_FRAC": 1.50}),
        ("addon_risk_frac_190", {"param_overrides.ADD_RISK_FRAC": 1.90}),
        ("addon_risk_frac_220", {"param_overrides.ADD_RISK_FRAC": 2.20}),
        ("addon_max_bars_14", {"param_overrides.ADD_MAX_BARS": 14}),
        ("addon_max_bars_22", {"param_overrides.ADD_MAX_BARS": 22}),
        ("addon_max_bars_28", {"param_overrides.ADD_MAX_BARS": 28}),
        ("addon_price_gate_025", {"param_overrides.ADD_PRICE_GATE_ATR_MULT": 0.25}),
        ("addon_price_gate_075", {"param_overrides.ADD_PRICE_GATE_ATR_MULT": 0.75}),
    ]


def _circuit_breaker_experiments() -> list[tuple[str, dict]]:
    """8 CIRCUIT_BREAKER experiments (Phase 4)."""
    return [
        ("circuit_daily_neg20", {"param_overrides.DAILY_STOP_R": -2.0}),
        ("circuit_daily_neg30", {"param_overrides.DAILY_STOP_R": -3.0}),
        ("circuit_weekly_neg40", {"param_overrides.WEEKLY_STOP_R": -4.0}),
        ("circuit_weekly_neg60", {"param_overrides.WEEKLY_STOP_R": -6.0}),
        ("circuit_consec_2", {"param_overrides.CONSEC_STOPS_HALVE": 2}),
        ("circuit_consec_4", {"param_overrides.CONSEC_STOPS_HALVE": 4}),
        ("circuit_disable", {"flags.disable_circuit_breaker": True}),
        ("early_stale_15", {"param_overrides.EARLY_STALE_BARS": 15}),
    ]


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

EXPERIMENT_CATEGORIES: dict[str, list[tuple[str, dict]]] = {
    "CLASS_PRUNING": _class_pruning_experiments(),
    "ENTRY_QUALITY": _entry_quality_experiments(),
    "CLASS_D_QUALITY": _class_d_quality_experiments(),
    "CLASS_D_ENTRY_DISCRIMINATOR": _class_d_entry_discriminator_experiments(),
    "STRUCTURAL_ABLATION": _structural_ablation_experiments(),
    "STOP_PLACEMENT": _stop_placement_experiments(),
    "TRAILING": _trailing_experiments(),
    "INLINE_TRAILING": _inline_trailing_experiments(),
    "CLASS_SPECIFIC_TRAILING": _class_specific_trailing_experiments(),
    "LEAKAGE_GUARD": _leakage_guard_experiments(),
    "STALE_BAIL": _stale_bail_experiments(),
    "PARTIALS_BE": _partials_be_experiments(),
    "VOLATILITY": _volatility_experiments(),
    "ADDON": _addon_experiments(),
    "ADDON_EXPANSION": _addon_expansion_experiments(),
    "CIRCUIT_BREAKER": _circuit_breaker_experiments(),
}

# Structural ablation split: some are P1, some are P3
_P1_STRUCTURAL = {"struct_no_corridor"}
_P3_STRUCTURAL = {
    "struct_no_extreme_vol", "struct_no_spread_gate",
    "struct_no_chandelier", "struct_no_basket", "partial_disable_all",
}


def get_all_experiments() -> list[tuple[str, dict]]:
    """Return all experiments across all categories."""
    all_exps = []
    for cat_exps in EXPERIMENT_CATEGORIES.values():
        all_exps.extend(cat_exps)
    return all_exps


def get_category_experiments(categories: list[str]) -> list[tuple[str, dict]]:
    """Return experiments for specific categories."""
    exps = []
    for cat in categories:
        exps.extend(EXPERIMENT_CATEGORIES.get(cat, []))
    return exps
