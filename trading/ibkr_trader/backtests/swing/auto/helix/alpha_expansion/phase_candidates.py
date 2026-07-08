"""Candidate schedule for the next Helix phased auto round."""
from __future__ import annotations

from typing import Any


def get_phase_candidates(
    phase: int,
    prior_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates()
    elif phase == 4:
        candidates = _phase_4_candidates(prior_mutations or {})
    else:
        candidates = []

    if suggested_experiments:
        seen = {name for name, _ in candidates}
        for name, mutations in suggested_experiments:
            if name not in seen:
                candidates.append((name, mutations))
                seen.add(name)
    return candidates


def _phase_1_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("p1_disable_class_c", {"flags.disable_class_c": True}),
        ("p1_class_c_smaller_trend", {"param_overrides.CLASS_C_SIZE_TREND": 0.25}),
        ("p1_class_c_smaller_all", {
            "param_overrides.CLASS_C_SIZE_CHOP": 0.80,
            "param_overrides.CLASS_C_SIZE_COUNTER": 0.65,
            "param_overrides.CLASS_C_SIZE_TREND": 0.25,
        }),
        ("p1_class_d_mom_lb2", {"param_overrides.CLASS_D_MOM_LOOKBACK": 2}),
        ("p1_class_d_mom_lb4", {"param_overrides.CLASS_D_MOM_LOOKBACK": 4}),
        ("p1_class_d_mom_lb5", {"param_overrides.CLASS_D_MOM_LOOKBACK": 5}),
        ("p1_class_d_size_070", {"param_overrides.CLASS_D_SIZE_TREND": 0.70}),
        ("p1_class_d_size_090", {"param_overrides.CLASS_D_SIZE_TREND": 0.90}),
        ("p1_class_b_min_adx_18", {"param_overrides.CLASS_B_MIN_ADX": 18.0}),
        ("p1_class_b_min_adx_24", {"param_overrides.CLASS_B_MIN_ADX": 24.0}),
        ("p1_class_b_min_adx_28", {"param_overrides.CLASS_B_MIN_ADX": 28.0}),
        ("p1_class_b_mom_lb4", {"param_overrides.CLASS_B_MOM_LOOKBACK": 4}),
        ("p1_class_b_mom_lb6", {"param_overrides.CLASS_B_MOM_LOOKBACK": 6}),
        ("p1_class_b_mom_lb7", {"param_overrides.CLASS_B_MOM_LOOKBACK": 7}),
        ("p1_div_floor_006", {"param_overrides.DIV_MAG_FLOOR": 0.06}),
        ("p1_div_pct_35", {"param_overrides.DIV_MAG_PERCENTILE": 35}),
        ("p1_div_floor_006_pct35", {
            "param_overrides.DIV_MAG_FLOOR": 0.06,
            "param_overrides.DIV_MAG_PERCENTILE": 35,
        }),
        ("p1_adx_cap_45", {"param_overrides.ADX_UPPER_GATE": 45.0}),
        ("p1_adx_cap_50", {"param_overrides.ADX_UPPER_GATE": 50.0}),
        ("p1_adx_cap_off", {"param_overrides.ADX_UPPER_GATE": 999.0}),
        ("p1_extreme_vol_97", {"param_overrides.EXTREME_VOL_PCT": 97.0}),
        ("p1_extreme_vol_off", {"param_overrides.EXTREME_VOL_PCT": 999.0}),
        ("p1_reenable_class_a_micro", {
            "flags.disable_class_a": False,
            "param_overrides.CLASS_A_SIZE_TREND": 0.35,
            "param_overrides.CLASS_A_SIZE_CHOP": 0.25,
            "param_overrides.CLASS_A_SIZE_COUNTER": 0.20,
            "param_overrides.DIV_MAG_FLOOR": 0.08,
        }),
    ]


def _phase_2_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("p2_ttl_1h_4", {"param_overrides.TTL_1H_HOURS": 4}),
        ("p2_ttl_1h_8", {"param_overrides.TTL_1H_HOURS": 8}),
        ("p2_ttl_1h_10", {"param_overrides.TTL_1H_HOURS": 10}),
        ("p2_ttl_1h_12", {"param_overrides.TTL_1H_HOURS": 12}),
        ("p2_ttl_4h_16", {"param_overrides.TTL_4H_HOURS": 16}),
        ("p2_ttl_4h_20", {"param_overrides.TTL_4H_HOURS": 20}),
        ("p2_ttl_4h_24", {"param_overrides.TTL_4H_HOURS": 24}),
        ("p2_ttl_combo_8_16", {
            "param_overrides.TTL_1H_HOURS": 8,
            "param_overrides.TTL_4H_HOURS": 16,
        }),
        ("p2_ttl_combo_10_20", {
            "param_overrides.TTL_1H_HOURS": 10,
            "param_overrides.TTL_4H_HOURS": 20,
        }),
        ("p2_stop_1h_std_045", {"param_overrides.STOP_1H_STD": 0.45}),
        ("p2_stop_1h_std_060", {"param_overrides.STOP_1H_STD": 0.60}),
        ("p2_stop_1h_highvol_065", {"param_overrides.STOP_1H_HIGHVOL": 0.65}),
        ("p2_stop_1h_highvol_085", {"param_overrides.STOP_1H_HIGHVOL": 0.85}),
        ("p2_stop_1h_wide_combo", {
            "param_overrides.STOP_1H_STD": 0.60,
            "param_overrides.STOP_1H_HIGHVOL": 0.85,
        }),
        ("p2_stop_4h_065", {"param_overrides.STOP_4H_MULT": 0.65}),
        ("p2_stop_4h_090", {"param_overrides.STOP_4H_MULT": 0.90}),
        ("p2_disable_corridor_cap", {"flags.disable_corridor_cap": True}),
        ("p2_adx_cap_55", {"param_overrides.ADX_UPPER_GATE": 55.0}),
        ("p2_extreme_vol_99", {"param_overrides.EXTREME_VOL_PCT": 99.0}),
        ("p2_extreme_vol_off", {"param_overrides.EXTREME_VOL_PCT": 999.0}),
    ]


def _phase_3_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("p3_add_risk_080", {"param_overrides.ADD_RISK_FRAC": 0.80}),
        ("p3_add_risk_095", {"param_overrides.ADD_RISK_FRAC": 0.95}),
        ("p3_add_risk_110", {"param_overrides.ADD_RISK_FRAC": 1.10}),
        ("p3_add_risk_120", {"param_overrides.ADD_RISK_FRAC": 1.20}),
        ("p3_add_risk_130", {"param_overrides.ADD_RISK_FRAC": 1.30}),
        ("p3_add_1h_r_065", {"param_overrides.ADD_1H_R": 0.65}),
        ("p3_add_1h_r_075", {"param_overrides.ADD_1H_R": 0.75}),
        ("p3_add_1h_r_105", {"param_overrides.ADD_1H_R": 1.05}),
        ("p3_add_1h_r_115", {"param_overrides.ADD_1H_R": 1.15}),
        ("p3_add_4h_r_025", {"param_overrides.ADD_4H_R": 0.25}),
        ("p3_add_4h_r_030", {"param_overrides.ADD_4H_R": 0.30}),
        ("p3_add_4h_r_050", {"param_overrides.ADD_4H_R": 0.50}),
        ("p3_add_4h_r_060", {"param_overrides.ADD_4H_R": 0.60}),
        ("p3_add_min_2", {"param_overrides.ADD_MIN_BARS": 2}),
        ("p3_add_min_3", {"param_overrides.ADD_MIN_BARS": 3}),
        ("p3_add_min_5", {"param_overrides.ADD_MIN_BARS": 5}),
        ("p3_add_min_6", {"param_overrides.ADD_MIN_BARS": 6}),
        ("p3_add_max_25", {"param_overrides.ADD_MAX_BARS": 25}),
        ("p3_add_max_45", {"param_overrides.ADD_MAX_BARS": 45}),
        ("p3_add_max_55", {"param_overrides.ADD_MAX_BARS": 55}),
        ("p3_add_gate_020", {"param_overrides.ADD_PRICE_GATE_ATR_MULT": 0.20}),
        ("p3_add_gate_035", {"param_overrides.ADD_PRICE_GATE_ATR_MULT": 0.35}),
        ("p3_add_gate_065", {"param_overrides.ADD_PRICE_GATE_ATR_MULT": 0.65}),
        ("p3_add_gate_080", {"param_overrides.ADD_PRICE_GATE_ATR_MULT": 0.80}),
        ("p3_partial_frac_055", {"param_overrides.PARTIAL_2P5_FRAC": 0.55}),
        ("p3_partial_frac_065", {"param_overrides.PARTIAL_2P5_FRAC": 0.65}),
        ("p3_partial_frac_080", {"param_overrides.PARTIAL_2P5_FRAC": 0.80}),
        ("p3_partial_225", {"param_overrides.R_PARTIAL_2P5": 2.25}),
        ("p3_partial_275", {"param_overrides.R_PARTIAL_2P5": 2.75}),
        ("p3_partial_300", {"param_overrides.R_PARTIAL_2P5": 3.00}),
        ("p3_disable_partial_2p5", {"flags.disable_partial_2p5r": True}),
        ("p3_partial_light_late", {
            "param_overrides.PARTIAL_2P5_FRAC": 0.55,
            "param_overrides.R_PARTIAL_2P5": 2.75,
        }),
    ]


def _phase_4_candidates(prior_mutations: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("p4_class_d_bail_6_m025", {
            "param_overrides.CLASS_D_BAIL_BARS": 6,
            "param_overrides.CLASS_D_BAIL_R_THRESH": -0.25,
        }),
        ("p4_class_d_bail_8_m025", {
            "param_overrides.CLASS_D_BAIL_BARS": 8,
            "param_overrides.CLASS_D_BAIL_R_THRESH": -0.25,
        }),
        ("p4_class_d_bail_10_m035", {
            "param_overrides.CLASS_D_BAIL_BARS": 10,
            "param_overrides.CLASS_D_BAIL_R_THRESH": -0.35,
        }),
        ("p4_class_d_bail_6_m050", {
            "param_overrides.CLASS_D_BAIL_BARS": 6,
            "param_overrides.CLASS_D_BAIL_R_THRESH": -0.50,
        }),
        ("p4_class_b_bail_8_m025", {
            "param_overrides.CLASS_B_BAIL_BARS": 8,
            "param_overrides.CLASS_B_BAIL_R_THRESH": -0.25,
        }),
        ("p4_class_b_bail_12_m025", {
            "param_overrides.CLASS_B_BAIL_BARS": 12,
            "param_overrides.CLASS_B_BAIL_R_THRESH": -0.25,
        }),
        ("p4_class_b_bail_8_m075", {
            "param_overrides.CLASS_B_BAIL_BARS": 8,
            "param_overrides.CLASS_B_BAIL_R_THRESH": -0.75,
        }),
        ("p4_early_stale_12", {"param_overrides.EARLY_STALE_BARS": 12}),
        ("p4_early_stale_16", {"param_overrides.EARLY_STALE_BARS": 16}),
        ("p4_early_stale_24", {"param_overrides.EARLY_STALE_BARS": 24}),
        ("p4_stale_1h_22", {"param_overrides.STALE_1H_BARS": 22}),
        ("p4_stale_1h_26", {"param_overrides.STALE_1H_BARS": 26}),
        ("p4_stale_1h_34", {"param_overrides.STALE_1H_BARS": 34}),
        ("p4_stale_1h_40", {"param_overrides.STALE_1H_BARS": 40}),
        ("p4_stale_thresh_030", {"param_overrides.STALE_R_THRESH": 0.30}),
        ("p4_stale_thresh_075", {"param_overrides.STALE_R_THRESH": 0.75}),
        ("p4_stale_floor_000", {"param_overrides.STALE_FLATTEN_R_FLOOR": 0.00}),
        ("p4_stale_floor_m010", {"param_overrides.STALE_FLATTEN_R_FLOOR": -0.10}),
        ("p4_stale_floor_m050", {"param_overrides.STALE_FLATTEN_R_FLOOR": -0.50}),
        ("p4_be_1h_060", {"param_overrides.R_BE_1H": 0.60}),
        ("p4_be_1h_085", {"param_overrides.R_BE_1H": 0.85}),
        ("p4_be_1h_105", {"param_overrides.R_BE_1H": 1.05}),
        ("p4_be_offset_016", {"param_overrides.BE_ATR1H_OFFSET": 0.16}),
        ("p4_be_offset_030", {"param_overrides.BE_ATR1H_OFFSET": 0.30}),
        ("p4_be_offset_036", {"param_overrides.BE_ATR1H_OFFSET": 0.36}),
        ("p4_slower_be_combo", {
            "param_overrides.R_BE_1H": 1.05,
            "param_overrides.BE_ATR1H_OFFSET": 0.36,
        }),
        ("p4_trail_stall_4", {"param_overrides.TRAIL_STALL_ONSET": 4}),
        ("p4_trail_stall_5", {"param_overrides.TRAIL_STALL_ONSET": 5}),
        ("p4_trail_stall_8", {"param_overrides.TRAIL_STALL_ONSET": 8}),
        ("p4_trail_fade_min_070", {"param_overrides.TRAIL_FADE_MIN_R": 0.70}),
        ("p4_trail_fade_min_130", {"param_overrides.TRAIL_FADE_MIN_R": 1.30}),
        ("p4_trail_delay_3", {"param_overrides.TRAIL_PROFIT_DELAY_BARS": 3}),
        ("p4_trail_delay_6", {"param_overrides.TRAIL_PROFIT_DELAY_BARS": 6}),
    ]

    candidates.extend(_local_finetune_candidates(prior_mutations))
    return candidates


def _local_finetune_candidates(prior_mutations: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    keys = {
        "param_overrides.ADD_1H_R",
        "param_overrides.ADD_4H_R",
        "param_overrides.ADD_RISK_FRAC",
        "param_overrides.PARTIAL_2P5_FRAC",
        "param_overrides.TRAIL_STALL_ONSET",
        "param_overrides.BE_ATR1H_OFFSET",
        "param_overrides.CLASS_B_BAIL_BARS",
        "param_overrides.CLASS_B_MOM_LOOKBACK",
        "param_overrides.ADX_UPPER_GATE",
    }
    finetune: list[tuple[str, dict[str, Any]]] = []
    for key in sorted(keys):
        if key not in prior_mutations:
            continue
        value = prior_mutations[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        for label, multiplier in (("m10", 0.90), ("p10", 1.10)):
            new_value: float | int
            if isinstance(value, int):
                new_value = int(round(value * multiplier))
                if new_value == value:
                    continue
            else:
                new_value = round(float(value) * multiplier, 6)
            finetune.append((f"p4_finetune_{key.replace('.', '_')}_{label}", {key: new_value}))
    return finetune
