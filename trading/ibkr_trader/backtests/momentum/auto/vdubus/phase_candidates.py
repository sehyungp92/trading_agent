"""VdubusNQ Round 3 phased auto-optimization candidates.

Round 3 starts from the Round 2 optimized config and targets the remaining
diagnostic alpha:

* hourly_align, slope, and no_signal shadow cohorts showed positive EV.
* Evening and daily-trend rejects looked correctly blocked.
* Winner capture and slow-death protection are still imperfect.

The experiments below are intentionally structural but disabled-by-default in
the shared Vdubus modules. They convert shadow alpha into explicit, testable
routes instead of globally removing gates.
"""
from __future__ import annotations

from typing import Any


def get_phase_candidates(
    phase: int,
    prior_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Return (name, mutations) pairs for the given phase."""
    prior = prior_mutations or {}
    candidates: list[tuple[str, dict]] = []

    if suggested_experiments:
        seen = set()
        for name, muts in suggested_experiments:
            if name not in seen:
                seen.add(name)
                candidates.append((name, muts))

    if phase == 1:
        candidates.extend(_phase_1_gate_alpha(prior))
    elif phase == 2:
        candidates.extend(_phase_2_no_signal_continuation(prior))
    elif phase == 3:
        candidates.extend(_phase_3_entry_frequency(prior))
    elif phase == 4:
        candidates.extend(_phase_4_exit_capture(prior))
    elif phase == 5:
        candidates.extend(_phase_5_session_cohorts(prior))
    elif phase == 6:
        candidates.extend(_phase_6_interactions(prior))

    return _dedupe(candidates)


def _dedupe(candidates: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen_names: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, muts in candidates:
        if name in seen_names:
            continue
        seen_names.add(name)
        unique.append((name, muts))
    return unique


def _phase_1_gate_alpha(prior: dict) -> list[tuple[str, dict]]:
    """Convert hourly_align and slope shadow alpha without disabling gates."""
    return [
        ("hourly_bypass_oc_eqs3_chop38", {
            "flags.hourly_bypass_quality": True,
            "param_overrides.HOURLY_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 3,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 38.0,
            "param_overrides.HOURLY_BYPASS_SIZE_MULT": 0.60,
        }),
        ("hourly_bypass_open_eqs3_chop35", {
            "flags.hourly_bypass_quality": True,
            "param_overrides.HOURLY_BYPASS_ALLOWED_WINDOWS": ["OPEN"],
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 3,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 35.0,
            "param_overrides.HOURLY_BYPASS_SIZE_MULT": 0.70,
        }),
        ("hourly_bypass_close_eqs3_chop42", {
            "flags.hourly_bypass_quality": True,
            "param_overrides.HOURLY_BYPASS_ALLOWED_WINDOWS": ["CLOSE"],
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 3,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 42.0,
            "param_overrides.HOURLY_BYPASS_SIZE_MULT": 0.70,
        }),
        ("hourly_bypass_rth_eqs4_fullsize", {
            "flags.hourly_bypass_quality": True,
            "param_overrides.HOURLY_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CORE", "CLOSE"],
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 4,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 36.0,
            "param_overrides.HOURLY_BYPASS_SIZE_MULT": 1.00,
        }),
        ("hourly_bypass_oc_eqs2_tiny", {
            "flags.hourly_bypass_quality": True,
            "param_overrides.HOURLY_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 2,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 32.0,
            "param_overrides.HOURLY_BYPASS_SIZE_MULT": 0.50,
        }),
        ("slope_bypass_rth_eqs3_chop38", {
            "flags.slope_bypass_quality": True,
            "param_overrides.SLOPE_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CORE", "CLOSE"],
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 3,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 38.0,
            "param_overrides.SLOPE_BYPASS_SIZE_MULT": 0.75,
        }),
        ("slope_bypass_open_close_eqs3", {
            "flags.slope_bypass_quality": True,
            "param_overrides.SLOPE_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 3,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 42.0,
            "param_overrides.SLOPE_BYPASS_SIZE_MULT": 0.85,
        }),
        ("slope_bypass_eqs4_fullsize", {
            "flags.slope_bypass_quality": True,
            "param_overrides.SLOPE_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CORE", "CLOSE"],
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 4,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 36.0,
            "param_overrides.SLOPE_BYPASS_SIZE_MULT": 1.00,
        }),
        ("slope_bypass_strong_mom", {
            "flags.slope_bypass_quality": True,
            "param_overrides.SLOPE_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CORE", "CLOSE"],
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 2,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 35.0,
            "param_overrides.SLOPE_BYPASS_MOM_ABS_MIN": 2.0,
            "param_overrides.SLOPE_BYPASS_SIZE_MULT": 0.90,
        }),
        ("combined_bypass_strict", {
            "flags.hourly_bypass_quality": True,
            "flags.slope_bypass_quality": True,
            "param_overrides.HOURLY_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 4,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 34.0,
            "param_overrides.HOURLY_BYPASS_SIZE_MULT": 0.60,
            "param_overrides.SLOPE_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 4,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 34.0,
            "param_overrides.SLOPE_BYPASS_SIZE_MULT": 0.75,
        }),
        ("legacy_disable_hourly_eqs3", {
            "flags.hourly_alignment": False,
            "flags.entry_quality_gate": True,
            "param_overrides.EQS_MIN_RTH": 3,
        }),
        ("legacy_disable_slope_eqs3", {
            "flags.slope_gate": False,
            "flags.entry_quality_gate": True,
            "param_overrides.EQS_MIN_RTH": 3,
        }),
    ]


def _phase_2_no_signal_continuation(prior: dict) -> list[tuple[str, dict]]:
    """Convert no_signal shadow alpha into explicit Type C continuation entries."""
    base = {
        "flags.type_c_enabled": True,
        "param_overrides.USE_TYPE_C": True,
    }
    return [
        ("type_c_default", dict(base)),
        ("type_c_open_close", {
            **base,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
        }),
        ("type_c_close_only", {
            **base,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["CLOSE"],
        }),
        ("type_c_open_only", {
            **base,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["OPEN"],
        }),
        ("type_c_core_only_strict", {
            **base,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["CORE"],
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.72,
            "param_overrides.TYPE_C_MAX_VWAP_DIST_ATR": 1.20,
        }),
        ("type_c_lb8_fast", {
            **base,
            "param_overrides.TYPE_C_LOOKBACK_15M": 8,
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.60,
        }),
        ("type_c_lb16_swing", {
            **base,
            "param_overrides.TYPE_C_LOOKBACK_15M": 16,
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.62,
        }),
        ("type_c_lb20_major", {
            **base,
            "param_overrides.TYPE_C_LOOKBACK_15M": 20,
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.65,
        }),
        ("type_c_strong_close", {
            **base,
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.72,
            "param_overrides.TYPE_C_MAX_BAR_ATR": 1.30,
        }),
        ("type_c_looser_close", {
            **base,
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.55,
            "param_overrides.TYPE_C_MAX_BAR_ATR": 1.80,
        }),
        ("type_c_vwap_tight", {
            **base,
            "param_overrides.TYPE_C_MAX_VWAP_DIST_ATR": 1.10,
        }),
        ("type_c_vwap_wide", {
            **base,
            "param_overrides.TYPE_C_MAX_VWAP_DIST_ATR": 2.40,
        }),
        ("type_c_no_vwap_side", {
            **base,
            "param_overrides.TYPE_C_REQUIRE_VWAP_SIDE": False,
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.70,
            "param_overrides.TYPE_C_MAX_BAR_ATR": 1.25,
        }),
        ("type_c_break_buffer", {
            **base,
            "param_overrides.TYPE_C_BREAK_BUFFER_ATR": 0.05,
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.60,
        }),
        ("type_c_with_slope_bypass", {
            **base,
            "flags.slope_bypass_quality": True,
            "param_overrides.SLOPE_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 3,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 35.0,
            "param_overrides.SLOPE_BYPASS_SIZE_MULT": 0.75,
        }),
        ("type_c_with_hourly_bypass", {
            **base,
            "flags.hourly_bypass_quality": True,
            "param_overrides.HOURLY_BYPASS_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 3,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 35.0,
            "param_overrides.HOURLY_BYPASS_SIZE_MULT": 0.60,
        }),
        ("type_b_c_combo", {
            **base,
            "param_overrides.USE_TYPE_B": True,
            "param_overrides.TYPE_B_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
        }),
    ]


def _phase_3_entry_frequency(prior: dict) -> list[tuple[str, dict]]:
    """Entry mechanics and controlled frequency expansion after new signals."""
    return [
        ("ttl_5", {"param_overrides.TTL_BARS": 5}),
        ("ttl_6", {"param_overrides.TTL_BARS": 6}),
        ("buffer_0", {"param_overrides.BUFFER_TICKS": 0}),
        ("buffer_2", {"param_overrides.BUFFER_TICKS": 2}),
        ("offset_atr_010", {"param_overrides.OFFSET_TICKS_ATR_FRAC": 0.10}),
        ("offset_atr_020", {"param_overrides.OFFSET_TICKS_ATR_FRAC": 0.20}),
        ("fallback_immediate", {"param_overrides.FALLBACK_WAIT_BARS": 0}),
        ("fallback_wait_2", {"param_overrides.FALLBACK_WAIT_BARS": 2}),
        ("micro_trigger_default", {"param_overrides.USE_MICRO_TRIGGER": True}),
        ("micro_trigger_window_2", {
            "param_overrides.USE_MICRO_TRIGGER": True,
            "param_overrides.MICRO_WINDOW_BARS": 2,
        }),
        ("touch_lb_12", {"param_overrides.TOUCH_LOOKBACK_15M": 12}),
        ("touch_lb_16", {"param_overrides.TOUCH_LOOKBACK_15M": 16}),
        ("vwap_cap_core_100", {"param_overrides.VWAP_CAP_CORE": 1.00}),
        ("vwap_cap_open_085", {"param_overrides.VWAP_CAP_OPEN_EVE": 0.85}),
        ("max_longs_3", {"param_overrides.MAX_LONGS_PER_DAY": 3}),
        ("max_shorts_3", {"param_overrides.MAX_SHORTS_PER_DAY": 3}),
        ("max_shorts_4", {"param_overrides.MAX_SHORTS_PER_DAY": 4}),
        ("caps_3x3", {
            "param_overrides.MAX_LONGS_PER_DAY": 3,
            "param_overrides.MAX_SHORTS_PER_DAY": 3,
        }),
        ("chop_caps_2", {
            "param_overrides.CHOP_MAX_LONGS": 2,
            "param_overrides.CHOP_MAX_SHORTS": 2,
        }),
        ("entry_recovery_combo", {
            "param_overrides.TTL_BARS": 5,
            "param_overrides.OFFSET_TICKS_ATR_FRAC": 0.20,
            "param_overrides.FALLBACK_WAIT_BARS": 0,
        }),
    ]


def _phase_4_exit_capture(prior: dict) -> list[tuple[str, dict]]:
    """Capture more MFE without clipping the 17+ bar right tail."""
    return [
        ("mfe_rescue_be_050_5", {
            "flags.mfe_rescue_stop": True,
            "param_overrides.MFE_RESCUE_MIN_R": 0.50,
            "param_overrides.MFE_RESCUE_AFTER_BARS": 5,
            "param_overrides.MFE_RESCUE_TRIGGER_R": 0.05,
            "param_overrides.MFE_RESCUE_LOCK_R": 0.00,
        }),
        ("mfe_rescue_lock010", {
            "flags.mfe_rescue_stop": True,
            "param_overrides.MFE_RESCUE_MIN_R": 0.50,
            "param_overrides.MFE_RESCUE_AFTER_BARS": 5,
            "param_overrides.MFE_RESCUE_TRIGGER_R": 0.10,
            "param_overrides.MFE_RESCUE_LOCK_R": 0.10,
        }),
        ("mfe_rescue_late", {
            "flags.mfe_rescue_stop": True,
            "param_overrides.MFE_RESCUE_MIN_R": 0.75,
            "param_overrides.MFE_RESCUE_AFTER_BARS": 7,
            "param_overrides.MFE_RESCUE_TRIGGER_R": 0.00,
            "param_overrides.MFE_RESCUE_LOCK_R": 0.05,
        }),
        ("mfe_rescue_early_small", {
            "flags.mfe_rescue_stop": True,
            "param_overrides.MFE_RESCUE_MIN_R": 0.35,
            "param_overrides.MFE_RESCUE_AFTER_BARS": 4,
            "param_overrides.MFE_RESCUE_TRIGGER_R": -0.05,
            "param_overrides.MFE_RESCUE_LOCK_R": 0.00,
        }),
        ("stale_mfe_exempt_035", {
            "flags.stale_mfe_exempt": True,
            "param_overrides.STALE_MFE_EXEMPT_R": 0.35,
        }),
        ("stale_mfe_exempt_050", {
            "flags.stale_mfe_exempt": True,
            "param_overrides.STALE_MFE_EXEMPT_R": 0.50,
        }),
        ("stale_bars_10", {"param_overrides.STALE_BARS_15M": 10}),
        ("stale_bars_12", {"param_overrides.STALE_BARS_15M": 12}),
        ("vwap_fail_3", {"param_overrides.VWAP_FAIL_CONSEC": 3}),
        ("late_trail_fast_capture", {
            "flags.late_trail": True,
            "param_overrides.LATE_TRAIL_ACTIVATE_R": 1.25,
            "param_overrides.LATE_TRAIL_MULT": 3.50,
            "param_overrides.LATE_TRAIL_MULT_MIN": 2.00,
        }),
        ("late_trail_wide_runner", {
            "flags.late_trail": True,
            "param_overrides.LATE_TRAIL_BE_R": 1.25,
            "param_overrides.LATE_TRAIL_ACTIVATE_R": 2.00,
            "param_overrides.LATE_TRAIL_MULT": 4.50,
            "param_overrides.LATE_TRAIL_MULT_MIN": 2.50,
        }),
        ("max_duration_64", {"param_overrides.MAX_POSITION_BARS_15M": 64}),
        ("max_duration_80", {"param_overrides.MAX_POSITION_BARS_15M": 80}),
        ("rescue_stale_combo", {
            "flags.mfe_rescue_stop": True,
            "flags.stale_mfe_exempt": True,
            "param_overrides.MFE_RESCUE_MIN_R": 0.50,
            "param_overrides.MFE_RESCUE_AFTER_BARS": 5,
            "param_overrides.STALE_MFE_EXEMPT_R": 0.50,
        }),
    ]


def _phase_5_session_cohorts(prior: dict) -> list[tuple[str, dict]]:
    """Session/day cohort controls after structural routes are known."""
    return [
        ("dow_disable", {"flags.dow_sizing": False}),
        ("dow_monday_only_cut", {
            "flags.dow_sizing": True,
            "param_overrides.DOW_SIZE_MULT": {0: 0.60},
        }),
        ("dow_thursday_only_cut", {
            "flags.dow_sizing": True,
            "param_overrides.DOW_SIZE_MULT": {3: 0.70},
        }),
        ("dow_monday_thu_harder", {
            "flags.dow_sizing": True,
            "param_overrides.DOW_SIZE_MULT": {0: 0.50, 3: 0.65},
        }),
        ("dow_monday_thu_softer", {
            "flags.dow_sizing": True,
            "param_overrides.DOW_SIZE_MULT": {0: 0.75, 3: 0.85},
        }),
        ("dow_tue_fri_boost", {
            "flags.dow_sizing": True,
            "param_overrides.DOW_SIZE_MULT": {0: 0.60, 1: 1.15, 3: 0.75, 4: 1.10},
        }),
        ("evening_zero_cap", {
            "flags.evening_vwap_cap": True,
            "param_overrides.VWAP_CAP_EVENING": 0.00,
        }),
        ("evening_cap_035", {
            "flags.evening_vwap_cap": True,
            "param_overrides.VWAP_CAP_EVENING": 0.35,
        }),
        ("allow_20h_quality4", {
            "flags.block_20h_hour": False,
            "flags.entry_quality_gate": True,
            "param_overrides.EQS_MIN_EVENING": 4,
            "param_overrides.VWAP_CAP_EVENING": 0.05,
        }),
        ("close_vwap_cap_060", {
            "param_overrides.VWAP_CAP_CLOSE": 0.60,
            "param_overrides.VWAP_CAP_CLOSE_STRICT": 0.45,
        }),
        ("close_mfe_floor_looser", {
            "param_overrides.CLOSE_MFE_RATCHET_TIERS": [[1.00, 0.60], [1.50, 0.95]],
        }),
        ("close_mfe_floor_tighter", {
            "param_overrides.CLOSE_MFE_RATCHET_TIERS": [[0.80, 0.55], [1.20, 0.95]],
        }),
        ("chop_28", {"param_overrides.CHOP_THRESHOLD": 28}),
        ("chop_36", {"param_overrides.CHOP_THRESHOLD": 36}),
        ("chop_44", {"param_overrides.CHOP_THRESHOLD": 44}),
    ]


def _phase_6_interactions(prior: dict) -> list[tuple[str, dict]]:
    """Curated structural interactions plus numeric fine-tuning."""
    c: list[tuple[str, dict]] = [
        ("combo_gate_typec_strict", {
            "flags.hourly_bypass_quality": True,
            "flags.slope_bypass_quality": True,
            "flags.type_c_enabled": True,
            "param_overrides.USE_TYPE_C": True,
            "param_overrides.HOURLY_BYPASS_EQS_MIN": 4,
            "param_overrides.HOURLY_BYPASS_MAX_CHOP": 34.0,
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 4,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 34.0,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.TYPE_C_MIN_CLOSE_FRAC": 0.70,
        }),
        ("combo_typec_rescue", {
            "flags.type_c_enabled": True,
            "flags.mfe_rescue_stop": True,
            "param_overrides.USE_TYPE_C": True,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.MFE_RESCUE_MIN_R": 0.50,
            "param_overrides.MFE_RESCUE_AFTER_BARS": 5,
        }),
        ("combo_slope_rescue", {
            "flags.slope_bypass_quality": True,
            "flags.mfe_rescue_stop": True,
            "param_overrides.SLOPE_BYPASS_EQS_MIN": 3,
            "param_overrides.SLOPE_BYPASS_MAX_CHOP": 38.0,
            "param_overrides.MFE_RESCUE_MIN_R": 0.50,
            "param_overrides.MFE_RESCUE_AFTER_BARS": 5,
        }),
        ("combo_frequency_guarded", {
            "flags.type_c_enabled": True,
            "param_overrides.USE_TYPE_C": True,
            "param_overrides.TTL_BARS": 5,
            "param_overrides.MAX_LONGS_PER_DAY": 3,
            "param_overrides.MAX_SHORTS_PER_DAY": 3,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
        }),
        ("combo_runner_guarded_expansion", {
            "flags.type_c_enabled": True,
            "flags.stale_mfe_exempt": True,
            "flags.late_trail": True,
            "param_overrides.USE_TYPE_C": True,
            "param_overrides.TYPE_C_ALLOWED_WINDOWS": ["OPEN", "CLOSE"],
            "param_overrides.STALE_MFE_EXEMPT_R": 0.50,
            "param_overrides.LATE_TRAIL_ACTIVATE_R": 1.50,
        }),
    ]

    numeric_keys = []
    for key, val in prior.items():
        if key.startswith("param_overrides.") and isinstance(val, (int, float)):
            if key.endswith("BASE_RISK_PCT"):
                continue
            numeric_keys.append((key, val))

    for key, val in numeric_keys:
        short = key.split(".")[-1].lower()
        for mult, label in [(0.90, "m10"), (1.05, "p05"), (1.10, "p10"), (1.20, "p20")]:
            new_val = round(val * mult, 6)
            if new_val != val:
                c.append((f"finetune_{short}_{label}", {key: new_val}))

    return c
