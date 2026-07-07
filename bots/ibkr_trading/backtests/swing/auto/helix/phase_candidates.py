"""Helix per-phase candidate selection from experiment categories.

Phase 1: CLASS_PRUNING + ENTRY_QUALITY + CLASS_D_QUALITY + P1 structural
Phase 2: LEAKAGE_GUARD + STALE_BAIL + PARTIALS_BE
Phase 3: STOP_PLACEMENT + TRAILING + INLINE_TRAILING + CLASS_SPECIFIC_TRAILING
Phase 4: VOLATILITY + ADDON + ADDON_EXPANSION + P3 structural
Phase 5: targeted RTS / exit-sensitive finetune
Phase 6: CIRCUIT_BREAKER + remaining numeric finetune
Phase 7: OOS repair probes for Class D frequency with entry discrimination
"""
from __future__ import annotations

from .experiment_categories import (
    _P1_STRUCTURAL,
    _P3_STRUCTURAL,
    get_category_experiments,
)


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates()
    elif phase == 4:
        candidates = _phase_4_candidates()
    elif phase == 5:
        candidates = _phase_5_candidates(prior_mutations or {})
    elif phase == 6:
        candidates = _phase_6_candidates(prior_mutations or {})
    elif phase == 7:
        candidates = _phase_7_candidates(prior_mutations or {})
    else:
        candidates = []

    if suggested_experiments:
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """Signal pruning + entry gates + P1 structural ablation."""
    pruning = get_category_experiments(["CLASS_PRUNING"])
    entry = get_category_experiments([
        "ENTRY_QUALITY",
        "CLASS_D_QUALITY",
        "CLASS_D_ENTRY_DISCRIMINATOR",
    ])
    structural = [
        (n, m) for n, m in get_category_experiments(["STRUCTURAL_ABLATION"])
        if n in _P1_STRUCTURAL
    ]
    return pruning + entry + structural


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """Fast leakage repair: RTS guard, stale/bail, and partial/BE controls."""
    return get_category_experiments([
        "LEAKAGE_GUARD", "STALE_BAIL", "PARTIALS_BE",
    ])


def _phase_3_candidates() -> list[tuple[str, dict]]:
    """Payoff repair: stop placement and trailing geometry."""
    return get_category_experiments([
        "STOP_PLACEMENT", "TRAILING", "INLINE_TRAILING",
        "CLASS_SPECIFIC_TRAILING",
    ])


def _phase_4_candidates() -> list[tuple[str, dict]]:
    """Volatility + addon + P3 structural ablation."""
    vol_addon = get_category_experiments(["VOLATILITY", "ADDON", "ADDON_EXPANSION"])
    structural = [
        (n, m) for n, m in get_category_experiments(["STRUCTURAL_ABLATION"])
        if n in _P3_STRUCTURAL
    ]
    return vol_addon + structural


def _phase_5_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Targeted exit-sensitive fine-tuning before broader cleanup."""
    return _targeted_rts_candidates() + _finetune_candidates(
        prior_mutations,
        include_key=_is_exit_sensitive_key,
    )


def _phase_6_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Circuit breakers plus the remaining numeric fine-tune surface."""
    circuit = get_category_experiments(["CIRCUIT_BREAKER"])
    return circuit + _finetune_candidates(
        prior_mutations,
        include_key=lambda key: not _is_exit_sensitive_key(key),
    )


def _phase_7_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """OOS frequency repair without dropping the Class D short guard blindly."""
    return [
        ("oos_d_short_adx_12", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 12.0}),
        ("oos_d_short_adx_16", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0}),
        ("oos_d_short_adx_18", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 18.0}),
        ("oos_d_short_adx_20", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0}),
        ("oos_d_short_adx_22", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 22.0}),
        ("oos_d_streak_1", {"param_overrides.CLASS_D_REGIME_STREAK_MIN": 1}),
        ("oos_d_short20_streak1", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_REGIME_STREAK_MIN": 1,
        }),
        ("oos_d_short16_streak1", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_REGIME_STREAK_MIN": 1,
        }),
        ("oos_d_short20_sep4", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
        }),
        ("oos_d_short16_sep4", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
        }),
        ("oos_d_short20_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short16_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short20_sep4_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short16_sep4_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short12_sep4_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 12.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short20_age24", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 24,
        }),
        ("oos_d_short16_age24", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 24,
        }),
        ("oos_d_short20_sep4_age24", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 24,
        }),
        ("oos_d_short16_sep4_age24", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 24,
        }),
        ("oos_d_short20_hist", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_HIST_SIGN_GATE": True,
        }),
        ("oos_d_short16_hist", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_HIST_SIGN_GATE": True,
        }),
        ("oos_d_short20_hist_sep4", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_HIST_SIGN_GATE": True,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
        }),
        ("oos_d_short16_hist_sep4", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_HIST_SIGN_GATE": True,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
        }),
        ("oos_d_short20_fresh010", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_FRESH_BREAK_ATR": 0.10,
        }),
        ("oos_d_short16_fresh010", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_FRESH_BREAK_ATR": 0.10,
        }),
        ("oos_d_short20_pullback_min025", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MIN_PULLBACK_ATR": 0.25,
        }),
        ("oos_d_short16_pullback_min025", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_MIN_PULLBACK_ATR": 0.25,
        }),
        ("oos_d_short20_arm_overext100", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MAX_ARM_OVEREXT_ATR": 1.00,
        }),
        ("oos_d_short16_arm_overext100", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 16.0,
            "param_overrides.CLASS_D_MAX_ARM_OVEREXT_ATR": 1.00,
        }),
        ("oos_d_streak1_sep4_dailyext300", {
            "param_overrides.CLASS_D_REGIME_STREAK_MIN": 1,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short0_sep4_age20_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 20,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short0_sep4_age24_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_PIVOT2_AGE_BARS": 24,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short0_sep6_dailyext300", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 6,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
        ("oos_d_short0_sep4_dailyext350", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.50,
        }),
        ("oos_d_short0_sep4_dailyext300_hist", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.CLASS_D_HIST_SIGN_GATE": True,
        }),
        ("oos_d_short0_sep4_dailyext300_fresh010", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.CLASS_D_FRESH_BREAK_ATR": 0.10,
        }),
        ("oos_d_short0_sep4_dailyext300_pullback025", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.CLASS_D_MIN_PULLBACK_ATR": 0.25,
        }),
        ("oos_d_short0_sep4_dailyext300_arm100", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.CLASS_D_MAX_ARM_OVEREXT_ATR": 1.00,
        }),
        ("oos_d_short0_sep4_dailyext300_baild6", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.CLASS_D_BAIL_BARS": 6,
            "param_overrides.CLASS_D_BAIL_R_THRESH": -0.30,
        }),
        ("oos_d_short0_sep4_dailyext300_baild8", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.CLASS_D_BAIL_BARS": 8,
            "param_overrides.CLASS_D_BAIL_R_THRESH": -0.30,
        }),
        ("oos_d_short0_sep4_dailyext300_highvol69", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.HIGH_VOL_PCT": 69,
        }),
        ("oos_d_short0_sep4_dailyext300_trailstall3", {
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 0.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "param_overrides.CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "param_overrides.TRAIL_STALL_ONSET": 3,
        }),
        ("oos_adx_upper_65", {"param_overrides.ADX_UPPER_GATE": 65.0}),
        ("oos_adx_upper_70", {"param_overrides.ADX_UPPER_GATE": 70.0}),
        ("oos_adx_upper_off", {"param_overrides.ADX_UPPER_GATE": 999.0}),
        ("oos_high_vol_62", {"param_overrides.HIGH_VOL_PCT": 62}),
        ("oos_high_vol_65", {"param_overrides.HIGH_VOL_PCT": 65}),
        ("oos_high_vol_70", {"param_overrides.HIGH_VOL_PCT": 70}),
        ("oos_high_vol_72", {"param_overrides.HIGH_VOL_PCT": 72}),
        ("oos_reenable_class_c", {"flags.disable_class_c": False}),
        ("oos_reenable_class_a", {"flags.disable_class_a": False}),
        ("oos_reenable_class_c_short20_sep4", {
            "flags.disable_class_c": False,
            "param_overrides.CLASS_D_SHORT_MIN_ADX": 20.0,
            "param_overrides.CLASS_D_MIN_PIVOT_SEP_BARS": 4,
        }),
    ]


def _targeted_rts_candidates() -> list[tuple[str, dict]]:
    """Smoke-tested RTS arming delays that preserved win-rate gains and more tail."""
    targeted = [
        ("rts_guard_minbars_8", {"param_overrides.RTS_GUARD_MIN_BARS": 8}),
        ("rts_guard_minbars_10", {"param_overrides.RTS_GUARD_MIN_BARS": 10}),
        ("rts_guard_minbars_12", {"param_overrides.RTS_GUARD_MIN_BARS": 12}),
    ]
    return targeted


def _finetune_candidates(
    prior_mutations: dict,
    *,
    include_key,
) -> list[tuple[str, dict]]:
    if not prior_mutations:
        return []

    finetune = []
    for key, value in prior_mutations.items():
        if not include_key(key):
            continue
        # bool must be checked first: isinstance(True, int) is True in Python
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue

        for pct_label, pct in [("m20", 0.80), ("m10", 0.90), ("p10", 1.10), ("p20", 1.20)]:
            new_val = value * pct
            if isinstance(value, int):
                new_val = int(round(new_val))
                if new_val == value:
                    continue
            else:
                new_val = round(new_val, 6)
                if abs(new_val - value) < 1e-12:
                    continue

            name = f"finetune_{key}_{pct_label}"
            finetune.append((name, {key: new_val}))

    return finetune


_EXIT_SENSITIVE_KEY_TOKENS = (
    "RTS_GUARD",
    "RTS_FAIL",
    "TRAIL",
    "STALE",
    "BAIL",
    "STOP",
    "R_BE",
    "R_PARTIAL",
    "PARTIAL",
    "CLASS_D",
    "PIVOT",
    "PULLBACK",
    "MACD",
    "HIST",
    "ENTRY_STOP",
    "DAILY_EXTENSION",
    "FRESH_BREAK",
)


def _is_exit_sensitive_key(key: str) -> bool:
    upper_key = key.upper()
    return any(token in upper_key for token in _EXIT_SENSITIVE_KEY_TOKENS)
