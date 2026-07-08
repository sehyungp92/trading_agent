"""Version-controlled research presets for the regime pipeline."""

from __future__ import annotations

import copy

R3_REFERENCE_MUTATIONS: dict[str, object] = {
    "sticky_diag": 15.0,
    "use_warm_start": False,
    "use_expanding_window": False,
    "rolling_window_years": 7,
    "refit_ll_tolerance": 5.0,
    "refit_freq": "QE",
    "z_minp": 30,
    "posterior_ema_alpha": 0.8,
    "per_strat_max": 0.5,
    "base_target_vol_annual": 0.08,
    "kappa_totalvol_cap": 1.5,
    "crisis_logit_a": 3.0,
    "stability_weight": 0.8,
    "weight_smoothing_alpha": 0.5,
    "L_max": 1.6,
    "delta_rho_exempt": 0.0,
    "sigma_floor_annual": 0.09,
    "delta_rho_threshold": 0.15,
    # 3-element: legacy crisis path without cross-asset corr trigger.
    # To test calibration on the full architecture, use recommended_full_stack.
    "crisis_weights": (0.3, 0.6, 0.1),
    "scanner_enabled": False,
    "n_ensemble_models": 1,
    "posterior_temperature": 1.0,
    "posterior_smoothing_eps": 0.0,
    "stress_model_enabled": False,
}

RECOMMENDED_FULL_STACK_MUTATIONS: dict[str, object] = {
    **R3_REFERENCE_MUTATIONS,
    "scanner_enabled": True,
    "n_ensemble_models": 5,
    "crisis_z_window": 21,
    "crisis_weights": (0.3, 0.6, 0.1, 0.1),
    "posterior_ema_risk_off_alpha": 1.0,
    "posterior_temperature": 1.0,
    "posterior_smoothing_eps": 0.0,
}

STEP9_R6_MUTATIONS: dict[str, object] = {
    **R3_REFERENCE_MUTATIONS,
    "scanner_enabled": True,
    "n_ensemble_models": 5,
    "crisis_z_window": 21,
    "posterior_temperature": 1.5,
    "posterior_smoothing_eps": 0.01,
    "posterior_ema_alpha": 0.7,
    "posterior_ema_risk_off_alpha": 1.0,
    "crisis_weights": (0.25, 0.50, 0.10, 0.15),
}

R7_OVERLAY_RECAL_MUTATIONS: dict[str, object] = {
    **STEP9_R6_MUTATIONS,
    # Track A: conjunction gating
    "conjunction_gating_enabled": True,
    "conjunction_active_threshold": 0.95,
    "conjunction_1_layer_max_reduction": 0.15,
    "conjunction_min_multiplier": 0.40,
    # Track A: tighter thresholds
    "crisis_leverage_threshold_low": 0.50,
    "crisis_leverage_threshold_high": 0.75,
    "scanner_threshold": 0.80,
    # Track B: crisis override
    "crisis_override_enabled": True,
    "crisis_override_threshold": 0.70,
    "scanner_override_threshold": 0.80,
}

R8_TWO_MODEL_MUTATIONS: dict[str, object] = {
    **R3_REFERENCE_MUTATIONS,
    # Pure R3 macro HMM (single model, no scanner overlay, no ensemble).
    # Stress model computes its own features via build_stress_features() --
    # scanner_enabled=False is fine, the stress pipeline calls build_scanner_features() directly.
    # n_ensemble_models and scanner_enabled are optimizer candidates (Phase 4), not forced defaults.
    # R8: stress model replaces all overlay leverage adjustments
    "stress_model_enabled": True,
    "stress_sticky_diag": 50.0,
    "stress_sticky_offdiag": 2.0,
    "stress_stressed_sticky": 10.0,
    "stress_reduction_max": 0.15,
    "stress_blend_threshold": 1.01,  # disabled: macro HMM handles allocation, stress handles sizing
    "stress_onset_threshold": 0.50,
    "stress_n_iter": 200,
    "stress_n_iter_first_fit": 400,
    "stress_refit_guard_tol": 2.0,
}

R9_BUDGET_MUTATIONS: dict[str, object] = {
    **R8_TWO_MODEL_MUTATIONS,
    # R8 greedy-accepted mutations (Phase 4)
    "posterior_ema_alpha": 0.6,
    "crisis_weights": (0.2, 0.4, 0.2, 0.2),
    "stress_reduction_max": 0.05,
    # All 25 budget fields at MetaConfig defaults (explicit baseline)
    "budget_G_spy": 0.40, "budget_G_efa": 0.10,
    "budget_G_tlt": 0.05, "budget_G_gld": 0.05, "budget_G_cash": 0.40,
    "budget_R_spy": 0.35, "budget_R_efa": 0.15, "budget_R_gld": 0.30,
    "budget_R_tlt": 0.00, "budget_R_cash": 0.20,
    "budget_S_gld": 0.50, "budget_S_cash": 0.30, "budget_S_spy": 0.10,
    "budget_S_efa": 0.05, "budget_S_tlt": 0.05,
    "budget_D_tlt": 0.50, "budget_D_cash": 0.30, "budget_D_gld": 0.10,
    "budget_D_spy": 0.05, "budget_D_efa": 0.05,
    "budget_neutral_spy": 0.20, "budget_neutral_efa": 0.10,
    "budget_neutral_tlt": 0.25, "budget_neutral_gld": 0.25,
    "budget_neutral_cash": 0.20,
}

PRESET_MUTATIONS: dict[str, dict[str, object]] = {
    "r3_reference": R3_REFERENCE_MUTATIONS,
    "recommended_full_stack": RECOMMENDED_FULL_STACK_MUTATIONS,
    "step9_r6": STEP9_R6_MUTATIONS,
    "r7_overlay_recal": R7_OVERLAY_RECAL_MUTATIONS,
    "r8_two_model": R8_TWO_MODEL_MUTATIONS,
    "r9_budget": R9_BUDGET_MUTATIONS,
}

DEFAULT_RESEARCH_PRESET = "recommended_full_stack"
DEFAULT_REFERENCE_PRESET = "r3_reference"
STEP9_RESEARCH_PRESET = "step9_r6"
R7_RESEARCH_PRESET = "r7_overlay_recal"
R8_RESEARCH_PRESET = "r8_two_model"
R9_RESEARCH_PRESET = "r9_budget"


def preset_choices() -> tuple[str, ...]:
    """Return the supported preset names for argparse choices/help text."""
    return tuple(PRESET_MUTATIONS)


def get_research_preset(name: str) -> dict[str, object]:
    """Return a defensive copy of a named preset."""
    if name not in PRESET_MUTATIONS:
        valid = ", ".join(sorted(PRESET_MUTATIONS))
        raise ValueError(f"Unknown preset {name!r}. Valid presets: {valid}")
    return copy.deepcopy(PRESET_MUTATIONS[name])
