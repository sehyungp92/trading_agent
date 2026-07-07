"""Configuration for the MR-AWQ Meta Allocator v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

REGIMES = ["G", "R", "S", "D"]  # Recovery, Reflation, Infl Hedge, Defensive

REGIME_TARGETS = {
    "G": np.array([+1.0, -1.0]),  # growth up, inflation down
    "R": np.array([+1.0, +1.0]),  # growth up, inflation up
    "S": np.array([-1.0, +1.0]),  # growth down, inflation up
    "D": np.array([-1.0, -1.0]),  # growth down, inflation down
}


@dataclass
class MetaConfig:
    # -- Rebalance --
    rebalance_freq: str = "W-FRI"
    cash_col: str = "CASH"
    ann_factor: float = 252.0

    # -- Macro standardisation --
    z_window: int = 252
    z_minp: int = 30

    # -- HMM --
    n_states: int = 4
    covariance_type: str = "full"
    n_iter_first_fit: int = 400
    n_iter_refit: int = 200
    tol: float = 1e-3
    min_covar: float = 1e-6
    random_state: int = 7
    n_ensemble_models: int = 1       # 1 = no ensemble (backward-compatible)
    use_forward_only: bool = False   # False = predict_proba (backward-compatible)
    sticky_diag: float = 15.0
    sticky_offdiag: float = 2.0

    # -- Refit --
    refit_freq: str = "QE"
    use_expanding_window: bool = False
    use_warm_start: bool = False
    refit_validation_window: int = 63
    refit_ll_tolerance: float = 5.0

    # -- Crisis overlay --
    crisis_weights: Tuple[float, ...] = (0.2, 0.4, 0.2, 0.2)
    crisis_logit_a: float = 3.0
    crisis_logit_b: float = 0.0

    # -- Confidence (entropy x stability) --
    conf_floor: float = 0.3
    stability_weight: float = 0.8

    # -- Risk budgeting (correlation-adjusted) --
    sigma_floor_annual: float = 0.09
    per_strat_max: float = 0.50
    strat_vol_span: int = 63
    cov_window: int = 63
    shrinkage_target: str = "ledoit_wolf"

    # -- Ventilator (delta-rho anticipatory) --
    ventilator_lambda: float = 0.8
    ventilator_vmin: float = 0.2
    rho_short_window: int = 20
    rho_long_window: int = 60
    delta_rho_threshold: float = 0.15
    delta_rho_exempt: float = 0.0
    pnl_confirm_days: int = 10
    risk_on_set: Tuple[str, ...] = ("SPY", "EFA")

    # -- Phase B leverage (fixed sigma-star, no confidence modulation) --
    L_max: float = 1.6
    kappa_totalvol_cap: float = 1.5
    base_target_vol_annual: float = 0.08
    ewma_downside_span: int = 20
    ewma_total_span: int = 60
    s_floor: float = 0.3
    gamma: float = 0.1
    dd_ladder: Tuple[Tuple[float, float], ...] = (
        (-0.08, 1.0),
        (-0.12, 0.7),
        (-0.16, 0.5),
        (-0.20, 0.3),
    )

    # -- Autoresearch phase additions --
    rolling_window_years: int = 7
    warm_start_perturb_std: float = 0.0
    use_commodity_feature: bool = False
    use_real_rates_feature: bool = False
    drop_momentum_breadth: bool = False
    drop_eq_bond_corr: bool = False

    # -- Posterior calibration (Imp 1) --
    posterior_temperature: float = 1.0       # T>1 softens, T<1 sharpens
    posterior_smoothing_eps: float = 0.0     # Dirichlet smoothing to eliminate exact zeros
    posterior_ema_alpha: float = 0.6         # R9: temporal smoothing (overrides R3's 0.8)

    # -- Label anchoring (Imp 2) --
    label_continuity_weight: float = 0.0     # L2 continuity cost in alignment

    # -- Regime momentum (Imp 3) --
    regime_momentum_lookback: int = 4        # weeks of posterior history for momentum

    # -- New features (Imp 4) --
    use_vix_feature: bool = False
    use_realized_vol_feature: bool = False
    use_trend_divergence_feature: bool = False

    # -- Weight smoothing (Imp 5) --
    weight_smoothing_alpha: float = 0.5      # R3: weight smoothing enabled

    # -- Crisis detection (Imp 6) --
    crisis_z_window: int = 21
    crisis_leverage_enabled: bool = True
    crisis_leverage_threshold_low: float = 0.35
    crisis_leverage_threshold_high: float = 0.65
    crisis_leverage_reduction_mid: float = 0.25     # reduction at threshold_high
    crisis_leverage_reduction_max: float = 0.50     # reduction at p_crisis=1.0
    posterior_ema_risk_off_alpha: float = 1.0     # EMA alpha when shifting toward S/D (1.0 = no smoothing)

    # -- Leading Indicator Scanner (Layer 2) --
    scanner_enabled: bool = False                   # opt-in, backward-compatible
    scanner_z_window: int = 252                     # z-score normalization window
    scanner_z_minp: int = 60                        # minimum periods for z-score
    scanner_steepness: float = 3.0                  # logistic steepness
    scanner_threshold: float = 0.6                  # shift_prob trigger for leverage reduction
    scanner_max_reduction: float = 0.30             # max 30% leverage reduction at prob=1.0
    scanner_feature_weights: Tuple[Tuple[str, float], ...] = (
        ("credit_spread_mom", 0.25),        # plan: "most reliable leading indicator"
        ("yield_curve_vel", 0.15),
        ("cross_asset_corr", 0.25),         # plan: "most important"
        ("breadth_deterioration", 0.15),
        ("realized_vol_ratio", 0.15),
        ("vix_momentum", 0.05),             # proxy for VIX term structure (VIX3M unavailable)
    )

    # -- Ensemble disagreement monitor (Layer 4) --
    disagreement_confidence_enabled: bool = True
    disagreement_leverage_enabled: bool = True
    disagreement_moderate_consensus: float = 0.90
    disagreement_low_consensus: float = 0.70
    disagreement_moderate_reduction: float = 0.20
    disagreement_high_reduction: float = 0.30
    disagreement_max_reduction: float = 0.35
    disagreement_trend_threshold: float = -0.15
    disagreement_trend_extra_reduction: float = 0.05
    disagreement_risk_off_extra_reduction: float = 0.05

    # -- Conjunction gating (R7 Track A) --
    conjunction_gating_enabled: bool = False        # opt-in, backward-compatible
    conjunction_active_threshold: float = 0.95      # adj < this = "active"
    conjunction_1_layer_max_reduction: float = 0.15 # max reduction when only 1 layer active
    conjunction_min_multiplier: float = 0.40        # floor when 2+ layers active

    # -- Crisis override (R7 Track B) --
    crisis_override_enabled: bool = False           # opt-in, backward-compatible
    crisis_override_threshold: float = 0.70         # p_crisis must exceed this
    scanner_override_threshold: float = 0.80        # shift_prob must exceed this

    # -- Stress HMM (R8 Two-Model) --
    stress_model_enabled: bool = True           # R8: two-model architecture
    stress_n_states: int = 2                    # always 2 for stress/normal
    stress_covariance_type: str = "full"
    stress_n_iter: int = 200
    stress_n_iter_first_fit: int = 400
    stress_sticky_diag: float = 50.0             # normal-state self-transition pseudocount (high = stays normal)
    stress_sticky_offdiag: float = 2.0          # off-diagonal prior pseudocount
    stress_stressed_sticky: float = 10.0        # stress-state self-transition pseudocount (lower = stress is transient)
    stress_reduction_max: float = 0.05          # R9: max leverage reduction at P(stress)=1.0
    stress_blend_threshold: float = 1.01        # R8: disabled (macro HMM handles allocation, stress handles sizing)
    stress_velocity_lookback: int = 4           # weeks for stress velocity calc
    stress_onset_threshold: float = 0.50        # P(stress) threshold for stress_onset flag
    stress_refit_guard_tol: float = 2.0         # OOS log-likelihood tolerance for refit guard

    # -- Regime budget overrides (R8) --
    # Defaults mirror the hardcoded values in portfolio.py:default_regime_budgets().
    # Recovery (G): risk-on tilt
    budget_G_spy: float = 0.40
    budget_G_efa: float = 0.10
    budget_G_tlt: float = 0.05
    budget_G_gld: float = 0.05
    budget_G_cash: float = 0.40
    # Reflation (R): commodity/equity tilt
    budget_R_spy: float = 0.35
    budget_R_efa: float = 0.15
    budget_R_gld: float = 0.30
    budget_R_tlt: float = 0.00
    budget_R_cash: float = 0.20
    # Infl Hedge (S): gold-dominated inflation hedge
    budget_S_gld: float = 0.50
    budget_S_cash: float = 0.30
    budget_S_spy: float = 0.10
    budget_S_efa: float = 0.05
    budget_S_tlt: float = 0.05
    # Defensive (D): bonds + cash safety
    budget_D_tlt: float = 0.50
    budget_D_cash: float = 0.30
    budget_D_gld: float = 0.10
    budget_D_spy: float = 0.05
    budget_D_efa: float = 0.05
    # Neutral: balanced fallback
    budget_neutral_spy: float = 0.20
    budget_neutral_efa: float = 0.10
    budget_neutral_tlt: float = 0.25
    budget_neutral_gld: float = 0.25
    budget_neutral_cash: float = 0.20
