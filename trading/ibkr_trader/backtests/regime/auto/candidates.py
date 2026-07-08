"""Candidate mutations for regime greedy optimization — 68 total."""
from __future__ import annotations


def get_all_candidates() -> list[tuple[str, dict]]:
    """Return all candidate mutations for greedy forward selection."""
    candidates = []

    # -- HMM Architecture (8) --
    candidates.extend([
        ("sticky_diag_30", {"sticky_diag": 30.0}),
        ("sticky_diag_70", {"sticky_diag": 70.0}),
        ("sticky_diag_100", {"sticky_diag": 100.0}),
        ("sticky_offdiag_1", {"sticky_offdiag": 1.0}),
        ("sticky_offdiag_5", {"sticky_offdiag": 5.0}),
        ("n_iter_first_200", {"n_iter_first_fit": 200}),
        ("n_iter_refit_100", {"n_iter_refit": 100}),
        ("covariance_diag", {"covariance_type": "diag"}),
    ])

    # -- Feature Engineering (6) --
    candidates.extend([
        ("z_window_126", {"z_window": 126}),
        ("z_window_504", {"z_window": 504}),
        ("z_minp_30", {"z_minp": 30}),
        ("z_minp_90", {"z_minp": 90}),
        ("cov_window_42", {"cov_window": 42}),
        ("cov_window_126", {"cov_window": 126}),
    ])

    # -- Refit Strategy (7) --
    candidates.extend([
        ("refit_6M", {"refit_freq": "6ME"}),
        ("refit_2A", {"refit_freq": "2YE"}),
        ("no_warm_start", {"use_warm_start": False}),
        ("val_window_42", {"refit_validation_window": 42}),
        ("val_window_126", {"refit_validation_window": 126}),
        ("ll_tol_0.25", {"refit_ll_tolerance": 0.25}),
        ("ll_tol_2.0", {"refit_ll_tolerance": 2.0}),
    ])

    # -- Crisis Overlay (7) --
    candidates.extend([
        ("crisis_vix_heavy", {"crisis_weights": (0.5, 0.3, 0.2)}),
        ("crisis_spread_heavy", {"crisis_weights": (0.3, 0.5, 0.2)}),
        ("crisis_logit_a_0.5", {"crisis_logit_a": 0.5}),
        ("crisis_logit_a_1.5", {"crisis_logit_a": 1.5}),
        ("crisis_logit_a_2.0", {"crisis_logit_a": 2.0}),
        ("crisis_logit_b_n05", {"crisis_logit_b": -0.5}),
        ("crisis_logit_b_p05", {"crisis_logit_b": 0.5}),
    ])

    # -- Confidence Blending (5) --
    candidates.extend([
        ("conf_floor_0.2", {"conf_floor": 0.2}),
        ("conf_floor_0.4", {"conf_floor": 0.4}),
        ("conf_floor_0.5", {"conf_floor": 0.5}),
        ("stability_weight_0.3", {"stability_weight": 0.3}),
        ("stability_weight_0.7", {"stability_weight": 0.7}),
    ])

    # -- Risk Budgeting (4) --
    candidates.extend([
        ("sigma_floor_0.03", {"sigma_floor_annual": 0.03}),
        ("sigma_floor_0.08", {"sigma_floor_annual": 0.08}),
        ("per_strat_max_0.30", {"per_strat_max": 0.30}),
        ("per_strat_max_0.50", {"per_strat_max": 0.50}),
    ])

    # -- Ventilator (14) --
    candidates.extend([
        ("vent_lambda_0.6", {"ventilator_lambda": 0.6}),
        ("vent_lambda_1.0", {"ventilator_lambda": 1.0}),
        ("vent_vmin_0.1", {"ventilator_vmin": 0.1}),
        ("vent_vmin_0.3", {"ventilator_vmin": 0.3}),
        ("rho_short_10", {"rho_short_window": 10}),
        ("rho_short_30", {"rho_short_window": 30}),
        ("rho_long_40", {"rho_long_window": 40}),
        ("rho_long_90", {"rho_long_window": 90}),
        ("drho_thresh_0.15", {"delta_rho_threshold": 0.15}),
        ("drho_thresh_0.35", {"delta_rho_threshold": 0.35}),
        ("drho_exempt_n20", {"delta_rho_exempt": -0.20}),
        ("drho_exempt_0", {"delta_rho_exempt": 0.0}),
        ("pnl_confirm_5", {"pnl_confirm_days": 5}),
        ("pnl_confirm_20", {"pnl_confirm_days": 20}),
    ])

    # -- Leverage Governor (17) --
    candidates.extend([
        ("L_max_0.8", {"L_max": 0.8}),
        ("L_max_1.2", {"L_max": 1.2}),
        ("L_max_1.5", {"L_max": 1.5}),
        ("target_vol_0.08", {"base_target_vol_annual": 0.08}),
        ("target_vol_0.10", {"base_target_vol_annual": 0.10}),
        ("target_vol_0.15", {"base_target_vol_annual": 0.15}),
        ("kappa_1.0", {"kappa_totalvol_cap": 1.0}),
        ("kappa_1.5", {"kappa_totalvol_cap": 1.5}),
        ("gamma_0.05", {"gamma": 0.05}),
        ("gamma_0.20", {"gamma": 0.20}),
        ("ewma_down_10", {"ewma_downside_span": 10}),
        ("ewma_down_30", {"ewma_downside_span": 30}),
        ("ewma_total_30", {"ewma_total_span": 30}),
        ("ewma_total_90", {"ewma_total_span": 90}),
        ("s_floor_0.2", {"s_floor": 0.2}),
        ("s_floor_0.4", {"s_floor": 0.4}),
        ("dd_ladder_aggressive", {"dd_ladder": ((-0.06, 1.0), (-0.10, 0.7), (-0.14, 0.5), (-0.18, 0.3))}),
    ])

    return candidates
