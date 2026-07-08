"""Per-phase candidate lists for multi-phase regime optimization.

Each phase targets specific root causes with tailored parameter ranges.
Phase 1: HMM dynamics (sticky prior, rolling window, warm start)
Phase 2: Features (new/drop features, z-score, covariance)
Phase 3: Crisis + portfolio (crisis overlay, confidence, ventilator, leverage, risk budget)
Phase 4: Fine-tuning (narrow ranges around prior phase optima)
"""
from __future__ import annotations

from typing import Any


def candidate_profile_choices() -> tuple[str, ...]:
    """Return supported candidate profile names for CLI wiring."""
    return ("default", "step9_r6", "r7_overlay", "r8_stress", "r9_budget")


def get_phase_candidates(
    phase: int,
    prior_diagnostics: dict | None = None,
    profile: str = "default",
) -> list[tuple[str, dict[str, Any]]]:
    """Return candidate list for a given phase.

    Args:
        phase: 1-4
        prior_diagnostics: Optional diagnostics from previous phase (for adaptive candidates).
        profile: Candidate profile name. ``default`` powers the standard
            research pipeline; ``step9_r6`` matches the assessment's R6 launch.

    Returns:
        List of (name, mutations_dict) tuples.
    """
    if phase not in {1, 2, 3, 4, 5}:
        raise ValueError(f"Unknown phase: {phase}. Valid phases: 1-5")
    if profile == "default":
        if phase == 1:
            return _phase_1_candidates()
        if phase == 2:
            return _phase_2_candidates()
        if phase == 3:
            return _phase_3_candidates()
        return _phase_4_candidates(prior_diagnostics)
    if profile == "r9_budget":
        if phase == 5:
            return _r9_budget_phase_5_candidates()
        return []
    if profile == "r8_stress":
        if phase == 1:
            return _r8_stress_phase_1_candidates()
        if phase == 2:
            return _r8_stress_phase_2_candidates()
        if phase == 3:
            return _r8_stress_phase_3_candidates()
        if phase == 4:
            return _r8_stress_phase_4_candidates()
        return []
    if profile == "r7_overlay":
        if phase == 1:
            return _r7_overlay_phase_1_candidates()
        if phase == 2:
            return _r7_overlay_phase_2_candidates()
        if phase == 3:
            return _r7_overlay_phase_3_candidates()
        return []  # no phase 4
    if profile == "step9_r6":
        if phase == 1:
            return _step9_r6_phase_1_candidates()
        if phase == 2:
            return _step9_r6_phase_2_candidates()
        if phase == 3:
            return _step9_r6_phase_3_candidates()
        return _step9_r6_phase_4_candidates()
    raise ValueError(f"Unknown phase candidate profile: {profile}")


def _dedupe_candidates(candidates: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Deduplicate candidate names while preserving order."""
    seen = set()
    deduped = []
    for name, muts in candidates:
        if name in seen:
            continue
        seen.add(name)
        deduped.append((name, muts))
    return deduped


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """Phase 1: Fix HMM Dynamics — all HMM-affecting candidates."""
    candidates = []

    # Sticky prior (reduce from default 50)
    for diag in [5, 10, 15, 20]:
        candidates.append((f"sticky_diag_{diag}", {"sticky_diag": float(diag)}))

    for offdiag in [3, 5, 8]:
        candidates.append((f"sticky_offdiag_{offdiag}", {"sticky_offdiag": float(offdiag)}))

    # Rolling window (switch from expanding)
    for years in [5, 7, 10]:
        candidates.append((
            f"rolling_{years}y",
            {"use_expanding_window": False, "rolling_window_years": years},
        ))

    # Warm start
    candidates.append(("no_warm_start", {"use_warm_start": False}))
    for std in [0.1, 0.3]:
        candidates.append((
            f"warm_start_perturb_{std}",
            {"warm_start_perturb_std": std},
        ))

    # Refit frequency
    for freq in ["6ME", "QE"]:
        candidates.append((f"refit_{freq}", {"refit_freq": freq}))

    # OOS guard relaxation (allow structural change)
    for tol in [1.0, 2.0, 5.0]:
        candidates.append((f"ll_tol_{tol}", {"refit_ll_tolerance": tol}))

    # --- Combination candidates ---
    # Single-param changes can't break the 2-state collapse alone;
    # the root causes are compounding. Test key multi-param combos.
    for diag in [5, 10]:
        for years in [5, 7]:
            candidates.append((
                f"combo_sd{diag}_roll{years}y",
                {"sticky_diag": float(diag), "use_expanding_window": False,
                 "rolling_window_years": years},
            ))
            candidates.append((
                f"combo_sd{diag}_roll{years}y_perturb",
                {"sticky_diag": float(diag), "use_expanding_window": False,
                 "rolling_window_years": years, "warm_start_perturb_std": 0.3},
            ))
    # Low sticky + no warm start (cold start each refit)
    for diag in [5, 10]:
        candidates.append((
            f"combo_sd{diag}_cold",
            {"sticky_diag": float(diag), "use_warm_start": False},
        ))
    # Aggressive: low sticky + rolling + relaxed OOS + quarterly refit
    candidates.append((
        "combo_aggressive_7y",
        {"sticky_diag": 5.0, "use_expanding_window": False,
         "rolling_window_years": 7, "refit_ll_tolerance": 5.0, "refit_freq": "QE"},
    ))
    candidates.append((
        "combo_aggressive_5y",
        {"sticky_diag": 5.0, "use_expanding_window": False,
         "rolling_window_years": 5, "refit_ll_tolerance": 5.0, "refit_freq": "QE"},
    ))
    # Rolling window + perturbation (keep default sticky, just fix window lock-in)
    for years in [5, 7]:
        candidates.append((
            f"combo_roll{years}y_perturb",
            {"use_expanding_window": False, "rolling_window_years": years,
             "warm_start_perturb_std": 0.3},
        ))

    # Imp 2: Label anchoring (HMM-affecting -- changes alignment)
    for w in [0.3, 0.5, 1.0]:
        candidates.append((f"label_anchor_{w}", {"label_continuity_weight": w}))

    # HMM Ensemble (HMM-affecting -- requires refit)
    # Standalone ensemble candidates (may fail if baseline has 2-state collapse)
    for n in [5, 10]:
        candidates.append((f"ensemble_{n}", {"n_ensemble_models": n}))
    # Ensemble + structural fix combos (ensemble alone can't break 2-state collapse;
    # must pair with low sticky + cold/rolling to get 3+ active regimes)
    for n in [5, 10]:
        candidates.append((
            f"combo_sd10_cold_ensemble_{n}",
            {"sticky_diag": 10.0, "use_warm_start": False, "n_ensemble_models": n},
        ))
        candidates.append((
            f"combo_aggressive_7y_ensemble_{n}",
            {"sticky_diag": 5.0, "use_expanding_window": False,
             "rolling_window_years": 7, "refit_ll_tolerance": 5.0,
             "refit_freq": "QE", "n_ensemble_models": n},
        ))
    # Ensemble + label anchoring combinations
    candidates.append(("ensemble_5_anchor_0.5", {"n_ensemble_models": 5, "label_continuity_weight": 0.5}))
    candidates.append(("ensemble_10_anchor_0.5", {"n_ensemble_models": 10, "label_continuity_weight": 0.5}))

    return candidates


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """Phase 2: Fix Features — feature engineering candidates."""
    candidates = []

    # New features (HMM-affecting)
    candidates.append(("add_commodity", {"use_commodity_feature": True}))
    candidates.append(("add_real_rates", {"use_real_rates_feature": True}))

    # Feature drops (HMM-affecting)
    candidates.append(("drop_momentum_breadth", {"drop_momentum_breadth": True}))
    candidates.append(("drop_eq_bond_corr", {"drop_eq_bond_corr": True}))

    # Z-score window (HMM-affecting)
    for window in [126, 504]:
        candidates.append((f"z_window_{window}", {"z_window": window}))
    for minp in [30, 90]:
        candidates.append((f"z_minp_{minp}", {"z_minp": minp}))

    # Covariance window (non-HMM)
    for cov_win in [42, 126]:
        candidates.append((f"cov_window_{cov_win}", {"cov_window": cov_win}))

    # HMM covariance type
    candidates.append(("cov_type_diag", {"covariance_type": "diag"}))

    # Feature swap combinations
    candidates.append((
        "swap_eqbond_for_commodity",
        {"use_commodity_feature": True, "drop_eq_bond_corr": True},
    ))
    candidates.append((
        "swap_breadth_for_realrates",
        {"use_real_rates_feature": True, "drop_momentum_breadth": True},
    ))
    candidates.append((
        "add_both_new_features",
        {"use_commodity_feature": True, "use_real_rates_feature": True},
    ))
    candidates.append((
        "add_both_drop_both",
        {
            "use_commodity_feature": True,
            "use_real_rates_feature": True,
            "drop_momentum_breadth": True,
            "drop_eq_bond_corr": True,
        },
    ))

    # Forward-only filtering (non-HMM -- fast eval via run_from_cache)
    candidates.append(("forward_only", {"use_forward_only": True}))

    # Step 5: moderate posterior calibration is tested as pairs, not independently.
    candidates.append((
        "posterior_pair_1p2_0p8",
        {
            "posterior_temperature": 1.2,
            "posterior_smoothing_eps": 0.01,
            "posterior_ema_alpha": 0.8,
        },
    ))
    candidates.append((
        "posterior_pair_1p5_0p7",
        {
            "posterior_temperature": 1.5,
            "posterior_smoothing_eps": 0.01,
            "posterior_ema_alpha": 0.7,
        },
    ))
    candidates.append((
        "posterior_pair_1p5_0p8",
        {
            "posterior_temperature": 1.5,
            "posterior_smoothing_eps": 0.01,
            "posterior_ema_alpha": 0.8,
        },
    ))

    return candidates


def _phase_3_candidates() -> list[tuple[str, dict]]:
    """Phase 3: Crisis Integration & Portfolio Tuning — mostly non-HMM."""
    candidates = []

    # Crisis overlay
    candidates.append(("crisis_weights_vix_heavy", {"crisis_weights": (0.6, 0.25, 0.1, 0.05)}))
    candidates.append(("crisis_weights_vix_dominant", {"crisis_weights": (0.8, 0.1, 0.05, 0.05)}))
    candidates.append(("crisis_weights_spread_heavy", {"crisis_weights": (0.3, 0.6, 0.1, 0.1)}))
    for a in [0.5, 2.0, 3.0]:
        candidates.append((f"crisis_logit_a_{a}", {"crisis_logit_a": a}))
    for b in [-0.5, 0.5]:
        candidates.append((f"crisis_logit_b_{b}", {"crisis_logit_b": b}))

    # Confidence
    for floor in [0.4, 0.5]:
        candidates.append((f"conf_floor_{floor}", {"conf_floor": floor}))
    for sw in [0.2, 0.8]:
        candidates.append((f"stability_weight_{sw}", {"stability_weight": sw}))

    # Ventilator
    for lam in [0.5, 1.0]:
        candidates.append((f"vent_lambda_{lam}", {"ventilator_lambda": lam}))
    for thresh in [0.15, 0.35]:
        candidates.append((f"delta_rho_thresh_{thresh}", {"delta_rho_threshold": thresh}))
    for exempt in [-0.20, 0.0]:
        candidates.append((f"delta_rho_exempt_{exempt}", {"delta_rho_exempt": exempt}))

    # Leverage
    for lmax in [1.0, 1.3, 1.5]:
        candidates.append((f"L_max_{lmax}", {"L_max": lmax}))
    for kappa in [1.0, 1.5]:
        candidates.append((f"kappa_cap_{kappa}", {"kappa_totalvol_cap": kappa}))
    for vol in [0.08, 0.10]:
        candidates.append((f"target_vol_{vol}", {"base_target_vol_annual": vol}))

    # Risk budget
    for floor in [0.03, 0.08]:
        candidates.append((f"sigma_floor_{floor}", {"sigma_floor_annual": floor}))
    for psm in [0.35, 0.50]:
        candidates.append((f"per_strat_max_{psm}", {"per_strat_max": psm}))

    # Imp 5: Weight smoothing (non-HMM)
    for alpha in [0.3, 0.5, 0.7]:
        candidates.append((f"weight_smooth_{alpha}", {"weight_smoothing_alpha": alpha}))

    # Imp 6: Crisis detection (non-HMM — fast eval via run_from_cache)
    # Shorter z-window for crisis indicators (more responsive to recent stress)
    for czw in [21, 42]:
        candidates.append((f"crisis_z_window_{czw}", {"crisis_z_window": czw}))
    # Asymmetric EMA: fast risk-off response (no smoothing when shifting to S/D)
    for alpha in [0.9, 1.0]:
        candidates.append((f"ema_risk_off_{alpha}", {"posterior_ema_risk_off_alpha": alpha}))
    # Combined: short crisis window + fast risk-off
    candidates.append((
        "crisis_fast_combo",
        {"crisis_z_window": 21, "posterior_ema_risk_off_alpha": 1.0},
    ))

    return candidates


def _phase_4_candidates(prior_diagnostics: dict | None = None) -> list[tuple[str, dict]]:
    """Phase 4: Fine-tuning — narrow ranges around prior optima.

    If prior_diagnostics has 'cumulative_mutations', generates candidates
    around accepted values (±20% or ±1 step). Otherwise, returns defaults.
    """
    candidates = []
    accepted = (prior_diagnostics or {}).get("cumulative_mutations", {})

    if accepted:
        # Generate narrow-range candidates around each accepted numeric param
        _FINE_TUNE = {
            "sticky_diag":           lambda v: [v * 0.8, v * 1.2],
            "sticky_offdiag":        lambda v: [max(1, v - 1), v + 1],
            "rolling_window_years":  lambda v: [max(3, v - 1), v + 1],
            "warm_start_perturb_std": lambda v: [max(0.01, v * 0.5), v * 1.5],
            "refit_ll_tolerance":    lambda v: [max(0.1, v * 0.7), v * 1.3],
            "conf_floor":            lambda v: [max(0.1, v - 0.05), min(0.9, v + 0.05)],
            "L_max":                 lambda v: [max(0.5, v - 0.1), v + 0.1],
            "base_target_vol_annual": lambda v: [max(0.03, v - 0.01), v + 0.01],
            "sigma_floor_annual":    lambda v: [max(0.01, v - 0.01), v + 0.01],
            "crisis_logit_a":        lambda v: [max(0.1, v - 0.5), v + 0.5],
            "posterior_temperature": lambda v: [max(1.0, v - 0.5), v + 0.5],
            "posterior_ema_alpha":   lambda v: [max(0.1, v - 0.1), min(1.0, v + 0.1)],
            "label_continuity_weight": lambda v: [max(0.0, v - 0.1), v + 0.2],
            "weight_smoothing_alpha": lambda v: [max(0.1, v - 0.1), min(1.0, v + 0.1)],
            "n_ensemble_models": lambda v: [max(3, v - 3), v + 5],
            "crisis_z_window": lambda v: [max(21, v - 21), v + 21],
            "posterior_ema_risk_off_alpha": lambda v: [max(0.5, v - 0.1), min(1.0, v + 0.1)],
        }
        for param, gen_fn in _FINE_TUNE.items():
            if param in accepted:
                val = accepted[param]
                if not isinstance(val, (int, float)):
                    continue
                for new_val in gen_fn(val):
                    new_val = round(new_val, 4)
                    if new_val != val:
                        candidates.append((
                            f"fine_{param}_{new_val}",
                            {param: float(new_val)},
                        ))
        # If rolling window was accepted, also try adjacent years
        if "rolling_window_years" in accepted:
            for adj in [accepted["rolling_window_years"] - 1,
                        accepted["rolling_window_years"] + 1]:
                if 3 <= adj <= 15:
                    candidates.append((
                        f"fine_rolling_{adj}y",
                        {"use_expanding_window": False, "rolling_window_years": adj},
                    ))

    # Always include some defaults (may overlap with adaptive — greedy handles dupes)
    for diag in [8, 12, 18]:
        candidates.append((f"fine_sticky_{diag}", {"sticky_diag": float(diag)}))

    candidates.append(("fine_refit_4ME", {"refit_freq": "4ME"}))

    for std in [0.05, 0.15, 0.2]:
        candidates.append((f"fine_perturb_{std}", {"warm_start_perturb_std": std}))

    for floor in [0.35, 0.45]:
        candidates.append((f"fine_conf_{floor}", {"conf_floor": floor}))

    for lmax in [1.1, 1.2]:
        candidates.append((f"fine_L_max_{lmax}", {"L_max": lmax}))

    # Deduplicate by name
    seen = set()
    deduped = []
    for name, muts in candidates:
        if name not in seen:
            seen.add(name)
            deduped.append((name, muts))
    return deduped


def _r7_overlay_phase_1_candidates() -> list[tuple[str, dict]]:
    """R7: Recalibrate crisis and scanner thresholds to reduce over-firing."""
    candidates = []
    # Crisis threshold low
    for tl in [0.45, 0.50, 0.55]:
        candidates.append((f"crisis_thr_low_{tl}", {"crisis_leverage_threshold_low": tl}))
    # Crisis threshold high
    for th in [0.70, 0.75, 0.80]:
        candidates.append((f"crisis_thr_high_{th}", {"crisis_leverage_threshold_high": th}))
    # Scanner threshold
    for st in [0.75, 0.80, 0.85]:
        candidates.append((f"scanner_thr_{st}", {"scanner_threshold": st}))
    # Key combos
    for tl, th in [(0.50, 0.75), (0.50, 0.80), (0.55, 0.80)]:
        candidates.append((
            f"crisis_combo_{tl}_{th}",
            {"crisis_leverage_threshold_low": tl, "crisis_leverage_threshold_high": th},
        ))
    # Crisis + scanner combos
    candidates.append((
        "overlay_tight",
        {"crisis_leverage_threshold_low": 0.55, "crisis_leverage_threshold_high": 0.80,
         "scanner_threshold": 0.85},
    ))
    candidates.append((
        "overlay_moderate",
        {"crisis_leverage_threshold_low": 0.50, "crisis_leverage_threshold_high": 0.75,
         "scanner_threshold": 0.80},
    ))
    return candidates


def _r7_overlay_phase_2_candidates() -> list[tuple[str, dict]]:
    """R7: Calibrate crisis override triggers and crisis channel weights."""
    candidates = []
    # Crisis override threshold
    for cot in [0.65, 0.70, 0.75, 0.80]:
        candidates.append((f"override_crisis_{cot}", {"crisis_override_threshold": cot}))
    # Scanner override threshold
    for sot in [0.75, 0.80, 0.85]:
        candidates.append((f"override_scanner_{sot}", {"scanner_override_threshold": sot}))
    # Crisis weights (4-element: vix, spread, realized_vol, spy_tlt_corr)
    for name, w in [
        ("spread_focus", (0.20, 0.50, 0.10, 0.20)),
        ("balanced", (0.25, 0.45, 0.10, 0.20)),
        ("corr_focus", (0.20, 0.40, 0.10, 0.30)),
        ("vol_focus", (0.20, 0.40, 0.20, 0.20)),
    ]:
        candidates.append((f"crisis_w_{name}", {"crisis_weights": w}))
    return candidates


def _r7_overlay_phase_3_candidates() -> list[tuple[str, dict]]:
    """R7: Fine-tune conjunction gating and risk budget parameters."""
    candidates = []
    # Conjunction 1-layer reduction
    for r in [0.10, 0.15, 0.20]:
        candidates.append((f"conj_1layer_{r}", {"conjunction_1_layer_max_reduction": r}))
    # Conjunction min multiplier
    for m in [0.35, 0.40, 0.50]:
        candidates.append((f"conj_min_{m}", {"conjunction_min_multiplier": m}))
    # Risk budget interaction
    candidates.append(("sigma_floor_0.08", {"sigma_floor_annual": 0.08}))
    candidates.append(("kappa_1.3", {"kappa_totalvol_cap": 1.3}))
    return candidates


def _step9_r6_phase_1_candidates() -> list[tuple[str, dict]]:
    """Assessment Step 9: optimize HMM structure, not ensemble size."""
    blocked = {
        "ensemble_5",
        "ensemble_10",
        "combo_sd10_cold_ensemble_5",
        "combo_aggressive_7y_ensemble_5",
        "combo_sd10_cold_ensemble_10",
        "combo_aggressive_7y_ensemble_10",
        "ensemble_5_anchor_0.5",
        "ensemble_10_anchor_0.5",
    }
    return [(name, muts) for name, muts in _phase_1_candidates() if name not in blocked]


def _step9_r6_phase_2_candidates() -> list[tuple[str, dict]]:
    """Assessment Step 9: moderate posterior calibration is tested jointly."""
    candidates = [
        (name, muts)
        for name, muts in _phase_2_candidates()
        if not name.startswith("posterior_pair_")
    ]
    candidates.extend(
        [
            (
                "posterior_pair_1p2_0p6",
                {
                    "posterior_temperature": 1.2,
                    "posterior_smoothing_eps": 0.01,
                    "posterior_ema_alpha": 0.6,
                },
            ),
            (
                "posterior_pair_1p2_0p65",
                {
                    "posterior_temperature": 1.2,
                    "posterior_smoothing_eps": 0.01,
                    "posterior_ema_alpha": 0.65,
                },
            ),
            (
                "posterior_pair_1p5_0p6",
                {
                    "posterior_temperature": 1.5,
                    "posterior_smoothing_eps": 0.01,
                    "posterior_ema_alpha": 0.6,
                },
            ),
            (
                "posterior_pair_1p5_0p65",
                {
                    "posterior_temperature": 1.5,
                    "posterior_smoothing_eps": 0.01,
                    "posterior_ema_alpha": 0.65,
                },
            ),
        ]
    )
    return candidates


def _step9_r6_phase_3_candidates() -> list[tuple[str, dict]]:
    """Assessment Step 9: narrow Layer 3 to targeted crisis/correlation tests."""
    return [
        (
            "crisis_weights_spread_focused",
            {"crisis_weights": (0.15, 0.55, 0.10, 0.20)},
        ),
        ("crisis_z_window_21", {"crisis_z_window": 21}),
        ("crisis_z_window_42", {"crisis_z_window": 42}),
        (
            "cross_asset_corr_weight_0.2",
            {"crisis_weights": (0.20, 0.50, 0.10, 0.20)},
        ),
        (
            "cross_asset_corr_weight_0.3",
            {"crisis_weights": (0.10, 0.50, 0.10, 0.30)},
        ),
    ]


def _step9_r6_phase_4_candidates() -> list[tuple[str, dict]]:
    """Assessment Step 9: no generic Phase 4 fine-tuning."""
    return []


# ---------------------------------------------------------------------------
# R8 Two-Model Architecture candidates
# ---------------------------------------------------------------------------

def _r8_stress_phase_1_candidates() -> list[tuple[str, dict]]:
    """R8 Phase 1: Stress leverage calibration.

    Baseline uses leverage-only mode (blend disabled). Phase 1 tunes how much
    leverage the stress model reduces, and optionally re-enables allocation blending
    with high thresholds as a candidate.
    """
    candidates = []
    # Leverage reduction: baseline=0.15, explore range
    for r in [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        candidates.append((f"stress_red_{r}", {"stress_reduction_max": r}))
    # Selective blending: only at very high stress (leverage-only + cautious blend)
    for t in [0.60, 0.70, 0.80]:
        candidates.append((f"stress_blend_{t}", {"stress_blend_threshold": t}))
    # Joint leverage + selective blend
    candidates.append(("stress_lev25_blend70", {"stress_reduction_max": 0.25, "stress_blend_threshold": 0.70}))
    candidates.append(("stress_lev30_blend80", {"stress_reduction_max": 0.30, "stress_blend_threshold": 0.80}))
    candidates.append(("stress_lev40_blend70", {"stress_reduction_max": 0.40, "stress_blend_threshold": 0.70}))
    return candidates


def _r8_stress_phase_2_candidates() -> list[tuple[str, dict]]:
    """R8 Phase 2: Stress HMM dynamics -- stickiness, covariance, refit guard.

    Now that blending magnitude is calibrated (Phase 1), tune when stress fires.
    """
    candidates = []
    # Normal-state stickiness (higher = stress fires less often)
    for sd in [30, 40, 60, 80]:
        candidates.append((f"stress_sticky_{sd}", {"stress_sticky_diag": float(sd)}))
    # Stress-state stickiness (lower = stress is more transient)
    for ss in [5, 8, 15, 20]:
        candidates.append((f"stress_stressed_{ss}", {"stress_stressed_sticky": float(ss)}))
    # Off-diagonal prior
    for so in [1.0, 4.0]:
        candidates.append((f"stress_offdiag_{so}", {"stress_sticky_offdiag": so}))
    # Key combos (normal sticky + stress sticky)
    for sd, ss in [(60, 5), (80, 8), (40, 15), (30, 20)]:
        candidates.append((f"stress_asym_{sd}_{ss}", {
            "stress_sticky_diag": float(sd), "stress_stressed_sticky": float(ss)}))
    # Covariance type
    candidates.append(("stress_cov_diag", {"stress_covariance_type": "diag"}))
    # Refit guard tolerance
    for tol in [1.0, 5.0]:
        candidates.append((f"stress_guard_{tol}", {"stress_refit_guard_tol": tol}))
    # Iteration count
    candidates.append(("stress_niter_400", {"stress_n_iter": 400}))
    return candidates


def _r8_stress_phase_3_candidates() -> list[tuple[str, dict]]:
    """R8 Phase 3: Regime budgets -- allocation tilts per regime."""
    candidates = []
    # Defensive: TLT-heavy vs GLD-heavy
    candidates.append(("budget_D_tlt_heavy", {"budget_D_tlt": 0.60, "budget_D_cash": 0.20}))
    candidates.append(("budget_D_gld_heavy", {"budget_D_tlt": 0.40, "budget_D_gld": 0.20}))
    # Goldilocks: aggressive vs conservative
    candidates.append(("budget_G_aggressive", {"budget_G_spy": 0.50, "budget_G_cash": 0.30}))
    candidates.append(("budget_G_conservative", {"budget_G_tlt": 0.10, "budget_G_gld": 0.10}))
    # Stagflation: GLD tilt
    candidates.append(("budget_S_gld_heavy", {"budget_S_gld": 0.60, "budget_S_cash": 0.20}))
    candidates.append(("budget_S_cash_heavy", {"budget_S_gld": 0.40, "budget_S_cash": 0.40}))
    # Reflation
    candidates.append(("budget_R_gld_heavy", {"budget_R_gld": 0.35, "budget_R_spy": 0.30}))
    # Crisis weights (still affects diagnostic p_crisis)
    candidates.append(("crisis_w_spread_focus", {"crisis_weights": (0.20, 0.50, 0.10, 0.20)}))
    candidates.append(("crisis_w_balanced", {"crisis_weights": (0.25, 0.45, 0.10, 0.20)}))
    candidates.append(("crisis_w_vol_focus", {"crisis_weights": (0.20, 0.40, 0.20, 0.20)}))
    return candidates


def _r8_stress_phase_4_candidates() -> list[tuple[str, dict]]:
    """R8 Phase 4: Fine-tuning -- macro HMM calibration, architecture, and risk budget."""
    candidates = []
    for a in [0.6, 0.8]:
        candidates.append((f"ema_alpha_{a}", {"posterior_ema_alpha": a}))
    for t in [1.2, 1.8]:
        candidates.append((f"post_temp_{t}", {"posterior_temperature": t}))
    for w in [0.4, 0.6]:
        candidates.append((f"w_smooth_{w}", {"weight_smoothing_alpha": w}))
    for s in [0.08, 0.10]:
        candidates.append((f"sigma_floor_{s}", {"sigma_floor_annual": s}))
    # Architecture candidates: ensemble and scanner (not forced in baseline)
    candidates.append(("ensemble_5", {"n_ensemble_models": 5}))
    candidates.append(("ensemble_3", {"n_ensemble_models": 3}))
    candidates.append(("scanner_on", {"scanner_enabled": True}))
    candidates.append(("full_stack", {
        "scanner_enabled": True, "n_ensemble_models": 5,
        "crisis_weights": (0.3, 0.6, 0.1, 0.1)}))
    return candidates


# ---------------------------------------------------------------------------
# R9 Budget-Only Optimization candidates
# ---------------------------------------------------------------------------

def _r9_budget_phase_5_candidates() -> list[tuple[str, dict]]:
    """R9 Phase 5: Per-regime complete budget profiles.

    Each candidate sets all active assets for one regime (sums to 1.0).
    The greedy optimizer applies these on top of the R9 baseline which
    already includes all 25 budget fields at MetaConfig defaults.
    """
    candidates: list[tuple[str, dict]] = []

    # --- Goldilocks (4 variants -- risk-on alternatives) ---
    candidates.append(("G_equity_heavy", {
        "budget_G_spy": 0.50, "budget_G_efa": 0.20,
        "budget_G_tlt": 0.00, "budget_G_gld": 0.05, "budget_G_cash": 0.25,
    }))
    candidates.append(("G_spy_tilt", {
        "budget_G_spy": 0.50, "budget_G_efa": 0.10,
        "budget_G_tlt": 0.05, "budget_G_gld": 0.05, "budget_G_cash": 0.30,
    }))
    candidates.append(("G_conservative", {
        "budget_G_spy": 0.35, "budget_G_efa": 0.15,
        "budget_G_tlt": 0.10, "budget_G_gld": 0.10, "budget_G_cash": 0.30,
    }))
    candidates.append(("G_no_cash", {
        "budget_G_spy": 0.55, "budget_G_efa": 0.25,
        "budget_G_tlt": 0.05, "budget_G_gld": 0.15, "budget_G_cash": 0.00,
    }))

    # --- Reflation (3 variants -- commodity/equity balance) ---
    candidates.append(("R_equity_tilt", {
        "budget_R_spy": 0.40, "budget_R_efa": 0.20, "budget_R_gld": 0.20,
        "budget_R_tlt": 0.00, "budget_R_cash": 0.20,
    }))
    candidates.append(("R_gld_dominant", {
        "budget_R_spy": 0.25, "budget_R_efa": 0.10, "budget_R_gld": 0.40,
        "budget_R_tlt": 0.05, "budget_R_cash": 0.20,
    }))
    candidates.append(("R_balanced", {
        "budget_R_spy": 0.30, "budget_R_efa": 0.15, "budget_R_gld": 0.25,
        "budget_R_tlt": 0.05, "budget_R_cash": 0.25,
    }))

    # --- Stagflation (3 variants -- safe haven mix) ---
    candidates.append(("S_deep_defensive", {
        "budget_S_gld": 0.55, "budget_S_cash": 0.35, "budget_S_spy": 0.05,
        "budget_S_efa": 0.00, "budget_S_tlt": 0.05,
    }))
    candidates.append(("S_tlt_hedge", {
        "budget_S_gld": 0.40, "budget_S_cash": 0.25, "budget_S_spy": 0.10,
        "budget_S_efa": 0.05, "budget_S_tlt": 0.20,
    }))
    candidates.append(("S_mild", {
        "budget_S_gld": 0.40, "budget_S_cash": 0.30, "budget_S_spy": 0.15,
        "budget_S_efa": 0.10, "budget_S_tlt": 0.05,
    }))

    # --- Defensive (3 variants -- bond/cash allocation) ---
    candidates.append(("D_tlt_dominant", {
        "budget_D_tlt": 0.60, "budget_D_cash": 0.25, "budget_D_gld": 0.05,
        "budget_D_spy": 0.05, "budget_D_efa": 0.05,
    }))
    candidates.append(("D_gld_tilt", {
        "budget_D_tlt": 0.40, "budget_D_cash": 0.20, "budget_D_gld": 0.25,
        "budget_D_spy": 0.10, "budget_D_efa": 0.05,
    }))
    candidates.append(("D_moderate", {
        "budget_D_tlt": 0.40, "budget_D_cash": 0.25, "budget_D_gld": 0.15,
        "budget_D_spy": 0.10, "budget_D_efa": 0.10,
    }))

    # --- Neutral fallback (4 variants -- affects all low-confidence periods) ---
    candidates.append(("N_equity_tilt", {
        "budget_neutral_spy": 0.25, "budget_neutral_efa": 0.15,
        "budget_neutral_tlt": 0.20, "budget_neutral_gld": 0.20,
        "budget_neutral_cash": 0.20,
    }))
    candidates.append(("N_safety_tilt", {
        "budget_neutral_spy": 0.15, "budget_neutral_efa": 0.10,
        "budget_neutral_tlt": 0.30, "budget_neutral_gld": 0.25,
        "budget_neutral_cash": 0.20,
    }))
    candidates.append(("N_gld_anchor", {
        "budget_neutral_spy": 0.20, "budget_neutral_efa": 0.10,
        "budget_neutral_tlt": 0.20, "budget_neutral_gld": 0.30,
        "budget_neutral_cash": 0.20,
    }))
    candidates.append(("N_minimal_cash", {
        "budget_neutral_spy": 0.25, "budget_neutral_efa": 0.15,
        "budget_neutral_tlt": 0.25, "budget_neutral_gld": 0.25,
        "budget_neutral_cash": 0.10,
    }))

    return candidates
