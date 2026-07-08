"""Crisis detection phase candidates -- round 3: recovery architecture + robustness.

Phase 1: Recovery Architecture (accel de-escalation, WATCH recalibration, hysteresis)
Phase 2: Threshold Re-optimization (VIX, Credit Spread, SPY DD around R2 optima)
Phase 3: Correlation + Yield Curve + Conjunction (stability tuning)
Phase 4: Fine-tuning (+/-4-8% perturbations on accepted mutations)
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
            if name in seen:
                continue
            seen.add(name)
            candidates.append((name, muts))

    if phase == 1:
        candidates.extend(_phase_1_recovery_architecture(prior))
    elif phase == 2:
        candidates.extend(_phase_2_threshold_reopt(prior))
    elif phase == 3:
        candidates.extend(_phase_3_corr_yield_conjunction(prior))
    elif phase == 4:
        candidates.extend(_phase_4_finetune(prior))

    seen_names: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, muts in candidates:
        if name in seen_names:
            continue
        seen_names.add(name)
        unique.append((name, muts))
    return unique


def _phase_1_recovery_architecture(prior: dict) -> list[tuple[str, dict]]:
    """Phase 1: Recovery + early advisory/action architecture."""
    candidates: list[tuple[str, dict]] = []

    # Accelerated de-escalation individual sweeps
    for val in [0, 3, 5, 7, 10]:
        candidates.append((
            f"accel_deesc_{val}d",
            {"ACCEL_DEESCALATE_NORMAL_DAYS": val},
        ))

    # WATCH recalibration
    for val in [1, 2, 3]:
        candidates.append((
            f"watch_min_primary_{val}",
            {"WATCH_MIN_PRIMARY": val},
        ))

    # Hysteresis retuning with recovery incentive
    for val in [1, 2, 3]:
        candidates.append((
            f"deesc_crisis_{val}d",
            {"DEESCALATE_CRISIS_DAYS": val},
        ))
    for val in [1, 2, 3]:
        candidates.append((
            f"deesc_warning_{val}d",
            {"DEESCALATE_WARNING_DAYS": val},
        ))
    for val in [1, 2]:
        candidates.append((
            f"deesc_watch_{val}d",
            {"DEESCALATE_WATCH_DAYS": val},
        ))

    # Architecture combos
    candidates.extend([
        ("arch_fast", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 3,
            "DEESCALATE_CRISIS_DAYS": 1,
            "DEESCALATE_WARNING_DAYS": 1,
            "DEESCALATE_WATCH_DAYS": 1,
        }),
        ("arch_balanced", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 5,
            "DEESCALATE_CRISIS_DAYS": 2,
            "DEESCALATE_WARNING_DAYS": 2,
            "DEESCALATE_WATCH_DAYS": 1,
        }),
        ("arch_conservative", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 7,
            "DEESCALATE_CRISIS_DAYS": 2,
            "DEESCALATE_WARNING_DAYS": 2,
            "DEESCALATE_WATCH_DAYS": 2,
        }),
    ])

    # ACCEL + WATCH_MIN pairs
    candidates.extend([
        ("accel3_watch2", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 3, "WATCH_MIN_PRIMARY": 2,
        }),
        ("accel5_watch2", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 5, "WATCH_MIN_PRIMARY": 2,
        }),
        ("accel7_watch2", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 7, "WATCH_MIN_PRIMARY": 2,
        }),
        ("accel5_watch3", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 5, "WATCH_MIN_PRIMARY": 3,
        }),
    ])

    # Full architecture combos with WATCH
    candidates.extend([
        ("arch_fast_watch2", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 3,
            "DEESCALATE_CRISIS_DAYS": 1,
            "DEESCALATE_WARNING_DAYS": 1,
            "DEESCALATE_WATCH_DAYS": 1,
            "WATCH_MIN_PRIMARY": 2,
        }),
        ("arch_balanced_watch2", {
            "ACCEL_DEESCALATE_NORMAL_DAYS": 5,
            "DEESCALATE_CRISIS_DAYS": 2,
            "DEESCALATE_WARNING_DAYS": 2,
            "DEESCALATE_WATCH_DAYS": 1,
            "WATCH_MIN_PRIMARY": 2,
        }),
    ])

    # Early stress-formation layer. These candidates target action latency
    # without loosening hard WARNING/CRISIS thresholds.
    for val in [1, 2, 3]:
        candidates.append((
            f"stress_min_score_{val}",
            {"STRESS_FORMATION_MIN_SCORE": val},
        ))

    candidates.extend([
        ("shock_fast", {
            "SHOCK_SPY_3D_RETURN": -0.03,
            "SHOCK_SPY_5D_RETURN": -0.05,
            "SHOCK_VIX_3D_CHANGE": 6.0,
            "SHOCK_MIN_VIX": 22.0,
            "SHOCK_CORR_MIN": 0.35,
            "SHOCK_CORR_SPY_5D_RETURN": -0.03,
        }),
        ("shock_early_equity_vol", {
            "SHOCK_SPY_3D_RETURN": -0.025,
            "SHOCK_SPY_5D_RETURN": -0.045,
            "SHOCK_VIX_3D_CHANGE": 5.0,
            "SHOCK_MIN_VIX": 22.0,
            "SHOCK_CORR_MIN": 0.45,
            "SHOCK_CORR_SPY_5D_RETURN": -0.02,
        }),
        ("shock_balanced_live", {
            "SHOCK_SPY_3D_RETURN": -0.04,
            "SHOCK_SPY_5D_RETURN": -0.06,
            "SHOCK_VIX_3D_CHANGE": 8.0,
            "SHOCK_MIN_VIX": 25.0,
            "SHOCK_CORR_MIN": 0.40,
            "SHOCK_CORR_SPY_5D_RETURN": -0.035,
        }),
        ("shock_strict", {
            "SHOCK_SPY_3D_RETURN": -0.05,
            "SHOCK_SPY_5D_RETURN": -0.075,
            "SHOCK_VIX_3D_CHANGE": 10.0,
            "SHOCK_MIN_VIX": 28.0,
            "SHOCK_CORR_MIN": 0.45,
            "SHOCK_CORR_SPY_5D_RETURN": -0.05,
        }),
        ("grind_fast", {
            "GRIND_SPREAD_20D_CHANGE_BPS": 50.0,
            "GRIND_SPY_20D_RETURN": -0.04,
            "GRIND_VIX_MIN": 22.0,
            "GRIND_VIX_PERSIST_DAYS": 3,
            "GRIND_SPREAD_CONFIRM_BPS": 225.0,
            "GRIND_SPY_CONFIRM_20D_RETURN": -0.03,
        }),
        ("grind_balanced_live", {
            "GRIND_SPREAD_20D_CHANGE_BPS": 75.0,
            "GRIND_SPY_20D_RETURN": -0.06,
            "GRIND_VIX_MIN": 25.0,
            "GRIND_VIX_PERSIST_DAYS": 5,
            "GRIND_SPREAD_CONFIRM_BPS": 250.0,
            "GRIND_SPY_CONFIRM_20D_RETURN": -0.04,
        }),
        ("grind_strict", {
            "GRIND_SPREAD_20D_CHANGE_BPS": 100.0,
            "GRIND_SPY_20D_RETURN": -0.08,
            "GRIND_VIX_MIN": 28.0,
            "GRIND_VIX_PERSIST_DAYS": 7,
            "GRIND_SPREAD_CONFIRM_BPS": 300.0,
            "GRIND_SPY_CONFIRM_20D_RETURN": -0.05,
        }),
        ("stress_fast_combo", {
            "STRESS_FORMATION_MIN_SCORE": 2,
            "SHOCK_SPY_3D_RETURN": -0.03,
            "SHOCK_SPY_5D_RETURN": -0.05,
            "SHOCK_VIX_3D_CHANGE": 6.0,
            "SHOCK_MIN_VIX": 22.0,
            "GRIND_SPREAD_20D_CHANGE_BPS": 50.0,
            "GRIND_SPY_20D_RETURN": -0.04,
            "GRIND_VIX_MIN": 22.0,
            "GRIND_VIX_PERSIST_DAYS": 3,
            "GRIND_SPREAD_CONFIRM_BPS": 225.0,
            "GRIND_SPY_CONFIRM_20D_RETURN": -0.03,
        }),
        ("stress_strict_combo", {
            "STRESS_FORMATION_MIN_SCORE": 3,
            "SHOCK_SPY_3D_RETURN": -0.04,
            "SHOCK_SPY_5D_RETURN": -0.06,
            "SHOCK_VIX_3D_CHANGE": 8.0,
            "SHOCK_MIN_VIX": 25.0,
            "GRIND_SPREAD_20D_CHANGE_BPS": 75.0,
            "GRIND_SPY_20D_RETURN": -0.06,
            "GRIND_VIX_MIN": 25.0,
            "GRIND_VIX_PERSIST_DAYS": 5,
        }),
        ("credit_impulse_warning_spread", {
            "CREDIT_IMPULSE_SPREAD_BPS": 336.0,
            "CREDIT_IMPULSE_SPY_3D_RETURN": -0.015,
            "CREDIT_IMPULSE_MIN_VIX": 18.0,
        }),
        ("credit_impulse_crisis_spread", {
            "CREDIT_IMPULSE_SPREAD_BPS": 450.0,
            "CREDIT_IMPULSE_SPY_3D_RETURN": -0.015,
            "CREDIT_IMPULSE_MIN_VIX": 18.0,
        }),
        ("credit_impulse_strict_spread", {
            "CREDIT_IMPULSE_SPREAD_BPS": 500.0,
            "CREDIT_IMPULSE_SPY_3D_RETURN": -0.015,
            "CREDIT_IMPULSE_MIN_VIX": 18.0,
        }),
        ("credit_impulse_deeper_equity", {
            "CREDIT_IMPULSE_SPREAD_BPS": 336.0,
            "CREDIT_IMPULSE_SPY_3D_RETURN": -0.02,
            "CREDIT_IMPULSE_MIN_VIX": 18.0,
        }),
        ("credit_impulse_vix20", {
            "CREDIT_IMPULSE_SPREAD_BPS": 336.0,
            "CREDIT_IMPULSE_SPY_3D_RETURN": -0.015,
            "CREDIT_IMPULSE_MIN_VIX": 20.0,
        }),
        ("hard_credit_impulse_warning_p2", {
            "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS": 2,
            "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY": 1,
        }),
        ("hard_credit_impulse_warning_p3", {
            "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS": 3,
            "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY": 1,
        }),
        ("hard_credit_impulse_warning_p4", {
            "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS": 4,
            "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY": 1,
        }),
        ("hard_credit_impulse_warning_p5", {
            "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS": 5,
            "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY": 1,
        }),
    ])

    # External advisory split tuning. Advisory has no direct risk impact unless
    # the action layer elects to use it economically.
    for val in [3, 4, 5]:
        candidates.append((
            f"advisory_watch_min_primary_{val}",
            {"ADVISORY_WATCH_MIN_PRIMARY": val},
        ))
    for val in [1, 2, 3]:
        candidates.append((
            f"advisory_watch_min_warning_{val}",
            {"ADVISORY_WATCH_MIN_WARNING": val},
        ))
    for val in [1, 2]:
        candidates.append((
            f"advisory_watch_min_crisis_{val}",
            {"ADVISORY_WATCH_MIN_CRISIS": val},
        ))

    return candidates


def _phase_2_threshold_reopt(prior: dict) -> list[tuple[str, dict]]:
    """Phase 2: Threshold re-optimization around R2 optima."""
    candidates: list[tuple[str, dict]] = []

    # VIX around R2 optima (27/34.56/38)
    for val in [25, 27, 29]:
        candidates.append((f"vix_watch_{val}", {"VIX_WATCH": float(val)}))
    for val in [32, 34.56, 37]:
        candidates.append((f"vix_warning_{val}", {"VIX_WARNING": float(val)}))
    for val in [36, 38, 40, 42]:
        candidates.append((f"vix_crisis_{val}", {"VIX_CRISIS": float(val)}))

    # VIX triplets
    candidates.extend([
        ("vix_tighter", {
            "VIX_WATCH": 25.0, "VIX_WARNING": 32.0, "VIX_CRISIS": 36.0,
        }),
        ("vix_r2_default", {
            "VIX_WATCH": 27.0, "VIX_WARNING": 34.56, "VIX_CRISIS": 38.0,
        }),
        ("vix_looser", {
            "VIX_WATCH": 29.0, "VIX_WARNING": 37.0, "VIX_CRISIS": 42.0,
        }),
    ])

    # Credit Spread around R2 optima (250/336/450)
    for val in [200, 250, 300]:
        candidates.append((f"spread_watch_{val}", {"SPREAD_WATCH_BPS": float(val)}))
    for val in [300, 336, 380]:
        candidates.append((f"spread_warning_{val}", {"SPREAD_WARNING_BPS": float(val)}))
    for val in [400, 450, 500]:
        candidates.append((f"spread_crisis_{val}", {"SPREAD_CRISIS_BPS": float(val)}))

    # Spread triplets
    candidates.extend([
        ("spread_tighter", {
            "SPREAD_WATCH_BPS": 200.0, "SPREAD_WARNING_BPS": 300.0,
            "SPREAD_CRISIS_BPS": 400.0,
        }),
        ("spread_r2_default", {
            "SPREAD_WATCH_BPS": 250.0, "SPREAD_WARNING_BPS": 336.0,
            "SPREAD_CRISIS_BPS": 450.0,
        }),
        ("spread_looser", {
            "SPREAD_WATCH_BPS": 300.0, "SPREAD_WARNING_BPS": 380.0,
            "SPREAD_CRISIS_BPS": 500.0,
        }),
    ])

    # SPY DD around R2 optima (-0.05/-0.0576/-0.12)
    for val in [-0.049, -0.054, -0.0576, -0.063, -0.070]:
        candidates.append((
            f"spy_dd_warning_{abs(val):.3f}",
            {"SPY_DD_WARNING": val},
        ))
    for val in [-0.10, -0.12, -0.15]:
        candidates.append((
            f"spy_dd_crisis_{abs(val):.2f}",
            {"SPY_DD_CRISIS": val},
        ))

    # Cross-channel combos: VIX + Spread
    candidates.extend([
        ("vix_spread_tighter", {
            "VIX_WATCH": 25.0, "VIX_WARNING": 32.0, "VIX_CRISIS": 36.0,
            "SPREAD_WATCH_BPS": 200.0, "SPREAD_WARNING_BPS": 300.0,
            "SPREAD_CRISIS_BPS": 400.0,
        }),
        ("vix_spread_looser", {
            "VIX_WATCH": 29.0, "VIX_WARNING": 37.0, "VIX_CRISIS": 42.0,
            "SPREAD_WATCH_BPS": 300.0, "SPREAD_WARNING_BPS": 380.0,
            "SPREAD_CRISIS_BPS": 500.0,
        }),
    ])

    return candidates


def _phase_3_corr_yield_conjunction(prior: dict) -> list[tuple[str, dict]]:
    """Phase 3: Correlation + Yield Curve + Conjunction -- stability tuning."""
    candidates: list[tuple[str, dict]] = []

    # Yield curve individual
    for val in [-0.30, -0.40, -0.50, -0.60]:
        candidates.append((
            f"slope_watch_{abs(val):.2f}",
            {"SLOPE_WATCH_THRESHOLD": val},
        ))
    for val in [0.35, 0.40, 0.50, 0.60]:
        candidates.append((
            f"slope_warn_{val:.2f}",
            {"SLOPE_STEEPEN_WARNING": val},
        ))
    for val in [0.50, 0.55, 0.60, 0.70]:
        candidates.append((
            f"slope_crisis_{val:.2f}",
            {"SLOPE_STEEPEN_CRISIS": val},
        ))
    for val in [60, 75, 90, 120]:
        candidates.append((
            f"slope_lookback_{val}",
            {"SLOPE_INVERSION_LOOKBACK": val},
        ))

    # SPY-TLT Correlation individual
    for val in [10, 15, 20]:
        candidates.append((f"corr_window_{val}", {"CORR_WINDOW": val}))
    for val in [0.30, 0.35, 0.40, 0.45]:
        candidates.append((
            f"corr_watch_{val:.2f}",
            {"CORR_WATCH": val},
        ))
    for val in [0.45, 0.50, 0.55, 0.60]:
        candidates.append((
            f"corr_warning_{val:.2f}",
            {"CORR_WARNING": val},
        ))
    for val in [-0.04, -0.05, -0.07, -0.09]:
        candidates.append((
            f"corr_spy_dd_{abs(val):.2f}",
            {"CORR_CRISIS_SPY_DD": val},
        ))

    # Yield curve combos
    candidates.extend([
        ("slope_tight", {
            "SLOPE_WATCH_THRESHOLD": -0.30,
            "SLOPE_STEEPEN_WARNING": 0.35,
            "SLOPE_STEEPEN_CRISIS": 0.50,
        }),
        ("slope_loose", {
            "SLOPE_WATCH_THRESHOLD": -0.60,
            "SLOPE_STEEPEN_WARNING": 0.50,
            "SLOPE_STEEPEN_CRISIS": 0.70,
        }),
    ])

    # Correlation combos
    candidates.extend([
        ("corr_tight", {
            "CORR_WATCH": 0.30, "CORR_WARNING": 0.45,
            "CORR_CRISIS": 0.45, "CORR_CRISIS_SPY_DD": -0.04,
        }),
        ("corr_loose", {
            "CORR_WATCH": 0.45, "CORR_WARNING": 0.60,
            "CORR_CRISIS": 0.60, "CORR_CRISIS_SPY_DD": -0.09,
        }),
    ])

    # Conjunction refinement
    for val in [2, 3]:
        candidates.append((f"crisis_min_{val}", {"CRISIS_MIN_PRIMARY": val}))
    for val in [3, 4]:
        candidates.append((f"crisis_alt_{val}", {"CRISIS_ALT_WARNING": val}))

    # Hybrid conjunction
    candidates.extend([
        ("hybrid_1_2", {
            "HYBRID_WARNING_MIN_CRISIS": 1, "HYBRID_WARNING_MIN_PRIMARY": 2,
        }),
        ("hybrid_1_3", {
            "HYBRID_WARNING_MIN_CRISIS": 1, "HYBRID_WARNING_MIN_PRIMARY": 3,
        }),
        ("hybrid_2_2", {
            "HYBRID_WARNING_MIN_CRISIS": 2, "HYBRID_WARNING_MIN_PRIMARY": 2,
        }),
    ])

    # Conjunction strict
    candidates.append(("conjunction_strict", {
        "WARNING_MIN_PRIMARY": 2, "CRISIS_MIN_PRIMARY": 3,
        "CRISIS_ALT_WARNING": 4,
    }))

    return candidates


def _phase_4_finetune(prior: dict) -> list[tuple[str, dict]]:
    """Phase 4: Fine-tuning -- +/-4-8% perturbations on accepted mutations."""
    candidates: list[tuple[str, dict]] = []

    _FLOAT_PARAMS = {
        "VIX_WATCH", "VIX_WARNING", "VIX_CRISIS",
        "SPREAD_WATCH_BPS", "SPREAD_WARNING_BPS", "SPREAD_CRISIS_BPS",
        "SLOPE_WATCH_THRESHOLD", "SLOPE_STEEPEN_WARNING", "SLOPE_STEEPEN_CRISIS",
        "CORR_WATCH", "CORR_WARNING", "CORR_CRISIS", "CORR_CRISIS_SPY_DD",
        "SPY_DD_WATCH", "SPY_DD_WARNING", "SPY_DD_CRISIS",
        "SHOCK_SPY_3D_RETURN", "SHOCK_SPY_5D_RETURN",
        "SHOCK_VIX_3D_CHANGE", "SHOCK_MIN_VIX", "SHOCK_CORR_MIN",
        "SHOCK_CORR_SPY_5D_RETURN", "GRIND_SPREAD_20D_CHANGE_BPS",
        "GRIND_SPY_20D_RETURN", "GRIND_VIX_MIN",
        "GRIND_SPREAD_CONFIRM_BPS", "GRIND_SPY_CONFIRM_20D_RETURN",
        "CREDIT_IMPULSE_SPREAD_BPS", "CREDIT_IMPULSE_SPY_3D_RETURN",
        "CREDIT_IMPULSE_MIN_VIX",
    }
    _INT_PARAMS = {
        "CORR_WINDOW", "SLOPE_INVERSION_LOOKBACK",
        "WATCH_MIN_PRIMARY", "WARNING_MIN_PRIMARY",
        "CRISIS_MIN_PRIMARY", "CRISIS_ALT_WARNING",
        "HYBRID_WARNING_MIN_CRISIS", "HYBRID_WARNING_MIN_PRIMARY",
        "DEESCALATE_CRISIS_DAYS", "DEESCALATE_WARNING_DAYS",
        "DEESCALATE_WATCH_DAYS", "ACCEL_DEESCALATE_NORMAL_DAYS",
        "ADVISORY_WATCH_MIN_PRIMARY", "ADVISORY_WATCH_MIN_WARNING",
        "ADVISORY_WATCH_MIN_CRISIS", "STRESS_FORMATION_MIN_SCORE",
        "GRIND_VIX_PERSIST_DAYS",
        "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS",
        "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY",
    }

    # Float perturbations: +/-4% and +/-8%
    for key in _FLOAT_PARAMS:
        value = prior.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        base_name = key.lower()
        for factor, label in [(0.92, "m08"), (0.96, "m04"), (1.04, "p04"), (1.08, "p08")]:
            adjusted = float(value) * factor
            candidates.append((f"finetune_{base_name}_{label}", {key: adjusted}))

    # Integer perturbations: +/-1
    for key in _INT_PARAMS:
        value = prior.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        base_name = key.lower()
        int_val = int(value)
        if int_val > 1:
            candidates.append((f"finetune_{base_name}_m1", {key: int_val - 1}))
        candidates.append((f"finetune_{base_name}_p1", {key: int_val + 1}))

    return candidates
