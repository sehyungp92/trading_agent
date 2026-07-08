"""ATRSS per-phase candidate experiments.

R1 phases (original independent-account mode):
  Phase 1: Exit Cleanup (26 candidates)
  Phase 2: Signal & Filtering (20 candidates)
  Phase 3: Entry & Fill Optimization (20 candidates)
  Phase 4: Sizing & Fine-tune (16 static + ~14 dynamic)

R9 phases (synchronized/fee-net mode):
  Phase 1: Structural Fixes (targeting capital blocking, symbol selection, sizing)
  Phase 2-4: Reuses R1 Phase 1-3 candidates with R9 scoring
"""
from __future__ import annotations


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Get experiment candidates for a specific phase (R1 mode)."""
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
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


def get_r9_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Get experiment candidates for synchronized ATRSS rounds.

    R2+ starts from the prior optimized config and targets the diagnosed
    alpha omissions directly: opportunity surface, signal geometry,
    execution/stop geometry, then exit/add-on/ranking management.
    """
    if phase == 1:
        candidates = _r10_phase_1_opportunity_surface()
    elif phase == 2:
        candidates = _r10_phase_2_signal_geometry()
    elif phase == 3:
        candidates = _r10_phase_3_execution_and_stops()
    elif phase == 4:
        candidates = _r10_phase_4_exits_addons_allocation(prior_mutations or {})
    else:
        candidates = []

    if suggested_experiments:
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


def get_risk_allocation_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Candidate slate for the post-alpha ATRSS risk-allocation round."""
    if phase == 1:
        candidates = _risk_phase_1_dynamic_risk_exposure()
    elif phase == 2:
        candidates = _risk_phase_2_risk_heat_calibration()
    elif phase == 3:
        candidates = _risk_phase_3_addon_b_winner_lean_in()
    elif phase == 4:
        candidates = _risk_phase_4_guardrails(prior_mutations or {})
    else:
        candidates = []

    if suggested_experiments:
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


# ---------------------------------------------------------------------------
# Phase 1: Exit Cleanup (26 candidates)
# ---------------------------------------------------------------------------

def _phase_1_candidates() -> list[tuple[str, dict]]:
    candidates = []

    # Structural (2)
    candidates.append(("disable_early_stall", {"flags.early_stall_exit": False}))
    candidates.append(("reenable_full_stall", {"flags.stall_exit": True}))

    # TP levels (8)
    candidates.append(("tp1_r_075", {"param_overrides.tp1_r": 0.75}))
    candidates.append(("tp1_r_125", {"param_overrides.tp1_r": 1.25}))
    candidates.append(("tp1_r_150", {"param_overrides.tp1_r": 1.50}))
    candidates.append(("tp1_frac_025", {"param_overrides.tp1_frac": 0.25}))
    candidates.append(("tp1_frac_050", {"param_overrides.tp1_frac": 0.50}))
    candidates.append(("tp2_r_150", {"param_overrides.tp2_r": 1.50}))
    candidates.append(("tp2_r_250", {"param_overrides.tp2_r": 2.50}))
    candidates.append(("tp2_frac_050", {"param_overrides.tp2_frac": 0.50}))

    # BE & trailing (5)
    candidates.append(("be_trigger_075r", {"param_overrides.be_trigger_r": 0.75}))
    candidates.append(("be_trigger_100r", {"param_overrides.be_trigger_r": 1.0}))
    candidates.append(("be_trigger_150r", {"param_overrides.be_trigger_r": 1.5}))
    candidates.append(("be_offset_005", {"param_overrides.be_atr_offset": 0.05}))
    candidates.append(("chand_trigger_075r", {"param_overrides.chandelier_trigger_r": 0.75}))

    # Time decay (5)
    candidates.append(("max_hold_240h", {"param_overrides.max_hold_hours": 240}))
    candidates.append(("max_hold_360h", {"param_overrides.max_hold_hours": 360}))
    candidates.append(("early_stall_8h", {"param_overrides.early_stall_check_hours": 8}))
    candidates.append(("early_stall_mfe_020", {"param_overrides.early_stall_mfe_threshold": 0.20}))
    candidates.append(("early_stall_mfe_050", {"param_overrides.early_stall_mfe_threshold": 0.50}))

    # Profit floor (4) -- dict type
    candidates.append(("floor_tight_low", {
        "param_overrides.profit_floor": {1.0: 0.1, 1.5: 0.35, 2.0: 0.85, 3.0: 1.6, 4.0: 2.6},
    }))
    candidates.append(("floor_tight_all", {
        "param_overrides.profit_floor": {1.0: 0.2, 1.5: 0.50, 2.0: 1.0, 3.0: 1.8, 4.0: 2.8},
    }))
    candidates.append(("floor_loose", {
        "param_overrides.profit_floor": {1.0: -0.25, 1.5: 0.15, 2.0: 0.60, 3.0: 1.3, 4.0: 2.3},
    }))
    candidates.append(("floor_short_tight", {
        "param_overrides.profit_floor_short": {0.75: 0.20, 1.0: 0.60, 1.5: 1.1, 2.0: 1.35},
    }))

    # Stop tightening (2)
    candidates.append(("trend_stop_tight_075", {"param_overrides.trend_stop_tightening": 0.75}))
    candidates.append(("trend_stop_tight_095", {"param_overrides.trend_stop_tightening": 0.95}))

    return candidates


# ---------------------------------------------------------------------------
# Phase 2: Signal & Filtering (20 candidates)
# ---------------------------------------------------------------------------

def _phase_2_candidates() -> list[tuple[str, dict]]:
    candidates = []

    # Structural toggles (5)
    candidates.append(("disable_breakout", {"flags.breakout_entries": False}))
    candidates.append(("disable_momentum_filter", {"flags.momentum_filter": False}))
    candidates.append(("disable_reset_requirement", {"flags.reset_requirement": False}))
    candidates.append(("disable_prior_high", {"flags.prior_high_confirm": False}))
    candidates.append(("disable_hysteresis_gap", {"flags.hysteresis_gap": False}))

    # Quality gate (3)
    candidates.append(("enable_quality_30", {
        "flags.quality_gate": True,
        "param_overrides.quality_gate_threshold": 3.0,
    }))
    candidates.append(("enable_quality_40", {
        "flags.quality_gate": True,
        "param_overrides.quality_gate_threshold": 4.0,
    }))
    candidates.append(("enable_quality_50", {
        "flags.quality_gate": True,
        "param_overrides.quality_gate_threshold": 5.0,
    }))

    # Confirmation tuning (5)
    candidates.append(("fast_confirm_score_45", {"param_overrides.fast_confirm_score": 45}))
    candidates.append(("fast_confirm_score_65", {"param_overrides.fast_confirm_score": 65}))
    candidates.append(("fast_confirm_adx_18", {"param_overrides.fast_confirm_adx": 18}))
    candidates.append(("confirm_days_0", {"param_overrides.confirm_days_normal": 0}))
    candidates.append(("adx_strong_25", {"param_overrides.adx_strong": 25}))

    # Regime sensitivity (3)
    candidates.append(("adx_on_15", {"param_overrides.adx_on": 15}))
    candidates.append(("adx_on_17", {"param_overrides.adx_on": 17}))
    candidates.append(("adx_off_14", {"param_overrides.adx_off": 14}))

    # Momentum/pullback tuning (4)
    candidates.append(("momentum_tol_005", {"param_overrides.momentum_tolerance_atr": 0.05}))
    candidates.append(("momentum_tol_020", {"param_overrides.momentum_tolerance_atr": 0.20}))
    candidates.append(("recovery_strong_070", {"param_overrides.recovery_tolerance_atr_strong": 0.70}))
    candidates.append(("recovery_trend_035", {"param_overrides.recovery_tolerance_atr_trend": 0.35}))

    return candidates


# ---------------------------------------------------------------------------
# Phase 3: Entry & Fill Optimization (20 candidates)
# ---------------------------------------------------------------------------

def _phase_3_candidates() -> list[tuple[str, dict]]:
    candidates = []

    # Order management (4)
    candidates.append(("expiry_6h", {"param_overrides.order_expiry_hours": 6}))
    candidates.append(("expiry_12h", {"param_overrides.order_expiry_hours": 12}))
    candidates.append(("expiry_24h", {"param_overrides.order_expiry_hours": 24}))
    candidates.append(("expiry_36h", {"param_overrides.order_expiry_hours": 36}))

    # Entry slip tolerance (2)
    candidates.append(("disable_slippage_abort", {"flags.slippage_abort": False}))
    candidates.append(("slip_atr_050", {"param_overrides.max_entry_slip_atr": 0.50}))

    # Recovery tolerance (4)
    candidates.append(("recov_strong_060", {"param_overrides.recovery_tolerance_atr_strong": 0.60}))
    candidates.append(("recov_strong_095", {"param_overrides.recovery_tolerance_atr_strong": 0.95}))
    candidates.append(("recov_trend_050", {"param_overrides.recovery_tolerance_atr_trend": 0.50}))
    candidates.append(("recov_base_030", {"param_overrides.recovery_tolerance_atr": 0.30}))

    # Cooldown tuning (4)
    candidates.append(("cd_range_2h", {"param_overrides.cooldown_range": 2}))
    candidates.append(("cd_range_6h", {"param_overrides.cooldown_range": 6}))
    candidates.append(("cd_trend_0h", {"param_overrides.cooldown_trend": 0}))
    candidates.append(("cd_strong_0h", {"param_overrides.cooldown_strong": 0}))

    # Voucher tuning (4)
    candidates.append(("voucher_12h", {"param_overrides.voucher_valid_hours": 12}))
    candidates.append(("voucher_36h", {"param_overrides.voucher_valid_hours": 36}))
    candidates.append(("voucher_48h", {"param_overrides.voucher_valid_hours": 48}))
    candidates.append(("disable_voucher", {"flags.voucher_system": False}))

    # Pullback & EMA (2)
    candidates.append(("ema_pull_normal_35", {"param_overrides.ema_pull_normal": 35}))
    candidates.append(("ema_pull_normal_55", {"param_overrides.ema_pull_normal": 55}))

    return candidates


# ---------------------------------------------------------------------------
# Phase 4: Sizing & Fine-tune (16 static + ~14 dynamic)
# ---------------------------------------------------------------------------

def _phase_4_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    candidates = []

    # Pyramiding (6)
    candidates.append(("addon_a_100r", {"param_overrides.addon_a_r": 1.00}))
    candidates.append(("addon_a_150r", {"param_overrides.addon_a_r": 1.50}))
    candidates.append(("addon_b_150r", {"param_overrides.addon_b_r": 1.50}))
    candidates.append(("addon_b_100r", {"param_overrides.addon_b_r": 1.00}))
    candidates.append(("addon_a_size_050", {"param_overrides.addon_a_size_mult": 0.50}))
    candidates.append(("addon_b_size_025", {"param_overrides.addon_b_size_mult": 0.25}))

    # Portfolio heat (4)
    candidates.append(("heat_4pct", {"param_overrides.max_portfolio_heat": 0.04}))
    candidates.append(("heat_5pct", {"param_overrides.max_portfolio_heat": 0.05}))
    candidates.append(("heat_8pct", {"param_overrides.max_portfolio_heat": 0.08}))
    candidates.append(("heat_10pct", {"param_overrides.max_portfolio_heat": 0.10}))

    # Per-symbol stop & chandelier (4)
    candidates.append(("chand_mult_25", {"param_overrides.chand_mult": 2.5}))
    candidates.append(("chand_mult_35", {"param_overrides.chand_mult": 3.5}))
    candidates.append(("base_risk_008", {"param_overrides.base_risk_pct": 0.008}))
    candidates.append(("base_risk_005", {"param_overrides.base_risk_pct": 0.005}))

    # ADX slope gate (2)
    candidates.append(("adx_slope_neg3", {"param_overrides.adx_slope_gate": -3.0}))
    candidates.append(("adx_slope_neg1", {"param_overrides.adx_slope_gate": -1.0}))

    # Dynamic fine-tuning from Phase 1-3 winners (+-10%, +-20%)
    candidates.extend(_dynamic_finetune(prior_mutations))

    return candidates


def _dynamic_finetune(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Generate +-10% and +-20% variants for top numeric param_override winners."""
    finetune = []
    numeric_winners = {}

    for key, value in prior_mutations.items():
        if key.startswith("param_overrides.") and isinstance(value, (int, float)):
            numeric_winners[key] = value

    # Take up to 4 numeric winners for fine-tuning (14-16 experiments)
    for key, base_val in list(numeric_winners.items())[:4]:
        short_name = key.split(".", 1)[1]
        for pct_label, factor in [("plus10", 1.10), ("plus20", 1.20),
                                   ("minus10", 0.90), ("minus20", 0.80)]:
            new_val = base_val * factor
            # Preserve int type if original was int
            if isinstance(base_val, int):
                new_val = int(round(new_val))
            name = f"{short_name}_{pct_label}"
            finetune.append((name, {key: new_val}))

    return finetune


# ---------------------------------------------------------------------------
# Risk-allocation round: scale strong ATRSS edge without reckless heat
# ---------------------------------------------------------------------------

def _risk_phase_1_dynamic_risk_exposure() -> list[tuple[str, dict]]:
    """Find the live-style dynamic risk band before enabling add-on complexity."""
    return [
        # Keep add-on B disabled in the exposure sweep so base risk is measured
        # cleanly before testing winner lean-in as a separate gated layer.
        ("risk_sized_150bp", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0150,
        }),
        ("risk_sized_175bp", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0175,
        }),
        ("risk_sized_200bp", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0200,
        }),
        ("risk_sized_225bp", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0225,
        }),
        ("risk_sized_240bp", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0240,
        }),
        ("risk_sized_250bp_guardrail_probe", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0250,
        }),

        # Dynamic regime overlays let the risk model lean into the best trend
        # states while cutting exposure when the trend score is marginal.
        ("risk_200bp_strong140_weak075", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0200,
            "param_overrides.dynamic_risk_strong_trend_mult": 1.40,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.75,
        }),
        ("risk_225bp_strong135_weak070", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0225,
            "param_overrides.dynamic_risk_strong_trend_mult": 1.35,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.70,
        }),
        ("risk_225bp_heat055_guarded", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0225,
            "param_overrides.max_portfolio_heat": 0.055,
        }),
        ("risk_240bp_heat055_guarded", {
            "fixed_qty": None,
            "flags.addon_b": False,
            "param_overrides.base_risk_pct": 0.0240,
            "param_overrides.max_portfolio_heat": 0.055,
        }),
    ]


def _risk_phase_2_risk_heat_calibration() -> list[tuple[str, dict]]:
    """Tune risk budget and heat once the exposure band is chosen."""
    return [
        # Local refinement around the aggressive-but-gated dynamic band.
        ("risk_sized_185bp", {"fixed_qty": None, "param_overrides.base_risk_pct": 0.0185}),
        ("risk_sized_200bp", {"fixed_qty": None, "param_overrides.base_risk_pct": 0.0200}),
        ("risk_sized_210bp", {"fixed_qty": None, "param_overrides.base_risk_pct": 0.0210}),
        ("risk_sized_220bp", {"fixed_qty": None, "param_overrides.base_risk_pct": 0.0220}),
        ("risk_sized_230bp", {"fixed_qty": None, "param_overrides.base_risk_pct": 0.0230}),
        ("risk_sized_240bp", {"fixed_qty": None, "param_overrides.base_risk_pct": 0.0240}),
        ("risk_sized_250bp_guardrail_probe", {"fixed_qty": None, "param_overrides.base_risk_pct": 0.0250}),

        # Heat caps stay explicit optimizer mutations so any extra return must
        # survive the shared-capital drawdown and PF gates.
        ("heat_05pct", {"param_overrides.max_portfolio_heat": 0.050}),
        ("heat_055pct", {"param_overrides.max_portfolio_heat": 0.055}),
        ("heat_06pct", {"param_overrides.max_portfolio_heat": 0.060}),
        ("heat_065pct", {"param_overrides.max_portfolio_heat": 0.065}),
        ("heat_07pct", {"param_overrides.max_portfolio_heat": 0.070}),

        # Regime scaling variants keep the dynamic risk model from being one
        # size fits all without changing the signal or execution clocks.
        ("dynamic_scale_default", {
            "param_overrides.dynamic_risk_strong_trend_mult": 1.25,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.75,
        }),
        ("dynamic_scale_quality_130_070", {
            "param_overrides.dynamic_risk_strong_trend_mult": 1.30,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.70,
        }),
        ("dynamic_scale_aggressive_140_070", {
            "param_overrides.dynamic_risk_strong_trend_mult": 1.40,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.70,
        }),
        ("dynamic_scale_defensive_115_080", {
            "param_overrides.dynamic_risk_strong_trend_mult": 1.15,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.80,
        }),
        ("risk_230bp_heat060", {
            "fixed_qty": None,
            "param_overrides.base_risk_pct": 0.0230,
            "param_overrides.max_portfolio_heat": 0.060,
        }),
        ("risk_240bp_heat055", {
            "fixed_qty": None,
            "param_overrides.base_risk_pct": 0.0240,
            "param_overrides.max_portfolio_heat": 0.055,
        }),
    ]


def _risk_phase_3_addon_b_winner_lean_in() -> list[tuple[str, dict]]:
    """Enable pyramiding only after base exposure has survived strict gates."""
    return [
        ("addon_b_200r_010", {
            "flags.addon_b": True,
            "param_overrides.addon_b_r": 2.00,
            "param_overrides.addon_b_size_mult": 0.10,
        }),
        ("addon_b_250r_010", {
            "flags.addon_b": True,
            "param_overrides.addon_b_r": 2.50,
            "param_overrides.addon_b_size_mult": 0.10,
        }),
        ("addon_b_250r_015", {
            "flags.addon_b": True,
            "param_overrides.addon_b_r": 2.50,
            "param_overrides.addon_b_size_mult": 0.15,
        }),
        ("addon_b_300r_015", {
            "flags.addon_b": True,
            "param_overrides.addon_b_r": 3.00,
            "param_overrides.addon_b_size_mult": 0.15,
        }),
        ("addon_b_300r_020", {
            "flags.addon_b": True,
            "param_overrides.addon_b_r": 3.00,
            "param_overrides.addon_b_size_mult": 0.20,
        }),
        ("addon_b_350r_015", {
            "flags.addon_b": True,
            "param_overrides.addon_b_r": 3.50,
            "param_overrides.addon_b_size_mult": 0.15,
        }),

        # Add-on A stays separate from add-on B so winner scaling is only
        # adopted when it improves net return without damaging expectancy.
        ("addon_a_100r_025", {
            "param_overrides.addon_a_r": 1.00,
            "param_overrides.addon_a_size_mult": 0.25,
        }),
        ("addon_a_125r_025", {
            "param_overrides.addon_a_r": 1.25,
            "param_overrides.addon_a_size_mult": 0.25,
        }),
        ("addon_a_150r_050", {
            "param_overrides.addon_a_r": 1.50,
            "param_overrides.addon_a_size_mult": 0.50,
        }),
        ("disable_addon_a", {"flags.addon_a": False}),
        ("disable_addon_b", {"flags.addon_b": False}),
    ]


def _risk_phase_4_guardrails(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Final guardrail pass after size and add-ons have been established."""
    candidates = [
        ("risk_pct_minus25bp", _base_risk_pct_delta(prior_mutations, -0.0025)),
        ("risk_pct_minus10bp", _base_risk_pct_delta(prior_mutations, -0.0010)),
        ("risk_pct_plus10bp", _base_risk_pct_delta(prior_mutations, 0.0010)),
        ("risk_pct_plus25bp", _base_risk_pct_delta(prior_mutations, 0.0025)),
        ("heat_055pct", {"param_overrides.max_portfolio_heat": 0.055}),
        ("heat_06pct", {"param_overrides.max_portfolio_heat": 0.060}),
        ("heat_065pct", {"param_overrides.max_portfolio_heat": 0.065}),
        ("heat_075pct", {"param_overrides.max_portfolio_heat": 0.075}),
        ("dynamic_scale_default", {
            "param_overrides.dynamic_risk_strong_trend_mult": 1.25,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.75,
        }),
        ("dynamic_scale_quality_130_070", {
            "param_overrides.dynamic_risk_strong_trend_mult": 1.30,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.70,
        }),
        ("dynamic_scale_aggressive_140_070", {
            "param_overrides.dynamic_risk_strong_trend_mult": 1.40,
            "param_overrides.dynamic_risk_weak_trend_mult": 0.70,
        }),

        # Preserve the high expectancy while checking whether exits need to
        # become more defensive after size increases.
        ("be_trigger_075", {"param_overrides.be_trigger_r": 0.75}),
        ("be_trigger_100", {"param_overrides.be_trigger_r": 1.00}),
        ("tp1_frac_025", {"param_overrides.tp1_frac": 0.25}),
        ("tp1_frac_040", {"param_overrides.tp1_frac": 0.40}),
        ("tp2_r_200", {"param_overrides.tp2_r": 2.00}),
        ("tp2_r_250", {"param_overrides.tp2_r": 2.50}),
        ("floor_guardrail", {
            "param_overrides.profit_floor": {0.75: 0.0, 1.0: 0.20, 1.5: 0.55, 2.0: 1.0, 3.0: 1.8, 4.0: 2.8},
        }),
        ("floor_loose", {
            "param_overrides.profit_floor": {1.0: -0.10, 1.5: 0.30, 2.0: 0.80, 3.0: 1.6, 4.0: 2.6},
        }),
        ("max_hold_72h", {"param_overrides.max_hold_hours": 72}),
        ("max_hold_88h", {"param_overrides.max_hold_hours": 88}),
        ("max_hold_104h", {"param_overrides.max_hold_hours": 104}),
        ("max_hold_120h", {"param_overrides.max_hold_hours": 120}),
    ]
    candidates.extend(_dynamic_finetune(prior_mutations))
    return [(name, muts) for name, muts in candidates if muts]


def _fixed_qty_delta(prior_mutations: dict, delta: int) -> dict:
    current = prior_mutations.get("fixed_qty")
    if current is None:
        return {}
    try:
        qty = int(current) + delta
    except (TypeError, ValueError):
        return {}
    if qty < 1:
        return {}
    return {"fixed_qty": qty}


def _base_risk_pct_delta(prior_mutations: dict, delta: float) -> dict:
    current = prior_mutations.get("param_overrides.base_risk_pct")
    if current is None:
        return {}
    try:
        risk_pct = float(current) + delta
    except (TypeError, ValueError):
        return {}
    if risk_pct <= 0:
        return {}
    return {"param_overrides.base_risk_pct": round(risk_pct, 4)}


# ---------------------------------------------------------------------------
# Synchronized R2+: alpha-expansion phases
# ---------------------------------------------------------------------------

def _r10_phase_1_opportunity_surface() -> list[tuple[str, dict]]:
    """Unlock missing opportunity before tuning the current pullback."""
    return [
        # Universe expansion remains opt-in; default ETF set is unchanged.
        ("add_uso_sleeve", {"symbols": ["QQQ", "GLD", "USO"]}),

        # Controlled short alpha probes. Shorts are additive to the existing
        # long engine and must pass full portfolio gates.
        ("enable_gld_shorts", {"param_overrides.shorts_enabled_GLD": 1}),
        ("enable_qqq_shorts", {"param_overrides.shorts_enabled_QQQ": 1}),
        ("enable_etf_shorts", {
            "param_overrides.shorts_enabled_QQQ": 1,
            "param_overrides.shorts_enabled_GLD": 1,
        }),
        ("enable_etf_shorts_loose_safety", {
            "param_overrides.shorts_enabled_QQQ": 1,
            "param_overrides.shorts_enabled_GLD": 1,
            "flags.short_safety": False,
        }),

        # Bias/regime extraction: primary way to reduce FLAT-bias idle time.
        ("confirm_days_0", {"param_overrides.confirm_days_normal": 0}),
        ("fast_confirm_score_45", {"param_overrides.fast_confirm_score": 45}),
        ("fast_confirm_score_50", {"param_overrides.fast_confirm_score": 50}),
        ("fast_confirm_adx_18", {"param_overrides.fast_confirm_adx": 18}),
        ("adx_on_13", {"param_overrides.adx_on": 13}),
        ("adx_on_14", {"param_overrides.adx_on": 14}),
        ("adx_on_17", {"param_overrides.adx_on": 17}),
        ("adx_off_14", {"param_overrides.adx_off": 14}),
        ("adx_strong_25", {"param_overrides.adx_strong": 25}),
        ("adx_strong_35", {"param_overrides.adx_strong": 35}),
        ("pathc_loose", {
            "param_overrides.di_min": 7,
            "param_overrides.sep_min": 0.12,
            "param_overrides.adx_min_struct": 16,
        }),
        ("pathc_tight", {
            "param_overrides.di_min": 14,
            "param_overrides.sep_min": 0.28,
            "param_overrides.adx_min_struct": 22,
        }),
        ("ema_daily_faster", {
            "param_overrides.daily_ema_fast": 15,
            "param_overrides.daily_ema_slow": 45,
        }),
        ("ema_daily_slower", {
            "param_overrides.daily_ema_fast": 25,
            "param_overrides.daily_ema_slow": 65,
        }),
        ("rank_stop_first", {"param_overrides.rank_mode": "stop_first"}),
        ("rank_score_per_risk", {"param_overrides.rank_mode": "score_per_risk"}),
    ]


def _r10_phase_2_signal_geometry() -> list[tuple[str, dict]]:
    """Repair signal extraction and test discrimination around pullbacks."""
    return [
        # Pullback discrimination.
        ("pullback_momentum_filter", {"param_overrides.pullback_momentum_filter": True}),
        ("pb_momentum_strict", {
            "param_overrides.pullback_momentum_filter": True,
            "param_overrides.momentum_tolerance_atr": 0.00,
        }),
        ("pb_momentum_loose", {
            "param_overrides.pullback_momentum_filter": True,
            "param_overrides.momentum_tolerance_atr": 0.20,
        }),
        ("quality_gate_30", {
            "flags.quality_gate": True,
            "param_overrides.quality_gate_threshold": 3.0,
        }),
        ("quality_gate_40", {
            "flags.quality_gate": True,
            "param_overrides.quality_gate_threshold": 4.0,
        }),

        # Pullback geometry.
        ("pullback_lb_4", {"param_overrides.pullback_lookback": 4}),
        ("pullback_lb_6", {"param_overrides.pullback_lookback": 6}),
        ("pullback_lb_12", {"param_overrides.pullback_lookback": 12}),
        ("pullback_lb_16", {"param_overrides.pullback_lookback": 16}),
        ("touch_tol_035", {"param_overrides.pullback_touch_tolerance_atr": 0.35}),
        ("touch_tol_045", {"param_overrides.pullback_touch_tolerance_atr": 0.45}),
        ("touch_tol_070", {"param_overrides.pullback_touch_tolerance_atr": 0.70}),
        ("touch_pct_015", {"param_overrides.pullback_touch_tolerance_pct": 0.0015}),
        ("touch_pct_050", {"param_overrides.pullback_touch_tolerance_pct": 0.0050}),
        ("recov_trend_020", {"param_overrides.recovery_tolerance_atr_trend": 0.20}),
        ("recov_trend_045", {"param_overrides.recovery_tolerance_atr_trend": 0.45}),
        ("recov_trend_055", {"param_overrides.recovery_tolerance_atr_trend": 0.55}),
        ("recov_strong_060", {"param_overrides.recovery_tolerance_atr_strong": 0.60}),
        ("recov_strong_100", {"param_overrides.recovery_tolerance_atr_strong": 1.00}),
        ("ema_pull_normal_35", {"param_overrides.ema_pull_normal": 35}),
        ("ema_pull_normal_55", {"param_overrides.ema_pull_normal": 55}),
        ("ema_pull_strong_21", {"param_overrides.ema_pull_strong": 21}),
        ("ema_pull_strong_50", {"param_overrides.ema_pull_strong": 50}),

        # Breakout repair: arms exist, but the current retrace never converts.
        ("breakout_direct", {"param_overrides.breakout_direct_entry": True}),
        ("breakout_no_candle_gate", {"param_overrides.breakout_require_directional_candle": False}),
        ("breakout_retrace_20_60", {
            "param_overrides.breakout_retrace_entry_frac": 0.20,
            "param_overrides.breakout_retrace_limit_frac": 0.60,
        }),
        ("breakout_retrace_10_75", {
            "param_overrides.breakout_retrace_entry_frac": 0.10,
            "param_overrides.breakout_retrace_limit_frac": 0.75,
            "param_overrides.breakout_require_directional_candle": False,
        }),
    ]


def _r10_phase_3_execution_and_stops() -> list[tuple[str, dict]]:
    """Tune entry execution and initial risk geometry after signal discovery."""
    return [
        # Entry/fill mechanics.
        ("expiry_6h", {"param_overrides.order_expiry_hours": 6}),
        ("expiry_12h", {"param_overrides.order_expiry_hours": 12}),
        ("expiry_24h", {"param_overrides.order_expiry_hours": 24}),
        ("stop_market_entries", {"slippage.use_stop_market": True}),
        ("disable_slippage_abort", {"flags.slippage_abort": False}),
        ("slip_atr_050", {"param_overrides.max_entry_slip_atr": 0.50}),
        ("slip_atr_075", {"param_overrides.max_entry_slip_atr": 0.75}),
        ("limit_ticks_0", {"param_overrides.limit_ticks": 0}),
        ("limit_ticks_4", {"param_overrides.limit_ticks": 4}),
        ("limit_pct_0025", {"param_overrides.limit_pct": 0.0025}),

        # Initial stop geometry. These are broad structural probes, not
        # symbol-specific curve fits.
        ("daily_mult_18", {"param_overrides.daily_mult": 1.8}),
        ("daily_mult_24", {"param_overrides.daily_mult": 2.4}),
        ("hourly_mult_22", {"param_overrides.hourly_mult": 2.2}),
        ("hourly_mult_32", {"param_overrides.hourly_mult": 3.2}),
        ("trend_stop_tight_050", {"param_overrides.trend_stop_tightening": 0.50}),
        ("trend_stop_tight_075", {"param_overrides.trend_stop_tightening": 0.75}),
        ("trend_stop_tight_095", {"param_overrides.trend_stop_tightening": 0.95}),
        ("qqq_stop_tighter", {
            "param_overrides.daily_mult_QQQ": 1.9,
            "param_overrides.hourly_mult_QQQ": 2.4,
        }),
        ("gld_stop_tighter", {
            "param_overrides.daily_mult_GLD": 1.8,
            "param_overrides.hourly_mult_GLD": 2.3,
        }),
    ]


def _r10_phase_4_exits_addons_allocation(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Harvest MFE leakage, add-on edge, and allocator mis-ranking."""
    candidates = [
        # Stall/time decay variants around the current winner.
        ("early_stall_8h_mfe02", {
            "flags.early_stall_exit": True,
            "param_overrides.early_stall_check_hours": 8,
            "param_overrides.early_stall_mfe_threshold": 0.20,
        }),
        ("early_stall_16h_mfe03", {
            "flags.early_stall_exit": True,
            "param_overrides.early_stall_check_hours": 16,
            "param_overrides.early_stall_mfe_threshold": 0.30,
        }),
        ("early_stall_partial_025", {"param_overrides.early_stall_partial_frac": 0.25}),
        ("early_stall_partial_075", {"param_overrides.early_stall_partial_frac": 0.75}),
        ("disable_early_stall", {"flags.early_stall_exit": False}),
        ("full_stall_24h_mfe03", {
            "flags.stall_exit": True,
            "param_overrides.stall_check_hours": 24,
            "param_overrides.stall_mfe_threshold": 0.30,
        }),
        ("max_hold_64h", {"param_overrides.max_hold_hours": 64}),
        ("max_hold_96h", {"param_overrides.max_hold_hours": 96}),
        ("max_hold_120h", {"param_overrides.max_hold_hours": 120}),

        # MFE leakage and partial capture.
        ("be_trigger_050", {"param_overrides.be_trigger_r": 0.50}),
        ("be_trigger_100", {"param_overrides.be_trigger_r": 1.00}),
        ("tp1_r_090", {"param_overrides.tp1_r": 0.90}),
        ("tp1_r_115", {"param_overrides.tp1_r": 1.15}),
        ("tp1_frac_025", {"param_overrides.tp1_frac": 0.25}),
        ("tp1_frac_050", {"param_overrides.tp1_frac": 0.50}),
        ("tp2_r_175", {"param_overrides.tp2_r": 1.75}),
        ("tp2_r_200", {"param_overrides.tp2_r": 2.00}),
        ("floor_pre_be", {
            "param_overrides.profit_floor": {0.75: 0.0, 1.0: 0.20, 1.5: 0.55, 2.0: 1.0, 3.0: 1.8, 4.0: 2.8},
        }),
        ("floor_loose_low", {
            "param_overrides.profit_floor": {1.0: -0.10, 1.5: 0.30, 2.0: 0.80, 3.0: 1.6, 4.0: 2.6},
        }),
        ("chand_trigger_100", {"param_overrides.chandelier_trigger_r": 1.00}),
        ("chand_mult_25", {"param_overrides.chand_mult": 2.5}),
        ("chand_mult_35", {"param_overrides.chand_mult": 3.5}),

        # Add-ons and allocation.
        ("addon_a_100r", {"param_overrides.addon_a_r": 1.00}),
        ("addon_a_125r", {"param_overrides.addon_a_r": 1.25}),
        ("addon_a_size_050", {"param_overrides.addon_a_size_mult": 0.50}),
        ("addon_b_150r", {"param_overrides.addon_b_r": 1.50}),
        ("dynamic_sizing", {"fixed_qty": None}),
        ("heat_08pct", {"param_overrides.max_portfolio_heat": 0.08}),
        ("heat_10pct", {"param_overrides.max_portfolio_heat": 0.10}),
        ("rank_gld_first", {"param_overrides.rank_mode": "gld_first"}),
        ("rank_qqq_first", {"param_overrides.rank_mode": "qqq_first"}),
    ]
    candidates.extend(_dynamic_finetune(prior_mutations))
    return candidates


# ---------------------------------------------------------------------------
# R9 Phase 1: Structural Fixes (targeting diagnosed weaknesses)
# ---------------------------------------------------------------------------

def _r9_phase_1_structural() -> list[tuple[str, dict]]:
    """R9 structural candidates targeting post-audit collapse root causes.

    Root causes diagnosed:
      1. QQQ's 4 trades (0% WR, avg hold 335 bars) block shared capital
      2. $1/contract commission eats 63% of GLD gross profit at fixed_qty=10
      3. Independent-mode capital isolation was masking QQQ destroys value
    """
    candidates = []

    # --- Symbol selection (1) ---
    # Validate Phase 0 finding that GLD-only removes QQQ capital drag
    candidates.append(("gld_only", {"symbols": ["GLD"]}))

    # --- Max hold caps (3) ---
    # Prevent QQQ capital hostage (1140-bar hold = ~71 days)
    candidates.append(("max_hold_48h", {"param_overrides.max_hold_hours": 48}))
    candidates.append(("max_hold_80h", {"param_overrides.max_hold_hours": 80}))
    candidates.append(("max_hold_120h", {"param_overrides.max_hold_hours": 120}))

    # --- Sizing: reduce commission burden (3) ---
    # At fixed_qty=10, $1 commission = 63% of GLD gross.
    # Larger qty makes commission a smaller % of risk.
    candidates.append(("fixed_qty_20", {"fixed_qty": 20}))
    candidates.append(("fixed_qty_50", {"fixed_qty": 50}))
    candidates.append(("dynamic_sizing", {"fixed_qty": None}))  # equity-based

    # --- Stall aggressiveness: free capital faster (3) ---
    candidates.append(("stall_24h_mfe03", {
        "flags.early_stall_exit": True,
        "param_overrides.early_stall_check_hours": 24,
        "param_overrides.early_stall_mfe_threshold": 0.30,
    }))
    candidates.append(("stall_12h_mfe02", {
        "flags.early_stall_exit": True,
        "param_overrides.early_stall_check_hours": 12,
        "param_overrides.early_stall_mfe_threshold": 0.20,
    }))
    candidates.append(("stall_6h_mfe02", {
        "flags.early_stall_exit": True,
        "param_overrides.early_stall_check_hours": 6,
        "param_overrides.early_stall_mfe_threshold": 0.20,
    }))

    # --- Portfolio heat: allow more simultaneous positions (2) ---
    candidates.append(("heat_10pct", {"param_overrides.max_portfolio_heat": 0.10}))
    candidates.append(("heat_15pct", {"param_overrides.max_portfolio_heat": 0.15}))

    # --- Risk reduction: less heat per position (2) ---
    candidates.append(("base_risk_005", {"param_overrides.base_risk_pct": 0.005}))
    candidates.append(("base_risk_003", {"param_overrides.base_risk_pct": 0.003}))

    # --- Commission reduction (2) ---
    candidates.append(("comm_050", {"slippage.commission_per_contract": 0.50}))
    candidates.append(("comm_035", {"slippage.commission_per_contract": 0.35}))

    # --- Full stall re-enable (1) ---
    candidates.append(("reenable_full_stall", {"flags.stall_exit": True}))

    return candidates
