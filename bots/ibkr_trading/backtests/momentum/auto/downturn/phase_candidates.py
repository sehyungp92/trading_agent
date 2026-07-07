"""Downturn per-phase candidate selection from experiment categories.

3-phase correction-capture specialist optimization:

Phase 1: Alpha extraction -- correction coverage, signal breadth, breakdown revival
Phase 2: Entry discrimination -- broker-safe entry mechanics and low-MFE suppression
Phase 3: Trade management -- sizing/risk polish and accepted-parameter finetuning
"""
from __future__ import annotations

from backtests.momentum.auto.downturn.experiment_categories import (
    get_category_experiments,
)


def get_phase_candidates(
    phase: int,
    prior_mutations: dict | None = None,
    suggested_experiments: list[tuple[str, dict]] | None = None,
) -> list[tuple[str, dict]]:
    """Get experiment candidates for a specific phase.

    Args:
        phase: Optimization phase (1-3)
        prior_mutations: Cumulative mutations from prior phases (for phase 3 finetune)
        suggested_experiments: Analyzer-suggested experiments (prepended)
    """
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    elif phase == 3:
        candidates = _phase_3_candidates(prior_mutations or {})
    else:
        candidates = []

    # Prepend suggested experiments from analyzer
    if suggested_experiments:
        # Deduplicate by name
        existing_names = {name for name, _ in candidates}
        new = [(n, m) for n, m in suggested_experiments if n not in existing_names]
        candidates = new + candidates

    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """Phase 1: Expand real correction alpha without broad no-op sweeps."""
    return get_category_experiments(["R3_ALPHA"])


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """Phase 2: Suppress weak entries and improve MFE capture."""
    return get_category_experiments(["R3_ENTRY", "R3_EXIT"])


def _phase_3_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Phase 3: Finetune accepted params, sizing, and engine contribution."""
    candidates: list[tuple[str, dict]] = []

    # A. Auto-generated finetune from accepted numeric params (+/-10%)
    candidates.extend(_finetune_candidates(prior_mutations))

    # B. Sizing/risk experiments
    candidates.extend(get_category_experiments(["R3_RISK"]))

    # C. Engine ablation (verify each engine still adds value post-regime-filtering)
    candidates.extend([
        ("r2_disable_reversal", {"flags.reversal_engine": False}),
        ("r2_disable_breakdown", {"flags.breakdown_engine": False}),
        ("r2_disable_fade", {"flags.fade_engine": False}),
        ("r2_enable_momentum", {"flags.momentum_signal": True}),
    ])

    return candidates


def _finetune_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Generate +/-10% variants around accepted numeric mutations."""
    candidates: list[tuple[str, dict]] = []

    if not prior_mutations:
        return candidates

    allowlist = {
        "param_overrides.base_risk_pct",
        "param_overrides.regime_mult_counter",
        "param_overrides.regime_mult_neutral",
        "param_overrides.regime_mult_range",
        "param_overrides.regime_mult_aligned",
        "param_overrides.regime_mult_emerging",
        "param_overrides.chandelier_lookback",
        "param_overrides.profit_floor_r_threshold",
        "param_overrides.be_trigger_r",
        "param_overrides.be_stop_buffer_mult",
        "param_overrides.min_hold_bars",
        "param_overrides.tp1_r_aligned",
        "param_overrides.tp1_r_emerging",
        "param_overrides.fade_stop_atr_mult",
        "param_overrides.entry_buffer_ticks",
        "param_overrides.entry_ttl_bars",
        "param_overrides.entry_limit_offset_ticks",
        "param_overrides.friction_min_atr_pctl",
        "param_overrides.vwap_cap_core",
        "param_overrides.vwap_cap_extended",
        "param_overrides.correction_sizing_mult",
        "param_overrides.non_correction_sizing_mult",
    }

    for key, value in prior_mutations.items():
        if key not in allowlist:
            continue
        if not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue

        for pct_label, pct in [("m10", 0.90), ("p10", 1.10)]:
            new_val = value * pct
            if isinstance(value, int):
                new_val = int(round(new_val))
                if new_val == value:
                    continue
            else:
                new_val = round(new_val, 6)

            name = f"finetune_{key.replace('.', '_')}_{pct_label}"
            candidates.append((name, {key: new_val}))

    return candidates
