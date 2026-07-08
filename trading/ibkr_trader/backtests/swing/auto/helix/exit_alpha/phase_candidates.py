"""Helix R2 per-phase candidate selection.

Phase 1: TRAILING_REPLAY + STOP_REPLAY + PARTIALS_BE_REPLAY + JOINT_TRAILING (~120 candidates)
Phase 2: R_BAND_TRAILING (~25 candidates)
Phase 3: CLASS_SPECIFIC_D + CLASS_SPECIFIC_B + CLASS_COMPOUND (~20 candidates)
Phase 4: FINETUNE of accepted mutations (~30+ auto-generated)
"""
from __future__ import annotations

from .experiment_categories import get_category_experiments


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
        candidates = _phase_4_candidates(prior_mutations or {})
    else:
        candidates = []

    if suggested_experiments:
        existing_names = {name for name, _ in candidates}
        for name, muts in suggested_experiments:
            if name not in existing_names:
                candidates.append((name, muts))

    return candidates


def _phase_1_candidates() -> list[tuple[str, dict]]:
    """Re-tested singles + joint trailing LHS + partials/BE replay."""
    return get_category_experiments([
        "TRAILING_REPLAY", "STOP_REPLAY", "PARTIALS_BE_REPLAY", "JOINT_TRAILING",
    ])


def _phase_2_candidates() -> list[tuple[str, dict]]:
    """R-band trailing profiles."""
    return get_category_experiments(["R_BAND_TRAILING"])


def _phase_3_candidates() -> list[tuple[str, dict]]:
    """Class-specific trailing experiments."""
    return get_category_experiments([
        "CLASS_SPECIFIC_D", "CLASS_SPECIFIC_B", "CLASS_COMPOUND",
    ])


def _phase_4_candidates(prior_mutations: dict) -> list[tuple[str, dict]]:
    """Auto-generated finetune of prior accepted mutations."""
    if not prior_mutations:
        return []

    finetune: list[tuple[str, dict]] = []
    for key, value in prior_mutations.items():
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue

        for pct_label, pct in [("m20", 0.80), ("m10", 0.90), ("p10", 1.10), ("p20", 1.20)]:
            new_val: int | float
            if isinstance(value, int):
                new_val = max(1, round(value * pct))
                if new_val == value:
                    continue
            else:
                new_val = round(value * pct, 4)

            short_key = key.rsplit(".", 1)[-1].lower()
            name = f"ft_{short_key}_{pct_label}"
            finetune.append((name, {key: new_val}))

    return finetune
