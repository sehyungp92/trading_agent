"""Helix R2 exit-alpha experiment definitions.

Categories:
  TRAILING_REPLAY     -- R1 trailing experiments (now functional after _AblationPatch fix)
  STOP_REPLAY         -- R1 stop placement experiments (now functional)
  PARTIALS_BE_REPLAY  -- R1 partials/BE experiments (now functional)
  JOINT_TRAILING      -- LHS-sampled 6D compound trailing mutations
  R_BAND_TRAILING     -- R-dependent trailing profiles (low/mid/high R bands)
  CLASS_SPECIFIC_D    -- Class D momentum tightening
  CLASS_SPECIFIC_B    -- Class B divergence widening
  CLASS_COMPOUND      -- Combined class D + B mutations
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Replay of R1 experiments that were silently broken (stops.py not patched)
# ---------------------------------------------------------------------------

def _trailing_replay_experiments() -> list[tuple[str, dict]]:
    """12 TRAILING experiments from R1 -- now actually functional."""
    return [
        ("r2_trail_base_30", {"param_overrides.TRAIL_BASE": 3.0}),
        ("r2_trail_base_35", {"param_overrides.TRAIL_BASE": 3.5}),
        ("r2_trail_base_50", {"param_overrides.TRAIL_BASE": 5.0}),
        ("r2_trail_min_15", {"param_overrides.TRAIL_MIN": 1.5}),
        ("r2_trail_min_25", {"param_overrides.TRAIL_MIN": 2.5}),
        ("r2_trail_r_div_30", {"param_overrides.TRAIL_R_DIV": 3.0}),
        ("r2_trail_r_div_40", {"param_overrides.TRAIL_R_DIV": 4.0}),
        ("r2_trail_r_div_70", {"param_overrides.TRAIL_R_DIV": 7.0}),
        ("r2_trail_mom_bonus_025", {"param_overrides.TRAIL_MOM_BONUS": 0.25}),
        ("r2_trail_mom_bonus_075", {"param_overrides.TRAIL_MOM_BONUS": 0.75}),
        ("r2_trail_delay_2", {"param_overrides.TRAIL_PROFIT_DELAY_BARS": 2}),
        ("r2_trail_delay_6", {"param_overrides.TRAIL_PROFIT_DELAY_BARS": 6}),
    ]


def _stop_replay_experiments() -> list[tuple[str, dict]]:
    """10 STOP_PLACEMENT experiments from R1 -- now actually functional."""
    return [
        ("r2_stop_1h_std_060", {"param_overrides.STOP_1H_STD": 0.60}),
        ("r2_stop_1h_std_070", {"param_overrides.STOP_1H_STD": 0.70}),
        ("r2_stop_1h_highvol_090", {"param_overrides.STOP_1H_HIGHVOL": 0.90}),
        ("r2_stop_1h_highvol_100", {"param_overrides.STOP_1H_HIGHVOL": 1.00}),
        ("r2_stop_4h_mult_060", {"param_overrides.STOP_4H_MULT": 0.60}),
        ("r2_stop_4h_mult_090", {"param_overrides.STOP_4H_MULT": 0.90}),
        ("r2_stop_be_offset_010", {"param_overrides.BE_ATR1H_OFFSET": 0.10}),
        ("r2_stop_be_offset_020", {"param_overrides.BE_ATR1H_OFFSET": 0.20}),
        ("r2_stop_emergency_neg15", {"param_overrides.EMERGENCY_STOP_R": -1.5}),
        ("r2_stop_emergency_neg25", {"param_overrides.EMERGENCY_STOP_R": -2.5}),
    ]


def _partials_be_replay_experiments() -> list[tuple[str, dict]]:
    """10 PARTIALS_BE experiments from R1 -- R_BE, R_BE_1H imported by stops.py."""
    return [
        ("r2_be_075", {"param_overrides.R_BE": 0.75}),
        ("r2_be_125", {"param_overrides.R_BE": 1.25}),
        ("r2_be_1h_050", {"param_overrides.R_BE_1H": 0.50}),
        ("r2_be_1h_100", {"param_overrides.R_BE_1H": 1.00}),
        ("r2_partial_2p5_frac_040", {"param_overrides.PARTIAL_2P5_FRAC": 0.40}),
        ("r2_partial_2p5_frac_060", {"param_overrides.PARTIAL_2P5_FRAC": 0.60}),
        ("r2_partial_5_frac_020", {"param_overrides.PARTIAL_5_FRAC": 0.20}),
        ("r2_partial_5_frac_033", {"param_overrides.PARTIAL_5_FRAC": 0.33}),
        ("r2_partial_2p5_at_3r", {"param_overrides.R_PARTIAL_2P5": 3.0}),
        ("r2_partial_5_at_6r", {"param_overrides.R_PARTIAL_5": 6.0}),
    ]


# ---------------------------------------------------------------------------
# Joint trailing LHS -- 6D compound mutations
# ---------------------------------------------------------------------------

def generate_joint_trailing_experiments(n_samples: int = 80) -> list[tuple[str, dict]]:
    """LHS-sample the 6D trailing space."""
    rng = np.random.default_rng(42)
    params = [
        ("param_overrides.TRAIL_BASE", 2.5, 5.0),
        ("param_overrides.TRAIL_R_DIV", 3.0, 8.0),
        ("param_overrides.TRAIL_STALL_ONSET", 3, 12),
        ("param_overrides.TRAIL_FADE_PENALTY", 0.3, 1.2),
        ("param_overrides.TRAIL_STALL_RATE", 0.04, 0.15),
        ("param_overrides.TRAIL_FADE_FLOOR", 1.0, 2.5),
    ]
    n_dims = len(params)
    # Latin hypercube: stratified random in each dimension
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            lo, hi = params[d][1], params[d][2]
            samples[perm[i], d] = lo + (hi - lo) * (i + rng.random()) / n_samples

    experiments = []
    for i in range(n_samples):
        muts: dict = {}
        for d, (key, _lo, _hi) in enumerate(params):
            val = samples[i, d]
            if "ONSET" in key:
                val = int(round(val))
            else:
                val = round(float(val), 3)
            muts[key] = val
        experiments.append((f"joint_trail_{i:03d}", muts))
    return experiments


# ---------------------------------------------------------------------------
# R-band trailing profiles
# ---------------------------------------------------------------------------

def _r_band_trailing_experiments() -> list[tuple[str, dict]]:
    """~25 R-band trailing profile experiments."""
    threshold_combos = [
        (1.5, 4.0), (2.0, 4.0), (2.0, 5.0), (2.5, 5.0), (1.5, 5.0),
    ]
    low_r_configs = [
        (2.5, 3.0), (3.0, 3.0), (3.0, 4.0), (3.5, 4.0), (2.5, 4.0),
    ]
    high_r_configs = [
        (4.0, 6.0), (4.5, 6.0), (4.5, 8.0), (5.0, 8.0), (4.0, 8.0),
    ]

    experiments = []
    for mid, high in threshold_combos:
        for (low_base, low_div), (hi_base, hi_div) in zip(low_r_configs, high_r_configs):
            name = f"rband_m{mid}_h{high}_lb{low_base}d{low_div}_hb{hi_base}d{hi_div}"
            name = name.replace(".", "p")
            muts = {
                "param_overrides.R_BAND_MID": mid,
                "param_overrides.R_BAND_HIGH": high,
                "param_overrides.TRAIL_BASE_LOW_R": low_base,
                "param_overrides.TRAIL_R_DIV_LOW_R": low_div,
                "param_overrides.TRAIL_BASE_HIGH_R": hi_base,
                "param_overrides.TRAIL_R_DIV_HIGH_R": hi_div,
            }
            experiments.append((name, muts))
    return experiments


# ---------------------------------------------------------------------------
# Class-specific trailing
# ---------------------------------------------------------------------------

def _class_d_tightening_experiments() -> list[tuple[str, dict]]:
    """~12 Class D momentum tightening experiments."""
    experiments = []
    # Single-param variants
    for onset in [3, 4, 5]:
        experiments.append((
            f"cls_d_stall_{onset}",
            {"param_overrides.TRAIL_STALL_ONSET_CLASS_D": onset},
        ))
    for penalty in [0.8, 1.0, 1.2]:
        name = f"cls_d_fade_{penalty}".replace(".", "p")
        experiments.append((name, {"param_overrides.TRAIL_FADE_PENALTY_CLASS_D": penalty}))
    for min_r in [0.3, 0.5]:
        name = f"cls_d_fade_min_r_{min_r}".replace(".", "p")
        experiments.append((name, {"param_overrides.TRAIL_FADE_MIN_R_CLASS_D": min_r}))
    for base in [2.5, 3.0, 3.5]:
        name = f"cls_d_base_{base}".replace(".", "p")
        experiments.append((name, {
            "param_overrides.TRAIL_BASE_CLASS_D": base,
            "param_overrides.TRAIL_R_DIV_CLASS_D": 4.0,
        }))
    # Compound: stall + fade
    experiments.append(("cls_d_compound_tight", {
        "param_overrides.TRAIL_STALL_ONSET_CLASS_D": 4,
        "param_overrides.TRAIL_FADE_PENALTY_CLASS_D": 1.0,
        "param_overrides.TRAIL_FADE_MIN_R_CLASS_D": 0.3,
    }))
    return experiments


def _class_b_widening_experiments() -> list[tuple[str, dict]]:
    """~4 Class B divergence widening experiments."""
    experiments = []
    for base in [4.0, 4.5, 5.0]:
        name = f"cls_b_base_{base}".replace(".", "p")
        experiments.append((name, {
            "param_overrides.TRAIL_BASE_CLASS_B": base,
            "param_overrides.TRAIL_R_DIV_CLASS_B": 6.0,
        }))
    for onset in [10, 12]:
        experiments.append((
            f"cls_b_stall_{onset}",
            {"param_overrides.TRAIL_STALL_ONSET_CLASS_B": onset},
        ))
    return experiments


def _class_compound_experiments() -> list[tuple[str, dict]]:
    """~4 combined Class D tightening + Class B widening."""
    return [
        ("cls_db_compound_1", {
            "param_overrides.TRAIL_STALL_ONSET_CLASS_D": 4,
            "param_overrides.TRAIL_FADE_PENALTY_CLASS_D": 1.0,
            "param_overrides.TRAIL_BASE_CLASS_B": 4.5,
            "param_overrides.TRAIL_R_DIV_CLASS_B": 6.0,
        }),
        ("cls_db_compound_2", {
            "param_overrides.TRAIL_STALL_ONSET_CLASS_D": 3,
            "param_overrides.TRAIL_FADE_MIN_R_CLASS_D": 0.3,
            "param_overrides.TRAIL_BASE_CLASS_B": 5.0,
            "param_overrides.TRAIL_STALL_ONSET_CLASS_B": 12,
        }),
        ("cls_db_compound_3", {
            "param_overrides.TRAIL_BASE_CLASS_D": 3.0,
            "param_overrides.TRAIL_R_DIV_CLASS_D": 4.0,
            "param_overrides.TRAIL_STALL_ONSET_CLASS_D": 5,
            "param_overrides.TRAIL_BASE_CLASS_B": 4.0,
            "param_overrides.TRAIL_STALL_ONSET_CLASS_B": 10,
        }),
        ("cls_db_compound_4", {
            "param_overrides.TRAIL_STALL_ONSET_CLASS_D": 4,
            "param_overrides.TRAIL_FADE_PENALTY_CLASS_D": 0.8,
            "param_overrides.TRAIL_FADE_MIN_R_CLASS_D": 0.5,
            "param_overrides.TRAIL_BASE_CLASS_B": 4.5,
            "param_overrides.TRAIL_R_DIV_CLASS_B": 6.0,
            "param_overrides.TRAIL_STALL_ONSET_CLASS_B": 10,
        }),
    ]


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "TRAILING_REPLAY": _trailing_replay_experiments,
    "STOP_REPLAY": _stop_replay_experiments,
    "PARTIALS_BE_REPLAY": _partials_be_replay_experiments,
    "JOINT_TRAILING": lambda: generate_joint_trailing_experiments(80),
    "R_BAND_TRAILING": _r_band_trailing_experiments,
    "CLASS_SPECIFIC_D": _class_d_tightening_experiments,
    "CLASS_SPECIFIC_B": _class_b_widening_experiments,
    "CLASS_COMPOUND": _class_compound_experiments,
}


def get_category_experiments(categories: list[str]) -> list[tuple[str, dict]]:
    """Return experiments for the given category names."""
    result: list[tuple[str, dict]] = []
    for cat in categories:
        fn = _CATEGORY_MAP.get(cat)
        if fn is not None:
            result.extend(fn())
    return result
