"""NQDTC v2.0 parameter space definition.

Reuses ParamRange, latin_hypercube_sample from the shared param_space module.
Organized into Tier A (signal/qualification), Tier B (entries/execution),
Tier C (risk/trailing/exit management).
"""
from __future__ import annotations

from backtests.momentum.optimization.param_space import ParamRange


# ---------------------------------------------------------------------------
# NQDTC parameter space (~30 params)
# ---------------------------------------------------------------------------

NQDTC_PARAM_SPACE: list[ParamRange] = [
    # --- Tier A: Signal / Qualification parameters ---
    # Displacement quantile (sweep param)
    ParamRange("q_disp", 0.55, 0.85, 0.05),

    # Regime slope threshold
    ParamRange("k_slope", 0.05, 0.20, 0.01),

    # ADX trending/range thresholds
    ParamRange("adx_trending", 20, 30, 1, True),
    ParamRange("adx_range", 15, 25, 1, True),

    # Evidence scorecard
    ParamRange("score_normal", 1.5, 3.0, 0.25),
    ParamRange("score_degraded", 2.5, 4.0, 0.25),
    ParamRange("rvol_score_thresh", 0.8, 1.6, 0.1),

    # Squeeze quantiles
    ParamRange("squeeze_good_quantile", 0.10, 0.30, 0.05),
    ParamRange("squeeze_loose_quantile", 0.45, 0.75, 0.05),

    # Breakout quality reject
    ParamRange("breakout_reject_range_mult", 1.3, 2.5, 0.1),
    ParamRange("breakout_reject_rvol", 1.5, 3.0, 0.25),

    # --- Tier B: Entry / Execution parameters ---
    # Entry A offsets and TTL
    ParamRange("a1_offset_ticks", 1, 4, 1, True),
    ParamRange("a2_buffer_ticks", 1, 4, 1, True),
    ParamRange("a_ttl_5m_bars", 2, 6, 1, True),

    # Entry B sweep depth
    ParamRange("b_sweep_depth_atr", 0.10, 0.35, 0.05),

    # Entry C
    ParamRange("c_hold_bars", 1, 4, 1, True),
    ParamRange("c_entry_offset_atr", 0.02, 0.10, 0.01),

    # DIRTY thresholds
    ParamRange("dirty_depth_frac", 0.25, 0.55, 0.05),
    ParamRange("dirty_reset_shift_atr", 0.30, 0.70, 0.05),

    # Chop thresholds
    ParamRange("chop_vwap_cross_1", 3, 8, 1, True),
    ParamRange("chop_vwap_cross_2", 6, 12, 1, True),
    ParamRange("chop_size_mult", 0.50, 0.90, 0.05),

    # --- Tier C: Risk and position management ---
    # Base risk
    ParamRange("base_risk_pct", 0.0020, 0.0050, 0.0005),

    # Chandelier parameters
    ParamRange("chandelier_mult_tier0", 2.0, 4.0, 0.2),
    ParamRange("chandelier_mult_tier1", 1.6, 3.0, 0.2),
    ParamRange("chandelier_mult_tier2", 1.2, 2.4, 0.2),

    # Stale exit
    ParamRange("stale_bars_normal", 8, 18, 2, True),
    ParamRange("stale_r_threshold", 0.3, 0.8, 0.1),

    # Daily stop
    ParamRange("daily_stop_r", -4.0, -1.5, 0.5),

    # Friction cap
    ParamRange("friction_cap", 0.05, 0.15, 0.01),
]


def nqdtc_params_to_overrides(sample: dict[str, float]) -> dict[str, float]:
    """Convert an NQDTC sample dict to config overrides.

    NQDTC params map 1:1 to strategy_2.config module-level constants.
    Names are lowercased versions of the constants.
    """
    overrides = {}
    for key, value in sample.items():
        overrides[key] = value
    return overrides
