"""VdubusNQ v4.0 parameter space definition.

Reuses ParamRange, latin_hypercube_sample from the shared param_space module.
Organized per backtesting_3.md Section 6: core entry, momentum, stops,
exits, decision gate, and risk parameters.
"""
from __future__ import annotations

from backtests.momentum.optimization.param_space import ParamRange


# ---------------------------------------------------------------------------
# VdubusNQ parameter space (~25 params)
# ---------------------------------------------------------------------------

VDUBUS_PARAM_SPACE: list[ParamRange] = [
    # --- Core entry parameters ---
    ParamRange("touch_lookback_15m", 6, 12, 1, True),
    ParamRange("vwap_cap_core", 0.75, 0.95, 0.05),
    ParamRange("vwap_cap_open_eve", 0.60, 0.85, 0.05),
    ParamRange("extension_skip_atr", 0.8, 1.5, 0.1),
    ParamRange("retest_tol_atr", 0.15, 0.35, 0.05),

    # --- Momentum / slope ---
    ParamRange("mom_n", 30, 80, 5, True),
    ParamRange("floor_pct", 0.20, 0.30, 0.01),
    ParamRange("slope_lb", 2, 4, 1, True),

    # --- Stops ---
    ParamRange("min_stop_points", 10, 30, 2, True),
    ParamRange("max_stop_points", 80, 160, 10, True),
    ParamRange("atr_stop_mult", 1.2, 1.6, 0.05),

    # --- Exits ---
    ParamRange("partial_pct", 0.25, 0.50, 0.05),
    ParamRange("vwap_fail_consec", 2, 3, 1, True),
    ParamRange("stale_bars_15m", 12, 20, 1, True),
    ParamRange("trail_lookback_15m", 12, 20, 1, True),
    ParamRange("trail_mult_base", 2.0, 4.0, 0.2),

    # --- Decision gate ---
    ParamRange("hold_weekday_r", 0.8, 1.2, 0.1),
    ParamRange("hold_friday_r", 1.2, 2.0, 0.1),
    ParamRange("weekend_lock_r", 0.25, 0.75, 0.05),

    # --- Risk ---
    ParamRange("base_risk_pct", 0.0020, 0.0035, 0.0005),
    ParamRange("class_mult_nopred", 0.60, 0.80, 0.05),
    ParamRange("session_mult_eve", 0.50, 1.00, 0.10),
    ParamRange("heat_cap_mult", 1.25, 1.50, 0.05),
]


def vdubus_params_to_overrides(sample: dict[str, float]) -> dict[str, float]:
    """Convert a VdubusNQ sample dict to config overrides.

    VdubusNQ params map 1:1 to strategy_3.config module-level constants.
    Names are lowercased versions of the constants.
    """
    overrides = {}
    for key, value in sample.items():
        overrides[key] = value
    return overrides
