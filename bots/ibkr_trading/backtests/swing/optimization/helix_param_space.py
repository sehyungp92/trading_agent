"""Helix-specific parameter space definition.

Reuses ParamRange, latin_hypercube_sample, params_to_overrides from the
strategy-agnostic param_space module.
"""
from __future__ import annotations

from backtests.swing.optimization.param_space import ParamRange


def _helix_symbols() -> list[str]:
    try:
        from strategies.swing.akc_helix.config import SYMBOLS
        return list(SYMBOLS)
    except ImportError:
        return ["QQQ", "GLD"]


_BTC_SYMS = {"MBT", "BT", "BRR"}


def _build_per_symbol_chandelier() -> list[ParamRange]:
    ranges: list[ParamRange] = []
    for sym in _helix_symbols():
        lo, hi = (15, 25) if sym in _BTC_SYMS else (20, 40)
        ranges.append(ParamRange(
            f"chandelier_lookback_{sym}", lo, hi, 1, True, symbol=sym,
        ))
    return ranges


# ---------------------------------------------------------------------------
# Helix parameter space (~35 params)
# ---------------------------------------------------------------------------

HELIX_PARAM_SPACE: list[ParamRange] = [
    # --- MACD periods (spec s3) ---
    ParamRange("macd_fast", 5, 12, 1, True),
    ParamRange("macd_slow", 15, 30, 1, True),
    ParamRange("macd_signal", 3, 9, 1, True),

    # --- Daily indicators ---
    ParamRange("daily_ema_fast", 15, 30, 1, True),
    ParamRange("daily_ema_slow", 40, 70, 1, True),
    ParamRange("atr_daily_period", 10, 20, 1, True),

    # --- Stop multipliers (spec s10) ---
    ParamRange("stop_4h_mult", 0.50, 1.00, 0.05),
    ParamRange("stop_1h_std", 0.30, 0.70, 0.05),
    ParamRange("stop_1h_highvol", 0.50, 1.00, 0.05),

    # --- R milestones (spec s13) ---
    ParamRange("r_be", 0.75, 1.50, 0.25),
    ParamRange("r_partial_2p5", 2.0, 3.5, 0.25),
    ParamRange("r_partial_5", 4.0, 6.0, 0.50),

    # --- Trailing (spec s14) ---
    ParamRange("trail_base", 3.0, 5.0, 0.25),
    ParamRange("trail_min", 1.5, 2.5, 0.25),
    ParamRange("trail_max", 3.0, 5.0, 0.25),
    ParamRange("trail_r_div", 3.0, 7.0, 0.50),
    ParamRange("trail_mom_bonus", 0.25, 0.75, 0.25),
    ParamRange("trail_chop_penalty", 0.10, 0.50, 0.10),
    ParamRange("trail_flip_penalty", 0.25, 0.75, 0.25),

    # --- Add-on R thresholds (spec s15) ---
    ParamRange("add_4h_r", 0.75, 1.50, 0.25),
    ParamRange("add_1h_r", 1.0, 2.0, 0.25),

    # --- Risk sizing ---
    ParamRange("base_risk_pct", 0.003, 0.010, 0.001),

    # --- Corridor caps ---
    ParamRange("corridor_cap_chop", 1.0, 1.8, 0.1),
    ParamRange("corridor_cap_trend", 1.2, 2.0, 0.1),
    ParamRange("corridor_cap_other", 1.1, 1.8, 0.1),

    # --- Stale exit ---
    ParamRange("stale_1h_bars", 30, 60, 5, True),
    ParamRange("stale_4h_bars", 10, 25, 1, True),
    ParamRange("stale_r_thresh", 0.3, 0.8, 0.1),

    # --- Per-symbol chandelier lookback ---
    *_build_per_symbol_chandelier(),
]


def helix_params_to_overrides(sample: dict[str, float]) -> dict[str, float]:
    """Convert a Helix sample dict to config overrides.

    Helix params map 1:1 to strategy_2.config module-level constants.
    Per-symbol params like chandelier_lookback_QQQ are passed through.
    """
    overrides = {}
    for key, value in sample.items():
        overrides[key] = value
    return overrides
