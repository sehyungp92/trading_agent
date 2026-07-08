"""Parameter space definition and Latin Hypercube Sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from strategies.swing.atrss.config import SYMBOLS, SYMBOL_CONFIGS


@dataclass
class ParamRange:
    """One optimizable parameter."""

    name: str
    low: float
    high: float
    step: float = 0.0  # 0 = continuous
    is_int: bool = False
    symbol: str = ""  # "" = global, "QQQ" = per-symbol

    def snap(self, value: float) -> float:
        """Snap value to grid if step > 0."""
        if self.step > 0:
            value = round(value / self.step) * self.step
        if self.is_int:
            value = round(value)
        return value


_BTC_SYMS = {"MBT", "BT", "BRR"}


def _build_per_symbol_ranges() -> list[ParamRange]:
    ranges: list[ParamRange] = []
    for sym in SYMBOLS:
        d_lo, h_lo = (2.0, 2.5) if sym in _BTC_SYMS else (1.5, 2.0)
        d_hi, h_hi = (4.5, 5.0) if sym in _BTC_SYMS else (3.5, 4.5)
        ranges.append(ParamRange(f"daily_mult_{sym}", d_lo, d_hi, 0.1, symbol=sym))
        ranges.append(ParamRange(f"hourly_mult_{sym}", h_lo, h_hi, 0.1, symbol=sym))
    return ranges


# ---------------------------------------------------------------------------
# Full parameter space (~30 params from backtesting.md Section 5.1)
# ---------------------------------------------------------------------------

PARAM_SPACE: list[ParamRange] = [
    # --- Core indicator periods ---
    ParamRange("daily_ema_fast", 15, 30, 1, True),
    ParamRange("daily_ema_slow", 40, 70, 1, True),
    ParamRange("adx_period", 10, 20, 1, True),
    ParamRange("atr_daily_period", 14, 30, 1, True),
    ParamRange("atr_hourly_period", 30, 72, 1, True),
    ParamRange("ema_mom_period", 15, 30, 1, True),
    ParamRange("ema_pull_strong", 25, 45, 1, True),
    ParamRange("ema_pull_normal", 40, 65, 1, True),
    ParamRange("donchian_period", 15, 30, 1, True),

    # --- Per-symbol ATR multipliers (generated from active SYMBOLS) ---
    *_build_per_symbol_ranges(),

    # --- Chandelier ---
    ParamRange("chand_mult", 2.0, 4.5, 0.1),

    # --- Regime thresholds ---
    ParamRange("adx_on", 16, 28, 1, True),
    # adx_off derived as adx_on - hysteresis_gap
    ParamRange("hysteresis_gap", 1, 5, 1, True),
    ParamRange("adx_strong", 25, 40, 1, True),

    # --- Entry thresholds ---
    ParamRange("score_reverse_min", 40, 80, 5, True),
    ParamRange("fast_confirm_score", 40, 80, 5, True),
    ParamRange("fast_confirm_adx", 16, 28, 1, True),

    # --- Stop management ---
    ParamRange("be_atr_offset", 0.05, 0.25, 0.01),

    # --- Risk sizing ---
    ParamRange("base_risk_pct", 0.005, 0.02, 0.001),

    # --- Stop management triggers ---
    ParamRange("be_trigger_r", 1.0, 2.0, 0.1),
    ParamRange("chandelier_trigger_r", 1.5, 2.5, 0.1),

    # --- Cooldown hours per regime ---
    ParamRange("cooldown_strong", 2, 8, 2, True),
    ParamRange("cooldown_trend", 6, 18, 2, True),
    ParamRange("cooldown_range", 12, 36, 4, True),

    # --- Re-entry / voucher ---
    ParamRange("voucher_valid_hours", 8, 48, 4, True),

    # --- Trend confirmation ---
    ParamRange("confirm_days_normal", 1, 3, 1, True),

    # --- Breakout regime gate ---
    ParamRange("adx_slope_gate", -3.0, 1.0, 0.5),

    # --- Execution parameters ---
    ParamRange("limit_ticks", 1, 5, 1, True),
    ParamRange("limit_pct", 0.0005, 0.0025, 0.0005),
]


def latin_hypercube_sample(
    param_space: list[ParamRange],
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate n_samples using Latin Hypercube Sampling.

    Returns a list of dicts, each mapping param name to sampled value.
    """
    rng = np.random.default_rng(seed)
    n_params = len(param_space)

    # LHS: divide each dimension into n_samples equal intervals
    # Sample one point per interval, then shuffle columns
    samples = np.zeros((n_samples, n_params))

    for j, p in enumerate(param_space):
        # Create stratified intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Sample uniformly within each interval
        points = rng.uniform(intervals[:-1], intervals[1:])
        # Shuffle
        rng.shuffle(points)
        # Scale to parameter range
        samples[:, j] = p.low + points * (p.high - p.low)

    # Snap to grid
    result = []
    for i in range(n_samples):
        sample = {}
        for j, p in enumerate(param_space):
            sample[p.name] = p.snap(samples[i, j])
        result.append(sample)

    return result


def params_to_overrides(sample: dict[str, float]) -> dict[str, float]:
    """Convert a sample dict to BacktestConfig.param_overrides format.

    Handles derived parameters:
    - adx_off = adx_on - hysteresis_gap
    - Per-symbol params like daily_mult_QQQ
    - Module-level constants (be_trigger_r, chandelier_trigger_r, cooldown_*,
      voucher_valid_hours, confirm_days_normal, adx_slope_gate) are passed
      through as overrides and applied by _AblationPatch / engine override logic.
    """
    overrides = {}

    for key, value in sample.items():
        # Derived: ADX_OFF
        if key == "adx_on":
            overrides["adx_on"] = value
            gap = sample.get("hysteresis_gap", 2)
            overrides["adx_off"] = value - gap
        elif key == "hysteresis_gap":
            continue  # Handled with adx_on
        else:
            overrides[key] = value

    return overrides
