"""Leading Indicator Scanner (Layer 2) -- rate-of-change features for early regime shift detection.

Provides 2-4 week advance warning of regime transitions by tracking 6 derivative
features. Modifies leverage only (not allocations), preserving the HMM's proven
allocation signal. Feature weights are empirically-calibrated, NOT optimized.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MetaConfig
from .utils import ewma_vol, rolling_zscore, sigmoid


@dataclass(frozen=True)
class ShiftSignal:
    regime_shift_probability: float   # 0-1, logistic output (0.5 = neutral)
    shift_direction: str              # "risk_off", "risk_on", or "neutral"
    dominant_leading_indicator: str    # feature with highest abs weighted score
    estimated_lead_weeks: int         # fixed at 3 (empirical median)
    raw_scores: dict                  # {feature_name: float} per-feature z-scores


def build_scanner_features(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cfg: MetaConfig,
) -> pd.DataFrame:
    """Precompute 6 leading indicator features on daily index, then z-score.

    Same precompute pattern as compute_crisis_prob in inference.py.
    Returns DataFrame indexed on daily dates with z-scored feature columns.
    """
    raw = {}

    # 1. Credit spread momentum (widening = risk-off)
    raw["credit_spread_mom"] = market_df["SPREAD"].diff(20)

    # 2. Yield curve velocity (flattening = risk-off, so negate)
    raw["yield_curve_vel"] = -market_df["SLOPE_10Y2Y"].diff(20)

    # 3. Cross-asset correlation (positive SPY-TLT corr = both selling = risk-off)
    raw["cross_asset_corr"] = strat_ret_df["SPY"].rolling(21, min_periods=10).corr(
        strat_ret_df["TLT"]
    )

    # 4. Breadth deterioration (falling breadth = risk-off, so negate diff)
    cash_col = cfg.cash_col
    non_cash = strat_ret_df.drop(columns=[cash_col], errors="ignore")
    mom_63 = non_cash.rolling(63).sum()
    momentum_breadth = (mom_63 > 0).mean(axis=1)
    raw["breadth_deterioration"] = -momentum_breadth.diff(20)

    # 5. Realized vol ratio (spike > 1 = risk-off)
    spy_ret = strat_ret_df["SPY"]
    vol_short = ewma_vol(spy_ret, 5)
    vol_long = ewma_vol(spy_ret, 63).clip(lower=1e-12)
    raw["realized_vol_ratio"] = vol_short / vol_long

    # 6. VIX momentum (proxy for VIX term structure; VIX3M data not in pipeline)
    if "VIX" in market_df.columns:
        raw["vix_momentum"] = market_df["VIX"].diff(20)
    else:
        import warnings
        warnings.warn("VIX column missing from market_df; vix_momentum set to 0.0", stacklevel=2)
        raw["vix_momentum"] = pd.Series(0.0, index=market_df.index)

    raw_df = pd.DataFrame(raw)
    return rolling_zscore(raw_df, cfg.scanner_z_window, cfg.scanner_z_minp).fillna(0.0)


def compute_shift_signal(row: pd.Series, cfg: MetaConfig) -> ShiftSignal:
    """Compute shift signal from a single row of scanner features.

    Takes a single row from scanner_features.loc[dt] lookup.
    """
    weights = dict(cfg.scanner_feature_weights)
    if not weights:
        return ShiftSignal(
            regime_shift_probability=0.5,
            shift_direction="neutral",
            dominant_leading_indicator="",
            estimated_lead_weeks=3,
            raw_scores={},
        )

    # Collect scores, replacing NaN with 0.0
    scores = {}
    for feat in weights:
        val = row.get(feat, 0.0)
        scores[feat] = 0.0 if pd.isna(val) else float(val)

    # Weighted composite
    composite = sum(weights[feat] * scores[feat] for feat in weights)

    # Logistic transform
    shift_prob = float(sigmoid(cfg.scanner_steepness * composite))

    # Direction
    if composite > 0:
        direction = "risk_off"
    elif composite < 0:
        direction = "risk_on"
    else:
        direction = "neutral"

    # Dominant indicator (highest abs weighted score)
    dominant = max(weights, key=lambda f: abs(weights[f] * scores[f]))

    return ShiftSignal(
        regime_shift_probability=shift_prob,
        shift_direction=direction,
        dominant_leading_indicator=dominant,
        estimated_lead_weeks=3,
        raw_scores=scores,
    )


def compute_shift_velocity(
    current_prob: float,
    prev_probs: list[float],
    lookback: int = 4,
) -> float:
    """Rate of change in shift_prob over lookback weeks.

    Returns positive when shift_prob is rising (acceleration into risk-off).
    """
    if len(prev_probs) < lookback:
        return 0.0
    past_prob = prev_probs[-lookback]
    return (current_prob - past_prob) / lookback


def compute_scanner_leverage_adj(shift: ShiftSignal | None, cfg: MetaConfig) -> float:
    """Return the scanner's multiplicative leverage adjustment."""
    if shift is None:
        return 1.0

    if (
        shift.regime_shift_probability > cfg.scanner_threshold
        and shift.shift_direction == "risk_off"
    ):
        reduction = cfg.scanner_max_reduction * (
            (shift.regime_shift_probability - cfg.scanner_threshold)
            / (1.0 - cfg.scanner_threshold)
        )
        return float(1.0 - reduction)

    return 1.0
