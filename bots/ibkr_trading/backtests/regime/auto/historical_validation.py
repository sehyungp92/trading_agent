"""Historical validation of regime assignments against known macro timeline.

Computes alignment score (fraction of weeks matching expected regime) and
transition latency (weeks delay before model detects known regime shifts).
"""
from __future__ import annotations

import pandas as pd

# Approximate macro consensus timeline (start, end, expected_regime)
KNOWN_REGIMES = [
    ("2003-01", "2007-06", "G"),   # Pre-GFC expansion
    ("2007-07", "2009-02", "D"),   # GFC
    ("2009-03", "2010-12", "R"),   # QE1 recovery + commodities
    ("2011-01", "2011-09", "S"),   # European debt + oil
    ("2011-10", "2014-06", "G"),   # Low-vol expansion
    ("2014-07", "2016-01", "D"),   # Oil crash + China
    ("2016-02", "2018-01", "R"),   # Reflation trade
    ("2018-02", "2018-12", "S"),   # Rate hikes + trade war
    ("2019-01", "2020-01", "G"),   # Late-cycle
    ("2020-02", "2020-03", "D"),   # COVID crash
    ("2020-04", "2021-12", "R"),   # Stimulus + reflation
    ("2022-01", "2022-10", "S"),   # Inflation + rate hikes
    ("2022-11", "2024-12", "G"),   # Soft landing
]

# Key transitions for latency measurement
KEY_TRANSITIONS = {
    "GFC_onset":      ("2007-07-01", "D"),
    "COVID_onset":    ("2020-02-15", "D"),
    "Inflation_2022": ("2022-01-01", "S"),
}

REGIME_COL_MAP = {"G": "P_G", "R": "P_R", "S": "P_S", "D": "P_D"}
REGIME_COLS = ["P_G", "P_R", "P_S", "P_D"]


def compute_historical_alignment(signals: pd.DataFrame) -> dict:
    """Compute alignment between model regimes and known timeline.

    Args:
        signals: DataFrame with P_G, P_R, P_S, P_D columns.

    Returns:
        dict with 'overall' alignment score and 'per_period' breakdown.
    """
    if signals.empty:
        return {"overall": 0.0, "per_period": {}}

    dominant = signals[REGIME_COLS].idxmax(axis=1).str.removeprefix("P_")
    per_period = {}
    total_weeks = 0
    total_matches = 0

    for start, end, expected in KNOWN_REGIMES:
        period_slice = dominant.loc[start:end]
        if period_slice.empty:
            continue
        n_weeks = len(period_slice)
        n_match = int((period_slice == expected).sum())
        frac = n_match / n_weeks
        label = f"{start}_{end}_{expected}"
        per_period[label] = frac
        total_weeks += n_weeks
        total_matches += n_match

    overall = total_matches / total_weeks if total_weeks > 0 else 0.0
    return {"overall": overall, "per_period": per_period}


def compute_transition_latency(signals: pd.DataFrame) -> dict:
    """Compute weeks delay before model transitions to expected regime.

    For each key transition point, counts how many weeks after the expected
    start date until the model's dominant regime matches the expected one.

    Args:
        signals: DataFrame with P_G, P_R, P_S, P_D columns.

    Returns:
        dict mapping transition name to latency in weeks.
    """
    if signals.empty:
        return {}

    dominant = signals[REGIME_COLS].idxmax(axis=1).str.removeprefix("P_")
    latencies = {}

    for name, (start_date, expected) in KEY_TRANSITIONS.items():
        ts = pd.Timestamp(start_date)
        post = dominant.loc[ts:]
        if post.empty:
            latencies[name] = float("inf")
            continue

        # Find first week where dominant regime matches expected
        matches = post == expected
        if matches.any():
            first_match_idx = matches.idxmax()
            delta_days = (first_match_idx - ts).days
            latencies[name] = max(0.0, delta_days / 7.0)
        else:
            latencies[name] = float("inf")

    return latencies
