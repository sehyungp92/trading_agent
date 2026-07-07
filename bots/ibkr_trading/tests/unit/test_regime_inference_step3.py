from __future__ import annotations

import numpy as np
import pandas as pd

from regime.config import MetaConfig
from regime.inference import (
    classify_crisis_severity,
    compute_crisis_leverage_adj,
    compute_disagreement_confidence,
    compute_disagreement_leverage_adj,
    compute_disagreement_warning,
    compute_crisis_signal,
    compute_ensemble_disagreement,
    classify_uncertainty_level,
)
from regime.scanner import ShiftSignal, compute_scanner_leverage_adj


def _make_market_inputs(tlt_sign: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2022-01-03", periods=80, freq="B")
    market_df = pd.DataFrame(
        {
            "VIX": np.linspace(15.0, 30.0, len(idx)),
            "SPREAD": np.linspace(1.0, 2.5, len(idx)),
        },
        index=idx,
    )
    spy = pd.Series(np.linspace(-0.01, 0.02, len(idx)), index=idx)
    tlt = spy * tlt_sign
    strat_ret_df = pd.DataFrame(
        {
            "SPY": spy,
            "EFA": spy * 0.9,
            "TLT": tlt,
            "GLD": spy * 0.1,
            "CASH": 0.0,
        },
        index=idx,
    )
    return market_df, strat_ret_df


def test_crisis_signal_supports_legacy_and_new_weight_shapes():
    market_df, strat_pos = _make_market_inputs(tlt_sign=0.8)
    _, strat_neg = _make_market_inputs(tlt_sign=-0.8)

    new_cfg = MetaConfig(crisis_weights=(0.3, 0.6, 0.1, 0.1), crisis_z_window=21)
    pos_signal = compute_crisis_signal(market_df, strat_pos, new_cfg)
    neg_signal = compute_crisis_signal(market_df, strat_neg, new_cfg)

    assert pos_signal["crisis_corr_component"].iloc[-1] > 0.0
    assert neg_signal["crisis_corr_component"].iloc[-1] == 0.0
    assert pos_signal["p_crisis"].iloc[-1] > neg_signal["p_crisis"].iloc[-1]

    legacy_cfg = MetaConfig(crisis_weights=(0.3, 0.6, 0.1), crisis_z_window=21)
    legacy_signal = compute_crisis_signal(market_df, strat_pos, legacy_cfg)
    assert "crisis_legacy_corr_component" in legacy_signal.columns
    assert np.isfinite(legacy_signal["p_crisis"].iloc[-1])


def test_crisis_severity_and_leverage_adjustments_follow_thresholds():
    assert classify_crisis_severity(0.10) == "none"
    assert classify_crisis_severity(0.35) == "elevated"
    assert classify_crisis_severity(0.80) == "acute"

    assert compute_crisis_leverage_adj(0.10) == 1.0
    assert 0.75 < compute_crisis_leverage_adj(0.50) < 1.0
    assert 0.50 <= compute_crisis_leverage_adj(0.90) < 0.75
    assert compute_crisis_leverage_adj(0.90, enabled=False) == 1.0


def test_scanner_multiplier_and_disagreement_metrics_cover_edge_cases():
    cfg = MetaConfig(scanner_threshold=0.6, scanner_max_reduction=0.30, n_ensemble_models=2)
    shift = ShiftSignal(
        regime_shift_probability=0.8,
        shift_direction="risk_off",
        dominant_leading_indicator="cross_asset_corr",
        estimated_lead_weeks=3,
        raw_scores={},
    )
    assert 0.7 < compute_scanner_leverage_adj(shift, cfg) < 1.0
    assert compute_scanner_leverage_adj(None, cfg) == 1.0

    single = compute_ensemble_disagreement([np.array([1.0, 0.0, 0.0, 0.0])])
    assert single["consensus_ratio"] == 1.0
    assert single["minority_regime"] == "G"
    assert single["uncertainty_level"] == "low"

    multi = compute_ensemble_disagreement(
        [
            np.array([0.7, 0.1, 0.1, 0.1]),
            np.array([0.1, 0.1, 0.1, 0.7]),
        ],
        consensus_history=[1.0, 0.95, 0.90, 0.85],
    )
    assert multi["consensus_ratio"] == 0.5
    assert multi["minority_regime"] == "D"
    assert multi["uncertainty_level"] == "high"
    assert multi["consensus_trend_4w"] < 0.0

    disagreement_conf = compute_disagreement_confidence(multi, cfg, posterior_conf=0.35)
    disagreement_adj = compute_disagreement_leverage_adj(multi, cfg)
    assert 0.69 <= disagreement_conf <= 0.80
    assert 0.65 <= disagreement_adj < 0.80
    assert compute_disagreement_warning(multi, cfg) is True

    single_cfg = MetaConfig(n_ensemble_models=1)
    assert compute_disagreement_confidence(single, single_cfg, posterior_conf=0.42) == 0.42
    assert compute_disagreement_leverage_adj(single, single_cfg) == 1.0
    assert compute_disagreement_warning(single, single_cfg) is False


def test_uncertainty_thresholds_and_warning_logic_are_normalized():
    assert (
        classify_uncertainty_level(0.80, high_consensus=0.70, moderate_consensus=0.90)
        == "moderate"
    )

    cfg = MetaConfig(
        n_ensemble_models=3,
        disagreement_low_consensus=0.70,
        disagreement_moderate_consensus=0.90,
    )
    malformed = {
        "model_count": "3",
        "consensus_ratio": "not-a-number",
        "consensus_trend_4w": 0.0,
    }

    assert compute_disagreement_confidence(malformed, cfg, posterior_conf=0.44) == 0.44
    assert compute_disagreement_leverage_adj(malformed, cfg) == 1.0
    assert compute_disagreement_warning(malformed, cfg) is False
