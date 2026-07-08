from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from regime.config import MetaConfig
from regime.scanner import ShiftSignal
import regime.engine as regime_engine
import backtests.regime.engine.cached_engine as cached_engine


class DummyHMM:
    def __init__(self, posterior):
        self.posterior = np.asarray(posterior, dtype=float)
        self.means_ = np.zeros((4, 2))

    def predict_proba(self, x):
        return np.asarray([self.posterior], dtype=float)


def test_full_and_cached_engines_match_on_new_step3_step4_columns(monkeypatch):
    weekly_idx = pd.date_range("2022-01-07", periods=6, freq="W-FRI")
    daily_idx = pd.date_range("2021-12-01", periods=60, freq="B")

    macro_df = pd.DataFrame({"GROWTH": 0.0, "INFLATION": 0.0}, index=daily_idx)
    market_df = pd.DataFrame(
        {"VIX": 20.0, "SPREAD": 1.0, "SLOPE_10Y2Y": 0.5},
        index=daily_idx,
    )
    strat_ret_df = pd.DataFrame(
        {
            "SPY": 0.001,
            "EFA": 0.001,
            "TLT": 0.001,
            "GLD": 0.001,
            "CASH": 0.0,
        },
        index=daily_idx,
    )
    Xz = pd.DataFrame({"x": 0.0, "y": 0.0}, index=weekly_idx)

    ensemble = [
        (DummyHMM([0.7, 0.1, 0.1, 0.1]), {}),
        (DummyHMM([0.1, 0.1, 0.1, 0.7]), {}),
    ]

    crisis_daily = pd.DataFrame(
        {
            "crisis_vix_z": 1.0,
            "crisis_spread_z": 0.5,
            "crisis_realized_vol_z": 0.2,
            "crisis_spy_tlt_corr": 0.3,
            "crisis_corr_trigger": 0.3,
            "crisis_pairwise_corr_z": 0.1,
            "crisis_vix_component": 0.3,
            "crisis_spread_component": 0.3,
            "crisis_vol_component": 0.02,
            "crisis_corr_component": 0.09,
            "crisis_legacy_corr_component": 0.0,
            "crisis_composite": 0.71,
            "p_crisis": 0.6,
        },
        index=weekly_idx,
    )

    shift_signal = ShiftSignal(
        regime_shift_probability=0.8,
        shift_direction="risk_off",
        dominant_leading_indicator="cross_asset_corr",
        estimated_lead_weeks=3,
        raw_scores={},
    )

    for module in (regime_engine, cached_engine):
        monkeypatch.setattr(module, "build_observation_matrix", lambda *args, **kwargs: (Xz, 0, 1))
        monkeypatch.setattr(module, "fit_ensemble_hmm", lambda *args, **kwargs: ensemble)
        monkeypatch.setattr(module, "estimate_shrunk_covariance", lambda hist, cfg: None)
        monkeypatch.setattr(module, "weights_from_risk_budget", lambda budget, cov, cfg: budget.copy())
        monkeypatch.setattr(module, "apply_ventilator", lambda w, p_crisis, hist_ret, cfg, exempt_state: (w, exempt_state))
        monkeypatch.setattr(module, "compute_leverage", lambda w, hist, cfg: 1.25)
        monkeypatch.setattr(module, "compute_crisis_signal", lambda *args, **kwargs: crisis_daily)
        monkeypatch.setattr(module, "build_scanner_features", lambda *args, **kwargs: pd.DataFrame({"x": 0.0}, index=weekly_idx))
        monkeypatch.setattr(module, "compute_shift_signal", lambda row, cfg: shift_signal)
        monkeypatch.setattr(module, "compute_scanner_leverage_adj", lambda shift, cfg: 0.9)

    cfg = MetaConfig(
        n_ensemble_models=2,
        scanner_enabled=True,
        refit_freq="W-FRI",
        stress_model_enabled=False,
    )

    full = regime_engine.run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )
    cache = cached_engine.build_hmm_cache(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="GROWTH",
        inflation_feature="INFLATION",
        cfg=cfg,
    )
    cached = cached_engine.run_from_cache(
        cache=cache,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        cfg=cfg,
    )

    cols = [
        "P_G",
        "P_D",
        "posterior_conf",
        "disagreement_conf",
        "Conf",
        "p_crisis",
        "crisis_leverage_adj",
        "scanner_leverage_adj",
        "disagreement_leverage_adj",
        "final_leverage",
        "consensus_ratio",
        "minority_regime",
        "avg_disagreement",
        "uncertainty_level",
        "disagreement_warning",
        "pi_SPY",
    ]
    assert_frame_equal(full[cols], cached[cols])
