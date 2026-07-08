"""Section 8/10: End-to-end weekly signal engine orchestrator."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from hmmlearn.hmm import GaussianHMM
except ModuleNotFoundError:  # pragma: no cover - optional dependency in test envs
    class GaussianHMM:  # type: ignore[override]
        pass

from .config import MetaConfig, REGIMES
from .features import build_observation_matrix
from .hmm import fit_ensemble_hmm
from .inference import (
    calibrate_posteriors,
    classify_crisis_severity,
    compute_confidence,
    compute_crisis_leverage_adj,
    compute_crisis_signal,
    compute_disagreement_confidence,
    compute_disagreement_leverage_adj,
    compute_disagreement_warning,
    compute_ensemble_disagreement,
    compute_regime_momentum,
    forward_filter_step,
)
from .leverage import compute_leverage
from .portfolio import (
    apply_ventilator,
    blend_policy_portfolios,
    confidence_fallback,
    default_regime_budgets,
    estimate_shrunk_covariance,
    smooth_weights,
    weights_from_risk_budget,
)
from .scanner import (
    ShiftSignal,
    build_scanner_features,
    compute_scanner_leverage_adj,
    compute_shift_signal,
    compute_shift_velocity,
)
from .stress import (
    StressSignal,
    apply_stress_blend,
    apply_stress_leverage,
    build_stress_features,
    compute_stress_signal,
    fit_stress_hmm,
    stress_forward_step,
)


def _crisis_override_allocations(
    w_post: pd.Series,
    p_crisis: float,
    shift: "ShiftSignal | None",
    dominant_regime: str,
    cfg: MetaConfig,
    wD: pd.Series,
) -> tuple[pd.Series, bool]:
    """Override allocations to Defensive when crisis+scanner both signal
    severe risk-off but HMM still says risk-on (G/R)."""
    if not cfg.crisis_override_enabled:
        return w_post, False
    if dominant_regime not in ("G", "R"):
        return w_post, False
    if p_crisis < cfg.crisis_override_threshold:
        return w_post, False
    if shift is None:
        return w_post, False
    if (shift.regime_shift_probability < cfg.scanner_override_threshold
            or shift.shift_direction != "risk_off"):
        return w_post, False
    return wD.copy(), True


def run_signal_engine(
    macro_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    market_df: pd.DataFrame,
    growth_feature: str,
    inflation_feature: str,
    cfg: MetaConfig,
) -> pd.DataFrame:
    required_cols = ["SPY", "EFA", "TLT", "GLD", cfg.cash_col]
    for c in required_cols:
        assert c in strat_ret_df.columns, f"Missing: {c}"

    Xz, g_idx, i_idx = build_observation_matrix(
        macro_df,
        market_df,
        strat_ret_df,
        cfg,
        growth_feature=growth_feature,
        inflation_feature=inflation_feature,
    )

    sleeves = strat_ret_df.columns.tolist()
    regime_budgets, w_neutral = default_regime_budgets(sleeves, cfg)

    crisis_daily = compute_crisis_signal(
        market_df.reindex(Xz.index).ffill(),
        strat_ret_df.reindex(Xz.index).ffill(),
        cfg,
    ).fillna(0.0)

    scanner_features_daily = None
    if cfg.scanner_enabled:
        scanner_features_daily = build_scanner_features(
            market_df.reindex(Xz.index).ffill(),
            strat_ret_df.reindex(Xz.index).ffill(),
            cfg,
        )

    stress_features_daily = None
    if cfg.stress_model_enabled:
        if cfg.scanner_enabled and scanner_features_daily is not None:
            stress_features_daily = scanner_features_daily
        else:
            stress_features_daily = build_stress_features(
                market_df.reindex(Xz.index).ffill(),
                strat_ret_df.reindex(Xz.index).ffill(),
                cfg,
            )

    refit_dates_raw = pd.date_range(Xz.index.min(), Xz.index.max(), freq=cfg.refit_freq)
    snap_idx = Xz.index.searchsorted(refit_dates_raw, side="right") - 1
    snap_idx = snap_idx[(snap_idx >= 0) & (snap_idx < len(Xz.index))]
    refit_dates = Xz.index[np.unique(snap_idx)]

    weekly_dates = pd.date_range(Xz.index.min(), Xz.index.max(), freq=cfg.rebalance_freq)
    weekly_dates = weekly_dates.intersection(Xz.index)

    rows = []
    ensemble: List[Tuple[GaussianHMM, dict]] = []
    model: Optional[GaussianHMM] = None
    refit_pointer = 0
    exempt_state: Dict[str, int] = {}
    P_prev: Optional[np.ndarray] = None
    P_history: list = []
    consensus_history: list[float] = []
    shift_prob_history: list[float] = []
    prev_means: Optional[np.ndarray] = None
    w_prev: Optional[pd.Series] = None
    fwd_log_alphas: Optional[List[Optional[np.ndarray]]] = None
    stress_model: Optional[GaussianHMM] = None
    stress_log_alpha: Optional[np.ndarray] = None
    stress_level_history: list[float] = []

    for dt in weekly_dates:
        while refit_pointer < len(refit_dates) and refit_dates[refit_pointer] <= dt:
            rd = refit_dates[refit_pointer]
            if cfg.use_expanding_window:
                X_train = Xz.loc[:rd].values
            else:
                window_start = rd - pd.DateOffset(years=cfg.rolling_window_years)
                X_train = Xz.loc[window_start:rd].values
                if len(X_train) < 252:
                    X_train = Xz.loc[:rd].values
            first_fit = model is None

            ensemble = fit_ensemble_hmm(
                X_train=X_train,
                cfg=cfg,
                growth_idx=g_idx,
                infl_idx=i_idx,
                prev_model=model,
                first_fit=first_fit,
                prev_means=prev_means,
            )
            model = ensemble[0][0]
            prev_means = model.means_.copy()
            fwd_log_alphas = [None] * len(ensemble) if cfg.use_forward_only else None
            refit_pointer += 1

            if cfg.stress_model_enabled and stress_features_daily is not None:
                stress_train = stress_features_daily.loc[:rd].dropna()
                if len(stress_train) >= 60:
                    stress_first = stress_model is None
                    stress_model, _ = fit_stress_hmm(
                        stress_train.values, cfg,
                        prev_model=stress_model,
                        first_fit=stress_first,
                    )
                    stress_log_alpha = None  # reset forward filter after refit

        if not ensemble:
            continue

        x_dt = Xz.loc[[dt]].values
        raw_per_model = []
        if cfg.use_forward_only:
            for i, (m, _) in enumerate(ensemble):
                posterior, fwd_log_alphas[i] = forward_filter_step(
                    m,
                    x_dt,
                    fwd_log_alphas[i],
                )
                raw_per_model.append(posterior)
        else:
            raw_per_model = [m.predict_proba(x_dt)[0] for m, _ in ensemble]

        calibrated_per_model = [
            calibrate_posteriors(
                posterior,
                temperature=cfg.posterior_temperature,
                smoothing_eps=cfg.posterior_smoothing_eps,
            )
            for posterior in raw_per_model
        ]
        P_grsd = np.mean(calibrated_per_model, axis=0)

        disagreement = compute_ensemble_disagreement(
            calibrated_per_model,
            consensus_history=consensus_history,
            high_consensus=cfg.disagreement_moderate_consensus,
            moderate_consensus=cfg.disagreement_low_consensus,
        )
        consensus_history.append(disagreement["consensus_ratio"])
        disagreement_warning = compute_disagreement_warning(disagreement, cfg)

        if P_prev is not None and cfg.posterior_ema_alpha < 1.0:
            defensive_rising = (P_grsd[2] + P_grsd[3]) > (P_prev[2] + P_prev[3])
            alpha = (
                cfg.posterior_ema_risk_off_alpha
                if defensive_rising
                else cfg.posterior_ema_alpha
            )
            if alpha < 1.0:
                P_grsd = alpha * P_grsd + (1.0 - alpha) * P_prev
                P_grsd /= P_grsd.sum()

        posterior_conf = compute_confidence(P_grsd, P_prev, cfg)
        disagreement_conf = (
            compute_disagreement_confidence(disagreement, cfg, posterior_conf)
            if len(ensemble) > 1
            else float("nan")
        )
        conf = disagreement_conf if len(ensemble) > 1 else posterior_conf
        P_prev = P_grsd.copy()

        P_history.append(P_grsd.copy())
        if len(P_history) > cfg.regime_momentum_lookback + 2:
            P_history.pop(0)
        momentum = compute_regime_momentum(P_history, cfg.regime_momentum_lookback)

        crisis_row = crisis_daily.loc[dt]
        p_crisis = float(crisis_row["p_crisis"])
        crisis_severity = classify_crisis_severity(p_crisis)

        stress_signal: StressSignal | None = None
        if (cfg.stress_model_enabled and stress_model is not None
                and stress_features_daily is not None
                and dt in stress_features_daily.index):
            x_stress = stress_features_daily.loc[[dt]].values
            stress_posterior, stress_log_alpha = stress_forward_step(
                stress_model, x_stress, stress_log_alpha,
            )
            stress_signal = compute_stress_signal(
                stress_posterior, stress_level_history,
                stress_features_daily.loc[dt], cfg,
            )
            stress_level_history.append(stress_signal.stress_level)
            if len(stress_level_history) > 20:
                stress_level_history.pop(0)

        hist = strat_ret_df.loc[:dt]
        cov_annual = estimate_shrunk_covariance(hist, cfg)

        wG = weights_from_risk_budget(regime_budgets["G"], cov_annual, cfg)
        wR = weights_from_risk_budget(regime_budgets["R"], cov_annual, cfg)
        wS = weights_from_risk_budget(regime_budgets["S"], cov_annual, cfg)
        wD = weights_from_risk_budget(regime_budgets["D"], cov_annual, cfg)

        w_active = blend_policy_portfolios(P_grsd, wG, wR, wS, wD)
        w_pre = confidence_fallback(w_active, w_neutral, conf)
        w_pre = smooth_weights(w_pre, w_prev, cfg.weight_smoothing_alpha)
        w_prev = w_pre.copy()

        w_post, exempt_state = apply_ventilator(
            w=w_pre.copy(),
            p_crisis=p_crisis,
            hist_ret=hist,
            cfg=cfg,
            exempt_state=exempt_state,
        )

        hmm_leverage = compute_leverage(w_post, hist, cfg)

        # -- Legacy overlays: always computed for diagnostic columns --
        shift = None
        scanner_leverage_adj = 1.0
        if scanner_features_daily is not None and dt in scanner_features_daily.index:
            shift = compute_shift_signal(scanner_features_daily.loc[dt], cfg)
            scanner_leverage_adj = compute_scanner_leverage_adj(shift, cfg)

        crisis_leverage_adj = compute_crisis_leverage_adj(
            p_crisis,
            enabled=cfg.crisis_leverage_enabled,
            threshold_low=cfg.crisis_leverage_threshold_low,
            threshold_high=cfg.crisis_leverage_threshold_high,
            reduction_mid=cfg.crisis_leverage_reduction_mid,
            reduction_max=cfg.crisis_leverage_reduction_max,
        )
        disagreement_leverage_adj = compute_disagreement_leverage_adj(disagreement, cfg)

        # -- Decision path --
        crisis_overridden = False
        if cfg.stress_model_enabled and stress_signal is not None:
            # R8: single stress model controls allocation blending + leverage
            p_stress = stress_signal.stress_level
            w_post = apply_stress_blend(w_post, wD, p_stress, cfg)
            final_leverage = apply_stress_leverage(hmm_leverage, p_stress, cfg)
        elif cfg.stress_model_enabled:
            # R8 pre-fit: stress model not yet ready, skip legacy overlays
            final_leverage = float(hmm_leverage)
        elif cfg.conjunction_gating_enabled:
            # R7: conjunction gating
            active_layers = []
            if crisis_leverage_adj < cfg.conjunction_active_threshold:
                active_layers.append(crisis_leverage_adj)
            if scanner_leverage_adj < cfg.conjunction_active_threshold:
                active_layers.append(scanner_leverage_adj)
            n_active = len(active_layers)
            if n_active == 0:
                overlay_mult = 1.0
            elif n_active == 1:
                raw_reduction = 1.0 - active_layers[0]
                capped_reduction = min(raw_reduction, cfg.conjunction_1_layer_max_reduction)
                overlay_mult = 1.0 - capped_reduction
            else:
                product = 1.0
                for adj in active_layers:
                    product *= adj
                overlay_mult = max(product, cfg.conjunction_min_multiplier)
            final_leverage = float(hmm_leverage * overlay_mult)

            # Crisis override (R7 Track B)
            dominant_regime = REGIMES[np.argmax(P_grsd)]
            w_post, crisis_overridden = _crisis_override_allocations(
                w_post, p_crisis, shift, dominant_regime, cfg, wD,
            )
        else:
            # Legacy multiplicative path
            final_leverage = float(
                hmm_leverage
                * crisis_leverage_adj
                * scanner_leverage_adj
                * disagreement_leverage_adj
            )

        row = {
            "date": dt,
            "P_G": P_grsd[0],
            "P_R": P_grsd[1],
            "P_S": P_grsd[2],
            "P_D": P_grsd[3],
            "posterior_conf": float(posterior_conf),
            "disagreement_conf": float(disagreement_conf),
            "Conf": conf,
            "p_crisis": p_crisis,
            "crisis_severity": crisis_severity,
            "hmm_leverage": float(hmm_leverage),
            "crisis_leverage_adj": float(crisis_leverage_adj),
            "scanner_leverage_adj": float(scanner_leverage_adj),
            "disagreement_leverage_adj": float(disagreement_leverage_adj),
            "final_leverage": final_leverage,
            "L": final_leverage,
            "consensus_ratio": disagreement["consensus_ratio"],
            "mode_regime": disagreement["mode_regime"],
            "minority_regime": disagreement["minority_regime"],
            "minority_fraction": disagreement["minority_fraction"],
            "avg_disagreement": disagreement["avg_disagreement"],
            "consensus_trend_4w": disagreement["consensus_trend_4w"],
            "uncertainty_level": disagreement["uncertainty_level"],
            "disagreement_warning": disagreement_warning,
            "crisis_overridden": crisis_overridden,
            **momentum,
        }

        if stress_signal is not None:
            row["stress_level"] = stress_signal.stress_level
            row["stress_onset"] = stress_signal.stress_onset
            row["stress_velocity"] = stress_signal.stress_velocity
            row["stress_dominant_feature"] = stress_signal.dominant_feature
        elif cfg.stress_model_enabled:
            row["stress_level"] = 0.0
            row["stress_onset"] = False
            row["stress_velocity"] = 0.0
            row["stress_dominant_feature"] = ""

        for regime in REGIMES:
            row[f"disagree_std_{regime}"] = disagreement["per_regime_std"][regime]

        for crisis_col in crisis_daily.columns:
            row[crisis_col] = float(crisis_row[crisis_col])

        if shift is not None:
            row["shift_prob"] = shift.regime_shift_probability
            row["shift_dir"] = shift.shift_direction
            row["dominant_indicator"] = shift.dominant_leading_indicator
            shift_prob_history.append(shift.regime_shift_probability)
            if len(shift_prob_history) > 10:
                shift_prob_history.pop(0)
            row["shift_velocity"] = compute_shift_velocity(
                shift.regime_shift_probability, shift_prob_history,
            )
        else:
            row["shift_velocity"] = 0.0

        for c in sleeves:
            row[f"w_{c}"] = float(w_post.get(c, 0.0))
            row[f"pi_{c}"] = float(final_leverage * w_post.get(c, 0.0))

        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    cfg = MetaConfig()

    macro_df = pd.read_parquet("macro_features.parquet")
    strat_ret_df = pd.read_parquet("strategy_returns.parquet")
    market_df = pd.read_parquet("market_stress.parquet")

    signals = run_signal_engine(
        macro_df=macro_df,
        strat_ret_df=strat_ret_df,
        market_df=market_df,
        growth_feature="growth_feature",
        inflation_feature="inflation_feature",
        cfg=cfg,
    )

    signals.to_parquet("meta_signals_weekly_v2.parquet")
    print(signals.tail())
