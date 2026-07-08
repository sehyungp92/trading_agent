"""2-state Market Stress Detector (R8) -- dedicated HMM for financial stress.

Runs alongside the macro HMM on fast-moving financial indicators.
Replaces crisis overlay + scanner leverage adj + disagreement leverage adj +
crisis override with a single P(stress) output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ModuleNotFoundError:  # pragma: no cover
    class GaussianHMM:  # type: ignore[override]
        pass

from .config import MetaConfig
from .hmm import warm_start_from_previous

logger = logging.getLogger(__name__)

STRESS_FEATURES = [
    "credit_spread_mom", "yield_curve_vel", "cross_asset_corr",
    "breadth_deterioration", "realized_vol_ratio", "vix_momentum",
]


@dataclass(frozen=True)
class StressSignal:
    stress_level: float         # P(stress), 0-1
    stress_onset: bool          # True if just transitioned normal -> stress
    stress_velocity: float      # dP(stress)/dt over lookback weeks
    dominant_feature: str       # feature with highest abs z-score


def build_stress_features(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cfg: MetaConfig,
) -> pd.DataFrame:
    """Build 6-feature z-scored DataFrame for stress model.

    Delegates to scanner.build_scanner_features which is a pure function
    (does NOT check scanner_enabled -- always computes features).
    """
    from .scanner import build_scanner_features
    return build_scanner_features(market_df, strat_ret_df, cfg)


def init_stress_hmm(cfg: MetaConfig, n_iter: int) -> GaussianHMM:
    """Create a 2-state GaussianHMM for stress detection."""
    model = GaussianHMM(
        n_components=cfg.stress_n_states,
        covariance_type=cfg.stress_covariance_type,
        n_iter=n_iter,
        tol=cfg.tol,
        min_covar=cfg.min_covar,
        random_state=cfg.random_state,
        params="stmc",
        init_params="stmc",
    )
    # Asymmetric prior: normal state is sticky (dominant), stress state is transient.
    # With defaults (50, 2, 10): P(normal->normal)=96.2%, P(stress->stress)=83.3%
    # → stress fires ~19% of the time, matching the 15-20% calibration target.
    model.transmat_prior = np.array([
        [cfg.stress_sticky_diag, cfg.stress_sticky_offdiag],      # normal row
        [cfg.stress_sticky_offdiag, cfg.stress_stressed_sticky],   # stress row
    ])
    return model


def align_stress_states(model: GaussianHMM) -> None:
    """Ensure state 0 = normal, state 1 = stress (in-place).

    The state with higher mean across all 6 z-scored features is stress
    (positive z-scores = credit widening, vol spiking, etc.).
    """
    mean_scores = [float(model.means_[k].mean()) for k in range(2)]
    if mean_scores[0] > mean_scores[1]:
        # Swap: state 0 has higher mean, but should be state 1 (stress)
        order = [1, 0]
        model.means_ = model.means_[order]
        model.covars_ = model.covars_[order]
        model.startprob_ = model.startprob_[order]
        model.transmat_ = model.transmat_[np.ix_(order, order)]


def fit_stress_hmm(
    X_train: np.ndarray,
    cfg: MetaConfig,
    prev_model: Optional[GaussianHMM] = None,
    first_fit: bool = True,
    feature_names: Optional[list[str]] = None,
) -> Tuple[GaussianHMM, dict]:
    """Fit or refit the 2-state stress HMM.

    Same pattern as hmm.fit_or_refit_hmm(): warm start, fit, align, OOS guard.
    """
    n_iter = cfg.stress_n_iter_first_fit if first_fit else cfg.stress_n_iter
    model = init_stress_hmm(cfg, n_iter)

    if cfg.use_warm_start and prev_model is not None and not first_fit:
        warm_start_from_previous(model, prev_model)

    try:
        model.fit(X_train)
    except Exception as exc:
        if prev_model is not None:
            logger.warning("Stress HMM fit failed (%s), keeping previous model", exc)
            return prev_model, {"refit_rejected": True, "fit_error": str(exc)}
        raise

    align_stress_states(model)

    # OOS refit guard
    if prev_model is not None and not first_fit:
        val_start = max(0, len(X_train) - cfg.refit_validation_window)
        X_val = X_train[val_start:]
        ll_new = model.score(X_val)
        ll_old = prev_model.score(X_val)
        if ll_new < ll_old - cfg.stress_refit_guard_tol:
            return prev_model, {
                "refit_rejected": True,
                "ll_new": ll_new,
                "ll_old": ll_old,
            }

    return model, {"refit_rejected": False}


def stress_forward_step(
    model: GaussianHMM,
    x_obs: np.ndarray,
    prev_log_alpha: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """One step of forward filtering for the stress model.

    Delegates to inference.forward_filter_step (works for any N-state HMM).
    """
    from .inference import forward_filter_step
    return forward_filter_step(model, x_obs, prev_log_alpha)


def compute_stress_signal(
    posterior: np.ndarray,
    prev_stress_levels: list[float],
    stress_features_row: pd.Series,
    cfg: MetaConfig,
) -> StressSignal:
    """Build a StressSignal from the 2-state posterior."""
    stress_level = float(posterior[1])  # state 1 = stress after alignment

    # Onset detection
    stress_onset = stress_level >= cfg.stress_onset_threshold and (
        not prev_stress_levels
        or prev_stress_levels[-1] < cfg.stress_onset_threshold
    )

    # Velocity
    lookback = cfg.stress_velocity_lookback
    if len(prev_stress_levels) >= lookback:
        stress_velocity = (stress_level - prev_stress_levels[-lookback]) / lookback
    else:
        stress_velocity = 0.0

    # Dominant feature
    dominant_feature = max(
        STRESS_FEATURES,
        key=lambda f: abs(float(stress_features_row.get(f, 0.0))),
    )

    return StressSignal(
        stress_level=stress_level,
        stress_onset=stress_onset,
        stress_velocity=stress_velocity,
        dominant_feature=dominant_feature,
    )


def apply_stress_leverage(hmm_leverage: float, p_stress: float, cfg: MetaConfig) -> float:
    """Scale leverage by stress probability.

    At P(stress)=0: full leverage. At P(stress)=1: reduced by stress_reduction_max.
    """
    return float(hmm_leverage * (1.0 - cfg.stress_reduction_max * p_stress))


def apply_stress_blend(
    w_macro: pd.Series,
    wD: pd.Series,
    p_stress: float,
    cfg: MetaConfig,
) -> pd.Series:
    """Blend macro allocations toward Defensive based on stress level.

    Below stress_blend_threshold: no blending.
    Above: linear interpolation toward Defensive allocations.
    """
    if p_stress < cfg.stress_blend_threshold:
        return w_macro.copy()

    blend_alpha = (p_stress - cfg.stress_blend_threshold) / (1.0 - cfg.stress_blend_threshold)
    w = (1.0 - blend_alpha) * w_macro + blend_alpha * wD
    w = w.clip(lower=0.0)
    total = w.sum()
    if total > 0:
        w = w / total
    return w
