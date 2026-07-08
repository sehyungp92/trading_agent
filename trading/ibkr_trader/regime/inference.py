"""Section 3: Phase A outputs for confidence, crisis, and disagreement."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import MetaConfig, REGIMES
from .utils import compute_avg_pairwise_corr, ewma_vol, rolling_zscore, sigmoid

UNCERTAINTY_LOW = 0.90
UNCERTAINTY_MODERATE = 0.70


def forward_filter_step(model, x_obs, prev_log_alpha=None):
    """One step of forward filtering. Maintains running log-alpha across calls."""
    from scipy.special import logsumexp

    log_emission = model._compute_log_likelihood(x_obs)[0]

    if prev_log_alpha is None:
        log_alpha = np.log(model.startprob_ + 1e-300) + log_emission
    else:
        log_transmat = np.log(model.transmat_ + 1e-300)
        n = len(log_emission)
        log_alpha = np.empty(n)
        for j in range(n):
            log_alpha[j] = (
                logsumexp(prev_log_alpha + log_transmat[:, j]) + log_emission[j]
            )

    posterior = np.exp(log_alpha - logsumexp(log_alpha))
    return posterior, log_alpha


def calibrate_posteriors(
    P_raw: np.ndarray,
    temperature: float = 1.0,
    smoothing_eps: float = 0.0,
) -> np.ndarray:
    """Calibrate HMM posteriors to reflect true uncertainty."""
    n = len(P_raw)

    if not np.isfinite(P_raw).all() or P_raw.sum() < 1e-12:
        return np.ones(n) / n

    if smoothing_eps > 0:
        P = (1.0 - smoothing_eps) * P_raw + smoothing_eps * np.ones(n) / n
    else:
        P = P_raw.copy()

    if temperature != 1.0 and temperature > 0:
        P_clipped = np.clip(P, 1e-300, 1.0)
        log_P = np.log(P_clipped)
        P = np.exp(log_P / temperature)
        total = P.sum()
        if total < 1e-300:
            P = np.ones(n) / n
        else:
            P /= total

    return P


def classify_uncertainty_level(
    consensus_ratio: float,
    high_consensus: float = UNCERTAINTY_LOW,
    moderate_consensus: float = UNCERTAINTY_MODERATE,
) -> str:
    """Map consensus to a discrete uncertainty bucket."""
    high_consensus, moderate_consensus = _normalize_consensus_thresholds(
        high_consensus,
        moderate_consensus,
    )
    if consensus_ratio >= high_consensus:
        return "low"
    if consensus_ratio >= moderate_consensus:
        return "moderate"
    return "high"


def _normalize_consensus_thresholds(
    high_consensus: float,
    moderate_consensus: float,
) -> tuple[float, float]:
    """Clamp and order the high/moderate consensus cutoffs."""
    high = float(np.clip(high_consensus, 0.0, 1.0))
    moderate = float(np.clip(moderate_consensus, 0.0, 1.0))
    if moderate > high:
        moderate, high = high, moderate
    if high - moderate < 1e-6:
        moderate = max(0.0, high - 1e-6)
    return high, moderate


def _disagreement_thresholds(cfg: MetaConfig) -> tuple[float, float]:
    """Normalize the disagreement thresholds into an ordered pair."""
    low = float(np.clip(cfg.disagreement_low_consensus, 0.0, 0.99))
    high = float(np.clip(cfg.disagreement_moderate_consensus, low + 1e-6, 1.0))
    return low, high


def _disagreement_model_count(disagreement: dict, cfg: MetaConfig) -> int:
    """Prefer the actual ensemble width from the disagreement payload."""
    model_count = disagreement.get("model_count", cfg.n_ensemble_models)
    try:
        return max(int(model_count), 0)
    except (TypeError, ValueError):
        return max(int(cfg.n_ensemble_models), 0)


def _base_disagreement_reduction(consensus_ratio: float, cfg: MetaConfig) -> float:
    """Map ensemble consensus to a base reduction schedule."""
    low, high = _disagreement_thresholds(cfg)
    consensus = float(np.clip(consensus_ratio, 0.0, 1.0))
    moderate_reduction = float(
        np.clip(cfg.disagreement_moderate_reduction, 0.0, cfg.disagreement_max_reduction)
    )
    high_reduction = float(
        np.clip(cfg.disagreement_high_reduction, moderate_reduction, cfg.disagreement_max_reduction)
    )

    if consensus >= high:
        return 0.0
    if consensus >= low:
        span = max(high - low, 1e-6)
        return moderate_reduction * ((high - consensus) / span)

    span = max(low, 1e-6)
    return moderate_reduction + (high_reduction - moderate_reduction) * (
        (low - consensus) / span
    )


def compute_disagreement_confidence(
    disagreement: dict,
    cfg: MetaConfig,
    posterior_conf: float,
) -> float:
    """Translate ensemble consensus into an effective allocation confidence."""
    if (
        _disagreement_model_count(disagreement, cfg) <= 1
        or not cfg.disagreement_confidence_enabled
    ):
        return float(np.clip(posterior_conf, cfg.conf_floor, 1.0))

    consensus_ratio = disagreement.get("consensus_ratio", 1.0)
    try:
        consensus_ratio = float(consensus_ratio)
    except (TypeError, ValueError):
        return float(np.clip(posterior_conf, cfg.conf_floor, 1.0))
    if not np.isfinite(consensus_ratio):
        return float(np.clip(posterior_conf, cfg.conf_floor, 1.0))

    reduction = _base_disagreement_reduction(consensus_ratio, cfg)
    conf = 1.0 - reduction
    return float(np.clip(conf, cfg.conf_floor, 1.0))


def compute_disagreement_leverage_adj(
    disagreement: dict,
    cfg: MetaConfig,
) -> float:
    """Return the Layer 4 leverage adjustment from ensemble disagreement."""
    if (
        _disagreement_model_count(disagreement, cfg) <= 1
        or not cfg.disagreement_leverage_enabled
    ):
        return 1.0

    low_consensus, high_consensus = _disagreement_thresholds(cfg)
    consensus = disagreement.get("consensus_ratio", 1.0)
    try:
        consensus = float(consensus)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(consensus):
        return 1.0
    consensus = float(np.clip(consensus, 0.0, 1.0))
    reduction = _base_disagreement_reduction(consensus, cfg)

    trend = float(disagreement.get("consensus_trend_4w", 0.0))
    if trend <= cfg.disagreement_trend_threshold:
        span = max(abs(cfg.disagreement_trend_threshold), 1e-6)
        trend_scale = np.clip(abs(trend) / span, 0.0, 1.0)
        reduction += float(cfg.disagreement_trend_extra_reduction) * trend_scale

    minority_regime = disagreement.get("minority_regime", "")
    if minority_regime in {"S", "D"} and consensus < high_consensus:
        span = max(high_consensus - low_consensus, 1e-6)
        risk_scale = np.clip((high_consensus - consensus) / span, 0.0, 1.0)
        reduction += float(cfg.disagreement_risk_off_extra_reduction) * risk_scale

    reduction = float(np.clip(reduction, 0.0, cfg.disagreement_max_reduction))
    return float(1.0 - reduction)


def compute_disagreement_warning(
    disagreement: dict,
    cfg: MetaConfig,
) -> bool:
    """Flag ambiguous or deteriorating consensus states for downstream consumers."""
    if _disagreement_model_count(disagreement, cfg) <= 1:
        return False

    low_consensus, _ = _disagreement_thresholds(cfg)
    consensus = disagreement.get("consensus_ratio", 1.0)
    trend = disagreement.get("consensus_trend_4w", 0.0)
    try:
        consensus = float(consensus)
        trend = float(trend)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(consensus) or not np.isfinite(trend):
        return False

    return bool(
        consensus < low_consensus or trend <= cfg.disagreement_trend_threshold
    )


def compute_regime_momentum(
    P_history: list,
    lookback: int = 4,
) -> dict:
    """Rate of change in posterior probabilities over lookback weeks."""
    if len(P_history) < lookback + 1:
        return {
            "mom_G": 0.0,
            "mom_R": 0.0,
            "mom_S": 0.0,
            "mom_D": 0.0,
            "risk_momentum": 0.0,
        }

    current = P_history[-1]
    past = P_history[-(lookback + 1)]
    delta = current - past
    risk_momentum = (delta[0] + delta[1]) - (delta[2] + delta[3])
    return {
        "mom_G": float(delta[0]),
        "mom_R": float(delta[1]),
        "mom_S": float(delta[2]),
        "mom_D": float(delta[3]),
        "risk_momentum": float(risk_momentum),
    }


def compute_confidence(
    P_grsd: np.ndarray,
    P_prev: Optional[np.ndarray],
    cfg: MetaConfig,
) -> float:
    """Entropy x posterior stability confidence score."""
    eps = 1e-12
    p = np.clip(P_grsd, eps, 1.0)

    H = -np.sum(p * np.log(p))
    Hmax = np.log(4.0)
    conf_entropy = 1.0 - H / Hmax

    if P_prev is not None:
        posterior_shift = np.sum(np.abs(P_grsd - P_prev))
        stability = 1.0 - 0.5 * posterior_shift
    else:
        stability = 1.0

    raw = conf_entropy * (
        (1.0 - cfg.stability_weight) + cfg.stability_weight * stability
    )
    conf = cfg.conf_floor + (1.0 - cfg.conf_floor) * raw
    return float(np.clip(conf, cfg.conf_floor, 1.0))


def compute_crisis_signal(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cfg: MetaConfig,
) -> pd.DataFrame:
    """Build the crisis overlay inputs and output probability on a daily index."""
    czw = cfg.crisis_z_window if cfg.crisis_z_window > 0 else cfg.z_window
    czmp = min(cfg.z_minp, max(2, czw // 2))

    vix_z = rolling_zscore(market_df[["VIX"]], czw, czmp)["VIX"]
    spr_z = rolling_zscore(market_df[["SPREAD"]], czw, czmp)["SPREAD"]

    spy_returns = strat_ret_df["SPY"].fillna(0.0)
    tlt_returns = strat_ret_df["TLT"].fillna(0.0)
    realized_vol = ewma_vol(spy_returns, span=21)
    realized_vol_z = rolling_zscore(
        realized_vol.to_frame("realized_vol"),
        czw,
        czmp,
    )["realized_vol"]

    spy_tlt_corr = spy_returns.rolling(21, min_periods=10).corr(tlt_returns)
    corr_trigger = spy_tlt_corr.clip(lower=0.0).fillna(0.0)

    non_cash = strat_ret_df.drop(columns=[cfg.cash_col], errors="ignore")
    avg_corr = compute_avg_pairwise_corr(non_cash, window=cfg.rho_short_window)
    pairwise_corr_z = rolling_zscore(
        avg_corr.to_frame("avg_pairwise_corr"),
        czw,
        czmp,
    )["avg_pairwise_corr"]

    weights = tuple(float(w) for w in cfg.crisis_weights)
    zero = pd.Series(0.0, index=vix_z.index)
    if len(weights) == 3:
        vix_component = weights[0] * vix_z
        spread_component = weights[1] * spr_z
        vol_component = zero
        corr_component = zero
        legacy_corr_component = weights[2] * pairwise_corr_z
        composite = vix_component + spread_component + legacy_corr_component
    elif len(weights) == 4:
        vix_component = weights[0] * vix_z
        spread_component = weights[1] * spr_z
        vol_component = weights[2] * realized_vol_z
        corr_component = weights[3] * corr_trigger * 3.0
        legacy_corr_component = zero
        composite = vix_component + spread_component + vol_component + corr_component
    else:
        raise ValueError(
            f"crisis_weights must have length 3 or 4, got {len(weights)}"
        )

    composite = composite.fillna(0.0)
    p_crisis = sigmoid(cfg.crisis_logit_a * (composite - cfg.crisis_logit_b))

    return pd.DataFrame(
        {
            "crisis_vix_z": vix_z,
            "crisis_spread_z": spr_z,
            "crisis_realized_vol_z": realized_vol_z,
            "crisis_spy_tlt_corr": spy_tlt_corr,
            "crisis_corr_trigger": corr_trigger,
            "crisis_pairwise_corr_z": pairwise_corr_z,
            "crisis_vix_component": vix_component,
            "crisis_spread_component": spread_component,
            "crisis_vol_component": vol_component,
            "crisis_corr_component": corr_component,
            "crisis_legacy_corr_component": legacy_corr_component,
            "crisis_composite": composite,
            "p_crisis": p_crisis,
        }
    )


def classify_crisis_severity(p_crisis: float) -> str:
    """Bucket crisis probability into a descriptive severity label."""
    if p_crisis < 0.35:
        return "none"
    if p_crisis < 0.65:
        return "elevated"
    return "acute"


def compute_crisis_leverage_adj(
    p_crisis: float,
    enabled: bool = True,
    threshold_low: float = 0.35,
    threshold_high: float = 0.65,
    reduction_mid: float = 0.25,
    reduction_max: float = 0.50,
) -> float:
    """Return the crisis overlay's multiplicative leverage adjustment."""
    if not enabled:
        return 1.0

    if p_crisis < threshold_low:
        return 1.0
    if p_crisis < threshold_high and threshold_high > threshold_low:
        span = (p_crisis - threshold_low) / (threshold_high - threshold_low)
        return float(1.0 - reduction_mid * span)

    if threshold_high < 1.0:
        span = min(max((p_crisis - threshold_high) / (1.0 - threshold_high), 0.0), 1.0)
        return float((1.0 - reduction_mid) - (reduction_max - reduction_mid) * span)

    return float(1.0 - reduction_max)


def compute_ensemble_disagreement(
    per_model_posteriors: list[np.ndarray],
    consensus_history: list[float] | None = None,
    high_consensus: float = UNCERTAINTY_LOW,
    moderate_consensus: float = UNCERTAINTY_MODERATE,
) -> dict:
    """Summarize ensemble disagreement before the per-model posteriors are averaged."""
    high_consensus, moderate_consensus = _normalize_consensus_thresholds(
        high_consensus,
        moderate_consensus,
    )
    if not per_model_posteriors:
        return {
            "model_count": 0,
            "consensus_ratio": 1.0,
            "mode_regime": REGIMES[0],
            "minority_regime": REGIMES[0],
            "minority_fraction": 0.0,
            "avg_disagreement": 0.0,
            "uncertainty_level": "low",
            "consensus_trend_4w": 0.0,
            "per_regime_std": {regime: 0.0 for regime in REGIMES},
        }

    stacked = np.stack(per_model_posteriors, axis=0)
    dominant = np.argmax(stacked, axis=1)
    counts = np.bincount(dominant, minlength=len(REGIMES))

    mode_idx = int(counts.argmax())
    consensus_ratio = float(counts[mode_idx] / len(dominant))

    minority_counts = counts.copy()
    minority_counts[mode_idx] = 0
    if minority_counts.max() > 0:
        minority_idx = int(minority_counts.argmax())
    else:
        minority_idx = mode_idx

    per_regime_std = stacked.std(axis=0)
    if consensus_history and len(consensus_history) >= 4:
        consensus_trend_4w = consensus_ratio - float(consensus_history[-4])
    else:
        consensus_trend_4w = 0.0

    return {
        "model_count": int(len(per_model_posteriors)),
        "consensus_ratio": consensus_ratio,
        "mode_regime": REGIMES[mode_idx],
        "minority_regime": REGIMES[minority_idx],
        "minority_fraction": float(1.0 - consensus_ratio),
        "avg_disagreement": float(per_regime_std.mean()),
        "uncertainty_level": classify_uncertainty_level(
            consensus_ratio,
            high_consensus=high_consensus,
            moderate_consensus=moderate_consensus,
        ),
        "consensus_trend_4w": float(consensus_trend_4w),
        "per_regime_std": {
            regime: float(per_regime_std[i])
            for i, regime in enumerate(REGIMES)
        },
    }


def compute_crisis_prob(
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    cfg: MetaConfig,
) -> pd.Series:
    """Compatibility helper returning only the crisis probability series."""
    return compute_crisis_signal(market_df, strat_ret_df, cfg)["p_crisis"].rename(
        "p_crisis"
    )
