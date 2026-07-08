"""Phase-specific scoring functions for multi-phase regime optimization."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtests.regime.analysis.metrics import PortfolioMetrics
from backtests.regime.auto.scoring import CompositeScore

CRISIS_PERIODS = {
    "GFC": ("2008-09-01", "2009-03-01", "D"),
    "COVID": ("2020-02-15", "2020-04-15", "D"),
    "Inflation": ("2022-01-01", "2022-10-01", "S"),
}

REGIME_COLS = ["P_G", "P_R", "P_S", "P_D"]
REGIME_LABELS = ["G", "R", "S", "D"]
ALLOC_ASSETS = ["SPY", "EFA", "TLT", "GLD", "CASH"]
ALLOC_DIFF_FLOOR = 0.015


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def compute_regime_stats(signals: pd.DataFrame, L_max: float = 0.0) -> dict:
    """Compute regime health statistics from a signals DataFrame."""
    if signals.empty or not all(c in signals.columns for c in REGIME_COLS):
        return _empty_stats()

    probs = signals[REGIME_COLS].values
    dominant = np.argmax(probs, axis=1)
    regime_counts = np.bincount(dominant, minlength=4) / len(dominant)

    n_active = int(np.sum(regime_counts > 0.02))
    regime_entropy = _shannon_entropy(regime_counts)

    transitions = np.sum(dominant[1:] != dominant[:-1])
    transition_rate = transitions / len(dominant) if len(dominant) > 1 else 0.0

    posterior_entropies = np.array([_shannon_entropy(row) for row in probs])
    avg_posterior_entropy = float(np.mean(posterior_entropies))
    avg_p_dom = float(np.max(probs, axis=1).mean())

    conf_std = float(signals["Conf"].std()) if "Conf" in signals.columns else 0.0
    dominant_dist = {REGIME_LABELS[i]: float(regime_counts[i]) for i in range(4)}
    crisis_response = _compute_crisis_response(signals)
    crisis_accuracy = _compute_crisis_accuracy(signals)
    alloc_differentiation = _compute_alloc_differentiation(signals)
    posterior_penalty = _compute_posterior_penalty(avg_p_dom)
    transition_penalty = _compute_transition_penalty(transition_rate)
    leverage_compression_penalty = _compute_leverage_compression_penalty(signals, L_max)

    stress_calibration = _compute_stress_calibration(signals)
    stress_crisis_capture = _compute_stress_crisis_capture(signals)

    return {
        "n_active_regimes": n_active,
        "regime_entropy": regime_entropy,
        "transition_rate": transition_rate,
        "avg_posterior_entropy": avg_posterior_entropy,
        "avg_p_dom": avg_p_dom,
        "conf_std": conf_std,
        "dominant_dist": dominant_dist,
        "crisis_response": crisis_response,
        "crisis_accuracy": crisis_accuracy,
        "alloc_differentiation": alloc_differentiation,
        "posterior_penalty": posterior_penalty,
        "transition_penalty": transition_penalty,
        "leverage_compression_penalty": leverage_compression_penalty,
        "stress_calibration": stress_calibration,
        "stress_crisis_capture": stress_crisis_capture,
    }


def _empty_stats() -> dict:
    return {
        "n_active_regimes": 0,
        "regime_entropy": 0.0,
        "transition_rate": 0.0,
        "avg_posterior_entropy": 0.0,
        "avg_p_dom": 0.0,
        "conf_std": 0.0,
        "dominant_dist": {"G": 0, "R": 0, "S": 0, "D": 0},
        "crisis_response": 0.0,
        "crisis_accuracy": 0.0,
        "alloc_differentiation": 0.0,
        "posterior_penalty": 1.0,
        "transition_penalty": 1.0,
        "leverage_compression_penalty": 0.0,
        "stress_calibration": 0.0,
        "stress_crisis_capture": 0.0,
    }


def _shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / np.log(4))


def _compute_crisis_response(signals: pd.DataFrame) -> float:
    regime_col_map = {"G": "P_G", "R": "P_R", "S": "P_S", "D": "P_D"}
    scores = []

    for start, end, expected in CRISIS_PERIODS.values():
        crisis_slice = signals.loc[start:end]
        if crisis_slice.empty:
            continue
        scores.append(float(crisis_slice[regime_col_map[expected]].mean()))

    return float(np.mean(scores)) if scores else 0.0


def _compute_crisis_accuracy(signals: pd.DataFrame) -> float:
    scores = []

    gfc = signals.loc["2008-09-01":"2009-03-01"]
    if not gfc.empty:
        scores.append(float(gfc["P_D"].mean()))

    covid = signals.loc["2020-02-15":"2020-04-15"]
    if not covid.empty:
        scores.append(float(covid[["P_D", "P_S"]].max(axis=1).mean()))

    inflation = signals.loc["2022-01-01":"2022-10-01"]
    if not inflation.empty:
        scores.append(float(inflation[["P_D", "P_S"]].max(axis=1).mean()))

    return float(np.mean(scores)) if scores else 0.0


def _compute_alloc_differentiation(signals: pd.DataFrame) -> float:
    alloc_cols = [f"w_{asset}" for asset in ALLOC_ASSETS if f"w_{asset}" in signals.columns]
    if not alloc_cols:
        alloc_cols = [f"pi_{asset}" for asset in ALLOC_ASSETS if f"pi_{asset}" in signals.columns]
    if not alloc_cols:
        return 0.0

    dominant = signals[REGIME_COLS].idxmax(axis=1).str.removeprefix("P_")
    grouped = signals[alloc_cols].groupby(dominant).mean()
    group_sizes = dominant.value_counts()
    min_group_size = max(2, min(10, len(dominant) // 50))
    active = [r for r in grouped.index.tolist() if group_sizes.get(r, 0) >= min_group_size]
    if len(active) < 2:
        return 0.0

    ranges = []
    for col in alloc_cols:
        col_vals = grouped.loc[active, col]
        ranges.append(float(col_vals.max() - col_vals.min()))
    return float(np.mean(ranges)) if ranges else 0.0


def _compute_stress_calibration(signals: pd.DataFrame) -> float:
    """Score how well stress firing rate matches calibration targets."""
    if "stress_level" not in signals.columns:
        return 0.0
    sl = signals["stress_level"]
    elevated_pct = float((sl > 0.3).mean())   # target: 15-20%
    acute_pct = float((sl > 0.7).mean())       # target: ~5%
    elevated_score = _clip01(1.0 - abs(elevated_pct - 0.175) / 0.15)
    acute_score = _clip01(1.0 - abs(acute_pct - 0.05) / 0.05)
    return 0.6 * elevated_score + 0.4 * acute_score


def _compute_stress_crisis_capture(signals: pd.DataFrame) -> float:
    """Mean P(stress) during known crisis periods."""
    if "stress_level" not in signals.columns:
        return 0.0
    scores = []
    gfc = signals.loc["2008-09-01":"2009-03-01"]
    if not gfc.empty:
        scores.append(float(gfc["stress_level"].mean()))
    covid = signals.loc["2020-02-15":"2020-04-15"]
    if not covid.empty:
        scores.append(float(covid["stress_level"].mean()))
    infl = signals.loc["2022-01-01":"2022-10-01"]
    if not infl.empty:
        scores.append(float(infl["stress_level"].mean()))
    return float(np.mean(scores)) if scores else 0.0


def _compute_posterior_penalty(avg_p_dom: float) -> float:
    if avg_p_dom < 0.60:
        return _clip01((0.60 - avg_p_dom) / 0.35)
    if avg_p_dom > 0.95:
        return _clip01((avg_p_dom - 0.95) / 0.05)
    return 0.0


def _compute_leverage_compression_penalty(signals: pd.DataFrame, L_max: float) -> float:
    """Penalize configs where mean leverage is below 50% of L_max."""
    if "L" not in signals.columns or L_max <= 0:
        return 0.0
    mean_leverage = float(signals["L"].mean())
    utilization = mean_leverage / L_max
    if utilization >= 0.50:
        return 0.0
    return _clip01((0.50 - utilization) / 0.30)


def _compute_transition_penalty(transition_rate: float) -> float:
    if transition_rate > 0.05:
        return _clip01((transition_rate - 0.05) / 0.10)
    if transition_rate < 0.005:
        return _clip01((0.005 - transition_rate) / 0.005)
    return 0.0


def _financial_floor(metrics: PortfolioMetrics) -> str | None:
    if metrics.sharpe < 1.2:
        return f"Sharpe {metrics.sharpe:.3f} < 1.2"
    if metrics.max_drawdown_pct > 0.12:
        return f"Max DD {metrics.max_drawdown_pct:.1%} > 12%"
    return None


def _allocation_differentiation_floor(regime_stats: dict) -> str | None:
    alloc_diff = float(regime_stats.get("alloc_differentiation", 0.0))
    if alloc_diff < ALLOC_DIFF_FLOOR:
        alloc_bp = int(round(alloc_diff * 10_000))
        floor_bp = int(round(ALLOC_DIFF_FLOOR * 10_000))
        return f"Alloc differentiation {alloc_bp}bp < {floor_bp}bp"
    return None


def _rejected_score(reason: str) -> CompositeScore:
    return CompositeScore(0, 0, 0, 0, 0, rejected=True, reject_reason=reason)


def phase_1_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    reject_reason = _financial_floor(metrics)
    if reject_reason:
        return _rejected_score(reject_reason)
    reject_reason = _allocation_differentiation_floor(regime_stats)
    if reject_reason:
        return _rejected_score(reject_reason)
    if regime_stats["n_active_regimes"] < 3:
        return _rejected_score(
            f"Only {regime_stats['n_active_regimes']} active regimes < 3"
        )
    if regime_stats["transition_rate"] < 0.003:
        return _rejected_score(
            f"Transition rate {regime_stats['transition_rate']:.4f} < 0.003"
        )

    sharpe_c = _clip01(metrics.sharpe / 1.5)
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.12)
    cagr_c = (
        _clip01(math.log(1 + metrics.cagr) / math.log(1.15))
        if metrics.cagr > 0
        else 0.0
    )
    entropy_c = _clip01(regime_stats["regime_entropy"])
    crisis_accuracy_c = _clip01(regime_stats["crisis_accuracy"])
    alloc_diff_c = _clip01(regime_stats["alloc_differentiation"] / 0.25)
    crisis_c = _clip01(regime_stats["crisis_response"])
    penalty_c = 0.5 * (
        regime_stats["posterior_penalty"] + regime_stats["transition_penalty"]
    )
    leverage_penalty = regime_stats.get("leverage_compression_penalty", 0.0)

    total = (
        0.25 * sharpe_c
        + 0.20 * inv_dd_c
        + 0.10 * cagr_c
        + 0.10 * entropy_c
        + 0.10 * crisis_accuracy_c
        + 0.10 * alloc_diff_c
        + 0.10 * crisis_c
        - 0.15 * penalty_c
        - 0.05 * leverage_penalty
    )

    # NOTE: Phase 1 repurposes CompositeScore financial fields:
    #   calmar_component  -> crisis_accuracy_c
    #   sortino_component -> entropy_c
    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=crisis_accuracy_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=entropy_c,
        total=total,
    )


def phase_2_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    reject_reason = _financial_floor(metrics)
    if reject_reason:
        return _rejected_score(reject_reason)
    reject_reason = _allocation_differentiation_floor(regime_stats)
    if reject_reason:
        return _rejected_score(reject_reason)
    if regime_stats["n_active_regimes"] < 3:
        return _rejected_score(
            f"Only {regime_stats['n_active_regimes']} active regimes < 3"
        )

    sharpe_c = _clip01(metrics.sharpe / 1.5)
    calmar_c = _clip01(metrics.calmar / 5.0)
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.35)
    cagr_c = (
        _clip01(math.log(1 + metrics.cagr) / math.log(1.15))
        if metrics.cagr > 0
        else 0.0
    )
    entropy_c = _clip01(regime_stats["regime_entropy"])
    crisis_accuracy_c = _clip01(regime_stats["crisis_accuracy"])
    alloc_diff_c = _clip01(regime_stats["alloc_differentiation"] / 0.25)
    penalty_c = 0.5 * (
        regime_stats["posterior_penalty"] + regime_stats["transition_penalty"]
    )
    leverage_penalty = regime_stats.get("leverage_compression_penalty", 0.0)

    total = (
        0.20 * sharpe_c
        + 0.15 * calmar_c
        + 0.15 * inv_dd_c
        + 0.10 * cagr_c
        + 0.10 * entropy_c
        + 0.10 * crisis_accuracy_c
        + 0.15 * alloc_diff_c
        - 0.10 * penalty_c
        - 0.05 * leverage_penalty
    )

    # NOTE: Phase 2 repurposes CompositeScore fields:
    #   sortino_component -> entropy_c
    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=entropy_c,
        total=total,
    )


def phase_3_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    reject_reason = _financial_floor(metrics)
    if reject_reason:
        return _rejected_score(reject_reason)
    reject_reason = _allocation_differentiation_floor(regime_stats)
    if reject_reason:
        return _rejected_score(reject_reason)
    if regime_stats["n_active_regimes"] < 3:
        return _rejected_score(
            f"Only {regime_stats['n_active_regimes']} active regimes < 3"
        )

    sharpe_c = _clip01(metrics.sharpe / 1.5)
    calmar_c = _clip01(metrics.calmar / 5.0)
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.12)
    cagr_c = (
        _clip01(math.log(1 + metrics.cagr) / math.log(1.15))
        if metrics.cagr > 0
        else 0.0
    )
    sortino_c = _clip01(metrics.sortino / 2.5)
    crisis_c = _clip01(regime_stats["crisis_response"])
    entropy_c = _clip01(regime_stats["regime_entropy"])
    crisis_accuracy_c = _clip01(regime_stats["crisis_accuracy"])
    alloc_diff_c = _clip01(regime_stats["alloc_differentiation"] / 0.25)
    penalty_c = 0.5 * (
        regime_stats["posterior_penalty"] + regime_stats["transition_penalty"]
    )
    leverage_penalty = regime_stats.get("leverage_compression_penalty", 0.0)

    total = (
        0.20 * sharpe_c
        + 0.15 * calmar_c
        + 0.10 * inv_dd_c
        + 0.10 * cagr_c
        + 0.10 * sortino_c
        + 0.05 * crisis_c
        + 0.05 * entropy_c
        + 0.10 * crisis_accuracy_c
        + 0.10 * alloc_diff_c
        - 0.05 * penalty_c
        - 0.05 * leverage_penalty
    )

    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=sortino_c,
        total=total,
    )


def _financial_floor_phase4(metrics: PortfolioMetrics) -> str | None:
    """Tighter floor for Phase 4 -- regime health should already be solved."""
    if metrics.sharpe < 1.3:
        return f"Sharpe {metrics.sharpe:.3f} < 1.3 (Phase 4 floor)"
    if metrics.max_drawdown_pct > 0.10:
        return f"Max DD {metrics.max_drawdown_pct:.1%} > 10% (Phase 4 floor)"
    return None


def phase_4_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    reject_reason = _financial_floor_phase4(metrics)
    if reject_reason:
        return _rejected_score(reject_reason)
    reject_reason = _allocation_differentiation_floor(regime_stats)
    if reject_reason:
        return _rejected_score(reject_reason)
    if regime_stats["n_active_regimes"] < 3:
        return _rejected_score(
            f"Only {regime_stats['n_active_regimes']} active regimes < 3"
        )

    sharpe_c = _clip01(metrics.sharpe / 1.5)
    calmar_c = _clip01(metrics.calmar / 5.0)
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.10)
    cagr_c = (
        _clip01(math.log(1 + metrics.cagr) / math.log(1.15))
        if metrics.cagr > 0
        else 0.0
    )
    sortino_c = _clip01(metrics.sortino / 2.5)
    crisis_accuracy_c = _clip01(regime_stats["crisis_accuracy"])
    alloc_diff_c = _clip01(regime_stats["alloc_differentiation"] / 0.25)
    penalty_c = 0.5 * (
        regime_stats["posterior_penalty"] + regime_stats["transition_penalty"]
    )
    leverage_penalty = regime_stats.get("leverage_compression_penalty", 0.0)

    # Phase 4: crisis_response and entropy weights zeroed (solved by Phases 1-3).
    # crisis_accuracy and alloc_diff increased to 0.15 each for final polish.
    total = (
        0.20 * sharpe_c
        + 0.15 * calmar_c
        + 0.10 * inv_dd_c
        + 0.10 * cagr_c
        + 0.10 * sortino_c
        + 0.15 * crisis_accuracy_c
        + 0.15 * alloc_diff_c
        - 0.05 * penalty_c
        - 0.05 * leverage_penalty
    )

    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=sortino_c,
        total=total,
    )


def phase_5_score(metrics: PortfolioMetrics, regime_stats: dict) -> CompositeScore:
    """Phase 5: Budget-only optimization -- reweights for drawdown-adjusted return."""
    reject_reason = _financial_floor_phase4(metrics)
    if reject_reason:
        return _rejected_score(reject_reason)
    reject_reason = _allocation_differentiation_floor(regime_stats)
    if reject_reason:
        return _rejected_score(reject_reason)
    if regime_stats["n_active_regimes"] < 3:
        return _rejected_score(
            f"Only {regime_stats['n_active_regimes']} active regimes < 3"
        )

    sharpe_c = _clip01(metrics.sharpe / 1.5)
    calmar_c = _clip01(metrics.calmar / 5.0)
    inv_dd_c = _clip01(1.0 - metrics.max_drawdown_pct / 0.10)
    cagr_c = (
        _clip01(math.log(1 + metrics.cagr) / math.log(1.15))
        if metrics.cagr > 0
        else 0.0
    )
    sortino_c = _clip01(metrics.sortino / 2.5)
    crisis_accuracy_c = _clip01(regime_stats["crisis_accuracy"])
    alloc_diff_c = _clip01(regime_stats["alloc_differentiation"] / 0.25)
    leverage_penalty = regime_stats.get("leverage_compression_penalty", 0.0)

    # Phase 5: Calmar and Sortino elevated (budget's main levers are
    # drawdown-adjusted return and downside risk). Penalty term dropped
    # (posterior/transition penalties are HMM-invariant under budget changes).
    total = (
        0.15 * sharpe_c
        + 0.20 * calmar_c
        + 0.10 * inv_dd_c
        + 0.10 * cagr_c
        + 0.20 * sortino_c
        + 0.10 * crisis_accuracy_c
        + 0.15 * alloc_diff_c
        - 0.05 * leverage_penalty
    )

    return CompositeScore(
        sharpe_component=sharpe_c,
        calmar_component=calmar_c,
        inv_dd_component=inv_dd_c,
        cagr_component=cagr_c,
        sortino_component=sortino_c,
        total=total,
    )


_PHASE_SCORERS = {
    1: phase_1_score,
    2: phase_2_score,
    3: phase_3_score,
    4: phase_4_score,
    5: phase_5_score,
}


def get_phase_scorer(phase: int):
    """Return the scoring function for a given phase."""
    if phase not in _PHASE_SCORERS:
        raise ValueError(f"Unknown phase: {phase}. Valid phases: 1-5")
    return _PHASE_SCORERS[phase]
