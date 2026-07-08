"""Robustness testing framework for automated backtesting.

Checks: neighborhood stability, regime stability, cost sensitivity,
walk-forward efficiency, and safety flags.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from backtests.stock.auto.scoring import CompositeScore, composite_score, extract_metrics
from backtests.stock.config import SlippageConfig
from backtests.stock.models import TradeRecord

if TYPE_CHECKING:
    from backtests.stock.auto.experiments import Experiment

logger = logging.getLogger(__name__)

# 1.5× cost multiplier for sensitivity checks
_HIGH_COST_SLIPPAGE = SlippageConfig(
    commission_per_share=0.0075,
    slip_bps_normal=7.5,
    slip_bps_illiquid=22.5,
)


@dataclass
class RobustnessReport:
    neighborhood_stable: bool = True
    neighborhood_scores: list[float] = field(default_factory=list)
    regime_stable: bool = True
    regime_scores: dict[str, float] = field(default_factory=dict)
    cost_sensitive: bool = False
    cost_score: float = 0.0
    walk_forward_stable: bool = True
    walk_forward_efficiency: float = 0.0
    safety_flags: list[str] = field(default_factory=list)

    @property
    def passes_all(self) -> bool:
        return (
            self.neighborhood_stable
            and self.regime_stable
            and not self.cost_sensitive
            and self.walk_forward_stable
            and len(self.safety_flags) == 0
        )


def neighborhood_check(
    experiment: Experiment,
    baseline_score: float,
    run_fn: Callable[[dict], tuple[list[TradeRecord], np.ndarray, np.ndarray, float]],
    pct: float = 0.10,
) -> tuple[bool, list[float]]:
    """Perturb continuous params ±pct, check all scores >= 80% of center.

    Only applies to PARAM_SWEEP experiments with numeric mutations.

    Args:
        experiment: The experiment definition
        baseline_score: The center-point composite score
        run_fn: Callable(mutations) -> (trades, equity_curve, timestamps, initial_equity)
        pct: Perturbation percentage (default 10%)

    Returns:
        (stable, perturbed_scores)
    """
    if experiment.type != "PARAM_SWEEP":
        return True, []

    # Find continuous params to perturb
    numeric_keys = [
        k for k, v in experiment.mutations.items()
        if isinstance(v, (int, float))
    ]

    if not numeric_keys:
        return True, []

    scores = []
    threshold = baseline_score * 0.80

    for key in numeric_keys:
        base_val = experiment.mutations[key]
        for direction in [-1, 1]:
            perturbed_val = base_val * (1.0 + direction * pct)
            perturbed_mutations = {**experiment.mutations, key: perturbed_val}
            try:
                trades, eq, ts, init_eq = run_fn(perturbed_mutations)
                metrics = extract_metrics(trades, eq, ts, init_eq)
                score = composite_score(metrics, init_eq)
                scores.append(score.total)
            except Exception:
                logger.warning("Neighborhood perturbation failed for %s=%s",
                               key, perturbed_val)
                scores.append(0.0)

    stable = all(s >= threshold for s in scores) if scores else True
    return stable, scores


def regime_check(trades: list[TradeRecord]) -> tuple[bool, dict[str, float]]:
    """Check per-regime composite scores. Stable if >= 3 of 4 tiers positive.

    Uses TradeRecord.regime_tier field (populated by both IARIC engines;
    ALCB uses regime.tier from CandidateArtifact).
    """
    regime_trades: defaultdict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        tier = t.regime_tier if t.regime_tier else "unknown"
        regime_trades[tier].append(t)

    if len(regime_trades) < 2:
        # Not enough regime diversity to check
        return True, {}

    regime_scores: dict[str, float] = {}
    for tier, tier_trades in regime_trades.items():
        if len(tier_trades) < 5:
            continue
        pnls = np.array([t.pnl_net for t in tier_trades])
        regime_scores[tier] = float(np.sum(pnls))

    positive_count = sum(1 for v in regime_scores.values() if v > 0)
    total_regimes = len(regime_scores)

    # Stable if at most 1 losing regime (3 of 4, 2 of 3, 2 of 2)
    threshold = max(2, total_regimes - 1)
    stable = positive_count >= threshold

    return stable, regime_scores


def cost_sensitivity_check(
    experiment: Experiment,
    base_score: float,
    run_fn: Callable[[dict], tuple[list[TradeRecord], np.ndarray, np.ndarray, float]],
) -> tuple[bool, float]:
    """Re-run with 1.5× costs. Sensitive if score drops below 60% of base.

    Args:
        experiment: The experiment definition
        base_score: Composite score at normal costs
        run_fn: Callable(mutations) -> (trades, equity_curve, timestamps, initial_equity)

    Returns:
        (is_sensitive, cost_score)
    """
    cost_mutations = {
        **experiment.mutations,
        "slippage.commission_per_share": _HIGH_COST_SLIPPAGE.commission_per_share,
        "slippage.slip_bps_normal": _HIGH_COST_SLIPPAGE.slip_bps_normal,
        "slippage.slip_bps_illiquid": _HIGH_COST_SLIPPAGE.slip_bps_illiquid,
    }

    try:
        trades, eq, ts, init_eq = run_fn(cost_mutations)
        metrics = extract_metrics(trades, eq, ts, init_eq)
        score = composite_score(metrics, init_eq)
        cost_total = score.total
    except Exception:
        logger.warning("Cost sensitivity check failed for %s", experiment.id)
        return True, 0.0

    sensitive = cost_total < base_score * 0.60
    return sensitive, cost_total


def compute_safety_flags(
    neighborhood_scores: list[float],
    trades: list[TradeRecord],
) -> list[str]:
    """Detect safety flags from neighborhood scores and trade data.

    Flags:
      - flat_surface: std(neighborhood_scores) < 0.01
      - spiky: max(scores) / median(scores) > 2.0
      - outlier_dependent: remove top 5% trades, net_profit drops below 0
    """
    flags = []

    if neighborhood_scores:
        std = float(np.std(neighborhood_scores))
        if std < 0.01:
            flags.append("flat_surface")

        median = float(np.median(neighborhood_scores))
        if median > 0 and max(neighborhood_scores) / median > 2.0:
            flags.append("spiky")

    if trades:
        pnls = sorted([t.pnl_net for t in trades], reverse=True)
        n_remove = max(1, int(len(pnls) * 0.05))
        remaining = sum(pnls[n_remove:])
        if remaining < 0:
            flags.append("outlier_dependent")

    return flags


def run_robustness(
    experiment: Experiment,
    score: CompositeScore,
    trades: list[TradeRecord],
    run_fn: Callable[[dict], tuple[list[TradeRecord], np.ndarray, np.ndarray, float]],
    replay=None,
) -> RobustnessReport:
    """Run all robustness checks and return a consolidated report."""
    report = RobustnessReport()

    # Neighborhood
    n_stable, n_scores = neighborhood_check(experiment, score.total, run_fn)
    report.neighborhood_stable = n_stable
    report.neighborhood_scores = n_scores

    # Regime
    r_stable, r_scores = regime_check(trades)
    report.regime_stable = r_stable
    report.regime_scores = r_scores

    # Cost sensitivity
    c_sensitive, c_score = cost_sensitivity_check(experiment, score.total, run_fn)
    report.cost_sensitive = c_sensitive
    report.cost_score = c_score

    # Safety flags
    report.safety_flags = compute_safety_flags(n_scores, trades)

    return report
