"""Robustness testing framework for automated momentum backtesting.

Checks: neighborhood stability, regime stability, cost sensitivity,
walk-forward efficiency, and safety flags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from backtests.momentum.auto.scoring import CompositeScore, composite_score, extract_metrics
from backtests.momentum.config import SlippageConfig

if TYPE_CHECKING:
    from backtests.momentum.auto.experiments import Experiment

logger = logging.getLogger(__name__)

# 1.5x cost multiplier for sensitivity checks (futures)
_HIGH_COST_SLIPPAGE = SlippageConfig(
    commission_per_contract=0.93,   # 1.5x of 0.62
    slip_ticks_normal=2,
    slip_ticks_illiquid=3,
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
    run_fn: Callable[[dict], tuple[list, np.ndarray, np.ndarray, float]],
    pct: float = 0.10,
) -> tuple[bool, list[float]]:
    """Perturb continuous params +/-pct, check all scores >= 80% of center.

    Only applies to PARAM_SWEEP and INTERACTION experiments with numeric
    mutations. Ablation experiments are binary and skip this check.
    """
    numeric_keys = [
        k for k, v in experiment.mutations.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]

    if not numeric_keys:
        return True, []

    threshold = baseline_score * 0.80
    scores: list[float] = []

    for key in numeric_keys:
        base_val = experiment.mutations[key]
        for direction in [-1, +1]:
            perturbed_val = base_val * (1.0 + direction * pct)
            perturbed_mutations = dict(experiment.mutations)
            perturbed_mutations[key] = perturbed_val

            try:
                trades, eq, ts, init_eq = run_fn(perturbed_mutations)
                strategy = experiment.strategy
                metrics = extract_metrics(trades, eq, ts, init_eq)
                score = composite_score(metrics, init_eq, strategy=strategy, equity_curve=eq)
                scores.append(score.total)
            except Exception:
                logger.warning("Neighborhood perturbation failed for %s=%s",
                               key, perturbed_val)
                scores.append(0.0)

    stable = all(s >= threshold for s in scores) if scores else True
    return stable, scores


def regime_check(trades) -> tuple[bool, dict[str, float]]:
    """Check per-regime trade PnL. Stable if >= 2/3 regimes are net positive.

    For momentum, uses session window, entry class, or date-based halves
    as regime proxy.
    """
    regime_pnls: dict[str, list[float]] = {}

    for t in trades:
        regime = _get_regime(t)
        pnl = t.pnl_dollars if hasattr(t, 'pnl_dollars') else getattr(t, 'pnl', 0.0)
        regime_pnls.setdefault(regime, []).append(pnl)

    if not regime_pnls:
        return True, {}

    regime_scores: dict[str, float] = {}
    for regime, pnls in regime_pnls.items():
        regime_scores[regime] = sum(pnls)

    positive_count = sum(1 for v in regime_scores.values() if v > 0)
    total_regimes = len(regime_scores)

    # Stable if >= 2/3 of regimes are net positive (rounded up)
    threshold = max(1, -(-2 * total_regimes // 3))  # ceiling of 2/3
    stable = positive_count >= threshold

    return stable, regime_scores


def cost_sensitivity_check(
    experiment: Experiment,
    base_score: float,
    run_fn: Callable[[dict], tuple[list, np.ndarray, np.ndarray, float]],
) -> tuple[bool, float]:
    """Re-run with 1.5x costs. Sensitive if score drops below 60% of base."""
    cost_mutations = dict(experiment.mutations)
    cost_mutations["slippage.commission_per_contract"] = _HIGH_COST_SLIPPAGE.commission_per_contract
    cost_mutations["slippage.slip_ticks_normal"] = _HIGH_COST_SLIPPAGE.slip_ticks_normal
    cost_mutations["slippage.slip_ticks_illiquid"] = _HIGH_COST_SLIPPAGE.slip_ticks_illiquid

    try:
        trades, eq, ts, init_eq = run_fn(cost_mutations)
        metrics = extract_metrics(trades, eq, ts, init_eq)
        score = composite_score(metrics, init_eq, strategy=experiment.strategy, equity_curve=eq)
        cost_score = score.total
    except Exception:
        logger.warning("Cost sensitivity re-run failed for %s", experiment.id)
        return True, 0.0

    if base_score <= 0:
        return True, cost_score

    sensitive = cost_score < (base_score * 0.60)
    return sensitive, cost_score


def walk_forward_check(
    trades,
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
    strategy: str | None = None,
) -> tuple[bool, float]:
    """Compare first-half vs second-half efficiency. Stable if ratio > 0.5."""
    if len(trades) < 10:
        return True, 1.0

    mid = len(trades) // 2
    first_half = trades[:mid]
    second_half = trades[mid:]

    pnl_first = sum(_get_trade_pnl(t) for t in first_half) / max(len(first_half), 1)
    pnl_second = sum(_get_trade_pnl(t) for t in second_half) / max(len(second_half), 1)

    if pnl_first <= 0:
        efficiency = 1.0 if pnl_second > 0 else 0.0
    else:
        efficiency = pnl_second / pnl_first

    stable = efficiency > 0.5
    return stable, efficiency


def compute_safety_flags(
    trades,
    equity_curve: np.ndarray,
) -> list[str]:
    """Detect: spiky equity, outlier concentration, drawdown spike."""
    flags: list[str] = []

    if len(equity_curve) < 10:
        return flags

    # Spiky equity: check if returns are dominated by a few large moves
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 0:
            sorted_abs = np.sort(np.abs(returns))[::-1]
            total_abs = np.sum(np.abs(returns))
            if total_abs > 0:
                top_5_pct = sorted_abs[:max(1, len(sorted_abs) // 20)]
                if np.sum(top_5_pct) / total_abs > 0.50:
                    flags.append("spiky_equity")

    # Outlier concentration: top 3 trades account for > 60% of PnL
    if trades:
        pnls = [_get_trade_pnl(t) for t in trades]
        total_pnl = sum(pnls)
        if total_pnl > 0:
            sorted_pnls = sorted(pnls, reverse=True)
            top_3 = sum(sorted_pnls[:3])
            if top_3 / total_pnl > 0.60:
                flags.append("outlier_dependent")

    # Drawdown spike: any drawdown > 25% of equity
    peak = equity_curve[0]
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > 0.25:
            flags.append("drawdown_spike")
            break

    return flags


def run_robustness(
    experiment: Experiment,
    trades,
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
    initial_equity: float,
    experiment_score: float,
    run_fn: Callable[[dict], tuple[list, np.ndarray, np.ndarray, float]],
    skip_walk_forward: bool = False,
) -> RobustnessReport:
    """Run all robustness checks and return the report."""
    report = RobustnessReport()

    # 1. Neighborhood
    report.neighborhood_stable, report.neighborhood_scores = neighborhood_check(
        experiment, experiment_score, run_fn,
    )

    # 2. Regime
    report.regime_stable, report.regime_scores = regime_check(trades)

    # 3. Cost sensitivity
    report.cost_sensitive, report.cost_score = cost_sensitivity_check(
        experiment, experiment_score, run_fn,
    )

    # 4. Walk-forward
    if not skip_walk_forward:
        report.walk_forward_stable, report.walk_forward_efficiency = walk_forward_check(
            trades, equity_curve, timestamps, initial_equity,
            strategy=experiment.strategy,
        )

    # 5. Safety flags
    report.safety_flags = compute_safety_flags(trades, equity_curve)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_regime(trade) -> str:
    """Extract regime label from various trade record types."""
    for attr in ("entry_class", "setup_class", "regime_at_entry", "session"):
        val = getattr(trade, attr, None)
        if val:
            return str(val)
    # Fall back to time-of-day bucket
    entry_time = getattr(trade, 'entry_time', None)
    if entry_time is not None:
        hour = getattr(entry_time, 'hour', None)
        if hour is not None:
            if hour < 8:
                return "ETH"
            elif hour < 16:
                return "RTH"
            else:
                return "EVENING"
    return "UNKNOWN"


def _get_trade_pnl(trade) -> float:
    if hasattr(trade, 'pnl_dollars'):
        return trade.pnl_dollars
    return getattr(trade, 'pnl', 0.0)
