"""Helix phase gate criteria -- success thresholds per phase.

Phase 1: signal viability after cumulative seed rebuild.
Phase 2: leakage/stale/partial repair with enough win-rate lift to adopt.
Phase 3: stop/trailing payoff repair to reach the >300 trade / >50% WR target.
Phase 4: volatility/add-on expansion without giving back prior phase expectancy.
Phase 5: exit-sensitive fine-tune with return and winner-count no-regression.
Phase 6: final circuit/remaining fine-tune with final target discipline.
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import HelixMetrics


def gate_criteria_phase_1(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    winners = _winning_trades(m)
    return [
        *_base_health_criteria(m),
        GateCriterion("total_trades", 270.0, float(m.total_trades), m.total_trades >= 270),
        GateCriterion("winning_trades", 103.0, winners, winners >= 103.0),
        GateCriterion("profit_factor", 1.20, m.profit_factor, m.profit_factor >= 1.20),
        GateCriterion("min_side_pf", 0.95, m.min_side_pf, m.min_side_pf >= 0.95),
        GateCriterion("net_return_pct", 25.0, m.net_return_pct, m.net_return_pct >= 25.0),
        GateCriterion("tail_pct", 0.30, m.tail_pct, m.tail_pct >= 0.30),
    ]


def gate_criteria_phase_2(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    winners = _winning_trades(m)
    return [
        *_base_health_criteria(m),
        GateCriterion("total_trades", 300.0, float(m.total_trades), m.total_trades >= 300),
        GateCriterion("win_rate", 47.0, m.win_rate, m.win_rate >= 47.0),
        GateCriterion("winning_trades", 145.0, winners, winners >= 145.0),
        GateCriterion("profit_factor", 1.25, m.profit_factor, m.profit_factor >= 1.25),
        GateCriterion("net_return_pct", 75.0, m.net_return_pct, m.net_return_pct >= 75.0),
        GateCriterion("exit_efficiency", 0.25, m.exit_efficiency, m.exit_efficiency >= 0.25),
        GateCriterion("waste_ratio", 0.55, m.waste_ratio, m.waste_ratio >= 0.55),
    ]


def gate_criteria_phase_3(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    winners = _winning_trades(m)
    return [
        *_base_health_criteria(m),
        GateCriterion("total_trades", 300.0, float(m.total_trades), m.total_trades >= 300),
        GateCriterion("win_rate", 50.0, m.win_rate, m.win_rate >= 50.0),
        GateCriterion("winning_trades", 150.0, winners, winners >= 150.0),
        GateCriterion("profit_factor", 1.30, m.profit_factor, m.profit_factor >= 1.30),
        GateCriterion("net_return_pct", 80.0, m.net_return_pct, m.net_return_pct >= 80.0),
        GateCriterion("total_r", 110.0, m.total_r, m.total_r >= 110.0),
        GateCriterion("avg_win_r", 1.50, m.avg_win_r, m.avg_win_r >= 1.50),
    ]


def gate_criteria_phase_4(
    metrics: dict[str, float],
    prior_phase_metrics: dict[str, float] | None = None,
) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    winners = _winning_trades(m)
    criteria = [
        *_base_health_criteria(m),
        GateCriterion("total_trades", 300.0, float(m.total_trades), m.total_trades >= 300),
        GateCriterion("win_rate", 50.0, m.win_rate, m.win_rate >= 50.0),
        GateCriterion("winning_trades", 150.0, winners, winners >= 150.0),
        GateCriterion("profit_factor", 1.30, m.profit_factor, m.profit_factor >= 1.30),
        GateCriterion("net_return_pct", 85.0, m.net_return_pct, m.net_return_pct >= 85.0),
        GateCriterion("total_r", 115.0, m.total_r, m.total_r >= 115.0),
    ]
    criteria.extend(_no_regression_criteria(m, prior_phase_metrics, net_ratio=0.90, total_r_ratio=0.90))
    return criteria


def gate_criteria_phase_5(
    metrics: dict[str, float],
    prior_phase_metrics: dict[str, float] | None = None,
) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    winners = _winning_trades(m)
    criteria = [
        *_base_health_criteria(m),
        GateCriterion("total_trades", 300.0, float(m.total_trades), m.total_trades >= 300),
        GateCriterion("win_rate", 50.0, m.win_rate, m.win_rate >= 50.0),
        GateCriterion("winning_trades", 150.0, winners, winners >= 150.0),
        GateCriterion("profit_factor", 1.35, m.profit_factor, m.profit_factor >= 1.35),
        GateCriterion("net_return_pct", 90.0, m.net_return_pct, m.net_return_pct >= 90.0),
        GateCriterion("total_r", 125.0, m.total_r, m.total_r >= 125.0),
        GateCriterion("avg_win_r", 1.55, m.avg_win_r, m.avg_win_r >= 1.55),
    ]
    criteria.extend(_no_regression_criteria(m, prior_phase_metrics, net_ratio=0.92, total_r_ratio=0.92))
    return criteria


def gate_criteria_phase_6(
    metrics: dict[str, float],
    prior_phase_metrics: dict[str, float] | None = None,
) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    winners = _winning_trades(m)
    criteria = [
        *_base_health_criteria(m),
        GateCriterion("total_trades", 300.0, float(m.total_trades), m.total_trades >= 300),
        GateCriterion("win_rate", 50.0, m.win_rate, m.win_rate >= 50.0),
        GateCriterion("winning_trades", 150.0, winners, winners >= 150.0),
        GateCriterion("profit_factor", 1.35, m.profit_factor, m.profit_factor >= 1.35),
        GateCriterion("net_return_pct", 90.0, m.net_return_pct, m.net_return_pct >= 90.0),
        GateCriterion("total_r", 125.0, m.total_r, m.total_r >= 125.0),
        GateCriterion("avg_win_r", 1.55, m.avg_win_r, m.avg_win_r >= 1.55),
        GateCriterion("calmar_r", 3.0, m.calmar_r, m.calmar_r >= 3.0),
        GateCriterion("max_r_dd", 20.0, m.max_r_dd, m.max_r_dd <= 20.0),
    ]
    criteria.extend(_no_regression_criteria(m, prior_phase_metrics, net_ratio=0.92, total_r_ratio=0.92))
    return criteria


def _base_health_criteria(metrics: HelixMetrics) -> list[GateCriterion]:
    return [
        GateCriterion("hard_min_trades", 270.0, float(metrics.total_trades), metrics.total_trades >= 270),
        GateCriterion("hard_min_pf", 1.15, metrics.profit_factor, metrics.profit_factor >= 1.15),
        GateCriterion("hard_max_r_dd", 22.0, metrics.max_r_dd, metrics.max_r_dd <= 22.0),
        GateCriterion("hard_min_regime_pf", 0.75, metrics.min_regime_pf, metrics.min_regime_pf >= 0.75),
        GateCriterion("hard_min_side_pf", 0.75, metrics.min_side_pf, metrics.min_side_pf >= 0.75),
    ]


def _no_regression_criteria(
    metrics: HelixMetrics,
    prior_phase_metrics: dict[str, float] | None,
    *,
    net_ratio: float,
    total_r_ratio: float,
) -> list[GateCriterion]:
    if not prior_phase_metrics:
        return []

    prior = _to_metrics(prior_phase_metrics)
    prior_winners = _winning_trades(prior)
    criteria: list[GateCriterion] = []
    if prior.profit_factor > 0:
        criteria.append(GateCriterion(
            "no_regression_pf",
            prior.profit_factor * 0.90,
            metrics.profit_factor,
            metrics.profit_factor >= prior.profit_factor * 0.90,
        ))
    if prior.net_return_pct > 0:
        criteria.append(GateCriterion(
            "no_regression_net_return",
            prior.net_return_pct * net_ratio,
            metrics.net_return_pct,
            metrics.net_return_pct >= prior.net_return_pct * net_ratio,
        ))
    if prior.total_r > 0:
        criteria.append(GateCriterion(
            "no_regression_total_r",
            prior.total_r * total_r_ratio,
            metrics.total_r,
            metrics.total_r >= prior.total_r * total_r_ratio,
        ))
    if prior.win_rate > 0:
        criteria.append(GateCriterion(
            "no_regression_win_rate",
            max(50.0, prior.win_rate - 1.0),
            metrics.win_rate,
            metrics.win_rate >= max(50.0, prior.win_rate - 1.0),
        ))
    if prior_winners > 0:
        current_winners = _winning_trades(metrics)
        criteria.append(GateCriterion(
            "no_regression_winning_trades",
            max(150.0, prior_winners * 0.97),
            current_winners,
            current_winners >= max(150.0, prior_winners * 0.97),
        ))
    return criteria


def _to_metrics(metrics: dict[str, float]) -> HelixMetrics:
    fields = HelixMetrics.__dataclass_fields__
    kwargs = {key: metrics.get(key, 0.0) for key in fields}
    kwargs["total_trades"] = int(kwargs.get("total_trades", 0))
    return HelixMetrics(**kwargs)


def _winning_trades(metrics: HelixMetrics) -> float:
    return float(metrics.winning_trades or (float(metrics.total_trades) * float(metrics.win_rate) / 100.0))
