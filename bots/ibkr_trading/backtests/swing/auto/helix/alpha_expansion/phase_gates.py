"""Phase gates for Helix alpha expansion.

Gates are anchored to the synchronized round 1 baseline so the optimizer can
increase frequency/return without accepting low-quality or brittle variants.
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion
from backtests.swing.auto.helix.scoring import HelixMetrics


def gate_criteria_phase_1(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 320.0, float(m.total_trades), m.total_trades >= 320),
        GateCriterion("hard_min_pf", 2.0, m.profit_factor, m.profit_factor >= 2.0),
        GateCriterion("hard_max_r_dd", 12.0, m.max_r_dd, m.max_r_dd <= 12.0),
        GateCriterion("total_trades", 330.0, float(m.total_trades), m.total_trades >= 330),
        GateCriterion("profit_factor", 2.35, m.profit_factor, m.profit_factor >= 2.35),
        GateCriterion("net_return_pct", 180.0, m.net_return_pct, m.net_return_pct >= 180.0),
        GateCriterion("tail_pct", 0.55, m.tail_pct, m.tail_pct >= 0.55),
    ]


def gate_criteria_phase_2(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 320.0, float(m.total_trades), m.total_trades >= 320),
        GateCriterion("hard_min_pf", 2.0, m.profit_factor, m.profit_factor >= 2.0),
        GateCriterion("hard_max_r_dd", 12.0, m.max_r_dd, m.max_r_dd <= 12.0),
        GateCriterion("total_trades", 345.0, float(m.total_trades), m.total_trades >= 345),
        GateCriterion("profit_factor", 2.25, m.profit_factor, m.profit_factor >= 2.25),
        GateCriterion("net_return_pct", 195.0, m.net_return_pct, m.net_return_pct >= 195.0),
        GateCriterion("exit_efficiency", 0.40, m.exit_efficiency, m.exit_efficiency >= 0.40),
    ]


def gate_criteria_phase_3(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 320.0, float(m.total_trades), m.total_trades >= 320),
        GateCriterion("hard_min_pf", 2.0, m.profit_factor, m.profit_factor >= 2.0),
        GateCriterion("hard_max_r_dd", 12.0, m.max_r_dd, m.max_r_dd <= 12.0),
        GateCriterion("total_trades", 340.0, float(m.total_trades), m.total_trades >= 340),
        GateCriterion("profit_factor", 2.25, m.profit_factor, m.profit_factor >= 2.25),
        GateCriterion("net_return_pct", 205.0, m.net_return_pct, m.net_return_pct >= 205.0),
        GateCriterion("tail_pct", 0.55, m.tail_pct, m.tail_pct >= 0.55),
    ]


def gate_criteria_phase_4(
    metrics: dict[str, float],
    prior_phase_metrics: dict[str, float] | None = None,
) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    criteria = [
        GateCriterion("hard_min_trades", 320.0, float(m.total_trades), m.total_trades >= 320),
        GateCriterion("hard_min_pf", 2.0, m.profit_factor, m.profit_factor >= 2.0),
        GateCriterion("hard_max_r_dd", 12.0, m.max_r_dd, m.max_r_dd <= 12.0),
        GateCriterion("total_trades", 340.0, float(m.total_trades), m.total_trades >= 340),
        GateCriterion("profit_factor", 2.25, m.profit_factor, m.profit_factor >= 2.25),
        GateCriterion("net_return_pct", 205.0, m.net_return_pct, m.net_return_pct >= 205.0),
        GateCriterion("calmar_r", 20.0, m.calmar_r, m.calmar_r >= 20.0),
    ]

    if prior_phase_metrics:
        p3_pf = float(prior_phase_metrics.get("profit_factor", 0.0))
        p3_nr = float(prior_phase_metrics.get("net_return_pct", 0.0))
        p3_trades = float(prior_phase_metrics.get("total_trades", 0.0))
        p3_dd = float(prior_phase_metrics.get("max_r_dd", 0.0))
        if p3_pf > 0:
            criteria.append(GateCriterion("no_regression_pf", p3_pf * 0.95, m.profit_factor, m.profit_factor >= p3_pf * 0.95))
        if p3_nr > 0:
            criteria.append(GateCriterion("no_regression_net_return", p3_nr * 0.95, m.net_return_pct, m.net_return_pct >= p3_nr * 0.95))
        if p3_trades > 0:
            criteria.append(GateCriterion("no_regression_trades", p3_trades * 0.95, float(m.total_trades), m.total_trades >= p3_trades * 0.95))
        if p3_dd > 0:
            criteria.append(GateCriterion("no_regression_r_dd", p3_dd * 1.10, m.max_r_dd, m.max_r_dd <= p3_dd * 1.10))

    return criteria


def _to_metrics(metrics: dict[str, float]) -> HelixMetrics:
    fields = HelixMetrics.__dataclass_fields__
    kwargs = {key: metrics.get(key, 0.0) for key in fields}
    kwargs["total_trades"] = int(kwargs.get("total_trades", 0))
    return HelixMetrics(**kwargs)
