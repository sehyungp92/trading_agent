"""Helix R2 phase gate criteria -- tighter thresholds starting from R1 optimized config.

Phase 1: trades>=350, PF>=2.0, return>=80%, exit_eff>=0.35
Phase 2: trades>=350, PF>=2.0, return>=90%, exit_eff>=0.38
Phase 3: trades>=350, PF>=2.2, return>=100%, waste>=0.65
Phase 4: trades>=350, calmar_r>=20, return>=105% + no-regression vs P3
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from backtests.swing.auto.helix.scoring import HelixMetrics


def gate_criteria_phase_1(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("hard_min_tail_pct", 0.30, m.tail_pct, m.tail_pct >= 0.30),
        GateCriterion("hard_min_regime_pf", 0.80, m.min_regime_pf, m.min_regime_pf >= 0.80),
        GateCriterion("total_trades", 350.0, float(m.total_trades), m.total_trades >= 350),
        GateCriterion("profit_factor", 2.0, m.profit_factor, m.profit_factor >= 2.0),
        GateCriterion("net_return_pct", 80.0, m.net_return_pct, m.net_return_pct >= 80.0),
        GateCriterion("exit_efficiency", 0.35, m.exit_efficiency, m.exit_efficiency >= 0.35),
    ]


def gate_criteria_phase_2(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("hard_min_tail_pct", 0.30, m.tail_pct, m.tail_pct >= 0.30),
        GateCriterion("hard_min_regime_pf", 0.80, m.min_regime_pf, m.min_regime_pf >= 0.80),
        GateCriterion("total_trades", 350.0, float(m.total_trades), m.total_trades >= 350),
        GateCriterion("profit_factor", 2.0, m.profit_factor, m.profit_factor >= 2.0),
        GateCriterion("net_return_pct", 90.0, m.net_return_pct, m.net_return_pct >= 90.0),
        GateCriterion("exit_efficiency", 0.38, m.exit_efficiency, m.exit_efficiency >= 0.38),
    ]


def gate_criteria_phase_3(metrics: dict[str, float]) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    return [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("hard_min_tail_pct", 0.30, m.tail_pct, m.tail_pct >= 0.30),
        GateCriterion("hard_min_regime_pf", 0.80, m.min_regime_pf, m.min_regime_pf >= 0.80),
        GateCriterion("total_trades", 350.0, float(m.total_trades), m.total_trades >= 350),
        GateCriterion("profit_factor", 2.2, m.profit_factor, m.profit_factor >= 2.2),
        GateCriterion("net_return_pct", 100.0, m.net_return_pct, m.net_return_pct >= 100.0),
        GateCriterion("waste_ratio", 0.65, m.waste_ratio, m.waste_ratio >= 0.65),
    ]


def gate_criteria_phase_4(
    metrics: dict[str, float],
    prior_phase_metrics: dict[str, float] | None = None,
) -> list[GateCriterion]:
    m = _to_metrics(metrics)
    criteria = [
        GateCriterion("hard_min_trades", 200.0, float(m.total_trades), m.total_trades >= 200),
        GateCriterion("hard_min_pf", 1.2, m.profit_factor, m.profit_factor >= 1.2),
        GateCriterion("hard_max_r_dd", 25.0, m.max_r_dd, m.max_r_dd <= 25.0),
        GateCriterion("hard_min_tail_pct", 0.30, m.tail_pct, m.tail_pct >= 0.30),
        GateCriterion("hard_min_regime_pf", 0.80, m.min_regime_pf, m.min_regime_pf >= 0.80),
        GateCriterion("total_trades", 350.0, float(m.total_trades), m.total_trades >= 350),
        GateCriterion("calmar_r", 20.0, m.calmar_r, m.calmar_r >= 20.0),
        GateCriterion("net_return_pct", 105.0, m.net_return_pct, m.net_return_pct >= 105.0),
    ]

    # No-regression gates vs Phase 3
    if prior_phase_metrics:
        p3 = _to_metrics(prior_phase_metrics)
        pf_floor = p3.profit_factor * 0.90
        ee_floor = p3.exit_efficiency * 0.90
        nr_floor = p3.total_r * 0.90
        criteria.extend([
            GateCriterion("no_regress_pf", pf_floor, m.profit_factor, m.profit_factor >= pf_floor),
            GateCriterion("no_regress_exit_eff", ee_floor, m.exit_efficiency, m.exit_efficiency >= ee_floor),
            GateCriterion("no_regress_total_r", nr_floor, m.total_r, m.total_r >= nr_floor),
        ])

    return criteria


# ---------------------------------------------------------------------------
# Gate routing
# ---------------------------------------------------------------------------

GATE_FN = {
    1: gate_criteria_phase_1,
    2: gate_criteria_phase_2,
    3: gate_criteria_phase_3,
    4: gate_criteria_phase_4,
}


def _to_metrics(metrics: dict[str, float]) -> HelixMetrics:
    fields = HelixMetrics.__dataclass_fields__
    kwargs = {key: metrics.get(key, 0.0) for key in fields}
    kwargs["total_trades"] = int(kwargs.get("total_trades", 0))
    return HelixMetrics(**kwargs)
