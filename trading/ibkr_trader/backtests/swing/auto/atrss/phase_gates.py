"""ATRSS phase gate criteria -- progressive thresholds per phase.

Two regimes:
  - R1: Original high thresholds for independent-account mode.
  - R9: Rescaled for honest synchronized/fee-net conditions.

Active regime is read from phase_scoring.SCORING_REGIME.
"""
from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import ATRSSMetrics


def gate_criteria_for_phase(
    phase: int,
    metrics: ATRSSMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Return gate criteria for *phase* given current *metrics*.

    Phase 4 adds no-regression checks against Phase 3 results.
    Thresholds auto-select between R1 and R9 regimes.
    """
    from .phase_scoring import SCORING_REGIME

    if SCORING_REGIME == "r9":
        return _gate_criteria_r9(phase, metrics, prior_phase_metrics)
    return _gate_criteria_r1(phase, metrics, prior_phase_metrics)


def risk_allocation_gate_criteria_for_phase(
    phase: int,
    metrics: ATRSSMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """Strict gates for the ATRSS capital-deployment optimization round."""
    criteria: list[GateCriterion] = [
        GateCriterion("hard_min_trades", 250.0, float(metrics.total_trades), metrics.total_trades >= 250),
        GateCriterion("hard_max_dd_pct", 0.080, metrics.max_dd_pct, metrics.max_dd_pct <= 0.080),
        GateCriterion("hard_min_pf", 4.5, metrics.profit_factor, metrics.profit_factor >= 4.5),
        GateCriterion("hard_min_wr", 0.78, metrics.win_rate, metrics.win_rate >= 0.78),
        GateCriterion("hard_min_net_return_pct", 45.0, metrics.net_return_pct, metrics.net_return_pct >= 45.0),
    ]

    if phase == 1:
        criteria.extend([
            GateCriterion("net_return_pct", 55.0, metrics.net_return_pct, metrics.net_return_pct >= 55.0),
            GateCriterion("profit_factor", 5.0, metrics.profit_factor, metrics.profit_factor >= 5.0),
            GateCriterion("max_dd_pct", 0.075, metrics.max_dd_pct, metrics.max_dd_pct <= 0.075),
            GateCriterion("total_r", 210.0, metrics.total_r, metrics.total_r >= 210.0),
        ])
    elif phase == 2:
        criteria.extend([
            GateCriterion("net_return_pct", 65.0, metrics.net_return_pct, metrics.net_return_pct >= 65.0),
            GateCriterion("calmar_r", 55.0, metrics.calmar_r, metrics.calmar_r >= 55.0),
            GateCriterion("max_dd_pct", 0.078, metrics.max_dd_pct, metrics.max_dd_pct <= 0.078),
        ])
    elif phase == 3:
        criteria.extend([
            GateCriterion("net_return_pct", 70.0, metrics.net_return_pct, metrics.net_return_pct >= 70.0),
            GateCriterion("profit_factor", 4.8, metrics.profit_factor, metrics.profit_factor >= 4.8),
            GateCriterion("mfe_capture", 0.66, metrics.mfe_capture, metrics.mfe_capture >= 0.66),
        ])
    elif phase == 4:
        criteria.extend([
            GateCriterion("net_return_pct", 75.0, metrics.net_return_pct, metrics.net_return_pct >= 75.0),
            GateCriterion("profit_factor", 5.0, metrics.profit_factor, metrics.profit_factor >= 5.0),
            GateCriterion("calmar_r", 60.0, metrics.calmar_r, metrics.calmar_r >= 60.0),
            GateCriterion("max_dd_pct", 0.080, metrics.max_dd_pct, metrics.max_dd_pct <= 0.080),
        ])

    if prior_phase_metrics:
        no_regress_keys = ("net_return_pct", "profit_factor", "win_rate", "total_r", "mfe_capture")
        for key in no_regress_keys:
            prior_val = prior_phase_metrics.get(key, 0.0)
            if prior_val > 0:
                cur_val = getattr(metrics, key, 0.0)
                floor = prior_val * 0.97
                criteria.append(
                    GateCriterion(f"no_regress_{key}", floor, cur_val, cur_val >= floor)
                )

    return criteria


def _gate_criteria_r9(
    phase: int,
    metrics: ATRSSMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """R9 gate criteria -- rescaled for synchronized/fee-net conditions."""
    criteria: list[GateCriterion] = [
        GateCriterion("hard_min_trades", 220.0, float(metrics.total_trades), metrics.total_trades >= 220),
        GateCriterion("hard_max_dd_pct", 0.06, metrics.max_dd_pct, metrics.max_dd_pct <= 0.06),
        GateCriterion("hard_min_pf", 1.8, metrics.profit_factor, metrics.profit_factor >= 1.8),
        GateCriterion("hard_min_wr", 0.58, metrics.win_rate, metrics.win_rate >= 0.58),
    ]

    if phase == 1:
        criteria.extend([
            GateCriterion("total_trades", 250.0, float(metrics.total_trades), metrics.total_trades >= 250),
            GateCriterion("trades_per_month", 4.5, metrics.trades_per_month, metrics.trades_per_month >= 4.5),
            GateCriterion("total_r", 190.0, metrics.total_r, metrics.total_r >= 190.0),
        ])
    elif phase == 2:
        criteria.extend([
            GateCriterion("total_trades", 255.0, float(metrics.total_trades), metrics.total_trades >= 255),
            GateCriterion("win_rate", 0.68, metrics.win_rate, metrics.win_rate >= 0.68),
            GateCriterion("profit_factor", 2.5, metrics.profit_factor, metrics.profit_factor >= 2.5),
        ])
    elif phase == 3:
        criteria.extend([
            GateCriterion("trades_per_month", 4.7, metrics.trades_per_month, metrics.trades_per_month >= 4.7),
            GateCriterion("total_trades", 260.0, float(metrics.total_trades), metrics.total_trades >= 260),
            GateCriterion("profit_factor", 2.8, metrics.profit_factor, metrics.profit_factor >= 2.8),
        ])
    elif phase == 4:
        criteria.extend([
            GateCriterion("calmar_r", 35.0, metrics.calmar_r, metrics.calmar_r >= 35.0),
            GateCriterion("total_r", 200.0, metrics.total_r, metrics.total_r >= 200.0),
            GateCriterion("sharpe", 1.0, metrics.sharpe, metrics.sharpe >= 1.0),
        ])
        if prior_phase_metrics:
            for key in ("profit_factor", "total_r", "win_rate", "calmar_r"):
                prior_val = prior_phase_metrics.get(key, 0.0)
                if prior_val > 0:
                    cur_val = getattr(metrics, key, 0.0)
                    floor = prior_val * 0.95
                    criteria.append(
                        GateCriterion(f"no_regress_{key}", floor, cur_val, cur_val >= floor)
                    )

    return criteria


def _gate_criteria_r1(
    phase: int,
    metrics: ATRSSMetrics,
    prior_phase_metrics: dict | None = None,
) -> list[GateCriterion]:
    """R1 gate criteria -- original thresholds for independent-account mode."""
    criteria: list[GateCriterion] = [
        GateCriterion("hard_min_trades", 100.0, float(metrics.total_trades), metrics.total_trades >= 100),
        GateCriterion("hard_max_dd_pct", 0.07, metrics.max_dd_pct, metrics.max_dd_pct <= 0.07),
        GateCriterion("hard_min_pf", 2.0, metrics.profit_factor, metrics.profit_factor >= 2.0),
        GateCriterion("hard_min_wr", 0.55, metrics.win_rate, metrics.win_rate >= 0.55),
    ]

    if phase == 1:
        criteria.extend([
            GateCriterion("profit_factor", 4.0, metrics.profit_factor, metrics.profit_factor >= 4.0),
            GateCriterion("total_r", 130.0, metrics.total_r, metrics.total_r >= 130.0),
            GateCriterion("mfe_capture", 0.35, metrics.mfe_capture, metrics.mfe_capture >= 0.35),
        ])
    elif phase == 2:
        criteria.extend([
            GateCriterion("total_trades", 150.0, float(metrics.total_trades), metrics.total_trades >= 150),
            GateCriterion("win_rate", 0.65, metrics.win_rate, metrics.win_rate >= 0.65),
            GateCriterion("profit_factor", 4.0, metrics.profit_factor, metrics.profit_factor >= 4.0),
        ])
    elif phase == 3:
        criteria.extend([
            GateCriterion("trades_per_month", 4.0, metrics.trades_per_month, metrics.trades_per_month >= 4.0),
            GateCriterion("total_trades", 160.0, float(metrics.total_trades), metrics.total_trades >= 160),
            GateCriterion("profit_factor", 4.5, metrics.profit_factor, metrics.profit_factor >= 4.5),
        ])
    elif phase == 4:
        criteria.extend([
            GateCriterion("calmar_r", 30.0, metrics.calmar_r, metrics.calmar_r >= 30.0),
            GateCriterion("total_r", 150.0, metrics.total_r, metrics.total_r >= 150.0),
            GateCriterion("sharpe", 3.0, metrics.sharpe, metrics.sharpe >= 3.0),
        ])
        if prior_phase_metrics:
            for key in ("profit_factor", "total_r", "win_rate", "calmar_r"):
                prior_val = prior_phase_metrics.get(key, 0.0)
                if prior_val > 0:
                    cur_val = getattr(metrics, key, 0.0)
                    floor = prior_val * 0.95
                    criteria.append(
                        GateCriterion(f"no_regress_{key}", floor, cur_val, cur_val >= floor)
                    )

    return criteria
