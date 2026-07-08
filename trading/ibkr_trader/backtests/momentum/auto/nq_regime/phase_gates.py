from __future__ import annotations

from backtests.shared.auto.types import GateCriterion

from .scoring import PHASE_HARD_REJECTS


def gate_criteria_for_phase(phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
    rejects = PHASE_HARD_REJECTS.get(phase, {})
    criteria = [
        _criterion("min_trades", rejects.get("min_trades", 0), metrics.get("total_trades", 0), ">="),
        _criterion("min_trades_per_month", rejects.get("min_trades_per_month", 0), metrics.get("trades_per_month", 0), ">="),
        _criterion("min_total_r_per_month", rejects.get("min_total_r_per_month", 0), metrics.get("total_r_per_month", 0), ">="),
        _criterion("min_pf", rejects.get("min_pf", 0), metrics.get("profit_factor", 0), ">="),
        _criterion("max_dd_pct", rejects.get("max_dd_pct", 1), metrics.get("max_drawdown_pct", 0), "<="),
        _criterion("min_avg_r", rejects.get("min_avg_r", -99), metrics.get("avg_r", 0), ">="),
        _criterion("min_module_coverage", rejects.get("min_module_coverage", 0), metrics.get("module_coverage", 0), ">="),
        _criterion("min_module_trades", rejects.get("min_module_trades", 0), metrics.get("min_module_trades", 0), ">="),
        _criterion("max_module_trade_share", rejects.get("max_module_trade_share", 1), _max_module_trade_share(metrics), "<="),
        _criterion("min_nq1_trades", rejects.get("min_nq1_trades", 0), metrics.get("module_second_wind_trades", 0), ">="),
        _criterion("min_nq2_trades", rejects.get("min_nq2_trades", 0), metrics.get("module_structural_expansion_trades", 0), ">="),
        _criterion("min_nq3_trades", rejects.get("min_nq3_trades", 0), metrics.get("module_liquidity_reversion_trades", 0), ">="),
        _criterion("min_nq1_avg_r", rejects.get("min_nq1_avg_r", -99), metrics.get("module_second_wind_avg_r", 0), ">="),
        _criterion("min_nq2_avg_r", rejects.get("min_nq2_avg_r", -99), metrics.get("module_structural_expansion_avg_r", 0), ">="),
        _criterion("min_nq3_avg_r", rejects.get("min_nq3_avg_r", -99), metrics.get("module_liquidity_reversion_avg_r", 0), ">="),
        _criterion("min_nq1_pf", rejects.get("min_nq1_pf", 0), metrics.get("module_second_wind_profit_factor", 0), ">="),
        _criterion("min_nq2_pf", rejects.get("min_nq2_pf", 0), metrics.get("module_structural_expansion_profit_factor", 0), ">="),
        _criterion("min_nq3_pf", rejects.get("min_nq3_pf", 0), metrics.get("module_liquidity_reversion_profit_factor", 0), ">="),
    ]
    if "min_execution_conversion" in rejects:
        criteria.append(_criterion("min_execution_conversion", rejects["min_execution_conversion"], metrics.get("execution_conversion", 0), ">="))
    if "max_positive_mfe_loser_rate" in rejects:
        criteria.append(_criterion("max_positive_mfe_loser_rate", rejects["max_positive_mfe_loser_rate"], metrics.get("positive_mfe_loser_rate", 0), "<="))
    return criteria


def _criterion(name: str, target: float, actual: float, op: str) -> GateCriterion:
    passed = actual <= target if op == "<=" else actual >= target
    return GateCriterion(name, target, actual, passed)


def _max_module_trade_share(metrics: dict[str, float]) -> float:
    counts = [
        metrics.get("module_second_wind_trades", 0.0),
        metrics.get("module_structural_expansion_trades", 0.0),
        metrics.get("module_liquidity_reversion_trades", 0.0),
    ]
    total = sum(counts)
    return max(counts) / total if total else 1.0

