from __future__ import annotations

import math

from backtests.swing.auto.scoring import CompositeScore


PHASE_WEIGHTS: dict[int, dict[str, float]] = {
    phase: {
        "edge_velocity": 0.30,
        "expectancy": 0.20,
        "trades": 0.15,
        "return": 0.20,
        "pf": 0.10,
        "drawdown": 0.05,
    }
    for phase in range(1, 5)
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 4, "min_pf": 1.05, "max_dd_pct": 0.20, "min_expectancy_dollar": -25.0, "min_tpm": 0.5},
    2: {"min_trades": 4, "min_pf": 1.10, "max_dd_pct": 0.18, "min_expectancy_dollar": -10.0, "min_tpm": 0.5},
    3: {"min_trades": 4, "min_pf": 1.15, "max_dd_pct": 0.16, "min_expectancy_dollar": 0.0, "min_tpm": 0.5},
    4: {"min_trades": 4, "min_pf": 1.20, "max_dd_pct": 0.15, "min_expectancy_dollar": 0.0, "min_tpm": 0.5},
}


def score_phase_metrics(
    phase: int,
    metrics,
    initial_equity: float,
    *,
    equity_curve=None,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> CompositeScore:
    del equity_curve
    edge_velocity = float(getattr(metrics, "edge_velocity", 0.0))
    weights = dict(PHASE_WEIGHTS.get(phase, PHASE_WEIGHTS[1]))
    weights.update(weight_overrides or {})
    total_weight = sum(weights.values()) or 1.0
    weights = {key: value / total_weight for key, value in weights.items()}
    components = {
        "edge_velocity": _scaled(edge_velocity, 120.0),
        "expectancy": _scaled(float(getattr(metrics, "expectancy_dollar", 0.0)), 80.0),
        "trades": min(math.sqrt(max(float(getattr(metrics, "trades_per_month", 0.0)), 0.0) / 8.0), 1.0),
        "return": _scaled(float(getattr(metrics, "net_profit", 0.0)) / max(initial_equity, 1.0), 0.40),
        "pf": _scaled_pf(float(getattr(metrics, "profit_factor", 0.0))),
        "drawdown": max(0.0, 1.0 - float(getattr(metrics, "max_drawdown_pct", 0.0)) / 0.20),
    }
    total = sum(weights.get(key, 0.0) * components[key] for key in components)
    reject_reason = _hard_reject(phase, metrics, hard_rejects)
    return CompositeScore(
        calmar_component=components["edge_velocity"],
        pf_component=components["pf"],
        inv_dd_component=components["drawdown"],
        net_profit_component=components["return"],
        total=total,
        rejected=reject_reason is not None,
        reject_reason=reject_reason or "",
    )


def _hard_reject(phase: int, metrics, overrides: dict[str, float] | None) -> str | None:
    rejects = dict(PHASE_HARD_REJECTS.get(phase, {}))
    rejects.update(overrides or {})
    if getattr(metrics, "total_trades", 0) < rejects.get("min_trades", 0):
        return "too_few_trades"
    if getattr(metrics, "profit_factor", 0.0) < rejects.get("min_pf", 0.0):
        return "low_profit_factor"
    if getattr(metrics, "max_drawdown_pct", 0.0) > rejects.get("max_dd_pct", 1.0):
        return "drawdown_too_high"
    if getattr(metrics, "expectancy_dollar", 0.0) < rejects.get("min_expectancy_dollar", -1e9):
        return "low_expectancy"
    if getattr(metrics, "trades_per_month", 0.0) < rejects.get("min_tpm", 0.0):
        return "low_frequency"
    return None


def _scaled(value: float, target: float) -> float:
    if value <= 0 or target <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(target), 1.0)


def _scaled_pf(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    if value <= 1:
        return 0.0
    return min(math.log(value) / math.log(3.0), 1.0)

