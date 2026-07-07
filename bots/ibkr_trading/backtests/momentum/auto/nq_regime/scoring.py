from __future__ import annotations

import math
from dataclasses import dataclass


IMMUTABLE_WEIGHTS: dict[str, float] = {
    "alpha_return": 0.24,
    "trade_frequency": 0.18,
    "expectancy_quality": 0.18,
    "component_synergy": 0.18,
    "module_guardrails": 0.10,
    "execution": 0.06,
    "drawdown_robustness": 0.06,
}

PHASE_WEIGHTS: dict[int, dict[str, float]] = {phase: dict(IMMUTABLE_WEIGHTS) for phase in range(1, 8)}
PHASE_WEIGHTS[4] = {
    "alpha_return": 0.18,
    "trade_frequency": 0.14,
    "expectancy_quality": 0.14,
    "component_synergy": 0.20,
    "module_guardrails": 0.22,
    "execution": 0.06,
    "drawdown_robustness": 0.06,
}
PHASE_WEIGHTS[7] = {
    "alpha_return": 0.28,
    "trade_frequency": 0.20,
    "expectancy_quality": 0.16,
    "component_synergy": 0.18,
    "module_guardrails": 0.08,
    "execution": 0.05,
    "drawdown_robustness": 0.05,
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "min_trades": 320,
        "min_trades_per_month": 6.50,
        "min_total_r_per_month": 7.50,
        "min_pf": 2.00,
        "max_dd_pct": 0.070,
        "min_avg_r": 0.60,
        "min_module_coverage": 1.0,
        "min_module_trades": 35,
        "min_nq1_trades": 100,
        "min_nq1_avg_r": 0.80,
        "min_nq1_pf": 1.50,
        "min_nq2_trades": 35,
        "min_nq2_avg_r": 0.05,
        "min_nq2_pf": 1.10,
        "min_nq3_trades": 100,
        "min_nq3_avg_r": 0.80,
        "min_nq3_pf": 3.00,
        "max_module_trade_share": 0.78,
    },
    2: {
        "min_trades": 340,
        "min_trades_per_month": 7.00,
        "min_total_r_per_month": 8.00,
        "min_pf": 2.20,
        "max_dd_pct": 0.070,
        "min_avg_r": 0.65,
        "min_module_coverage": 1.0,
        "min_module_trades": 40,
        "min_nq1_trades": 100,
        "min_nq1_avg_r": 0.85,
        "min_nq1_pf": 1.60,
        "min_nq2_trades": 40,
        "min_nq2_avg_r": 0.06,
        "min_nq2_pf": 1.15,
        "min_nq3_trades": 110,
        "min_nq3_avg_r": 0.85,
        "min_nq3_pf": 3.50,
        "max_module_trade_share": 0.76,
    },
    3: {
        "min_trades": 340,
        "min_trades_per_month": 7.00,
        "min_total_r_per_month": 8.00,
        "min_pf": 2.20,
        "max_dd_pct": 0.070,
        "min_avg_r": 0.65,
        "min_module_coverage": 1.0,
        "min_module_trades": 40,
        "min_nq1_trades": 100,
        "min_nq1_avg_r": 0.90,
        "min_nq1_pf": 1.65,
        "min_nq2_trades": 40,
        "min_nq2_avg_r": 0.06,
        "min_nq2_pf": 1.15,
        "min_nq3_trades": 110,
        "min_nq3_avg_r": 0.85,
        "min_nq3_pf": 3.50,
        "max_module_trade_share": 0.76,
    },
    4: {
        "min_trades": 350,
        "min_trades_per_month": 7.20,
        "min_total_r_per_month": 8.00,
        "min_pf": 2.25,
        "max_dd_pct": 0.065,
        "min_avg_r": 0.65,
        "min_module_coverage": 1.0,
        "min_module_trades": 45,
        "min_nq1_trades": 100,
        "min_nq1_avg_r": 0.90,
        "min_nq1_pf": 1.65,
        "min_nq2_trades": 45,
        "min_nq2_avg_r": 0.08,
        "min_nq2_pf": 1.25,
        "min_nq3_trades": 110,
        "min_nq3_avg_r": 0.85,
        "min_nq3_pf": 3.50,
        "max_module_trade_share": 0.75,
    },
    5: {
        "min_trades": 360,
        "min_trades_per_month": 7.40,
        "min_total_r_per_month": 8.20,
        "min_pf": 2.30,
        "max_dd_pct": 0.065,
        "min_avg_r": 0.70,
        "min_module_coverage": 1.0,
        "min_module_trades": 45,
        "min_nq1_trades": 100,
        "min_nq1_avg_r": 0.90,
        "min_nq1_pf": 1.65,
        "min_nq2_trades": 45,
        "min_nq2_avg_r": 0.08,
        "min_nq2_pf": 1.25,
        "min_nq3_trades": 120,
        "min_nq3_avg_r": 0.90,
        "min_nq3_pf": 4.00,
        "max_module_trade_share": 0.74,
    },
    6: {
        "min_trades": 380,
        "min_trades_per_month": 7.80,
        "min_total_r_per_month": 8.50,
        "min_pf": 2.40,
        "max_dd_pct": 0.065,
        "min_avg_r": 0.75,
        "min_module_coverage": 1.0,
        "min_module_trades": 45,
        "min_nq1_trades": 100,
        "min_nq1_avg_r": 0.90,
        "min_nq1_pf": 1.65,
        "min_nq2_trades": 45,
        "min_nq2_avg_r": 0.08,
        "min_nq2_pf": 1.25,
        "min_nq3_trades": 120,
        "min_nq3_avg_r": 0.90,
        "min_nq3_pf": 4.00,
        "max_module_trade_share": 0.72,
        "max_positive_mfe_loser_rate": 0.15,
    },
    7: {
        "min_trades": 400,
        "min_trades_per_month": 8.00,
        "min_total_r_per_month": 8.80,
        "min_pf": 2.50,
        "max_dd_pct": 0.060,
        "min_avg_r": 0.80,
        "min_module_coverage": 1.0,
        "min_module_trades": 45,
        "min_nq1_trades": 100,
        "min_nq1_avg_r": 0.95,
        "min_nq1_pf": 1.70,
        "min_nq2_trades": 45,
        "min_nq2_avg_r": 0.08,
        "min_nq2_pf": 1.25,
        "min_nq3_trades": 120,
        "min_nq3_avg_r": 0.90,
        "min_nq3_pf": 4.00,
        "max_module_trade_share": 0.70,
        "min_execution_conversion": 0.15,
        "max_positive_mfe_loser_rate": 0.15,
    },
}


@dataclass(frozen=True)
class NqRegimeRound5Score:
    score: float
    rejected: bool = False
    reject_reason: str = ""


def composite_score(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> NqRegimeRound5Score:
    rejects = hard_rejects or {}
    total_trades = metrics.get("total_trades", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    dd = metrics.get("max_drawdown_pct", 0.0)
    avg_r = metrics.get("avg_r", 0.0)
    total_r_per_month = metrics.get("total_r_per_month", 0.0)
    trades_per_month = metrics.get("trades_per_month", 0.0)
    coverage = metrics.get("module_coverage", 0.0)
    min_module_trades = metrics.get("min_module_trades", 0.0)
    module_share = _max_module_trade_share(metrics)

    checks = (
        ("min_trades", total_trades, ">="),
        ("min_trades_per_month", trades_per_month, ">="),
        ("min_total_r_per_month", total_r_per_month, ">="),
        ("min_pf", pf, ">="),
        ("max_dd_pct", dd, "<="),
        ("min_avg_r", avg_r, ">="),
        ("min_module_coverage", coverage, ">="),
        ("min_module_trades", min_module_trades, ">="),
        ("max_module_trade_share", module_share, "<="),
        ("min_execution_conversion", metrics.get("execution_conversion", 0.0), ">="),
        ("max_positive_mfe_loser_rate", metrics.get("positive_mfe_loser_rate", 0.0), "<="),
        ("min_nq1_trades", metrics.get("module_second_wind_trades", 0.0), ">="),
        ("min_nq1_avg_r", metrics.get("module_second_wind_avg_r", 0.0), ">="),
        ("min_nq1_pf", metrics.get("module_second_wind_profit_factor", 0.0), ">="),
        ("min_nq2_trades", metrics.get("module_structural_expansion_trades", 0.0), ">="),
        ("min_nq2_avg_r", metrics.get("module_structural_expansion_avg_r", 0.0), ">="),
        ("min_nq2_pf", metrics.get("module_structural_expansion_profit_factor", 0.0), ">="),
        ("min_nq3_trades", metrics.get("module_liquidity_reversion_trades", 0.0), ">="),
        ("min_nq3_avg_r", metrics.get("module_liquidity_reversion_avg_r", 0.0), ">="),
        ("min_nq3_pf", metrics.get("module_liquidity_reversion_profit_factor", 0.0), ">="),
    )
    for name, actual, op in checks:
        if name not in rejects:
            continue
        target = rejects[name]
        failed = actual > target if op == "<=" else actual < target
        if failed:
            return NqRegimeRound5Score(0.0, True, name)

    n_shrink = _shrink(total_trades, 250.0)
    concentration_penalty = 1.0 - 0.35 * _clip((module_share - 0.60) / 0.25, 0.0, 1.0)
    components = {
        "alpha_return": _clip(total_r_per_month * n_shrink * concentration_penalty / 10.0, -1.0, 1.5),
        "trade_frequency": _clip(trades_per_month / 10.0, 0.0, 1.5),
        "expectancy_quality": _expectancy_quality(metrics, n_shrink),
        "component_synergy": _component_synergy(metrics),
        "module_guardrails": _module_guardrails(metrics),
        "execution": _execution_score(metrics),
        "drawdown_robustness": _drawdown_score(metrics),
    }
    active_weights = weights or IMMUTABLE_WEIGHTS
    total_weight = sum(active_weights.values()) or 1.0
    score = sum(components.get(key, 0.0) * value for key, value in active_weights.items()) / total_weight
    return NqRegimeRound5Score(float(score))


def _expectancy_quality(metrics: dict[str, float], n_shrink: float) -> float:
    avg_r = _clip(metrics.get("avg_r", 0.0) / 1.20, -1.0, 1.5)
    pf = _clip(math.log(max(metrics.get("profit_factor", 0.01), 0.01)) / math.log(5.0), -1.0, 1.5)
    win = _clip((metrics.get("win_rate", 0.0) - 0.50) / 0.25, -0.5, 1.0)
    expectancy = _clip(metrics.get("expectancy_dollar", 0.0) / 55.0, -1.0, 1.5)
    capture = _clip(metrics.get("mfe_capture", 0.0) / 0.75, 0.0, 1.5)
    return _clip((0.30 * avg_r + 0.25 * pf + 0.15 * win + 0.20 * expectancy + 0.10 * capture) * n_shrink, -1.0, 1.5)


def _component_synergy(metrics: dict[str, float]) -> float:
    trades = _module_trade_counts(metrics)
    total = sum(trades.values())
    if total <= 0:
        return 0.0
    coverage = _clip(metrics.get("module_coverage", 0.0) / 1.0, 0.0, 1.0)
    min_trade_score = _clip(metrics.get("min_module_trades", 0.0) / 60.0, 0.0, 1.5)
    trade_entropy = _normalized_entropy([value / total for value in trades.values()])
    total_r_values = [
        max(0.0, metrics.get("module_second_wind_total_r", 0.0)),
        max(0.0, metrics.get("module_structural_expansion_total_r", 0.0)),
        max(0.0, metrics.get("module_liquidity_reversion_total_r", 0.0)),
    ]
    r_total = sum(total_r_values)
    return_entropy = _normalized_entropy([value / r_total for value in total_r_values]) if r_total else 0.0
    positive_edges = sum(
        1
        for prefix in ("module_second_wind", "module_structural_expansion", "module_liquidity_reversion")
        if metrics.get(f"{prefix}_avg_r", 0.0) > 0.0 and metrics.get(f"{prefix}_profit_factor", 0.0) > 1.0
    ) / 3.0
    return _clip(
        0.25 * coverage
        + 0.20 * min_trade_score
        + 0.25 * trade_entropy
        + 0.15 * return_entropy
        + 0.15 * positive_edges,
        0.0,
        1.5,
    )


def _module_guardrails(metrics: dict[str, float]) -> float:
    nq1 = _module_quality(metrics, "module_second_wind", avg_target=1.20, pf_target=3.0, trade_target=120.0)
    nq2 = _module_quality(metrics, "module_structural_expansion", avg_target=0.20, pf_target=2.0, trade_target=60.0)
    nq3 = _module_quality(metrics, "module_liquidity_reversion", avg_target=1.20, pf_target=8.0, trade_target=180.0)
    nq3_capture = _clip(metrics.get("module_liquidity_reversion_mfe_capture", 0.0) / 0.75, 0.0, 1.5)
    nq1_leak = 1.0 - _clip(metrics.get("module_second_wind_positive_mfe_loser_rate", 0.0) / 0.30, 0.0, 1.0)
    return _clip(0.30 * nq1 + 0.20 * nq2 + 0.35 * nq3 + 0.10 * nq3_capture + 0.05 * nq1_leak, -1.0, 1.5)


def _module_quality(metrics: dict[str, float], prefix: str, *, avg_target: float, pf_target: float, trade_target: float) -> float:
    trades = metrics.get(f"{prefix}_trades", 0.0)
    shrink = _shrink(trades, trade_target)
    avg_r = _clip(metrics.get(f"{prefix}_avg_r", 0.0) / avg_target, -1.0, 1.5)
    pf = _clip(math.log(max(metrics.get(f"{prefix}_profit_factor", 0.01), 0.01)) / math.log(pf_target), -1.0, 1.5)
    win = _clip((metrics.get(f"{prefix}_win_rate", 0.0) - 0.45) / 0.30, -0.5, 1.0)
    capture = _clip(metrics.get(f"{prefix}_mfe_capture", 0.0) / 0.65, 0.0, 1.5)
    return _clip((0.45 * avg_r + 0.30 * pf + 0.10 * win + 0.15 * capture) * shrink, -1.0, 1.5)


def _execution_score(metrics: dict[str, float]) -> float:
    overall = _clip(metrics.get("execution_conversion", 0.0) / 0.25, 0.0, 1.5)
    selected_rates = []
    for module in ("second_wind", "structural_expansion", "liquidity_reversion"):
        selected = metrics.get(f"routing_{module}_selected", 0.0)
        trades = metrics.get(f"module_{module}_trades", 0.0)
        if selected > 0:
            selected_rates.append(trades / selected)
    selected_avg = sum(selected_rates) / len(selected_rates) if selected_rates else 0.0
    request_rates = [
        metrics.get("routing_second_wind_request_to_fill_rate", 0.0),
        metrics.get("routing_liquidity_reversion_request_to_fill_rate", 0.0),
    ]
    request_avg = sum(request_rates) / len(request_rates)
    return _clip(0.45 * overall + 0.35 * _clip(selected_avg / 0.25, 0.0, 1.5) + 0.20 * _clip(request_avg / 0.30, 0.0, 1.5), 0.0, 1.5)


def _drawdown_score(metrics: dict[str, float]) -> float:
    dd = metrics.get("max_drawdown_pct", 0.0)
    rolling_floor = metrics.get("rolling_20_min_avg_r", 0.0)
    dd_score = _clip(1.0 - dd / 0.06, -1.0, 1.0)
    rolling_score = _clip(rolling_floor / 0.25, -0.5, 1.0) if rolling_floor else 0.50
    return _clip(0.75 * dd_score + 0.25 * rolling_score, -1.0, 1.0)


def _module_trade_counts(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "second_wind": metrics.get("module_second_wind_trades", 0.0),
        "structural_expansion": metrics.get("module_structural_expansion_trades", 0.0),
        "liquidity_reversion": metrics.get("module_liquidity_reversion_trades", 0.0),
    }


def _max_module_trade_share(metrics: dict[str, float]) -> float:
    counts = _module_trade_counts(metrics)
    total = sum(counts.values())
    if total <= 0:
        return 1.0
    return max(counts.values()) / total


def _normalized_entropy(shares: list[float]) -> float:
    cleaned = [share for share in shares if share > 0.0]
    if not cleaned:
        return 0.0
    entropy = -sum(share * math.log(share) for share in cleaned)
    return _clip(entropy / math.log(3.0), 0.0, 1.0)


def _shrink(n: float, k: float) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(n / (n + k))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
