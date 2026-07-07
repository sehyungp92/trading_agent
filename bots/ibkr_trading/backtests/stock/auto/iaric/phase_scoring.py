from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


_BASE_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "avg_r": 0.20,
        "profit_factor": 0.14,
        "managed_exit_share": 0.20,
        "eod_flatten_inverse": 0.10,
        "expected_total_r": 0.14,
        "protection_candidate_inverse": 0.08,
        "inv_dd": 0.06,
        "total_trades": 0.08,
    },
    2: {
        "avg_r": 0.20,
        "profit_factor": 0.16,
        "signal_score_edge": 0.16,
        "crowded_day_discrimination": 0.16,
        "expected_total_r": 0.14,
        "total_trades": 0.12,
        "inv_dd": 0.06,
    },
    3: {
        "avg_r": 0.14,
        "profit_factor": 0.12,
        "route_diversity": 0.18,
        "route_score_monotonicity": 0.14,
        "expected_total_r": 0.18,
        "total_trades": 0.16,
        "inv_dd": 0.08,
    },
    4: {
        "avg_r": 0.20,
        "profit_factor": 0.14,
        "carry_avg_r": 0.18,
        "carry_trade_share": 0.08,
        "managed_exit_share": 0.14,
        "eod_flatten_inverse": 0.10,
        "expected_total_r": 0.10,
        "total_trades": 0.06,
    },
    5: {
        "avg_r": 0.18,
        "profit_factor": 0.14,
        "sharpe": 0.16,
        "expected_total_r": 0.20,
        "total_trades": 0.12,
        "inv_dd": 0.12,
        "eod_flatten_inverse": 0.08,
    },
}

_MAINLINE_EXPECTED_TOTAL_R_WEIGHTS = {
    1: 0.12,
    2: 0.12,
    3: 0.12,
    4: 0.12,
    5: 0.16,
}

_AGGRESSIVE_EXPECTED_TOTAL_R_WEIGHTS = {
    1: 0.14,
    2: 0.14,
    3: 0.16,
    4: 0.16,
    5: 0.18,
}

_AGGRESSIVE_TOTAL_TRADES_WEIGHTS = {
    1: 0.12,
    2: 0.10,
    3: 0.16,
    4: 0.12,
    5: 0.18,
}

_AGGRESSIVE_INV_DD_WEIGHTS = {
    1: 0.02,
    2: 0.02,
    3: 0.04,
    4: 0.04,
    5: 0.04,
}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def _build_mainline_phase_weights() -> dict[int, dict[str, float]]:
    mainline: dict[int, dict[str, float]] = {}
    for phase, weights in _BASE_PHASE_SCORING_WEIGHTS.items():
        updated = dict(weights)
        updated["expected_total_r"] = _MAINLINE_EXPECTED_TOTAL_R_WEIGHTS[phase]
        mainline[phase] = _normalize_weights(updated)
    return mainline


def _build_aggressive_phase_weights() -> dict[int, dict[str, float]]:
    aggressive: dict[int, dict[str, float]] = {}
    for phase, weights in _BASE_PHASE_SCORING_WEIGHTS.items():
        updated = dict(weights)
        updated["expected_total_r"] = _AGGRESSIVE_EXPECTED_TOTAL_R_WEIGHTS[phase]
        updated["total_trades"] = _AGGRESSIVE_TOTAL_TRADES_WEIGHTS[phase]
        updated["inv_dd"] = _AGGRESSIVE_INV_DD_WEIGHTS[phase]
        aggressive[phase] = _normalize_weights(updated)
    return aggressive


PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = _build_mainline_phase_weights()
AGGRESSIVE_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = _build_aggressive_phase_weights()
PHASE_SCORING_WEIGHTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": PHASE_SCORING_WEIGHTS,
    "aggressive": AGGRESSIVE_PHASE_SCORING_WEIGHTS,
}


def get_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    return PHASE_SCORING_WEIGHTS_BY_PROFILE.get(str(profile or "mainline").lower(), PHASE_SCORING_WEIGHTS)


_NORMALIZATION_CEILINGS: dict[str, float] = {
    "avg_r": 0.40,
    "expected_total_r": 40.0,
    "profit_factor": 3.50,
    "sharpe": 2.50,
    "rsi_depth_edge": 0.15,
    "trend_band_edge": 0.15,
    "late_rank_edge": 0.25,
    "managed_exit_share": 0.85,
    "carry_trade_share": 0.40,
    "carry_avg_r": 0.80,
    "total_trades": 160.0,
    "inv_dd": 1.0,
    "eod_flatten_inverse": 0.85,
    "stop_hit_inverse": 1.0,
    "signal_score_edge": 0.25,
    "crowded_day_discrimination": 0.15,
    "crowded_day_missed_alpha_inverse": 1.0,
    "route_score_monotonicity": 1.0,
    "open_scored_inverse": 1.0,
    "route_diversity": 1.0,
    "protection_candidate_inverse": 1.0,
}


def merge_pullback_metrics(
    performance_metrics: Any,
    trades: list[Any],
    *,
    candidate_ledger: dict | None = None,
    selection_attribution: dict | None = None,
) -> dict[str, float]:
    metrics = asdict(performance_metrics) if is_dataclass(performance_metrics) else dict(performance_metrics)
    metrics.update(
        compute_pullback_phase_metrics(
            trades,
            candidate_ledger=candidate_ledger,
            selection_attribution=selection_attribution,
        )
    )
    return metrics


def score_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    if scoring_weights:
        weights = dict(scoring_weights)
    else:
        weights = dict(PHASE_SCORING_WEIGHTS.get(phase, {}))

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_phase_score_metrics(metrics)
    return sum(
        (weight / total_weight) * _normalize_metric(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


def enrich_phase_score_metrics(metrics: dict[str, float]) -> dict[str, float]:
    enriched = dict(metrics)
    max_dd = float(metrics.get("max_drawdown_pct", 0.0))
    stop_hit_share = float(metrics.get("stop_hit_share", 0.0))
    eod_flatten_share = float(metrics.get("eod_flatten_share", 0.0))
    avg_r = float(metrics.get("avg_r", 0.0))
    total_trades = float(metrics.get("total_trades", 0.0))

    enriched["inv_dd"] = _clip01(1.0 - max_dd / 0.06)
    enriched["eod_flatten_inverse"] = _clip01(1.0 - eod_flatten_share)
    enriched["stop_hit_inverse"] = _clip01(1.0 - stop_hit_share / 0.10)
    enriched["expected_total_r"] = avg_r * total_trades
    enriched["protection_candidate_inverse"] = _clip01(
        1.0 - float(metrics.get("protection_candidate_share", 0.0)) / 0.30
    )
    return enriched


def compute_pullback_phase_metrics(
    trades: list[Any],
    *,
    candidate_ledger: dict | None = None,
    selection_attribution: dict | None = None,
) -> dict[str, float]:
    del candidate_ledger
    total = float(len(trades))
    exit_counts = _count_by_exit_reason(trades)
    eod_trades = [trade for trade in trades if (trade.exit_reason or "") == "EOD_FLATTEN"]
    carry_trades = [trade for trade in trades if trade.hold_bars > 1]

    gap_down = [trade for trade in trades if _meta_float(trade, "entry_gap_pct") < -0.5]
    gap_up = [trade for trade in trades if _meta_float(trade, "entry_gap_pct") > 0.5]
    sweet_trend = [trade for trade in trades if 2.0 <= _meta_float(trade, "entry_sma_dist_pct") <= 10.0]
    outside_trend = [trade for trade in trades if not (2.0 <= _meta_float(trade, "entry_sma_dist_pct") <= 10.0)]
    deep_rsi = [trade for trade in trades if _meta_float(trade, "entry_rsi") < 5.0]
    shallow_rsi = [trade for trade in trades if 5.0 <= _meta_float(trade, "entry_rsi") < 10.0]
    late_rank = [trade for trade in trades if _meta_int(trade, "entry_rank") >= 4]
    early_rank = [trade for trade in trades if 1 <= _meta_int(trade, "entry_rank") <= 3]
    rsi_exits = [trade for trade in trades if (trade.exit_reason or "") == "RSI_EXIT"]
    stop_hits = [trade for trade in trades if (trade.exit_reason or "") == "STOP_HIT"]

    crowded_day_discrimination = 0.0
    crowded_day_missed_alpha_inverse = 0.0
    crowded_day_count = 0.0
    crowded_day_missed_alpha_days = 0.0
    if selection_attribution:
        crowded = [
            item for item in selection_attribution.values()
            if float(item.get("candidate_count", 0) or 0.0) > float(item.get("entered_count", 0) or 0.0)
        ]
        if crowded:
            crowded_day_count = float(len(crowded))
            entered_avg = float(np.mean([float(item.get("entered_avg_r", 0.0) or 0.0) for item in crowded]))
            skipped_avg = float(np.mean([float(item.get("skipped_avg_shadow_r", 0.0) or 0.0) for item in crowded]))
            crowded_day_discrimination = entered_avg - skipped_avg
            missed_days = sum(int(item.get("skipped_beating_worst_entered", 0) or 0) > 0 for item in crowded)
            crowded_day_missed_alpha_days = float(missed_days)
            crowded_day_missed_alpha_inverse = _clip01(1.0 - (missed_days / max(len(crowded), 1)))

    route_labels = [_route_label(trade) for trade in trades]
    route_trade_counts = {
        route: sum(1 for label in route_labels if label == route)
        for route in {"OPEN_SCORED_ENTRY", "OPENING_RECLAIM", "DELAYED_CONFIRM"}
    }
    open_scored_share = _share(sum(1 for label in route_labels if label == "OPEN_SCORED_ENTRY"), total)
    delayed_confirm_share = _share(route_trade_counts["DELAYED_CONFIRM"], total)
    route_diversity = _share(len({label for label in route_labels if label in {"OPEN_SCORED_ENTRY", "OPENING_RECLAIM", "DELAYED_CONFIRM"}}), 3.0)
    routes_with_min_10_trades = float(sum(count >= 10 for count in route_trade_counts.values()))

    carried_trades = [trade for trade in trades if _hold_days(trade) > 1 or str(_meta_str(trade, "carry_decision_path")) in {"binary", "score_fallback"}]
    binary_carried = [trade for trade in carried_trades if str(_meta_str(trade, "carry_decision_path")) == "binary"]
    delayed_confirm_carries = [
        trade for trade in carried_trades
        if _route_label(trade) == "DELAYED_CONFIRM"
    ]
    protection_candidates = [
        trade for trade in trades
        if float(trade.r_multiple) < 0 and _meta_float(trade, "mfe_before_negative_exit_r", _meta_float(trade, "mfe_r", 0.0)) > 0.25
    ]

    return {
        "avg_r": _avg_r(trades),
        "mean_entry_rank": _avg(_meta_int(trade, "entry_rank") for trade in trades),
        "mean_n_candidates": _avg(_meta_int(trade, "n_candidates") for trade in trades),
        "gap_down_avg_r": _avg_r(gap_down),
        "gap_up_avg_r": _avg_r(gap_up),
        "gap_selectivity_edge": _avg_r(gap_down) - _avg_r(gap_up),
        "trend_band_avg_r": _avg_r(sweet_trend),
        "trend_outside_avg_r": _avg_r(outside_trend),
        "trend_band_edge": _avg_r(sweet_trend) - _avg_r(outside_trend),
        "deep_rsi_avg_r": _avg_r(deep_rsi),
        "shallow_rsi_avg_r": _avg_r(shallow_rsi),
        "rsi_depth_edge": _avg_r(deep_rsi) - _avg_r(shallow_rsi),
        "late_rank_edge": _avg_r(late_rank) - _avg_r(early_rank),
        "signal_score_edge": _bucket_edge(trades, lambda trade: _meta_float(trade, "daily_signal_score", np.nan)),
        "crowded_day_count": crowded_day_count,
        "crowded_day_discrimination": crowded_day_discrimination,
        "crowded_day_missed_alpha_days": crowded_day_missed_alpha_days,
        "crowded_day_missed_alpha_inverse": crowded_day_missed_alpha_inverse,
        "route_score_monotonicity": _route_score_monotonicity(trades),
        "route_middle_bucket_deficit": _route_middle_bucket_deficit(trades),
        "open_scored_share": open_scored_share,
        "open_scored_inverse": _clip01(1.0 - open_scored_share / 0.60),
        "open_scored_trades": float(route_trade_counts["OPEN_SCORED_ENTRY"]),
        "opening_reclaim_trades": float(route_trade_counts["OPENING_RECLAIM"]),
        "delayed_confirm_trades": float(route_trade_counts["DELAYED_CONFIRM"]),
        "delayed_confirm_share": delayed_confirm_share,
        "routes_with_min_10_trades": routes_with_min_10_trades,
        "route_diversity": route_diversity,
        "eod_flatten_share": _share(len(eod_trades), total),
        "managed_exit_share": _share(total - len(eod_trades), total),
        "stop_hit_share": _share(exit_counts.get("STOP_HIT", 0), total),
        "rsi_exit_share": _share(exit_counts.get("RSI_EXIT", 0), total),
        "rsi_exit_avg_r": _avg_r(rsi_exits),
        "profit_target_share": _share(exit_counts.get("PROFIT_TARGET", 0), total),
        "positive_eod_share": _share(sum(1 for trade in eod_trades if trade.is_winner), len(eod_trades)),
        "carry_trade_share": _share(len(carry_trades), total),
        "carry_avg_r": _avg_r(carry_trades),
        "actual_carried_count": float(len(carried_trades)),
        "binary_carried_share": _share(len(binary_carried), len(carried_trades)),
        "delayed_confirm_carry_avg_r": _avg_r(delayed_confirm_carries),
        "stop_hit_avg_r": _avg_r(stop_hits),
        "stop_hit_total_r": float(sum(float(trade.r_multiple) for trade in stop_hits)),
        "protection_candidate_share": _share(len(protection_candidates), total),
    }


def _normalize_metric(metric_name: str, metrics: dict[str, float]) -> float:
    value = float(metrics.get(metric_name, 0.0))
    ceiling = _NORMALIZATION_CEILINGS.get(metric_name, 1.0)
    if ceiling <= 0:
        return 0.0
    return _clip01(value / ceiling)


def _count_by_exit_reason(trades: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        key = trade.exit_reason or "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _avg_r(trades: list[Any]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([float(trade.r_multiple) for trade in trades]))


def _avg(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values))


def _share(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _meta_float(trade: Any, key: str, default: float = 0.0) -> float:
    value = trade.metadata.get(key, default) if getattr(trade, "metadata", None) else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _meta_int(trade: Any, key: str, default: int = 0) -> int:
    value = trade.metadata.get(key, default) if getattr(trade, "metadata", None) else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _route_label(trade: Any) -> str:
    if not getattr(trade, "metadata", None):
        return "UNKNOWN"
    return str(
        trade.metadata.get("entry_route_family")
        or trade.metadata.get("selected_route")
        or trade.metadata.get("entry_trigger")
        or "UNKNOWN"
    ).upper()


def _route_score(trade: Any) -> float:
    for key in ("route_score", "intraday_score", "daily_signal_score"):
        value = _meta_float(trade, key, np.nan)
        if np.isfinite(value):
            return float(value)
    return float("nan")


def _bucket_edge(trades: list[Any], value_getter) -> float:
    pairs = [
        (float(value_getter(trade)), float(trade.r_multiple))
        for trade in trades
        if np.isfinite(float(value_getter(trade)))
    ]
    if len(pairs) < 8:
        return 0.0
    pairs.sort(key=lambda item: item[0])
    bucket = max(len(pairs) // 4, 1)
    lo = [r for _, r in pairs[:bucket]]
    hi = [r for _, r in pairs[-bucket:]]
    return float(np.mean(hi) - np.mean(lo))


def _route_score_monotonicity(trades: list[Any]) -> float:
    rows: list[float] = []
    for route in {"OPEN_SCORED_ENTRY", "OPENING_RECLAIM", "DELAYED_CONFIRM"}:
        pairs = [
            (_route_score(trade), float(trade.r_multiple))
            for trade in trades
            if _route_label(trade) == route and np.isfinite(_route_score(trade))
        ]
        score = _monotonicity_from_pairs(pairs)
        if score > 0:
            rows.append(score)
    return float(np.mean(rows)) if rows else 0.0


def _route_middle_bucket_deficit(trades: list[Any]) -> float:
    deficits: list[float] = []
    for route in {"OPEN_SCORED_ENTRY", "OPENING_RECLAIM", "DELAYED_CONFIRM"}:
        pairs = [
            (_route_score(trade), float(trade.r_multiple))
            for trade in trades
            if _route_label(trade) == route and np.isfinite(_route_score(trade))
        ]
        avgs = _quartile_avg_rs(pairs)
        if len(avgs) < 4:
            continue
        edge_worst = min(avgs[0], avgs[-1])
        middle_worst = min(avgs[1:-1])
        deficits.append(max(0.0, edge_worst - middle_worst))
    return float(max(deficits)) if deficits else 0.0


def _quartile_avg_rs(pairs: list[tuple[float, float]]) -> list[float]:
    if len(pairs) < 8:
        return []
    arr = np.array(sorted(pairs, key=lambda item: item[0]), dtype=float)
    chunks = [chunk for chunk in np.array_split(arr, 4) if len(chunk)]
    if len(chunks) < 4:
        return []
    return [float(np.mean(chunk[:, 1])) for chunk in chunks]


def _monotonicity_from_pairs(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 8:
        return 0.0
    arr = np.array(sorted(pairs, key=lambda item: item[0]), dtype=float)
    chunks = [chunk for chunk in np.array_split(arr, 4) if len(chunk)]
    if len(chunks) < 2:
        return 0.0
    avgs = [float(np.mean(chunk[:, 1])) for chunk in chunks]
    wins = sum(cur >= prev - 1e-9 for prev, cur in zip(avgs, avgs[1:]))
    return _clip01(wins / max(len(avgs) - 1, 1))


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _hold_days(trade: Any) -> int:
    if getattr(trade, "metadata", None):
        try:
            return int(trade.metadata.get("hold_days", getattr(trade, "hold_bars", 1)) or 1)
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(trade, "hold_bars", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _meta_str(trade: Any, key: str, default: str = "") -> str:
    value = trade.metadata.get(key, default) if getattr(trade, "metadata", None) else default
    return str(value or default)


# ═══════════════════════════════════════════════════════════════════════
# R5 -- Rebalanced scoring toward trade volume and total R
# ═══════════════════════════════════════════════════════════════════════

_R5_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "total_trades": 0.22,
        "expected_total_r": 0.20,
        "avg_r": 0.16,
        "profit_factor": 0.14,
        "signal_score_edge": 0.14,
        "inv_dd": 0.08,
        "sharpe": 0.06,
    },
    2: {
        "total_trades": 0.20,
        "expected_total_r": 0.22,
        "avg_r": 0.18,
        "profit_factor": 0.14,
        "signal_score_edge": 0.12,
        "inv_dd": 0.08,
        "sharpe": 0.06,
    },
    3: {
        "total_trades": 0.22,
        "expected_total_r": 0.22,
        "avg_r": 0.16,
        "profit_factor": 0.14,
        "inv_dd": 0.10,
        "sharpe": 0.08,
        "managed_exit_share": 0.08,
    },
    4: {
        "carry_avg_r": 0.18,
        "expected_total_r": 0.18,
        "avg_r": 0.16,
        "profit_factor": 0.14,
        "total_trades": 0.12,
        "carry_trade_share": 0.10,
        "inv_dd": 0.06,
        "sharpe": 0.06,
    },
    5: {
        "expected_total_r": 0.22,
        "sharpe": 0.18,
        "total_trades": 0.16,
        "avg_r": 0.14,
        "profit_factor": 0.12,
        "inv_dd": 0.12,
        "eod_flatten_inverse": 0.06,
    },
}

R5_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(weights)
    for phase, weights in _R5_PHASE_SCORING_WEIGHTS.items()
}


def get_r5_phase_scoring_weights(profile: str = "aggressive") -> dict[int, dict[str, float]]:
    del profile  # R5 uses a single weight set (no mainline/aggressive split)
    return R5_PHASE_SCORING_WEIGHTS


R5_ULTIMATE_TARGETS: dict[str, float] = {
    "avg_r": 0.18,
    "expected_total_r": 30.0,
    "profit_factor": 2.00,
    "sharpe": 1.20,
    "max_drawdown_pct": 0.07,
    "managed_exit_share": 0.55,
    "eod_flatten_inverse": 0.50,
    "total_trades": 140.0,
}

R5_MAINLINE_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 35, "max_dd_pct": 0.08, "min_pf": 1.40},
    2: {"min_trades": 45, "max_dd_pct": 0.07, "min_pf": 1.50},
    3: {"min_trades": 55, "max_dd_pct": 0.07, "min_pf": 1.60},
    4: {"min_trades": 55, "max_dd_pct": 0.065, "min_pf": 1.65},
    5: {"min_trades": 60, "max_dd_pct": 0.06, "min_pf": 1.70},
}

R5_AGGRESSIVE_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 30, "max_dd_pct": 0.09, "min_pf": 1.30},
    2: {"min_trades": 40, "max_dd_pct": 0.08, "min_pf": 1.40},
    3: {"min_trades": 50, "max_dd_pct": 0.07, "min_pf": 1.50},
    4: {"min_trades": 50, "max_dd_pct": 0.07, "min_pf": 1.55},
    5: {"min_trades": 55, "max_dd_pct": 0.065, "min_pf": 1.60},
}

R5_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": R5_MAINLINE_PHASE_HARD_REJECTS,
    "aggressive": R5_AGGRESSIVE_PHASE_HARD_REJECTS,
}


def score_r5_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    weights = dict(R5_PHASE_SCORING_WEIGHTS.get(phase, {}))
    if scoring_weights:
        weights.update(scoring_weights)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_phase_score_metrics(metrics)
    return sum(
        (weight / total_weight) * _normalize_metric(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


# ═══════════════════════════════════════════════════════════════════════
# V2R1 -- Phased Auto-Optimization for V2 Hybrid Engine
# Baseline: 815 trades, PF 1.37, avg_r +0.054, DD 3.3%, Sharpe 1.80
# ═══════════════════════════════════════════════════════════════════════

_V2R1_NORMALIZATION_CEILINGS: dict[str, float] = {
    "avg_r": 0.15,
    "expected_total_r": 100.0,
    "profit_factor": 3.00,
    "sharpe": 3.00,
    "total_trades": 1000.0,
    "inv_dd": 1.0,
    "managed_exit_share": 1.0,
    "eod_flatten_inverse": 1.0,
    "stop_hit_inverse": 1.0,
    "signal_score_edge": 0.20,
    "crowded_day_discrimination": 0.15,
    "route_diversity": 1.0,
    "route_score_monotonicity": 1.0,
    "protection_candidate_inverse": 1.0,
    "carry_avg_r": 0.50,
    "carry_trade_share": 0.60,
}

_V2R1_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    # Phase 1: Stop, MFE Protection & Exit Priority
    1: {
        "avg_r": 0.20,
        "expected_total_r": 0.18,
        "profit_factor": 0.14,
        "protection_candidate_inverse": 0.12,
        "stop_hit_inverse": 0.12,
        "managed_exit_share": 0.08,
        "total_trades": 0.08,
        "inv_dd": 0.08,
    },
    # Phase 2: Signal & Scoring
    2: {
        "signal_score_edge": 0.20,
        "expected_total_r": 0.18,
        "avg_r": 0.18,
        "profit_factor": 0.14,
        "total_trades": 0.14,
        "crowded_day_discrimination": 0.10,
        "inv_dd": 0.06,
    },
    # Phase 3: Route Diversification
    3: {
        "route_diversity": 0.18,
        "expected_total_r": 0.18,
        "total_trades": 0.18,
        "avg_r": 0.14,
        "profit_factor": 0.12,
        "route_score_monotonicity": 0.10,
        "inv_dd": 0.10,
    },
    # Phase 4: Carry & Overnight
    4: {
        "carry_avg_r": 0.18,
        "expected_total_r": 0.18,
        "avg_r": 0.16,
        "profit_factor": 0.14,
        "eod_flatten_inverse": 0.12,
        "total_trades": 0.12,
        "inv_dd": 0.10,
    },
    # Phase 5: Robustness & Sizing
    5: {
        "sharpe": 0.20,
        "expected_total_r": 0.20,
        "avg_r": 0.16,
        "inv_dd": 0.14,
        "profit_factor": 0.12,
        "total_trades": 0.12,
        "eod_flatten_inverse": 0.06,
    },
}

V2R1_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(weights)
    for phase, weights in _V2R1_PHASE_SCORING_WEIGHTS.items()
}


def get_v2r1_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile  # V2R1 uses a single weight set
    return V2R1_PHASE_SCORING_WEIGHTS


V2R1_ULTIMATE_TARGETS: dict[str, float] = {
    "avg_r": 0.10,
    "expected_total_r": 80.0,
    "profit_factor": 2.00,
    "sharpe": 2.50,
    "max_drawdown_pct": 0.04,
    "total_trades": 800.0,
    "route_diversity": 0.67,
    "managed_exit_share": 0.95,
    "eod_flatten_inverse": 0.95,
}

V2R1_MAINLINE_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 600, "max_dd_pct": 0.05, "min_pf": 1.15},
    2: {"min_trades": 600, "max_dd_pct": 0.05, "min_pf": 1.20},
    3: {"min_trades": 600, "max_dd_pct": 0.05, "min_pf": 1.20},
    4: {"min_trades": 650, "max_dd_pct": 0.045, "min_pf": 1.25},
    5: {"min_trades": 650, "max_dd_pct": 0.045, "min_pf": 1.30},
}

V2R1_AGGRESSIVE_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 500, "max_dd_pct": 0.06, "min_pf": 1.10},
    2: {"min_trades": 500, "max_dd_pct": 0.06, "min_pf": 1.15},
    3: {"min_trades": 500, "max_dd_pct": 0.06, "min_pf": 1.15},
    4: {"min_trades": 550, "max_dd_pct": 0.055, "min_pf": 1.20},
    5: {"min_trades": 550, "max_dd_pct": 0.055, "min_pf": 1.25},
}

V2R1_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V2R1_MAINLINE_PHASE_HARD_REJECTS,
    "aggressive": V2R1_AGGRESSIVE_PHASE_HARD_REJECTS,
}


def _normalize_v2r1_metric(metric_name: str, metrics: dict[str, float]) -> float:
    value = float(metrics.get(metric_name, 0.0))
    ceiling = _V2R1_NORMALIZATION_CEILINGS.get(metric_name, 1.0)
    if ceiling <= 0:
        return 0.0
    return _clip01(value / ceiling)


def enrich_v2r1_phase_score_metrics(metrics: dict[str, float]) -> dict[str, float]:
    enriched = dict(metrics)
    max_dd = float(metrics.get("max_drawdown_pct", 0.0))
    stop_hit_share = float(metrics.get("stop_hit_share", 0.0))
    eod_flatten_share = float(metrics.get("eod_flatten_share", 0.0))
    avg_r = float(metrics.get("avg_r", 0.0))
    total_trades = float(metrics.get("total_trades", 0.0))

    enriched["inv_dd"] = _clip01(1.0 - max_dd / 0.06)
    enriched["eod_flatten_inverse"] = _clip01(1.0 - eod_flatten_share)
    enriched["stop_hit_inverse"] = _clip01(1.0 - stop_hit_share / 0.30)
    enriched["expected_total_r"] = avg_r * total_trades
    enriched["protection_candidate_inverse"] = _clip01(
        1.0 - float(metrics.get("protection_candidate_share", 0.0)) / 0.30
    )
    return enriched


def score_v2r1_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    if scoring_weights:
        weights = dict(scoring_weights)
    else:
        weights = dict(V2R1_PHASE_SCORING_WEIGHTS.get(phase, {}))

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_v2r1_phase_score_metrics(metrics)
    return sum(
        (weight / total_weight) * _normalize_v2r1_metric(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


# ═══════════════════════════════════════════════════════════════════════
# V2R2 -- Alpha-Maximizing Phased Optimization (Post-Phase-3 Restart)
# Baseline: 810 trades, PF 1.55, avg_r +0.052, DD 1.9%, Sharpe 2.30
# Key change: immutable scoring (80% alpha, 20% risk) for ALL phases
# ═══════════════════════════════════════════════════════════════════════

_V2R2_NORMALIZATION_CEILINGS: dict[str, float] = {
    "avg_r": 0.12,
    "expected_total_r": 80.0,
    "profit_factor": 3.00,
    "total_trades": 1000.0,
    "inv_dd": 1.0,
}

# Immutable: same weights for ALL phases (80% alpha+frequency, 20% risk)
_V2R2_IMMUTABLE_WEIGHTS: dict[str, float] = {
    "expected_total_r": 0.40,
    "avg_r": 0.22,
    "total_trades": 0.18,
    "profit_factor": 0.10,
    "inv_dd": 0.10,
}

V2R2_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(dict(_V2R2_IMMUTABLE_WEIGHTS))
    for phase in (1, 2, 3)
}


def get_v2r2_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile  # V2R2 uses immutable weights -- no profile split
    return V2R2_PHASE_SCORING_WEIGHTS


V2R2_ULTIMATE_TARGETS: dict[str, float] = {
    "avg_r": 0.08,
    "expected_total_r": 65.0,
    "profit_factor": 2.00,
    "sharpe": 2.80,
    "max_drawdown_pct": 0.025,
    "total_trades": 850.0,
    "managed_exit_share": 0.97,
    "eod_flatten_inverse": 0.97,
}

# Same hard rejects for ALL phases (no per-phase escalation)
_V2R2_HARD_REJECTS: dict[str, float] = {
    "min_trades": 650,
    "max_dd_pct": 0.035,
    "min_pf": 1.25,
    "min_avg_r": 0.035,
    "min_expected_total_r": 35.0,
}

V2R2_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    phase: dict(_V2R2_HARD_REJECTS) for phase in (1, 2, 3)
}

V2R2_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V2R2_PHASE_HARD_REJECTS,
    "aggressive": V2R2_PHASE_HARD_REJECTS,
}


def _normalize_v2r2_metric(metric_name: str, metrics: dict[str, float]) -> float:
    value = float(metrics.get(metric_name, 0.0))
    ceiling = _V2R2_NORMALIZATION_CEILINGS.get(metric_name, 1.0)
    if ceiling <= 0:
        return 0.0
    return _clip01(value / ceiling)


def score_v2r2_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    if scoring_weights:
        weights = dict(scoring_weights)
    else:
        weights = dict(V2R2_PHASE_SCORING_WEIGHTS.get(phase, {}))

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_v2r1_phase_score_metrics(metrics)
    return sum(
        (weight / total_weight) * _normalize_v2r2_metric(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


# ═══════════════════════════════════════════════════════════════════════
# V2R3 -- Structural Engine Fixes + Re-optimization
# Baseline: V2R2 final (832 trades, PF 1.55, avg_r +0.055, DD 2.1%)
# 2 phases: carry quality tuning, route expansion
# Key change: V2 carry quality gate now active, DELAYED_CONFIRM rescue
# ═══════════════════════════════════════════════════════════════════════

# V2R3: own hard rejects (carry gate shifts baseline expected_total_r from ~46 to ~32)
V2R3_NORMALIZATION_CEILINGS = _V2R2_NORMALIZATION_CEILINGS
V2R3_IMMUTABLE_WEIGHTS = _V2R2_IMMUTABLE_WEIGHTS
_V2R3_HARD_REJECTS: dict[str, float] = {
    "min_trades": 650,
    "max_dd_pct": 0.035,
    "min_pf": 1.25,
    "min_avg_r": 0.030,
    "min_expected_total_r": 25.0,
}
V2R3_HARD_REJECTS = _V2R3_HARD_REJECTS
V2R3_ULTIMATE_TARGETS = V2R2_ULTIMATE_TARGETS

V2R3_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    1: _normalize_weights(dict(_V2R2_IMMUTABLE_WEIGHTS)),
}

V2R3_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: dict(_V2R3_HARD_REJECTS),
}

V2R3_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V2R3_PHASE_HARD_REJECTS,
    "aggressive": V2R3_PHASE_HARD_REJECTS,
}


def get_v2r3_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile  # V2R3 uses immutable weights -- no profile split
    return V2R3_PHASE_SCORING_WEIGHTS


def score_v2r3_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    return score_v2r2_pullback_phase(phase, metrics, scoring_weights)


# ═══════════════════════════════════════════════════════════════════════
# V2R4 -- Overnight Profit Lock + Intraday Protection
# Baseline: V2R3 final (832 trades, PF 1.58, avg_r +0.054, DD 2.1%)
# 2 phases: overnight profit lock, intraday protection + capacity
# Key change: parameterized overnight stop ratchet (profit_lock_r)
# ═══════════════════════════════════════════════════════════════════════

V2R4_NORMALIZATION_CEILINGS = _V2R2_NORMALIZATION_CEILINGS
V2R4_IMMUTABLE_WEIGHTS = _V2R2_IMMUTABLE_WEIGHTS
_V2R4_HARD_REJECTS: dict[str, float] = {
    "min_trades": 700,
    "max_dd_pct": 0.035,
    "min_pf": 1.35,
    "min_avg_r": 0.035,
    "min_expected_total_r": 30.0,
}
V2R4_HARD_REJECTS = _V2R4_HARD_REJECTS
V2R4_ULTIMATE_TARGETS = V2R2_ULTIMATE_TARGETS

V2R4_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(dict(_V2R2_IMMUTABLE_WEIGHTS))
    for phase in (1, 2)
}

V2R4_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    phase: dict(_V2R4_HARD_REJECTS) for phase in (1, 2)
}

V2R4_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V2R4_PHASE_HARD_REJECTS,
    "aggressive": V2R4_PHASE_HARD_REJECTS,
}


def get_v2r4_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile  # V2R4 uses immutable weights -- no profile split
    return V2R4_PHASE_SCORING_WEIGHTS


def score_v2r4_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    return score_v2r2_pullback_phase(phase, metrics, scoring_weights)


# ═══════════════════════════════════════════════════════════════════════
# V3R1 -- Ablation-First Integration (Multiphase + V2R4 + Tier B)
# Baseline: multiphase final + tier B (634 trades, PF 1.42, avg_r +0.082, 51.89R)
# Phase 1 tests REMOVING subsystems; Phases 2-4 add V2R4 enhancements
# Scoring: immutable V2R2 formula with updated ceilings for 634-trade base
# ═══════════════════════════════════════════════════════════════════════

_V3R1_NORMALIZATION_CEILINGS: dict[str, float] = {
    "avg_r": 0.15,
    "expected_total_r": 80.0,
    "profit_factor": 3.00,
    "total_trades": 750.0,
    "inv_dd": 1.0,
}

_V3R1_IMMUTABLE_WEIGHTS: dict[str, float] = {
    "expected_total_r": 0.40,
    "avg_r": 0.22,
    "total_trades": 0.18,
    "profit_factor": 0.10,
    "inv_dd": 0.10,
}

V3R1_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(dict(_V3R1_IMMUTABLE_WEIGHTS))
    for phase in (1, 2, 3, 4)
}

V3R1_ULTIMATE_TARGETS: dict[str, float] = {
    "avg_r": 0.10,
    "expected_total_r": 60.0,
    "profit_factor": 1.80,
    "sharpe": 2.50,
    "max_drawdown_pct": 0.030,
    "total_trades": 650.0,
    "managed_exit_share": 0.95,
    "eod_flatten_inverse": 0.95,
}

# Progressive hard rejects: relaxed for ablation (P1), tighten P2-P4
_V3R1_HARD_REJECTS_P1: dict[str, float] = {
    "min_trades": 400,
    "max_dd_pct": 0.050,
    "min_pf": 1.15,
    "min_avg_r": 0.025,
    "min_expected_total_r": 20.0,
}

_V3R1_HARD_REJECTS_P2: dict[str, float] = {
    "min_trades": 450,
    "max_dd_pct": 0.045,
    "min_pf": 1.18,
    "min_avg_r": 0.030,
    "min_expected_total_r": 22.0,
}

_V3R1_HARD_REJECTS_P3: dict[str, float] = {
    "min_trades": 500,
    "max_dd_pct": 0.040,
    "min_pf": 1.20,
    "min_avg_r": 0.035,
    "min_expected_total_r": 25.0,
}

_V3R1_HARD_REJECTS_P4: dict[str, float] = {
    "min_trades": 500,
    "max_dd_pct": 0.038,
    "min_pf": 1.25,
    "min_avg_r": 0.035,
    "min_expected_total_r": 27.0,
}

V3R1_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: dict(_V3R1_HARD_REJECTS_P1),
    2: dict(_V3R1_HARD_REJECTS_P2),
    3: dict(_V3R1_HARD_REJECTS_P3),
    4: dict(_V3R1_HARD_REJECTS_P4),
}

V3R1_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V3R1_PHASE_HARD_REJECTS,
    "aggressive": V3R1_PHASE_HARD_REJECTS,
}


def get_v3r1_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile  # V3R1 uses immutable weights -- no profile split
    return V3R1_PHASE_SCORING_WEIGHTS


def score_v3r1_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    if scoring_weights:
        weights = dict(scoring_weights)
    else:
        weights = dict(V3R1_PHASE_SCORING_WEIGHTS.get(phase, {}))

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_phase_score_metrics(metrics)
    return sum(
        (weight / total_weight) * min(max(float(enriched.get(metric_name, 0.0)), 0.0) / _V3R1_NORMALIZATION_CEILINGS.get(metric_name, 1.0), 1.0)
        for metric_name, weight in weights.items()
    )


# ═══════════════════════════════════════════════════════════════════════
# V4R1 -- Comprehensive Auto-Optimization (V2R4 base + all lineages)
# Baseline: V2R4 final (634 trades, PF 1.42, avg_r +0.082, 51.89R, Sharpe 2.04)
# Immutable scoring weights for all 5 phases; progressive hard rejects
# ═══════════════════════════════════════════════════════════════════════

_V4R1_NORMALIZATION_CEILINGS: dict[str, float] = {
    "avg_r": 0.15,
    "expected_total_r": 80.0,
    "profit_factor": 2.5,
    "total_trades": 800.0,
    "inv_dd": 0.97,
    "sharpe": 3.5,
}

# Immutable: same weights for ALL 5 phases (sum = 1.0)
_V4R1_IMMUTABLE_WEIGHTS: dict[str, float] = {
    "avg_r": 0.25,
    "expected_total_r": 0.22,
    "profit_factor": 0.18,
    "total_trades": 0.13,
    "inv_dd": 0.12,
    "sharpe": 0.10,
}

V4R1_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(dict(_V4R1_IMMUTABLE_WEIGHTS))
    for phase in (1, 2, 3, 4, 5)
}

V4R1_ULTIMATE_TARGETS: dict[str, float] = {
    "avg_r": 0.10,
    "expected_total_r": 65.0,
    "profit_factor": 1.80,
    "sharpe": 2.50,
    "max_drawdown_pct": 0.035,
    "total_trades": 700.0,
    "managed_exit_share": 0.95,
    "eod_flatten_inverse": 0.95,
}

# Progressive hard rejects: relaxed P1-P2, tighten P3-P4, strictest P5
_V4R1_HARD_REJECTS_P1P2: dict[str, float] = {
    "min_trades": 500,
    "min_pf": 1.20,
    "min_avg_r": 0.040,
    "min_expected_total_r": 30.0,
    "max_dd_pct": 0.050,
    "min_sharpe": 1.0,
}

_V4R1_HARD_REJECTS_P3P4: dict[str, float] = {
    "min_trades": 520,
    "min_pf": 1.25,
    "min_avg_r": 0.050,
    "min_expected_total_r": 35.0,
    "max_dd_pct": 0.045,
    "min_sharpe": 1.2,
}

_V4R1_HARD_REJECTS_P5: dict[str, float] = {
    "min_trades": 550,
    "min_pf": 1.30,
    "min_avg_r": 0.055,
    "min_expected_total_r": 40.0,
    "max_dd_pct": 0.040,
    "min_sharpe": 1.4,
}

V4R1_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: dict(_V4R1_HARD_REJECTS_P1P2),
    2: dict(_V4R1_HARD_REJECTS_P1P2),
    3: dict(_V4R1_HARD_REJECTS_P3P4),
    4: dict(_V4R1_HARD_REJECTS_P3P4),
    5: dict(_V4R1_HARD_REJECTS_P5),
}

V4R1_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V4R1_PHASE_HARD_REJECTS,
    "aggressive": V4R1_PHASE_HARD_REJECTS,
}


def get_v4r1_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile  # V4R1 uses immutable weights -- no profile split
    return V4R1_PHASE_SCORING_WEIGHTS


def _normalize_v4r1_metric(metric_name: str, metrics: dict[str, float]) -> float:
    value = float(metrics.get(metric_name, 0.0))
    ceiling = _V4R1_NORMALIZATION_CEILINGS.get(metric_name, 1.0)
    if ceiling <= 0:
        return 0.0
    return _clip01(value / ceiling)


def score_v4r1_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    if scoring_weights:
        weights = dict(scoring_weights)
    else:
        weights = dict(V4R1_PHASE_SCORING_WEIGHTS.get(phase, {}))

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_phase_score_metrics(metrics)
    return sum(
        (weight / total_weight) * _normalize_v4r1_metric(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


# ---------------------------------------------------------------------------
# V5R1 -- alpha/frequency score from the round-1 optimized baseline
#
# The immutable score intentionally has seven public components. The
# alpha_discrimination component is a stabilized composite of two pre-entry
# diagnostics: gap selectivity and crowded-day selected-vs-skipped shadow edge.
# ---------------------------------------------------------------------------
_V5R1_NORMALIZATION_CEILINGS: dict[str, float] = {
    "avg_r": 0.25,
    "expected_total_r": 150.0,
    "profit_factor": 3.25,
    "total_trades": 1000.0,
    "inv_dd": 1.0,
    "sharpe": 6.0,
    "alpha_discrimination": 1.0,
}

_V5R1_IMMUTABLE_WEIGHTS: dict[str, float] = {
    "expected_total_r": 0.30,
    "total_trades": 0.15,
    "avg_r": 0.15,
    "profit_factor": 0.12,
    "sharpe": 0.10,
    "inv_dd": 0.08,
    "alpha_discrimination": 0.10,
}

V5R1_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(dict(_V5R1_IMMUTABLE_WEIGHTS))
    for phase in (1, 2, 3, 4, 5)
}

V5R1_ULTIMATE_TARGETS: dict[str, float] = {
    "avg_r": 0.17,
    "expected_total_r": 125.0,
    "profit_factor": 2.35,
    "sharpe": 4.75,
    "max_drawdown_pct": 0.040,
    "total_trades": 800.0,
    "managed_exit_share": 0.97,
    "eod_flatten_inverse": 0.98,
}

_V5R1_HARD_REJECTS_P1: dict[str, float] = {
    "min_trades": 450,
    "min_pf": 1.15,
    "min_avg_r": 0.025,
    "min_expected_total_r": 20.0,
    "max_dd_pct": 0.080,
    "min_sharpe": 0.8,
}

_V5R1_HARD_REJECTS_P2: dict[str, float] = {
    "min_trades": 450,
    "min_pf": 1.15,
    "min_avg_r": 0.025,
    "min_expected_total_r": 20.0,
    "max_dd_pct": 0.080,
    "min_sharpe": 0.8,
}

_V5R1_HARD_REJECTS_P3: dict[str, float] = {
    "min_trades": 500,
    "min_pf": 1.20,
    "min_avg_r": 0.030,
    "min_expected_total_r": 22.0,
    "max_dd_pct": 0.075,
    "min_sharpe": 1.0,
}

_V5R1_HARD_REJECTS_P4: dict[str, float] = {
    "min_trades": 500,
    "min_pf": 1.20,
    "min_avg_r": 0.030,
    "min_expected_total_r": 22.0,
    "max_dd_pct": 0.070,
    "min_sharpe": 1.0,
}

_V5R1_HARD_REJECTS_P5: dict[str, float] = {
    "min_trades": 550,
    "min_pf": 1.25,
    "min_avg_r": 0.035,
    "min_expected_total_r": 25.0,
    "max_dd_pct": 0.065,
    "min_sharpe": 1.1,
}

V5R1_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: dict(_V5R1_HARD_REJECTS_P1),
    2: dict(_V5R1_HARD_REJECTS_P2),
    3: dict(_V5R1_HARD_REJECTS_P3),
    4: dict(_V5R1_HARD_REJECTS_P4),
    5: dict(_V5R1_HARD_REJECTS_P5),
}

V5R1_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V5R1_PHASE_HARD_REJECTS,
    "aggressive": V5R1_PHASE_HARD_REJECTS,
}


def get_v5r1_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile
    return V5R1_PHASE_SCORING_WEIGHTS


def _v5r1_alpha_discrimination(metrics: dict[str, float]) -> float:
    gap_component = _clip01(float(metrics.get("gap_selectivity_edge", 0.0)) / 0.30)
    crowded_component = _clip01(float(metrics.get("crowded_day_discrimination", 0.0)) / 0.20)
    return 0.50 * gap_component + 0.50 * crowded_component


def _normalize_v5r1_metric(metric_name: str, metrics: dict[str, float]) -> float:
    value = float(metrics.get(metric_name, 0.0))
    ceiling = _V5R1_NORMALIZATION_CEILINGS.get(metric_name, 1.0)
    if ceiling <= 0:
        return 0.0
    return _clip01(value / ceiling)


def score_v5r1_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    if scoring_weights:
        weights = dict(scoring_weights)
    else:
        weights = dict(V5R1_PHASE_SCORING_WEIGHTS.get(phase, {}))

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_phase_score_metrics(metrics)
    enriched["alpha_discrimination"] = _v5r1_alpha_discrimination(enriched)
    return sum(
        (weight / total_weight) * _normalize_v5r1_metric(metric_name, enriched)
        for metric_name, weight in weights.items()
    )


# ---------------------------------------------------------------------------
# V5R2 -- targeted residual-alpha score from the corrected round-2 baseline
#
# The score intentionally stays at seven components. It directly includes
# net PnL to avoid the R-score/dollar-PnL divergence observed in round 2,
# while still rewarding risk, sample size, and residual carry/selection alpha.
# ---------------------------------------------------------------------------
_V5R2_NORMALIZATION_CEILINGS: dict[str, float] = {
    "net_profit": 16_000.0,
    "expected_total_r": 135.0,
    "profit_factor": 2.15,
    "sharpe": 5.00,
    "inv_dd": 1.0,
    "total_trades": 1_100.0,
    "residual_alpha_quality": 1.0,
}

_V5R2_WEIGHTS: dict[str, float] = {
    "net_profit": 0.25,
    "expected_total_r": 0.20,
    "profit_factor": 0.15,
    "sharpe": 0.12,
    "inv_dd": 0.10,
    "total_trades": 0.10,
    "residual_alpha_quality": 0.08,
}

V5R2_PHASE_SCORING_WEIGHTS: dict[int, dict[str, float]] = {
    phase: _normalize_weights(dict(_V5R2_WEIGHTS))
    for phase in (1, 2, 3, 4)
}

V5R2_ULTIMATE_TARGETS: dict[str, float] = {
    "net_profit": 15_000.0,
    "avg_r": 0.13,
    "expected_total_r": 125.0,
    "profit_factor": 1.95,
    "sharpe": 4.75,
    "max_drawdown_pct": 0.035,
    "total_trades": 950.0,
}

_V5R2_HARD_REJECTS: dict[str, float] = {
    "min_trades": 900,
    "min_net_profit": 13_350.0,
    "min_pf": 1.75,
    "min_avg_r": 0.105,
    "min_expected_total_r": 112.0,
    "max_dd_pct": 0.040,
    "min_sharpe": 4.10,
}

V5R2_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    phase: dict(_V5R2_HARD_REJECTS)
    for phase in (1, 2, 3, 4)
}

V5R2_PHASE_HARD_REJECTS_BY_PROFILE: dict[str, dict[int, dict[str, float]]] = {
    "mainline": V5R2_PHASE_HARD_REJECTS,
    "aggressive": V5R2_PHASE_HARD_REJECTS,
}


def get_v5r2_phase_scoring_weights(profile: str = "mainline") -> dict[int, dict[str, float]]:
    del profile
    return V5R2_PHASE_SCORING_WEIGHTS


def _v5r2_residual_alpha_quality(metrics: dict[str, float]) -> float:
    alpha_component = _v5r1_alpha_discrimination(metrics)
    carry_avg_r = float(metrics.get("carry_avg_r", 0.0))
    carry_share = float(metrics.get("carry_trade_share", 0.0))
    if carry_share <= 0.01:
        carry_drag_inverse = 1.0
    else:
        carry_drag_inverse = _clip01((carry_avg_r + 0.30) / 0.30)
    return 0.50 * alpha_component + 0.50 * carry_drag_inverse


def _normalize_v5r2_metric(metric_name: str, metrics: dict[str, float]) -> float:
    if metric_name == "inv_dd":
        return _clip01(1.0 - float(metrics.get("max_drawdown_pct", 0.0)) / 0.05)
    if metric_name == "residual_alpha_quality":
        return _v5r2_residual_alpha_quality(metrics)
    value = float(metrics.get(metric_name, 0.0))
    ceiling = _V5R2_NORMALIZATION_CEILINGS.get(metric_name, 1.0)
    if ceiling <= 0:
        return 0.0
    return _clip01(value / ceiling)


def score_v5r2_pullback_phase(
    phase: int,
    metrics: dict[str, float],
    scoring_weights: dict[str, float] | None = None,
) -> float:
    if scoring_weights:
        weights = dict(scoring_weights)
    else:
        weights = dict(V5R2_PHASE_SCORING_WEIGHTS.get(phase, {}))

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    enriched = enrich_phase_score_metrics(metrics)
    enriched["expected_total_r"] = float(
        enriched.get("expected_total_r", 0.0)
        or float(enriched.get("avg_r", 0.0)) * float(enriched.get("total_trades", 0.0))
    )
    enriched["residual_alpha_quality"] = _v5r2_residual_alpha_quality(enriched)
    return sum(
        (weight / total_weight) * _normalize_v5r2_metric(metric_name, enriched)
        for metric_name, weight in weights.items()
    )
