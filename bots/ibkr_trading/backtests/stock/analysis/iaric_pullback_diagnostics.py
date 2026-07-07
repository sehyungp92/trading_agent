"""IARIC pullback diagnostics focused on alpha extraction and discrimination."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np

from backtests.stock.models import TradeRecord


def _meta(trade: TradeRecord, key: str, default=None):
    return trade.metadata.get(key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _share(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _pf(values: list[float]) -> float:
    gross_p = sum(v for v in values if v > 0)
    gross_l = abs(sum(v for v in values if v < 0))
    if gross_l <= 0:
        return float("inf") if gross_p > 0 else 0.0
    return gross_p / gross_l


def _hdr(title: str) -> str:
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}"


def _bootstrap_mean_ci(values: list[float], *, iterations: int = 200, seed: int = 7) -> tuple[float, float]:
    if len(values) < 2:
        mean = float(np.mean(values)) if values else 0.0
        return mean, mean
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(iterations)]
    lo, hi = np.percentile(means, [10, 90])
    return float(lo), float(hi)


def _trade_stats(trades: list[TradeRecord]) -> dict[str, float]:
    if not trades:
        return {"n": 0.0, "wr": 0.0, "avg_r": 0.0, "median_r": 0.0, "total_r": 0.0, "pf": 0.0, "pnl": 0.0}
    rs = [float(t.r_multiple) for t in trades]
    return {
        "n": float(len(trades)),
        "wr": _share(sum(1 for t in trades if t.is_winner), len(trades)),
        "avg_r": float(np.mean(rs)),
        "median_r": float(np.median(rs)),
        "total_r": float(sum(rs)),
        "pf": float(_pf(rs)),
        "pnl": float(sum(t.pnl_net for t in trades)),
    }


def _stats_from_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0.0, "wr": 0.0, "avg_r": 0.0, "median_r": 0.0, "total_r": 0.0, "pf": 0.0}
    return {
        "n": float(len(values)),
        "wr": _share(sum(1 for v in values if v > 0), len(values)),
        "avg_r": float(np.mean(values)),
        "median_r": float(np.median(values)),
        "total_r": float(sum(values)),
        "pf": float(_pf(values)),
    }


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _fmt_stats(stats: dict[str, float]) -> str:
    return f"n={int(stats['n'])}, WR={stats['wr']:.1%}, AvgR={stats['avg_r']:+.3f}, PF={stats['pf']:.2f}, TotalR={stats['total_r']:+.2f}"


def _quantile_indices(values: list[float], bins: int = 5) -> list[tuple[float, float]]:
    clean = sorted(v for v in values if np.isfinite(v))
    if len(clean) < max(8, bins):
        return []
    percentiles = np.linspace(0, 100, bins + 1)
    edges = [float(np.percentile(clean, p)) for p in percentiles]
    ranges: list[tuple[float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            hi = lo + 1e-9
        ranges.append((lo, hi))
    return ranges


def _group_by_quantiles(items: list[Any], value_getter, target_getter, *, bins: int = 5) -> list[dict[str, Any]]:
    values = [_safe_float(value_getter(item), np.nan) for item in items]
    ranges = _quantile_indices([v for v in values if np.isfinite(v)], bins=bins)
    if not ranges:
        return []
    rows: list[dict[str, Any]] = []
    for idx, (lo, hi) in enumerate(ranges, start=1):
        if idx == len(ranges):
            group = [item for item in items if lo <= _safe_float(value_getter(item), np.nan) <= hi]
        else:
            group = [item for item in items if lo <= _safe_float(value_getter(item), np.nan) < hi]
        targets = [_safe_float(target_getter(item)) for item in group]
        ci_lo, ci_hi = _bootstrap_mean_ci(targets)
        rows.append({
            "label": f"Q{idx} [{lo:.2f}, {hi:.2f}{']' if idx == len(ranges) else ')'}",
            "n": len(group),
            "wr": _share(sum(1 for val in targets if val > 0), len(targets)),
            "avg_r": float(np.mean(targets)) if targets else 0.0,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
    return rows


def _interaction_table(trades: list[TradeRecord], x_key: str, y_key: str, *, x_bins: int = 3, y_bins: int = 3) -> dict[str, Any]:
    x_ranges = _quantile_indices([_safe_float(_meta(t, x_key, np.nan), np.nan) for t in trades], bins=x_bins)
    y_ranges = _quantile_indices([_safe_float(_meta(t, y_key, np.nan), np.nan) for t in trades], bins=y_bins)
    if not x_ranges or not y_ranges:
        return {"cells": []}
    cells: list[list[dict[str, Any]]] = []
    for y_lo, y_hi in y_ranges:
        row: list[dict[str, Any]] = []
        for x_lo, x_hi in x_ranges:
            group = [
                t for t in trades
                if x_lo <= _safe_float(_meta(t, x_key, np.nan), np.nan) <= x_hi
                and y_lo <= _safe_float(_meta(t, y_key, np.nan), np.nan) <= y_hi
            ]
            rs = [float(t.r_multiple) for t in group]
            row.append({"n": len(group), "avg_r": float(np.mean(rs)) if rs else 0.0})
        cells.append(row)
    return {"cells": cells}


def _close_in_range_pct(high: float, low: float, close: float) -> float:
    if high <= low:
        return 1.0
    return float(min(max((close - low) / (high - low), 0.0), 1.0))


def _next_flow_reversal(replay, symbol: str, trade_date: date, lookback: int) -> bool:
    if replay is None:
        return False
    last_n = replay.get_flow_proxy_last_n(symbol, trade_date, max(1, lookback))
    return bool(last_n is not None and all(v < 0 for v in last_n))


def _simulate_daily_policy(
    trade: TradeRecord,
    replay,
    *,
    hold_days: int,
    carry_min_r: float,
    close_pct_min: float = 0.0,
    mfe_gate_r: float = 0.0,
    flow_lookback: int = 1,
    close_stop: bool = False,
) -> dict[str, Any] | None:
    if replay is None or trade.risk_per_share <= 0:
        return None
    sym = trade.symbol
    trade_date = trade.entry_time.date()
    entry_price = float(trade.entry_price)
    risk_per_share = float(trade.risk_per_share)
    stop_price = entry_price - risk_per_share
    ohlc = replay.get_daily_ohlc(sym, trade_date)
    if ohlc is None:
        return None
    _, H, L, C = ohlc
    max_favorable = H
    max_adverse = L
    close_pct = _close_in_range_pct(H, L, C)
    close_r = (C - entry_price) / risk_per_share
    exit_price = None
    exit_reason = ""
    days_held = 1

    if L <= stop_price:
        exit_price = C if close_stop else stop_price
        exit_reason = "STOP_HIT"
    elif hold_days <= 0 or close_r < carry_min_r or close_pct < close_pct_min or ((H - entry_price) / risk_per_share) < mfe_gate_r:
        exit_price = C
        exit_reason = "EOD_FLATTEN"
    else:
        current_date = trade_date
        for _ in range(hold_days):
            next_date = replay.get_next_trading_date(current_date)
            if next_date is None:
                break
            current_date = next_date
            ohlc_next = replay.get_daily_ohlc(sym, current_date)
            if ohlc_next is None:
                break
            O2, H2, L2, C2 = ohlc_next
            days_held += 1
            max_favorable = max(max_favorable, H2)
            max_adverse = min(max_adverse, L2)
            if _next_flow_reversal(replay, sym, replay.get_prev_trading_date(current_date) or trade_date, flow_lookback):
                exit_price = O2
                exit_reason = "FLOW_REVERSAL"
                break
            if L2 <= stop_price:
                exit_price = C2 if close_stop else stop_price
                exit_reason = "STOP_HIT"
                break
            exit_price = C2
            exit_reason = "TIME_STOP"
    if exit_price is None:
        exit_price = C
        exit_reason = "EOD_FLATTEN"
    r_mult = (float(exit_price) - entry_price) / risk_per_share
    return {"r": float(r_mult), "exit_reason": exit_reason, "hold_days": int(days_held), "close_pct": float(close_pct), "close_r": float(close_r)}


def _entry_variant_result(trade: TradeRecord, replay, variant: str) -> dict[str, Any] | None:
    if replay is None or trade.risk_per_share <= 0:
        return None
    arrs = getattr(replay, "get_5m_arrays_for_date", None)
    if arrs is None:
        return None
    arrs = replay.get_5m_arrays_for_date(trade.symbol, trade.entry_time.date())
    if arrs is None:
        return None
    opens = arrs["open"]
    highs = arrs["high"]
    lows = arrs["low"]
    closes = arrs["close"]
    n = len(opens)
    if n == 0:
        return None
    index = 0
    entry_price = float(opens[0])
    label = "Open"
    if variant == "delay_30m":
        index = min(6, n - 1)
        entry_price = float(opens[index])
        label = "Delay 30m"
    elif variant == "delay_60m":
        index = min(12, n - 1)
        entry_price = float(opens[index])
        label = "Delay 60m"
    elif variant == "first_reversal_close":
        chosen = None
        for idx in range(1, n):
            if closes[idx] > closes[idx - 1] and closes[idx - 1] <= opens[idx - 1]:
                chosen = idx
                break
        if chosen is None:
            return None
        index = chosen
        entry_price = float(closes[index])
        label = "First reversal close"
    risk_per_share = float(trade.risk_per_share)
    stop_price = entry_price - risk_per_share
    exit_price = float(closes[-1])
    exit_reason = "EOD_FLATTEN"
    for idx in range(index, n):
        if lows[idx] <= stop_price:
            exit_price = stop_price
            exit_reason = "STOP_HIT"
            break
        exit_price = float(closes[idx])
    day_low = float(lows.min())
    day_high = float(highs.max())
    entry_location = _share(entry_price - day_low, max(day_high - day_low, 1e-9))
    return {
        "label": label,
        "r": float((exit_price - entry_price) / risk_per_share),
        "exit_reason": exit_reason,
        "entry_location": float(entry_location),
        "entry_price": float(entry_price),
        "entry_index": int(index),
        "entry_time": None,
    }


def _best_feasible_entry_result(trade: TradeRecord, replay) -> dict[str, Any] | None:
    if replay is None or trade.risk_per_share <= 0:
        return None
    arrs = getattr(replay, "get_5m_arrays_for_date", None)
    if arrs is None:
        return None
    arrs = replay.get_5m_arrays_for_date(trade.symbol, trade.entry_time.date())
    if arrs is None:
        return None
    opens = arrs["open"]
    highs = arrs["high"]
    lows = arrs["low"]
    closes = arrs["close"]
    n = len(opens)
    if n == 0:
        return None
    risk_per_share = float(trade.risk_per_share)
    day_low = float(lows.min())
    day_high = float(highs.max())
    best: dict[str, Any] | None = None
    for idx in range(n):
        entry_price = float(closes[idx])
        stop_price = entry_price - risk_per_share
        exit_price = float(closes[-1])
        exit_reason = "EOD_FLATTEN"
        for jdx in range(idx, n):
            if lows[jdx] <= stop_price:
                exit_price = stop_price
                exit_reason = "STOP_HIT"
                break
            exit_price = float(closes[jdx])
        r_val = float((exit_price - entry_price) / risk_per_share)
        if best is None or r_val > float(best["r"]):
            best = {
                "label": "Best feasible close",
                "r": r_val,
                "exit_reason": exit_reason,
                "entry_location": float(_share(entry_price - day_low, max(day_high - day_low, 1e-9))),
                "entry_price": float(entry_price),
                "entry_index": int(idx),
                "entry_time": None,
            }
    return best


def _compute_funnel(trades: list[TradeRecord], funnel_counters: dict[str, int] | None, rejection_log: list[dict[str, Any]] | None) -> dict[str, Any]:
    counters = dict(funnel_counters or {})
    entered = int(counters.get("entered", len(trades)))
    pool = int(counters.get("candidate_pool", entered))
    actual = _trade_stats(trades)
    by_gate: dict[str, list[float]] = defaultdict(list)
    for item in rejection_log or []:
        gate = str(item.get("gate") or "unknown")
        if item.get("shadow_r") is not None:
            by_gate[gate].append(float(item["shadow_r"]))
    gate_rows = []
    for gate, values in sorted(by_gate.items(), key=lambda item: (-len(item[1]), item[0])):
        stats = _stats_from_values(values)
        gate_rows.append({
            "gate": gate,
            "count": len(values),
            "avg_r": stats["avg_r"],
            "wr": stats["wr"],
            "false_positive": _share(sum(1 for v in values if v > 0), len(values)),
            "verdict": "KEEP" if stats["avg_r"] <= actual["avg_r"] else "REVIEW",
        })
    return {"counters": counters, "entered": entered, "candidate_pool": pool, "accept_rate": _share(entered, pool), "gate_rows": gate_rows}


def _compute_shadow_summary(rejection_log: list[dict[str, Any]] | None, trades: list[TradeRecord]) -> dict[str, Any]:
    actual = _trade_stats(trades)
    shadow_vals = [float(item["shadow_r"]) for item in rejection_log or [] if item.get("shadow_r") is not None]
    shadow = _stats_from_values(shadow_vals)
    return {"actual": actual, "shadow": shadow, "delta_avg_r": actual["avg_r"] - shadow["avg_r"], "delta_wr": actual["wr"] - shadow["wr"]}


def _compute_selection_summary(selection_attribution: dict[date, dict[str, Any]] | None) -> dict[str, Any]:
    if not selection_attribution:
        return {"crowded_days": 0, "entered_avg_r": 0.0, "skipped_avg_shadow_r": 0.0, "days_with_missed_alpha": 0, "top_days": []}
    crowded = [item for item in selection_attribution.values() if item.get("candidate_count", 0) > item.get("entered_count", 0)]
    top_days = sorted(
        (
            {
                "trade_date": trade_date,
                **item,
                "selection_delta": _safe_float(item.get("entered_avg_r")) - _safe_float(item.get("skipped_avg_shadow_r")),
            }
            for trade_date, item in selection_attribution.items()
            if item.get("candidate_count", 0) > item.get("entered_count", 0)
        ),
        key=lambda item: (-_safe_int(item.get("skipped_beating_worst_entered")), -_safe_float(item.get("best_skipped_shadow_r"))),
    )
    return {
        "crowded_days": len(crowded),
        "entered_avg_r": float(np.mean([_safe_float(item.get("entered_avg_r")) for item in crowded])) if crowded else 0.0,
        "skipped_avg_shadow_r": float(np.mean([_safe_float(item.get("skipped_avg_shadow_r")) for item in crowded])) if crowded else 0.0,
        "days_with_missed_alpha": sum(1 for item in crowded if _safe_float(item.get("skipped_avg_shadow_r")) > _safe_float(item.get("entered_avg_r"))),
        "top_days": top_days[:5],
    }


def _compute_monotonicity(trades: list[TradeRecord]) -> dict[str, list[dict[str, Any]]]:
    features = {
        "entry_rsi": lambda t: _meta(t, "entry_rsi", np.nan),
        "entry_rank": lambda t: _meta(t, "entry_rank", np.nan),
        "entry_rank_pct": lambda t: _meta(t, "entry_rank_pct", np.nan),
        "daily_signal_score": lambda t: _meta(t, "daily_signal_score", np.nan),
        "daily_signal_rank_pct": lambda t: _meta(t, "daily_signal_rank_pct", np.nan),
        "route_score": lambda t: _meta(t, "route_score", np.nan),
        "intraday_score": lambda t: _meta(t, "intraday_score", np.nan),
        "entry_sma_dist_pct": lambda t: _meta(t, "entry_sma_dist_pct", np.nan),
        "entry_cdd": lambda t: _meta(t, "entry_cdd", np.nan),
        "entry_gap_pct": lambda t: _meta(t, "entry_gap_pct", np.nan),
        "close_pct": lambda t: _meta(t, "close_pct", np.nan),
        "mfe_r": lambda t: _meta(t, "mfe_r", np.nan),
    }
    return {key: _group_by_quantiles(trades, getter, lambda t: float(t.r_multiple)) for key, getter in features.items()}


def _compute_mfe_capture(trades: list[TradeRecord]) -> dict[str, Any]:
    by_reason: dict[str, list[TradeRecord]] = defaultdict(list)
    lost_alpha: list[dict[str, Any]] = []
    for trade in trades:
        reason = trade.exit_reason or "UNKNOWN"
        by_reason[reason].append(trade)
        mfe_r = _safe_float(_meta(trade, "mfe_r", 0.0))
        if reason == "EOD_FLATTEN" and mfe_r > 0:
            lost_alpha.append({"symbol": trade.symbol, "trade_date": trade.entry_time.date(), "actual_r": float(trade.r_multiple), "mfe_r": mfe_r, "lost_r": mfe_r - float(trade.r_multiple)})
    rows = []
    for reason, group in sorted(by_reason.items(), key=lambda item: -len(item[1])):
        capture = []
        giveback = []
        for trade in group:
            mfe_r = _safe_float(_meta(trade, "mfe_r", 0.0))
            if mfe_r > 0:
                capture.append(float(trade.r_multiple) / mfe_r)
                giveback.append(mfe_r - float(trade.r_multiple))
        rows.append({"reason": reason, "count": len(group), "avg_r": float(np.mean([float(t.r_multiple) for t in group])) if group else 0.0, "capture": float(np.mean(capture)) if capture else 0.0, "giveback": float(np.mean(giveback)) if giveback else 0.0})
    return {"rows": rows, "lost_alpha": sorted(lost_alpha, key=lambda item: item["lost_r"], reverse=True)[:10]}


def _compute_exit_frontier(trades: list[TradeRecord], replay) -> list[dict[str, Any]]:
    actual = _trade_stats(trades)
    frontier = [{"label": "Actual", "n": int(actual["n"]), "avg_r": actual["avg_r"], "pf": actual["pf"], "wr": actual["wr"]}]
    if replay is None:
        return frontier
    variants = [
        ("Carry 1d r>=0.00", dict(hold_days=1, carry_min_r=0.0)),
        ("Carry 2d r>=0.00", dict(hold_days=2, carry_min_r=0.0)),
        ("Carry 3d r>=0.10", dict(hold_days=3, carry_min_r=0.10)),
        ("Carry 5d r>=0.10", dict(hold_days=5, carry_min_r=0.10)),
        ("Carry 5d r>=0.25", dict(hold_days=5, carry_min_r=0.25)),
        ("Carry 3d + close>=0.65", dict(hold_days=3, carry_min_r=0.10, close_pct_min=0.65)),
        ("Carry 3d + close>=0.65 + mfe>=0.25", dict(hold_days=3, carry_min_r=0.10, close_pct_min=0.65, mfe_gate_r=0.25)),
        ("Carry 3d + flowrev 2", dict(hold_days=3, carry_min_r=0.10, flow_lookback=2)),
        ("Carry 3d + flowrev 3", dict(hold_days=3, carry_min_r=0.10, flow_lookback=3)),
        ("Carry 3d + close stop", dict(hold_days=3, carry_min_r=0.10, close_stop=True)),
    ]
    for label, params in variants:
        sims = [_simulate_daily_policy(trade, replay, **params) for trade in trades]
        sims = [item for item in sims if item is not None]
        if not sims:
            continue
        rs = [float(item["r"]) for item in sims]
        frontier.append({"label": label, "n": len(sims), "avg_r": float(np.mean(rs)), "pf": float(_pf(rs)), "wr": _share(sum(1 for v in rs if v > 0), len(rs))})
    return frontier


def _compute_carry_funnel(trades: list[TradeRecord], replay) -> dict[str, Any]:
    eod = [trade for trade in trades if (trade.exit_reason or "UNKNOWN") == "EOD_FLATTEN"]
    profitable = [trade for trade in eod if _safe_float(_meta(trade, "close_r", 0.0)) > 0]
    close_pct_gate = [trade for trade in profitable if _safe_float(_meta(trade, "close_pct", 0.0)) >= 0.65]
    mfe_gate = [trade for trade in close_pct_gate if _safe_float(_meta(trade, "mfe_r", 0.0)) >= 0.25]
    flow_ok = [trade for trade in mfe_gate if replay is None or not _next_flow_reversal(replay, trade.symbol, trade.entry_time.date(), 1)]
    forward_rows = []
    route_rows = []
    if replay is not None and flow_ok:
        for horizon in [1, 3, 5]:
            sims = [_simulate_daily_policy(trade, replay, hold_days=horizon, carry_min_r=0.10, close_pct_min=0.65, mfe_gate_r=0.25) for trade in flow_ok]
            sims = [item for item in sims if item is not None]
            if sims:
                rs = [float(item["r"]) for item in sims]
                forward_rows.append({"label": f"Carry +{horizon}d", "n": len(sims), "avg_r": float(np.mean(rs)), "pf": float(_pf(rs))})
    for route in _ordered_route_labels(sorted({_route_label(trade) for trade in eod})):
        group = [trade for trade in eod if _route_label(trade) == route]
        if not group:
            continue
        route_rows.append({
            "route": route,
            "eod": len(group),
            "profitable": sum(1 for trade in group if _safe_float(_meta(trade, "close_r", 0.0)) > 0),
            "carry_binary_ok": sum(1 for trade in group if bool(_meta(trade, "carry_binary_ok", False))),
            "carry_score_ok": sum(1 for trade in group if bool(_meta(trade, "carry_score_ok", False))),
        })
    return {
        "eod": len(eod),
        "profitable": len(profitable),
        "close_pct_gate": len(close_pct_gate),
        "mfe_gate": len(mfe_gate),
        "flow_ok": len(flow_ok),
        "forward_rows": forward_rows,
        "route_rows": route_rows,
    }


def _compute_entry_timing(trades: list[TradeRecord], replay) -> list[dict[str, Any]]:
    if replay is None:
        return []
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        for variant in ["open", "delay_30m", "delay_60m", "first_reversal_close"]:
            result = _entry_variant_result(trade, replay, variant)
            if result is not None:
                by_label[result["label"]].append(result)
    rows = []
    for label in ["Open", "Delay 30m", "Delay 60m", "First reversal close"]:
        results = by_label.get(label, [])
        if not results:
            continue
        rs = [float(item["r"]) for item in results]
        rows.append({"label": label, "n": len(results), "avg_r": float(np.mean(rs)), "pf": float(_pf(rs)), "wr": _share(sum(1 for v in rs if v > 0), len(rs)), "entry_location": float(np.mean([float(item["entry_location"]) for item in results]))})
    return rows


def _compute_low_trade_days(candidate_ledger: dict[date, list[dict[str, Any]]] | None) -> list[dict[str, Any]]:
    if not candidate_ledger:
        return []
    rows = []
    for trade_date, records in candidate_ledger.items():
        entered = [item for item in records if item.get("disposition") == "entered"]
        skipped = [item for item in records if item.get("disposition") != "entered"]
        shadow_vals = [float(item["shadow_r"]) for item in skipped if item.get("shadow_r") is not None]
        gate_counts: dict[str, int] = defaultdict(int)
        for item in skipped:
            gate_counts[str(item.get("disposition") or "unknown")] += 1
        rows.append({
            "trade_date": trade_date,
            "candidate_count": len(records),
            "entered_count": len(entered),
            "avg_skipped_shadow_r": float(np.mean(shadow_vals)) if shadow_vals else 0.0,
            "best_skipped_shadow_r": max(shadow_vals) if shadow_vals else 0.0,
            "top_gates": sorted(gate_counts.items(), key=lambda item: (-item[1], item[0]))[:3],
        })
    return sorted([item for item in rows if item["candidate_count"] >= 5 and item["entered_count"] <= 1], key=lambda item: (-item["best_skipped_shadow_r"], -item["candidate_count"]))[:5]


def _compute_concentration(trades: list[TradeRecord]) -> dict[str, Any]:
    by_sector: dict[str, list[TradeRecord]] = defaultdict(list)
    by_day: dict[int, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        by_sector[trade.sector or "UNKNOWN"].append(trade)
        by_day[trade.entry_time.weekday()].append(trade)
    sector_rows = [{"label": sector, "n": len(group), "avg_r": float(np.mean([float(t.r_multiple) for t in group])) if group else 0.0} for sector, group in sorted(by_sector.items(), key=lambda item: -len(item[1]))[:5]]
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    day_rows = [{"label": weekday_labels[idx], "n": len(by_day.get(idx, [])), "avg_r": float(np.mean([float(t.r_multiple) for t in by_day.get(idx, [])])) if by_day.get(idx) else 0.0} for idx in range(5)]
    return {"sector_rows": sector_rows, "day_rows": day_rows}


def _compute_intraday_summary(
    trades: list[TradeRecord],
    candidate_ledger: dict[date, list[dict[str, Any]]] | None,
    fsm_log: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    records = [record for rows in (candidate_ledger or {}).values() for record in rows]
    intraday_records = [
        record
        for record in records
        if "intraday_setup_type" in record or str(record.get("disposition") or "") == "no_intraday_data"
    ]
    if not intraday_records and not fsm_log:
        return {
            "stage_counts": {
                "watchlist": 0,
                "flush_locked": 0,
                "reclaiming": 0,
                "ready": 0,
                "entered": 0,
                "partial": 0,
                "trailed": 0,
                "carried": 0,
            },
            "trigger_rows": [],
            "hour_rows": [],
            "transition_rows": [],
            "coverage": {"with_5m": 0, "missing_5m": 0, "missing_5m_share": 0.0, "fallback_share": 0.0},
            "live_selector": {
                "considered": 0,
                "accepted": 0,
                "rejected": 0,
                "accepted_avg_r": 0.0,
                "rejected_avg_shadow_r": 0.0,
                "delta_avg_r": 0.0,
            },
        }
    live_records = [record for record in intraday_records if bool(record.get("intraday_data_available"))]
    fallback_records = [record for record in intraday_records if not bool(record.get("intraday_data_available"))]
    stage_counts = {
        "watchlist": len(intraday_records),
        "flush_locked": sum(1 for record in intraday_records if record.get("stage_flush_locked")),
        "reclaiming": sum(1 for record in intraday_records if record.get("stage_reclaiming")),
        "ready": sum(1 for record in intraday_records if record.get("stage_ready")),
        "entered": sum(1 for record in intraday_records if record.get("disposition") == "entered"),
        "partial": sum(1 for record in intraday_records if record.get("partial_taken")),
        "trailed": sum(1 for record in intraday_records if record.get("trail_active")),
        "carried": sum(1 for record in intraday_records if _safe_int(record.get("actual_hold_days"), 0) > 1),
    }
    trigger_rows: list[dict[str, Any]] = []
    by_trigger: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        trigger = str(_meta(trade, "entry_trigger", "UNKNOWN") or "UNKNOWN")
        by_trigger[trigger].append(trade)
    for trigger, group in sorted(by_trigger.items(), key=lambda item: (-len(item[1]), item[0])):
        trigger_rows.append({
            "label": trigger,
            "n": len(group),
            "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])) if group else 0.0,
            "wr": _share(sum(1 for trade in group if trade.is_winner), len(group)),
        })

    by_hour: dict[int, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        by_hour[trade.entry_time.hour].append(trade)
    hour_rows = [
        {
            "label": f"{hour:02d}:00",
            "n": len(group),
            "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])) if group else 0.0,
        }
        for hour, group in sorted(by_hour.items())
    ]
    transitions: dict[str, int] = defaultdict(int)
    for row in fsm_log or []:
        transitions[f"{row.get('from_state', '?')}->{row.get('to_state', '?')}"] += 1
    transition_rows = sorted(
        ({"label": label, "count": count} for label, count in transitions.items()),
        key=lambda item: (-item["count"], item["label"]),
    )[:8]
    accepted_live = [record for record in live_records if str(record.get("disposition") or "") == "entered"]
    rejected_live = [record for record in live_records if str(record.get("disposition") or "") != "entered"]
    accepted_live_r = [_safe_float(record.get("actual_r"), np.nan) for record in accepted_live if record.get("actual_r") is not None]
    rejected_live_shadow = [_safe_float(record.get("shadow_r"), np.nan) for record in rejected_live if record.get("shadow_r") is not None]
    return {
        "stage_counts": stage_counts,
        "trigger_rows": trigger_rows,
        "hour_rows": hour_rows,
        "transition_rows": transition_rows,
        "coverage": {
            "with_5m": len(live_records),
            "missing_5m": len(fallback_records),
            "missing_5m_share": _share(len(fallback_records), len(intraday_records)),
            "fallback_share": _share(len(fallback_records), len(intraday_records)),
        },
        "live_selector": {
            "considered": len(live_records),
            "accepted": len(accepted_live),
            "rejected": len(rejected_live),
            "accepted_avg_r": float(np.mean(accepted_live_r)) if accepted_live_r else 0.0,
            "rejected_avg_shadow_r": float(np.mean(rejected_live_shadow)) if rejected_live_shadow else 0.0,
            "delta_avg_r": (float(np.mean(accepted_live_r)) if accepted_live_r else 0.0) - (float(np.mean(rejected_live_shadow)) if rejected_live_shadow else 0.0),
        },
    }


def _all_candidate_records(candidate_ledger: dict[date, list[dict[str, Any]]] | None) -> list[dict[str, Any]]:
    return [record for rows in (candidate_ledger or {}).values() for record in rows]


def _record_trade_date(record: dict[str, Any]) -> date | None:
    value = record.get("trade_date")
    return value if isinstance(value, date) else None


def _record_timestamp(record: dict[str, Any], key: str) -> Any:
    value = record.get(key)
    return value


def _blocked_capacity_reason(record: dict[str, Any]) -> str:
    blocked = str(record.get("blocked_by_capacity_reason") or "")
    if blocked:
        return blocked
    disposition = str(record.get("disposition") or "")
    if disposition == "intraday_priority_reserve":
        return "intraday_priority_reserve"
    if disposition == "sector_cap_reject":
        return "sector_cap"
    if disposition == "buying_power_reject":
        return "buying_power"
    if disposition in {"priority_reject", "position_cap_reject"}:
        return "slot_cap"
    if disposition == "rescue_cap_reject":
        return "rescue_cap"
    return ""


def _route_label(trade: TradeRecord) -> str:
    return str(
        _meta(trade, "entry_route_family", None)
        or _meta(trade, "selected_route", None)
        or _meta(trade, "entry_trigger", "UNKNOWN")
        or "UNKNOWN"
    ).upper()


def _trade_route_score(trade: TradeRecord) -> float:
    for key in ("route_score", "intraday_score", "daily_signal_score"):
        value = _safe_float(_meta(trade, key, np.nan), np.nan)
        if np.isfinite(value):
            return float(value)
    return float("nan")


def _ordered_route_labels(labels: list[str]) -> list[str]:
    preferred = [
        "OPEN_SCORED_ENTRY",
        "OPENING_RECLAIM",
        "DELAYED_CONFIRM",
        "NO_TRADE",
        "PM_REENTRY",
        "UNKNOWN",
    ]
    rank = {label: idx for idx, label in enumerate(preferred)}
    return sorted(labels, key=lambda label: (rank.get(label, len(preferred)), label))


def _half_hour_bucket(ts) -> str:
    if ts is None:
        return "N/A"
    minute = 30 if getattr(ts, "minute", 0) >= 30 else 0
    return f"{getattr(ts, 'hour', 0):02d}:{minute:02d}"


def _bars_to_exit(trade: TradeRecord) -> int:
    bars = _safe_int(_meta(trade, "bars_to_exit", 0), 0)
    if bars > 0:
        return bars
    return max(int(round(trade.hold_hours * 12.0)), 1)


def _bars_to_mfe(trade: TradeRecord) -> int:
    bars = _safe_int(_meta(trade, "bars_to_mfe", 0), 0)
    if bars > 0:
        return bars
    return _bars_to_exit(trade)


def _selector_value(record: dict[str, Any], *, accepted: bool) -> float | None:
    if accepted:
        if record.get("actual_r") is not None:
            return float(record["actual_r"])
        if record.get("shadow_r") is not None:
            return float(record["shadow_r"])
        return None
    if record.get("shadow_r") is not None:
        return float(record["shadow_r"])
    if record.get("actual_r") is not None:
        return float(record["actual_r"])
    return None


def _record_route(record: dict[str, Any]) -> str:
    return str(
        record.get("entry_route_family")
        or record.get("refinement_route")
        or record.get("entry_trigger")
        or record.get("intraday_setup_type")
        or ""
    ).upper()


def _route_matches(record: dict[str, Any], *labels: str) -> bool:
    route = _record_route(record)
    return route in {label.upper() for label in labels}


def _threshold_sweep_rows(
    live_records: list[dict[str, Any]],
    *,
    param_key: str,
    active_value: float,
    values: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed_reject_dispositions = {
        "priority_reject",
        "sector_cap_reject",
        "position_cap_reject",
        "buying_power_reject",
        "sizing_reject",
        "rescue_cap_reject",
    }
    for value in values:
        accepted_vals: list[float] = []
        rejected_vals: list[float] = []
        accepted_count = 0
        for record in live_records:
            disposition = str(record.get("disposition") or "")
            ready_idx = _safe_int(record.get("ready_bar_index"), -1)
            feasible_score = _safe_float(record.get("max_feasible_intraday_score"), np.nan)
            if not np.isfinite(feasible_score):
                feasible_score = _safe_float(record.get("intraday_score"), np.nan)
            ready_cpr = _safe_float(record.get("ready_cpr"), np.nan)
            ready_vol = _safe_float(record.get("ready_volume_ratio"), np.nan)
            accepted = disposition == "entered"
            blocked_by_capacity = bool(record.get("blocked_by_capacity_reason")) or disposition in fixed_reject_dispositions
            entry_feasible = accepted or bool(record.get("entry_window_feasible"))
            open_scored_param = param_key in {"pb_open_scored_min_score", "pb_open_scored_rank_pct_max"}
            if open_scored_param:
                can_reclass = not blocked_by_capacity and _route_matches(record, "OPEN_SCORED_ENTRY")
            elif param_key == "pb_daily_signal_min_score":
                can_reclass = not blocked_by_capacity
            else:
                can_reclass = ready_idx >= 0 and not blocked_by_capacity and entry_feasible
            relevant = True
            if param_key == "pb_entry_score_min":
                relevant = _route_matches(record, "OPENING_RECLAIM", "PM_REENTRY")
            elif param_key == "pb_delayed_confirm_score_min":
                relevant = _route_matches(record, "DELAYED_CONFIRM")
            elif open_scored_param:
                relevant = _route_matches(record, "OPEN_SCORED_ENTRY")
            elif param_key == "pb_daily_signal_min_score":
                relevant = not bool(record.get("rescue_flow_candidate"))
            elif param_key == "pb_daily_rescue_min_score":
                relevant = bool(record.get("rescue_flow_candidate"))
            elif param_key == "pb_delayed_confirm_after_bar":
                relevant = _route_matches(record, "DELAYED_CONFIRM")
            if not relevant:
                continue
            if can_reclass:
                if param_key == "pb_entry_score_min":
                    accepted = np.isfinite(feasible_score) and feasible_score >= value
                elif param_key == "pb_delayed_confirm_score_min":
                    accepted = np.isfinite(feasible_score) and feasible_score >= value
                elif param_key == "pb_open_scored_min_score":
                    accepted = _safe_float(record.get("daily_signal_score"), np.nan) >= value
                elif param_key == "pb_open_scored_rank_pct_max":
                    accepted = _safe_float(record.get("daily_signal_rank_pct"), np.nan) <= value
                elif param_key == "pb_daily_signal_min_score":
                    accepted = _safe_float(record.get("daily_signal_score"), np.nan) >= value
                elif param_key == "pb_daily_rescue_min_score":
                    accepted = _safe_float(record.get("daily_signal_score"), np.nan) >= value
                elif param_key == "pb_ready_min_cpr":
                    accepted = np.isfinite(ready_cpr) and ready_cpr >= value
                elif param_key == "pb_ready_min_volume_ratio":
                    accepted = np.isfinite(ready_vol) and ready_vol >= value
                elif param_key == "pb_delayed_confirm_after_bar":
                    accepted = ready_idx >= int(round(value))
            eff = _selector_value(record, accepted=accepted)
            if eff is None:
                continue
            if accepted:
                accepted_count += 1
                accepted_vals.append(float(eff))
            else:
                rejected_vals.append(float(eff))
        rows.append({
            "value": float(value),
            "accepted": int(accepted_count),
            "accepted_avg_r": float(np.mean(accepted_vals)) if accepted_vals else 0.0,
            "accepted_total_r": float(sum(accepted_vals)),
            "rejected_avg_shadow_r": float(np.mean(rejected_vals)) if rejected_vals else 0.0,
            "selector_delta": (float(np.mean(accepted_vals)) if accepted_vals else 0.0) - (float(np.mean(rejected_vals)) if rejected_vals else 0.0),
            "active": abs(float(value) - float(active_value)) < 1e-9,
        })
    return rows


def _compute_selector_frontier(candidate_ledger: dict[date, list[dict[str, Any]]] | None) -> dict[str, Any]:
    records = _all_candidate_records(candidate_ledger)
    live_records = [record for record in records if bool(record.get("intraday_data_available"))]
    rejected_records = [record for record in records if str(record.get("disposition") or "") != "entered"]
    open_scored_missing = [
        record
        for record in records
        if str(record.get("disposition") or "") == "entered"
        and not bool(record.get("intraday_data_available"))
        and _route_matches(record, "OPEN_SCORED_ENTRY")
    ]
    open_scored_live = [
        record
        for record in live_records
        if str(record.get("disposition") or "") == "entered" and _route_matches(record, "OPEN_SCORED_ENTRY")
    ]
    live_entered = [
        record
        for record in live_records
        if str(record.get("disposition") or "") == "entered" and not _route_matches(record, "OPEN_SCORED_ENTRY")
    ]
    live_rejected = [record for record in live_records if str(record.get("disposition") or "") != "entered"]
    cohorts = [
        {
            "label": "open-scored missing 5m",
            "count": len(open_scored_missing),
            "avg_r": _mean_or_zero([_safe_float(record.get("actual_r")) for record in open_scored_missing if record.get("actual_r") is not None]),
            "shadow_avg_r": _mean_or_zero([_safe_float(record.get("shadow_r")) for record in open_scored_missing if record.get("shadow_r") is not None]),
        },
        {
            "label": "open-scored live",
            "count": len(open_scored_live),
            "avg_r": _mean_or_zero([_safe_float(record.get("actual_r")) for record in open_scored_live if record.get("actual_r") is not None]),
            "shadow_avg_r": _mean_or_zero([_safe_float(record.get("shadow_r")) for record in open_scored_live if record.get("shadow_r") is not None]),
        },
        {
            "label": "routed 5m accepted",
            "count": len(live_entered),
            "avg_r": _mean_or_zero([_safe_float(record.get("actual_r")) for record in live_entered if record.get("actual_r") is not None]),
            "shadow_avg_r": _mean_or_zero([_safe_float(record.get("shadow_r")) for record in live_entered if record.get("shadow_r") is not None]),
        },
        {
            "label": "live 5m rejected",
            "count": len(live_rejected),
            "avg_r": _mean_or_zero([_safe_float(record.get("actual_r")) for record in live_rejected if record.get("actual_r") is not None]),
            "shadow_avg_r": _mean_or_zero([_safe_float(record.get("shadow_r")) for record in live_rejected if record.get("shadow_r") is not None]),
        },
    ]
    gate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rejected_records:
        gate = str(record.get("disposition") or record.get("intraday_invalid_reason") or "unknown")
        gate_groups[gate].append(record)
    gate_rows = []
    for gate, group in sorted(gate_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        shadows = [_safe_float(item.get("shadow_r"), np.nan) for item in group if item.get("shadow_r") is not None]
        shadows = [val for val in shadows if np.isfinite(val)]
        missed_total_r = float(sum(val for val in shadows if val > 0))
        avoided_total_r = float(abs(sum(val for val in shadows if val < 0)))
        avg_r = float(np.mean(shadows)) if shadows else 0.0
        gate_rows.append({
            "gate": gate,
            "count": len(group),
            "shadow_wr": _share(sum(1 for val in shadows if val > 0), len(shadows)),
            "shadow_avg_r": avg_r,
            "missed_total_r": missed_total_r,
            "avoided_total_r": avoided_total_r,
            "verdict": "KEEP" if avg_r <= 0 or avoided_total_r >= missed_total_r else "REVIEW",
        })
    active_entry = float(np.median([_safe_float(record.get("entry_score_threshold"), 0.0) for record in live_records])) if live_records else 0.0
    active_delayed = float(np.median([_safe_float(record.get("delayed_confirm_score_threshold"), 0.0) for record in live_records])) if live_records else 0.0
    active_cpr = float(np.median([_safe_float(record.get("ready_min_cpr_threshold"), 0.0) for record in live_records])) if live_records else 0.0
    active_vol = float(np.median([_safe_float(record.get("ready_min_volume_ratio_threshold"), 0.0) for record in live_records])) if live_records else 0.0
    active_after = float(np.median([_safe_float(record.get("delayed_confirm_after_bar_threshold"), 0.0) for record in live_records])) if live_records else 0.0
    active_daily_floor = float(np.median([_safe_float(record.get("daily_signal_min_score_threshold"), 0.0) for record in records])) if records else 0.0
    open_scored_records = [record for record in records if _route_matches(record, "OPEN_SCORED_ENTRY")]
    rescue_records = [record for record in records if bool(record.get("rescue_flow_candidate"))]
    active_open_score = float(np.median([_safe_float(record.get("open_scored_min_score_threshold"), 0.0) for record in open_scored_records])) if open_scored_records else 0.0
    active_open_rank = float(np.median([_safe_float(record.get("open_scored_rank_pct_max_threshold"), 100.0) for record in open_scored_records])) if open_scored_records else 0.0
    active_rescue_score = float(np.median([_safe_float(record.get("daily_rescue_min_score_threshold"), 0.0) for record in rescue_records])) if rescue_records else 0.0
    threshold_sweeps = {
        "pb_entry_score_min": _threshold_sweep_rows(live_records, param_key="pb_entry_score_min", active_value=active_entry, values=[max(active_entry - 10.0, 0.0), max(active_entry - 5.0, 0.0), active_entry, active_entry + 5.0, active_entry + 10.0] if live_records else []),
        "pb_delayed_confirm_score_min": _threshold_sweep_rows(live_records, param_key="pb_delayed_confirm_score_min", active_value=active_delayed, values=[max(active_delayed - 10.0, 0.0), max(active_delayed - 5.0, 0.0), active_delayed, active_delayed + 5.0, active_delayed + 10.0] if live_records else []),
        "pb_daily_signal_min_score": _threshold_sweep_rows(records, param_key="pb_daily_signal_min_score", active_value=active_daily_floor, values=[max(active_daily_floor - 10.0, 0.0), max(active_daily_floor - 5.0, 0.0), active_daily_floor, active_daily_floor + 5.0, active_daily_floor + 10.0] if records else []),
        "pb_open_scored_min_score": _threshold_sweep_rows(open_scored_records, param_key="pb_open_scored_min_score", active_value=active_open_score, values=[max(active_open_score - 10.0, 0.0), max(active_open_score - 5.0, 0.0), active_open_score, active_open_score + 5.0, active_open_score + 10.0] if open_scored_records else []),
        "pb_open_scored_rank_pct_max": _threshold_sweep_rows(open_scored_records, param_key="pb_open_scored_rank_pct_max", active_value=active_open_rank, values=[max(active_open_rank - 10.0, 0.0), max(active_open_rank - 5.0, 0.0), active_open_rank, min(active_open_rank + 5.0, 100.0), min(active_open_rank + 10.0, 100.0)] if open_scored_records else []),
        "pb_daily_rescue_min_score": _threshold_sweep_rows(rescue_records, param_key="pb_daily_rescue_min_score", active_value=active_rescue_score, values=[max(active_rescue_score - 10.0, 0.0), max(active_rescue_score - 5.0, 0.0), active_rescue_score, active_rescue_score + 5.0, active_rescue_score + 10.0] if rescue_records else []),
        "pb_ready_min_cpr": _threshold_sweep_rows(live_records, param_key="pb_ready_min_cpr", active_value=active_cpr, values=[max(active_cpr - 0.15, 0.0), max(active_cpr - 0.05, 0.0), active_cpr, min(active_cpr + 0.05, 1.0), min(active_cpr + 0.15, 1.0)] if live_records else []),
        "pb_ready_min_volume_ratio": _threshold_sweep_rows(live_records, param_key="pb_ready_min_volume_ratio", active_value=active_vol, values=[max(active_vol - 0.5, 0.0), max(active_vol - 0.25, 0.0), active_vol, active_vol + 0.25, active_vol + 0.5] if live_records else []),
        "pb_delayed_confirm_after_bar": _threshold_sweep_rows(live_records, param_key="pb_delayed_confirm_after_bar", active_value=active_after, values=[max(active_after - 2.0, 0.0), active_after, active_after + 2.0, active_after + 4.0] if live_records else []),
    }
    invalidated_rows = []
    for gate in ["daily_signal_floor_reject", "flow_reject", "flush_stale", "never_ready", "no_intraday_setup", "entry_window_expired", "intraday_priority_reserve"]:
        group = [record for record in rejected_records if str(record.get("disposition") or "") == gate]
        shadows = [_safe_float(record.get("shadow_r"), np.nan) for record in group if record.get("shadow_r") is not None]
        shadows = [val for val in shadows if np.isfinite(val)]
        invalidated_rows.append({
            "gate": gate,
            "count": len(group),
            "worked_later": sum(1 for val in shadows if val > 0),
            "worked_share": _share(sum(1 for val in shadows if val > 0), len(shadows)),
            "avg_shadow_r": float(np.mean(shadows)) if shadows else 0.0,
            "positive_shadow_avg_r": float(np.mean([val for val in shadows if val > 0])) if any(val > 0 for val in shadows) else 0.0,
        })
    daily_score_rows = _group_by_quantiles(
        records,
        lambda record: record.get("daily_signal_score", np.nan),
        lambda record: _selector_value(record, accepted=str(record.get("disposition") or "") == "entered"),
    )
    signal_model_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if bool(record.get("flow_negative")):
            if str(record.get("disposition") or "") == "flow_reject":
                label = "flow_hard_reject"
            elif bool(record.get("rescue_flow_candidate")):
                label = "flow_rescue_lane"
            else:
                label = "flow_soft_penalty"
        else:
            label = "core_signal"
        signal_model_groups[label].append(record)
    signal_model_rows = []
    for label, group in sorted(signal_model_groups.items()):
        values = [
            _selector_value(record, accepted=str(record.get("disposition") or "") == "entered")
            for record in group
        ]
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        signal_model_rows.append({
            "label": label,
            "count": len(group),
            "entered": sum(1 for record in group if str(record.get("disposition") or "") == "entered"),
            "avg_effective_r": float(np.mean(values)) if values else 0.0,
        })
    return {
        "cohorts": cohorts,
        "gate_rows": gate_rows,
        "threshold_sweeps": threshold_sweeps,
        "invalidated_rows": invalidated_rows,
        "daily_score_rows": daily_score_rows,
        "signal_model_rows": signal_model_rows,
    }


def _compute_capacity_opportunity(
    trades: list[TradeRecord],
    candidate_ledger: dict[date, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    records = _all_candidate_records(candidate_ledger)
    blocked_rows = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        reason = _blocked_capacity_reason(record)
        if reason:
            groups[reason].append(record)
    for reason, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        blocked_shadows = [_safe_float(item.get("shadow_r"), np.nan) for item in group if item.get("shadow_r") is not None]
        blocked_shadows = [val for val in blocked_shadows if np.isfinite(val)]
        occupant_vals: list[float] = []
        for record in group:
            trade_date = _record_trade_date(record)
            if trade_date is None:
                continue
            for peer in records:
                if peer is record or _record_trade_date(peer) != trade_date or str(peer.get("disposition") or "") != "entered":
                    continue
                if reason == "sector_cap" and str(peer.get("sector") or "") != str(record.get("sector") or ""):
                    continue
                if peer.get("actual_r") is not None:
                    occupant_vals.append(float(peer["actual_r"]))
        blocked_rows.append({
            "reason": reason,
            "count": len(group),
            "blocked_shadow_avg_r": float(np.mean(blocked_shadows)) if blocked_shadows else 0.0,
            "blocked_shadow_wr": _share(sum(1 for val in blocked_shadows if val > 0), len(blocked_shadows)),
            "occupant_avg_r": float(np.mean(occupant_vals)) if occupant_vals else 0.0,
            "occupant_count": len(occupant_vals),
            "verdict": "REVIEW" if blocked_shadows and float(np.mean(blocked_shadows)) > float(np.mean(occupant_vals)) else "KEEP",
        })
    replacement_rows = []
    for reason in ["QUICK_EXIT", "STALE_EXIT", "VWAP_FAIL", "FLOW_REVERSAL"]:
        group = [trade for trade in trades if (trade.exit_reason or "UNKNOWN") == reason]
        if reason == "FLOW_REVERSAL":
            group = [trade for trade in group if _bars_to_exit(trade) <= 24]
        replacement_shadow: list[float] = []
        replacement_hits = 0
        actual_reuse_rs: list[float] = []
        for trade in group:
            day_records = [
                record for record in records
                if _record_trade_date(record) == trade.entry_time.date()
                and record.get("symbol") != trade.symbol
            ]
            later = [
                record for record in day_records
                if _record_timestamp(record, "ready_timestamp") is not None
                and _record_timestamp(record, "ready_timestamp") > trade.exit_time
                and _blocked_capacity_reason(record) in {"slot_cap", "sector_cap", "buying_power", "intraday_priority_reserve"}
                and record.get("shadow_r") is not None
            ]
            if later:
                replacement_hits += 1
                replacement_shadow.extend(float(record["shadow_r"]) for record in later)
            actual_replacements = [
                peer for peer in trades
                if peer.symbol != trade.symbol
                and peer.entry_time.date() == trade.entry_time.date()
                and peer.entry_time > trade.exit_time
            ]
            actual_reuse_rs.extend(float(peer.r_multiple) for peer in actual_replacements)
        replacement_rows.append({
            "reason": reason,
            "exit_count": len(group),
            "replacement_windows": replacement_hits,
            "replacement_count": len(replacement_shadow),
            "replacement_avg_shadow_r": float(np.mean(replacement_shadow)) if replacement_shadow else 0.0,
            "actual_reuse_count": len(actual_reuse_rs),
            "actual_reuse_avg_r": float(np.mean(actual_reuse_rs)) if actual_reuse_rs else 0.0,
            "exit_avg_r": float(np.mean([float(trade.r_multiple) for trade in group])) if group else 0.0,
            "net_delta": (float(np.mean(replacement_shadow)) if replacement_shadow else 0.0) - (float(np.mean([float(trade.r_multiple) for trade in group])) if group else 0.0),
        })
    return {
        "blocked_rows": blocked_rows,
        "replacement_rows": replacement_rows,
        "replacement_verdict": (
            "Replacement-ready candidates did appear after some early exits."
            if any(row["replacement_count"] > 0 for row in replacement_rows)
            else (
                "No replacement-ready candidates appeared after early exits, but later same-day capital reuse did occur."
                if any(row["actual_reuse_count"] > 0 for row in replacement_rows)
                else "No replacement-ready candidates appeared after early exits."
            )
        ),
    }


def _simulate_intraday_trade_policy(trade: TradeRecord, replay, *, mode: str) -> dict[str, Any] | None:
    if replay is None or trade.risk_per_share <= 0:
        return None
    arrs = getattr(replay, "get_5m_arrays_for_date", None)
    if arrs is None:
        return None
    arrs = replay.get_5m_arrays_for_date(trade.symbol, trade.entry_time.date())
    if arrs is None:
        return None
    lows = arrs["low"]
    closes = arrs["close"]
    n = len(lows)
    if n == 0:
        return None
    entry_idx = 0  # no end_time available in array mode
    entry_price = float(trade.entry_price)
    risk = float(trade.risk_per_share)
    base_stop = entry_price - risk
    exit_price = float(closes[-1])
    exit_reason = "EOD_FLATTEN"
    for idx in range(entry_idx, n):
        stop_price = base_stop
        if mode == "tight_first_hour" and idx - entry_idx < 12:
            stop_price = max(stop_price, entry_price - 0.5 * risk)
        if lows[idx] <= stop_price:
            exit_price = stop_price
            exit_reason = "STOP_HIT"
            break
        exit_price = float(closes[idx])
    return {"r": float((exit_price - entry_price) / risk), "exit_reason": exit_reason}


def _compute_entry_quality(trades: list[TradeRecord], replay) -> dict[str, Any]:
    route_frontier_rows: list[dict[str, Any]] = []
    entry_tax_rows: list[dict[str, Any]] = []
    route_bucket_rows: list[dict[str, Any]] = []
    short_loser_rows: list[dict[str, Any]] = []
    score_bucket_rows: list[dict[str, Any]] = []
    score_cross_rows: list[dict[str, Any]] = []
    route_monotonicity_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    routes = _ordered_route_labels(sorted({_route_label(trade) for trade in trades}))
    by_route = {route: [trade for trade in trades if _route_label(trade) == route] for route in routes}
    for route, group in by_route.items():
        if not group:
            continue
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        price_taxes: list[float] = []
        best_r_deltas: list[float] = []
        for trade in group:
            actual = {
                "label": "Actual",
                "r": float(trade.r_multiple),
                "entry_price": float(trade.entry_price),
                "entry_location": _safe_float(_meta(trade, "entry_location", 0.0)),
            }
            by_label["Actual"].append(actual)
            results = [result for result in [
                _entry_variant_result(trade, replay, "open"),
                _entry_variant_result(trade, replay, "delay_30m"),
                _entry_variant_result(trade, replay, "delay_60m"),
                _entry_variant_result(trade, replay, "first_reversal_close"),
                _best_feasible_entry_result(trade, replay),
            ] if result is not None]
            for result in results:
                by_label[str(result["label"])].append(result)
            if results:
                best = max(results, key=lambda item: float(item["r"]))
                price_taxes.append(float(trade.entry_price) - float(best["entry_price"]))
                best_r_deltas.append(float(best["r"]) - float(trade.r_multiple))
        for label, rows in sorted(by_label.items(), key=lambda item: ("Actual" not in item[0], item[0])):
            rs = [float(row["r"]) for row in rows]
            route_frontier_rows.append({
                "route": route,
                "label": label,
                "n": len(rows),
                "avg_r": float(np.mean(rs)) if rs else 0.0,
                "pf": float(_pf(rs)),
                "wr": _share(sum(1 for val in rs if val > 0), len(rs)),
                "entry_location": float(np.mean([float(row.get("entry_location", 0.0)) for row in rows])) if rows else 0.0,
            })
        if price_taxes:
            entry_tax_rows.append({
                "route": route,
                "count": len(price_taxes),
                "avg_price_tax": float(np.mean(price_taxes)),
                "avg_r_tax": float(np.mean(best_r_deltas)),
            })
        bucket_groups: dict[str, list[TradeRecord]] = defaultdict(list)
        for trade in group:
            bucket_groups[_half_hour_bucket(trade.entry_time)].append(trade)
        for bucket, bucket_group in sorted(bucket_groups.items()):
            route_bucket_rows.append({
                "route": route,
                "bucket": bucket,
                "n": len(bucket_group),
                "avg_r": float(np.mean([float(trade.r_multiple) for trade in bucket_group])),
            })
    short_losers = [
        trade for trade in trades
        if float(trade.r_multiple) < 0 and _bars_to_exit(trade) <= 12
    ]
    bucket_labels = [(0, 2, "0-2 bars"), (3, 6, "3-6 bars"), (7, 12, "7-12 bars")]
    for trade in sorted(short_losers, key=lambda item: float(item.r_multiple))[:15]:
        bucket = next((label for lo, hi, label in bucket_labels if lo <= _bars_to_exit(trade) <= hi), ">12 bars")
        short_loser_rows.append({
            "bucket": bucket,
            "symbol": trade.symbol,
            "route": _route_label(trade),
            "entry_hour": _half_hour_bucket(trade.entry_time),
            "intraday_score": _safe_float(_meta(trade, "intraday_score", 0.0)),
            "reclaim_bars": _safe_int(_meta(trade, "reclaim_bars", 0)),
            "micropressure": str(_meta(trade, "micropressure_signal", "N/A") or "N/A"),
            "mfe_before_loss_r": _safe_float(_meta(trade, "mfe_before_negative_exit_r", _meta(trade, "mfe_r", 0.0))),
            "r": float(trade.r_multiple),
        })
    scored_trades = [trade for trade in trades if np.isfinite(_trade_route_score(trade))]
    if scored_trades:
        score_bucket_rows = _group_by_quantiles(scored_trades, _trade_route_score, lambda trade: float(trade.r_multiple))
        for route in _ordered_route_labels(sorted({_route_label(trade) for trade in scored_trades})):
            group = [trade for trade in scored_trades if _route_label(trade) == route]
            if not group:
                continue
            quant_rows = _group_by_quantiles(group, _trade_route_score, lambda trade: float(trade.r_multiple), bins=4)
            if quant_rows:
                score_cross_rows.extend(
                    {
                        "route": route,
                        "label": row["label"],
                        "n": row["n"],
                        "avg_r": row["avg_r"],
                    }
                    for row in quant_rows
                )
                monotonic_pairs = 0
                comparisons = 0
                for prev, cur in zip(quant_rows, quant_rows[1:]):
                    comparisons += 1
                    if float(cur["avg_r"]) >= float(prev["avg_r"]) - 1e-9:
                        monotonic_pairs += 1
                route_monotonicity_rows.append({
                    "route": route,
                    "bucket_count": len(quant_rows),
                    "monotonic_share": _share(monotonic_pairs, comparisons),
                    "best_bucket_avg_r": max(float(row["avg_r"]) for row in quant_rows),
                    "worst_bucket_avg_r": min(float(row["avg_r"]) for row in quant_rows),
                })
    live_scored = [
        trade for trade in trades
        if _route_label(trade) in {"OPENING_RECLAIM", "DELAYED_CONFIRM", "PM_REENTRY"}
        and _meta(trade, "intraday_score") is not None
    ]
    if live_scored:
        component_names = [
            "daily_signal",
            "reclaim",
            "volume",
            "vwap_hold",
            "cpr",
            "speed",
            "context_adjust",
            "micro_penalty",
            "weak_vwap_penalty",
            "rescue_penalty",
        ]
        winners = [trade for trade in live_scored if trade.is_winner]
        losers = [trade for trade in live_scored if not trade.is_winner]
        for name in component_names:
            component_rows.append({
                "component": name,
                "winner_mean": float(np.mean([_safe_float(_meta(trade, f"entry_score_component_{name}", 0.0)) for trade in winners])) if winners else 0.0,
                "loser_mean": float(np.mean([_safe_float(_meta(trade, f"entry_score_component_{name}", 0.0)) for trade in losers])) if losers else 0.0,
                "delta": (float(np.mean([_safe_float(_meta(trade, f"entry_score_component_{name}", 0.0)) for trade in winners])) if winners else 0.0) - (float(np.mean([_safe_float(_meta(trade, f"entry_score_component_{name}", 0.0)) for trade in losers])) if losers else 0.0),
            })
    return {
        "route_frontier_rows": route_frontier_rows,
        "entry_tax_rows": entry_tax_rows,
        "route_bucket_rows": route_bucket_rows,
        "short_loser_rows": short_loser_rows,
        "score_bucket_rows": score_bucket_rows,
        "score_cross_rows": score_cross_rows,
        "route_monotonicity_rows": route_monotonicity_rows,
        "component_rows": component_rows,
    }


def _compute_management_forensics(trades: list[TradeRecord], replay) -> dict[str, Any]:
    hold_rows = []
    giveback_rows = []
    feature_rows = []
    short_loser_rows = []
    stale_counterfactual = {"count": 0, "hold_to_eod_avg_r": 0.0, "tight_first_hour_avg_r": 0.0}
    hold_buckets = [
        ("0-2 bars", lambda trade: _bars_to_exit(trade) <= 2),
        ("3-6 bars", lambda trade: 3 <= _bars_to_exit(trade) <= 6),
        ("7-12 bars", lambda trade: 7 <= _bars_to_exit(trade) <= 12),
        ("13-24 bars", lambda trade: 13 <= _bars_to_exit(trade) <= 24),
        (">24 bars / overnight", lambda trade: _bars_to_exit(trade) > 24 or _safe_int(_meta(trade, "hold_days", 1), 1) > 1),
    ]
    for label, predicate in hold_buckets:
        group = [trade for trade in trades if predicate(trade)]
        if not group:
            continue
        hold_rows.append({
            "label": label,
            "count": len(group),
            "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])),
            "avg_hours": float(np.mean([trade.hold_hours for trade in group])),
        })
    short_losers = [trade for trade in trades if float(trade.r_multiple) < 0 and _bars_to_exit(trade) <= 12]
    if short_losers:
        mfe_vals = [_safe_float(_meta(trade, "mfe_before_negative_exit_r", _meta(trade, "mfe_r", 0.0))) for trade in short_losers]
        short_loser_rows.append({
            "count": len(short_losers),
            "avg_mfe_before_negative_exit_r": float(np.mean(mfe_vals)),
            "mfe_gt_0p1": _share(sum(1 for val in mfe_vals if val > 0.1), len(mfe_vals)),
            "mfe_gt_0p25": _share(sum(1 for val in mfe_vals if val > 0.25), len(mfe_vals)),
            "mfe_gt_0p5": _share(sum(1 for val in mfe_vals if val > 0.5), len(mfe_vals)),
        })
    for feature, label in [("partial_taken", "partial"), ("trail_active", "trail"), ("breakeven_activated", "breakeven")]:
        yes_group = [trade for trade in trades if bool(_meta(trade, feature, False))]
        no_group = [trade for trade in trades if not bool(_meta(trade, feature, False))]
        for bucket_label, group in [("yes", yes_group), ("no", no_group)]:
            if not group:
                continue
            feature_rows.append({
                "feature": label,
                "bucket": bucket_label,
                "count": len(group),
                "wr": _share(sum(1 for trade in group if trade.is_winner), len(group)),
                "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])),
                "mean_mfe": float(np.mean([_safe_float(_meta(trade, "mfe_r", 0.0)) for trade in group])),
                "mean_giveback": float(np.mean([max(_safe_float(_meta(trade, "mfe_r", 0.0)) - float(trade.r_multiple), 0.0) for trade in group])),
            })
    by_reason_route: dict[tuple[str, str], list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        by_reason_route[(trade.exit_reason or "UNKNOWN", _route_label(trade))].append(trade)
    for (reason, route), group in sorted(by_reason_route.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1]))[:12]:
        giveback_rows.append({
            "reason": reason,
            "route": route,
            "count": len(group),
            "avg_giveback": float(np.mean([max(_safe_float(_meta(trade, "mfe_r", 0.0)) - float(trade.r_multiple), 0.0) for trade in group])),
        })
    stale_trades = [trade for trade in trades if (trade.exit_reason or "UNKNOWN") == "STALE_EXIT"]
    if stale_trades:
        eod_vals = []
        tight_vals = []
        for trade in stale_trades:
            eod = _simulate_intraday_trade_policy(trade, replay, mode="hold_to_eod")
            tight = _simulate_intraday_trade_policy(trade, replay, mode="tight_first_hour")
            if eod is not None:
                eod_vals.append(float(eod["r"]))
            if tight is not None:
                tight_vals.append(float(tight["r"]))
        stale_counterfactual = {
            "count": len(stale_trades),
            "hold_to_eod_avg_r": float(np.mean(eod_vals)) if eod_vals else 0.0,
            "tight_first_hour_avg_r": float(np.mean(tight_vals)) if tight_vals else 0.0,
        }
    protection_candidates = [trade for trade in trades if float(trade.r_multiple) < 0 and _safe_float(_meta(trade, "mfe_before_negative_exit_r", _meta(trade, "mfe_r", 0.0))) > 0.25]
    return {
        "hold_rows": hold_rows,
        "short_loser_rows": short_loser_rows,
        "feature_rows": feature_rows,
        "giveback_rows": giveback_rows,
        "stale_counterfactual": stale_counterfactual,
        "protection_summary": {
            "count": len(protection_candidates),
            "share": _share(len(protection_candidates), len(trades)),
            "avg_loser_r": float(np.mean([float(trade.r_multiple) for trade in protection_candidates])) if protection_candidates else 0.0,
        },
    }


def _group_contribution_rows(
    trades: list[TradeRecord],
    label_fn,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        groups[str(label_fn(trade) or "UNKNOWN")].append(trade)
    rows = []
    for label, group in groups.items():
        rows.append({
            "label": label,
            "count": len(group),
            "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])),
            "total_r": float(sum(float(trade.r_multiple) for trade in group)),
        })
    return sorted(rows, key=lambda item: (-item["total_r"], -item["avg_r"], item["label"]))[:limit]


def _compute_exit_replacement(trades: list[TradeRecord], replay) -> dict[str, Any]:
    route_rows = []
    reason_rows = []
    carry_audit = {
        "binary_eligible": 0,
        "score_fallback_eligible": 0,
        "actually_carried": 0,
        "should_have_carried_but_flattened": 0,
        "carried_underperformed_flatten": 0,
    }
    alpha_curve_rows = []
    flow_rows = []
    variants = [
        ("EOD flatten", dict(hold_days=0, carry_min_r=99.0)),
        ("Carry 1d", dict(hold_days=1, carry_min_r=0.0)),
        ("Carry 3d", dict(hold_days=3, carry_min_r=0.10)),
        ("Carry 5d", dict(hold_days=5, carry_min_r=0.10)),
        ("Carry 3d + close>=0.65", dict(hold_days=3, carry_min_r=0.10, close_pct_min=0.65)),
        ("Carry 3d + close stop", dict(hold_days=3, carry_min_r=0.10, close_stop=True)),
        ("Carry 3d + flowrev 2", dict(hold_days=3, carry_min_r=0.10, flow_lookback=2)),
    ]
    if replay is not None:
        by_route: dict[str, list[TradeRecord]] = defaultdict(list)
        by_reason: dict[str, list[TradeRecord]] = defaultdict(list)
        for trade in trades:
            by_route[_route_label(trade)].append(trade)
            by_reason[trade.exit_reason or "UNKNOWN"].append(trade)
        for route, group in sorted(by_route.items()):
            sims = [(label, [_simulate_daily_policy(trade, replay, **params) for trade in group]) for label, params in variants]
            best_label = "Actual"
            best_avg = float(np.mean([float(trade.r_multiple) for trade in group])) if group else 0.0
            for label, sim_values in sims:
                sim_values = [item for item in sim_values if item is not None]
                if not sim_values:
                    continue
                avg_r = float(np.mean([float(item["r"]) for item in sim_values]))
                if avg_r > best_avg:
                    best_avg = avg_r
                    best_label = label
            route_rows.append({
                "route": route,
                "count": len(group),
                "actual_avg_r": float(np.mean([float(trade.r_multiple) for trade in group])) if group else 0.0,
                "best_variant": best_label,
                "best_avg_r": best_avg,
            })
        for reason, group in sorted(by_reason.items()):
            row = {
                "reason": reason,
                "count": len(group),
                "actual_avg_r": float(np.mean([float(trade.r_multiple) for trade in group])) if group else 0.0,
            }
            for label, params in variants:
                sims = [_simulate_daily_policy(trade, replay, **params) for trade in group]
                sims = [item for item in sims if item is not None]
                row[label] = float(np.mean([float(item["r"]) for item in sims])) if sims else 0.0
            reason_rows.append(row)
    for trade in trades:
        binary_ok = bool(_meta(trade, "carry_binary_ok", False))
        score_ok = bool(_meta(trade, "carry_score_ok", False))
        decision_path = str(_meta(trade, "carry_decision_path", "") or "")
        carried = decision_path in {"binary", "score_fallback"} or _safe_int(_meta(trade, "hold_days", 1), 1) > 1
        flattened_eligible = (trade.exit_reason or "UNKNOWN") == "EOD_FLATTEN" and (binary_ok or score_ok)
        if binary_ok:
            carry_audit["binary_eligible"] += 1
        if score_ok and not binary_ok:
            carry_audit["score_fallback_eligible"] += 1
        if carried:
            carry_audit["actually_carried"] += 1
        if flattened_eligible and not carried:
            carry_audit["should_have_carried_but_flattened"] += 1
        if replay is not None and carried:
            flatten_sim = _simulate_daily_policy(trade, replay, hold_days=0, carry_min_r=99.0)
            if flatten_sim is not None and float(flatten_sim["r"]) > float(trade.r_multiple):
                carry_audit["carried_underperformed_flatten"] += 1
    flow_group = [trade for trade in trades if (trade.exit_reason or "UNKNOWN") == "FLOW_REVERSAL"]
    for label, predicate in [
        ("early", lambda trade: _safe_int(_meta(trade, "hold_days", 1), 1) <= 2),
        ("mid", lambda trade: 3 <= _safe_int(_meta(trade, "hold_days", 1), 1) <= 4),
        ("late", lambda trade: _safe_int(_meta(trade, "hold_days", 1), 1) >= 5),
    ]:
        group = [trade for trade in flow_group if predicate(trade)]
        if not group:
            continue
        flow_rows.append({
            "label": label,
            "count": len(group),
            "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])),
            "total_r": float(sum(float(trade.r_multiple) for trade in group)),
            "routes": ", ".join(sorted({_route_label(trade) for trade in group})),
        })
    for label, predicate in [
        ("intraday-only", lambda trade: _safe_int(_meta(trade, "hold_days", 1), 1) <= 1),
        ("overnight", lambda trade: _safe_int(_meta(trade, "hold_days", 1), 1) > 1),
    ]:
        group = [trade for trade in trades if predicate(trade)]
        if not group:
            continue
        alpha_curve_rows.append({
            "label": label,
            "count": len(group),
            "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])),
            "total_r": float(sum(float(trade.r_multiple) for trade in group)),
        })
    for route, group in sorted(defaultdict(list, {route: [trade for trade in trades if _route_label(trade) == route] for route in sorted({_route_label(trade) for trade in trades})}).items()):
        if not group:
            continue
        alpha_curve_rows.append({
            "label": f"route:{route}",
            "count": len(group),
            "avg_r": float(np.mean([float(trade.r_multiple) for trade in group])),
            "total_r": float(sum(float(trade.r_multiple) for trade in group)),
        })
    return {
        "route_rows": route_rows,
        "reason_rows": reason_rows,
        "carry_audit": carry_audit,
        "flow_rows": flow_rows,
        "alpha_curve_rows": alpha_curve_rows,
    }


def _compute_alpha_sources(
    trades: list[TradeRecord],
    selector_frontier: dict[str, Any],
) -> dict[str, Any]:
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    feature_bucket_rows = {
        "entry_gap_pct": _group_contribution_rows(trades, lambda trade: "<-1%" if _safe_float(_meta(trade, "entry_gap_pct", 0.0)) < -1.0 else "-1% to 0%" if _safe_float(_meta(trade, "entry_gap_pct", 0.0)) < 0.0 else "0% to 1%" if _safe_float(_meta(trade, "entry_gap_pct", 0.0)) < 1.0 else ">=1%"),
        "entry_cdd": _group_contribution_rows(trades, lambda trade: "0-1" if _safe_int(_meta(trade, "entry_cdd", 0)) <= 1 else "2-3" if _safe_int(_meta(trade, "entry_cdd", 0)) <= 3 else "4-5" if _safe_int(_meta(trade, "entry_cdd", 0)) <= 5 else "6+"),
        "entry_rank_pct": _group_contribution_rows(trades, lambda trade: "0-20%" if _safe_float(_meta(trade, "entry_rank_pct", 100.0)) <= 20.0 else "20-35%" if _safe_float(_meta(trade, "entry_rank_pct", 100.0)) <= 35.0 else "35-50%" if _safe_float(_meta(trade, "entry_rank_pct", 100.0)) <= 50.0 else ">50%"),
        "daily_signal_score": _group_contribution_rows(trades, lambda trade: "<50" if _safe_float(_meta(trade, "daily_signal_score", 0.0)) < 50.0 else "50-60" if _safe_float(_meta(trade, "daily_signal_score", 0.0)) < 60.0 else "60-70" if _safe_float(_meta(trade, "daily_signal_score", 0.0)) < 70.0 else "70+"),
        "daily_signal_rank_pct": _group_contribution_rows(trades, lambda trade: "0-20%" if _safe_float(_meta(trade, "daily_signal_rank_pct", 100.0)) <= 20.0 else "20-35%" if _safe_float(_meta(trade, "daily_signal_rank_pct", 100.0)) <= 35.0 else "35-50%" if _safe_float(_meta(trade, "daily_signal_rank_pct", 100.0)) <= 50.0 else ">50%"),
        "close_pct": _group_contribution_rows(trades, lambda trade: "<0.40" if _safe_float(_meta(trade, "close_pct", 0.0)) < 0.40 else "0.40-0.65" if _safe_float(_meta(trade, "close_pct", 0.0)) < 0.65 else "0.65-0.80" if _safe_float(_meta(trade, "close_pct", 0.0)) < 0.80 else "0.80+"),
        "mfe_r": _group_contribution_rows(trades, lambda trade: "<0.25R" if _safe_float(_meta(trade, "mfe_r", 0.0)) < 0.25 else "0.25-0.50R" if _safe_float(_meta(trade, "mfe_r", 0.0)) < 0.50 else "0.50-1.00R" if _safe_float(_meta(trade, "mfe_r", 0.0)) < 1.00 else "1.00R+"),
    }
    gate_rows = [row for row in selector_frontier.get("gate_rows", []) if row.get("verdict") == "KEEP"]
    best_routes = [row for row in _group_contribution_rows(trades, _route_label) if row["avg_r"] > 0]
    return {
        "route_rows": _group_contribution_rows(trades, _route_label),
        "sector_rows": _group_contribution_rows(trades, lambda trade: trade.sector or "UNKNOWN"),
        "weekday_rows": _group_contribution_rows(trades, lambda trade: weekday_labels[trade.entry_time.weekday()] if 0 <= trade.entry_time.weekday() < len(weekday_labels) else "N/A"),
        "hour_rows": _group_contribution_rows(trades, lambda trade: _half_hour_bucket(trade.entry_time)),
        "feature_bucket_rows": feature_bucket_rows,
        "best_keepers": {
            "gates": gate_rows[:5],
            "routes": best_routes[:5],
        },
    }

def compute_pullback_diagnostic_snapshot(
    trades: list[TradeRecord],
    *,
    metrics: dict[str, float] | None = None,
    replay=None,
    daily_selections: dict | None = None,
    candidate_ledger: dict[date, list[dict[str, Any]]] | None = None,
    funnel_counters: dict[str, int] | None = None,
    rejection_log: list[dict[str, Any]] | None = None,
    shadow_outcomes: list[dict[str, Any]] | None = None,
    selection_attribution: dict[date, dict[str, Any]] | None = None,
    fsm_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    del daily_selections, shadow_outcomes
    actual = _trade_stats(trades)
    sharpe_proxy = 0.0
    if len(trades) > 1:
        rs = np.array([float(trade.r_multiple) for trade in trades], dtype=float)
        std = float(np.std(rs))
        if std > 0:
            sharpe_proxy = float(np.mean(rs) / std * np.sqrt(252))
    overview = {**actual, "sharpe": _safe_float(metrics.get("sharpe")) if metrics else sharpe_proxy, "sharpe_label": "Sharpe (backtest)" if metrics else "Sharpe (trade proxy)", "max_drawdown_pct": _safe_float(metrics.get("max_drawdown_pct")) if metrics else 0.0}
    funnel = _compute_funnel(trades, funnel_counters, rejection_log)
    shadow = _compute_shadow_summary(rejection_log, trades)
    selection = _compute_selection_summary(selection_attribution)
    monotonicity = _compute_monotonicity(trades)
    interactions = {
        "rank_pct_x_sma_dist": _interaction_table(trades, "entry_rank_pct", "entry_sma_dist_pct"),
        "cdd_x_rsi": _interaction_table(trades, "entry_cdd", "entry_rsi"),
        "gap_x_close_pct": _interaction_table(trades, "entry_gap_pct", "close_pct"),
        "close_pct_x_mfe_r": _interaction_table(trades, "close_pct", "mfe_r"),
    }
    mfe_capture = _compute_mfe_capture(trades)
    exit_frontier = _compute_exit_frontier(trades, replay)
    carry_funnel = _compute_carry_funnel(trades, replay)
    entry_timing = _compute_entry_timing(trades, replay)
    low_trade_days = _compute_low_trade_days(candidate_ledger)
    concentration = _compute_concentration(trades)
    intraday = _compute_intraday_summary(trades, candidate_ledger, fsm_log)
    selector_frontier = _compute_selector_frontier(candidate_ledger)
    capacity_opportunity = _compute_capacity_opportunity(trades, candidate_ledger)
    entry_quality = _compute_entry_quality(trades, replay)
    management_forensics = _compute_management_forensics(trades, replay)
    exit_replacement = _compute_exit_replacement(trades, replay)
    alpha_sources = _compute_alpha_sources(trades, selector_frontier)
    best_timing = max(entry_timing, key=lambda item: item["avg_r"], default=None)
    best_exit = max(exit_frontier, key=lambda item: item["avg_r"], default=None)
    daily_score_rows = selector_frontier.get("daily_score_rows", [])
    daily_score_monotonic = bool(daily_score_rows) and float(daily_score_rows[-1]["avg_r"]) >= float(daily_score_rows[0]["avg_r"])
    route_monotonicity_rows = entry_quality.get("route_monotonicity_rows", [])
    route_monotonicity_avg = float(np.mean([float(row["monotonic_share"]) for row in route_monotonicity_rows])) if route_monotonicity_rows else 0.0
    verdicts = {
        "signal_extraction": "GOOD" if shadow["delta_avg_r"] > 0.03 and daily_score_monotonic else "REVIEW",
        "signal_discrimination": "GOOD" if selection["skipped_avg_shadow_r"] <= selection["entered_avg_r"] and shadow["shadow"]["avg_r"] <= actual["avg_r"] else "REVIEW",
        "entry_mechanism": "GOOD" if route_monotonicity_avg >= 0.50 and (best_timing is None or best_timing["label"] == "Open" or best_timing["avg_r"] <= actual["avg_r"] + 0.02) else "REVIEW",
        "trade_management": "GOOD" if metrics and _safe_float(metrics.get("managed_exit_share")) >= 0.50 and _safe_float(metrics.get("eod_flatten_share")) <= 0.50 else "REVIEW",
        "exit_mechanism": "GOOD" if best_exit is None or best_exit["label"] == "Actual" or best_exit["avg_r"] <= actual["avg_r"] + 0.02 else "REVIEW",
        "primary_bottleneck": (
            "EOD flatten / carry conversion"
            if metrics and _safe_float(metrics.get("eod_flatten_share")) > 0.50
            else "route score calibration"
            if route_monotonicity_rows and route_monotonicity_avg < 0.50
            else "selection discrimination"
        ),
    }
    return {
        "overview": overview,
        "verdicts": verdicts,
        "funnel": funnel,
        "shadow": shadow,
        "selection": selection,
        "monotonicity": monotonicity,
        "interactions": interactions,
        "mfe_capture": mfe_capture,
        "exit_frontier": exit_frontier,
        "carry_funnel": carry_funnel,
        "entry_timing": entry_timing,
        "low_trade_days": low_trade_days,
        "concentration": concentration,
        "intraday": intraday,
        "selector_frontier": selector_frontier,
        "capacity_opportunity": capacity_opportunity,
        "entry_quality": entry_quality,
        "management_forensics": management_forensics,
        "exit_replacement": exit_replacement,
        "alpha_sources": alpha_sources,
    }


def _render_overview(snapshot: dict[str, Any]) -> str:
    overview = snapshot["overview"]
    lines = [_hdr("1. Overview")]
    lines.append(f"  Trades: {int(overview['n'])}")
    lines.append(f"  Win Rate: {overview['wr']:.1%}")
    lines.append(f"  Mean R: {overview['avg_r']:+.3f}  |  Median R: {overview['median_r']:+.3f}")
    lines.append(f"  Total R: {overview['total_r']:+.2f}  |  Total PnL: ${overview['pnl']:,.2f}")
    lines.append(f"  Profit Factor: {overview['pf']:.2f}  |  {overview['sharpe_label']}: {overview['sharpe']:.2f}")
    if overview["max_drawdown_pct"] > 0:
        lines.append(f"  Max Drawdown: {overview['max_drawdown_pct']:.1%}")
    return "\n".join(lines)


def _render_verdicts(snapshot: dict[str, Any]) -> str:
    verdicts = snapshot["verdicts"]
    selection = snapshot["selection"]
    shadow = snapshot["shadow"]
    timing_best = max(snapshot["entry_timing"], key=lambda item: item["avg_r"], default=None)
    exit_best = max(snapshot["exit_frontier"], key=lambda item: item["avg_r"], default=None)
    lines = [_hdr("Executive Verdicts")]
    lines.append(f"  Signal extraction: {verdicts['signal_extraction']} (accepted avg_r {shadow['actual']['avg_r']:+.3f} vs rejected shadow {shadow['shadow']['avg_r']:+.3f}).")
    lines.append(f"  Discrimination: {verdicts['signal_discrimination']} (crowded-day entered avg_r {selection['entered_avg_r']:+.3f} vs skipped shadow {selection['skipped_avg_shadow_r']:+.3f}).")
    if timing_best is not None:
        lines.append(f"  Entry mechanism: {verdicts['entry_mechanism']} (best timing variant {timing_best['label']} @ {timing_best['avg_r']:+.3f}R).")
    else:
        lines.append("  Entry mechanism: REVIEW (no 5m replay available for timing comparison).")
    if exit_best is not None:
        lines.append(f"  Exit mechanism: {verdicts['exit_mechanism']} (best frontier variant {exit_best['label']} @ {exit_best['avg_r']:+.3f}R).")
    else:
        lines.append("  Exit mechanism: REVIEW (no replay available for economic frontier).")
    lines.append(f"  Trade management: {verdicts['trade_management']}.")
    lines.append(f"  Primary bottleneck: {verdicts['primary_bottleneck']}.")
    return "\n".join(lines)


def _render_exit_mix(trades: list[TradeRecord]) -> str:
    by_reason: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        by_reason[trade.exit_reason or "UNKNOWN"].append(trade)
    lines = [_hdr("2. Exit Reason Decomposition")]
    lines.append(f"  {'Reason':<22s} {'Count':>6s} {'WR':>6s} {'AvgR':>8s} {'TotR':>8s}")
    lines.append("  " + "-" * 56)
    for reason, group in sorted(by_reason.items(), key=lambda item: -len(item[1])):
        stats = _trade_stats(group)
        lines.append(f"  {reason:<22s} {int(stats['n']):>6d} {stats['wr']:>5.1%} {stats['avg_r']:>+7.3f} {stats['total_r']:>+7.2f}")
    return "\n".join(lines)


def _render_funnel(snapshot: dict[str, Any]) -> str:
    funnel = snapshot["funnel"]
    lines = [_hdr("3. Signal Funnel & Gate Attribution")]
    if not funnel["counters"]:
        lines.append("  (no candidate funnel data -- rerun with upgraded engine diagnostics)")
        return "\n".join(lines)
    counters = funnel["counters"]
    lines.append(f"  Universe seen: {counters.get('universe_seen', 0)}")
    lines.append(f"  Triggered: {counters.get('triggered', 0)}")
    lines.append(f"  Candidate pool: {counters.get('candidate_pool', 0)}")
    pool = int(funnel.get("candidate_pool", 0))
    entered = int(funnel.get("entered", 0))
    if pool > 0 and entered <= pool:
        lines.append(f"  Entered: {entered} ({funnel['accept_rate']:.1%} of candidate pool)")
    else:
        lines.append(
            f"  Entered: {entered} (mixed cohorts; headline accept rate omitted, see 3C for cohort splits)"
        )
    lines.append("")
    lines.append(f"  {'Gate':<24s} {'Count':>7s} {'AvgShadowR':>12s} {'FP Rate':>9s} {'Verdict':>8s}")
    lines.append("  " + "-" * 68)
    for row in funnel["gate_rows"]:
        lines.append(f"  {row['gate']:<24s} {row['count']:>7d} {row['avg_r']:>+11.3f} {row['false_positive']:>8.1%} {row['verdict']:>8s}")
    return "\n".join(lines)


def _render_intraday(snapshot: dict[str, Any]) -> str:
    intraday = snapshot["intraday"]
    lines = [_hdr("3B. Intraday Hybrid Funnel")]
    stage_counts = intraday["stage_counts"]
    if stage_counts["watchlist"] == 0 and not intraday["trigger_rows"] and not intraday["transition_rows"]:
        lines.append("  (no intraday hybrid data in this run)")
        return "\n".join(lines)
    lines.append(
        "  Funnel: "
        f"watchlist={stage_counts['watchlist']}, "
        f"flush={stage_counts['flush_locked']}, "
        f"reclaim={stage_counts['reclaiming']}, "
        f"ready={stage_counts['ready']}, "
        f"entered={stage_counts['entered']}, "
        f"partial={stage_counts['partial']}, "
        f"trailed={stage_counts['trailed']}, "
        f"carried={stage_counts['carried']}"
    )
    coverage = intraday.get("coverage", {})
    if coverage:
        missing_5m_share = _safe_float(
            coverage.get("missing_5m_share", coverage.get("fallback_share"))
        )
        lines.append(
            "  Coverage: "
            f"5m-ready={coverage.get('with_5m', 0)}, "
            f"missing_5m={coverage.get('missing_5m', 0)}, "
            f"missing_5m share={missing_5m_share:.1%}"
        )
    live_selector = intraday.get("live_selector", {})
    if _safe_int(live_selector.get("considered")) > 0:
        lines.append(
            "  Live 5m cohort: "
            f"entered={_safe_int(live_selector.get('accepted'))}/{_safe_int(live_selector.get('considered'))}, "
            f"accepted avg_r={_safe_float(live_selector.get('accepted_avg_r')):+.3f}, "
            f"rejected shadow={_safe_float(live_selector.get('rejected_avg_shadow_r')):+.3f}, "
            f"delta={_safe_float(live_selector.get('delta_avg_r')):+.3f}"
        )
    if intraday["trigger_rows"]:
        lines.append("")
        lines.append(f"  {'Trigger':<16s} {'n':>5s} {'WR':>6s} {'AvgR':>8s}")
        lines.append("  " + "-" * 40)
        for row in intraday["trigger_rows"]:
            lines.append(f"  {row['label']:<16s} {row['n']:>5d} {row['wr']:>5.1%} {row['avg_r']:>+7.3f}")
    if intraday["hour_rows"]:
        lines.append("")
        lines.append("  Entry time-of-day expectancy:")
        for row in intraday["hour_rows"]:
            lines.append(f"    {row['label']}: n={row['n']}, avg_r={row['avg_r']:+.3f}")
    if intraday["transition_rows"]:
        lines.append("")
        lines.append("  Most common FSM transitions:")
        for row in intraday["transition_rows"]:
            lines.append(f"    {row['label']}: {row['count']}")
    return "\n".join(lines)


def _render_selector_frontier(snapshot: dict[str, Any]) -> str:
    frontier = snapshot["selector_frontier"]
    lines = [_hdr("3C. Selector Frontier & Filter Attribution")]
    if not frontier["cohorts"]:
        lines.append("  (no selector frontier data)")
        return "\n".join(lines)
    lines.append(f"  {'Cohort':<22s} {'Count':>6s} {'AvgR':>8s} {'ShadowR':>9s}")
    lines.append("  " + "-" * 52)
    for row in frontier["cohorts"]:
        lines.append(f"  {row['label']:<22s} {row['count']:>6d} {row['avg_r']:>+7.3f} {row['shadow_avg_r']:>+8.3f}")
    if frontier["gate_rows"]:
        lines.append("")
        lines.append(f"  {'Gate':<24s} {'Count':>6s} {'ShWR':>6s} {'ShAvgR':>8s} {'MissedR':>9s} {'AvoidedR':>10s} {'Verdict':>8s}")
        lines.append("  " + "-" * 86)
        for row in frontier["gate_rows"][:12]:
            lines.append(f"  {row['gate']:<24s} {row['count']:>6d} {row['shadow_wr']:>5.1%} {row['shadow_avg_r']:>+7.3f} {row['missed_total_r']:>+8.2f} {row['avoided_total_r']:>+9.2f} {row['verdict']:>8s}")
    if frontier.get("signal_model_rows"):
        lines.append("")
        lines.append("  Signal model attribution:")
        for row in frontier["signal_model_rows"]:
            lines.append(f"    {row['label']}: n={row['count']}, entered={row['entered']}, avg_effective_r={row['avg_effective_r']:+.3f}")
    if frontier.get("daily_score_rows"):
        lines.append("")
        lines.append("  Daily signal score monotonicity:")
        for row in frontier["daily_score_rows"]:
            lines.append(f"    {row['label']}: n={row['n']}, WR={row['wr']:.1%}, avg_r={row['avg_r']:+.3f}")
    for label, rows in frontier["threshold_sweeps"].items():
        if not rows:
            continue
        lines.append("")
        lines.append(f"  Threshold sweep: {label}")
        lines.append(f"    {'Value':<8s} {'Acc':>5s} {'AccAvgR':>9s} {'RejShR':>9s} {'TotAccR':>9s} {'Delta':>8s}")
        for row in rows:
            active = "*" if row["active"] else " "
            lines.append(f"    {active}{row['value']:<7.2f} {row['accepted']:>5d} {row['accepted_avg_r']:>+8.3f} {row['rejected_avg_shadow_r']:>+8.3f} {row['accepted_total_r']:>+8.2f} {row['selector_delta']:>+7.3f}")
    if frontier["invalidated_rows"]:
        lines.append("")
        lines.append("  Invalidated but worked later:")
        for row in frontier["invalidated_rows"]:
            lines.append(f"    {row['gate']}: n={row['count']}, worked={row['worked_later']} ({row['worked_share']:.1%}), avg_shadow={row['avg_shadow_r']:+.3f}, positive_avg={row['positive_shadow_avg_r']:+.3f}")
    return "\n".join(lines)


def _render_capacity_opportunity(snapshot: dict[str, Any]) -> str:
    capacity = snapshot["capacity_opportunity"]
    lines = [_hdr("3D. Capacity & Replacement Analysis")]
    if capacity["blocked_rows"]:
        lines.append(f"  {'Reason':<24s} {'Count':>6s} {'BlockedShR':>11s} {'OccupAvgR':>10s} {'Verdict':>8s}")
        lines.append("  " + "-" * 70)
        for row in capacity["blocked_rows"]:
            lines.append(f"  {row['reason']:<24s} {row['count']:>6d} {row['blocked_shadow_avg_r']:>+10.3f} {row['occupant_avg_r']:>+9.3f} {row['verdict']:>8s}")
    else:
        lines.append("  (no blocked-by-capacity records)")
    if capacity["replacement_rows"]:
        lines.append("")
        lines.append(f"  {'Exit':<16s} {'n':>5s} {'Windows':>8s} {'ReplN':>6s} {'ReplShR':>9s} {'ReuseN':>7s} {'ReuseR':>8s} {'ExitAvgR':>9s} {'Delta':>8s}")
        lines.append("  " + "-" * 90)
        for row in capacity["replacement_rows"]:
            lines.append(f"  {row['reason']:<16s} {row['exit_count']:>5d} {row['replacement_windows']:>8d} {row['replacement_count']:>6d} {row['replacement_avg_shadow_r']:>+8.3f} {row['actual_reuse_count']:>7d} {row['actual_reuse_avg_r']:>+7.3f} {row['exit_avg_r']:>+8.3f} {row['net_delta']:>+7.3f}")
    lines.append(f"  Verdict: {capacity['replacement_verdict']}")
    return "\n".join(lines)


def _render_entry_quality(snapshot: dict[str, Any]) -> str:
    entry = snapshot["entry_quality"]
    lines = [_hdr("11B. Entry Quality Deep Dive")]
    if entry["route_frontier_rows"]:
        current_route = None
        for row in entry["route_frontier_rows"]:
            if row["route"] != current_route:
                current_route = row["route"]
                lines.append(f"\n  {current_route}:")
                lines.append(f"    {'Variant':<22s} {'n':>5s} {'WR':>6s} {'AvgR':>8s} {'PF':>6s}")
            lines.append(f"    {row['label']:<22s} {row['n']:>5d} {row['wr']:>5.1%} {row['avg_r']:>+7.3f} {row['pf']:>6.2f}")
    else:
        lines.append("  (no route-level entry frontier data)")
    if entry["entry_tax_rows"]:
        lines.append("\n  Entry tax vs best feasible close:")
        for row in entry["entry_tax_rows"]:
            lines.append(f"    {row['route']}: n={row['count']}, avg_price_tax={row['avg_price_tax']:+.3f}, avg_r_tax={row['avg_r_tax']:+.3f}")
    if entry["route_bucket_rows"]:
        lines.append("\n  Entry timing buckets:")
        for row in entry["route_bucket_rows"][:18]:
            lines.append(f"    {row['route']} @ {row['bucket']}: n={row['n']}, avg_r={row['avg_r']:+.3f}")
    if entry["short_loser_rows"]:
        lines.append("\n  Short-hold loser audit:")
        for row in entry["short_loser_rows"][:10]:
            lines.append(f"    {row['bucket']} {row['symbol']} {row['route']} @ {row['entry_hour']}: score={row['intraday_score']:.1f}, reclaim={row['reclaim_bars']}, micro={row['micropressure']}, mfe={row['mfe_before_loss_r']:+.3f}, r={row['r']:+.3f}")
    if entry["score_bucket_rows"]:
        lines.append("\n  Intraday score buckets:")
        for row in entry["score_bucket_rows"]:
            lines.append(f"    {row['label']}: n={row['n']}, WR={row['wr']:.1%}, avg_r={row['avg_r']:+.3f}")
    if entry["score_cross_rows"]:
        lines.append("\n  Route x score bucket:")
        for row in entry["score_cross_rows"][:18]:
            lines.append(f"    {row['route']} x {row['label']}: n={row['n']}, avg_r={row['avg_r']:+.3f}")
    if entry.get("route_monotonicity_rows"):
        lines.append("\n  Route score monotonicity:")
        for row in entry["route_monotonicity_rows"]:
            lines.append(
                f"    {row['route']}: buckets={row['bucket_count']}, monotonic={row['monotonic_share']:.1%}, "
                f"best={row['best_bucket_avg_r']:+.3f}, worst={row['worst_bucket_avg_r']:+.3f}"
            )
    if entry["component_rows"]:
        lines.append("\n  Score component attribution (winners vs losers):")
        for row in entry["component_rows"]:
            lines.append(f"    {row['component']}: win={row['winner_mean']:+.2f}, loss={row['loser_mean']:+.2f}, delta={row['delta']:+.2f}")
    return "\n".join(lines)


def _render_management_forensics(snapshot: dict[str, Any]) -> str:
    mgmt = snapshot["management_forensics"]
    lines = [_hdr("12B. Trade Management Deep Dive")]
    if mgmt["hold_rows"]:
        lines.append("  Hold duration distribution:")
        for row in mgmt["hold_rows"]:
            lines.append(f"    {row['label']}: n={row['count']}, avg_r={row['avg_r']:+.3f}, avg_hours={row['avg_hours']:.2f}")
    if mgmt["short_loser_rows"]:
        row = mgmt["short_loser_rows"][0]
        lines.append("")
        lines.append(f"  Short losers: n={row['count']}, avg_mfe_before_negative_exit={row['avg_mfe_before_negative_exit_r']:+.3f}, >0.1R={row['mfe_gt_0p1']:.1%}, >0.25R={row['mfe_gt_0p25']:.1%}, >0.5R={row['mfe_gt_0p5']:.1%}")
    if mgmt["feature_rows"]:
        lines.append("")
        lines.append(f"  {'Feature':<12s} {'Bucket':<4s} {'Count':>6s} {'WR':>6s} {'AvgR':>8s} {'MeanMFE':>9s} {'Giveback':>9s}")
        lines.append("  " + "-" * 74)
        for row in mgmt["feature_rows"]:
            lines.append(f"  {row['feature']:<12s} {row['bucket']:<4s} {row['count']:>6d} {row['wr']:>5.1%} {row['avg_r']:>+7.3f} {row['mean_mfe']:>+8.3f} {row['mean_giveback']:>+8.3f}")
    if mgmt["giveback_rows"]:
        lines.append("\n  Giveback decomposition:")
        for row in mgmt["giveback_rows"]:
            lines.append(f"    {row['reason']} / {row['route']}: n={row['count']}, avg_giveback={row['avg_giveback']:+.3f}")
    stale = mgmt["stale_counterfactual"]
    if stale["count"] > 0:
        lines.append("")
        lines.append(f"  Stale-exit counterfactuals: n={stale['count']}, hold_to_eod={stale['hold_to_eod_avg_r']:+.3f}, tight_first_hour={stale['tight_first_hour_avg_r']:+.3f}")
    protect = mgmt["protection_summary"]
    lines.append(f"  Could-be-saved with protection: n={protect['count']} ({protect['share']:.1%}), avg_loser_r={protect['avg_loser_r']:+.3f}")
    return "\n".join(lines)


def _render_exit_replacement(snapshot: dict[str, Any]) -> str:
    audit = snapshot["exit_replacement"]
    lines = [_hdr("12C. Exit Replacement & Carry Audit")]
    if audit["route_rows"]:
        lines.append("  Route-level exit frontier:")
        for row in audit["route_rows"]:
            lines.append(f"    {row['route']}: n={row['count']}, actual={row['actual_avg_r']:+.3f}, best={row['best_variant']} @ {row['best_avg_r']:+.3f}")
    if audit["reason_rows"]:
        lines.append("")
        lines.append(f"  {'Reason':<16s} {'n':>5s} {'Actual':>8s} {'C1d':>8s} {'C3d':>8s} {'C5d':>8s} {'CloseG':>8s} {'CStop':>8s} {'Flow2':>8s}")
        lines.append("  " + "-" * 88)
        for row in audit["reason_rows"][:10]:
            lines.append(f"  {row['reason']:<16s} {row['count']:>5d} {row['actual_avg_r']:>+7.3f} {row['Carry 1d']:>+7.3f} {row['Carry 3d']:>+7.3f} {row['Carry 5d']:>+7.3f} {row['Carry 3d + close>=0.65']:>+7.3f} {row['Carry 3d + close stop']:>+7.3f} {row['Carry 3d + flowrev 2']:>+7.3f}")
    carry = audit["carry_audit"]
    lines.append("")
    lines.append(f"  Carry decision audit: binary={carry['binary_eligible']}, score_fallback={carry['score_fallback_eligible']}, carried={carry['actually_carried']}, should_have_carried_but_flattened={carry['should_have_carried_but_flattened']}, carried_underperformed_flatten={carry['carried_underperformed_flatten']}")
    if audit["flow_rows"]:
        lines.append("\n  Flow reversal timing:")
        for row in audit["flow_rows"]:
            lines.append(f"    {row['label']}: n={row['count']}, avg_r={row['avg_r']:+.3f}, total_r={row['total_r']:+.2f}, routes={row['routes']}")
    if audit["alpha_curve_rows"]:
        lines.append("\n  Intraday alpha curve:")
        for row in audit["alpha_curve_rows"][:10]:
            lines.append(f"    {row['label']}: n={row['count']}, avg_r={row['avg_r']:+.3f}, total_r={row['total_r']:+.2f}")
    return "\n".join(lines)


def _render_alpha_sources(snapshot: dict[str, Any]) -> str:
    alpha = snapshot["alpha_sources"]
    lines = [_hdr("13B. Positive Contribution / Alpha Source")]
    for title, rows in [
        ("Entry routes", alpha["route_rows"]),
        ("Sectors", alpha["sector_rows"]),
        ("Weekdays", alpha["weekday_rows"]),
        ("Entry hours", alpha["hour_rows"]),
    ]:
        lines.append(f"  {title}:")
        for row in rows:
            lines.append(f"    {row['label']}: n={row['count']}, avg_r={row['avg_r']:+.3f}, total_r={row['total_r']:+.2f}")
    lines.append("  Feature buckets:")
    for feature, rows in alpha["feature_bucket_rows"].items():
        top = ", ".join(f"{row['label']} {row['total_r']:+.2f}R" for row in rows[:3]) or "-"
        lines.append(f"    {feature}: {top}")
    keepers = alpha["best_keepers"]
    if keepers["gates"]:
        lines.append("  Best keepers (gates):")
        for row in keepers["gates"]:
            lines.append(f"    {row['gate']}: avoided={row['avoided_total_r']:+.2f}R, missed={row['missed_total_r']:+.2f}R")
    if keepers["routes"]:
        lines.append("  Best keepers (routes):")
        for row in keepers["routes"]:
            lines.append(f"    {row['label']}: avg_r={row['avg_r']:+.3f}, total_r={row['total_r']:+.2f}")
    return "\n".join(lines)


def _render_shadow(snapshot: dict[str, Any]) -> str:
    shadow = snapshot["shadow"]
    lines = [_hdr("4. Rejected Candidate Shadow Analysis")]
    lines.append(f"  Accepted trades: {_fmt_stats(shadow['actual'])}")
    lines.append(f"  Rejected shadows: {_fmt_stats(shadow['shadow'])}")
    lines.append(f"  Delta: avg_r {shadow['delta_avg_r']:+.3f}, WR {shadow['delta_wr']:+.1%} (positive means the filters are helping).")
    return "\n".join(lines)


def _render_selection(snapshot: dict[str, Any]) -> str:
    selection = snapshot["selection"]
    lines = [_hdr("5. Chosen vs Skipped Same-Day Attribution")]
    lines.append(f"  Crowded days: {selection['crowded_days']}")
    lines.append(f"  Entered avg_r on crowded days: {selection['entered_avg_r']:+.3f}")
    lines.append(f"  Skipped shadow avg_r on crowded days: {selection['skipped_avg_shadow_r']:+.3f}")
    lines.append(f"  Days with skipped shadow alpha > entered alpha: {selection['days_with_missed_alpha']}")
    for item in selection["top_days"]:
        lines.append(f"  {item['trade_date']}: entered {item['entered_avg_r']:+.3f}, skipped {item['skipped_avg_shadow_r']:+.3f}, best skipped {item.get('best_skipped_symbol') or '-'} {(_safe_float(item.get('best_skipped_shadow_r'))):+.3f}, skipped beating worst entered={_safe_int(item.get('skipped_beating_worst_entered'))}")
    return "\n".join(lines)


def _render_monotonicity(snapshot: dict[str, Any]) -> str:
    lines = [_hdr("6. Feature Monotonicity Tables")]
    for feature, rows in snapshot["monotonicity"].items():
        if not rows:
            continue
        lines.append(f"\n  {feature}:")
        lines.append(f"    {'Bin':<22s} {'n':>5s} {'WR':>6s} {'AvgR':>8s} {'CI10-90':>18s}")
        for row in rows:
            lines.append(f"    {row['label']:<22s} {row['n']:>5d} {row['wr']:>5.1%} {row['avg_r']:>+7.3f} [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]")
    return "\n".join(lines)


def _render_interactions(snapshot: dict[str, Any]) -> str:
    lines = [_hdr("7. Interaction Tables")]
    for name, table in snapshot["interactions"].items():
        if not table["cells"]:
            continue
        lines.append(f"\n  {name}:")
        for row in table["cells"]:
            lines.append("    " + " | ".join(f"n={cell['n']},R={cell['avg_r']:+.2f}" for cell in row))
    return "\n".join(lines)


def _render_mfe_capture(snapshot: dict[str, Any]) -> str:
    lines = [_hdr("8. MFE Capture by Exit Path")]
    lines.append(f"  {'Reason':<22s} {'Count':>6s} {'AvgR':>8s} {'Capture':>9s} {'Giveback':>10s}")
    lines.append("  " + "-" * 64)
    for row in snapshot["mfe_capture"]["rows"]:
        lines.append(f"  {row['reason']:<22s} {row['count']:>6d} {row['avg_r']:>+7.3f} {row['capture']:>8.1%} {row['giveback']:>+9.3f}")
    if snapshot["mfe_capture"]["lost_alpha"]:
        lines.append("\n  Top lost-alpha EOD flatten trades:")
        for item in snapshot["mfe_capture"]["lost_alpha"][:5]:
            lines.append(f"    {item['trade_date']} {item['symbol']}: actual {item['actual_r']:+.3f} vs MFE {item['mfe_r']:+.3f} (lost {item['lost_r']:+.3f}R)")
    return "\n".join(lines)


def _render_frontier(snapshot: dict[str, Any]) -> str:
    lines = [_hdr("9. Economic Exit Frontier")]
    lines.append(f"  {'Variant':<34s} {'n':>5s} {'WR':>6s} {'AvgR':>8s} {'PF':>6s}")
    lines.append("  " + "-" * 67)
    for row in snapshot["exit_frontier"]:
        lines.append(f"  {row['label']:<34s} {row['n']:>5d} {row['wr']:>5.1%} {row['avg_r']:>+7.3f} {row['pf']:>6.2f}")
    return "\n".join(lines)


def _render_carry_funnel(snapshot: dict[str, Any]) -> str:
    carry = snapshot["carry_funnel"]
    lines = [_hdr("10. Carry Eligibility Funnel")]
    lines.append(f"  EOD flatten trades: {carry['eod']}")
    lines.append(f"  Profitable at close: {carry['profitable']}")
    lines.append(f"  Close-in-range >= 0.65: {carry['close_pct_gate']}")
    lines.append(f"  MFE >= 0.25R: {carry['mfe_gate']}")
    lines.append(f"  No immediate flow reversal: {carry['flow_ok']}")
    if carry.get("route_rows"):
        lines.append("  Route carry gates:")
        for row in carry["route_rows"]:
            lines.append(
                f"    {row['route']}: eod={row['eod']}, profitable={row['profitable']}, "
                f"binary_ok={row['carry_binary_ok']}, score_ok={row['carry_score_ok']}"
            )
    for row in carry["forward_rows"]:
        lines.append(f"  {row['label']}: n={row['n']}, avg_r={row['avg_r']:+.3f}, PF={row['pf']:.2f}")
    return "\n".join(lines)


def _render_entry_timing(snapshot: dict[str, Any]) -> str:
    lines = [_hdr("11. Entry Timing Counterfactuals")]
    if not snapshot["entry_timing"]:
        lines.append("  (no 5m replay data available)")
        return "\n".join(lines)
    lines.append(f"  {'Variant':<24s} {'n':>5s} {'WR':>6s} {'AvgR':>8s} {'PF':>6s} {'EntryLoc':>10s}")
    lines.append("  " + "-" * 66)
    for row in snapshot["entry_timing"]:
        lines.append(f"  {row['label']:<24s} {row['n']:>5d} {row['wr']:>5.1%} {row['avg_r']:>+7.3f} {row['pf']:>6.2f} {row['entry_location']:>9.1%}")
    lines.append("  EntryLoc is the average entry position within the day's range; lower is better for pullback buys.")
    return "\n".join(lines)


def _render_stop_calibration(trades: list[TradeRecord]) -> str:
    stops = [trade for trade in trades if (trade.exit_reason or "UNKNOWN") == "STOP_HIT"]
    all_mae = [_safe_float(_meta(trade, "mae_r", 0.0)) for trade in trades]
    lines = [_hdr("12. Stop Calibration")]
    lines.append(f"  Stop hits: {len(stops)}/{len(trades)} ({_share(len(stops), len(trades)):.1%})" if trades else "  N/A")
    if stops:
        lines.append(f"  Stop-hit profile: {_fmt_stats(_trade_stats(stops))}")
    if all_mae:
        lines.append(f"  MAE exceedance: >0.5R {_share(sum(1 for v in all_mae if v > 0.5), len(all_mae)):.0%}, >0.75R {_share(sum(1 for v in all_mae if v > 0.75), len(all_mae)):.0%}, >1.0R {_share(sum(1 for v in all_mae if v > 1.0), len(all_mae)):.0%}")
    return "\n".join(lines)


def _render_concentration(snapshot: dict[str, Any]) -> str:
    concentration = snapshot["concentration"]
    lines = [_hdr("13. Regime / Sector / Weekday Concentration")]
    lines.append("  Top sectors by participation:")
    for row in concentration["sector_rows"]:
        lines.append(f"    {row['label']}: n={row['n']}, avg_r={row['avg_r']:+.3f}")
    lines.append("  Weekday mix:")
    for row in concentration["day_rows"]:
        lines.append(f"    {row['label']}: n={row['n']}, avg_r={row['avg_r']:+.3f}")
    return "\n".join(lines)


def _render_low_trade_days(snapshot: dict[str, Any]) -> str:
    lines = [_hdr("14. No-Trade / Low-Trade Day Forensics")]
    if not snapshot["low_trade_days"]:
        lines.append("  (no rich candidate days with low entry counts captured)")
        return "\n".join(lines)
    for item in snapshot["low_trade_days"]:
        gate_summary = ", ".join(f"{gate}:{count}" for gate, count in item["top_gates"]) or "-"
        lines.append(f"  {item['trade_date']}: candidates={item['candidate_count']}, entered={item['entered_count']}, skipped_avg={item['avg_skipped_shadow_r']:+.3f}, best_skipped={item['best_skipped_shadow_r']:+.3f}, gates={gate_summary}")
    return "\n".join(lines)


def _render_monthly(trades: list[TradeRecord]) -> str:
    by_month: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        by_month[trade.exit_time.strftime('%Y-%m')].append(trade)
    lines = [_hdr("15. Monthly PnL")]
    cum_r = 0.0
    lines.append(f"  {'Month':>8s} {'Trades':>6s} {'WR':>6s} {'NetR':>8s} {'CumR':>8s}")
    lines.append("  " + "-" * 44)
    for month in sorted(by_month):
        stats = _trade_stats(by_month[month])
        cum_r += stats["total_r"]
        lines.append(f"  {month:>8s} {int(stats['n']):>6d} {stats['wr']:>5.1%} {stats['total_r']:>+7.2f} {cum_r:>+7.2f}")
    return "\n".join(lines)


def _render_rolling(trades: list[TradeRecord]) -> str:
    lines = [_hdr("16. Rolling Expectancy")]
    if len(trades) < 20:
        lines.append("  Insufficient trades for rolling analysis.")
        return "\n".join(lines)
    rs = [float(trade.r_multiple) for trade in trades]
    rolling = [float(np.mean(rs[idx:idx + 20])) for idx in range(len(rs) - 19)]
    lines.append(f"  Start: {rolling[0]:+.3f}R  |  End: {rolling[-1]:+.3f}R")
    lines.append(f"  Min: {min(rolling):+.3f}R  |  Max: {max(rolling):+.3f}R")
    lines.append(f"  Positive windows: {_share(sum(1 for val in rolling if val > 0), len(rolling)):.0%}")
    return "\n".join(lines)


def _render_drawdowns(trades: list[TradeRecord]) -> str:
    lines = [_hdr("17. Drawdown Episodes")]
    if not trades:
        lines.append("  No trades.")
        return "\n".join(lines)
    rs = np.array([float(trade.r_multiple) for trade in trades], dtype=float)
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    lines.append(f"  Max drawdown: {float(np.min(dd)):+.2f}R")
    lines.append(f"  Worst point: trade #{int(np.argmin(dd)) + 1 if len(dd) else 0}")
    return "\n".join(lines)


def _render_appendix(trades: list[TradeRecord]) -> str:
    carry = [trade for trade in trades if _safe_int(_meta(trade, "hold_days", 1), 1) > 1]
    lines = [_hdr("Appendix")]
    lines.append("  RSI-only same-day exit threshold sweeps are intentionally downgraded because they mainly relabel close exits.")
    if carry:
        lines.append(f"  Nontrivial carry trades: {len(carry)}.")
    else:
        lines.append("  Hold-duration table is intentionally omitted from the main body because the run is overwhelmingly same-day.")
    return "\n".join(lines)


def pullback_full_diagnostic(
    trades: list[TradeRecord],
    *,
    metrics: dict[str, float] | None = None,
    replay=None,
    daily_selections: dict | None = None,
    candidate_ledger: dict[date, list[dict[str, Any]]] | None = None,
    funnel_counters: dict[str, int] | None = None,
    rejection_log: list[dict[str, Any]] | None = None,
    shadow_outcomes: list[dict[str, Any]] | None = None,
    selection_attribution: dict[date, dict[str, Any]] | None = None,
    fsm_log: list[dict[str, Any]] | None = None,
) -> str:
    if not trades:
        return "No trades to diagnose."
    snapshot = compute_pullback_diagnostic_snapshot(
        trades,
        metrics=metrics,
        replay=replay,
        daily_selections=daily_selections,
        candidate_ledger=candidate_ledger,
        funnel_counters=funnel_counters,
        rejection_log=rejection_log,
        shadow_outcomes=shadow_outcomes,
        selection_attribution=selection_attribution,
        fsm_log=fsm_log,
    )
    sections = [
        _render_verdicts(snapshot),
        _render_overview(snapshot),
        _render_exit_mix(trades),
        _render_funnel(snapshot),
        _render_intraday(snapshot),
        _render_selector_frontier(snapshot),
        _render_capacity_opportunity(snapshot),
        _render_shadow(snapshot),
        _render_selection(snapshot),
        _render_monotonicity(snapshot),
        _render_interactions(snapshot),
        _render_mfe_capture(snapshot),
        _render_frontier(snapshot),
        _render_carry_funnel(snapshot),
        _render_entry_timing(snapshot),
        _render_entry_quality(snapshot),
        _render_stop_calibration(trades),
        _render_management_forensics(snapshot),
        _render_exit_replacement(snapshot),
        _render_concentration(snapshot),
        _render_alpha_sources(snapshot),
        _render_low_trade_days(snapshot),
        _render_monthly(trades),
        _render_rolling(trades),
        _render_drawdowns(trades),
        _render_appendix(trades),
    ]
    return "\n".join(sections)
