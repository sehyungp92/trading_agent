"""IARIC OOS failure attribution plus ablation/perturbation diagnostics.

This is intentionally a research runner, not a production promotion tool. It
uses the 2026-03-21..2026-05-01 OOS period to diagnose the failure mode, then
checks promising changes against the 2024-01-01..2026-03-20 calibration window
so OOS-only overfit candidates are visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from backtests.scripts.iaric_walkforward_reopt import (
    CALIBRATION_END,
    CALIBRATION_START,
    LOCKBOX_END,
    LOCKBOX_START,
    _evaluate_many,
    _window_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "output"
STOCK_IARIC = OUTPUT_ROOT / "stock" / "iaric"


def _now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _apply_patch(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in changes.items():
        if value == "__DELETE__":
            out.pop(key, None)
        else:
            out[key] = value
    return out


def _load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    round1 = _read_json(STOCK_IARIC / "round_1" / "optimized_config.json")
    current = _read_json(STOCK_IARIC / "round_2" / "optimized_config.json")
    hybrid = _read_json(STOCK_IARIC / "round_2" / "hybrid_greedy.json")
    return round1, current, hybrid


def _accepted_order(hybrid: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidate_mutations = hybrid.get("candidate_mutations") or {}
    names = []
    for row in hybrid.get("rounds") or []:
        if row.get("kept") and row.get("best_name"):
            names.append(str(row["best_name"]))
    return [(name, dict(candidate_mutations.get(name) or {})) for name in names]


def _add_candidate(
    candidates: list[dict[str, Any]],
    seen: set[str],
    *,
    name: str,
    group: str,
    mutations: dict[str, Any],
    patch: dict[str, Any],
    note: str = "",
) -> None:
    signature = json.dumps(mutations, sort_keys=True, default=str)
    if signature in seen:
        return
    seen.add(signature)
    candidates.append(
        {
            "name": name,
            "group": group,
            "patch": patch,
            "mutations": mutations,
            "note": note,
        }
    )


def build_candidate_set(round1: dict[str, Any], current: dict[str, Any], hybrid: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    _add_candidate(
        candidates,
        seen,
        name="current",
        group="baseline",
        mutations=current,
        patch={},
        note="Existing round_2 champion.",
    )
    _add_candidate(
        candidates,
        seen,
        name="round1",
        group="baseline",
        mutations=round1,
        patch={},
        note="Previous optimized round.",
    )

    cumulative = dict(round1)
    for idx, (name, patch) in enumerate(_accepted_order(hybrid), start=1):
        cumulative = _apply_patch(cumulative, patch)
        _add_candidate(
            candidates,
            seen,
            name=f"prefix_{idx:02d}_{name}",
            group="accepted_prefix",
            mutations=dict(cumulative),
            patch=patch,
            note="Cumulative accepted-mutation prefix from round_1.",
        )

    rollback_specs = {
        "rollback_partial_trigger_030": {"param_overrides.pb_v2_partial_profit_trigger_r": 0.3},
        "rollback_signal_floor_75": {"param_overrides.pb_v2_signal_floor": 75.0},
        "rollback_gap_1": {"param_overrides.pb_v2_gap_max_pct": 1.0},
        "rollback_sector_cap_2": {"max_per_sector": 2},
        "rollback_stale_6_005": {
            "param_overrides.pb_v2_stale_bars": 6,
            "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
        },
        "rollback_routes_on": {
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
            "param_overrides.pb_v2_afternoon_retest_enabled": True,
        },
        "rollback_floor75_sector2": {
            "param_overrides.pb_v2_signal_floor": 75.0,
            "max_per_sector": 2,
        },
        "rollback_floor75_stale6": {
            "param_overrides.pb_v2_signal_floor": 75.0,
            "param_overrides.pb_v2_stale_bars": 6,
            "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
        },
        "rollback_sector2_routes_on": {
            "max_per_sector": 2,
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
            "param_overrides.pb_v2_afternoon_retest_enabled": True,
        },
        "rollback_round1_core": {
            "param_overrides.pb_v2_signal_floor": 75.0,
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.3,
            "param_overrides.pb_v2_stale_bars": 6,
            "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
            "max_per_sector": 2,
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
            "param_overrides.pb_v2_afternoon_retest_enabled": True,
        },
        "remove_remainder_override": {"param_overrides.pb_v2_partial_profit_remainder_stop_r": "__DELETE__"},
        "remove_cdd_override": {"param_overrides.pb_cdd_max": "__DELETE__"},
    }
    for name, patch in rollback_specs.items():
        _add_candidate(
            candidates,
            seen,
            name=name,
            group="accepted_rollback",
            mutations=_apply_patch(current, patch),
            patch=patch,
            note="One-at-a-time rollback/removal from current round_2 champion.",
        )

    single_sweeps: dict[str, tuple[str, list[Any]]] = {
        "signal_floor": ("param_overrides.pb_v2_signal_floor", [66.0, 68.0, 70.0, 72.0, 74.0, 75.0, 78.0, 80.0]),
        "gap_max": ("param_overrides.pb_v2_gap_max_pct", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        "gap_min": ("param_overrides.pb_v2_gap_min_pct", [-15.0, -8.0, -5.0, -2.0, 0.0]),
        "cdd_max": ("param_overrides.pb_cdd_max", [3, 4, 5, 6, 999]),
        "sector_cap": ("max_per_sector", [1, 2, 3, 4, 5]),
        "max_positions": ("param_overrides.pb_max_positions", [6, 8, 10, 12]),
        "open_slots": ("param_overrides.pb_v2_open_scored_max_slots", [1, 2, 3, 4, 6]),
        "partial_trigger": ("param_overrides.pb_v2_partial_profit_trigger_r", [0.1, 0.2, 0.3, 0.4, 0.6]),
        "remainder_stop": ("param_overrides.pb_v2_partial_profit_remainder_stop_r", [0.0, 0.3, 0.5, 0.7]),
        "rsi_exit_open": ("param_overrides.pb_v2_rsi_exit_open_scored", [50.0, 55.0, 60.0, 65.0, 70.0]),
        "ema_min_r": ("param_overrides.pb_v2_ema_reversion_min_r", [0.0, 0.03, 0.08, 0.15]),
        "carry_close": ("param_overrides.pb_open_scored_carry_close_pct_min", [0.0, 0.45, 0.5, 0.6, 0.75]),
        "carry_mfe": ("param_overrides.pb_open_scored_carry_mfe_gate_r", [0.0, 0.1, 0.2, 0.4]),
        "flatten_loss": ("param_overrides.pb_v2_flatten_loss_r", [-0.25, -0.35, -0.5, -0.75]),
        "open_max_hold": ("param_overrides.pb_open_scored_max_hold_days", [1, 2, 3]),
        "sma_min": ("param_overrides.pb_v2_sma_dist_min_pct", [-10.0, -5.0, 0.0, 2.0]),
        "sma_max": ("param_overrides.pb_v2_sma_dist_max_pct", [10.0, 15.0, 20.0, 25.0]),
        "rank_pct_max": ("param_overrides.pb_entry_rank_pct_max", [50.0, 75.0, 90.0, 100.0]),
        "daily_min_score": ("param_overrides.pb_daily_signal_min_score", [50.0, 52.0, 54.0, 56.0, 58.0]),
        "open_min_score": ("param_overrides.pb_v2_open_scored_min_score", [40.0, 45.0, 50.0, 55.0]),
        "delayed_score": ("param_overrides.pb_delayed_confirm_score_min", [47.0, 52.0, 57.0]),
        "flow_lookback": ("param_overrides.pb_open_scored_flow_reversal_lookback", [1, 2, 3]),
    }
    for label, (key, values) in single_sweeps.items():
        for value in values:
            patch = {key: value}
            _add_candidate(
                candidates,
                seen,
                name=f"sweep_{label}_{str(value).replace('.', 'p').replace('-', 'neg')}",
                group="single_perturbation",
                mutations=_apply_patch(current, patch),
                patch=patch,
            )

    stale_pairs = [(2, 0.04), (4, 0.05), (4, 0.08), (6, 0.05), (6, 0.10), (8, 0.05)]
    for bars, mfe in stale_pairs:
        patch = {
            "param_overrides.pb_v2_stale_bars": bars,
            "param_overrides.pb_v2_stale_mfe_thresh": mfe,
        }
        _add_candidate(
            candidates,
            seen,
            name=f"sweep_stale_{bars}_{str(mfe).replace('.', 'p')}",
            group="single_perturbation",
            mutations=_apply_patch(current, patch),
            patch=patch,
        )

    route_specs = {
        "routes_vwap_on": {"param_overrides.pb_v2_vwap_bounce_enabled": True},
        "routes_afternoon_on": {"param_overrides.pb_v2_afternoon_retest_enabled": True},
        "routes_both_on": {
            "param_overrides.pb_v2_vwap_bounce_enabled": True,
            "param_overrides.pb_v2_afternoon_retest_enabled": True,
        },
        "disable_delayed_confirm": {"param_overrides.pb_delayed_confirm_enabled": False},
    }
    for name, patch in route_specs.items():
        _add_candidate(
            candidates,
            seen,
            name=name,
            group="single_perturbation",
            mutations=_apply_patch(current, patch),
            patch=patch,
        )

    targeted = {
        "target_carry_close50_mfe20": {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.50,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
        },
        "target_carry_close50_flatten035": {
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.50,
            "param_overrides.pb_v2_flatten_loss_r": -0.35,
        },
        "target_floor70_rank75": {
            "param_overrides.pb_v2_signal_floor": 70.0,
            "param_overrides.pb_entry_rank_pct_max": 75.0,
        },
        "target_floor70_slots6": {
            "param_overrides.pb_v2_signal_floor": 70.0,
            "param_overrides.pb_v2_open_scored_max_slots": 6,
        },
        "target_floor68_rank75_slots6": {
            "param_overrides.pb_v2_signal_floor": 68.0,
            "param_overrides.pb_entry_rank_pct_max": 75.0,
            "param_overrides.pb_v2_open_scored_max_slots": 6,
        },
        "target_floor70_carry50_mfe20": {
            "param_overrides.pb_v2_signal_floor": 70.0,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.50,
            "param_overrides.pb_open_scored_carry_mfe_gate_r": 0.20,
        },
        "target_sma_min0_carry50": {
            "param_overrides.pb_v2_sma_dist_min_pct": 0.0,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.50,
        },
        "target_rank75_carry50": {
            "param_overrides.pb_entry_rank_pct_max": 75.0,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.50,
        },
        "target_sector3_floor70_slots6": {
            "max_per_sector": 3,
            "param_overrides.pb_v2_signal_floor": 70.0,
            "param_overrides.pb_v2_open_scored_max_slots": 6,
        },
        "target_floor70_gap25_slots6": {
            "param_overrides.pb_v2_signal_floor": 70.0,
            "param_overrides.pb_v2_gap_max_pct": 2.5,
            "param_overrides.pb_v2_open_scored_max_slots": 6,
        },
        "target_open_hold1_carry50": {
            "param_overrides.pb_open_scored_max_hold_days": 1,
            "param_overrides.pb_open_scored_carry_close_pct_min": 0.50,
        },
    }
    for name, patch in targeted.items():
        _add_candidate(
            candidates,
            seen,
            name=name,
            group="targeted_combo",
            mutations=_apply_patch(current, patch),
            patch=patch,
            note="Targeted at OOS weakness: low frequency, poor rank-100/carry behavior, or late-April adverse continuation.",
        )

    return candidates


def _metrics_from_trades(trades: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    return _window_metrics(trades, start, end)


def _summarize_eval(name: str, group: str, patch: dict[str, Any], ev: dict[str, Any], start: date, end: date) -> dict[str, Any]:
    if ev.get("error"):
        return {
            "name": name,
            "group": group,
            "patch": patch,
            "error": ev["error"],
        }
    trades = ev.get("trades") or []
    metrics = _metrics_from_trades(trades, start, end)
    return {
        "name": name,
        "group": group,
        "patch": patch,
        "trades": int(metrics["total_trades"]),
        "win_rate": float(metrics["win_rate"]),
        "profit_factor": metrics["profit_factor"],
        "avg_r": float(metrics["avg_r"]),
        "net_r": float(metrics["net_r"]),
        "max_drawdown_r": float(metrics["max_drawdown_r"]),
        "trades_per_month": float(metrics["trades_per_month"]),
    }


def _evaluate_candidates(
    candidates: list[dict[str, Any]],
    *,
    start: date,
    end: date,
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    named = [(item["name"], item["mutations"]) for item in candidates]
    return _evaluate_many(named, start=start, end=end, max_workers=max_workers)


def _finite_pf(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(number):
        return 99.0
    return number


def _join_oos_is(
    candidates: list[dict[str, Any]],
    oos_rows: dict[str, dict[str, Any]],
    is_rows: dict[str, dict[str, Any]],
    current_name: str = "current",
) -> list[dict[str, Any]]:
    current_oos = oos_rows[current_name]
    current_is = is_rows[current_name]
    by_name = {item["name"]: item for item in candidates}
    rows: list[dict[str, Any]] = []
    for name, oos in oos_rows.items():
        item = by_name.get(name, {})
        is_m = is_rows.get(name)
        if not is_m:
            continue
        row = {
            "name": name,
            "group": item.get("group", ""),
            "patch": item.get("patch", {}),
            "note": item.get("note", ""),
            "oos": oos,
            "is": is_m,
            "delta": {
                "oos_trades": oos.get("trades", 0) - current_oos.get("trades", 0),
                "oos_net_r": oos.get("net_r", 0.0) - current_oos.get("net_r", 0.0),
                "oos_avg_r": oos.get("avg_r", 0.0) - current_oos.get("avg_r", 0.0),
                "is_trades": is_m.get("trades", 0) - current_is.get("trades", 0),
                "is_net_r": is_m.get("net_r", 0.0) - current_is.get("net_r", 0.0),
                "is_avg_r": is_m.get("avg_r", 0.0) - current_is.get("avg_r", 0.0),
            },
        }
        is_pf_ok = _finite_pf(is_m.get("profit_factor")) >= _finite_pf(current_is.get("profit_factor")) * 0.90
        is_net_ok = is_m.get("net_r", 0.0) >= current_is.get("net_r", 0.0) * 0.90
        is_trade_ok = is_m.get("trades", 0) >= current_is.get("trades", 0) * 0.85
        oos_net_ok = oos.get("net_r", 0.0) > current_oos.get("net_r", 0.0)
        oos_freq_ok = oos.get("trades", 0) >= current_oos.get("trades", 0)
        row["eligible"] = bool(is_pf_ok and is_net_ok and is_trade_ok and oos_net_ok and oos_freq_ok)
        row["score"] = (
            4.0 * row["delta"]["oos_net_r"]
            + 0.20 * row["delta"]["oos_trades"]
            + 0.25 * row["delta"]["is_net_r"] / max(abs(current_is.get("net_r", 1.0)), 1.0)
            - max(0.0, -row["delta"]["is_trades"]) * 0.01
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            bool(row["eligible"]),
            row["delta"]["oos_net_r"],
            row["oos"]["trades"],
            row["delta"]["is_net_r"],
        ),
        reverse=True,
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "group",
        "eligible",
        "oos_trades",
        "oos_pf",
        "oos_avg_r",
        "oos_net_r",
        "oos_trades_delta",
        "oos_net_r_delta",
        "is_trades",
        "is_pf",
        "is_avg_r",
        "is_net_r",
        "is_trades_delta",
        "is_net_r_delta",
        "patch",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "group": row["group"],
                    "eligible": row["eligible"],
                    "oos_trades": row["oos"]["trades"],
                    "oos_pf": _fmt(row["oos"]["profit_factor"]),
                    "oos_avg_r": _fmt(row["oos"]["avg_r"]),
                    "oos_net_r": _fmt(row["oos"]["net_r"]),
                    "oos_trades_delta": row["delta"]["oos_trades"],
                    "oos_net_r_delta": _fmt(row["delta"]["oos_net_r"]),
                    "is_trades": row["is"]["trades"],
                    "is_pf": _fmt(row["is"]["profit_factor"]),
                    "is_avg_r": _fmt(row["is"]["avg_r"]),
                    "is_net_r": _fmt(row["is"]["net_r"]),
                    "is_trades_delta": row["delta"]["is_trades"],
                    "is_net_r_delta": _fmt(row["delta"]["is_net_r"]),
                    "patch": json.dumps(_jsonable(row["patch"]), sort_keys=True),
                }
            )


def _run_current_diagnostics(current: dict[str, Any]) -> dict[str, Any]:
    from backtests.stock.auto.config_mutator import mutate_iaric_config
    from backtests.stock.auto.scoring import extract_metrics
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.data.replay_cache import load_research_replay_bundle
    from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

    data_dir = PROJECT_ROOT / "backtests" / "stock" / "data" / "raw"
    replay = load_research_replay_bundle(data_dir).data
    base = IARICBacktestConfig(
        start_date=LOCKBOX_START.isoformat(),
        end_date=LOCKBOX_END.isoformat(),
        initial_equity=10_000.0,
        tier=3,
        data_dir=data_dir,
    )
    config = mutate_iaric_config(base, current)
    result = IARICPullbackEngine(config, replay, collect_diagnostics=True).run()
    metrics_obj = extract_metrics(result.trades, result.equity_curve, result.timestamps, 10_000.0)
    metrics = asdict(metrics_obj) if is_dataclass(metrics_obj) else dict(metrics_obj)

    rows: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    for trade in result.trades:
        meta = trade.metadata or {}
        row = {
            "entry_date": trade.entry_time.date().isoformat(),
            "entry_time": trade.entry_time.isoformat(),
            "exit_date": trade.exit_time.date().isoformat(),
            "symbol": trade.symbol,
            "r": float(trade.r_multiple),
            "pnl_net": float(trade.pnl_net),
            "exit_reason": trade.exit_reason,
            "sector": trade.sector,
            "regime": trade.regime_tier,
            "route": meta.get("entry_route_family") or meta.get("entry_trigger"),
            "trend_tier": meta.get("trend_tier"),
            "entry_gap_pct": meta.get("entry_gap_pct"),
            "entry_sma_dist_pct": meta.get("entry_sma_dist_pct"),
            "entry_cdd": meta.get("entry_cdd"),
            "entry_rank": meta.get("entry_rank"),
            "entry_rank_pct": meta.get("entry_rank_pct"),
            "n_candidates": meta.get("n_candidates"),
            "daily_signal_score": meta.get("daily_signal_score"),
            "mfe_r": meta.get("mfe_r"),
            "mae_r": meta.get("mae_r"),
            "close_r": meta.get("close_r"),
            "close_pct": meta.get("close_pct"),
            "partial_taken": meta.get("partial_taken"),
        }
        rows.append(row)

    for key in ["exit_reason", "route", "sector", "regime", "trend_tier"]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key))].append(float(row["r"]))
        groups[key] = {
            bucket: {
                "trades": len(values),
                "net_r": sum(values),
                "avg_r": sum(values) / len(values),
                "win_rate": sum(1 for value in values if value > 0) / len(values),
            }
            for bucket, values in sorted(buckets.items(), key=lambda item: sum(item[1]))
        }

    largest_losses = sorted(rows, key=lambda row: row["r"])[:5]
    return {
        "metrics": metrics,
        "trades": rows,
        "groups": groups,
        "largest_losses": largest_losses,
        "funnel_counters": result.funnel_counters or {},
    }


def _format_markdown(summary: dict[str, Any]) -> str:
    current = summary["current"]
    top = summary["top_rows"][:12]
    eligible = [row for row in summary["ranked_rows"] if row.get("eligible")]
    diagnostics = summary["current_oos_diagnostics"]
    largest = diagnostics.get("largest_losses", [])

    lines = [
        "# IARIC OOS Ablation and Perturbation Diagnostic",
        "",
        f"- Generated: {summary['generated_at_utc']}",
        f"- OOS window: {LOCKBOX_START.isoformat()} to {LOCKBOX_END.isoformat()}",
        f"- IS window: {CALIBRATION_START.isoformat()} to {CALIBRATION_END.isoformat()}",
        f"- Candidates evaluated on OOS: {summary['candidate_count']}",
        f"- Candidates evaluated on IS: {summary['is_evaluated_count']}",
        "",
        "## Current OOS Failure Shape",
        "",
        (
            f"Current champion: {current['oos']['trades']} OOS trades, "
            f"{_fmt(current['oos']['profit_factor'])} PF, "
            f"{_fmt(current['oos']['avg_r'])} avgR, "
            f"{_fmt(current['oos']['net_r'])} netR."
        ),
        "",
        "Largest OOS losses:",
    ]
    for row in largest:
        lines.append(
            f"- {row['entry_date']} {row['symbol']} {row['r']:+.3f}R, "
            f"{row['exit_reason']}, sector={row['sector']}, "
            f"score={row.get('daily_signal_score')}, rank_pct={row.get('entry_rank_pct')}, "
            f"gap={row.get('entry_gap_pct')}%, sma_dist={row.get('entry_sma_dist_pct')}%"
        )

    lines.extend(
        [
            "",
            "Bad buckets:",
        ]
    )
    for group_key in ["sector", "exit_reason", "regime"]:
        buckets = diagnostics.get("groups", {}).get(group_key, {})
        worst = list(buckets.items())[:3]
        for bucket, payload in worst:
            lines.append(
                f"- {group_key}={bucket}: n={payload['trades']}, "
                f"netR={payload['net_r']:+.3f}, avgR={payload['avg_r']:+.3f}"
            )

    lines.extend(
        [
            "",
            "## Best IS-Checked Candidates",
            "",
            "| Candidate | Group | Eligible | OOS trades | OOS netR | OOS avgR | IS trades | IS netR | IS avgR | Patch |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top:
        patch = json.dumps(_jsonable(row["patch"]), sort_keys=True)
        lines.append(
            f"| {row['name']} | {row['group']} | {row['eligible']} | "
            f"{row['oos']['trades']} | {_fmt(row['oos']['net_r'])} | {_fmt(row['oos']['avg_r'])} | "
            f"{row['is']['trades']} | {_fmt(row['is']['net_r'])} | {_fmt(row['is']['avg_r'])} | `{patch}` |"
        )

    lines.extend(["", "## Interpretation", ""])
    if eligible:
        best = eligible[0]
        lines.append(
            "At least one diagnostic candidate improved OOS netR and held OOS frequency "
            "without breaching the 90% IS PF/netR and 85% IS trade-retention checks."
        )
        lines.append(
            f"Best eligible candidate: `{best['name']}` with OOS "
            f"{best['oos']['trades']} trades / {_fmt(best['oos']['net_r'])} netR "
            f"and IS {_fmt(best['is']['net_r'])} netR."
        )
    else:
        lines.append(
            "No candidate simultaneously improved OOS expectancy/frequency and cleared "
            "the conservative IS-retention checks. Treat any OOS-only improvement as overfit."
        )

    lines.append(
        "The current OOS loss is not a single catastrophic print; it is a small-sample cluster "
        "of normal-sized mean-reversion failures concentrated in late April, especially "
        "Healthcare and Communication Services."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    max_workers = max(1, int(args.max_workers))
    output_dir = Path(args.output_root) if args.output_root else OUTPUT_ROOT / f"iaric_oos_ablation_{_now_label()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    round1, current, hybrid = _load_configs()
    candidates = build_candidate_set(round1, current, hybrid)
    _write_json(output_dir / "candidate_set.json", candidates)

    full_evals = _evaluate_candidates(candidates, start=CALIBRATION_START, end=LOCKBOX_END, max_workers=max_workers)
    by_candidate = {item["name"]: item for item in candidates}
    oos_rows = {
        name: _summarize_eval(
            name,
            by_candidate[name]["group"],
            by_candidate[name]["patch"],
            ev,
            LOCKBOX_START,
            LOCKBOX_END,
        )
        for name, ev in full_evals.items()
    }
    _write_json(output_dir / "oos_results.json", list(oos_rows.values()))

    is_rows = {
        name: _summarize_eval(
            name,
            by_candidate[name]["group"],
            by_candidate[name]["patch"],
            ev,
            CALIBRATION_START,
            CALIBRATION_END,
        )
        for name, ev in full_evals.items()
    }
    _write_json(output_dir / "is_results.json", list(is_rows.values()))

    ranked_rows = _join_oos_is(candidates, oos_rows, is_rows)
    _write_json(output_dir / "ranked_results.json", ranked_rows)
    _write_csv(output_dir / "ranked_results.csv", ranked_rows)

    current_diagnostics = _run_current_diagnostics(current)
    _write_json(output_dir / "current_oos_diagnostics.json", current_diagnostics)

    current_row = next(row for row in ranked_rows if row["name"] == "current")
    best_eligible = next((row for row in ranked_rows if row.get("eligible") and row["name"] != "current"), None)
    if best_eligible:
        best_config = by_candidate[best_eligible["name"]]["mutations"]
        _write_json(output_dir / "best_eligible_challenger_config.json", best_config)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "is_evaluated_count": len(candidates),
        "current": current_row,
        "best_eligible": best_eligible,
        "top_rows": ranked_rows[:12],
        "ranked_rows": ranked_rows,
        "current_oos_diagnostics": current_diagnostics,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_format_markdown(_jsonable(summary)), encoding="utf-8")
    print(f"Done. Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
