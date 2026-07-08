"""Analyze and repair breakout round-2 out-of-sample performance.

The requested split is:
- in sample: 2026-01-04 through 2026-04-20
- out of sample: 2026-04-21 through 2026-05-23

This script evaluates checkpoints, granular ablations across both historical
and current accepted mutations, local perturbations, and a second targeted
repair phase generated after the OOS trade autopsy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE, build_backtest_config_from_profile
from crypto_trader.backtest.runner import run, run_split_continuation
from crypto_trader.optimize.breakout_round3_pre_round1 import build_pre_round1_config
from crypto_trader.optimize.config_mutator import apply_mutations
from crypto_trader.optimize.revalidation import local_perturbation_values
from crypto_trader.strategy.breakout.config import BreakoutConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SYMBOLS = ["BTC", "ETH", "SOL"]

IS_START = "2026-01-04"
IS_END = "2026-04-20"
OOS_START = "2026-04-21"
OOS_END = "2026-05-23"
FULL_END = OOS_END

CURRENT_ROUND1_PATH = ROOT / "output" / "breakout" / "round_1" / "optimized_config.json"
CURRENT_ROUND2_PATH = ROOT / "output" / "breakout" / "round_2" / "optimized_config.json"
CURRENT_PHASE_STATE_PATH = ROOT / "output" / "breakout" / "round_2" / "phase_state.json"
CURRENT_RAW_PHASE6_PATH = ROOT / "output" / "breakout" / "round_2" / "exploratory_phase_6_config.json"
CURRENT_MANIFEST_PATH = ROOT / "output" / "breakout" / "rounds_manifest.json"

ARCHIVE_ROOT = ROOT / "output" / "breakout" / "archive" / "pre_parity_reset_20260525T111754Z"
ARCHIVE_MANIFEST_PATH = ARCHIVE_ROOT / "rounds_manifest.json"
ARCHIVE_ROUND2_PATH = ARCHIVE_ROOT / "round_2" / "optimized_config.json"
ARCHIVE_ROUND3_PATH = ARCHIVE_ROOT / "round_3" / "optimized_config.json"

METRIC_KEYS = [
    "net_return_pct",
    "net_profit",
    "total_trades",
    "win_rate",
    "expectancy_r",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "exit_efficiency",
    "avg_mae_r",
    "avg_mfe_r",
    "realized_pnl_net",
    "terminal_mark_pnl_net",
    "terminal_mark_count",
    "total_fees",
    "funding_cost_total",
    "max_consecutive_losses",
]


def _quiet_logging() -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_strategy_payload(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    payload = raw.get("strategy", raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return payload


def _load_config(path: Path) -> BreakoutConfig:
    return BreakoutConfig.from_dict(_load_strategy_payload(path))


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _diff_payloads(base_payload: dict[str, Any], candidate_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_flat = _flatten(base_payload)
    cand_flat = _flatten(candidate_payload)
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(set(base_flat) | set(cand_flat)):
        if key == "symbols":
            continue
        if base_flat.get(key) != cand_flat.get(key):
            diff[key] = {
                "base_value": base_flat.get(key),
                "candidate_value": cand_flat.get(key),
            }
    return diff


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_path(payload: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    updated = deepcopy(payload)
    section, field = path.split(".", 1)
    if section not in updated or not isinstance(updated[section], dict):
        updated[section] = {}
    updated[section][field] = value
    return updated


def _metrics_subset(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS if key in metrics}


def _finite(value: float, cap: float) -> float:
    if not math.isfinite(value):
        return cap
    return value


def _window_score(metrics: dict[str, float], *, oos: bool) -> float:
    """Rank candidates by expected return, then frequency, edge, and drawdown."""
    ret = float(metrics.get("net_return_pct", 0.0))
    trades = float(metrics.get("total_trades", 0.0))
    expectancy = float(metrics.get("expectancy_r", 0.0))
    pf = _finite(float(metrics.get("profit_factor", 0.0)), 8.0)
    dd = float(metrics.get("max_drawdown_pct", 0.0))
    eff = float(metrics.get("exit_efficiency", 0.0))
    trade_weight = 0.45 if oos else 0.14
    return (
        ret
        + trade_weight * min(trades, 35.0)
        + 4.0 * expectancy
        + 0.38 * min(pf, 8.0)
        + 1.35 * eff
        - 0.30 * dd
    )


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _trade_to_dict(trade: Any) -> dict[str, Any]:
    entry_time = trade.entry_time
    direction = _enum_text(trade.direction).lower()
    grade = _enum_text(getattr(trade, "setup_grade", ""))
    r_value = trade.economic_r_multiple
    return {
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "direction": direction,
        "entry_time": entry_time.isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "entry_hour_utc": int(entry_time.hour),
        "entry_weekday_utc": int(entry_time.weekday()),
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "net_pnl": trade.net_pnl,
        "r_multiple": r_value,
        "geometric_r_multiple": trade.r_multiple,
        "realized_r_multiple": trade.realized_r_multiple,
        "commission": trade.commission,
        "funding_paid": trade.funding_paid,
        "bars_held": trade.bars_held,
        "setup_grade": grade,
        "exit_reason": trade.exit_reason,
        "confluences_used": list(trade.confluences_used or []),
        "confluence_count": len(trade.confluences_used or []),
        "confirmation_type": trade.confirmation_type,
        "entry_method": trade.entry_method,
        "mae_r": trade.mae_r,
        "mfe_r": trade.mfe_r,
        "signal_variant": trade.signal_variant,
    }


def _evaluate_payload(payload: dict[str, Any], windows: dict[str, tuple[str, str]]) -> dict[str, Any]:
    _quiet_logging()
    config = BreakoutConfig.from_dict(deepcopy(payload))
    result: dict[str, Any] = {}
    for window_name, (start, end) in windows.items():
        if window_name == "oos":
            bt_config = build_backtest_config_from_profile(
                profile=LIVE_PARITY_PROFILE,
                symbols=SYMBOLS,
                start_date=date.fromisoformat(IS_START),
                end_date=date.fromisoformat(end),
            )
            split = run_split_continuation(
                config,
                bt_config,
                split_date=date.fromisoformat(start),
                data_dir=DATA_DIR,
                strategy_type="breakout",
            )
            backtest = split.out_of_sample
        else:
            bt_config = build_backtest_config_from_profile(
                profile=LIVE_PARITY_PROFILE,
                symbols=SYMBOLS,
                start_date=date.fromisoformat(start),
                end_date=date.fromisoformat(end),
            )
            backtest = run(config, bt_config, data_dir=DATA_DIR, strategy_type="breakout")
        metrics = metrics_to_dict(backtest.metrics)
        result[window_name] = {
            "metrics": _metrics_subset(metrics),
            "score": _window_score(metrics, oos=window_name == "oos"),
            "trades": [_trade_to_dict(trade) for trade in backtest.trades],
            "terminal_marks": len(backtest.terminal_marks),
            "stateful_continuation": window_name == "oos",
        }
    return result


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    try:
        windows = task.get("windows") or {"oos": (OOS_START, OOS_END)}
        evaluation = _evaluate_payload(task["payload"], windows)
        return {**task, "evaluation": evaluation, "error": ""}
    except Exception as exc:  # Keep long sweeps alive; surface errors in outputs.
        return {**task, "evaluation": {}, "error": repr(exc)}


def _empty_bucket() -> dict[str, float]:
    return {"n": 0, "wins": 0, "net_pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0, "total_r": 0.0}


def _finalize_buckets(buckets: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for key, bucket in sorted(buckets.items()):
        n = bucket["n"]
        gross_loss = bucket["gross_loss"]
        finalized[key] = {
            **bucket,
            "win_rate": 100.0 * bucket["wins"] / n if n else 0.0,
            "avg_r": bucket["total_r"] / n if n else 0.0,
            "profit_factor": bucket["gross_win"] / gross_loss if gross_loss else math.inf,
        }
    return finalized


def _add_bucket(buckets: dict[str, dict[str, float]], key: str, trade: dict[str, Any]) -> None:
    bucket = buckets[key]
    pnl = float(trade["net_pnl"])
    r_value = float(trade.get("r_multiple") or 0.0)
    bucket["n"] += 1
    bucket["net_pnl"] += pnl
    bucket["total_r"] += r_value
    if pnl >= 0.0:
        bucket["wins"] += 1
        bucket["gross_win"] += pnl
    else:
        bucket["gross_loss"] += abs(pnl)


def _trade_autopsy(trades: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(float(t["net_pnl"]) for t in trades)
    losers = sorted((t for t in trades if float(t["net_pnl"]) < 0.0), key=lambda t: float(t["net_pnl"]))
    winners = sorted((t for t in trades if float(t["net_pnl"]) > 0.0), key=lambda t: float(t["net_pnl"]), reverse=True)
    gross_loss = sum(abs(float(t["net_pnl"])) for t in losers)
    gross_win = sum(float(t["net_pnl"]) for t in winners)

    by_symbol = defaultdict(_empty_bucket)
    by_symbol_direction = defaultdict(_empty_bucket)
    by_confirmation = defaultdict(_empty_bucket)
    by_variant = defaultdict(_empty_bucket)
    by_exit = defaultdict(_empty_bucket)
    by_hour = defaultdict(_empty_bucket)
    by_weekday = defaultdict(_empty_bucket)
    by_confluence = defaultdict(_empty_bucket)
    by_duration = defaultdict(_empty_bucket)

    for trade in trades:
        symbol = str(trade["symbol"]).upper()
        direction = str(trade["direction"]).lower()
        duration = int(trade.get("bars_held") or 0)
        duration_bucket = "0-1" if duration <= 1 else "2-4" if duration <= 4 else "5-8" if duration <= 8 else "9+"
        for buckets, key in (
            (by_symbol, symbol),
            (by_symbol_direction, f"{symbol}_{direction}"),
            (by_confirmation, str(trade.get("confirmation_type") or "")),
            (by_variant, str(trade.get("signal_variant") or "")),
            (by_exit, str(trade.get("exit_reason") or "")),
            (by_hour, f"{int(trade.get('entry_hour_utc') or 0):02d}h"),
            (by_weekday, str(trade.get("entry_weekday_utc"))),
            (by_confluence, str(trade.get("confluence_count"))),
            (by_duration, duration_bucket),
        ):
            _add_bucket(buckets, key, trade)

    right_then_stopped = [
        t for t in losers
        if float(t.get("mfe_r") or 0.0) >= 0.5 and float(t.get("r_multiple") or 0.0) < 0.0
    ]
    same_bar_losses = [t for t in losers if int(t.get("bars_held") or 0) <= 0]
    top_loss_values = [abs(float(t["net_pnl"])) for t in losers[:5]]

    return {
        "net_pnl": total,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "trade_count": len(trades),
        "winner_count": len(winners),
        "loser_count": len(losers),
        "worst_trades": losers[:8],
        "best_trades": winners[:8],
        "same_bar_losses": same_bar_losses[:8],
        "right_then_stopped": right_then_stopped[:8],
        "loss_concentration": {
            "top1_loss_share_of_gross_loss": top_loss_values[0] / gross_loss if gross_loss and top_loss_values else 0.0,
            "top2_loss_share_of_gross_loss": sum(top_loss_values[:2]) / gross_loss if gross_loss else 0.0,
            "top3_loss_share_of_gross_loss": sum(top_loss_values[:3]) / gross_loss if gross_loss else 0.0,
            "top1_loss_share_of_abs_net": top_loss_values[0] / abs(total) if total and top_loss_values else None,
        },
        "by_symbol": _finalize_buckets(by_symbol),
        "by_symbol_direction": _finalize_buckets(by_symbol_direction),
        "by_confirmation": _finalize_buckets(by_confirmation),
        "by_signal_variant": _finalize_buckets(by_variant),
        "by_exit_reason": _finalize_buckets(by_exit),
        "by_entry_hour": _finalize_buckets(by_hour),
        "by_weekday": _finalize_buckets(by_weekday),
        "by_confluence_count": _finalize_buckets(by_confluence),
        "by_duration_bucket": _finalize_buckets(by_duration),
    }


def _trade_signature(trade: dict[str, Any]) -> tuple[str, str, str]:
    return (trade["entry_time"][:16], str(trade["symbol"]).upper(), str(trade["direction"]).lower())


def _compare_trade_sets(base_trades: list[dict[str, Any]], candidate_trades: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_sig = {_trade_signature(t): t for t in base_trades}
    cand_by_sig = {_trade_signature(t): t for t in candidate_trades}
    missed = [base_by_sig[key] for key in sorted(set(base_by_sig) - set(cand_by_sig))]
    added = [cand_by_sig[key] for key in sorted(set(cand_by_sig) - set(base_by_sig))]
    common = sorted(set(base_by_sig) & set(cand_by_sig))
    common_delta = [
        {
            "signature": key,
            "base_net_pnl": base_by_sig[key]["net_pnl"],
            "candidate_net_pnl": cand_by_sig[key]["net_pnl"],
            "delta": cand_by_sig[key]["net_pnl"] - base_by_sig[key]["net_pnl"],
        }
        for key in common
        if abs(cand_by_sig[key]["net_pnl"] - base_by_sig[key]["net_pnl"]) > 1e-9
    ]
    return {
        "missed_base_trades": missed,
        "added_candidate_trades": added,
        "common_pnl_deltas": common_delta,
        "missed_base_net_pnl": sum(float(t["net_pnl"]) for t in missed),
        "added_candidate_net_pnl": sum(float(t["net_pnl"]) for t in added),
        "common_net_pnl_delta": sum(float(item["delta"]) for item in common_delta),
    }


def _load_archive_cumulative_mutations() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not ARCHIVE_MANIFEST_PATH.exists():
        return {}, {}, {}
    manifest = _load_json(ARCHIVE_MANIFEST_PATH)
    cumulative: dict[str, Any] = {}
    round2: dict[str, Any] = {}
    round3_delta: dict[str, Any] = {}
    for entry in sorted(manifest.get("rounds", []), key=lambda item: int(item["round"])):
        mutations = dict(entry.get("mutations") or {})
        before = dict(cumulative)
        cumulative.update(mutations)
        if int(entry["round"]) == 2:
            round2 = dict(cumulative)
        if int(entry["round"]) == 3:
            round3_delta = {key: value for key, value in mutations.items() if before.get(key) != value}
    return dict(cumulative), round2, round3_delta


def _load_current_round2_manifest_mutations() -> dict[str, Any]:
    if not CURRENT_MANIFEST_PATH.exists():
        return {}
    manifest = _load_json(CURRENT_MANIFEST_PATH)
    for entry in manifest.get("rounds", []):
        if int(entry.get("round", 0)) == 2:
            return dict(entry.get("mutations") or {})
    return {}


def _raw_phase6_payload(round1_config: BreakoutConfig) -> dict[str, Any] | None:
    if CURRENT_RAW_PHASE6_PATH.exists():
        return _load_strategy_payload(CURRENT_RAW_PHASE6_PATH)
    if not CURRENT_PHASE_STATE_PATH.exists():
        return None
    state = _load_json(CURRENT_PHASE_STATE_PATH)
    mutations = dict(state.get("cumulative_mutations") or {})
    if not mutations:
        return None
    return apply_mutations(round1_config, mutations).to_dict()


def _make_checkpoint_tasks(
    *,
    pre_payload: dict[str, Any],
    round1_config: BreakoutConfig,
    round2_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = [
        {
            "label": "checkpoint:pre_round1_reconstructed",
            "stage": "checkpoint",
            "group": "checkpoint",
            "mutation_key": "",
            "candidate_value": None,
            "payload": pre_payload,
        }
    ]
    for label, path in (
        ("archive_round2", ARCHIVE_ROUND2_PATH),
        ("archive_round3_current_round1_equivalent", ARCHIVE_ROUND3_PATH),
        ("current_round1", CURRENT_ROUND1_PATH),
        ("current_round2", CURRENT_ROUND2_PATH),
    ):
        if path.exists():
            tasks.append(
                {
                    "label": f"checkpoint:{label}",
                    "stage": "checkpoint",
                    "group": "checkpoint",
                    "mutation_key": "",
                    "candidate_value": None,
                    "payload": _load_strategy_payload(path),
                }
            )
    raw_payload = _raw_phase6_payload(round1_config)
    if raw_payload is not None:
        tasks.append(
            {
                "label": "checkpoint:current_round2_raw_phase6_gate_failed",
                "stage": "checkpoint",
                "group": "checkpoint",
                "mutation_key": "",
                "candidate_value": None,
                "payload": raw_payload,
            }
        )
    if CURRENT_PHASE_STATE_PATH.exists():
        phase_state = _load_json(CURRENT_PHASE_STATE_PATH)
        for phase_key, phase_result in sorted(phase_state.get("phase_results", {}).items(), key=lambda item: int(item[0])):
            mutations = dict(phase_result.get("final_mutations") or {})
            if not mutations:
                payload = round1_config.to_dict()
            else:
                payload = apply_mutations(round1_config, mutations).to_dict()
            tasks.append(
                {
                    "label": f"checkpoint:current_round2_phase_{phase_key}_cumulative",
                    "stage": "checkpoint",
                    "group": "phase_checkpoint",
                    "mutation_key": "",
                    "candidate_value": mutations,
                    "payload": payload,
                }
            )
    tasks.append(
        {
            "label": "checkpoint:current_round2_promoted_copy",
            "stage": "checkpoint",
            "group": "checkpoint",
            "mutation_key": "",
            "candidate_value": None,
            "payload": round2_payload,
        }
    )
    return tasks


def _make_ablation_tasks_from_diffs(
    *,
    final_payload: dict[str, Any],
    diffs: dict[str, dict[str, Any]],
    group: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for key, detail in sorted(diffs.items()):
        if key == "symbols":
            continue
        reset_value = detail["base_value"]
        payload = _set_path(final_payload, key, reset_value)
        tasks.append(
            {
                "label": f"ablation:{group}:remove:{key}",
                "stage": "ablation",
                "group": group,
                "mutation_key": key,
                "base_value": reset_value,
                "candidate_value": detail["candidate_value"],
                "payload": payload,
            }
        )
    return tasks


def _make_ablation_tasks_from_keys(
    *,
    final_payload: dict[str, Any],
    reset_payload: dict[str, Any],
    keys: list[str],
    group: str,
) -> list[dict[str, Any]]:
    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(set(keys)):
        if key == "symbols":
            continue
        base_value = _get_path(reset_payload, key)
        current_value = _get_path(final_payload, key)
        if base_value != current_value:
            diffs[key] = {"base_value": base_value, "candidate_value": current_value}
    return _make_ablation_tasks_from_diffs(final_payload=final_payload, diffs=diffs, group=group)


def _domain_values(key: str, value: Any, *, base_value: Any | None = None) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, bool):
        values.append(not value)
    elif isinstance(value, str):
        if key.startswith("symbol_filter."):
            values.extend(["both", "long_only", "short_only", "disabled"])
        elif key == "exits.time_stop_action":
            values.extend(["close", "reduce"])
    elif isinstance(value, int) and not isinstance(value, bool):
        values.extend(local_perturbation_values(key, value))
        if "bars" in key or "lookback" in key or "age" in key:
            values.extend([2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 60])
        if "max_trades_per_day" in key:
            values.extend([3, 4, 5, 6, 8])
        if "max_consecutive_losses" in key:
            values.extend([1, 2, 3, 4, 5])
    elif isinstance(value, float):
        values.extend(local_perturbation_values(key, value))
        if "risk_pct" in key:
            values.extend([0.004, 0.006, 0.0075, 0.01, 0.015, 0.018, 0.02, 0.024, 0.02625, 0.0277725])
        elif "tp1_frac" in key or "tp2_frac" in key or "runner_frac" in key:
            values.extend([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65])
        elif "tp1_r" in key:
            values.extend([0.6, 0.8, 1.0, 1.2, 1.5])
        elif "tp2_r" in key:
            values.extend([1.5, 1.8, 2.0, 2.2, 2.5, 3.0])
        elif "body_ratio" in key or "relaxed_body_min" in key:
            values.extend([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
        elif "volume_mult" in key or "volume_surge" in key:
            values.extend([1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.5])
        elif "trail_buffer_tight" in key:
            values.extend([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.1575, 0.2])
        elif "trail_activation_r" in key:
            values.extend([0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7])
        elif "trail_buffer_wide" in key or "trail_r_ceiling" in key:
            values.extend([0.8, 1.0, 1.2, 1.5, 2.0, 2.5])
        elif "be_buffer_r" in key:
            values.extend([0.1, 0.2, 0.3, 0.45, 0.525, 0.6])
        elif "quick_exit" in key or "early_lock" in key:
            values.extend([-0.3, -0.2, -0.1, -0.05, 0.0, 0.1, 0.15, 0.2, 0.3, 0.45, 0.55])
        elif "hvn_threshold" in key:
            values.extend([1.0, 1.2, 1.35, 1.4, 1.47, 1.5, 1.65, 1.8])
        elif "lvn_threshold" in key:
            values.extend([0.3, 0.4, 0.5, 0.6, 0.7])
        elif "atr" in key or "room" in key:
            values.extend([0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        elif "adx" in key:
            values.extend([8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0])
    if base_value is not None:
        values.append(base_value)

    deduped: list[Any] = []
    for item in values:
        if item == value:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            if isinstance(item, float) and not item.is_integer():
                continue
            item = int(item)
            if item <= 0:
                continue
        if isinstance(value, float) and isinstance(item, (int, float)):
            item = round(float(item), 6)
            if any(token in key for token in ("frac", "body_ratio", "relaxed_body_min", "value_area")):
                if not (0.0 < item < 1.0):
                    continue
            if "risk_pct" in key and not (0.0 < item <= 0.05):
                continue
            if item <= 0 and "quick_exit_max_r" not in key:
                continue
        if item not in deduped:
            deduped.append(item)
    return deduped


def _make_perturbation_tasks(
    *,
    final_payload: dict[str, Any],
    diffs: dict[str, dict[str, Any]],
    max_per_key: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for key, detail in sorted(diffs.items()):
        current_value = detail["candidate_value"]
        base_value = detail.get("base_value")
        for value in _domain_values(key, current_value, base_value=base_value)[:max_per_key]:
            payload = _set_path(final_payload, key, value)
            tasks.append(
                {
                    "label": f"perturb:{key}:{value}",
                    "stage": "perturbation",
                    "group": "active_cumulative",
                    "mutation_key": key,
                    "base_value": current_value,
                    "candidate_value": value,
                    "payload": payload,
                }
            )
    return tasks


def _task_from_mutations(
    base_payload: dict[str, Any],
    label: str,
    mutations: dict[str, Any],
    thesis: str,
) -> dict[str, Any] | None:
    try:
        cfg = apply_mutations(BreakoutConfig.from_dict(base_payload), mutations)
    except Exception as exc:
        print(f"skip targeted {label}: {exc}")
        return None
    return {
        "label": f"targeted:{label}",
        "stage": "targeted",
        "group": "repair",
        "mutation_key": ",".join(sorted(mutations)),
        "base_value": None,
        "candidate_value": mutations,
        "thesis": thesis,
        "payload": cfg.to_dict(),
    }


def _static_targeted_sets() -> list[tuple[str, dict[str, Any], str]]:
    return [
        ("sol_disabled", {"symbol_filter.sol_direction": "disabled"}, "Remove SOL, a likely fragile high-beta pocket."),
        ("sol_both", {"symbol_filter.sol_direction": "both"}, "Undo current SOL short-only pruning."),
        ("sol_long_only", {"symbol_filter.sol_direction": "long_only"}, "Test if SOL weakness is isolated to shorts."),
        ("eth_long_only", {"symbol_filter.eth_direction": "long_only"}, "Remove ETH shorts without touching BTC/SOL."),
        ("btc_long_only", {"symbol_filter.btc_direction": "long_only"}, "Test if BTC shorts are the OOS drag."),
        ("relaxed_off", {"setup.relaxed_body_enabled": False}, "Remove the relaxed-body branch entirely."),
        ("relaxed_min_045", {"setup.relaxed_body_min": 0.45}, "Require stronger candle body for relaxed setups."),
        ("relaxed_min_050", {"setup.relaxed_body_min": 0.50}, "Strict relaxed-body quality gate."),
        ("relaxed_conf_6", {"setup.relaxed_body_min_confluences": 6}, "Require more confluence for relaxed setups."),
        ("relaxed_room_16", {"setup.relaxed_body_min_room_r": 1.6}, "Require more room for relaxed setups."),
        ("all_relaxed_disabled", {
            "symbol_filter.btc_relaxed_body_direction": "disabled",
            "symbol_filter.eth_relaxed_body_direction": "disabled",
            "symbol_filter.sol_relaxed_body_direction": "disabled",
        }, "Disable all relaxed-body direction pockets."),
        ("all_relaxed_long_only", {
            "symbol_filter.btc_relaxed_body_direction": "long_only",
            "symbol_filter.eth_relaxed_body_direction": "long_only",
            "symbol_filter.sol_relaxed_body_direction": "long_only",
        }, "Keep relaxed-body only with the post-crash long recovery bias."),
        ("model1_vol_13", {"confirmation.model1_min_volume_mult": 1.3}, "Demand stronger breakout-close participation."),
        ("model1_vol_15", {"confirmation.model1_min_volume_mult": 1.5}, "Strict model1 volume gate."),
        ("model1_volume_off", {"confirmation.model1_require_volume": False}, "Test if volume confirmation delayed or distorted OOS entries."),
        ("model2_only", {"confirmation.enable_model1": False, "confirmation.enable_model2": True}, "Remove aggressive close entries, keep retest entries."),
        ("model2_off", {"confirmation.enable_model2": False}, "Remove retest model if it overfit IS."),
        ("strict_retest", {
            "confirmation.retest_require_rejection": True,
            "confirmation.retest_require_volume_decline": True,
            "confirmation.retest_zone_atr": 0.35,
            "confirmation.retest_max_bars": 4,
        }, "Require cleaner retest acceptance."),
        ("model2_break_entry", {"entry.model2_entry_on_close": False, "entry.model2_entry_on_break": True}, "Use stop entries for retests."),
        ("entry_ttl_2", {"entry.max_bars_after_signal": 2}, "Reduce stale signal chasing."),
        ("body_070", {"setup.body_ratio_min": 0.70}, "Tighten core body quality."),
        ("body_075", {"setup.body_ratio_min": 0.75}, "Strict core body quality."),
        ("min_conf_b_1", {"setup.min_confluences_b": 1}, "Remove zero-confluence B setups."),
        ("min_conf_b_2", {"setup.min_confluences_b": 2}, "Require real confluence for B setups."),
        ("room_b_14", {"setup.min_room_r_b": 1.4}, "Improve expected reward room for B setups."),
        ("room_b_16", {"setup.min_room_r_b": 1.6}, "Strict B setup room."),
        ("no_countertrend", {"context.allow_countertrend": False}, "Remove countertrend breakouts."),
        ("h4_adx_15", {"context.h4_adx_threshold": 15.0}, "Require stronger H4 context."),
        ("trail_tight_004", {"trail.trail_buffer_tight": 0.04}, "Undo current round-2 tighter trail change."),
        ("trail_tight_008", {"trail.trail_buffer_tight": 0.08}, "Midpoint tight trail."),
        ("trail_tight_010", {"trail.trail_buffer_tight": 0.10}, "Looser trail to avoid noise stops."),
        ("trail_activation_03_4", {"trail.trail_activation_r": 0.3, "trail.trail_activation_bars": 4}, "Protect OOS winners earlier."),
        ("trail_activation_04_5", {"trail.trail_activation_r": 0.4, "trail.trail_activation_bars": 5}, "Moderate trail timing."),
        ("quick_exit_on", {"exits.quick_exit_enabled": True}, "Cut failed breakouts before full stop."),
        ("quick_exit_4_045_neg005", {
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 4,
            "exits.quick_exit_max_mfe_r": 0.45,
            "exits.quick_exit_max_r": -0.05,
        }, "Early failure handling for low-MFE losers."),
        ("early_lock_055_0", {
            "exits.early_lock_enabled": True,
            "exits.early_lock_mfe_r": 0.55,
            "exits.early_lock_stop_r": 0.0,
        }, "Prevent small winners from reversing."),
        ("early_lock_075_01", {
            "exits.early_lock_enabled": True,
            "exits.early_lock_mfe_r": 0.75,
            "exits.early_lock_stop_r": 0.1,
        }, "Lock only more mature winners."),
        ("tp1_frac_005", {"exits.tp1_frac": 0.05}, "Let winners run more than current round-2."),
        ("tp1_frac_015", {"exits.tp1_frac": 0.15}, "Midpoint TP1 fraction."),
        ("tp1_frac_020", {"exits.tp1_frac": 0.20}, "Undo current TP1 fraction change."),
        ("tp1_r_08", {"exits.tp1_r": 0.8}, "Earlier first partial."),
        ("tp1_r_12", {"exits.tp1_r": 1.2}, "Later first partial."),
        ("be_buffer_015", {"exits.be_buffer_r": 0.15}, "Reduce BE giveback buffer."),
        ("be_buffer_045", {"exits.be_buffer_r": 0.45}, "Wider BE buffer."),
        ("invalidation_deeper_12", {"exits.invalidation_depth_atr": 1.2}, "Avoid shallow invalidation exits."),
        ("structure_trail_off", {"trail.structure_trail_enabled": False}, "Remove structure trail noise."),
        ("hvn_147_zone_42", {"profile.hvn_threshold_pct": 1.47, "balance.max_zone_age_bars": 42}, "Retest gate-failed phase-5/6 structural changes."),
        ("hvn_14", {"profile.hvn_threshold_pct": 1.4}, "Looser HVN threshold from failed phase 5."),
        ("hvn_16", {"profile.hvn_threshold_pct": 1.6}, "Tighter HVN threshold."),
        ("lookback_32", {"profile.lookback_bars": 32}, "Shorter profile memory."),
        ("lookback_42", {"profile.lookback_bars": 42}, "Longer profile memory."),
        ("zone_age_30", {"balance.max_zone_age_bars": 30}, "Moderate zone age."),
        ("zone_age_42", {"balance.max_zone_age_bars": 42}, "Longer zone persistence."),
        ("min_zone_5", {"balance.min_bars_in_zone": 5}, "More frequent zones than current."),
        ("min_zone_8", {"balance.min_bars_in_zone": 8}, "Higher-quality balance zones."),
        ("reentry_off", {"reentry.enabled": False}, "Remove reentry branch."),
        ("reentry_guarded", {"reentry.cooldown_bars": 6, "reentry.min_confluences_override": 1}, "Allow only guarded reentries."),
        ("max_trades_5", {"limits.max_trades_per_day": 5}, "Recover frequency with current quality rules."),
        ("risk_b_0200", {"risk.risk_pct_b": 0.020}, "Trim B-risk tail losses."),
        ("risk_b_0150", {"risk.risk_pct_b": 0.015}, "Strong B-risk tail mitigation."),
        ("funding_filter", {"filters.funding_filter_enabled": True, "filters.funding_extreme_threshold": 0.001}, "Avoid extreme funding context."),
        ("avoid_15h", {"filters.session_filter_enabled": True, "filters.session_avoid_hours": [15]}, "Diagnostic hour filter for common loss hour."),
        ("avoid_overlap_14_16", {"filters.session_filter_enabled": True, "filters.session_avoid_hours": [14, 15, 16]}, "Diagnostic overlap-session risk filter."),
        ("sol_disabled_model1_vol_13", {"symbol_filter.sol_direction": "disabled", "confirmation.model1_min_volume_mult": 1.3}, "Remove SOL and demand stronger model1 volume."),
        ("sol_disabled_relaxed_strict", {"symbol_filter.sol_direction": "disabled", "setup.relaxed_body_min": 0.45}, "Remove SOL and tighten relaxed-body branch."),
        ("eth_long_sol_disabled", {"symbol_filter.eth_direction": "long_only", "symbol_filter.sol_direction": "disabled"}, "BTC plus ETH-long-only defensive set."),
        ("quality_guard_bundle", {
            "setup.min_confluences_b": 1,
            "setup.min_room_r_b": 1.4,
            "confirmation.model1_min_volume_mult": 1.3,
        }, "Broad B-quality guard without symbol pruning."),
        ("failure_guard_bundle", {
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 4,
            "exits.quick_exit_max_mfe_r": 0.45,
            "exits.quick_exit_max_r": -0.05,
            "exits.early_lock_enabled": True,
            "exits.early_lock_mfe_r": 0.75,
            "exits.early_lock_stop_r": 0.1,
        }, "Reduce OOS loser and giveback profile."),
    ]


def _make_autopsy_targeted_sets(autopsy: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    targeted: list[tuple[str, dict[str, Any], str]] = []
    for key, stats in autopsy.get("by_symbol_direction", {}).items():
        if stats.get("net_pnl", 0.0) >= 0.0 or stats.get("n", 0) <= 0:
            continue
        symbol, direction = key.split("_", 1)
        field = f"symbol_filter.{symbol.lower()}_direction"
        opposite = "short_only" if direction == "long" else "long_only"
        targeted.append((f"autopsy_{key}_disabled", {field: "disabled"}, f"Disable losing OOS bucket {key}."))
        targeted.append((f"autopsy_{key}_opposite_only", {field: opposite}, f"Keep only the opposite side of losing OOS bucket {key}."))

    for key, stats in autopsy.get("by_signal_variant", {}).items():
        if key == "relaxed_body" and stats.get("net_pnl", 0.0) < 0.0:
            targeted.extend([
                ("autopsy_relaxed_off", {"setup.relaxed_body_enabled": False}, "OOS relaxed-body bucket lost money."),
                ("autopsy_relaxed_strict", {"setup.relaxed_body_min": 0.5, "setup.relaxed_body_min_confluences": 6}, "Strict relaxed-body repair."),
            ])

    for key, stats in autopsy.get("by_confirmation", {}).items():
        if key == "model1_close" and stats.get("net_pnl", 0.0) < 0.0:
            targeted.extend([
                ("autopsy_model1_off", {"confirmation.enable_model1": False}, "OOS model1_close bucket lost money."),
                ("autopsy_model1_strict_volume", {"confirmation.model1_min_volume_mult": 1.5}, "Tighten model1_close volume gate."),
            ])
        if key == "model2_retest" and stats.get("net_pnl", 0.0) < 0.0:
            targeted.extend([
                ("autopsy_model2_off", {"confirmation.enable_model2": False}, "OOS model2_retest bucket lost money."),
                ("autopsy_strict_retest", {
                    "confirmation.retest_require_rejection": True,
                    "confirmation.retest_require_volume_decline": True,
                    "confirmation.retest_zone_atr": 0.35,
                }, "Tighten losing retest bucket."),
            ])

    if autopsy.get("same_bar_losses"):
        targeted.extend([
            ("autopsy_same_bar_stop_wider", {"stops.min_stop_atr": 1.0, "stops.buffer_pct": 0.0015}, "Same-bar OOS losses suggest stop/friction sensitivity."),
            ("autopsy_same_bar_quality", {"setup.min_confluences_b": 1, "entry.max_bars_after_signal": 2}, "Reduce weak and stale entries behind same-bar losses."),
        ])

    if autopsy.get("right_then_stopped"):
        targeted.extend([
            ("autopsy_right_then_stopped_lock", {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.55,
                "exits.early_lock_stop_r": 0.05,
            }, "Trades moved favorably then closed red."),
            ("autopsy_right_then_stopped_trail", {"trail.trail_activation_r": 0.3, "trail.trail_activation_bars": 4}, "Earlier trail for reversals after MFE."),
        ])

    for key, stats in autopsy.get("by_entry_hour", {}).items():
        if stats.get("n", 0) >= 2 and stats.get("net_pnl", 0.0) < 0.0:
            hour = int(str(key).replace("h", ""))
            targeted.append((f"autopsy_avoid_{hour:02d}h", {"filters.session_filter_enabled": True, "filters.session_avoid_hours": [hour]}, f"Diagnostic avoidance of losing OOS hour {key}."))
    return targeted


def _make_targeted_tasks(final_payload: dict[str, Any], autopsy: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for label, mutations, thesis in _static_targeted_sets() + _make_autopsy_targeted_sets(autopsy):
        task = _task_from_mutations(final_payload, label, mutations, thesis)
        if task is not None:
            tasks.append(task)
    return tasks


def _make_second_phase_tasks(seed_payload: dict[str, Any], current_round2_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Trend-reference style pass: recover frequency from the first repair seed."""
    tasks: list[dict[str, Any]] = [
        {
            "label": "second_phase_checkpoint:current_round2",
            "stage": "second_phase_checkpoint",
            "group": "checkpoint",
            "mutation_key": "",
            "base_value": None,
            "candidate_value": None,
            "thesis": "Current promoted round-2 baseline for the seeded second phase.",
            "payload": current_round2_payload,
        },
        {
            "label": "second_phase_checkpoint:first_phase_recommended",
            "stage": "second_phase_checkpoint",
            "group": "checkpoint",
            "mutation_key": "",
            "base_value": None,
            "candidate_value": None,
            "thesis": "Seed from the first OOS repair pass.",
            "payload": seed_payload,
        },
    ]

    single_sets: list[tuple[str, dict[str, Any], str]] = []
    for max_trades in [3, 4, 5, 6, 8]:
        single_sets.append((f"max_trades_{max_trades}", {"limits.max_trades_per_day": max_trades}, "Frequency frontier for daily trade cap."))
    for ttl in [1, 2, 3, 4, 5]:
        single_sets.append((f"entry_ttl_{ttl}", {"entry.max_bars_after_signal": ttl}, "Stale-signal/frequency frontier."))
    for body in [0.55, 0.60, 0.65, 0.70, 0.75]:
        single_sets.append((f"body_{body:.2f}", {"setup.body_ratio_min": body}, "Core breakout body quality frontier."))
    for relaxed in [0.35, 0.40, 0.45, 0.50, 0.55]:
        single_sets.append((f"relaxed_min_{relaxed:.2f}", {"setup.relaxed_body_min": relaxed}, "Relaxed-body quality frontier."))
    for conf_b in [0, 1, 2, 3]:
        single_sets.append((f"min_conf_b_{conf_b}", {"setup.min_confluences_b": conf_b}, "B-setup quality/frequency frontier."))
    for room_b in [0.8, 1.0, 1.2, 1.4, 1.6]:
        single_sets.append((f"room_b_{room_b:.1f}", {"setup.min_room_r_b": room_b}, "B-setup reward-room frontier."))
    for vol in [1.0, 1.1, 1.2, 1.3, 1.5]:
        single_sets.append((f"model1_vol_{vol:.1f}", {"confirmation.model1_min_volume_mult": vol}, "Breakout-close participation frontier."))
    for zone_age in [24, 30, 36, 42, 48]:
        single_sets.append((f"zone_age_{zone_age}", {"balance.max_zone_age_bars": zone_age}, "Balance-zone recency/frequency frontier."))
    for hvn in [1.2, 1.35, 1.47, 1.6, 1.8]:
        single_sets.append((f"hvn_{hvn:.2f}", {"profile.hvn_threshold_pct": hvn}, "HVN structural threshold frontier."))
    for trail in [0.04, 0.06, 0.08, 0.10, 0.12]:
        single_sets.append((f"trail_tight_{trail:.2f}", {"trail.trail_buffer_tight": trail}, "Noise-stop versus protection frontier."))
    for tp1_frac in [0.05, 0.10, 0.15, 0.20, 0.25]:
        single_sets.append((f"tp1_frac_{tp1_frac:.2f}", {"exits.tp1_frac": tp1_frac}, "Partial-taking frontier."))
    for risk_b in [0.012, 0.015, 0.018, 0.020, 0.024]:
        single_sets.append((f"risk_b_{risk_b:.3f}", {"risk.risk_pct_b": risk_b}, "B-risk frontier."))

    single_sets.extend(
        [
            ("sol_disabled", {"symbol_filter.sol_direction": "disabled"}, "Retest SOL removal after first repair seed."),
            ("sol_long_only", {"symbol_filter.sol_direction": "long_only"}, "Retain SOL longs while blocking SOL shorts."),
            ("sol_both", {"symbol_filter.sol_direction": "both"}, "Recover SOL frequency if the first pass over-pruned it."),
            ("btc_eth_only", {"symbol_filter.sol_direction": "disabled", "symbol_filter.btc_direction": "both", "symbol_filter.eth_direction": "both"}, "BTC/ETH-only frequency recovery."),
            ("eth_long_sol_disabled", {"symbol_filter.eth_direction": "long_only", "symbol_filter.sol_direction": "disabled"}, "Defensive ETH-long plus no-SOL test."),
            ("all_relaxed_disabled", {
                "symbol_filter.btc_relaxed_body_direction": "disabled",
                "symbol_filter.eth_relaxed_body_direction": "disabled",
                "symbol_filter.sol_relaxed_body_direction": "disabled",
            }, "Check whether relaxed-body pockets still overfit after repair."),
            ("all_relaxed_both", {
                "symbol_filter.btc_relaxed_body_direction": "both",
                "symbol_filter.eth_relaxed_body_direction": "both",
                "symbol_filter.sol_relaxed_body_direction": "both",
            }, "Recover relaxed-body frequency from the repair seed."),
            ("model1_only", {"confirmation.enable_model1": True, "confirmation.enable_model2": False}, "Breakout-close only after first repair."),
            ("model2_only", {"confirmation.enable_model1": False, "confirmation.enable_model2": True}, "Retest-only after first repair."),
            ("strict_retest", {
                "confirmation.retest_require_rejection": True,
                "confirmation.retest_require_volume_decline": True,
                "confirmation.retest_zone_atr": 0.35,
                "confirmation.retest_max_bars": 4,
            }, "Cleaner retest acceptance after first repair."),
            ("quick_exit_guard", {
                "exits.quick_exit_enabled": True,
                "exits.quick_exit_bars": 4,
                "exits.quick_exit_max_mfe_r": 0.45,
                "exits.quick_exit_max_r": -0.05,
            }, "Failed-breakout guard after first repair."),
            ("early_lock_guard", {
                "exits.early_lock_enabled": True,
                "exits.early_lock_mfe_r": 0.75,
                "exits.early_lock_stop_r": 0.1,
            }, "Winner giveback guard after first repair."),
            ("reentry_off", {"reentry.enabled": False}, "Remove reentry churn after first repair."),
            ("funding_filter", {"filters.funding_filter_enabled": True, "filters.funding_extreme_threshold": 0.001}, "Avoid adverse funding extremes after first repair."),
            ("session_overlap_avoid", {"filters.session_filter_enabled": True, "filters.session_avoid_hours": [14, 15, 16]}, "Diagnostic overlap-session guard after first repair."),
        ]
    )

    for label, mutations, thesis in single_sets:
        task = _task_from_mutations(seed_payload, f"second_phase:{label}", mutations, thesis)
        if task is not None:
            task["label"] = f"second_phase:{label}"
            task["stage"] = "second_phase"
            task["group"] = "frequency_recovery"
            tasks.append(task)

    guards: list[tuple[str, dict[str, Any]]] = [
        ("sol_disabled", {"symbol_filter.sol_direction": "disabled"}),
        ("sol_long_only", {"symbol_filter.sol_direction": "long_only"}),
        ("body_065", {"setup.body_ratio_min": 0.65}),
        ("body_070", {"setup.body_ratio_min": 0.70}),
        ("conf_b_1", {"setup.min_confluences_b": 1}),
        ("conf_b_2", {"setup.min_confluences_b": 2}),
        ("risk_b_018", {"risk.risk_pct_b": 0.018}),
        ("risk_b_020", {"risk.risk_pct_b": 0.020}),
        ("quick_exit", {
            "exits.quick_exit_enabled": True,
            "exits.quick_exit_bars": 4,
            "exits.quick_exit_max_mfe_r": 0.45,
            "exits.quick_exit_max_r": -0.05,
        }),
        ("early_lock", {
            "exits.early_lock_enabled": True,
            "exits.early_lock_mfe_r": 0.75,
            "exits.early_lock_stop_r": 0.1,
        }),
    ]
    for max_trades in [4, 5, 6]:
        for guard_label, guard in guards:
            mutations = {"limits.max_trades_per_day": max_trades, **guard}
            task = _task_from_mutations(
                seed_payload,
                f"second_phase:max_trades_{max_trades}_{guard_label}",
                mutations,
                "Recover frequency with one guardrail.",
            )
            if task is not None:
                task["label"] = f"second_phase:max_trades_{max_trades}_{guard_label}"
                task["stage"] = "second_phase"
                task["group"] = "frequency_recovery_combo"
                tasks.append(task)

    combo_sets = [
        ("freq_sol_disabled_body065", {"limits.max_trades_per_day": 5, "symbol_filter.sol_direction": "disabled", "setup.body_ratio_min": 0.65}),
        ("freq_sol_long_body065", {"limits.max_trades_per_day": 5, "symbol_filter.sol_direction": "long_only", "setup.body_ratio_min": 0.65}),
        ("freq_conf1_room12", {"limits.max_trades_per_day": 5, "setup.min_confluences_b": 1, "setup.min_room_r_b": 1.2}),
        ("freq_conf1_model1vol13", {"limits.max_trades_per_day": 5, "setup.min_confluences_b": 1, "confirmation.model1_min_volume_mult": 1.3}),
        ("freq_quick_sol_disabled", {"limits.max_trades_per_day": 5, "symbol_filter.sol_direction": "disabled", "exits.quick_exit_enabled": True, "exits.quick_exit_bars": 4, "exits.quick_exit_max_mfe_r": 0.45, "exits.quick_exit_max_r": -0.05}),
        ("freq_hvn147_zone42", {"limits.max_trades_per_day": 5, "profile.hvn_threshold_pct": 1.47, "balance.max_zone_age_bars": 42}),
        ("freq_trail008_tp015", {"limits.max_trades_per_day": 5, "trail.trail_buffer_tight": 0.08, "exits.tp1_frac": 0.15}),
        ("freq_model2_strict_sol_disabled", {"limits.max_trades_per_day": 5, "symbol_filter.sol_direction": "disabled", "confirmation.enable_model1": False, "confirmation.enable_model2": True, "confirmation.retest_require_rejection": True}),
    ]
    for label, mutations in combo_sets:
        task = _task_from_mutations(seed_payload, f"second_phase:{label}", mutations, "Second-order recovery combo after OOS weakness review.")
        if task is not None:
            task["label"] = f"second_phase:{label}"
            task["stage"] = "second_phase"
            task["group"] = "frequency_recovery_combo"
            tasks.append(task)

    return tasks


def _new_process_pool(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max_workers)


def _payload_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _run_oos_tasks(
    tasks: list[dict[str, Any]],
    *,
    max_workers: int,
    executor_kind: str,
    process_batch_size: int,
    seen_payloads: set[str] | None = None,
    label: str,
) -> list[dict[str, Any]]:
    seen_payloads = seen_payloads if seen_payloads is not None else set()
    unique_tasks: list[dict[str, Any]] = []
    for task in tasks:
        key = _payload_key(task["payload"])
        if key in seen_payloads:
            continue
        seen_payloads.add(key)
        task = dict(task)
        task["windows"] = {"oos": (OOS_START, OOS_END)}
        unique_tasks.append(task)

    print(
        f"evaluating {label} OOS candidates: {len(unique_tasks)} tasks "
        f"with {max_workers} {executor_kind} workers",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    if executor_kind == "sequential":
        for idx, task in enumerate(unique_tasks, start=1):
            item = _worker(task)
            item.pop("windows", None)
            results.append(item)
            if idx % 10 == 0 or idx == len(unique_tasks):
                errors = sum(1 for result in results if result.get("error"))
                print(f"  {label} OOS progress {idx}/{len(unique_tasks)} errors={errors}", flush=True)
        return results

    if executor_kind == "process":
        completed = 0
        batch_size = max(1, process_batch_size)
        for start in range(0, len(unique_tasks), batch_size):
            batch = unique_tasks[start:start + batch_size]
            batch_end = start + len(batch)
            with _new_process_pool(max_workers) as executor:
                futures = [executor.submit(_worker, task) for task in batch]
                for future in as_completed(futures):
                    item = future.result()
                    item.pop("windows", None)
                    results.append(item)
                    completed += 1
                    if completed % 10 == 0 or completed == len(unique_tasks) or completed == batch_end:
                        errors = sum(1 for result in results if result.get("error"))
                        print(f"  {label} OOS progress {completed}/{len(unique_tasks)} errors={errors}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, task) for task in unique_tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            item.pop("windows", None)
            results.append(item)
            if idx % 10 == 0 or idx == len(unique_tasks):
                errors = sum(1 for result in results if result.get("error"))
                print(f"  {label} OOS progress {idx}/{len(unique_tasks)} errors={errors}", flush=True)
    return results


def _valid_oos(result: dict[str, Any]) -> bool:
    return not result.get("error") and "oos" in result.get("evaluation", {})


def _add_deltas(results: list[dict[str, Any]]) -> None:
    current = next((item for item in results if item["label"] == "checkpoint:current_round2"), None)
    if current is None:
        current = next((item for item in results if item["label"] == "checkpoint:current_round2_promoted_copy"), None)
    if current is None or not _valid_oos(current):
        return
    current_metrics = current["evaluation"]["oos"]["metrics"]
    current_score = current["evaluation"]["oos"]["score"]
    for result in results:
        if not _valid_oos(result):
            continue
        metrics = result["evaluation"]["oos"]["metrics"]
        result["oos_score_delta_vs_current_round2"] = result["evaluation"]["oos"]["score"] - current_score
        result["oos_net_return_delta_vs_current_round2"] = metrics["net_return_pct"] - current_metrics["net_return_pct"]
        result["oos_trades_delta_vs_current_round2"] = metrics["total_trades"] - current_metrics["total_trades"]


def _evaluate_is_full_for_top(
    results: list[dict[str, Any]],
    *,
    max_workers: int,
    executor_kind: str,
    process_batch_size: int,
    top_full: int,
) -> None:
    valid = [item for item in results if _valid_oos(item)]
    current = next(item for item in valid if item["label"] in {"checkpoint:current_round2", "checkpoint:current_round2_promoted_copy"})
    current_oos = current["evaluation"]["oos"]["metrics"]
    checkpoint_labels = {item["label"] for item in valid if item.get("stage") == "checkpoint"}
    improving_labels = {
        item["label"]
        for item in valid
        if item["evaluation"]["oos"]["metrics"]["net_return_pct"] > current_oos["net_return_pct"]
    }
    ranked_by_return = sorted(
        valid,
        key=lambda item: (
            item["evaluation"]["oos"]["metrics"]["net_return_pct"],
            item["evaluation"]["oos"]["metrics"]["total_trades"],
            item["evaluation"]["oos"]["metrics"]["expectancy_r"],
        ),
        reverse=True,
    )
    ranked_by_score = sorted(valid, key=lambda item: item["evaluation"]["oos"]["score"], reverse=True)
    full_labels = checkpoint_labels | improving_labels
    full_labels.update(item["label"] for item in ranked_by_return[:top_full])
    full_labels.update(item["label"] for item in ranked_by_score[:top_full])

    tasks = [
        {**item, "windows": {"is": (IS_START, IS_END), "full": (IS_START, FULL_END)}}
        for item in valid
        if item["label"] in full_labels
        and not all(window in item.get("evaluation", {}) for window in ("is", "full"))
    ]
    print(
        f"evaluating IS/full for {len(tasks)} checkpoint/top/improving candidates "
        f"with {max_workers} {executor_kind} workers",
        flush=True,
    )
    full_by_label: dict[str, dict[str, Any]] = {}
    if executor_kind == "sequential":
        for idx, task in enumerate(tasks, start=1):
            item = _worker(task)
            if not item.get("error"):
                full_by_label[item["label"]] = item["evaluation"]
            if idx % 10 == 0 or idx == len(tasks):
                print(f"  IS/full progress {idx}/{len(tasks)}", flush=True)
    elif executor_kind == "process":
        completed = 0
        batch_size = max(1, process_batch_size)
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            batch_end = start + len(batch)
            with _new_process_pool(max_workers) as executor:
                futures = [executor.submit(_worker, task) for task in batch]
                for future in as_completed(futures):
                    item = future.result()
                    if not item.get("error"):
                        full_by_label[item["label"]] = item["evaluation"]
                    completed += 1
                    if completed % 10 == 0 or completed == len(tasks) or completed == batch_end:
                        print(f"  IS/full progress {completed}/{len(tasks)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                item = future.result()
                if not item.get("error"):
                    full_by_label[item["label"]] = item["evaluation"]
                if idx % 10 == 0 or idx == len(tasks):
                    print(f"  IS/full progress {idx}/{len(tasks)}", flush=True)

    for item in results:
        if item["label"] in full_by_label:
            item["evaluation"].update(full_by_label[item["label"]])


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "stage",
        "group",
        "mutation_key",
        "candidate_value",
        "thesis",
        "error",
        "oos_score",
        "oos_net_return_pct",
        "oos_total_trades",
        "oos_win_rate",
        "oos_expectancy_r",
        "oos_profit_factor",
        "oos_max_drawdown_pct",
        "oos_exit_efficiency",
        "is_score",
        "is_net_return_pct",
        "is_total_trades",
        "is_win_rate",
        "is_expectancy_r",
        "is_profit_factor",
        "is_max_drawdown_pct",
        "full_score",
        "full_net_return_pct",
        "full_total_trades",
        "full_win_rate",
        "full_expectancy_r",
        "full_profit_factor",
        "full_max_drawdown_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row: dict[str, Any] = {
                "label": item["label"],
                "stage": item.get("stage", ""),
                "group": item.get("group", ""),
                "mutation_key": item.get("mutation_key", ""),
                "candidate_value": json.dumps(item.get("candidate_value"), default=str),
                "thesis": item.get("thesis", ""),
                "error": item.get("error", ""),
            }
            for window in ("oos", "is", "full"):
                if window not in item.get("evaluation", {}):
                    continue
                row[f"{window}_score"] = item["evaluation"][window].get("score")
                metrics = item["evaluation"][window]["metrics"]
                for metric in (
                    "net_return_pct",
                    "total_trades",
                    "win_rate",
                    "expectancy_r",
                    "profit_factor",
                    "max_drawdown_pct",
                    "exit_efficiency",
                ):
                    col = f"{window}_{metric}"
                    if col in fieldnames:
                        row[col] = metrics.get(metric)
            writer.writerow(row)


def _compact_result(item: dict[str, Any], *, include_payload: bool = False) -> dict[str, Any]:
    payload = {
        "label": item["label"],
        "stage": item.get("stage"),
        "group": item.get("group"),
        "mutation_key": item.get("mutation_key"),
        "candidate_value": item.get("candidate_value"),
        "thesis": item.get("thesis"),
        "error": item.get("error", ""),
    }
    if _valid_oos(item):
        payload["oos_metrics"] = item["evaluation"]["oos"]["metrics"]
        payload["oos_score"] = item["evaluation"]["oos"]["score"]
    if "is" in item.get("evaluation", {}):
        payload["is_metrics"] = item["evaluation"]["is"]["metrics"]
        payload["is_score"] = item["evaluation"]["is"]["score"]
    if "full" in item.get("evaluation", {}):
        payload["full_metrics"] = item["evaluation"]["full"]["metrics"]
        payload["full_score"] = item["evaluation"]["full"]["score"]
    if include_payload:
        payload["payload"] = item["payload"]
    return payload


def _select_recommendation(results: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    valid = [item for item in results if _valid_oos(item) and "is" in item.get("evaluation", {})]
    current = next(item for item in valid if item["label"] in {"checkpoint:current_round2", "checkpoint:current_round2_promoted_copy"})
    current_oos = current["evaluation"]["oos"]["metrics"]
    current_is = current["evaluation"]["is"]["metrics"]

    def combined_score(item: dict[str, Any]) -> float:
        oos = item["evaluation"]["oos"]
        is_eval = item["evaluation"]["is"]
        full = item["evaluation"].get("full", {})
        oos_trades = oos["metrics"].get("total_trades", 0.0)
        return (
            0.62 * oos["score"]
            + 0.33 * is_eval["score"]
            + 0.05 * float(full.get("score", 0.0))
            + 0.08 * oos_trades
        )

    strict: list[dict[str, Any]] = []
    relaxed: list[dict[str, Any]] = []
    for item in valid:
        if item.get("stage") == "checkpoint":
            continue
        oos = item["evaluation"]["oos"]["metrics"]
        is_metrics = item["evaluation"]["is"]["metrics"]
        improves_oos = oos["net_return_pct"] > current_oos["net_return_pct"]
        no_frequency_loss = oos["total_trades"] >= current_oos["total_trades"]
        is_ok = is_metrics["net_return_pct"] >= current_is["net_return_pct"] * 0.90
        pf_ok = is_metrics["profit_factor"] >= max(1.5, current_is["profit_factor"] * 0.55)
        if improves_oos and no_frequency_loss and is_ok and pf_ok:
            strict.append(item)
        elif improves_oos and oos["total_trades"] >= max(1.0, current_oos["total_trades"] - 1.0) and is_metrics["net_return_pct"] >= current_is["net_return_pct"] * 0.80:
            relaxed.append(item)
    if strict:
        return max(strict, key=combined_score), "strict: OOS return improves, OOS frequency is preserved or better, and IS deterioration is under 10%."
    if relaxed:
        return max(relaxed, key=combined_score), "relaxed: OOS improves with at most one fewer OOS trade and IS remains within 20%."
    fallback = max(valid, key=lambda item: (
        item["evaluation"]["oos"]["metrics"]["net_return_pct"],
        item["evaluation"]["oos"]["metrics"]["total_trades"],
        item["evaluation"]["is"]["metrics"]["net_return_pct"],
    ))
    return fallback, "fallback: no candidate satisfied the IS preservation and OOS frequency constraints."


def _summarize_acceptance_risk(results: list[dict[str, Any]]) -> dict[str, Any]:
    current = next(item for item in results if item["label"] in {"checkpoint:current_round2", "checkpoint:current_round2_promoted_copy"} and _valid_oos(item))
    current_oos = current["evaluation"]["oos"]["metrics"]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        if item.get("stage") != "ablation" or not _valid_oos(item):
            continue
        metrics = item["evaluation"]["oos"]["metrics"]
        improvement = metrics["net_return_pct"] - current_oos["net_return_pct"]
        if improvement <= 0.0:
            continue
        by_group[str(item.get("group", ""))].append(
            {
                "label": item["label"],
                "mutation_key": item.get("mutation_key"),
                "reset_to": item.get("base_value"),
                "removed_value": item.get("candidate_value"),
                "oos_metrics": metrics,
                "oos_net_return_delta": improvement,
                "oos_trade_delta": metrics["total_trades"] - current_oos["total_trades"],
            }
        )
    for values in by_group.values():
        values.sort(key=lambda item: (item["oos_net_return_delta"], item["oos_trade_delta"]), reverse=True)
    return dict(by_group)


def _write_strategy_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"strategy": payload}, indent=2, sort_keys=True), encoding="utf-8")


def _diagnosis(summary: dict[str, Any]) -> dict[str, Any]:
    checkpoints = summary["checkpoint_metrics"]
    current = checkpoints["checkpoint:current_round2"]
    is_metrics = current["is"]
    oos_metrics = current["oos"]
    autopsy = summary["current_round2_oos_trade_autopsy"]
    loss_conc = autopsy["loss_concentration"]
    acceptance = summary["acceptance_risk"]
    high_loss_concentration = loss_conc["top2_loss_share_of_gross_loss"] >= 0.70 if autopsy["gross_loss"] else False
    ablation_fix_count = sum(len(items) for items in acceptance.values())
    return {
        "is_oos_gap": {
            "is_net_return_pct": is_metrics["net_return_pct"],
            "oos_net_return_pct": oos_metrics["net_return_pct"],
            "is_win_rate": is_metrics["win_rate"],
            "oos_win_rate": oos_metrics["win_rate"],
            "is_total_trades": is_metrics["total_trades"],
            "oos_total_trades": oos_metrics["total_trades"],
        },
        "edge_case_loss_concentration": {
            "gross_loss": autopsy["gross_loss"],
            "top1_loss_share_of_gross_loss": loss_conc["top1_loss_share_of_gross_loss"],
            "top2_loss_share_of_gross_loss": loss_conc["top2_loss_share_of_gross_loss"],
            "assessment": "losses are concentrated enough to consider targeted mitigation" if high_loss_concentration else "losses are not dominated by only one or two catastrophic trades",
        },
        "accepted_mutation_overfit_signal": {
            "ablation_removals_that_improve_oos": ablation_fix_count,
            "assessment": "one or more accepted mutations look suspect OOS" if ablation_fix_count else "simple one-at-a-time removals did not improve OOS",
        },
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    current = summary["checkpoint_metrics"]["checkpoint:current_round2"]
    diagnosis = summary["diagnosis"]
    recommendation = summary["recommended_candidate"]
    lines = [
        "# Breakout Round 2 OOS Repair",
        "",
        f"Generated at: {summary['generated_at']}",
        "",
        "## Current Round 2",
        "",
        f"- IS {IS_START} to {IS_END}: return {current['is']['net_return_pct']:.2f}%, trades {current['is']['total_trades']:.0f}, win rate {current['is']['win_rate']:.1f}%, PF {current['is']['profit_factor']:.2f}.",
        f"- OOS {OOS_START} to {OOS_END}: return {current['oos']['net_return_pct']:.2f}%, trades {current['oos']['total_trades']:.0f}, win rate {current['oos']['win_rate']:.1f}%, PF {current['oos']['profit_factor']:.2f}.",
        "",
        "## Diagnosis",
        "",
        f"- Loss concentration: {diagnosis['edge_case_loss_concentration']['assessment']}. Top two losses are {diagnosis['edge_case_loss_concentration']['top2_loss_share_of_gross_loss']:.1%} of gross loss.",
        f"- Accepted mutation risk: {diagnosis['accepted_mutation_overfit_signal']['assessment']} ({diagnosis['accepted_mutation_overfit_signal']['ablation_removals_that_improve_oos']} improving OOS removals).",
        "",
        "## Top OOS Candidates",
        "",
        "| label | OOS return | OOS trades | IS return | IS trades | thesis |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summary["top_dual_window_candidates"][:15]:
        oos = item.get("oos_metrics", {})
        is_metrics = item.get("is_metrics", {})
        thesis = str(item.get("thesis") or item.get("candidate_value") or "")[:80]
        lines.append(
            f"| {item['label']} | {oos.get('net_return_pct', 0.0):.2f}% | {oos.get('total_trades', 0.0):.0f} | "
            f"{is_metrics.get('net_return_pct', 0.0):.2f}% | {is_metrics.get('total_trades', 0.0):.0f} | {thesis} |"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"- {summary['recommendation_reason']}",
        f"- First-phase seed: {summary['first_phase_recommended_candidate']['label']} ({summary['first_phase_recommendation_reason']})",
        f"- Second-phase candidates evaluated: {summary['second_phase_candidate_count']}",
        f"- Candidate: {recommendation['label']}",
        f"- OOS: return {recommendation['oos_metrics']['net_return_pct']:.2f}%, trades {recommendation['oos_metrics']['total_trades']:.0f}, win rate {recommendation['oos_metrics']['win_rate']:.1f}%, PF {recommendation['oos_metrics']['profit_factor']:.2f}.",
        f"- IS: return {recommendation['is_metrics']['net_return_pct']:.2f}%, trades {recommendation['is_metrics']['total_trades']:.0f}, win rate {recommendation['is_metrics']['win_rate']:.1f}%, PF {recommendation['is_metrics']['profit_factor']:.2f}.",
        "",
        f"Full results: `{summary['candidate_results_csv']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--executor", choices=("process", "thread", "sequential"), default="process")
    parser.add_argument("--process-batch-size", type=int, default=8)
    parser.add_argument("--max-perturbations-per-key", type=int, default=12)
    parser.add_argument("--top-full", type=int, default=80)
    parser.add_argument("--skip-second-phase", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "breakout" / "round_2_oos_repair" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()

    _quiet_logging()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_config = build_pre_round1_config()
    current_round1 = _load_config(CURRENT_ROUND1_PATH)
    current_round2 = _load_config(CURRENT_ROUND2_PATH)
    pre_payload = pre_config.to_dict()
    round1_payload = current_round1.to_dict()
    round2_payload = current_round2.to_dict()

    archive_cumulative, archive_round2_mutations, archive_round3_delta = _load_archive_cumulative_mutations()
    current_round2_manifest_mutations = _load_current_round2_manifest_mutations()

    active_diffs = _diff_payloads(pre_payload, round2_payload)
    current_round2_latest_diffs = _diff_payloads(round1_payload, round2_payload)

    initial_tasks: list[dict[str, Any]] = []
    initial_tasks.extend(
        _make_checkpoint_tasks(
            pre_payload=pre_payload,
            round1_config=current_round1,
            round2_payload=round2_payload,
        )
    )
    initial_tasks.extend(
        _make_ablation_tasks_from_diffs(
            final_payload=round2_payload,
            diffs=active_diffs,
            group="active_cumulative_vs_pre_round1",
        )
    )
    initial_tasks.extend(
        _make_ablation_tasks_from_diffs(
            final_payload=round2_payload,
            diffs=current_round2_latest_diffs,
            group="current_round2_latest_vs_round1",
        )
    )
    if archive_round2_mutations:
        initial_tasks.extend(
            _make_ablation_tasks_from_keys(
                final_payload=round2_payload,
                reset_payload=pre_payload,
                keys=list(archive_round2_mutations),
                group="historical_archive_round2_mutations",
            )
        )
    if archive_round3_delta:
        archive_round2_payload = _load_strategy_payload(ARCHIVE_ROUND2_PATH) if ARCHIVE_ROUND2_PATH.exists() else round1_payload
        initial_tasks.extend(
            _make_ablation_tasks_from_keys(
                final_payload=round2_payload,
                reset_payload=archive_round2_payload,
                keys=list(archive_round3_delta),
                group="historical_archive_round3_delta",
            )
        )
    if current_round2_manifest_mutations:
        initial_tasks.extend(
            _make_ablation_tasks_from_keys(
                final_payload=round2_payload,
                reset_payload=round1_payload,
                keys=list(current_round2_manifest_mutations),
                group="current_round2_manifest_mutations",
            )
        )
    initial_tasks.extend(
        _make_perturbation_tasks(
            final_payload=round2_payload,
            diffs=active_diffs,
            max_per_key=args.max_perturbations_per_key,
        )
    )

    seen_payloads: set[str] = set()
    results = _run_oos_tasks(
        initial_tasks,
        max_workers=args.max_workers,
        executor_kind=args.executor,
        process_batch_size=args.process_batch_size,
        seen_payloads=seen_payloads,
        label="checkpoint/ablation/perturbation",
    )
    _add_deltas(results)

    current_result = next(item for item in results if item["label"] in {"checkpoint:current_round2", "checkpoint:current_round2_promoted_copy"} and _valid_oos(item))
    autopsy = _trade_autopsy(current_result["evaluation"]["oos"]["trades"])

    targeted_tasks = _make_targeted_tasks(round2_payload, autopsy)
    targeted_results = _run_oos_tasks(
        targeted_tasks,
        max_workers=args.max_workers,
        executor_kind=args.executor,
        process_batch_size=args.process_batch_size,
        seen_payloads=seen_payloads,
        label="targeted repair",
    )
    results.extend(targeted_results)
    _add_deltas(results)

    _evaluate_is_full_for_top(
        results,
        max_workers=args.max_workers,
        executor_kind=args.executor,
        process_batch_size=args.process_batch_size,
        top_full=args.top_full,
    )
    _add_deltas(results)

    first_recommendation, first_recommendation_reason = _select_recommendation(results)
    _write_strategy_config(output_dir / "first_phase_recommended_config.json", first_recommendation["payload"])

    second_phase_results: list[dict[str, Any]] = []
    if not args.skip_second_phase:
        second_phase_tasks = _make_second_phase_tasks(first_recommendation["payload"], round2_payload)
        second_phase_results = _run_oos_tasks(
            second_phase_tasks,
            max_workers=args.max_workers,
            executor_kind=args.executor,
            process_batch_size=args.process_batch_size,
            seen_payloads=seen_payloads,
            label="second phase frequency recovery",
        )
        results.extend(second_phase_results)
        _add_deltas(results)
        _evaluate_is_full_for_top(
            results,
            max_workers=args.max_workers,
            executor_kind=args.executor,
            process_batch_size=args.process_batch_size,
            top_full=args.top_full,
        )
        _add_deltas(results)

    valid = [item for item in results if _valid_oos(item)]
    current_round1_result = next(item for item in valid if item["label"] == "checkpoint:current_round1")
    current_round2_result = next(item for item in valid if item["label"] in {"checkpoint:current_round2", "checkpoint:current_round2_promoted_copy"})
    recommendation, recommendation_reason = _select_recommendation(results)
    _write_strategy_config(output_dir / "recommended_config.json", recommendation["payload"])

    second_phase_ranked = sorted(
        (item for item in valid if item.get("stage") == "second_phase" and "is" in item.get("evaluation", {})),
        key=lambda item: (
            item["evaluation"]["oos"]["metrics"]["net_return_pct"],
            item["evaluation"]["oos"]["metrics"]["total_trades"],
            item["evaluation"]["is"]["metrics"]["net_return_pct"],
        ),
        reverse=True,
    )
    if second_phase_ranked:
        _write_strategy_config(output_dir / "second_phase_best_config.json", second_phase_ranked[0]["payload"])

    checkpoint_metrics = {
        item["label"]: {
            window: item["evaluation"][window]["metrics"]
            for window in ("oos", "is", "full")
            if window in item["evaluation"]
        }
        for item in results
        if item.get("stage") == "checkpoint" and _valid_oos(item)
    }
    acceptance_risk = _summarize_acceptance_risk(results)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": {
            "in_sample": {"start": IS_START, "end": IS_END},
            "out_of_sample": {"start": OOS_START, "end": OOS_END},
            "full": {"start": IS_START, "end": FULL_END},
        },
        "candidate_count": len(results),
        "error_count": sum(1 for item in results if item.get("error")),
        "active_cumulative_diff_count_vs_pre_round1": len(active_diffs),
        "active_cumulative_diffs_vs_pre_round1": active_diffs,
        "archive_cumulative_mutations": archive_cumulative,
        "archive_round2_mutations": archive_round2_mutations,
        "archive_round3_delta": archive_round3_delta,
        "current_round2_latest_deltas_vs_round1": current_round2_latest_diffs,
        "current_round2_manifest_mutations": current_round2_manifest_mutations,
        "checkpoint_metrics": checkpoint_metrics,
        "current_round2_oos_trade_autopsy": autopsy,
        "current_round1_vs_round2_oos_trade_set_delta": _compare_trade_sets(
            current_round1_result["evaluation"]["oos"]["trades"],
            current_round2_result["evaluation"]["oos"]["trades"],
        ),
        "acceptance_risk": acceptance_risk,
        "first_phase_recommended_candidate": _compact_result(first_recommendation, include_payload=True),
        "first_phase_recommendation_reason": first_recommendation_reason,
        "first_phase_recommended_config_path": str(output_dir / "first_phase_recommended_config.json"),
        "second_phase_enabled": not args.skip_second_phase,
        "second_phase_candidate_count": len(second_phase_results),
        "second_phase_frontier": [
            _compact_result(item)
            for item in second_phase_ranked
            if item["evaluation"]["oos"]["metrics"]["net_return_pct"] > current_round2_result["evaluation"]["oos"]["metrics"]["net_return_pct"]
            and item["evaluation"]["oos"]["metrics"]["total_trades"] >= max(1.0, current_round2_result["evaluation"]["oos"]["metrics"]["total_trades"] - 1.0)
        ][:30],
        "second_phase_best_config_path": str(output_dir / "second_phase_best_config.json") if second_phase_ranked else "",
        "top_oos_candidates": [
            _compact_result(item)
            for item in sorted(
                valid,
                key=lambda item: (
                    item["evaluation"]["oos"]["metrics"]["net_return_pct"],
                    item["evaluation"]["oos"]["metrics"]["total_trades"],
                    item["evaluation"]["oos"]["metrics"]["expectancy_r"],
                ),
                reverse=True,
            )[:40]
        ],
        "top_dual_window_candidates": [
            _compact_result(item)
            for item in sorted(
                (item for item in valid if "is" in item.get("evaluation", {})),
                key=lambda item: (
                    item["evaluation"]["oos"]["metrics"]["net_return_pct"],
                    item["evaluation"]["is"]["metrics"]["net_return_pct"],
                    item["evaluation"]["oos"]["metrics"]["total_trades"],
                ),
                reverse=True,
            )[:40]
        ],
        "recommended_candidate": _compact_result(recommendation, include_payload=True),
        "recommendation_reason": recommendation_reason,
        "recommended_config_path": str(output_dir / "recommended_config.json"),
        "candidate_results_csv": str(output_dir / "candidate_results.csv"),
        "all_results_json": str(output_dir / "all_results.json"),
    }
    summary["diagnosis"] = _diagnosis(summary)

    (output_dir / "all_results_pre_csv.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(output_dir / "candidate_results.csv", results)
    (output_dir / "all_results.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_report(output_dir / "report.md", summary)

    print(json.dumps({
        "output_dir": str(output_dir),
        "candidates": len(results),
        "errors": summary["error_count"],
        "current_round2_is": checkpoint_metrics["checkpoint:current_round2"]["is"],
        "current_round2_oos": checkpoint_metrics["checkpoint:current_round2"]["oos"],
        "diagnosis": summary["diagnosis"],
        "recommended": _compact_result(recommendation),
        "report": str(output_dir / "report.md"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
