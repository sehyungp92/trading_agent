"""Reusable OOS repair sweep for any optimized strategy round.

The script generalizes the original trend round-2 repair workflow:

* evaluate checkpoints across the requested strategy's previous rounds
* ablate active cumulative diffs versus a baseline config
* ablate latest round diffs versus the previous round
* perturb active parameters and try targeted repair mutations
* optionally run a second phase from a seed config to recover frequency
* evaluate OOS by checkpointing at the IS/OOS split and continuing from IS state

Example:
    python scripts/analyze_oos_repair.py --strategy trend --round 2 \
        --is-start 2026-02-25 --is-end 2026-04-20 \
        --oos-start 2026-04-21 --oos-end 2026-05-23 \
        --phase both --max-workers 6
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import yaml

from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.profiles import (
    LIVE_PARITY_PROFILE,
    build_backtest_config_from_profile,
)
from crypto_trader.backtest.runner import run, run_split_continuation
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.config_mutator import apply_mutations
from crypto_trader.optimize.revalidation import local_perturbation_values


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL"]

STRATEGY_CONFIG_CLASS = {
    "trend": ("crypto_trader.strategy.trend.config", "TrendConfig"),
    "momentum": ("crypto_trader.strategy.momentum.config", "MomentumConfig"),
    "breakout": ("crypto_trader.strategy.breakout.config", "BreakoutConfig"),
}

STRATEGY_TIMEFRAMES = {
    "trend": ("15m", "1h", "1d"),
    "momentum": ("15m", "1h", "4h"),
    "breakout": ("30m", "4h"),
}

_EVAL_STORES: dict[tuple[str, str, tuple[str, ...]], "_CachedEvaluationStore"] = {}

METRIC_KEYS = [
    "net_return_pct",
    "net_profit",
    "total_trades",
    "win_rate",
    "expectancy_r",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe_ratio",
    "calmar_ratio",
    "exit_efficiency",
    "avg_mae_r",
    "avg_mfe_r",
    "realized_pnl_net",
    "total_fees",
    "funding_cost_total",
]


class _CachedEvaluationStore:
    """Per-worker read-through cache for repeated candidate backtests."""

    def __init__(self, data_dir: Path, symbols: list[str], timeframes: tuple[str, ...]) -> None:
        store = ParquetStore(base_dir=data_dir)
        self._candles = {
            (symbol, timeframe): store.load_candles(symbol, timeframe)
            for symbol in symbols
            for timeframe in timeframes
        }
        self._funding = {
            symbol: store.load_funding(symbol)
            for symbol in symbols
        }

    def load_candles(self, coin: str, interval: str):
        return self._candles.get((coin, interval))

    def load_funding(self, coin: str):
        return self._funding.get(coin)


def _evaluation_store(
    *,
    strategy: str,
    symbols: list[str],
    data_dir: Path,
) -> _CachedEvaluationStore:
    key = (
        str(data_dir.resolve()),
        strategy,
        tuple(str(symbol).upper() for symbol in symbols),
    )
    if key not in _EVAL_STORES:
        _EVAL_STORES[key] = _CachedEvaluationStore(
            data_dir=data_dir,
            symbols=list(key[2]),
            timeframes=STRATEGY_TIMEFRAMES[strategy],
        )
    return _EVAL_STORES[key]


@dataclass(frozen=True)
class RepairContext:
    strategy: str
    round_num: int
    symbols: list[str]
    data_dir: Path
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str

    @property
    def current_label(self) -> str:
        return f"checkpoint:current_round{self.round_num}"

    @property
    def previous_label(self) -> str | None:
        if self.round_num <= 1:
            return None
        return f"checkpoint:previous_round{self.round_num - 1}"

    @property
    def full_end(self) -> str:
        return self.oos_end


def _quiet_logging() -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))


def _config_class(strategy: str) -> type:
    try:
        module_name, class_name = STRATEGY_CONFIG_CLASS[strategy]
    except KeyError as exc:
        raise ValueError(f"Unsupported strategy {strategy!r}") from exc
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _config_from_payload(strategy: str, payload: dict[str, Any]) -> Any:
    return _config_class(strategy).from_dict(deepcopy(payload))


def _load_strategy_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("strategy", raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected strategy mapping in {path}")
    return payload


def _load_config(strategy: str, path: Path) -> Any:
    return _config_from_payload(strategy, _load_strategy_payload(path))


def _round_config_path(strategy: str, round_num: int) -> Path:
    return ROOT / "output" / strategy / f"round_{round_num}" / "optimized_config.json"


def _default_base_config_path(strategy: str) -> Path:
    candidates = [
        ROOT / "config" / f"{strategy}_pre_round1.yaml",
        ROOT / "config" / f"{strategy}_pre_round1.json",
        ROOT / "config" / "strategies" / f"{strategy}.json",
        _round_config_path(strategy, 1),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No baseline config found for {strategy}. Pass --base-config explicitly."
    )


def _default_base_config(strategy: str) -> tuple[str, Any]:
    if strategy == "breakout":
        from crypto_trader.optimize.breakout_round3_pre_round1 import build_pre_round1_config

        return (
            "builtin:crypto_trader.optimize.breakout_round3_pre_round1.build_pre_round1_config",
            build_pre_round1_config(),
        )
    path = _default_base_config_path(strategy)
    return str(path), _load_config(strategy, path)


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _has_path(payload: dict[str, Any], path: str) -> bool:
    section, dot, field = path.partition(".")
    return bool(dot) and isinstance(payload.get(section), dict) and field in payload[section]


def _set_path(payload: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    section, dot, field = path.partition(".")
    if not dot:
        raise ValueError(f"Expected section.field path, got {path!r}")
    updated = deepcopy(payload)
    if section not in updated or not isinstance(updated[section], dict):
        updated[section] = {}
    updated[section][field] = value
    return updated


def _diff_configs(base: Any, candidate: Any) -> dict[str, dict[str, Any]]:
    base_flat = _flatten(base.to_dict())
    cand_flat = _flatten(candidate.to_dict())
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


def _metrics_subset(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS if key in metrics}


def _window_score(metrics: dict[str, float], *, oos: bool) -> float:
    ret = float(metrics.get("net_return_pct", 0.0))
    trades = float(metrics.get("total_trades", 0.0))
    expectancy = float(metrics.get("expectancy_r", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    dd = float(metrics.get("max_drawdown_pct", 0.0))
    eff = float(metrics.get("exit_efficiency", 0.0))
    trade_weight = 0.35 if oos else 0.10
    return (
        ret
        + trade_weight * min(trades, 20.0)
        + 5.0 * expectancy
        + 0.45 * min(pf, 5.0)
        + 1.5 * eff
        - 0.25 * dd
    )


def _trade_to_dict(trade: Any) -> dict[str, Any]:
    direction_obj = getattr(trade, "direction", "")
    grade_obj = getattr(trade, "setup_grade", "")
    direction = getattr(direction_obj, "value", str(direction_obj))
    grade = getattr(grade_obj, "value", str(grade_obj))
    economic_r = getattr(
        trade,
        "economic_r_multiple",
        getattr(trade, "realized_r_multiple", getattr(trade, "r_multiple", 0.0)),
    )
    return {
        "trade_id": getattr(trade, "trade_id", ""),
        "symbol": getattr(trade, "symbol", ""),
        "direction": direction,
        "entry_time": getattr(trade, "entry_time", "").isoformat()
        if hasattr(getattr(trade, "entry_time", ""), "isoformat")
        else str(getattr(trade, "entry_time", "")),
        "exit_time": getattr(trade, "exit_time", "").isoformat()
        if hasattr(getattr(trade, "exit_time", ""), "isoformat")
        else str(getattr(trade, "exit_time", "")),
        "entry_price": getattr(trade, "entry_price", 0.0),
        "exit_price": getattr(trade, "exit_price", 0.0),
        "net_pnl": getattr(trade, "net_pnl", 0.0),
        "r_multiple": economic_r,
        "geometric_r_multiple": getattr(trade, "r_multiple", None),
        "realized_r_multiple": getattr(trade, "realized_r_multiple", None),
        "commission": getattr(trade, "commission", 0.0),
        "funding_paid": getattr(trade, "funding_paid", 0.0),
        "bars_held": getattr(trade, "bars_held", 0),
        "setup_grade": grade,
        "exit_reason": getattr(trade, "exit_reason", ""),
        "confluences_used": list(getattr(trade, "confluences_used", None) or []),
        "confirmation_type": getattr(trade, "confirmation_type", ""),
        "entry_method": getattr(trade, "entry_method", ""),
        "mae_r": getattr(trade, "mae_r", None),
        "mfe_r": getattr(trade, "mfe_r", None),
        "signal_variant": getattr(trade, "signal_variant", ""),
    }


def _evaluate_payload(
    *,
    strategy: str,
    payload: dict[str, Any],
    windows: dict[str, tuple[str, str]],
    symbols: list[str],
    data_dir: Path,
    continuation_start: str | None = None,
) -> dict[str, Any]:
    _quiet_logging()
    store = _evaluation_store(
        strategy=strategy,
        symbols=symbols,
        data_dir=data_dir,
    )
    result: dict[str, Any] = {}
    for window_name, (start, end) in windows.items():
        config = _config_from_payload(strategy, payload)
        if window_name == "oos" and continuation_start is not None:
            bt_config = build_backtest_config_from_profile(
                profile=LIVE_PARITY_PROFILE,
                symbols=symbols,
                start_date=date.fromisoformat(continuation_start),
                end_date=date.fromisoformat(end),
            )
            split = run_split_continuation(
                config,
                bt_config,
                split_date=date.fromisoformat(start),
                data_dir=data_dir,
                store=store,
                strategy_type=strategy,
            )
            backtest = split.out_of_sample
        else:
            bt_config = build_backtest_config_from_profile(
                profile=LIVE_PARITY_PROFILE,
                symbols=symbols,
                start_date=date.fromisoformat(start),
                end_date=date.fromisoformat(end),
            )
            backtest = run(
                config,
                bt_config,
                data_dir=data_dir,
                store=store,
                strategy_type=strategy,
            )
        metrics = metrics_to_dict(backtest.metrics)
        result[window_name] = {
            "metrics": _metrics_subset(metrics),
            "score": _window_score(metrics, oos=window_name == "oos"),
            "trades": [_trade_to_dict(trade) for trade in backtest.trades],
            "terminal_marks": len(backtest.terminal_marks),
            "stateful_continuation": window_name == "oos" and continuation_start is not None,
        }
    return result


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    try:
        evaluation = _evaluate_payload(
            strategy=task["strategy"],
            payload=task["payload"],
            windows=task["windows"],
            symbols=task["symbols"],
            data_dir=Path(task["data_dir"]),
            continuation_start=task.get("continuation_start"),
        )
    except Exception as exc:
        label = task.get("label", "<unknown>")
        raise RuntimeError(f"candidate {label!r} failed in worker") from exc
    item = {
        key: value
        for key, value in task.items()
        if key not in {"windows", "symbols", "data_dir", "strategy", "evaluation"}
    }
    item["evaluation"] = evaluation
    return item


def _trade_autopsy(trades: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(float(t["net_pnl"]) for t in trades)
    losers = sorted(
        (t for t in trades if float(t["net_pnl"]) < 0.0),
        key=lambda t: float(t["net_pnl"]),
    )
    winners = sorted(
        (t for t in trades if float(t["net_pnl"]) > 0.0),
        key=lambda t: float(t["net_pnl"]),
        reverse=True,
    )
    by_symbol = defaultdict(lambda: {"n": 0, "net_pnl": 0.0, "total_r": 0.0})
    by_symbol_direction = defaultdict(lambda: {"n": 0, "net_pnl": 0.0, "total_r": 0.0})
    by_exit = defaultdict(lambda: {"n": 0, "net_pnl": 0.0, "total_r": 0.0})
    by_day = defaultdict(lambda: {"n": 0, "net_pnl": 0.0, "total_r": 0.0})
    for trade in trades:
        pnl = float(trade["net_pnl"])
        r_value = float(trade["r_multiple"] or 0.0)
        symbol = str(trade["symbol"])
        direction = str(trade["direction"])
        exit_reason = str(trade["exit_reason"])
        day = str(trade["entry_time"])[:10]
        for bucket, key in (
            (by_symbol, symbol),
            (by_symbol_direction, f"{symbol}_{direction}"),
            (by_exit, exit_reason),
            (by_day, day),
        ):
            bucket[key]["n"] += 1
            bucket[key]["net_pnl"] += pnl
            bucket[key]["total_r"] += r_value
    return {
        "net_pnl": total,
        "trade_count": len(trades),
        "loser_count": len(losers),
        "winner_count": len(winners),
        "worst_trades": losers[:5],
        "best_trades": winners[:5],
        "largest_loss_share_of_net_loss": (
            abs(float(losers[0]["net_pnl"])) / abs(total)
            if losers and total < 0
            else None
        ),
        "top_two_loss_share_of_net_loss": (
            sum(abs(float(t["net_pnl"])) for t in losers[:2]) / abs(total)
            if len(losers) >= 2 and total < 0
            else None
        ),
        "by_symbol": dict(sorted(by_symbol.items())),
        "by_symbol_direction": dict(sorted(by_symbol_direction.items())),
        "by_exit_reason": dict(sorted(by_exit.items())),
        "by_entry_day": dict(sorted(by_day.items())),
    }


def _trade_signature(trade: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(trade["entry_time"])[:16],
        str(trade["symbol"]),
        str(trade["direction"]),
        str(trade["exit_reason"]),
    )


def _compare_trade_sets(
    base_trades: list[dict[str, Any]],
    candidate_trades: list[dict[str, Any]],
) -> dict[str, Any]:
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


def _task(
    *,
    label: str,
    stage: str,
    group: str,
    payload: dict[str, Any],
    mutation_key: str = "",
    base_value: Any = None,
    candidate_value: Any = None,
    thesis: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "stage": stage,
        "group": group,
        "mutation_key": mutation_key,
        "base_value": base_value,
        "candidate_value": candidate_value,
        "thesis": thesis,
        "payload": payload,
    }


def _task_from_mutations(
    *,
    strategy: str,
    label: str,
    stage: str,
    group: str,
    base_payload: dict[str, Any],
    mutations: dict[str, Any],
    thesis: str,
) -> dict[str, Any] | None:
    if not all(_has_path(base_payload, key) for key in mutations):
        return None
    try:
        cfg = apply_mutations(_config_from_payload(strategy, base_payload), mutations)
    except Exception:
        return None
    return _task(
        label=label,
        stage=stage,
        group=group,
        mutation_key=",".join(mutations),
        candidate_value=mutations,
        thesis=thesis,
        payload=cfg.to_dict(),
    )


def _make_ablation_tasks(
    *,
    final_payload: dict[str, Any],
    diffs: dict[str, dict[str, Any]],
    group: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for key, detail in diffs.items():
        if not _has_path(final_payload, key):
            continue
        reset_value = detail["base_value"]
        tasks.append(
            _task(
                label=f"{group}:remove:{key}",
                stage="ablation",
                group=group,
                mutation_key=key,
                base_value=reset_value,
                candidate_value=detail["candidate_value"],
                payload=_set_path(final_payload, key, reset_value),
            )
        )
    return tasks


def _domain_values(key: str, value: Any, *, base_value: Any | None = None) -> list[Any]:
    values: list[Any] = []
    key_l = key.lower()
    if isinstance(value, bool):
        values.append(not value)
    elif isinstance(value, str):
        if key_l.endswith("_direction") or key_l.startswith("symbol_filter."):
            values.extend(["both", "long_only", "short_only", "disabled"])
        elif key_l.endswith(".mode") or "mode" in key_l:
            values.extend(["legacy", "close", "break", "hybrid_grade", "confirm_preferred"])
        elif "action" in key_l:
            values.extend(["reduce", "exit", "hold"])
    elif isinstance(value, int) and not isinstance(value, bool):
        values.extend(local_perturbation_values(key, value))
        if any(token in key_l for token in ("bars", "lookback", "window", "wait")):
            values.extend([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 24, 30, 36, 48])
        if "ema" in key_l:
            values.extend([9, 12, 20, 21, 25, 30, 35, 40, 50, 100, 200])
        if "period" in key_l:
            values.extend([7, 10, 14, 20, 21, 30, 50])
    elif isinstance(value, float):
        values.extend(local_perturbation_values(key, value))
        if "risk_pct" in key_l:
            values.extend([0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.020, 0.0216, 0.024, 0.03])
        elif "adx" in key_l:
            values.extend([8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 24.0, 25.0, 28.0, 30.0])
        elif "room" in key_l:
            values.extend([0.0, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0])
        elif "score" in key_l:
            values.extend([1.0, 1.1, 1.25, 1.35, 1.5, 1.75, 2.0, 2.2, 2.5])
        elif any(token in key_l for token in ("volume", "body", "ratio")):
            values.extend([0.5, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.1, 1.2])
        elif any(token in key_l for token in ("tp", "frac", "runner")):
            values.extend([0.15, 0.25, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2])
        elif any(token in key_l for token in ("scratch", "quick_exit", "mfe_lock", "be_buffer")):
            values.extend([-0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        elif any(token in key_l for token in ("trail", "atr", "buffer", "mult")):
            values.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
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
            if any(token in key_l for token in ("frac", "body", "ratio")) and not (0.0 < item < 1.5):
                continue
            if "risk_pct" in key_l and not (0.0 < item <= 0.05):
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
    if max_per_key <= 0:
        return []
    tasks: list[dict[str, Any]] = []
    for key, detail in diffs.items():
        current_value = detail["candidate_value"]
        base_value = detail.get("base_value")
        for value in _domain_values(key, current_value, base_value=base_value)[:max_per_key]:
            if not _has_path(final_payload, key):
                continue
            tasks.append(
                _task(
                    label=f"perturb:{key}:{value}",
                    stage="perturbation",
                    group="active_cumulative",
                    mutation_key=key,
                    base_value=current_value,
                    candidate_value=value,
                    payload=_set_path(final_payload, key, value),
                )
            )
    return tasks


def _interesting_key(key: str, value: Any) -> bool:
    if key == "symbols":
        return False
    key_l = key.lower()
    tokens = (
        "risk",
        "room",
        "pullback",
        "body",
        "volume",
        "adx",
        "trail",
        "scratch",
        "quick_exit",
        "mfe",
        "entry",
        "confirm",
        "reentry",
        "direction",
        "filter",
        "stop",
        "tp",
        "bars",
        "lookback",
        "threshold",
    )
    if any(token in key_l for token in tokens):
        return isinstance(value, (bool, int, float, str))
    return False


def _generic_single_mutation_sets(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    flat = _flatten(payload)
    sets: list[tuple[str, dict[str, Any], str]] = []

    explicit: list[tuple[str, dict[str, Any], str]] = [
        ("weekly_room_off", {"setup.weekly_room_filter_enabled": False}, "Disable weekly room gate."),
        (
            "weekly_room_0_0",
            {"setup.weekly_room_filter_enabled": True, "setup.min_weekly_room_r": 0.0},
            "Keep weekly room enabled but remove the threshold.",
        ),
        ("reentry_off", {"reentry.enabled": False}, "Disable reentry branch."),
        ("scratch_off", {"exits.scratch_exit_enabled": False}, "Disable scratch exit branch."),
        ("volume_confirm_off", {
            "confirmation.require_volume_confirm": False,
            "confirmation.enforce_volume_on_trigger": False,
        }, "Disable volume confirmation requirement."),
        ("require_confirm_all", {"confirmation.require_confirmation": True}, "Require confirmation for all setups."),
        ("require_confirm_b", {"confirmation.require_confirmation_for_b": True}, "Require confirmation for weaker setups."),
    ]
    sets.extend(explicit)

    for key, value in flat.items():
        if key.endswith("_direction") or key.startswith("symbol_filter."):
            for direction in ("both", "long_only", "short_only", "disabled"):
                sets.append((f"{key}_{direction}", {key: direction}, "Per-symbol direction gate."))
        elif _interesting_key(key, value):
            for candidate in _domain_values(key, value)[:6]:
                sets.append((f"{key}_{candidate}", {key: candidate}, "Generic one-field repair frontier."))

    valid: list[tuple[str, dict[str, Any], str]] = []
    seen: set[str] = set()
    for label, mutations, thesis in sets:
        if not all(_has_path(payload, key) for key in mutations):
            continue
        signature = json.dumps(mutations, sort_keys=True, separators=(",", ":"), default=str)
        if signature in seen:
            continue
        seen.add(signature)
        safe_label = label.replace(".", "_").replace(":", "_").replace(" ", "_")
        valid.append((safe_label, mutations, thesis))
    return valid


def _make_targeted_tasks(
    *,
    strategy: str,
    final_payload: dict[str, Any],
    max_tasks: int,
) -> list[dict[str, Any]]:
    if max_tasks <= 0:
        return []
    tasks: list[dict[str, Any]] = []
    for label, mutations, thesis in _generic_single_mutation_sets(final_payload):
        task = _task_from_mutations(
            strategy=strategy,
            label=f"targeted:{label}",
            stage="targeted",
            group="repair",
            base_payload=final_payload,
            mutations=mutations,
            thesis=thesis,
        )
        if task is not None:
            tasks.append(task)
        if len(tasks) >= max_tasks:
            break
    return tasks


def _frequency_mutation_sets(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    flat = _flatten(payload)
    sets: list[tuple[str, dict[str, Any], str]] = []
    for key, value in flat.items():
        key_l = key.lower()
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        if not any(token in key_l for token in ("bars", "lookback", "window", "wait")):
            continue
        values = _domain_values(key, value)[:10]
        for candidate in values:
            sets.append((f"{key}_{candidate}", {key: candidate}, "Frequency/quality frontier."))
    return sets


def _guard_mutation_sets(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    flat = _flatten(payload)
    guards: list[tuple[str, dict[str, Any], str]] = []
    for key, value in flat.items():
        key_l = key.lower()
        if key.endswith("_direction") or key.startswith("symbol_filter."):
            guards.append((f"{key}_disabled", {key: "disabled"}, "Disable weak symbol/direction sleeve."))
            guards.append((f"{key}_long_only", {key: "long_only"}, "Keep long side only."))
        elif "body" in key_l and isinstance(value, float):
            guards.append((f"{key}_075", {key: 0.75}, "Tighten body quality."))
            guards.append((f"{key}_080", {key: 0.80}, "Tighten body quality."))
        elif "volume" in key_l and isinstance(value, float):
            guards.append((f"{key}_095", {key: 0.95}, "Tighten volume quality."))
            guards.append((f"{key}_100", {key: 1.0}, "Normalize volume quality."))
        elif "risk_pct" in key_l and isinstance(value, float):
            guards.append((f"{key}_020", {key: 0.020}, "Moderate risk."))
            guards.append((f"{key}_018", {key: 0.018}, "Lower risk."))
        elif key_l.endswith(".enabled") and isinstance(value, bool) and "reentry" in key_l:
            guards.append((f"{key}_false", {key: False}, "Disable reentry churn."))
        elif "weekly_room_filter_enabled" in key_l:
            guards.append((f"{key}_false", {key: False}, "Disable weekly room gate."))
        elif "min_weekly_room_r" in key_l:
            guards.append((f"{key}_0_0", {key: 0.0}, "Remove weekly room threshold."))
    valid: list[tuple[str, dict[str, Any], str]] = []
    seen: set[str] = set()
    for label, mutations, thesis in guards:
        if not all(_has_path(payload, key) for key in mutations):
            continue
        signature = json.dumps(mutations, sort_keys=True, separators=(",", ":"), default=str)
        if signature in seen:
            continue
        seen.add(signature)
        safe_label = label.replace(".", "_")
        valid.append((safe_label, mutations, thesis))
    return valid


def _make_second_phase_tasks(
    *,
    strategy: str,
    seed_payload: dict[str, Any],
    current_payload: dict[str, Any],
    max_combos: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = [
        _task(
            label="checkpoint:current_round",
            stage="checkpoint",
            group="checkpoint",
            payload=current_payload,
        ),
        _task(
            label="checkpoint:first_phase_recommended",
            stage="checkpoint",
            group="checkpoint",
            payload=seed_payload,
        ),
    ]
    singles = _generic_single_mutation_sets(seed_payload)
    for label, mutations, thesis in singles:
        task = _task_from_mutations(
            strategy=strategy,
            label=f"second_phase:{label}",
            stage="second_phase",
            group="single_repair",
            base_payload=seed_payload,
            mutations=mutations,
            thesis=thesis,
        )
        if task is not None:
            tasks.append(task)

    combo_count = 0
    frequency_sets = _frequency_mutation_sets(seed_payload)
    guard_sets = _guard_mutation_sets(seed_payload)
    for freq_label, freq_mutation, _freq_thesis in frequency_sets:
        for guard_label, guard_mutation, guard_thesis in guard_sets:
            mutations = {**freq_mutation, **guard_mutation}
            if len(mutations) != len(freq_mutation) + len(guard_mutation):
                continue
            task = _task_from_mutations(
                strategy=strategy,
                label=f"second_phase:{freq_label}_{guard_label}",
                stage="second_phase",
                group="frequency_recovery",
                base_payload=seed_payload,
                mutations=mutations,
                thesis=f"Recover frequency with guardrail: {guard_thesis}",
            )
            if task is not None:
                tasks.append(task)
                combo_count += 1
            if combo_count >= max_combos:
                return tasks
    return tasks


def _checkpoint_tasks(
    *,
    strategy: str,
    round_num: int,
    base_payload: dict[str, Any],
    base_config_source: str,
    include_archives: bool,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = [
        _task(
            label="checkpoint:base_config",
            stage="checkpoint",
            group="checkpoint",
            candidate_value=base_config_source,
            payload=base_payload,
        )
    ]
    for idx in range(1, round_num + 1):
        path = _round_config_path(strategy, idx)
        if not path.exists():
            continue
        if idx == round_num:
            label = f"checkpoint:current_round{idx}"
        elif idx == round_num - 1:
            label = f"checkpoint:previous_round{idx}"
        else:
            label = f"checkpoint:round_{idx}"
        tasks.append(
            _task(
                label=label,
                stage="checkpoint",
                group="checkpoint",
                payload=_load_config(strategy, path).to_dict(),
            )
        )
    if include_archives:
        archive_root = ROOT / "output" / strategy / "archive"
        if archive_root.exists():
            for path in sorted(archive_root.glob("**/round_*/optimized_config.json")):
                rel = path.relative_to(archive_root).as_posix().replace("/", "_")
                tasks.append(
                    _task(
                        label=f"checkpoint:archive_{rel}",
                        stage="checkpoint",
                        group="checkpoint",
                        payload=_load_config(strategy, path).to_dict(),
                    )
                )
    return tasks


def _dedupe_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        key = json.dumps(task["payload"], sort_keys=True, separators=(",", ":"), default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)
    return deduped


_TASK_HASH_IGNORED_KEYS = {
    "evaluation",
    "error",
    "task_hash",
    "task_index",
    "worker_attempt",
    "oos_score_delta_vs_current",
    "oos_net_return_delta_vs_current",
    "oos_trades_delta_vs_current",
}


def _task_hash(task: dict[str, Any]) -> str:
    identity = {
        key: value
        for key, value in task.items()
        if key not in _TASK_HASH_IGNORED_KEYS
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    import hashlib

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_incremental_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if item.get("error"):
            continue
        task_hash = item.get("task_hash")
        if not task_hash:
            task_hash = _task_hash(item)
            item["task_hash"] = task_hash
        results[task_hash] = item
    return results


def _append_incremental_result(path: Path | None, item: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True, default=str) + "\n")
        handle.flush()


def _execute_tasks(
    tasks: list[dict[str, Any]],
    *,
    max_workers: int,
    executor_kind: str,
    process_batch_size: int,
    progress_label: str,
    subprocess_timeout: int,
    incremental_results_file: Path | None = None,
) -> list[dict[str, Any]]:
    total = len(tasks)
    if total == 0:
        return []
    completed_by_hash = (
        _load_incremental_results(incremental_results_file)
        if incremental_results_file is not None
        else {}
    )
    indexed_pending: list[tuple[int, str, dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_hash = _task_hash(task)
        existing = completed_by_hash.get(task_hash)
        if existing is not None:
            existing.setdefault("task_index", idx)
            existing.setdefault("task_hash", task_hash)
            results.append(existing)
            continue
        indexed_pending.append((idx, task_hash, task))
    completed = len(results)
    if completed:
        print(
            f"  {progress_label} resumed {completed}/{total} completed tasks",
            flush=True,
        )
    if not indexed_pending:
        return sorted(results, key=lambda item: item.get("task_index", 10**9))

    def finish_item(item: dict[str, Any], task_index: int, task_hash: str) -> dict[str, Any]:
        item["task_index"] = task_index
        item["task_hash"] = task_hash
        _append_incremental_result(incremental_results_file, item)
        return item

    if executor_kind == "sequential":
        for task_index, task_hash, task in indexed_pending:
            try:
                item = _worker(task)
            except Exception as exc:
                item = {**task, "evaluation": {}, "error": repr(exc)}
            results.append(finish_item(item, task_index, task_hash))
            completed += 1
            if completed % 10 == 0 or completed == total:
                errors = sum(1 for result in results if result.get("error"))
                print(f"  {progress_label} progress {completed}/{total} errors={errors}", flush=True)
        return sorted(results, key=lambda item: item.get("task_index", 10**9))
    if executor_kind == "thread":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, task): (task_index, task_hash, task)
                for task_index, task_hash, task in indexed_pending
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                task_index, task_hash, task = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    item = {**task, "evaluation": {}, "error": repr(exc)}
                results.append(finish_item(item, task_index, task_hash))
                completed += 1
                if completed % 10 == 0 or completed == total:
                    errors = sum(1 for result in results if result.get("error"))
                    print(f"  {progress_label} progress {completed}/{total} errors={errors}", flush=True)
        return sorted(results, key=lambda item: item.get("task_index", 10**9))
    if executor_kind == "subprocess":
        worker_dir = ROOT / "tmp" / "oos_repair_worker_tasks"
        worker_dir.mkdir(parents=True, exist_ok=True)
        for task_index, task_hash, task in indexed_pending:
            token = uuid.uuid4().hex
            task_path = worker_dir / f"{token}.task.json"
            result_path = worker_dir / f"{token}.result.json"
            task_path.write_text(json.dumps(task, default=str), encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-task-file",
                        str(task_path),
                        "--worker-output-file",
                        str(result_path),
                    ],
                    cwd=str(ROOT),
                    text=True,
                    capture_output=True,
                    timeout=subprocess_timeout,
                )
                worker_error = (
                    f"worker_returncode={proc.returncode}; "
                    f"stdout_tail={proc.stdout[-1000:]!r}; "
                    f"stderr_tail={proc.stderr[-1000:]!r}"
                )
            except subprocess.TimeoutExpired as exc:
                proc = None
                worker_error = (
                    f"worker_timeout={subprocess_timeout}; "
                    f"stdout_tail={str(exc.stdout)[-1000:]!r}; "
                    f"stderr_tail={str(exc.stderr)[-1000:]!r}"
                )
            if proc is not None and proc.returncode == 0 and result_path.exists():
                item = json.loads(result_path.read_text(encoding="utf-8-sig"))
            else:
                item = {**task, "evaluation": {}, "error": worker_error}
            results.append(finish_item(item, task_index, task_hash))
            task_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            completed += 1
            if completed % 10 == 0 or completed == total:
                errors = sum(1 for item in results if item.get("error"))
                print(f"  {progress_label} progress {completed}/{total} errors={errors}", flush=True)
        return sorted(results, key=lambda item: item.get("task_index", 10**9))

    batch_size = max(1, process_batch_size)
    pending_total = len(indexed_pending)
    for start in range(0, pending_total, batch_size):
        batch = indexed_pending[start:start + batch_size]
        batch_end = completed + len(batch)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            futures = {
                executor.submit(_worker, task): (task_index, task_hash, task)
                for task_index, task_hash, task in batch
            }
            for future in as_completed(futures):
                task_index, task_hash, task = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    item = {**task, "evaluation": {}, "error": repr(exc)}
                results.append(finish_item(item, task_index, task_hash))
                completed += 1
                if completed % 10 == 0 or completed == total or completed == batch_end:
                    errors = sum(1 for result in results if result.get("error"))
                    print(f"  {progress_label} progress {completed}/{total} errors={errors}", flush=True)
    return sorted(results, key=lambda item: item.get("task_index", 10**9))


def _run_tasks(
    tasks: list[dict[str, Any]],
    *,
    context: RepairContext,
    max_workers: int,
    executor_kind: str,
    process_batch_size: int,
    subprocess_timeout: int,
    top_full: int,
    output_dir: Path,
    max_oos_tasks: int | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _dedupe_tasks(tasks)
    if max_oos_tasks is not None:
        checkpoints = [task for task in tasks if task.get("stage") == "checkpoint"]
        others = [task for task in tasks if task.get("stage") != "checkpoint"]
        tasks = checkpoints + others[: max(0, max_oos_tasks - len(checkpoints))]

    oos_tasks = [
        {
            **task,
            "strategy": context.strategy,
            "symbols": context.symbols,
            "data_dir": str(context.data_dir),
            "windows": {"oos": (context.oos_start, context.oos_end)},
            "continuation_start": context.is_start,
        }
        for task in tasks
    ]
    print(
        f"evaluating OOS candidates: {len(oos_tasks)} tasks with "
        f"{max_workers} {executor_kind} workers",
        flush=True,
    )
    results = _execute_tasks(
        oos_tasks,
        max_workers=max_workers,
        executor_kind=executor_kind,
        process_batch_size=process_batch_size,
        subprocess_timeout=subprocess_timeout,
        progress_label="OOS",
        incremental_results_file=output_dir / "oos_results.jsonl",
    )

    successful_oos = [
        item
        for item in results
        if not item.get("error") and "oos" in item.get("evaluation", {})
    ]
    baseline_oos = next((r for r in successful_oos if r["label"] == context.current_label), None)
    if baseline_oos is None:
        baseline_oos = next((r for r in successful_oos if r["label"] == "checkpoint:current_round"), None)
    if baseline_oos is None:
        raise RuntimeError(
            "Current-round OOS checkpoint did not complete successfully; "
            f"errors={sum(1 for item in results if item.get('error'))}"
        )
    baseline_score = baseline_oos["evaluation"]["oos"]["score"]
    baseline_metrics = baseline_oos["evaluation"]["oos"]["metrics"]
    for result in results:
        if result.get("error") or "oos" not in result.get("evaluation", {}):
            result["oos_score_delta_vs_current"] = None
            result["oos_net_return_delta_vs_current"] = None
            result["oos_trades_delta_vs_current"] = None
            continue
        oos_metrics = result["evaluation"]["oos"]["metrics"]
        result["oos_score_delta_vs_current"] = result["evaluation"]["oos"]["score"] - baseline_score
        result["oos_net_return_delta_vs_current"] = oos_metrics["net_return_pct"] - baseline_metrics.get("net_return_pct", 0.0)
        result["oos_trades_delta_vs_current"] = oos_metrics["total_trades"] - baseline_metrics.get("total_trades", 0.0)

    ranked = sorted(
        successful_oos,
        key=lambda item: (
            item["evaluation"]["oos"]["metrics"]["net_return_pct"],
            item["evaluation"]["oos"]["metrics"]["total_trades"],
            item["evaluation"]["oos"]["metrics"].get("expectancy_r", 0.0),
        ),
        reverse=True,
    )
    full_labels = {item["label"] for item in results if item.get("stage") == "checkpoint"}
    full_labels.update(item["label"] for item in ranked[:top_full])
    full_tasks = [
        {
            **{
                key: value
                for key, value in item.items()
                if key not in {"evaluation", "oos_score_delta_vs_current", "oos_net_return_delta_vs_current", "oos_trades_delta_vs_current"}
            },
            "strategy": context.strategy,
            "symbols": context.symbols,
            "data_dir": str(context.data_dir),
            "windows": {
                "is": (context.is_start, context.is_end),
                "full": (context.is_start, context.full_end),
            },
        }
        for item in ranked
        if item["label"] in full_labels
    ]
    print(
        f"evaluating IS/full for top/checkpoint candidates: {len(full_tasks)} tasks "
        f"with {max_workers} {executor_kind} workers",
        flush=True,
    )
    full_by_label: dict[str, dict[str, Any]] = {}
    for item in _execute_tasks(
        full_tasks,
        max_workers=max_workers,
        executor_kind=executor_kind,
        process_batch_size=process_batch_size,
        subprocess_timeout=subprocess_timeout,
        progress_label="IS/full",
        incremental_results_file=output_dir / "is_full_results.jsonl",
    ):
        if not item.get("error"):
            full_by_label[item["label"]] = item["evaluation"]
    for item in results:
        if item["label"] in full_by_label:
            item["evaluation"].update(full_by_label[item["label"]])

    _write_csv(output_dir / "candidate_results.csv", results)
    (output_dir / "all_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return results


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "stage",
        "group",
        "mutation_key",
        "candidate_value",
        "error",
        "oos_score",
        "oos_net_return_pct",
        "oos_total_trades",
        "oos_win_rate",
        "oos_expectancy_r",
        "oos_profit_factor",
        "oos_max_drawdown_pct",
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
                "error": item.get("error", ""),
            }
            evaluation = item.get("evaluation", {})
            for window in ("oos", "is", "full"):
                if window not in evaluation:
                    continue
                row[f"{window}_score"] = evaluation[window].get("score")
                metrics = evaluation[window]["metrics"]
                for metric in (
                    "net_return_pct",
                    "total_trades",
                    "win_rate",
                    "expectancy_r",
                    "profit_factor",
                    "max_drawdown_pct",
                ):
                    row[f"{window}_{metric}"] = metrics.get(metric)
            writer.writerow(row)


def _write_strategy_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"strategy": payload}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _compact_result(item: dict[str, Any], *, include_payload: bool = False) -> dict[str, Any]:
    payload = {
        "label": item["label"],
        "stage": item.get("stage"),
        "group": item.get("group"),
        "mutation_key": item.get("mutation_key"),
        "candidate_value": item.get("candidate_value"),
        "thesis": item.get("thesis"),
        "oos_metrics": item["evaluation"]["oos"]["metrics"],
        "oos_score": item["evaluation"]["oos"]["score"],
    }
    if "is" in item["evaluation"]:
        payload["is_metrics"] = item["evaluation"]["is"]["metrics"]
        payload["is_score"] = item["evaluation"]["is"]["score"]
    if "full" in item["evaluation"]:
        payload["full_metrics"] = item["evaluation"]["full"]["metrics"]
    if include_payload:
        payload["payload"] = item["payload"]
    return payload


def _select_first_phase_recommendation(results: list[dict[str, Any]], current_label: str) -> dict[str, Any]:
    current = next((item for item in results if item["label"] == current_label), None)
    if current is None:
        current = next(item for item in results if item["stage"] == "checkpoint")
    current_oos = current["evaluation"]["oos"]["metrics"]
    current_is = current["evaluation"].get("is", {}).get("metrics", {})
    candidates = []
    for item in results:
        if "is" not in item["evaluation"]:
            continue
        oos = item["evaluation"]["oos"]["metrics"]
        is_metrics = item["evaluation"]["is"]["metrics"]
        if oos["net_return_pct"] <= current_oos["net_return_pct"]:
            continue
        if oos["total_trades"] < max(3.0, current_oos["total_trades"] - 2.0):
            continue
        if current_is and is_metrics["net_return_pct"] < current_is.get("net_return_pct", 0.0) * 0.75:
            continue
        combined = (
            0.62 * item["evaluation"]["oos"]["score"]
            + 0.38 * item["evaluation"]["is"]["score"]
            + 0.05 * oos["total_trades"]
        )
        candidates.append((combined, item))
    if candidates:
        return sorted(candidates, key=lambda pair: pair[0], reverse=True)[0][1]
    return max(
        (item for item in results if "is" in item["evaluation"]),
        key=lambda item: item["evaluation"]["oos"]["metrics"]["net_return_pct"],
    )


def _second_phase_is_floor(
    results: list[dict[str, Any]],
    *,
    min_is_return_pct: float,
    min_is_retention: float = 0.85,
) -> float:
    checkpoint_is_returns = [
        item["evaluation"]["is"]["metrics"]["net_return_pct"]
        for item in results
        if item.get("stage") == "checkpoint" and "is" in item["evaluation"]
    ]
    return max(
        [min_is_return_pct]
        + [
            min_is_retention * ret
            for ret in checkpoint_is_returns
            if ret > 0
        ]
    )


def _select_second_phase_recommendation(
    results: list[dict[str, Any]],
    *,
    min_oos_trades: float,
    min_is_return_pct: float,
    min_is_retention: float = 0.85,
) -> dict[str, Any]:
    ranked = sorted(
        (item for item in results if "is" in item["evaluation"]),
        key=lambda item: (
            item["evaluation"]["oos"]["metrics"]["net_return_pct"],
            item["evaluation"]["oos"]["metrics"]["total_trades"],
            item["evaluation"]["is"]["metrics"]["net_return_pct"],
        ),
        reverse=True,
    )
    is_floor = _second_phase_is_floor(
        ranked,
        min_is_return_pct=min_is_return_pct,
        min_is_retention=min_is_retention,
    )
    frontier = [
        item
        for item in ranked
        if item["evaluation"]["oos"]["metrics"]["net_return_pct"] > 0
        and item["evaluation"]["oos"]["metrics"]["total_trades"] >= min_oos_trades
        and item["evaluation"]["is"]["metrics"]["net_return_pct"] >= is_floor
    ]
    if frontier:
        return frontier[0]
    relaxed_frontier = [
        item
        for item in ranked
        if item["evaluation"]["oos"]["metrics"]["net_return_pct"] > 0
        and item["evaluation"]["oos"]["metrics"]["total_trades"] >= min_oos_trades
        and item["evaluation"]["is"]["metrics"]["net_return_pct"] >= min_is_return_pct
    ]
    return relaxed_frontier[0] if relaxed_frontier else ranked[0]


def _summarize_acceptance_risk(
    results: list[dict[str, Any]],
    *,
    current_label: str,
) -> dict[str, Any]:
    current = next((item for item in results if item["label"] == current_label), None)
    if current is None:
        return {}
    current_oos = current["evaluation"]["oos"]["metrics"]

    def improved_ablation(group: str) -> list[dict[str, Any]]:
        items = []
        for item in results:
            if item.get("stage") != "ablation" or item.get("group") != group:
                continue
            metrics = item["evaluation"]["oos"]["metrics"]
            if metrics["net_return_pct"] <= current_oos["net_return_pct"]:
                continue
            items.append(
                {
                    "label": item["label"],
                    "mutation_key": item["mutation_key"],
                    "candidate_value": item.get("base_value"),
                    "removed_value": item.get("candidate_value"),
                    "oos_metrics": metrics,
                    "oos_net_return_delta": metrics["net_return_pct"] - current_oos["net_return_pct"],
                    "oos_trade_delta": metrics["total_trades"] - current_oos["total_trades"],
                }
            )
        return sorted(
            items,
            key=lambda item: (item["oos_net_return_delta"], item["oos_trade_delta"]),
            reverse=True,
        )

    return {
        "latest_round_removals_that_improve_oos": improved_ablation("latest_round_delta")[:20],
        "active_cumulative_removals_that_improve_oos": improved_ablation("active_cumulative")[:30],
    }


def _build_first_phase_tasks(
    *,
    strategy: str,
    round_num: int,
    base_config: Any,
    base_config_source: str,
    include_archives: bool,
    max_perturbations_per_key: int,
    max_targeted_tasks: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_path = _round_config_path(strategy, round_num)
    if not current_path.exists():
        raise FileNotFoundError(f"Missing current round config: {current_path}")
    current_config = _load_config(strategy, current_path)
    previous_path = _round_config_path(strategy, round_num - 1)
    previous_config = _load_config(strategy, previous_path) if round_num > 1 and previous_path.exists() else None

    current_payload = current_config.to_dict()
    active_diffs = _diff_configs(base_config, current_config)
    latest_diffs = _diff_configs(previous_config, current_config) if previous_config is not None else {}

    tasks = _checkpoint_tasks(
        strategy=strategy,
        round_num=round_num,
        base_payload=base_config.to_dict(),
        base_config_source=base_config_source,
        include_archives=include_archives,
    )
    tasks.extend(_make_ablation_tasks(final_payload=current_payload, diffs=active_diffs, group="active_cumulative"))
    tasks.extend(_make_ablation_tasks(final_payload=current_payload, diffs=latest_diffs, group="latest_round_delta"))
    tasks.extend(
        _make_perturbation_tasks(
            final_payload=current_payload,
            diffs=active_diffs,
            max_per_key=max_perturbations_per_key,
        )
    )
    tasks.extend(
        _make_targeted_tasks(
            strategy=strategy,
            final_payload=current_payload,
            max_tasks=max_targeted_tasks,
        )
    )
    metadata = {
        "base_config_source": base_config_source,
        "current_round_config_path": str(current_path),
        "previous_round_config_path": str(previous_path) if previous_config is not None else "",
        "active_cumulative_diff_count_vs_base": len(active_diffs),
        "active_cumulative_diffs_vs_base": active_diffs,
        "latest_round_delta_count_vs_previous_round": len(latest_diffs),
        "latest_round_deltas_vs_previous_round": latest_diffs,
    }
    return tasks, metadata


def _latest_first_phase_seed(strategy: str, round_num: int) -> Path:
    root = ROOT / "output" / strategy / f"round_{round_num}_oos_repair"
    candidates = sorted(path for path in root.glob("*/recommended_config.json") if path.exists())
    if not candidates:
        raise FileNotFoundError(
            f"No first-phase seed found under {root}. Pass --seed-config."
        )
    return candidates[-1]


def _run_first_phase(args: argparse.Namespace, context: RepairContext, output_dir: Path) -> Path:
    if args.base_config:
        base_config_source = str(Path(args.base_config))
        base_config = _load_config(args.strategy, Path(args.base_config))
    else:
        base_config_source, base_config = _default_base_config(args.strategy)
    tasks, metadata = _build_first_phase_tasks(
        strategy=args.strategy,
        round_num=args.round_num,
        base_config=base_config,
        base_config_source=base_config_source,
        include_archives=args.include_archives,
        max_perturbations_per_key=args.max_perturbations_per_key,
        max_targeted_tasks=args.max_targeted_tasks,
    )
    if args.dump_tasks_file:
        tasks = _dedupe_tasks(tasks)
        if args.max_oos_tasks is not None:
            checkpoints = [task for task in tasks if task.get("stage") == "checkpoint"]
            others = [task for task in tasks if task.get("stage") != "checkpoint"]
            tasks = checkpoints + others[: max(0, args.max_oos_tasks - len(checkpoints))]
        oos_tasks = [
            {
                **task,
                "strategy": context.strategy,
                "symbols": context.symbols,
                "data_dir": str(context.data_dir),
                "windows": {"oos": (context.oos_start, context.oos_end)},
                "continuation_start": context.is_start,
            }
            for task in tasks
        ]
        args.dump_tasks_file.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_tasks_file.open("w", encoding="utf-8") as handle:
            for task in oos_tasks:
                handle.write(json.dumps(task, default=str) + "\n")
        metadata_path = args.dump_tasks_file.with_suffix(args.dump_tasks_file.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "strategy": context.strategy,
                    "round": context.round_num,
                    "phase": "first",
                    "task_count": len(oos_tasks),
                    "windows": {
                        "in_sample": {"start": context.is_start, "end": context.is_end},
                        "out_of_sample": {"start": context.oos_start, "end": context.oos_end},
                        "full": {"start": context.is_start, "end": context.full_end},
                    },
                    **metadata,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"dump_tasks_file": str(args.dump_tasks_file), "metadata_file": str(metadata_path), "task_count": len(oos_tasks)}, indent=2))
        return output_dir / "recommended_config.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "strategy": context.strategy,
                    "round": context.round_num,
                    "phase": "first",
                    "task_count": len(_dedupe_tasks(tasks)),
                    **metadata,
                },
                indent=2,
                default=str,
            )
        )
        return output_dir / "recommended_config.json"
    results = _run_tasks(
        tasks,
        context=context,
        max_workers=args.max_workers,
        executor_kind=args.executor,
        process_batch_size=args.process_batch_size,
        subprocess_timeout=args.subprocess_timeout,
        top_full=args.top_full,
        output_dir=output_dir,
        max_oos_tasks=args.max_oos_tasks,
    )
    recommendation = _select_first_phase_recommendation(results, context.current_label)
    recommended_path = output_dir / "recommended_config.json"
    _write_strategy_config(recommended_path, recommendation["payload"])

    current = next(item for item in results if item["label"] == context.current_label)
    previous = next((item for item in results if item["label"] == context.previous_label), None)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": context.strategy,
        "round": context.round_num,
        "phase": "first",
        "windows": {
            "in_sample": {"start": context.is_start, "end": context.is_end},
            "out_of_sample": {"start": context.oos_start, "end": context.oos_end},
            "full": {"start": context.is_start, "end": context.full_end},
        },
        **metadata,
        "checkpoint_metrics": {
            item["label"]: {
                window: item["evaluation"][window]["metrics"]
                for window in ("oos", "is", "full")
                if window in item["evaluation"]
            }
            for item in results
            if item["stage"] == "checkpoint"
        },
        "current_round_oos_trade_autopsy": _trade_autopsy(current["evaluation"]["oos"]["trades"]),
        "previous_vs_current_oos_trade_set_delta": (
            _compare_trade_sets(previous["evaluation"]["oos"]["trades"], current["evaluation"]["oos"]["trades"])
            if previous is not None
            else {}
        ),
        "acceptance_risk": _summarize_acceptance_risk(results, current_label=context.current_label),
        "top_oos_candidates": [
            _compact_result(item)
            for item in sorted(
                results,
                key=lambda item: (
                    item["evaluation"]["oos"]["metrics"]["net_return_pct"],
                    item["evaluation"]["oos"]["metrics"]["total_trades"],
                    item["evaluation"]["oos"]["metrics"].get("expectancy_r", 0.0),
                ),
                reverse=True,
            )[:30]
        ],
        "top_dual_window_candidates": [
            _compact_result(item)
            for item in sorted(
                (item for item in results if "is" in item["evaluation"]),
                key=lambda item: (
                    item["evaluation"]["oos"]["metrics"]["net_return_pct"],
                    item["evaluation"]["is"]["metrics"]["net_return_pct"],
                    item["evaluation"]["oos"]["metrics"]["total_trades"],
                ),
                reverse=True,
            )[:30]
        ],
        "recommended_candidate": _compact_result(recommendation, include_payload=True),
        "recommended_config_path": str(recommended_path),
        "candidate_results_csv": str(output_dir / "candidate_results.csv"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidates": len(results),
                "current_oos": current["evaluation"]["oos"]["metrics"],
                "recommended": _compact_result(recommendation),
            },
            indent=2,
            default=str,
        )
    )
    return recommended_path


def _run_second_phase(args: argparse.Namespace, context: RepairContext, output_dir: Path, seed_config: Path | None) -> Path:
    seed_path = seed_config or (Path(args.seed_config) if args.seed_config else _latest_first_phase_seed(args.strategy, args.round_num))
    current_payload = _load_config(args.strategy, _round_config_path(args.strategy, args.round_num)).to_dict()
    seed_payload = _load_config(args.strategy, seed_path).to_dict()
    tasks = _make_second_phase_tasks(
        strategy=args.strategy,
        seed_payload=seed_payload,
        current_payload=current_payload,
        max_combos=args.max_second_phase_combos,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "strategy": context.strategy,
                    "round": context.round_num,
                    "phase": "second",
                    "seed_config": str(seed_path),
                    "task_count": len(_dedupe_tasks(tasks)),
                },
                indent=2,
                default=str,
            )
        )
        return output_dir / "recommended_second_phase_config.json"
    results = _run_tasks(
        tasks,
        context=context,
        max_workers=args.max_workers,
        executor_kind=args.executor,
        process_batch_size=args.process_batch_size,
        subprocess_timeout=args.subprocess_timeout,
        top_full=args.top_full,
        output_dir=output_dir,
        max_oos_tasks=args.max_oos_tasks,
    )
    best = _select_second_phase_recommendation(
        results,
        min_oos_trades=args.second_phase_min_oos_trades,
        min_is_return_pct=args.second_phase_min_is_return_pct,
    )
    recommended_path = output_dir / "recommended_second_phase_config.json"
    _write_strategy_config(recommended_path, best["payload"])
    ranked = sorted(
        (item for item in results if "is" in item["evaluation"]),
        key=lambda item: (
            item["evaluation"]["oos"]["metrics"]["net_return_pct"],
            item["evaluation"]["oos"]["metrics"]["total_trades"],
            item["evaluation"]["is"]["metrics"]["net_return_pct"],
        ),
        reverse=True,
    )
    is_floor = _second_phase_is_floor(
        ranked,
        min_is_return_pct=args.second_phase_min_is_return_pct,
    )
    frontier = [
        item
        for item in ranked
        if item["evaluation"]["oos"]["metrics"]["net_return_pct"] > 0
        and item["evaluation"]["oos"]["metrics"]["total_trades"] >= args.second_phase_min_oos_trades
        and item["evaluation"]["is"]["metrics"]["net_return_pct"] >= is_floor
    ]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": context.strategy,
        "round": context.round_num,
        "phase": "second",
        "seed_config": str(seed_path),
        "candidate_count": len(results),
        "best": _compact_result(best, include_payload=True),
        "frontier": [_compact_result(item) for item in frontier[:20]],
        "top": [_compact_result(item) for item in ranked[:30]],
        "recommended_config_path": str(recommended_path),
        "candidate_results_csv": str(output_dir / "candidate_results.csv"),
        "selection_guardrails": {
            "min_oos_trades": args.second_phase_min_oos_trades,
            "min_is_return_pct": args.second_phase_min_is_return_pct,
            "effective_is_floor_pct": is_floor,
        },
    }
    (output_dir / "second_phase_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "best": _compact_result(best)}, indent=2, default=str))
    return recommended_path


def _default_output_dir(strategy: str, round_num: int, phase: str) -> Path:
    suffix = "round_{}_oos_repair".format(round_num)
    if phase == "second":
        suffix += "_second_phase"
    return ROOT / "output" / strategy / suffix / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=sorted(STRATEGY_CONFIG_CLASS), required=True)
    parser.add_argument("--round", dest="round_num", type=int, required=True)
    parser.add_argument("--phase", choices=("first", "second", "both"), default="first")
    parser.add_argument("--is-start", default="2026-02-25")
    parser.add_argument("--is-end", default="2026-04-20")
    parser.add_argument("--oos-start", default="2026-04-21")
    parser.add_argument("--oos-end", default="2026-05-23")
    parser.add_argument("--symbols", default="BTC,ETH,SOL")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--seed-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--second-output-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--executor", choices=("auto", "process", "thread", "sequential", "subprocess"), default="auto")
    parser.add_argument("--process-batch-size", type=int, default=16)
    parser.add_argument("--subprocess-timeout", type=int, default=900)
    parser.add_argument("--max-perturbations-per-key", type=int, default=8)
    parser.add_argument("--max-targeted-tasks", type=int, default=120)
    parser.add_argument("--max-second-phase-combos", type=int, default=160)
    parser.add_argument("--top-full", type=int, default=40)
    parser.add_argument("--max-oos-tasks", type=int, default=None)
    parser.add_argument("--include-archives", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dump-tasks-file", type=Path, default=None)
    parser.add_argument("--second-phase-min-oos-trades", type=float, default=10.0)
    parser.add_argument("--second-phase-min-is-return-pct", type=float, default=0.0)
    return parser.parse_args(argv)


def _resolve_executor(args: argparse.Namespace) -> str:
    if args.executor != "auto":
        if os.name == "nt" and args.executor == "process" and args.max_workers > 1:
            print(
                "warning: Windows process executor with max_workers > 1 can terminate "
                "abruptly under heavy backtest/data-load pressure; consider "
                "--executor thread or --max-workers 1.",
                flush=True,
            )
        return args.executor
    return "thread" if os.name == "nt" else "process"


def _run_worker_task_file(argv: list[str]) -> bool:
    if "--worker-task-file" not in argv:
        return False
    task_idx = argv.index("--worker-task-file")
    output_idx = argv.index("--worker-output-file")
    task_path = Path(argv[task_idx + 1])
    output_path = Path(argv[output_idx + 1])
    _quiet_logging()
    task = json.loads(task_path.read_text(encoding="utf-8-sig"))
    result = _worker(task)
    output_path.write_text(json.dumps(result, default=str), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> None:
    mp.freeze_support()
    argv = sys.argv[1:] if argv is None else argv
    if _run_worker_task_file(argv):
        return
    args = parse_args(argv)
    args.executor = _resolve_executor(args)
    _quiet_logging()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    context = RepairContext(
        strategy=args.strategy,
        round_num=args.round_num,
        symbols=symbols or DEFAULT_SYMBOLS,
        data_dir=args.data_dir,
        is_start=args.is_start,
        is_end=args.is_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
    )

    first_seed: Path | None = None
    if args.phase in {"first", "both"}:
        first_output = args.output_dir or _default_output_dir(args.strategy, args.round_num, "first")
        first_seed = _run_first_phase(args, context, first_output)
    if args.phase in {"second", "both"}:
        second_output = (
            args.second_output_dir
            or (args.output_dir if args.phase == "second" and args.output_dir else None)
            or _default_output_dir(args.strategy, args.round_num, "second")
        )
        _run_second_phase(args, context, second_output, first_seed)


if __name__ == "__main__":
    main()
