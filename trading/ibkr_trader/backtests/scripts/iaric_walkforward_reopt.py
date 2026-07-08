"""Purged walk-forward IARIC challenger optimization.

This workflow keeps the current production optimization artifacts untouched.
It uses data only through 2026-03-20 for selection, then compares the selected
candidate against the existing champion on the 2026-03-21 to 2026-05-01
lockbox as a diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.phase_state import PhaseState, _atomic_write_json
from backtests.shared.auto.types import Experiment, GateCriterion, ScoredCandidate
from backtests.shared.auto.round_manager import RoundManager
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin
from backtests.stock.auto.iaric.phase_candidates import V5R2_PHASE_CANDIDATES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backtests" / "stock" / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "output"

CALIBRATION_START = date(2024, 1, 1)
CALIBRATION_END = date(2026, 3, 20)
LOCKBOX_START = date(2026, 3, 21)
LOCKBOX_END = date(2026, 5, 1)
EMBARGO_DAYS = 5
DELETE = "__DELETE__"

FOLDS = [
    ("2024Q3", date(2024, 7, 1), date(2024, 9, 30)),
    ("2024Q4", date(2024, 10, 1), date(2024, 12, 31)),
    ("2025Q1", date(2025, 1, 1), date(2025, 3, 31)),
    ("2025Q2", date(2025, 4, 1), date(2025, 6, 30)),
    ("2025Q3", date(2025, 7, 1), date(2025, 9, 30)),
    ("2025Q4", date(2025, 10, 1), date(2025, 12, 31)),
    ("2026Q1_PARTIAL", date(2026, 1, 1), CALIBRATION_END),
]

_WORKER_REPLAY = None
_WORKER_CONFIG = None
_WORKER_EQUITY = 10_000.0


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
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_latest_champion() -> tuple[dict[str, Any], Path, int]:
    manager = RoundManager("stock", "iaric")
    latest = manager.get_latest_round()
    if latest < 1:
        raise FileNotFoundError("No IARIC optimization round found.")
    path = manager.optimized_config_path(manager.round_path(latest))
    return _read_json(path), path, latest


def _load_round(round_num: int) -> tuple[dict[str, Any], Path] | None:
    manager = RoundManager("stock", "iaric")
    path = manager.optimized_config_path(manager.round_path(round_num))
    if not path.exists():
        return None
    return _read_json(path), path


def _config_signature(mutations: dict[str, Any]) -> str:
    raw = json.dumps(mutations, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _apply_patch_mutations(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if value == DELETE:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _clean_delete_markers(mutations: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mutations.items() if value != DELETE}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _init_eval_worker(data_dir: str, start_date: str, end_date: str, equity: float) -> None:
    global _WORKER_REPLAY, _WORKER_CONFIG, _WORKER_EQUITY

    if sys.stdout.encoding != "utf-8":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.data.replay_cache import load_research_replay_bundle

    _WORKER_EQUITY = float(equity)
    _WORKER_REPLAY = load_research_replay_bundle(Path(data_dir)).data
    _WORKER_CONFIG = IARICBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=_WORKER_EQUITY,
        tier=3,
        data_dir=Path(data_dir),
    )


def _evaluate_task(payload: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    name, mutations = payload
    try:
        from backtests.stock.auto.config_mutator import mutate_iaric_config
        from backtests.stock.auto.scoring import extract_metrics
        from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine

        config = mutate_iaric_config(_WORKER_CONFIG, mutations)
        result = IARICPullbackEngine(config, _WORKER_REPLAY, collect_diagnostics=False).run()
        metrics = extract_metrics(result.trades, result.equity_curve, result.timestamps, _WORKER_EQUITY)
        rows = [
            {
                "entry_date": trade.entry_time.date().isoformat(),
                "entry_time": trade.entry_time.isoformat(),
                "symbol": trade.symbol,
                "r": float(trade.r_multiple),
                "pnl_net": float(trade.pnl_net),
                "exit_reason": trade.exit_reason,
            }
            for trade in result.trades
        ]
        return {
            "name": name,
            "error": None,
            "metrics": _jsonable(metrics),
            "trades": rows,
        }
    except Exception:
        return {
            "name": name,
            "error": traceback.format_exc(),
            "metrics": {},
            "trades": [],
        }


def _evaluate_many(
    named_mutations: list[tuple[str, dict[str, Any]]],
    *,
    start: date,
    end: date,
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    if not named_mutations:
        return {}
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_eval_worker,
        initargs=(str(DATA_DIR), start.isoformat(), end.isoformat(), 10_000.0),
    ) as pool:
        futures = {
            pool.submit(_evaluate_task, item): item[0]
            for item in named_mutations
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = {
                    "name": name,
                    "error": traceback.format_exc(),
                    "metrics": {},
                    "trades": [],
                }
    return results


def _months(start: date, end: date) -> float:
    return max((end - start).days / 30.44, 0.1)


def _window_metrics(trades: list[dict[str, Any]], start: date, end: date) -> dict[str, Any]:
    rs = [
        float(trade["r"])
        for trade in trades
        if start <= date.fromisoformat(str(trade["entry_date"])) <= end
    ]
    months = _months(start, end)
    if not rs:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "max_drawdown_r": 0.0,
            "trades_per_month": 0.0,
            "months": months,
        }

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    gross_win = float(sum(wins))
    gross_loss = abs(float(sum(losses)))
    cum = np.cumsum(rs)
    drawdowns = np.maximum.accumulate(cum) - cum
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return {
        "total_trades": len(rs),
        "winning_trades": len(wins),
        "win_rate": len(wins) / len(rs),
        "profit_factor": pf,
        "net_r": float(sum(rs)),
        "avg_r": float(sum(rs) / len(rs)),
        "max_drawdown_r": float(np.max(drawdowns)) if len(drawdowns) else 0.0,
        "trades_per_month": len(rs) / months if months > 0 else 0.0,
        "months": months,
    }


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _finite_pf(value: Any) -> float:
    try:
        pf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(pf):
        return 4.0
    return pf


def _fold_score(metrics: dict[str, Any]) -> float:
    trades = float(metrics.get("total_trades", 0) or 0)
    if trades < 4:
        return 0.0

    pf = _finite_pf(metrics.get("profit_factor"))
    avg_r = float(metrics.get("avg_r", 0.0) or 0.0)
    net_r = float(metrics.get("net_r", 0.0) or 0.0)
    max_dd = float(metrics.get("max_drawdown_r", 0.0) or 0.0)

    avg_score = _clip01((avg_r + 0.05) / 0.24)
    pf_score = _clip01((pf - 0.80) / 1.50)
    net_score = _clip01((net_r + 1.0) / 26.0)
    dd_score = _clip01(1.0 - max_dd / 10.0)
    trade_score = _clip01(trades / 55.0)
    return (
        0.36 * net_score
        + 0.24 * trade_score
        + 0.18 * avg_score
        + 0.14 * pf_score
        + 0.08 * dd_score
    )


def _summarize_walkforward(evaluation: dict[str, Any]) -> dict[str, Any]:
    trades = evaluation.get("trades") or []
    fold_rows: list[dict[str, Any]] = []
    val_scores: list[float] = []
    train_scores: list[float] = []

    for name, val_start, val_end in FOLDS:
        train_start = CALIBRATION_START
        train_end = min(val_start - timedelta(days=EMBARGO_DAYS), CALIBRATION_END)
        score_start = min(val_start + timedelta(days=EMBARGO_DAYS), val_end)
        score_end = max(val_end - timedelta(days=EMBARGO_DAYS), score_start)
        train_metrics = _window_metrics(trades, train_start, train_end)
        valid_metrics = _window_metrics(trades, score_start, score_end)
        train_score = _fold_score(train_metrics)
        valid_score = _fold_score(valid_metrics)
        train_scores.append(train_score)
        val_scores.append(valid_score)
        fold_rows.append(
            {
                "fold": name,
                "train_start": train_start,
                "train_end": train_end,
                "valid_start": score_start,
                "valid_end": score_end,
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "train_score": train_score,
                "valid_score": valid_score,
                "score_decay": train_score - valid_score,
            }
        )

    full_metrics = evaluation.get("metrics") or {}
    mean_val = float(np.mean(val_scores)) if val_scores else 0.0
    median_val = float(np.median(val_scores)) if val_scores else 0.0
    worst_val = float(np.min(val_scores)) if val_scores else 0.0
    std_val = float(np.std(val_scores)) if val_scores else 0.0
    mean_train = float(np.mean(train_scores)) if train_scores else 0.0
    positive_folds = sum(
        1
        for fold in fold_rows
        if float(fold["valid_metrics"].get("avg_r", 0.0) or 0.0) > 0
        and float(fold["valid_metrics"].get("net_r", 0.0) or 0.0) > 0
    )
    total_validation_trades = sum(int(fold["valid_metrics"].get("total_trades", 0) or 0) for fold in fold_rows)
    total_validation_net_r = sum(float(fold["valid_metrics"].get("net_r", 0.0) or 0.0) for fold in fold_rows)
    decay_penalty = _clip01(max(0.0, mean_train - mean_val - 0.10))
    instability_penalty = _clip01(std_val / 0.25)
    trade_adequacy = _clip01(total_validation_trades / 250.0)
    positive_share = positive_folds / len(fold_rows) if fold_rows else 0.0
    validation_months = sum(float(fold["valid_metrics"].get("months", 0.0) or 0.0) for fold in fold_rows)
    validation_trades_per_month = total_validation_trades / validation_months if validation_months > 0 else 0.0
    validation_avg_r = total_validation_net_r / total_validation_trades if total_validation_trades > 0 else 0.0
    worst_validation_net_r = min(
        (float(fold["valid_metrics"].get("net_r", 0.0) or 0.0) for fold in fold_rows),
        default=0.0,
    )
    full_total_trades = float(full_metrics.get("total_trades", 0.0) or 0.0)
    full_trades_per_month = float(full_metrics.get("trades_per_month", 0.0) or 0.0)
    full_expected_r = float(full_metrics.get("avg_r", 0.0) or 0.0) * full_total_trades
    full_pf = _finite_pf(full_metrics.get("profit_factor"))
    full_avg_r = float(full_metrics.get("avg_r", 0.0) or 0.0)
    full_dd = float(full_metrics.get("max_drawdown_pct", 0.0) or 0.0)

    validation_alpha = (
        0.40 * math.tanh(max(total_validation_net_r, 0.0) / 95.0)
        + 0.25 * _clip01(validation_trades_per_month / 38.0)
        + 0.15 * _clip01((validation_avg_r + 0.03) / 0.22)
        + 0.10 * _clip01((worst_validation_net_r + 2.0) / 18.0)
        + 0.10 * positive_share
    )
    in_sample_alpha = (
        0.38 * math.tanh(max(full_expected_r, 0.0) / 130.0)
        + 0.27 * _clip01(full_trades_per_month / 38.0)
        + 0.17 * _clip01((full_avg_r + 0.02) / 0.22)
        + 0.12 * _clip01((full_pf - 0.9) / 1.4)
        + 0.06 * _clip01(1.0 - full_dd / 0.08)
    )
    robust_score = (
        0.58 * validation_alpha
        + 0.34 * in_sample_alpha
        + 0.05 * median_val
        + 0.03 * worst_val
        - 0.03 * decay_penalty
        - 0.02 * instability_penalty
    )

    return {
        "full_metrics": full_metrics,
        "folds": _jsonable(fold_rows),
        "robust_score": robust_score,
        "alpha_frequency_score": robust_score,
        "validation_alpha": validation_alpha,
        "in_sample_alpha": in_sample_alpha,
        "validation_trades_per_month": validation_trades_per_month,
        "validation_avg_r": validation_avg_r,
        "worst_validation_net_r": worst_validation_net_r,
        "full_expected_r": full_expected_r,
        "mean_validation_score": mean_val,
        "median_validation_score": median_val,
        "worst_validation_score": worst_val,
        "validation_score_std": std_val,
        "mean_train_score": mean_train,
        "positive_folds": positive_folds,
        "fold_count": len(fold_rows),
        "positive_fold_share": positive_share,
        "total_validation_trades": total_validation_trades,
        "total_validation_net_r": total_validation_net_r,
        "decay_penalty": decay_penalty,
        "instability_penalty": instability_penalty,
    }


def _fold_relative_summary(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_folds = candidate.get("folds") or []
    baseline_by_name = {
        str(fold.get("fold")): fold
        for fold in baseline.get("folds", [])
    }
    rows: list[dict[str, Any]] = []
    help_count = 0
    hurt_count = 0
    total_net_r_delta = 0.0
    total_trade_delta = 0.0

    for fold in candidate_folds:
        name = str(fold.get("fold"))
        base = baseline_by_name.get(name)
        if not base:
            continue
        cand_metrics = fold.get("valid_metrics") or {}
        base_metrics = base.get("valid_metrics") or {}
        score_delta = float(fold.get("valid_score", 0.0) or 0.0) - float(base.get("valid_score", 0.0) or 0.0)
        net_r_delta = float(cand_metrics.get("net_r", 0.0) or 0.0) - float(base_metrics.get("net_r", 0.0) or 0.0)
        trade_delta = float(cand_metrics.get("total_trades", 0.0) or 0.0) - float(base_metrics.get("total_trades", 0.0) or 0.0)
        helped = score_delta > 0.002 and (net_r_delta > 0.0 or (trade_delta > 0 and net_r_delta > -0.75))
        hurt = score_delta < -0.002 and net_r_delta < 0.0
        help_count += int(helped)
        hurt_count += int(hurt)
        total_net_r_delta += net_r_delta
        total_trade_delta += trade_delta
        rows.append({
            "fold": name,
            "score_delta": score_delta,
            "net_r_delta": net_r_delta,
            "trade_delta": trade_delta,
            "helped": helped,
            "hurt": hurt,
        })

    fold_count = len(rows) or 1
    return {
        "folds": rows,
        "help_count": help_count,
        "hurt_count": hurt_count,
        "help_share": help_count / fold_count,
        "hurt_share": hurt_count / fold_count,
        "total_validation_net_r_delta": total_net_r_delta,
        "total_validation_trades_delta": total_trade_delta,
    }


def _candidate_catalog(current_champion: dict[str, Any], round1: dict[str, Any] | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, group: str, patch: dict[str, Any], note: str = "") -> None:
        signature = json.dumps(patch, sort_keys=True, default=str)
        if signature in seen or not patch:
            return
        seen.add(signature)
        candidates.append({
            "name": _safe_name(name),
            "group": group,
            "patch": patch,
            "note": note,
        })

    for phase, phase_candidates in sorted(V5R2_PHASE_CANDIDATES.items()):
        for name, patch in phase_candidates:
            add(f"v5r2_p{phase}_{name}", f"v5r2_phase_{phase}", dict(patch), "Existing V5R2 maintenance candidate.")

    if round1:
        diff_keys = sorted(set(current_champion) | set(round1))
        rollback_all: dict[str, Any] = {}
        for key in diff_keys:
            old = round1.get(key, DELETE)
            new = current_champion.get(key, DELETE)
            if old == new:
                continue
            patch = {key: old}
            rollback_all[key] = old
            add(f"rollback_one_{key}", "old_result_rollback", patch, "Single-key rollback toward round_1.")

        groups = {
            "rollback_signal_aperture": [
                "param_overrides.pb_v2_signal_floor",
                "param_overrides.pb_v2_gap_max_pct",
                "param_overrides.pb_cdd_max",
            ],
            "rollback_delayed_confirm": [
                "param_overrides.pb_delayed_confirm_after_bar",
                "param_overrides.pb_delayed_confirm_score_min",
            ],
            "rollback_exit_timing": [
                "param_overrides.pb_v2_partial_profit_trigger_r",
                "param_overrides.pb_v2_partial_profit_remainder_stop_r",
                "param_overrides.pb_v2_stale_bars",
                "param_overrides.pb_v2_stale_mfe_thresh",
            ],
            "rollback_disabled_routes": [
                "param_overrides.pb_v2_vwap_bounce_enabled",
                "param_overrides.pb_v2_afternoon_retest_enabled",
            ],
            "rollback_capacity": ["max_per_sector"],
        }
        for name, keys in groups.items():
            patch = {key: rollback_all[key] for key in keys if key in rollback_all}
            add(name, "old_result_bundle", patch, "Grouped rollback based on round_1 to round_2 differences.")
        add("rollback_full_round1", "old_result_bundle", rollback_all, "Full rollback to the previous optimized round.")

    conservative = [
        ("floor_75_capacity_2", {
            "param_overrides.pb_v2_signal_floor": 75.0,
            "max_per_sector": 2,
        }),
        ("floor_75_gap_1", {
            "param_overrides.pb_v2_signal_floor": 75.0,
            "param_overrides.pb_v2_gap_max_pct": 1.0,
        }),
        ("partial_030_stale6", {
            "param_overrides.pb_v2_partial_profit_trigger_r": 0.3,
            "param_overrides.pb_v2_stale_bars": 6,
        }),
        ("disable_delayed_confirm", {
            "param_overrides.pb_delayed_confirm_enabled": False,
        }),
        ("capacity_3", {
            "max_per_sector": 3,
        }),
    ]
    for name, patch in conservative:
        add(f"conservative_{name}", "conservative_plateau", patch, "Small conservative plateau candidate.")

    return candidates


def _rank_candidates(
    evaluations: dict[str, dict[str, Any]],
    catalog_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for name, evaluation in evaluations.items():
        candidate = dict(catalog_by_name.get(name, {}))
        candidate["name"] = name
        candidate["error"] = evaluation.get("error")
        if evaluation.get("error"):
            candidate["summary"] = {}
            candidate["score"] = -1e9
        else:
            summary = _summarize_walkforward(evaluation)
            candidate["summary"] = summary
            candidate["score"] = float(summary["robust_score"])
        ranked.append(candidate)
    ranked.sort(key=lambda item: float(item.get("score", -1e9)), reverse=True)
    return ranked


def _accept_candidate(best: dict[str, Any], current: dict[str, Any], *, min_delta: float) -> tuple[bool, str]:
    if best.get("error"):
        return False, "best candidate errored"
    best_summary = best.get("summary") or {}
    current_summary = current.get("summary") or {}
    best_score = float(best_summary.get("robust_score", 0.0) or 0.0)
    current_score = float(current_summary.get("robust_score", 0.0) or 0.0)
    delta = best_score - current_score
    if delta < min_delta:
        return False, f"score delta {delta:.4f} below min_delta {min_delta:.4f}"

    if float(best_summary.get("positive_fold_share", 0.0) or 0.0) < float(current_summary.get("positive_fold_share", 0.0) or 0.0):
        return False, "positive validation fold share worsened"

    best_full = best_summary.get("full_metrics") or {}
    current_full = current_summary.get("full_metrics") or {}
    best_pf = _finite_pf(best_full.get("profit_factor"))
    current_pf = _finite_pf(current_full.get("profit_factor"))
    if current_pf > 0 and best_pf < current_pf * 0.90:
        return False, f"full calibration PF fell too far ({best_pf:.2f} vs {current_pf:.2f})"

    best_trades = float(best_full.get("total_trades", 0.0) or 0.0)
    current_trades = float(current_full.get("total_trades", 0.0) or 0.0)
    if current_trades > 0 and best_trades < current_trades * 0.75:
        return False, f"full calibration trade count fell too far ({best_trades:.0f} vs {current_trades:.0f})"

    return True, f"accepted robust score delta {delta:.4f}"


def _round1_rollback_candidates_by_phase(
    champion: dict[str, Any],
    round1: dict[str, Any] | None,
) -> dict[int, list[Experiment]]:
    if not round1:
        return {}

    keys_by_phase = {
        1: [
            "param_overrides.pb_v2_partial_profit_trigger_r",
            "param_overrides.pb_v2_stale_bars",
            "param_overrides.pb_v2_stale_mfe_thresh",
            "param_overrides.pb_v2_vwap_bounce_enabled",
            "param_overrides.pb_v2_afternoon_retest_enabled",
        ],
        2: [
            "param_overrides.pb_delayed_confirm_after_bar",
            "param_overrides.pb_delayed_confirm_score_min",
        ],
        3: [
            "param_overrides.pb_v2_signal_floor",
            "max_per_sector",
        ],
        4: [
            "param_overrides.pb_v2_partial_profit_trigger_r",
            "param_overrides.pb_v2_stale_bars",
            "param_overrides.pb_v2_stale_mfe_thresh",
            "param_overrides.pb_v2_vwap_bounce_enabled",
            "param_overrides.pb_v2_afternoon_retest_enabled",
            "param_overrides.pb_v2_signal_floor",
            "max_per_sector",
        ],
    }
    result: dict[int, list[Experiment]] = {}
    for phase, keys in keys_by_phase.items():
        experiments: list[Experiment] = []
        bundle: dict[str, Any] = {}
        for key in keys:
            if key not in round1:
                continue
            if champion.get(key) == round1.get(key):
                continue
            patch = {key: round1[key]}
            short = key.replace("param_overrides.", "").replace("pb_", "")
            experiments.append(Experiment(f"old_result_rollback_{short}", patch))
            bundle[key] = round1[key]
        if len(bundle) > 1:
            experiments.append(Experiment(f"old_result_phase_{phase}_bundle", bundle))
        if experiments:
            result[phase] = experiments
    return result


def _ablation_candidates_by_phase() -> dict[int, list[Experiment]]:
    """High-signal one-at-a-time rule relaxations/removals."""
    return {
        1: [
            Experiment("ablate_open_scored_carry", {
                "param_overrides.pb_open_scored_carry_close_pct_min": 999.0,
                "param_overrides.pb_open_scored_carry_mfe_gate_r": 999.0,
            }),
            Experiment("ablate_all_route_carry", {
                "param_overrides.pb_carry_close_pct_min": 999.0,
                "param_overrides.pb_carry_mfe_gate_r": 999.0,
                "param_overrides.pb_open_scored_carry_close_pct_min": 999.0,
                "param_overrides.pb_open_scored_carry_mfe_gate_r": 999.0,
                "param_overrides.pb_delayed_confirm_carry_close_pct_min": 999.0,
                "param_overrides.pb_delayed_confirm_carry_mfe_gate_r": 999.0,
            }),
            Experiment("ablate_fast_partial_profit", {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.30,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": DELETE,
            }),
            Experiment("ablate_stale_tightening", {
                "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
                "param_overrides.pb_v2_stale_bars": 6,
            }),
        ],
        2: [
            Experiment("ablate_delayed_confirm_route", {
                "param_overrides.pb_delayed_confirm_enabled": False,
            }),
            Experiment("relax_delayed_confirm_score", {
                "param_overrides.pb_delayed_confirm_score_min": 47.0,
            }),
            Experiment("tighten_delayed_confirm_score", {
                "param_overrides.pb_delayed_confirm_score_min": 57.0,
            }),
        ],
        3: [
            Experiment("ablate_crowded_day_limit", {
                "param_overrides.pb_cdd_max": 999,
            }),
            Experiment("restore_signal_floor_75", {
                "param_overrides.pb_v2_signal_floor": 75.0,
            }),
            Experiment("tighten_signal_floor_78", {
                "param_overrides.pb_v2_signal_floor": 78.0,
            }),
            Experiment("ablate_sector_capacity_expansion", {
                "max_per_sector": 2,
            }),
            Experiment("moderate_sector_capacity", {
                "max_per_sector": 3,
            }),
        ],
        4: [
            Experiment("ablate_round2_exit_bundle", {
                "param_overrides.pb_v2_partial_profit_trigger_r": 0.30,
                "param_overrides.pb_v2_partial_profit_remainder_stop_r": DELETE,
                "param_overrides.pb_v2_stale_mfe_thresh": 0.05,
                "param_overrides.pb_v2_stale_bars": 6,
            }),
            Experiment("ablate_round2_route_disable_bundle", {
                "param_overrides.pb_v2_vwap_bounce_enabled": True,
                "param_overrides.pb_v2_afternoon_retest_enabled": True,
            }),
            Experiment("defensive_frequency_balance", {
                "param_overrides.pb_v2_signal_floor": 75.0,
                "max_per_sector": 3,
                "param_overrides.pb_delayed_confirm_score_min": 47.0,
            }),
        ],
    }


def _curate_phase_candidates(phase: int, candidates: list[Experiment]) -> list[Experiment]:
    allow: dict[int, set[str]] = {
        1: {
            "open_carry_off",
            "all_route_carry_off",
            "profit_lock_050",
            "old_result_phase_1_bundle",
            "old_result_rollback_v2_partial_profit_trigger_r",
            "old_result_rollback_v2_stale_bars",
            "ablate_open_scored_carry",
            "ablate_all_route_carry",
            "ablate_fast_partial_profit",
            "ablate_stale_tightening",
        },
        2: {
            "delayed_score_47",
            "delayed_score_57",
            "delayed_bar7_score47",
            "delayed_carry_relaxed_045_010",
            "ablate_delayed_confirm_route",
            "relax_delayed_confirm_score",
            "tighten_delayed_confirm_score",
        },
        3: {
            "gap_max_1",
            "gap_down_focus",
            "max_pos_11",
            "old_result_phase_3_bundle",
            "ablate_crowded_day_limit",
            "restore_signal_floor_75",
            "tighten_signal_floor_78",
            "ablate_sector_capacity_expansion",
            "moderate_sector_capacity",
        },
        4: {
            "carry_quality_delayed47",
            "carry_quality_capacity11",
            "defensive_gap_carry_quality",
            "old_result_phase_4_bundle",
            "ablate_round2_exit_bundle",
            "ablate_round2_route_disable_bundle",
            "defensive_frequency_balance",
        },
    }
    selected = allow.get(phase)
    if not selected:
        return candidates
    curated = [candidate for candidate in candidates if candidate.name in selected]
    return curated or candidates


class _WalkForwardAlphaEvaluator:
    def __init__(self, *, max_workers: int, cache: dict[str, dict[str, Any]], cache_path: Path | None = None):
        self.max_workers = max_workers
        self.cache = cache
        self.cache_path = cache_path

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        current_clean = _clean_delete_markers(current_mutations)
        current_key = _config_signature(current_clean)
        pending: list[tuple[str, dict[str, Any], str]] = []
        if current_key not in self.cache:
            pending.append(("__phase_current__", current_clean, current_key))

        for candidate in candidates:
            mutations = _clean_delete_markers(_apply_patch_mutations(current_mutations, candidate.mutations))
            key = _config_signature(mutations)
            if key not in self.cache:
                pending.append((candidate.name, mutations, key))

        if pending:
            self._evaluate_pending(pending)

        baseline_summary = self.cache.get(current_key, {}).get("summary", {})
        results: list[ScoredCandidate] = []
        for candidate in candidates:
            mutations = _clean_delete_markers(_apply_patch_mutations(current_mutations, candidate.mutations))
            key = _config_signature(mutations)
            results.append(self._score_cached(candidate.name, self.cache[key], baseline_summary, key == current_key))
        return results

    def _evaluate_pending(self, pending: list[tuple[str, dict[str, Any], str]]) -> None:
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_eval_worker,
            initargs=(str(DATA_DIR), CALIBRATION_START.isoformat(), CALIBRATION_END.isoformat(), 10_000.0),
        ) as pool:
            futures = {
                pool.submit(_evaluate_task, (name, mutations)): (name, key)
                for name, mutations, key in pending
            }
            for future in as_completed(futures):
                name, key = futures[future]
                try:
                    evaluation = future.result()
                except Exception:
                    evaluation = {
                        "name": name,
                        "error": traceback.format_exc(),
                        "metrics": {},
                        "trades": [],
                    }
                if evaluation.get("error"):
                    self.cache[key] = {"name": name, "error": evaluation["error"], "summary": {}}
                else:
                    self.cache[key] = {
                        "name": name,
                        "error": None,
                        "summary": _summarize_walkforward(evaluation),
                    }
                self._save_cache()

    def _save_cache(self) -> None:
        if self.cache_path is not None:
            _atomic_write_json(_jsonable(self.cache), self.cache_path)

    @staticmethod
    def _score_cached(
        name: str,
        cached: dict[str, Any],
        baseline_summary: dict[str, Any],
        is_current: bool,
    ) -> ScoredCandidate:
        if cached.get("error"):
            return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=str(cached["error"])[:1000])

        summary = dict(cached.get("summary") or {})
        full_metrics = dict(summary.get("full_metrics") or {})
        validation_net_r = float(summary.get("total_validation_net_r", 0.0) or 0.0)
        positive_folds = int(summary.get("positive_folds", 0) or 0)
        validation_trades = int(summary.get("total_validation_trades", 0) or 0)

        rel = _fold_relative_summary(summary, baseline_summary) if baseline_summary and not is_current else {}
        help_count = int(rel.get("help_count", 0) or 0)
        hurt_count = int(rel.get("hurt_count", 0) or 0)
        delta_net_r = float(rel.get("total_validation_net_r_delta", 0.0) or 0.0)
        delta_trades = float(rel.get("total_validation_trades_delta", 0.0) or 0.0)
        relative_bonus = (
            0.035 * float(rel.get("help_share", 0.0) or 0.0)
            + 0.025 * math.tanh(max(delta_net_r, 0.0) / 25.0)
            + 0.015 * math.tanh(max(delta_trades, 0.0) / 45.0)
            - 0.025 * float(rel.get("hurt_share", 0.0) or 0.0)
            if rel else 0.0
        )

        reject_reason = ""
        if is_current:
            reject_reason = ""
        elif validation_trades < 120:
            reject_reason = f"validation_frequency_too_low ({validation_trades} < 120)"
        elif validation_net_r <= 0.0:
            reject_reason = f"validation_net_r_non_positive ({validation_net_r:.2f})"
        elif positive_folds < 4:
            reject_reason = f"not_enough_positive_forward_folds ({positive_folds} < 4)"
        elif help_count <= 2:
            reject_reason = f"fold_help_count_too_low ({help_count} <= 2)"
        elif help_count < 4 and delta_net_r < 6.0:
            reject_reason = f"weak_fold_support ({help_count}/7, delta_net_r={delta_net_r:.2f})"
        elif hurt_count >= 4 and delta_net_r <= 0.0:
            reject_reason = f"too_many_hurt_folds ({hurt_count}/7)"
        elif _finite_pf(full_metrics.get("profit_factor")) < 1.0:
            reject_reason = f"full_calibration_pf_too_low ({_finite_pf(full_metrics.get('profit_factor')):.2f} < 1.00)"

        score = float(summary["alpha_frequency_score"]) + relative_bonus
        metrics = dict(full_metrics)
        metrics.update({
            "wf_alpha_frequency_score": summary["alpha_frequency_score"],
            "wf_relative_alpha_frequency_score": score,
            "wf_validation_alpha": summary["validation_alpha"],
            "wf_in_sample_alpha": summary["in_sample_alpha"],
            "wf_positive_folds": positive_folds,
            "wf_fold_count": summary["fold_count"],
            "wf_total_validation_trades": validation_trades,
            "wf_total_validation_net_r": validation_net_r,
            "wf_validation_trades_per_month": summary["validation_trades_per_month"],
            "wf_validation_avg_r": summary["validation_avg_r"],
            "wf_worst_validation_net_r": summary["worst_validation_net_r"],
            "wf_mean_validation_score": summary["mean_validation_score"],
            "wf_worst_validation_score": summary["worst_validation_score"],
            "wf_help_count": help_count,
            "wf_hurt_count": hurt_count,
            "wf_total_validation_net_r_delta": delta_net_r,
            "wf_total_validation_trades_delta": delta_trades,
        })
        return ScoredCandidate(
            name=name,
            score=score,
            rejected=bool(reject_reason),
            reject_reason=reject_reason,
            metrics=_jsonable(metrics),
        )

    def close(self) -> None:
        return None


class IARICWalkForwardAlphaPlugin(IARICPullbackPlugin):
    def __init__(
        self,
        data_dir: Path,
        *,
        max_workers: int,
        round1_mutations: dict[str, Any] | None,
        cache_path: Path | None = None,
    ):
        super().__init__(
            data_dir=data_dir,
            start_date=CALIBRATION_START.isoformat(),
            end_date=CALIBRATION_END.isoformat(),
            initial_equity=10_000.0,
            max_workers=max_workers,
            num_phases=4,
            profile="mainline",
            round_name="v5r2",
        )
        self.name = "iaric_v5r2_walkforward_alpha_frequency"
        self.max_workers = max_workers
        self._wf_cache_path = cache_path
        self._wf_candidate_cache: dict[str, dict[str, Any]] = self._load_candidate_cache(cache_path)
        self._wf_metrics_cache: dict[str, dict[str, Any]] = {}
        self._round1_mutations = dict(round1_mutations or {})
        self._rollback_candidates_by_phase: dict[int, list[Experiment]] = {}

    @staticmethod
    def _load_candidate_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
        if cache_path is None or not cache_path.exists():
            return {}
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def set_champion_seed(self, champion: dict[str, Any]) -> None:
        self.initial_mutations = dict(champion)
        self._rollback_candidates_by_phase = _round1_rollback_candidates_by_phase(
            self.initial_mutations,
            self._round1_mutations,
        )

    def get_phase_spec(self, phase: int, state: PhaseState):
        spec = super().get_phase_spec(phase, state)
        extra = [
            *self._rollback_candidates_by_phase.get(phase, []),
            *_ablation_candidates_by_phase().get(phase, []),
        ]
        if extra:
            existing = {candidate.name for candidate in spec.candidates}
            candidates = list(spec.candidates) + [candidate for candidate in extra if candidate.name not in existing]
        else:
            candidates = list(spec.candidates)
        candidates = _curate_phase_candidates(phase, candidates)
        return replace(
            spec,
            candidates=candidates,
            hard_rejects={},
            prune_threshold=0.0,
            reject_streak_limit=2,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        del phase, cumulative_mutations, scoring_weights, hard_rejects
        return _WalkForwardAlphaEvaluator(
            max_workers=int(self.max_workers or 1),
            cache=self._wf_candidate_cache,
            cache_path=self._wf_cache_path,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        mutations = _clean_delete_markers(mutations)
        metrics = super().compute_final_metrics(mutations)
        key = _config_signature(mutations)
        if key not in self._wf_metrics_cache:
            evaluation = _evaluate_many(
                [("__metrics__", mutations)],
                start=CALIBRATION_START,
                end=CALIBRATION_END,
                max_workers=int(self.max_workers or 1),
            )["__metrics__"]
            if evaluation.get("error"):
                self._wf_metrics_cache[key] = {"wf_error": evaluation["error"]}
            else:
                summary = _summarize_walkforward(evaluation)
                self._wf_metrics_cache[key] = {
                    "wf_alpha_frequency_score": summary["alpha_frequency_score"],
                    "wf_validation_alpha": summary["validation_alpha"],
                    "wf_in_sample_alpha": summary["in_sample_alpha"],
                    "wf_positive_folds": summary["positive_folds"],
                    "wf_fold_count": summary["fold_count"],
                    "wf_total_validation_trades": summary["total_validation_trades"],
                    "wf_total_validation_net_r": summary["total_validation_net_r"],
                    "wf_validation_trades_per_month": summary["validation_trades_per_month"],
                    "wf_validation_avg_r": summary["validation_avg_r"],
                    "wf_worst_validation_net_r": summary["worst_validation_net_r"],
                    "wf_mean_validation_score": summary["mean_validation_score"],
                    "wf_worst_validation_score": summary["worst_validation_score"],
                }
        metrics.update(_jsonable(self._wf_metrics_cache[key]))
        return metrics

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        del phase, state
        baseline = self._baseline_metrics()
        base_trades = float(baseline.get("total_trades", 0.0) or 0.0)
        base_expected_r = float(baseline.get("avg_r", 0.0) or 0.0) * base_trades
        trades = float(metrics.get("total_trades", 0.0) or 0.0)
        expected_r = float(metrics.get("avg_r", 0.0) or 0.0) * trades
        pf = _finite_pf(metrics.get("profit_factor"))
        max_dd = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
        validation_net_r = float(metrics.get("wf_total_validation_net_r", 0.0) or 0.0)
        validation_trades = float(metrics.get("wf_total_validation_trades", 0.0) or 0.0)
        positive_folds = float(metrics.get("wf_positive_folds", 0.0) or 0.0)
        return [
            GateCriterion("is_expected_total_r_retention", base_expected_r * 0.90, expected_r, expected_r >= base_expected_r * 0.90),
            GateCriterion("is_trade_count_retention", base_trades * 0.85, trades, trades >= base_trades * 0.85),
            GateCriterion("is_profit_factor_floor", 1.10, pf, pf >= 1.10),
            GateCriterion("is_max_drawdown_pct", 0.08, max_dd, max_dd <= 0.08),
            GateCriterion("wf_validation_net_r_positive", 0.0, validation_net_r, validation_net_r > 0.0),
            GateCriterion("wf_validation_trades", 120.0, validation_trades, validation_trades >= 120.0),
            GateCriterion("wf_positive_folds", 4.0, positive_folds, positive_folds >= 4.0),
        ]


def _run_greedy_walkforward(
    champion: dict[str, Any],
    catalog: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_workers: int,
    max_rounds: int,
    min_delta: float,
) -> dict[str, Any]:
    current_mutations = dict(champion)
    accepted: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    remaining = list(catalog)

    baseline_eval = _evaluate_many(
        [("__current__", current_mutations)],
        start=CALIBRATION_START,
        end=CALIBRATION_END,
        max_workers=max_workers,
    )["__current__"]
    current = {
        "name": "__current__",
        "patch": {},
        "summary": _summarize_walkforward(baseline_eval),
    }

    _atomic_write_json(_jsonable(current), output_dir / "baseline_walkforward.json")

    for round_num in range(1, max_rounds + 1):
        named: list[tuple[str, dict[str, Any]]] = []
        catalog_by_eval_name: dict[str, dict[str, Any]] = {}
        seen_configs = {_config_signature(current_mutations)}

        for candidate in remaining:
            mutated = _apply_patch_mutations(current_mutations, candidate["patch"])
            signature = _config_signature(mutated)
            if signature in seen_configs:
                continue
            seen_configs.add(signature)
            eval_name = f"r{round_num}_{candidate['name']}"
            catalog_by_eval_name[eval_name] = {**candidate, "final_mutations": mutated}
            named.append((eval_name, mutated))

        if not named:
            rounds.append({"round": round_num, "stop_reason": "no_remaining_distinct_candidates"})
            break

        started = time.time()
        evaluations = _evaluate_many(
            named,
            start=CALIBRATION_START,
            end=CALIBRATION_END,
            max_workers=max_workers,
        )
        ranked = _rank_candidates(evaluations, catalog_by_eval_name)
        best = ranked[0] if ranked else {}
        keep, reason = _accept_candidate(best, current, min_delta=min_delta) if best else (False, "no candidates")
        round_payload = {
            "round": round_num,
            "elapsed_seconds": round(time.time() - started, 2),
            "current_score": current["summary"]["robust_score"],
            "candidate_count": len(named),
            "best": best,
            "accepted": keep,
            "decision_reason": reason,
            "top_candidates": ranked[:10],
        }
        rounds.append(round_payload)
        _atomic_write_json(_jsonable(round_payload), output_dir / f"round_{round_num}_walkforward.json")

        if not keep:
            break

        accepted.append(best)
        current_mutations = dict(best["final_mutations"])
        current = {
            "name": best["name"],
            "patch": best["patch"],
            "summary": best["summary"],
        }
        accepted_names = {best["name"].split(f"r{round_num}_", 1)[-1]}
        remaining = [candidate for candidate in remaining if candidate["name"] not in accepted_names]

    return {
        "accepted": accepted,
        "rounds": rounds,
        "final_mutations": current_mutations,
        "final_summary": current["summary"],
        "baseline_summary": _summarize_walkforward(baseline_eval),
    }


def _compare_configs(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    *,
    max_workers: int,
) -> dict[str, Any]:
    evaluations = _evaluate_many(
        [("existing_champion", champion), ("new_walkforward_candidate", challenger)],
        start=CALIBRATION_START,
        end=LOCKBOX_END,
        max_workers=max_workers,
    )
    comparison: dict[str, Any] = {}
    for name, evaluation in evaluations.items():
        trades = evaluation.get("trades") or []
        comparison[name] = {
            "error": evaluation.get("error"),
            "full_metrics": evaluation.get("metrics") or {},
            "is_metrics": _window_metrics(trades, CALIBRATION_START, CALIBRATION_END),
            "oos_metrics": _window_metrics(trades, LOCKBOX_START, LOCKBOX_END),
            "walkforward": _summarize_walkforward({
                "metrics": evaluation.get("metrics") or {},
                "trades": [
                    trade for trade in trades
                    if date.fromisoformat(str(trade["entry_date"])) <= CALIBRATION_END
                ],
            }),
        }
    return comparison


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    if value == "inf":
        return "inf"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _format_markdown(summary: dict[str, Any]) -> str:
    comparison = summary.get("comparison") or {}
    old = comparison.get("existing_champion") or {}
    new = comparison.get("new_walkforward_candidate") or {}
    accepted = summary.get("optimization") or {}
    accepted_items = accepted.get("accepted") or []
    old_is = old.get("is_metrics") or {}
    old_oos = old.get("oos_metrics") or {}
    new_is = new.get("is_metrics") or {}
    new_oos = new.get("oos_metrics") or {}
    old_wf = (old.get("walkforward") or {})
    new_wf = (new.get("walkforward") or {})

    lines = [
        "# IARIC Purged Walk-Forward Challenger",
        "",
        f"- Selection data: {CALIBRATION_START.isoformat()} to {CALIBRATION_END.isoformat()}",
        f"- Lockbox comparison only: {LOCKBOX_START.isoformat()} to {LOCKBOX_END.isoformat()}",
        f"- Embargo: {EMBARGO_DAYS} calendar days around validation folds",
        f"- Existing champion: `{summary.get('champion_config_path')}`",
        f"- Challenger config: `{summary.get('challenger_config_path')}`",
        "",
        "## Optimization Decision",
        "",
    ]
    if accepted_items:
        lines.append(f"Accepted {len(accepted_items)} mutation set(s):")
        for item in accepted_items:
            patch = item.get("patch") or {}
            lines.append(f"- {item.get('name')}: `{patch}`")
    else:
        lines.append("No mutation set cleared the walk-forward robustness gate; the challenger is identical to the existing champion.")

    lines.extend([
        "",
        "## Existing vs New",
        "",
        "| Config | WF score | Pos folds | Worst fold | IS trades | IS PF | IS avgR | IS netR | OOS trades | OOS PF | OOS avgR | OOS netR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Existing champion | {_fmt(old_wf.get('robust_score'), 4)} | "
            f"{old_wf.get('positive_folds', '')}/{old_wf.get('fold_count', '')} | "
            f"{_fmt(old_wf.get('worst_validation_score'), 4)} | "
            f"{old_is.get('total_trades', '')} | {_fmt(old_is.get('profit_factor'))} | "
            f"{_fmt(old_is.get('avg_r'), 3)} | {_fmt(old_is.get('net_r'))} | "
            f"{old_oos.get('total_trades', '')} | {_fmt(old_oos.get('profit_factor'))} | "
            f"{_fmt(old_oos.get('avg_r'), 3)} | {_fmt(old_oos.get('net_r'))} |"
        ),
        (
            f"| New candidate | {_fmt(new_wf.get('robust_score'), 4)} | "
            f"{new_wf.get('positive_folds', '')}/{new_wf.get('fold_count', '')} | "
            f"{_fmt(new_wf.get('worst_validation_score'), 4)} | "
            f"{new_is.get('total_trades', '')} | {_fmt(new_is.get('profit_factor'))} | "
            f"{_fmt(new_is.get('avg_r'), 3)} | {_fmt(new_is.get('net_r'))} | "
            f"{new_oos.get('total_trades', '')} | {_fmt(new_oos.get('profit_factor'))} | "
            f"{_fmt(new_oos.get('avg_r'), 3)} | {_fmt(new_oos.get('net_r'))} |"
        ),
        "",
        "## Candidate Universe",
        "",
        f"- Total candidates considered per first round: {summary.get('candidate_count')}",
        "- Included curated V5R2 phase candidates, fold-scored one-at-a-time ablations, and round_1 rollback probes.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.005)
    args = parser.parse_args()

    max_workers = max(1, int(args.max_workers))
    output_dir = Path(args.output_root) if args.output_root else OUTPUT_ROOT / f"iaric_walkforward_reopt_{_now_label()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    champion, champion_path, champion_round = _load_latest_champion()
    round1_payload = _load_round(1)
    round1 = round1_payload[0] if round1_payload else None
    plugin = IARICWalkForwardAlphaPlugin(
        DATA_DIR,
        max_workers=max_workers,
        round1_mutations=round1,
        cache_path=output_dir / "candidate_eval_cache.json",
    )
    plugin.set_champion_seed(champion)

    candidate_catalog: dict[str, Any] = {}
    for phase in range(1, plugin.num_phases + 1):
        spec = plugin.get_phase_spec(phase, PhaseState(cumulative_mutations=dict(champion)))
        candidate_catalog[str(phase)] = [
            {"name": candidate.name, "mutations": candidate.mutations}
            for candidate in spec.candidates
        ]
    _atomic_write_json(_jsonable(candidate_catalog), output_dir / "candidate_catalog.json")

    started = time.time()

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="iaric_v5r2_walkforward_alpha_frequency",
        max_rounds=int(args.max_rounds),
        min_delta=float(args.min_delta),
        max_retries=0,
        max_diagnostic_retries=0,
    )
    state = runner.run_all_phases()

    challenger = _clean_delete_markers(dict(state.cumulative_mutations))
    challenger_path = output_dir / "iaric_walkforward_challenger_config.json"
    _atomic_write_json(_jsonable(challenger), challenger_path)

    accepted: list[dict[str, Any]] = []
    for phase in state.completed_phases:
        result = state.phase_results.get(phase, {})
        phase_new_mutations = result.get("new_mutations", {})
        if not result.get("applied_phase_mutations") or not phase_new_mutations:
            continue
        accepted.append({
            "name": f"phase_{phase}",
            "patch": phase_new_mutations,
            "kept_features": result.get("kept_features", []),
            "score": result.get("final_score"),
            "base_score": result.get("base_score"),
            "gate": state.phase_gate_results.get(phase, {}),
        })

    comparison = _compare_configs(champion, challenger, max_workers=max_workers)
    optimization = {
        "accepted": accepted,
        "phase_state_path": str(runner.state_path),
        "completed_phases": list(state.completed_phases),
        "final_mutations": challenger,
        "baseline_summary": comparison.get("existing_champion", {}).get("walkforward", {}),
        "final_summary": comparison.get("new_walkforward_candidate", {}).get("walkforward", {}),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_dir),
        "selection_start": CALIBRATION_START,
        "selection_end": CALIBRATION_END,
        "lockbox_start": LOCKBOX_START,
        "lockbox_end": LOCKBOX_END,
        "embargo_days": EMBARGO_DAYS,
        "max_workers": max_workers,
        "champion_round": champion_round,
        "champion_config_path": str(champion_path),
        "round1_config_path": str(round1_payload[1]) if round1_payload else None,
        "candidate_count": sum(len(items) for items in candidate_catalog.values()),
        "candidate_catalog_path": str(output_dir / "candidate_catalog.json"),
        "optimization": optimization,
        "challenger_config_path": str(challenger_path),
        "new_mutations": {
            key: value
            for key, value in challenger.items()
            if champion.get(key) != value
        },
        "removed_mutations": [
            key for key in champion
            if key not in challenger
        ],
        "comparison": comparison,
        "phase_state_path": str(runner.state_path),
        "round_evaluation_path": str(output_dir / "round_evaluation.txt"),
        "round_diagnostics_path": str(output_dir / "round_final_diagnostics.txt"),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    _atomic_write_json(_jsonable(summary), output_dir / "summary.json")
    (output_dir / "summary.md").write_text(_format_markdown(_jsonable(summary)), encoding="utf-8")
    print(f"Done. Summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
