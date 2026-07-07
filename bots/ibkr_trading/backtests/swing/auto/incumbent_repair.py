"""Incumbent repair runner for non-ATRSS swing strategies.

This is the companion to ATRSS' bespoke repair pass. It evaluates each
strategy with its native engine, then performs bounded granular ablation,
local perturbation, and repo-native targeted candidate probes. The OOS window
is used for selection, so reports call it selection OOS rather than an
untouched holdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.validation.oos_validation import (
    BACKTEST_START,
    BACKTEST_START_DATE,
    IS_BASELINES,
    LAST_SEEN_DATA_DATE,
    OOS_CUTOFF,
    OOS_CUTOFF_DATE,
    WindowMetrics,
    _assess,
    _coerce_json_config_values,
    _collect_symbol_trades,
    _compute_oos_months,
    _get_entry_time,
    _get_r_multiple,
    _to_naive_utc,
    _window_months,
    _with_backtest_window,
    compute_window_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MISSING = "__INCUMBENT_REPAIR_MISSING__"

STRATEGY_CONFIGS = {
    "helix_swing": PROJECT_ROOT / "backtests/output/swing/helix/round_2/optimized_config.json",
    "breakout": PROJECT_ROOT / "backtests/output/swing/breakout/round_5/optimized_config.json",
    "brs": PROJECT_ROOT / "backtests/output/swing/brs/round_1/optimized_config.json",
    "helix_momentum": PROJECT_ROOT / "backtests/output/momentum/helix/round_5/optimized_config.json",
}

CONFIG_OVERRIDES: dict[str, Path] = {}

STRATEGY_ROUND_ROOTS = {
    "helix_swing": PROJECT_ROOT / "backtests/output/swing/helix",
    "breakout": PROJECT_ROOT / "backtests/output/swing/breakout",
    "brs": PROJECT_ROOT / "backtests/output/swing/brs",
    "helix_momentum": PROJECT_ROOT / "backtests/output/momentum/helix",
}

STRATEGY_FAMILY = {
    "helix_swing": "swing",
    "breakout": "swing",
    "brs": "swing",
    "helix_momentum": "momentum",
}

DEFAULT_STRATEGIES = ["helix_swing", "breakout", "brs", "helix_momentum"]


@dataclass(frozen=True)
class RepairCandidate:
    name: str
    stage: str
    mutations: dict[str, Any]
    intent: str
    source: str = ""


@dataclass
class FoldMetrics:
    name: str
    start: str
    end: str
    metrics: WindowMetrics


@dataclass
class StrategyRun:
    strategy: str
    mutations: dict[str, Any]
    is_metrics: WindowMetrics
    oos_metrics: WindowMetrics
    fold_metrics: list[FoldMetrics]
    assessment: str
    action: str
    error: str = ""


@dataclass
class CandidateEvaluation:
    candidate: RepairCandidate
    run: StrategyRun
    objective_delta: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyRepairResult:
    strategy: str
    family: str
    pre: StrategyRun
    post: StrategyRun
    accepted: list[CandidateEvaluation]
    stage_reports: list[dict[str, Any]]
    pre_config_path: str
    post_config_path: str


def run_repair(args: argparse.Namespace) -> dict[str, Any]:
    configure_config_overrides(getattr(args, "config", None))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = [item.strip() for item in args.strategies if item.strip()]
    max_workers = max(1, int(args.max_workers))
    run_spec = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategies": strategies,
        "data_end": args.data_end,
        "max_workers": max_workers,
        "stage_sequence": args.stage_sequence,
        "max_stage_rounds": args.max_stage_rounds,
        "max_candidates_per_stage": args.max_candidates_per_stage,
        "max_ablation_candidates": args.max_ablation_candidates,
        "max_perturbation_candidates": args.max_perturbation_candidates,
        "max_targeted_candidates": args.max_targeted_candidates,
        "resolved_stage_candidate_limits": {
            stage: stage_candidate_limit(args, stage)
            for stage in sorted(
                {"ablation", "perturbation", "targeted"}
                | {item.strip() for item in args.stage_sequence.split(",") if item.strip()}
            )
        },
        "selection_oos_note": (
            "The OOS window is used for tuning/selection in this repair run; "
            "it is no longer an untouched holdout."
        ),
        "config_overrides": {
            strategy: str(path)
            for strategy, path in sorted(CONFIG_OVERRIDES.items())
        },
    }
    write_json(output_dir / "run_spec.json", run_spec)

    results: list[StrategyRepairResult] = []
    for strategy in strategies:
        print(f"[{strategy}] starting repair", flush=True)
        result = repair_strategy(strategy=strategy, output_dir=output_dir, args=args)
        results.append(result)
        write_json(output_dir / f"{strategy}_summary.json", serialize(result))
        print(
            f"[{strategy}] post OOS trades={result.post.oos_metrics.total_trades}, "
            f"PF={result.post.oos_metrics.profit_factor:.2f}, "
            f"avgR={result.post.oos_metrics.avg_r:.3f}, accepted={len(result.accepted)}",
            flush=True,
        )

    summary = {
        "output_dir": str(output_dir.resolve()),
        "run_spec": run_spec,
        "results": [serialize(item) for item in results],
    }
    write_json(output_dir / "summary.json", summary)
    report = format_pre_post_report(results, args.data_end, run_spec["selection_oos_note"])
    (output_dir / "pre_post_oos_comparison.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


def repair_strategy(*, strategy: str, output_dir: Path, args: argparse.Namespace) -> StrategyRepairResult:
    if strategy not in STRATEGY_CONFIGS:
        raise ValueError(f"Unsupported strategy: {strategy}")

    config_path = config_path_for(strategy)
    current = read_json(config_path)
    pre = evaluate_strategy(strategy, current, args.data_end)
    accepted: list[CandidateEvaluation] = []
    stage_reports: list[dict[str, Any]] = []
    stage_sequence = [item.strip() for item in args.stage_sequence.split(",") if item.strip()]

    for stage in stage_sequence:
        for round_num in range(1, args.max_stage_rounds + 1):
            candidates = build_candidates(strategy, stage, current, args)
            if not candidates:
                stage_reports.append(
                    {
                        "stage": stage,
                        "round_num": round_num,
                        "stop_reason": "no_candidates",
                        "candidate_count": 0,
                    }
                )
                break

            evaluated = evaluate_candidates(
                strategy=strategy,
                baseline=pre if not accepted else evaluate_strategy(strategy, current, args.data_end),
                candidates=candidates,
                data_end=args.data_end,
                max_workers=args.max_workers,
            )
            evaluated_sorted = sorted(evaluated, key=lambda item: item.objective_delta, reverse=True)
            eligible = [
                item
                for item in evaluated_sorted
                if item.passed and item.objective_delta >= args.min_objective_delta
            ]
            stage_reports.append(
                {
                    "stage": stage,
                    "round_num": round_num,
                    "candidate_count": len(candidates),
                    "accepted": serialize(eligible[0]) if eligible else None,
                    "best": serialize(evaluated_sorted[0]) if evaluated_sorted else None,
                    "top_evaluations": [serialize(item) for item in evaluated_sorted[: args.report_top_n]],
                    "stop_reason": None if eligible else "no_candidate_met_repair_acceptance",
                }
            )
            if not eligible:
                break
            best = eligible[0]
            accepted.append(best)
            current = dict(best.candidate.mutations)
            print(
                f"  [{strategy} {stage} round {round_num}] accepted {best.candidate.name} "
                f"objective +{best.objective_delta * 100:.2f}%",
                flush=True,
            )

    post = evaluate_strategy(strategy, current, args.data_end)
    strategy_dir = output_dir / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    post_config_path = strategy_dir / "optimized_config.json"
    write_json(post_config_path, current)
    write_json(strategy_dir / "stage_reports.json", stage_reports)

    return StrategyRepairResult(
        strategy=strategy,
        family=STRATEGY_FAMILY[strategy],
        pre=pre,
        post=post,
        accepted=accepted,
        stage_reports=stage_reports,
        pre_config_path=str(config_path),
        post_config_path=str(post_config_path.resolve()),
    )


def configure_config_overrides(raw_items: list[str] | None) -> None:
    CONFIG_OVERRIDES.clear()
    for raw in raw_items or []:
        if "=" not in raw:
            raise ValueError(f"Config override must be strategy=path, got: {raw!r}")
        strategy, raw_path = raw.split("=", 1)
        strategy = strategy.strip()
        if strategy not in STRATEGY_CONFIGS:
            raise ValueError(f"Unsupported config override strategy: {strategy}")
        path = Path(raw_path.strip())
        CONFIG_OVERRIDES[strategy] = path if path.is_absolute() else PROJECT_ROOT / path


def config_path_for(strategy: str) -> Path:
    path = CONFIG_OVERRIDES.get(strategy, STRATEGY_CONFIGS[strategy])
    if not path.exists():
        raise FileNotFoundError(f"Missing optimized config for {strategy}: {path}")
    return path


def evaluate_candidates(
    *,
    strategy: str,
    baseline: StrategyRun,
    candidates: list[RepairCandidate],
    data_end: str,
    max_workers: int,
) -> list[CandidateEvaluation]:
    payloads = [(strategy, candidate, baseline, data_end) for candidate in candidates]
    if max_workers <= 1 or len(payloads) <= 1:
        return [_evaluate_candidate_payload(payload) for payload in payloads]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        return pool.map(_evaluate_candidate_payload, payloads)


def _evaluate_candidate_payload(payload: tuple[Any, ...]) -> CandidateEvaluation:
    strategy, candidate, baseline, data_end = payload
    try:
        run = evaluate_strategy(strategy, candidate.mutations, data_end)
        return score_candidate(candidate, baseline, run)
    except Exception as exc:
        run = StrategyRun(
            strategy=strategy,
            mutations=dict(candidate.mutations),
            is_metrics=WindowMetrics(),
            oos_metrics=WindowMetrics(),
            fold_metrics=[],
            assessment="ERROR",
            action="Error",
            error=str(exc),
        )
        return CandidateEvaluation(
            candidate=candidate,
            run=run,
            objective_delta=-999.0,
            passed=False,
            reasons=[f"error: {exc}"],
        )


def evaluate_strategy(strategy: str, mutations: dict[str, Any], data_end: str) -> StrategyRun:
    _quiet_noisy_loggers()
    if strategy == "helix_swing":
        trades = run_helix_swing_trades(mutations, data_end)
    elif strategy == "breakout":
        trades = run_breakout_trades(mutations, data_end)
    elif strategy == "brs":
        trades = run_brs_trades(mutations, data_end)
    elif strategy == "helix_momentum":
        trades = run_helix_momentum_trades(mutations, data_end)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = split_and_analyze(trades, BACKTEST_START, OOS_CUTOFF, data_end, is_months, oos_months)
    folds = build_fold_metrics(trades, data_end)
    assessment, action = _assess(strategy, is_m, oos_m)
    return StrategyRun(
        strategy=strategy,
        mutations=dict(mutations),
        is_metrics=is_m,
        oos_metrics=oos_m,
        fold_metrics=folds,
        assessment=assessment,
        action=action,
    )


def run_helix_swing_trades(mutations: dict[str, Any], data_end: str) -> list[Any]:
    from backtests.swing.auto.config_mutator import mutate_helix_config
    from backtests.swing.auto.helix.worker import load_helix_worker_data
    from backtests.swing.config_helix import HelixBacktestConfig
    from backtests.swing.engine.helix_portfolio_engine import run_helix_independent

    data_dir = PROJECT_ROOT / "backtests/swing/data/raw"
    base_config = HelixBacktestConfig(initial_equity=10_000, data_dir=data_dir)
    config = finalize_config(mutate_helix_config(base_config, mutations), data_end)
    data = load_helix_worker_data(config.symbols, data_dir)
    result = run_helix_independent(data, config)
    return _collect_symbol_trades(result)


def run_breakout_trades(mutations: dict[str, Any], data_end: str) -> list[Any]:
    from backtests.swing.auto.config_mutator import mutate_breakout_config
    from backtests.swing.config_breakout import BreakoutBacktestConfig
    from backtests.swing.data.replay_cache import load_breakout_replay_bundle
    from backtests.swing.engine.breakout_portfolio_engine import run_breakout_synchronized

    data_dir = PROJECT_ROOT / "backtests/swing/data/raw"
    base_config = BreakoutBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        track_signals=False,
        track_shadows=False,
    )
    config = finalize_config(mutate_breakout_config(base_config, mutations), data_end)
    data = load_breakout_replay_bundle(config.symbols, data_dir).data
    result = run_breakout_synchronized(data, config)
    return _collect_symbol_trades(result)


def run_brs_trades(mutations: dict[str, Any], data_end: str) -> list[Any]:
    from backtests.swing.auto.brs.config_mutator import mutate_brs_config
    from backtests.swing.config_brs import BRSConfig
    from backtests.swing.data.replay_cache import load_brs_replay_bundle
    from backtests.swing.engine.brs_portfolio_engine import run_brs_synchronized

    data_dir = PROJECT_ROOT / "backtests/swing/data/raw"
    base_config = BRSConfig(initial_equity=10_000, data_dir=data_dir)
    config = finalize_config(mutate_brs_config(base_config, mutations), data_end)
    data = load_brs_replay_bundle(config).data
    result = run_brs_synchronized(data, config)
    return _collect_symbol_trades(result)


def run_helix_momentum_trades(mutations: dict[str, Any], data_end: str) -> list[Any]:
    from backtests.momentum.auto.config_mutator import mutate_helix_config
    from backtests.momentum.cli import _load_helix_data_cached
    from backtests.momentum.config_helix import Helix4BacktestConfig
    from backtests.momentum.engine.helix_engine import Helix4Engine

    data_dir = PROJECT_ROOT / "backtests/momentum/data/raw"
    base_config = Helix4BacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        fixed_qty=10,
        track_signals=False,
        track_shadows=False,
    )
    config = finalize_config(mutate_helix_config(base_config, mutations), data_end)
    data = _load_helix_data_cached("NQ", data_dir)
    engine = Helix4Engine("NQ", config)
    result = engine.run(
        data["minute_bars"],
        data["hourly"],
        data["four_hour"],
        data["daily"],
        data["hourly_idx_map"],
        data["four_hour_idx_map"],
        data["daily_idx_map"],
    )
    return list(result.trades)


def finalize_config(config: Any, data_end: str) -> Any:
    return _with_backtest_window(_coerce_json_config_values(config), data_end)


def split_and_analyze(
    trades: list[Any],
    is_start: datetime,
    oos_start: datetime,
    data_end: str,
    is_months: float,
    oos_months: float,
) -> tuple[WindowMetrics, WindowMetrics]:
    is_rs: list[float] = []
    oos_rs: list[float] = []
    end = datetime.combine(date.fromisoformat(data_end) + timedelta(days=1), datetime.min.time())
    for trade in trades:
        entry = _get_entry_time(trade)
        if entry < is_start or entry >= end:
            continue
        r = float(_get_r_multiple(trade))
        if entry < oos_start:
            is_rs.append(r)
        else:
            oos_rs.append(r)
    return compute_window_metrics(is_rs, is_months), compute_window_metrics(oos_rs, oos_months)


def build_fold_metrics(trades: list[Any], data_end: str) -> list[FoldMetrics]:
    folds: list[FoldMetrics] = []
    start = date(2024, 4, 1)
    dev_end = LAST_SEEN_DATA_DATE
    while start <= dev_end:
        next_start = add_months(start, 3)
        end = min(next_start - timedelta(days=1), dev_end)
        rs: list[float] = []
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())
        for trade in trades:
            entry = _get_entry_time(trade)
            if start_dt <= entry < end_dt:
                rs.append(float(_get_r_multiple(trade)))
        months = _window_months(start, end + timedelta(days=1))
        folds.append(
            FoldMetrics(
                name=f"F{len(folds) + 1:02d}_{start:%Y%m%d}_{end:%Y%m%d}",
                start=start.isoformat(),
                end=end.isoformat(),
                metrics=compute_window_metrics(rs, months),
            )
        )
        start = next_start
    return folds


def add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    day = min(value.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


def score_candidate(
    candidate: RepairCandidate,
    baseline: StrategyRun,
    run: StrategyRun,
) -> CandidateEvaluation:
    deltas = build_deltas(baseline, run)
    objective = (
        0.22 * deltas["is_net_r_delta"]
        + 0.14 * deltas["is_winner_delta"]
        + 0.08 * deltas["is_win_rate_delta"]
        + 0.24 * deltas["oos_net_r_delta"]
        + 0.16 * deltas["oos_winner_delta"]
        + 0.08 * deltas["oos_win_rate_delta"]
        + 0.08 * deltas["oos_trade_delta"]
    )
    reasons = acceptance_reasons(baseline, run, deltas)
    return CandidateEvaluation(
        candidate=candidate,
        run=run,
        objective_delta=objective,
        passed=not reasons,
        reasons=reasons,
        deltas=deltas,
    )


def build_deltas(baseline: StrategyRun, run: StrategyRun) -> dict[str, float]:
    fold_deltas = []
    for base_fold, cand_fold in zip(baseline.fold_metrics, run.fold_metrics):
        fold_deltas.append(window_score_delta(base_fold.metrics, cand_fold.metrics))
    fold_mean = float(np.mean(fold_deltas)) if fold_deltas else 0.0
    fold_std = float(np.std(fold_deltas)) if fold_deltas else 0.0
    return {
        "is_net_r_delta": metric_delta(run.is_metrics.net_r, baseline.is_metrics.net_r, min_scale=5.0),
        "is_winner_delta": metric_delta(run.is_metrics.winning_trades, baseline.is_metrics.winning_trades, min_scale=5.0),
        "is_trade_delta": metric_delta(run.is_metrics.total_trades, baseline.is_metrics.total_trades, min_scale=10.0),
        "is_win_rate_delta": float(run.is_metrics.win_rate) - float(baseline.is_metrics.win_rate),
        "is_score_delta": window_score_delta(baseline.is_metrics, run.is_metrics),
        "oos_net_r_delta": metric_delta(run.oos_metrics.net_r, baseline.oos_metrics.net_r, min_scale=1.0),
        "oos_winner_delta": metric_delta(run.oos_metrics.winning_trades, baseline.oos_metrics.winning_trades, min_scale=2.0),
        "oos_trade_delta": metric_delta(run.oos_metrics.total_trades, baseline.oos_metrics.total_trades, min_scale=3.0),
        "oos_win_rate_delta": float(run.oos_metrics.win_rate) - float(baseline.oos_metrics.win_rate),
        "oos_score_delta": window_score_delta(baseline.oos_metrics, run.oos_metrics),
        "fold_mean_delta": fold_mean,
        "fold_std_delta": fold_std,
        "fold_robust_delta": fold_mean - 0.50 * fold_std,
    }


def acceptance_reasons(baseline: StrategyRun, run: StrategyRun, deltas: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if run.error:
        return [run.error]
    if run.is_metrics.total_trades < max(1, int(baseline.is_metrics.total_trades * 0.80)):
        reasons.append("development_trade_floor")
    if baseline.is_metrics.net_r > 0 and run.is_metrics.net_r < baseline.is_metrics.net_r * 0.90:
        reasons.append("development_net_r_floor")
    if run.is_metrics.avg_r < baseline.is_metrics.avg_r - 0.15:
        reasons.append("development_avg_r_floor")
    if run.is_metrics.win_rate < 0.50 and run.is_metrics.win_rate < baseline.is_metrics.win_rate - 0.01:
        reasons.append("development_win_rate_goal_floor")
    if (
        run.is_metrics.winning_trades < baseline.is_metrics.winning_trades
        and run.is_metrics.net_r <= baseline.is_metrics.net_r
    ):
        reasons.append("development_winning_trade_floor")
    if run.oos_metrics.total_trades < max(1, int(baseline.oos_metrics.total_trades * 0.75)):
        reasons.append("selection_oos_trade_floor")
    if run.oos_metrics.net_r < baseline.oos_metrics.net_r - max(1.0, abs(baseline.oos_metrics.net_r) * 0.25):
        reasons.append("selection_oos_net_r_floor")
    if run.oos_metrics.total_trades >= baseline.oos_metrics.total_trades and run.oos_metrics.win_rate < baseline.oos_metrics.win_rate - 0.10:
        reasons.append("selection_oos_win_rate_deterioration")
    if (
        run.oos_metrics.winning_trades < baseline.oos_metrics.winning_trades
        and run.oos_metrics.net_r <= baseline.oos_metrics.net_r
    ):
        reasons.append("selection_oos_winning_trade_floor")
    if deltas["fold_robust_delta"] < -0.10:
        reasons.append("fold_robust_deterioration")
    if run.is_metrics.max_drawdown_r > max(baseline.is_metrics.max_drawdown_r * 1.50, baseline.is_metrics.max_drawdown_r + 4.0):
        reasons.append("development_dd_expansion")
    return reasons


def window_score_delta(baseline: WindowMetrics, candidate: WindowMetrics) -> float:
    return metric_delta(window_score(candidate), window_score(baseline), min_scale=0.05)


def window_score(metrics: WindowMetrics) -> float:
    if metrics.total_trades <= 0:
        return 0.0
    pf = 8.0 if math.isinf(metrics.profit_factor) else max(metrics.profit_factor, 0.0)
    pf_component = clamp((pf - 1.0) / 5.0)
    avg_component = clamp((metrics.avg_r + 0.25) / 1.50)
    net_component = clamp((metrics.net_r + 5.0) / 80.0)
    freq_component = clamp(metrics.trades_per_month / 12.0)
    dd_component = 1.0 / (1.0 + max(metrics.max_drawdown_r, 0.0) / 6.0)
    return (
        0.25 * pf_component
        + 0.25 * avg_component
        + 0.25 * net_component
        + 0.15 * freq_component
        + 0.10 * dd_component
    )


def metric_delta(candidate: float, baseline: float, *, min_scale: float = 1.0) -> float:
    scale = max(abs(float(baseline)), min_scale)
    return (float(candidate) - float(baseline)) / scale


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def build_candidates(
    strategy: str,
    stage: str,
    current: dict[str, Any],
    args: argparse.Namespace,
) -> list[RepairCandidate]:
    if stage == "ablation":
        raw = build_ablation_candidates(strategy, current)
    elif stage == "perturbation":
        raw = build_perturbation_candidates(strategy, current)
    elif stage == "targeted":
        raw = build_targeted_candidates(strategy, current)
    elif stage == "helix_oos":
        raw = build_helix_oos_candidates(strategy, current)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return dedupe_and_rank(raw, current, max_candidates=stage_candidate_limit(args, stage))


def stage_candidate_limit(args: argparse.Namespace, stage: str) -> int:
    if stage == "ablation":
        if args.max_ablation_candidates is not None:
            return int(args.max_ablation_candidates)
        return 0
    if stage == "perturbation":
        if args.max_perturbation_candidates is not None:
            return int(args.max_perturbation_candidates)
        if args.max_candidates_per_stage is not None:
            return int(args.max_candidates_per_stage)
        return 48
    if stage == "targeted":
        if args.max_targeted_candidates is not None:
            return int(args.max_targeted_candidates)
        if args.max_candidates_per_stage is not None:
            return int(args.max_candidates_per_stage)
        return 48
    if stage == "helix_oos":
        if args.max_targeted_candidates is not None:
            return int(args.max_targeted_candidates)
        if args.max_candidates_per_stage is not None:
            return int(args.max_candidates_per_stage)
        return 0
    raise ValueError(f"Unknown stage: {stage}")


def build_ablation_candidates(strategy: str, current: dict[str, Any]) -> list[RepairCandidate]:
    features = build_historical_features(strategy)
    candidates: list[RepairCandidate] = []
    previous_by_key: dict[str, Any] = {}
    if features:
        for name, mutations, previous_values in features:
            active_keys = [key for key, value in mutations.items() if current.get(key, MISSING) == value]
            if not active_keys:
                continue
            for key in active_keys:
                previous_by_key[key] = previous_values.get(key, MISSING)
            reverted = dict(current)
            for key in active_keys:
                previous = previous_values.get(key, MISSING)
                if previous == MISSING:
                    reverted.pop(key, None)
                else:
                    reverted[key] = previous
            candidates.append(
                RepairCandidate(
                    name=f"ablate_cluster_{name}",
                    stage="ablation",
                    mutations=reverted,
                    intent="Remove an accepted historical mutation cluster.",
                    source=name,
                )
            )
            for key in active_keys:
                key_reverted = dict(current)
                previous = previous_values.get(key, MISSING)
                if previous == MISSING:
                    key_reverted.pop(key, None)
                else:
                    key_reverted[key] = previous
                candidates.append(
                    RepairCandidate(
                        name=f"ablate_key_{short_key(key)}",
                        stage="ablation",
                        mutations=key_reverted,
                        intent="Remove one accepted mutation key.",
                        source=name,
                    )
                )
    for key in sorted(current):
        dropped = dict(current)
        dropped.pop(key, None)
        candidates.append(
            RepairCandidate(
                name=f"ablate_key_drop_{short_key(key)}",
                stage="ablation",
                mutations=dropped,
                intent="Remove one incumbent mutation key.",
                source="all_round_active_key_inventory",
            )
        )
        previous = previous_by_key.get(key, MISSING)
        if previous != MISSING and previous != current.get(key, MISSING):
            reverted = dict(current)
            reverted[key] = previous
            candidates.append(
                RepairCandidate(
                    name=f"ablate_key_prior_{short_key(key)}",
                    stage="ablation",
                    mutations=reverted,
                    intent="Revert one incumbent mutation key to its last known prior value.",
                    source="all_round_active_key_inventory",
                )
            )
    return candidates


def build_historical_features(strategy: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    root = STRATEGY_ROUND_ROOTS.get(strategy)
    if root is None or not root.exists():
        return []
    features: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for round_dir in sorted(
        (item for item in root.glob("round_*") if item.is_dir()),
        key=lambda item: round_number(item),
    ):
        features.extend(build_phase_state_features(round_dir / "phase_state.json", round_dir.name))
    return features


def build_phase_state_features(path: Path, round_name: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if not path.exists():
        return []
    try:
        state = read_json(path)
    except Exception:
        return []
    phase_results = state.get("phase_results", {})
    features: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for phase_key in sorted(phase_results, key=phase_number):
        report = phase_results[phase_key]
        base = report.get("base_mutations") or {}
        final = report.get("final_mutations") or {}
        grouped: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for key, value in final.items():
            previous = base.get(key, MISSING)
            if previous == value:
                continue
            group = mutation_cluster(key)
            mutations, previous_values = grouped.setdefault(group, ({}, {}))
            mutations[key] = value
            previous_values[key] = previous
        for group, (mutations, previous_values) in grouped.items():
            features.append((f"{round_name}_phase_{phase_key}_{group}", mutations, previous_values))
    return features


def round_number(path: Path) -> tuple[int, str]:
    try:
        return (int(path.name.rsplit("_", 1)[1]), path.name)
    except (IndexError, ValueError):
        return (9999, path.name)


def phase_number(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return (int(text.rsplit("_", 1)[-1]), text)
    except ValueError:
        return (9999, text)


def build_perturbation_candidates(strategy: str, current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    for key in sorted(current, key=mutation_priority):
        value = current[key]
        if isinstance(value, bool):
            patch_candidate(candidates, current, "perturbation", f"flip_{short_key(key)}", {key: not value})
            continue
        if value is None or not isinstance(value, (int, float)):
            continue
        if isinstance(value, int):
            variants = sorted({max(0, int(round(value + delta))) for delta in (-2, -1, 1, 2)})
        else:
            variants = [round(float(value) * pct, 6) for pct in (0.80, 0.90, 1.10, 1.20)]
        for new_value in variants:
            if new_value == value:
                continue
            patch_candidate(
                candidates,
                current,
                "perturbation",
                f"perturb_{short_key(key)}_{format_value(new_value)}",
                {key: new_value},
                source=key,
            )
    return candidates


def build_targeted_candidates(strategy: str, current: dict[str, Any]) -> list[RepairCandidate]:
    raw: list[tuple[str, dict[str, Any]]] = []
    if strategy == "helix_swing":
        from backtests.swing.auto.helix.phase_candidates import get_phase_candidates
        for phase in range(1, 8):
            raw.extend(get_phase_candidates(phase, current))
    elif strategy == "breakout":
        from backtests.swing.auto.breakout.phase_candidates import get_phase_candidates
        for phase in range(1, 5):
            raw.extend(get_phase_candidates(phase, current))
    elif strategy == "brs":
        from backtests.swing.auto.brs.phase_candidates import get_phase_candidates
        for phase in range(1, 5):
            raw.extend(get_phase_candidates(phase, current))
    elif strategy == "helix_momentum":
        from backtests.momentum.auto.akc_helix.phase_candidates import get_phase_candidates
        for phase in range(1, 6):
            raw.extend(get_phase_candidates(phase))
    candidates: list[RepairCandidate] = []
    for name, mutations in raw:
        merged = dict(current)
        changed = False
        for key, value in mutations.items():
            if merged.get(key, MISSING) != value:
                merged[key] = value
                changed = True
        if changed:
            candidates.append(
                RepairCandidate(
                    name=f"target_{name}",
                    stage="targeted",
                    mutations=merged,
                    intent="Repo-native targeted candidate applied on top of the incumbent.",
                    source=name,
                )
            )
    return candidates


def build_helix_oos_candidates(strategy: str, current: dict[str, Any]) -> list[RepairCandidate]:
    if strategy != "helix_swing":
        return []
    from backtests.swing.auto.helix.phase_candidates import get_phase_candidates

    candidates: list[RepairCandidate] = []
    for name, mutations in get_phase_candidates(7, current):
        merged = dict(current)
        changed = False
        for key, value in mutations.items():
            if merged.get(key, MISSING) != value:
                merged[key] = value
                changed = True
        if changed:
            candidates.append(
                RepairCandidate(
                    name=f"helix_oos_{name}",
                    stage="helix_oos",
                    mutations=merged,
                    intent="Helix OOS repair candidate focused on Class D frequency and entry discrimination.",
                    source=name,
                )
            )
    return candidates


def patch_candidate(
    candidates: list[RepairCandidate],
    current: dict[str, Any],
    stage: str,
    name: str,
    patch: dict[str, Any],
    source: str = "",
) -> None:
    merged = dict(current)
    changed = False
    for key, value in patch.items():
        if merged.get(key, MISSING) != value:
            merged[key] = value
            changed = True
    if changed:
        candidates.append(
            RepairCandidate(
                name=name,
                stage=stage,
                mutations=merged,
                intent="Local incumbent perturbation.",
                source=source,
            )
        )


def dedupe_and_rank(
    candidates: list[RepairCandidate],
    current: dict[str, Any],
    *,
    max_candidates: int,
) -> list[RepairCandidate]:
    current_sig = signature(current)
    seen: set[str] = set()
    deduped: list[RepairCandidate] = []
    for candidate in candidates:
        sig = signature(candidate.mutations)
        if sig == current_sig or sig in seen:
            continue
        seen.add(sig)
        deduped.append(candidate)
    ranked = sorted(deduped, key=lambda item: candidate_priority(item), reverse=True)
    if max_candidates <= 0:
        return ranked
    return ranked[:max_candidates]


def candidate_priority(candidate: RepairCandidate) -> tuple[int, int, int]:
    text = f"{candidate.name} {' '.join(candidate.mutations)}".lower()
    score = 0
    for token in (
        "enable", "entry", "risk", "size", "qty", "trail", "stale", "exit",
        "partial", "add", "quality", "score", "ttl", "bars", "mfe", "vol",
        "short", "long",
    ):
        if token in text:
            score += 1
    return (score, -len(candidate.mutations), -len(candidate.name))


def mutation_priority(key: str) -> tuple[int, str]:
    text = key.lower()
    score = 0
    for token in (
        "risk", "qty", "size", "entry", "score", "quality", "trail", "partial",
        "stale", "exit", "bars", "ttl", "adx", "vol", "stop", "mfe",
    ):
        if token in text:
            score -= 1
    return (score, key)


def mutation_cluster(key: str) -> str:
    clean = key.replace("param_overrides.", "").replace("flags.", "")
    upper = clean.upper()
    for prefix in (
        "ENTRY_C_FRESH", "ENTRY_C_MOMENTUM", "ENTRY_OUTSIDE_WINDOW_CARRY",
        "FAST_BRANCH", "MICRO_TREND_RETEST", "M15_TREND_RETEST",
        "FAILED_RECLAIM_SHORT", "FOLLOWTHROUGH", "CLASS_T", "NORM_VOL",
        "HIGH_VOL", "CLASS_F",
    ):
        if upper.startswith(prefix):
            return prefix.lower()
    for token in ("risk", "size", "qty", "trail", "partial", "stale", "entry", "exit", "vol", "adx"):
        if token in clean.lower():
            return token
    return clean.split("_", 1)[0].lower()


def format_pre_post_report(
    results: list[StrategyRepairResult],
    data_end: str,
    note: str,
) -> str:
    lines: list[str] = []
    lines.append("=" * 116)
    lines.append("SWING INCUMBENT REPAIR PRE/POST OOS COMPARISON")
    lines.append(
        f"IS Period: {BACKTEST_START_DATE.isoformat()} to {LAST_SEEN_DATA_DATE.isoformat()} "
        f"(~{_window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE):.1f} months)"
    )
    lines.append(f"Selection OOS Period: {OOS_CUTOFF_DATE.isoformat()} to {data_end}")
    lines.append(note)
    lines.append("=" * 116)
    lines.append("")
    header = (
        f"{'Strategy':<16} {'Pre OOS#':>8} {'Pre PF':>8} {'Pre AvgR':>9} "
        f"{'Post OOS#':>9} {'Post PF':>8} {'Post AvgR':>10} "
        f"{'Pre IS PF':>9} {'Post IS PF':>10} {'Accepted':>8} {'Assessment':<17}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        pre_oos = result.pre.oos_metrics
        post_oos = result.post.oos_metrics
        assessment = f"{result.pre.assessment}->{result.post.assessment}"
        lines.append(
            f"{result.strategy:<16} {pre_oos.total_trades:>8} {pre_oos.profit_factor:>8.2f} "
            f"{pre_oos.avg_r:>9.3f} {post_oos.total_trades:>9} {post_oos.profit_factor:>8.2f} "
            f"{post_oos.avg_r:>10.3f} {result.pre.is_metrics.profit_factor:>9.2f} "
            f"{result.post.is_metrics.profit_factor:>10.2f} {len(result.accepted):>8} {assessment:<17}"
        )

    lines.append("")
    lines.append("=" * 116)
    lines.append("DETAIL")
    lines.append("=" * 116)
    for result in results:
        lines.append("")
        lines.append(f"Strategy: {result.strategy} ({result.family})")
        lines.append("-" * 72)
        lines.append(f"{'':20} {'Pre IS':>10} {'Post IS':>10} {'Pre OOS':>10} {'Post OOS':>10}")
        lines.append(
            f"{'Trades:':<20} {result.pre.is_metrics.total_trades:>10} "
            f"{result.post.is_metrics.total_trades:>10} "
            f"{result.pre.oos_metrics.total_trades:>10} {result.post.oos_metrics.total_trades:>10}"
        )
        lines.append(
            f"{'Win Rate:':<20} {result.pre.is_metrics.win_rate:>9.1%} "
            f"{result.post.is_metrics.win_rate:>9.1%} "
            f"{result.pre.oos_metrics.win_rate:>9.1%} {result.post.oos_metrics.win_rate:>9.1%}"
        )
        lines.append(
            f"{'Profit Factor:':<20} {result.pre.is_metrics.profit_factor:>10.2f} "
            f"{result.post.is_metrics.profit_factor:>10.2f} "
            f"{result.pre.oos_metrics.profit_factor:>10.2f} {result.post.oos_metrics.profit_factor:>10.2f}"
        )
        lines.append(
            f"{'Avg R/trade:':<20} {result.pre.is_metrics.avg_r:>10.3f} "
            f"{result.post.is_metrics.avg_r:>10.3f} "
            f"{result.pre.oos_metrics.avg_r:>10.3f} {result.post.oos_metrics.avg_r:>10.3f}"
        )
        lines.append(
            f"{'Net R:':<20} {result.pre.is_metrics.net_r:>10.1f} "
            f"{result.post.is_metrics.net_r:>10.1f} "
            f"{result.pre.oos_metrics.net_r:>10.1f} {result.post.oos_metrics.net_r:>10.1f}"
        )
        lines.append(
            f"{'Max DD (R):':<20} {result.pre.is_metrics.max_drawdown_r:>10.2f} "
            f"{result.post.is_metrics.max_drawdown_r:>10.2f} "
            f"{result.pre.oos_metrics.max_drawdown_r:>10.2f} {result.post.oos_metrics.max_drawdown_r:>10.2f}"
        )
        lines.append(
            f"{'Trades/month:':<20} {result.pre.is_metrics.trades_per_month:>10.1f} "
            f"{result.post.is_metrics.trades_per_month:>10.1f} "
            f"{result.pre.oos_metrics.trades_per_month:>10.1f} {result.post.oos_metrics.trades_per_month:>10.1f}"
        )
        if result.accepted:
            lines.append("Accepted repair candidates:")
            for item in result.accepted:
                lines.append(
                    f"  {item.candidate.stage}/{item.candidate.name}: "
                    f"objective {item.objective_delta * 100:+.2f}%"
                )
        else:
            lines.append("Accepted repair candidates: None")
        lines.append(f"Pre config:  {result.pre_config_path}")
        lines.append(f"Post config: {result.post_config_path}")
        lines.append(f"Assessment:  {result.pre.assessment} -> {result.post.assessment}")
    return "\n".join(lines) + "\n"


def serialize(value: Any) -> Any:
    if isinstance(value, (StrategyRepairResult, CandidateEvaluation, RepairCandidate, StrategyRun, FoldMetrics, WindowMetrics)):
        return {k: serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, (Path, date, datetime)):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize(value), indent=2), encoding="utf-8")


def signature(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def short_key(key: str) -> str:
    return (
        key.replace("param_overrides.", "")
        .replace("symbol_configs.", "sym_")
        .replace("flags.", "flag_")
        .replace(".", "_")
    )


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", "p")
    return str(value).replace(".", "p")


def _quiet_noisy_loggers() -> None:
    logging.getLogger("strategies.momentum.helix_v40.gates").setLevel(logging.WARNING)
    logging.getLogger("backtests.momentum.cli").setLevel(logging.WARNING)
    logging.getLogger("backtests.swing.engine.breakout_portfolio_engine").setLevel(logging.WARNING)
    logging.getLogger("backtests.swing.engine.brs_portfolio_engine").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair non-ATRSS swing incumbents and report pre/post OOS state")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Override a starting config as strategy=path. Can be repeated.",
    )
    parser.add_argument("--data-end", default="2026-05-01")
    parser.add_argument("--output-dir", default="backtests/output/swing/incumbent_repair_20260504_other_swing")
    parser.add_argument("--max-workers", type=int, default=max(1, min(2, os.cpu_count() or 1)))
    parser.add_argument("--stage-sequence", default="ablation,perturbation,targeted")
    parser.add_argument("--max-stage-rounds", type=int, default=1)
    parser.add_argument(
        "--max-candidates-per-stage",
        type=int,
        default=None,
        help="Legacy cap used for perturbation/targeted when stage-specific caps are omitted.",
    )
    parser.add_argument(
        "--max-ablation-candidates",
        type=int,
        default=0,
        help="Ablation cap; 0 means evaluate all cluster and individual active-key ablations.",
    )
    parser.add_argument("--max-perturbation-candidates", type=int, default=48)
    parser.add_argument("--max-targeted-candidates", type=int, default=48)
    parser.add_argument("--min-objective-delta", type=float, default=0.005)
    parser.add_argument("--report-top-n", type=int, default=8)
    return parser


def main() -> None:
    start = time.time()
    args = build_parser().parse_args()
    summary = run_repair(args)
    elapsed = (time.time() - start) / 60.0
    print(f"Complete in {elapsed:.1f} min")
    print(f"Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()
