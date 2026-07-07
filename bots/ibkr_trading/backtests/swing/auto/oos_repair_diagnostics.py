"""Focused OOS repair diagnostics for weak swing-subset incumbents.

This runner reuses the incumbent repair candidate builders, but evaluates
candidate changes independently against the frozen incumbent instead of doing a
slow sequential greedy run. The OOS window is used for diagnosis/selection, so
outputs should not be treated as a fresh holdout validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.swing.auto.incumbent_repair import (
    BACKTEST_START,
    BACKTEST_START_DATE,
    DEFAULT_STRATEGIES,
    MISSING,
    OOS_CUTOFF,
    STRATEGY_CONFIGS,
    CandidateEvaluation,
    RepairCandidate,
    StrategyRun,
    WindowMetrics,
    _assess,
    _compute_oos_months,
    _window_months,
    build_ablation_candidates,
    build_fold_metrics,
    build_helix_oos_candidates,
    build_phase_state_features,
    build_perturbation_candidates,
    build_targeted_candidates,
    evaluate_strategy as evaluate_existing_strategy,
    finalize_config,
    round_number,
    score_candidate,
    serialize,
    short_key,
    split_and_analyze,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTRA_STRATEGY_CONFIGS = {
    "nqdtc": PROJECT_ROOT / "backtests/output/momentum/nqdtc/round_3/optimized_config.json",
}
EXTRA_STRATEGY_ROUND_ROOTS = {
    "nqdtc": PROJECT_ROOT / "backtests/output/momentum/nqdtc",
}
CONFIG_OVERRIDES: dict[str, Path] = {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_config_overrides(getattr(args, "config", None))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "candidate_progress.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "summary.txt"

    strategies = [item.strip() for item in args.strategies if item.strip()]
    spec = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategies": strategies,
        "data_end": args.data_end,
        "max_workers": args.max_workers,
        "max_perturbation_candidates": args.max_perturbation_candidates,
        "max_targeted_candidates": args.max_targeted_candidates,
        "top_n": args.top_n,
        "selection_oos_note": (
            "The OOS window is used for diagnosis/selection in this run; "
            "it is no longer an untouched holdout."
        ),
        "config_overrides": {
            strategy: str(path)
            for strategy, path in sorted(CONFIG_OVERRIDES.items())
        },
    }
    write_json(output_dir / "run_spec.json", spec)
    progress_path.write_text("", encoding="utf-8")

    strategy_results: list[dict[str, Any]] = []
    for strategy in strategies:
        started = time.time()
        print(f"[{strategy}] loading incumbent", flush=True)
        current = read_json(config_path_for(strategy))
        baseline = evaluate_strategy(strategy, current, args.data_end)
        candidates = build_candidate_suite(strategy, current, args)
        print(
            f"[{strategy}] baseline OOS trades={baseline.oos_metrics.total_trades} "
            f"netR={baseline.oos_metrics.net_r:.3f} avgR={baseline.oos_metrics.avg_r:.3f}; "
            f"evaluating {len(candidates)} candidates",
            flush=True,
        )
        evaluations = evaluate_candidate_suite(
            strategy=strategy,
            baseline=baseline,
            candidates=candidates,
            data_end=args.data_end,
            max_workers=args.max_workers,
            progress_path=progress_path,
        )
        top = sorted(evaluations, key=lambda item: item.objective_delta, reverse=True)
        passed = [item for item in top if item.passed]
        result = {
            "strategy": strategy,
            "baseline": serialize(baseline),
            "candidate_count": len(candidates),
            "elapsed_seconds": round(time.time() - started, 2),
            "top": [serialize(item) for item in top[: args.top_n]],
            "passed": [serialize(item) for item in passed[: args.top_n]],
            "stage_leaders": stage_leaders(top, args.top_n),
            "oos_net_leaders": [
                serialize(item)
                for item in sorted(
                    evaluations,
                    key=lambda item: (
                        item.run.oos_metrics.net_r,
                        item.run.oos_metrics.total_trades,
                        item.run.is_metrics.net_r,
                    ),
                    reverse=True,
                )[: args.top_n]
            ],
        }
        strategy_results.append(result)
        write_json(summary_path, {"run_spec": spec, "results": strategy_results})
        report_path.write_text(format_report(spec, strategy_results), encoding="utf-8")
        print(
            f"[{strategy}] complete in {(time.time() - started) / 60.0:.1f} min; "
            f"best={top[0].candidate.name if top else 'none'}",
            flush=True,
        )

    summary = {"run_spec": spec, "results": strategy_results}
    write_json(summary_path, summary)
    report_path.write_text(format_report(spec, strategy_results), encoding="utf-8")
    print(f"Output: {output_dir.resolve()}", flush=True)
    return summary


def build_candidate_suite(
    strategy: str,
    current: dict[str, Any],
    args: argparse.Namespace,
) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    stages = {item.strip() for item in args.stages.split(",") if item.strip()}
    if "ablation" in stages:
        if strategy in EXTRA_STRATEGY_CONFIGS:
            candidates.extend(build_extra_ablation_candidates(strategy, current))
        else:
            candidates.extend(build_ablation_candidates(strategy, current))
    if "perturbation" in stages:
        candidates.extend(
            limit_candidates(build_perturbation_candidates(strategy, current), args.max_perturbation_candidates)
        )
    if "targeted" in stages:
        if strategy in EXTRA_STRATEGY_CONFIGS:
            candidates.extend(limit_candidates(build_extra_targeted_candidates(strategy, current), args.max_targeted_candidates))
        else:
            candidates.extend(limit_candidates(build_targeted_candidates(strategy, current), args.max_targeted_candidates))
    if "helix_oos" in stages:
        candidates.extend(limit_candidates(build_helix_oos_candidates(strategy, current), args.max_targeted_candidates))
    return dedupe(candidates, current)


def limit_candidates(candidates: list[RepairCandidate], limit: int | None) -> list[RepairCandidate]:
    if limit is None or int(limit) <= 0:
        return candidates
    return candidates[: int(limit)]


def config_path_for(strategy: str) -> Path:
    if strategy in CONFIG_OVERRIDES:
        path = CONFIG_OVERRIDES[strategy]
        if not path.exists():
            raise FileNotFoundError(f"Missing override optimized config for {strategy}: {path}")
        return path
    if strategy in EXTRA_STRATEGY_CONFIGS:
        return EXTRA_STRATEGY_CONFIGS[strategy]
    return STRATEGY_CONFIGS[strategy]


def configure_config_overrides(raw_items: list[str] | None) -> None:
    CONFIG_OVERRIDES.clear()
    supported = set(STRATEGY_CONFIGS) | set(EXTRA_STRATEGY_CONFIGS)
    for raw in raw_items or []:
        if "=" not in raw:
            raise ValueError(f"Config override must be strategy=path, got: {raw!r}")
        strategy, raw_path = raw.split("=", 1)
        strategy = strategy.strip()
        if strategy not in supported:
            raise ValueError(f"Unsupported config override strategy: {strategy}")
        path = Path(raw_path.strip())
        CONFIG_OVERRIDES[strategy] = path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_strategy(strategy: str, mutations: dict[str, Any], data_end: str) -> StrategyRun:
    if strategy == "nqdtc":
        return evaluate_nqdtc_strategy(mutations, data_end)
    return evaluate_existing_strategy(strategy, mutations, data_end)


def evaluate_nqdtc_strategy(mutations: dict[str, Any], data_end: str) -> StrategyRun:
    trades = run_nqdtc_trades(mutations, data_end)
    is_months = _window_months(BACKTEST_START_DATE, OOS_CUTOFF.date())
    oos_months = _compute_oos_months(data_end)
    is_m, oos_m = split_and_analyze(trades, BACKTEST_START, OOS_CUTOFF, data_end, is_months, oos_months)
    folds = build_fold_metrics(trades, data_end)
    assessment, action = _assess("nqdtc", is_m, oos_m)
    return StrategyRun(
        strategy="nqdtc",
        mutations=dict(mutations),
        is_metrics=is_m,
        oos_metrics=oos_m,
        fold_metrics=folds,
        assessment=assessment,
        action=action,
    )


def run_nqdtc_trades(mutations: dict[str, Any], data_end: str) -> list[Any]:
    from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
    from backtests.momentum.auto.nqdtc.worker import load_worker_data
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine

    data_dir = PROJECT_ROOT / "backtests/momentum/data/raw"
    base_config = NQDTCBacktestConfig(
        initial_equity=10_000,
        data_dir=data_dir,
        fixed_qty=10,
        track_signals=False,
        track_shadows=False,
        scoring_mode=False,
        max_dd_abort=0.0,
    )
    config = finalize_config(mutate_nqdtc_config(base_config, mutations), data_end)
    bundle = load_worker_data("NQ", data_dir)
    kwargs = replay_engine_kwargs(bundle)
    engine = NQDTCEngine("MNQ", config)
    result = engine.run(**kwargs)
    return list(result.trades)


def build_extra_ablation_candidates(strategy: str, current: dict[str, Any]) -> list[RepairCandidate]:
    features = build_extra_historical_features(strategy)
    candidates: list[RepairCandidate] = []
    previous_by_key: dict[str, Any] = {}
    missing = MISSING
    for name, mutations, previous_values in features:
        active_keys = [key for key, value in mutations.items() if current.get(key, missing) == value]
        if not active_keys:
            continue
        for key in active_keys:
            previous_by_key[key] = previous_values.get(key, missing)
        reverted = dict(current)
        for key in active_keys:
            previous = previous_values.get(key, missing)
            if previous == missing:
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
            previous = previous_values.get(key, missing)
            if previous == missing:
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
                source="active_key_inventory",
            )
        )
        previous = previous_by_key.get(key, missing)
        if previous != missing and previous != current.get(key, missing):
            reverted = dict(current)
            reverted[key] = previous
            candidates.append(
                RepairCandidate(
                    name=f"ablate_key_prior_{short_key(key)}",
                    stage="ablation",
                    mutations=reverted,
                    intent="Revert one incumbent mutation key to its last known prior value.",
                    source="active_key_inventory",
                )
            )
    return candidates


def build_extra_historical_features(strategy: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    root = EXTRA_STRATEGY_ROUND_ROOTS[strategy]
    features: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for round_dir in sorted(
        (item for item in root.glob("round_*") if item.is_dir()),
        key=round_number,
    ):
        features.extend(build_phase_state_features(round_dir / "phase_state.json", round_dir.name))
    return features


def build_extra_targeted_candidates(strategy: str, current: dict[str, Any]) -> list[RepairCandidate]:
    raw: list[tuple[str, dict[str, Any]]] = []
    if strategy == "nqdtc":
        from backtests.momentum.auto.nqdtc.phase_candidates import get_phase_candidates

        for phase in range(1, 5):
            raw.extend(get_phase_candidates(phase, current))
    candidates: list[RepairCandidate] = []
    missing = MISSING
    for name, mutations in raw:
        merged = dict(current)
        changed = False
        for key, value in mutations.items():
            if merged.get(key, missing) != value:
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


def evaluate_candidate_suite(
    *,
    strategy: str,
    baseline: StrategyRun,
    candidates: list[RepairCandidate],
    data_end: str,
    max_workers: int,
    progress_path: Path,
) -> list[CandidateEvaluation]:
    if max_workers <= 1:
        evaluations = []
        for index, candidate in enumerate(candidates, start=1):
            evaluation = evaluate_one(strategy, baseline, candidate, data_end)
            evaluations.append(evaluation)
            append_progress(progress_path, strategy, index, len(candidates), evaluation)
            print_progress(strategy, index, len(candidates), evaluation)
        return evaluations

    evaluations: list[CandidateEvaluation] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(evaluate_one, strategy, baseline, candidate, data_end): idx
            for idx, candidate in enumerate(candidates, start=1)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            evaluation = future.result()
            evaluations.append(evaluation)
            append_progress(progress_path, strategy, completed, len(candidates), evaluation)
            print_progress(strategy, completed, len(candidates), evaluation)
    return evaluations


def evaluate_one(
    strategy: str,
    baseline: StrategyRun,
    candidate: RepairCandidate,
    data_end: str,
) -> CandidateEvaluation:
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


def append_progress(
    path: Path,
    strategy: str,
    completed: int,
    total: int,
    evaluation: CandidateEvaluation,
) -> None:
    payload = {
        "strategy": strategy,
        "completed": completed,
        "total": total,
        "candidate": evaluation.candidate.name,
        "stage": evaluation.candidate.stage,
        "objective_delta": evaluation.objective_delta,
        "passed": evaluation.passed,
        "reasons": evaluation.reasons,
        "is_metrics": serialize(evaluation.run.is_metrics),
        "oos_metrics": serialize(evaluation.run.oos_metrics),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(serialize(payload), default=str) + "\n")


def print_progress(
    strategy: str,
    completed: int,
    total: int,
    evaluation: CandidateEvaluation,
) -> None:
    oos = evaluation.run.oos_metrics
    print(
        f"[{strategy}] {completed}/{total} {evaluation.candidate.stage}/"
        f"{evaluation.candidate.name} obj={evaluation.objective_delta:+.3%} "
        f"oos={oos.total_trades} {oos.net_r:+.2f}R passed={evaluation.passed}",
        flush=True,
    )


def stage_leaders(evaluations: list[CandidateEvaluation], top_n: int) -> dict[str, list[Any]]:
    by_stage: dict[str, list[CandidateEvaluation]] = {}
    for item in evaluations:
        by_stage.setdefault(item.candidate.stage, []).append(item)
    return {
        stage: [serialize(item) for item in items[:top_n]]
        for stage, items in by_stage.items()
    }


def format_report(spec: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "OOS Repair Diagnostic Summary",
        "=" * 92,
        spec["selection_oos_note"],
        f"Data end: {spec['data_end']}",
        "",
    ]
    for result in results:
        baseline = result["baseline"]
        base_is = baseline["is_metrics"]
        base_oos = baseline["oos_metrics"]
        lines.extend(
            [
                f"Strategy: {result['strategy']}",
                "-" * 92,
                (
                    f"Baseline IS: trades={base_is['total_trades']} "
                    f"PF={fmt(base_is['profit_factor'])} avgR={base_is['avg_r']:.3f} "
                    f"netR={base_is['net_r']:.1f}"
                ),
                (
                    f"Baseline OOS: trades={base_oos['total_trades']} "
                    f"PF={fmt(base_oos['profit_factor'])} avgR={base_oos['avg_r']:.3f} "
                    f"netR={base_oos['net_r']:.1f}"
                ),
                f"Candidates evaluated: {result['candidate_count']}",
                "Top objective candidates:",
            ]
        )
        for item in result["top"][: min(8, len(result["top"]))]:
            candidate = item["candidate"]
            run = item["run"]
            is_m = run["is_metrics"]
            oos_m = run["oos_metrics"]
            lines.append(
                f"  {candidate['stage']}/{candidate['name']}: "
                f"obj={item['objective_delta']:+.2%}, passed={item['passed']}, "
                f"OOS trades={oos_m['total_trades']} netR={oos_m['net_r']:.1f} "
                f"avgR={oos_m['avg_r']:.3f}, IS trades={is_m['total_trades']} "
                f"netR={is_m['net_r']:.1f}"
            )
        lines.append("Top OOS-net candidates:")
        for item in result["oos_net_leaders"][: min(8, len(result["oos_net_leaders"]))]:
            candidate = item["candidate"]
            run = item["run"]
            is_m = run["is_metrics"]
            oos_m = run["oos_metrics"]
            lines.append(
                f"  {candidate['stage']}/{candidate['name']}: "
                f"OOS trades={oos_m['total_trades']} netR={oos_m['net_r']:.1f} "
                f"avgR={oos_m['avg_r']:.3f}, IS trades={is_m['total_trades']} "
                f"netR={is_m['net_r']:.1f}, obj={item['objective_delta']:+.2%}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def dedupe(candidates: list[RepairCandidate], current: dict[str, Any]) -> list[RepairCandidate]:
    current_sig = signature(current)
    seen: set[str] = set()
    deduped: list[RepairCandidate] = []
    for candidate in candidates:
        sig = signature(candidate.mutations)
        if sig == current_sig or sig in seen:
            continue
        seen.add(sig)
        deduped.append(candidate)
    return deduped


def signature(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(serialize(value), indent=2, default=str), encoding="utf-8")


def fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Focused OOS repair diagnostics")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Override a starting config as strategy=path. Can be repeated.",
    )
    parser.add_argument("--data-end", default="2026-05-01")
    parser.add_argument("--output-dir", default="backtests/output/swing/oos_repair_diagnostics")
    parser.add_argument("--stages", default="ablation,perturbation,targeted")
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--max-perturbation-candidates", type=int, default=48)
    parser.add_argument("--max-targeted-candidates", type=int, default=48)
    parser.add_argument("--top-n", type=int, default=16)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
