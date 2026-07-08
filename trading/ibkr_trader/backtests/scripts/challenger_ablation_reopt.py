"""Controlled challenger ablation and narrow reoptimization pass.

This script is intentionally conservative:

* Calibration data ends at 2026-03-20, so the 2026-03-21+ OOS window stays
  available for a single promotion check.
* Each strategy is first scanned with one-at-a-time diagnostic ablations.
* Only the best diagnostic phase is used for one narrow greedy pass.
* Results are written under backtests/output/challenger_reopt_* and never into
  the production round directories.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backtests.shared.auto.greedy_optimizer import run_greedy
from backtests.shared.auto.phase_state import PhaseState, _atomic_write_json
from backtests.shared.auto.round_manager import RoundManager
from backtests.shared.auto.types import Experiment, ScoredCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_START = "2024-01-01"
CALIBRATION_END = "2026-03-20"
OOS_END = "2026-05-01"


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
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_latest_mutations(family: str, strategy: str) -> tuple[dict[str, Any], Path, int]:
    manager = RoundManager(family, strategy)
    latest = manager.get_latest_round()
    if latest < 1:
        raise FileNotFoundError(f"No optimized round found for {family}/{strategy}")
    path = manager.optimized_config_path(manager.round_path(latest))
    data = _read_json(path)
    if "mutations" in data and isinstance(data["mutations"], dict):
        data = data["mutations"]
    elif "cumulative_mutations" in data and isinstance(data["cumulative_mutations"], dict):
        data = data["cumulative_mutations"]
    return dict(data), path, latest


def _slice_parquet_dir(source_dir: Path, target_dir: Path, *, start: str, end: str) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    summary: dict[str, Any] = {"files": 0, "rows": 0, "empty": [], "errors": []}

    for source in sorted(source_dir.glob("*.parquet")):
        target = target_dir / source.name
        try:
            df = pd.read_parquet(source)
            if not df.empty:
                idx = pd.DatetimeIndex(df.index)
                if idx.tz is not None:
                    start_bound = start_ts.tz_localize("UTC")
                    end_bound = end_ts.tz_localize("UTC")
                else:
                    start_bound = start_ts
                    end_bound = end_ts
                df = df.loc[(idx >= start_bound) & (idx <= end_bound)]
            if df.empty:
                summary["empty"].append(source.name)
            df.to_parquet(target)
            summary["files"] += 1
            summary["rows"] += int(len(df))
        except Exception as exc:
            summary["errors"].append(f"{source.name}: {exc}")
    return summary


def prepare_calibration_data(root: Path) -> dict[str, Any]:
    data_root = root / "calibration_data"
    return {
        "swing": _slice_parquet_dir(
            PROJECT_ROOT / "backtests" / "swing" / "data" / "raw",
            data_root / "swing" / "raw",
            start=CALIBRATION_START,
            end=CALIBRATION_END,
        ),
        "momentum": _slice_parquet_dir(
            PROJECT_ROOT / "backtests" / "momentum" / "data" / "raw",
            data_root / "momentum" / "raw",
            start=CALIBRATION_START,
            end=CALIBRATION_END,
        ),
    }


def build_plugin(strategy: str, data_root: Path, max_workers: int):
    if strategy == "iaric":
        from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin

        plugin = IARICPullbackPlugin(
            data_dir=PROJECT_ROOT / "backtests" / "stock" / "data" / "raw",
            start_date=CALIBRATION_START,
            end_date=CALIBRATION_END,
            initial_equity=10_000.0,
            max_workers=max_workers,
            num_phases=4,
            profile="mainline",
            round_name="v5r2",
        )
        return plugin

    if strategy == "helix_swing":
        from backtests.swing.auto.helix.plugin import HelixPlugin

        return HelixPlugin(data_root / "swing" / "raw", initial_equity=10_000.0, max_workers=max_workers)

    if strategy == "breakout":
        from backtests.swing.auto.breakout.plugin import BreakoutPlugin

        return BreakoutPlugin(data_root / "swing" / "raw", initial_equity=10_000.0, max_workers=max_workers)

    if strategy == "atrss":
        from backtests.swing.auto.atrss.plugin import ATRSSPlugin

        return ATRSSPlugin(
            data_root / "swing" / "raw",
            initial_equity=10_000.0,
            max_workers=max_workers,
            mode="synchronized",
            candidate_profile="alpha",
        )

    if strategy == "brs":
        from backtests.swing.auto.brs.plugin import BRSPlugin

        return BRSPlugin(data_root / "swing" / "raw", initial_equity=10_000.0, max_workers=max_workers)

    if strategy == "nqdtc":
        from backtests.momentum.auto.nqdtc.plugin import NQDTCPlugin

        return NQDTCPlugin(data_root / "momentum" / "raw", initial_equity=10_000.0, max_workers=max_workers)

    if strategy == "helix_momentum":
        from backtests.momentum.auto.akc_helix.plugin import AKCHelixPlugin

        return AKCHelixPlugin(data_root / "momentum" / "raw", initial_equity=10_000.0, max_workers=max_workers)

    raise KeyError(strategy)


def strategy_meta(strategy: str) -> tuple[str, str, list[int], str]:
    if strategy == "iaric":
        return "stock", "iaric", [1, 2, 3, 4], "V5R2 maintenance set: carry drag, route quality, capacity, exits"
    if strategy == "helix_swing":
        return "swing", "helix", [1, 2, 3], "Signal pruning, exit management, volatility/add-on only"
    if strategy == "breakout":
        return "swing", "breakout", [1, 2, 3, 4], "Existing compact Breakout maintenance phases"
    if strategy == "atrss":
        return "swing", "atrss", [1, 2, 3], "Alpha maintenance phases; no risk-only sizing pass"
    if strategy == "brs":
        return "swing", "brs", [2, 3], "BRS kept narrow: signal selection and exit/volatility only"
    if strategy == "nqdtc":
        return "momentum", "nqdtc", [1, 2, 3], "Session harvest, robust protection, selective recovery only"
    if strategy == "helix_momentum":
        return "momentum", "helix", [1, 2, 3, 4, 5], "Momentum Helix compact maintenance phases"
    raise KeyError(strategy)


def _close_plugin(plugin: Any) -> None:
    close_pool = getattr(plugin, "close_pool", None)
    if callable(close_pool):
        close_pool()
    destroy_pool = getattr(plugin, "_destroy_pool", None)
    if callable(destroy_pool):
        destroy_pool()


def _score_sort_key(item: ScoredCandidate) -> float:
    return float(item.score if not item.rejected else -1e100)


def _delta_pct(score: float, baseline: float) -> float:
    if baseline > 0:
        return (score - baseline) / baseline * 100.0
    return (score - baseline) * 100.0


def evaluate_phase(
    plugin: Any,
    phase: int,
    base_mutations: dict[str, Any],
) -> dict[str, Any]:
    state = PhaseState(cumulative_mutations=dict(base_mutations))
    spec = plugin.get_phase_spec(phase, state)
    evaluator = plugin.create_evaluate_batch(
        phase,
        base_mutations,
        scoring_weights=spec.scoring_weights,
        hard_rejects=spec.hard_rejects,
    )
    try:
        baseline = evaluator([Experiment("__baseline__", {})], base_mutations)[0]
        scored = evaluator(spec.candidates, base_mutations)
    finally:
        close = getattr(evaluator, "close", None)
        if callable(close):
            close()

    valid = [item for item in scored if not item.rejected]
    rejected = [item for item in scored if item.rejected]
    top = sorted(valid, key=_score_sort_key, reverse=True)[:8]
    baseline_score = float(baseline.score)
    best = top[0] if top else None
    return {
        "phase": phase,
        "focus": spec.focus,
        "candidate_count": len(spec.candidates),
        "valid_count": len(valid),
        "rejected_count": len(rejected),
        "baseline": _jsonable(baseline),
        "baseline_score": baseline_score,
        "baseline_rejected": bool(baseline.rejected),
        "baseline_reject_reason": baseline.reject_reason,
        "best_delta_pct": _delta_pct(float(best.score), baseline_score) if best else None,
        "top_candidates": [
            {
                "name": item.name,
                "score": float(item.score),
                "delta_pct": _delta_pct(float(item.score), baseline_score),
                "metrics": _jsonable(item.metrics),
                "mutations": next((c.mutations for c in spec.candidates if c.name == item.name), {}),
            }
            for item in top
        ],
        "sample_rejections": [
            {"name": item.name, "reason": item.reject_reason[:500]}
            for item in rejected[:10]
        ],
    }


def choose_phase(phase_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [
        result for result in phase_results
        if result.get("top_candidates") and result.get("best_delta_pct") is not None
    ]
    if not viable:
        return None
    viable.sort(key=lambda r: float(r["best_delta_pct"]), reverse=True)
    return viable[0]


def run_narrow_greedy(
    plugin: Any,
    phase: int,
    base_mutations: dict[str, Any],
    output_dir: Path,
    *,
    max_rounds: int,
) -> dict[str, Any]:
    state = PhaseState(cumulative_mutations=dict(base_mutations))
    spec = plugin.get_phase_spec(phase, state)
    evaluator = plugin.create_evaluate_batch(
        phase,
        base_mutations,
        scoring_weights=spec.scoring_weights,
        hard_rejects=spec.hard_rejects,
    )
    try:
        result = run_greedy(
            spec.candidates,
            base_mutations,
            evaluator,
            max_rounds=max_rounds,
            min_delta=0.005,
            prune_threshold=spec.prune_threshold if spec.prune_threshold is not None else 0.05,
            reject_streak_limit=spec.reject_streak_limit if spec.reject_streak_limit is not None else 1,
            checkpoint_path=output_dir / f"phase_{phase}_narrow_greedy_checkpoint.json",
            checkpoint_context={
                "phase": phase,
                "focus": spec.focus,
                "calibration_start": CALIBRATION_START,
                "calibration_end": CALIBRATION_END,
            },
        )
    finally:
        close = getattr(evaluator, "close", None)
        if callable(close):
            close()

    try:
        final_metrics = plugin.compute_final_metrics(result.final_mutations)
    except Exception as exc:
        final_metrics = {"error": str(exc)}
    result.final_metrics = dict(final_metrics or {})
    return _jsonable(result)


def run_challenger_oos(strategy: str, config_path: Path) -> dict[str, Any]:
    import backtests.shared.validation.oos_validation as oos

    key = strategy
    relative = config_path.relative_to(PROJECT_ROOT).as_posix()
    old_path = oos.OPTIMIZED_CONFIG_PATHS.get(key)
    oos.OPTIMIZED_CONFIG_PATHS[key] = relative
    try:
        result = oos.RUNNERS[key](OOS_END)
        return _jsonable(result)
    finally:
        if old_path is None:
            oos.OPTIMIZED_CONFIG_PATHS.pop(key, None)
        else:
            oos.OPTIMIZED_CONFIG_PATHS[key] = old_path


def run_strategy(strategy: str, root: Path, max_workers: int, max_rounds: int, *, validate_oos: bool) -> dict[str, Any]:
    family, round_strategy, phases, scope_note = strategy_meta(strategy)
    output_dir = root / "strategies" / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "calibration_data"
    base_mutations, champion_path, champion_round = _load_latest_mutations(family, round_strategy)

    plugin = build_plugin(strategy, data_root, max_workers)
    plugin.initial_mutations = dict(base_mutations)

    started = time.time()
    payload: dict[str, Any] = {
        "strategy": strategy,
        "family": family,
        "round_strategy": round_strategy,
        "champion_round": champion_round,
        "champion_config_path": str(champion_path),
        "scope_note": scope_note,
        "calibration_start": CALIBRATION_START,
        "calibration_end": CALIBRATION_END,
        "diagnostic_phases": phases,
    }
    try:
        champion_metrics = plugin.compute_final_metrics(base_mutations)
        payload["champion_calibration_metrics"] = _jsonable(champion_metrics)
    except Exception as exc:
        payload["champion_calibration_metrics_error"] = traceback.format_exc()
        champion_metrics = {}

    phase_results: list[dict[str, Any]] = []
    try:
        for phase in phases:
            phase_result = evaluate_phase(plugin, phase, base_mutations)
            phase_results.append(phase_result)
            _atomic_write_json(_jsonable(phase_result), output_dir / f"diagnostic_phase_{phase}.json")
        payload["diagnostic_ablation"] = phase_results

        selected = choose_phase(phase_results)
        payload["selected_phase"] = selected
        if selected is None or float(selected.get("best_delta_pct") or 0.0) <= 0.0:
            payload["decision"] = "no_challenger"
            payload["decision_reason"] = "No diagnostic phase produced a positive valid score lift."
            final_mutations = dict(base_mutations)
            greedy = None
        else:
            phase = int(selected["phase"])
            greedy = run_narrow_greedy(plugin, phase, base_mutations, output_dir, max_rounds=max_rounds)
            payload["narrow_reoptimization"] = greedy
            final_mutations = dict(greedy.get("final_mutations", base_mutations))
            accepted_count = int(greedy.get("accepted_count", 0) or 0)
            payload["decision"] = "challenger_created" if accepted_count > 0 else "no_challenger"
            payload["decision_reason"] = (
                f"Accepted {accepted_count} mutations in phase {phase}."
                if accepted_count > 0
                else "Narrow greedy pass found no candidate clearing the minimum score delta."
            )

        config_path = output_dir / "challenger_optimized_config.json"
        _atomic_write_json(_jsonable(final_mutations), config_path)
        payload["challenger_config_path"] = str(config_path)
        payload["new_mutations"] = {
            key: value
            for key, value in final_mutations.items()
            if base_mutations.get(key) != value
        }

        if validate_oos and payload["decision"] == "challenger_created":
            try:
                payload["challenger_oos_validation"] = run_challenger_oos(strategy, config_path)
            except Exception:
                payload["challenger_oos_validation_error"] = traceback.format_exc()
    finally:
        _close_plugin(plugin)

    payload["elapsed_seconds"] = round(time.time() - started, 2)
    _atomic_write_json(_jsonable(payload), output_dir / "strategy_result.json")
    return payload


def format_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Challenger Diagnostic Ablation and Narrow Reoptimization")
    lines.append("")
    lines.append(f"- Calibration: {CALIBRATION_START} to {CALIBRATION_END}")
    lines.append(f"- OOS promotion check end: {OOS_END}")
    lines.append(f"- Output root: `{summary['output_root']}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Strategy | Decision | Selected phase | Accepted | Calibration PF | OOS trades | OOS PF | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for strategy, result in summary["strategies"].items():
        selected = result.get("selected_phase") or {}
        greedy = result.get("narrow_reoptimization") or {}
        metrics = result.get("champion_calibration_metrics") or {}
        oos_payload = result.get("challenger_oos_validation") or {}
        oos_metrics = oos_payload.get("oos_metrics") or {}
        lines.append(
            "| {strategy} | {decision} | {phase} | {accepted} | {pf} | {oos_trades} | {oos_pf} | {note} |".format(
                strategy=strategy,
                decision=result.get("decision", "error"),
                phase=selected.get("phase", ""),
                accepted=greedy.get("accepted_count", 0),
                pf=_fmt(metrics.get("profit_factor")),
                oos_trades=oos_metrics.get("total_trades", ""),
                oos_pf=_fmt(oos_metrics.get("profit_factor")),
                note=(result.get("decision_reason") or "").replace("|", "/"),
            )
        )
    lines.append("")
    lines.append("## Data Slice")
    lines.append("")
    for family, data in summary.get("calibration_data", {}).items():
        lines.append(f"- {family}: {data.get('files')} files, {data.get('rows')} rows, errors={len(data.get('errors', []))}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if value == "inf" or value == float("inf"):
        return "inf"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["iaric", "helix_swing", "breakout", "atrss", "brs", "nqdtc", "helix_momentum"],
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--skip-oos", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root) if args.output_root else PROJECT_ROOT / "backtests" / "output" / f"challenger_reopt_{_now_label()}"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "output_root": str(root),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_start": CALIBRATION_START,
            "calibration_end": CALIBRATION_END,
            "oos_end": OOS_END,
            "strategies": {},
        }
    summary["output_root"] = str(root)
    summary["calibration_start"] = CALIBRATION_START
    summary["calibration_end"] = CALIBRATION_END
    summary["oos_end"] = OOS_END
    summary.setdefault("strategies", {})

    calibration_summary_path = root / "calibration_data_summary.json"
    if calibration_summary_path.exists():
        summary["calibration_data"] = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    else:
        summary["calibration_data"] = prepare_calibration_data(root)
        _atomic_write_json(_jsonable(summary["calibration_data"]), calibration_summary_path)

    for strategy in args.strategies:
        print(f"[{strategy}] diagnostic ablation and narrow reoptimization...")
        try:
            summary["strategies"][strategy] = run_strategy(
                strategy,
                root,
                args.max_workers,
                args.max_rounds,
                validate_oos=not args.skip_oos,
            )
        except Exception:
            summary["strategies"][strategy] = {
                "strategy": strategy,
                "error": traceback.format_exc(),
            }
        _atomic_write_json(_jsonable(summary), root / "summary.json")

    (root / "summary.md").write_text(format_markdown(_jsonable(summary)), encoding="utf-8")
    print(f"Done. Summary: {root / 'summary.md'}")


if __name__ == "__main__":
    main()
