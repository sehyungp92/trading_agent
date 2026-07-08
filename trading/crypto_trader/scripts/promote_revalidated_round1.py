"""Promote cleaned-seed revalidation reruns into live round_1 strategy folders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVALIDATED_ROOT = ROOT / "output" / "revalidated"
STRATEGIES = ("momentum", "trend", "breakout")
MANIFEST_METRIC_KEYS = (
    "total_trades",
    "win_rate",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe_ratio",
    "calmar_ratio",
    "net_return_pct",
)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _detect_latest_run_root() -> Path:
    candidates = sorted(
        path
        for path in REVALIDATED_ROOT.iterdir()
        if path.is_dir()
    )
    if not candidates:
        raise RuntimeError(f"No revalidation runs found under {REVALIDATED_ROOT}.")
    return candidates[-1]


def _build_manifest_entry(run_summary: dict) -> dict:
    mutations = dict(run_summary.get("cumulative_mutations") or {})
    metrics = dict(run_summary.get("final_metrics") or {})
    entry = {
        "round": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mutations_count": len(mutations),
        "mutations": mutations,
    }
    for key in MANIFEST_METRIC_KEYS:
        if key in metrics:
            entry[key] = metrics[key]
    return entry


def _build_trend_round1_summary(target_dir: Path, run_summary: dict) -> dict:
    return {
        "round": 1,
        "output_dir": str(target_dir),
        "completed_phases": list(run_summary.get("completed_phases") or []),
        "mutations": dict(run_summary.get("cumulative_mutations") or {}),
        "final_metrics": dict(run_summary.get("final_metrics") or {}),
        "baseline_seed": run_summary.get("baseline_seed"),
    }


def _promote_strategy(run_root: Path, strategy: str) -> dict:
    source_dir = run_root / strategy / "cleaned_seed_rerun"
    if not source_dir.exists():
        raise RuntimeError(
            f"Expected cleaned-seed rerun artifacts for {strategy} at {source_dir}."
        )

    target_base = ROOT / "output" / strategy
    target_dir = target_base / "round_1"
    manifest_path = target_base / "rounds_manifest.json"
    run_summary_path = source_dir / "run_summary.json"

    if target_dir.exists():
        raise RuntimeError(f"Target round_1 already exists for {strategy}: {target_dir}")

    if not run_summary_path.exists():
        raise RuntimeError(f"Missing run summary for {strategy}: {run_summary_path}")

    shutil.copytree(source_dir, target_dir)
    run_summary = _load_json(target_dir / "run_summary.json")
    manifest = {"rounds": [_build_manifest_entry(run_summary)]}
    _write_json(manifest_path, manifest)

    if strategy == "trend":
        _write_json(
            target_dir / "round1_summary.json",
            _build_trend_round1_summary(target_dir, run_summary),
        )

    return {
        "strategy": strategy,
        "target_dir": str(target_dir),
        "manifest_path": str(manifest_path),
        "mutations": run_summary.get("cumulative_mutations") or {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default=None,
        help="Specific revalidation run root to promote. Defaults to the latest run.",
    )
    parser.add_argument(
        "--strategy",
        choices=("all",) + STRATEGIES,
        default="all",
        help="Strategy to promote.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root) if args.run_root else _detect_latest_run_root()
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)

    if not run_root.exists():
        raise RuntimeError(f"Revalidation run root does not exist: {run_root}")

    results = [_promote_strategy(run_root, strategy) for strategy in strategies]
    print(json.dumps({"run_root": str(run_root), "results": results}, indent=2))


if __name__ == "__main__":
    main()
