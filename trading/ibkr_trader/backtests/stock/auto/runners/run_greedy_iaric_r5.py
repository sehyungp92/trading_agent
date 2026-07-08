"""Run IARIC Pullback Round 5 -- Exploit Unexplored Parameter Space.

Stage 1: Run with existing engine code but better-designed candidates
covering the unexplored signal families, FSM params, entry thresholds,
rescue flow, and rebalanced scoring (trade volume + total R weighted).

Uses max_workers=2 by default.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.stock.auto.iaric.phase_candidates import (
    R5_BASE_MUTATIONS,
    R5_PHASE_CANDIDATES,
    R5_PHASE_FOCUS,
)
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin

DATA_DIR = Path("backtests/stock/data/raw")
DEFAULT_OUTPUT_DIR = Path("backtests/stock/auto/iaric/output_r5")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0
PROFILE = "aggressive"
NUM_PHASES = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--profile", default=PROFILE, choices=["mainline", "aggressive"])
    parser.add_argument("--num-phases", type=int, default=NUM_PHASES)
    return parser.parse_args()


def _write_manifest(output_dir: Path, *, start_date: str, end_date: str, max_workers: int, profile: str, num_phases: int) -> None:
    phase_counts = {
        str(phase): {
            "focus": R5_PHASE_FOCUS[phase][0],
            "count": len(R5_PHASE_CANDIDATES.get(phase, [])),
            "queue_ids": [name for name, _ in R5_PHASE_CANDIDATES.get(phase, [])],
        }
        for phase in sorted(R5_PHASE_FOCUS)
        if phase <= num_phases
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "backtests/shared/auto",
        "plugin": "IARICPullbackPlugin",
        "round": "R5",
        "profile": profile,
        "start_date": start_date,
        "end_date": end_date,
        "initial_equity": INITIAL_EQUITY,
        "max_workers": max_workers,
        "num_phases": num_phases,
        "base_mutations": dict(sorted(R5_BASE_MUTATIONS.items())),
        "phase_count": len(phase_counts),
        "queue_count": sum(item["count"] for item in phase_counts.values()),
        "phases": phase_counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 72)
    print("IARIC Pullback Round 5 -- Exploit Unexplored Parameter Space")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Profile: {args.profile}")
    print(f"Phases: {args.num_phases}")
    print(f"Max workers: {args.max_workers}")
    for phase in sorted(R5_PHASE_FOCUS):
        if phase > args.num_phases:
            break
        candidates = R5_PHASE_CANDIDATES.get(phase, [])
        print(f"Phase {phase}: {len(candidates)} candidates | {R5_PHASE_FOCUS[phase][0]}")
    print(flush=True)

    _write_manifest(
        output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        profile=args.profile,
        num_phases=args.num_phases,
    )

    plugin = IARICPullbackPlugin(
        DATA_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=INITIAL_EQUITY,
        max_workers=args.max_workers,
        num_phases=args.num_phases,
        profile=args.profile,
        round_name="r5",
    )
    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="r5",
        max_retries=0,
        max_diagnostic_retries=0,
    )
    runner.run_all_phases()


if __name__ == "__main__":
    main()
