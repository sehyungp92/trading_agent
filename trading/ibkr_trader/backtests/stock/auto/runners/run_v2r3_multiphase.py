"""IARIC Pullback V2R3 -- Structural Engine Fixes + Re-optimization.

2-phase optimization with immutable alpha-focused scoring (80/20):
  Phase 1: Carry Quality Tuning (12 candidates)
  Phase 2: Route Expansion (10 candidates)

Engine fixes applied:
  - V2 carry quality gate (close_pct + MFE check before overnight carry)
  - DELAYED_CONFIRM allow_rescue (unlocks 76% rescue pool for highest-alpha route)

Baseline (post-V2R2): 832 trades, PF 1.55, avg_r +0.055, DD 2.1%, Sharpe 2.42.

Usage::

    python -m backtests.stock.auto.runners.run_v2r3_multiphase
    python -m backtests.stock.auto.runners.run_v2r3_multiphase --max-workers 2
    python -m backtests.stock.auto.runners.run_v2r3_multiphase --num-phases 1
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
    V2R3_BASE_MUTATIONS,
    V2R3_PHASE_CANDIDATES,
    V2R3_PHASE_FOCUS,
)
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin

DATA_DIR = Path("backtests/stock/data/raw")
DEFAULT_OUTPUT_DIR = Path("backtests/stock/auto/iaric/output_v2r3")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0
PROFILE = "mainline"
NUM_PHASES = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--profile", default=PROFILE, choices=["mainline", "aggressive"])
    parser.add_argument("--num-phases", type=int, default=NUM_PHASES)
    return parser.parse_args()


def _write_manifest(
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
    max_workers: int,
    profile: str,
    num_phases: int,
) -> None:
    phase_counts = {
        str(phase): {
            "focus": V2R3_PHASE_FOCUS[phase][0],
            "count": len(V2R3_PHASE_CANDIDATES.get(phase, [])),
            "queue_ids": [name for name, _ in V2R3_PHASE_CANDIDATES.get(phase, [])],
        }
        for phase in sorted(V2R3_PHASE_FOCUS)
        if phase <= num_phases
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "backtests/shared/auto",
        "plugin": "IARICPullbackPlugin",
        "round": "V2R3",
        "profile": profile,
        "start_date": start_date,
        "end_date": end_date,
        "initial_equity": INITIAL_EQUITY,
        "max_workers": max_workers,
        "num_phases": num_phases,
        "base_mutations": dict(sorted(V2R3_BASE_MUTATIONS.items())),
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
    print("IARIC Pullback V2R3 -- Structural Engine Fixes + Re-optimization")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Profile: {args.profile}")
    print(f"Phases: {args.num_phases}")
    print(f"Max workers: {args.max_workers}")
    total_candidates = 0
    for phase in sorted(V2R3_PHASE_FOCUS):
        if phase > args.num_phases:
            break
        candidates = V2R3_PHASE_CANDIDATES.get(phase, [])
        total_candidates += len(candidates)
        print(f"Phase {phase}: {len(candidates):>3} candidates | {V2R3_PHASE_FOCUS[phase][0]}")
    print(f"Total: {total_candidates} candidates across {args.num_phases} phases")
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
        round_name="v2r3",
    )
    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="v2r3",
        max_retries=0,
        max_diagnostic_retries=0,
    )
    runner.run_all_phases()


if __name__ == "__main__":
    main()
