"""IARIC Pullback V4R1 -- Comprehensive Auto-Optimization.

Correctly starts from V2R4's proven V2 engine base (46 mutations),
systematically tests all features from all three lineages (V2R4,
multiphase, tier B), and re-evaluates tier B options on the clean V2 base.

5-phase greedy selection with immutable scoring and progressive hard rejects:
  Phase 1: V2R4 Structural Ablation (15 candidates)
  Phase 2: Tier B Full Grid (11 candidates)
  Phase 3: Multiphase Feature Adoption (19 candidates)
  Phase 4: Exit Mechanics Tuning (20 candidates)
  Phase 5: Signal Threshold Sweep (10 candidates)

Usage::

    python -m backtests.stock.auto.runners.run_v4r1_comprehensive
    python -m backtests.stock.auto.runners.run_v4r1_comprehensive --max-workers 2
    python -m backtests.stock.auto.runners.run_v4r1_comprehensive --num-phases 1
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
from backtests.shared.auto.round_manager import RoundManager
from backtests.stock.auto.iaric.phase_candidates import (
    V4R1_BASE_MUTATIONS,
    V4R1_PHASE_CANDIDATES,
    V4R1_PHASE_FOCUS,
)
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin

DATA_DIR = Path("backtests/stock/data/raw")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0
PROFILE = "mainline"
NUM_PHASES = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--profile", default=PROFILE, choices=["mainline", "aggressive"])
    parser.add_argument("--num-phases", type=int, default=NUM_PHASES)
    parser.add_argument("--round", type=int, default=None)
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
            "focus": V4R1_PHASE_FOCUS[phase][0],
            "count": len(V4R1_PHASE_CANDIDATES.get(phase, [])),
            "queue_ids": [name for name, _ in V4R1_PHASE_CANDIDATES.get(phase, [])],
        }
        for phase in sorted(V4R1_PHASE_FOCUS)
        if phase <= num_phases
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "backtests/shared/auto",
        "plugin": "IARICPullbackPlugin",
        "round": "V4R1",
        "profile": profile,
        "start_date": start_date,
        "end_date": end_date,
        "initial_equity": INITIAL_EQUITY,
        "max_workers": max_workers,
        "num_phases": num_phases,
        "base_mutations": dict(sorted(V4R1_BASE_MUTATIONS.items())),
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
    round_manager = None
    round_num = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        round_manager = RoundManager("stock", "iaric")
        round_num, output_dir = round_manager.resolve_round(
            args.round,
            for_write=True,
            expected_phases=args.num_phases,
        )

    print("=" * 72)
    print("IARIC Pullback V4R1 -- Comprehensive Auto-Optimization")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Profile: {args.profile}")
    print(f"Phases: {args.num_phases}")
    print(f"Max workers: {args.max_workers}")
    total_candidates = 0
    for phase in sorted(V4R1_PHASE_FOCUS):
        if phase > args.num_phases:
            break
        candidates = V4R1_PHASE_CANDIDATES.get(phase, [])
        total_candidates += len(candidates)
        print(f"Phase {phase}: {len(candidates):>3} candidates | {V4R1_PHASE_FOCUS[phase][0]}")
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
        round_name="v4r1",
    )
    if round_manager and round_num and round_num > 1:
        plugin.initial_mutations = round_manager.get_previous_mutations(
            round_num,
            current_provenance=plugin.build_provenance(),
        )
    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="v4r1",
        max_retries=0,
        max_diagnostic_retries=0,
        round_manager=round_manager,
        round_num=round_num,
    )
    runner.run_all_phases()


if __name__ == "__main__":
    main()
