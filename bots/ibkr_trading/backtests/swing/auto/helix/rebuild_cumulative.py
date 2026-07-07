"""Run Helix phased optimization from an explicit cumulative mutation seed."""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.round_manager import RoundManager
from backtests.swing.auto.helix.plugin import HelixPlugin


def _load_mutations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("mutations"), dict):
        return dict(payload["mutations"])
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Unexpected mutation seed payload in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", default="", help="JSON mutation seed for round 1 rebuilds.")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    manager = RoundManager("swing", "helix")
    plugin = HelixPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=args.max_workers,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    round_num, round_dir = manager.resolve_round(
        args.round,
        for_write=True,
        expected_phases=plugin.num_phases,
    )

    baseline_source: Path | None = None
    if args.seed:
        baseline_source = Path(args.seed)
        plugin.initial_mutations = _load_mutations(baseline_source)
    elif round_num > 1:
        plugin.initial_mutations = manager.get_previous_mutations(
            round_num,
            current_provenance=plugin.build_provenance(),
        )

    description = (
        f"Round {round_num} Helix cumulative rebuild/optimization "
        f"(IS through {args.end_date or 'latest'}, equity ${args.equity:,.0f})"
    )
    manager.write_run_spec(
        round_dir,
        round_num,
        plugin.name,
        description=description,
        baseline_mutations=dict(plugin.initial_mutations or {}),
        baseline_source=baseline_source,
        provenance=plugin.build_provenance(),
        provenance_status="complete",
        execution_context={
            "data_dir": str(Path(args.data_dir)),
            "initial_equity": args.equity,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "max_workers": args.max_workers,
            "max_rounds": args.max_rounds,
            "min_delta": args.min_delta,
            "max_retries": args.max_retries,
            "execution_mode": "independent_optimizer_replay",
        },
        overwrite=True,
    )

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        round_name=description,
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        max_retries=args.max_retries,
        round_manager=manager,
        round_num=round_num,
    )
    state = runner.run_all_phases()
    print("Helix cumulative rebuild complete.")
    print(f"Round: {round_num}")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")
    print(f"Round dir: {round_dir}")


if __name__ == "__main__":
    main()
