"""CLI for the Helix leakage-control optimizer."""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parents[5]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.round_manager import RoundManager
from backtests.swing.auto.helix.leakage_control.plugin import HelixLeakageControlPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROUND_MANAGER = RoundManager("swing", "helix")


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    plugin = HelixLeakageControlPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", 2),
    )
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if round_num > 1:
        plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(
            round_num,
            current_provenance=plugin.build_provenance(),
        )
    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        round_name="Helix right-then-stopped leakage control",
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 1),
        round_manager=ROUND_MANAGER,
        round_num=round_num,
    )


def cmd_phase_run(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_phase(args.phase)
    result = state.phase_results.get(args.phase, {})
    print(f"Phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print("Helix leakage-control auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    _, round_dir = ROUND_MANAGER.resolve_round(getattr(args, "round", None), for_write=False)
    diag_path = round_dir / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.")
        return
    print(diag_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="helix-leakage-control",
        description="Helix right-then-stopped leakage-control phased auto-optimization",
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", default="backtests/swing/data/raw")
        command.add_argument("--equity", type=float, default=10_000.0)
        command.add_argument("--round", type=int, default=None)

    phase_run = sub.add_parser("phase-run", help="Run one leakage-control phase")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=[1, 2])
    phase_run.add_argument("--max-rounds", type=int, default=10)
    phase_run.add_argument("--max-workers", type=int, default=2)
    phase_run.add_argument("--min-delta", type=float, default=0.001)
    phase_run.add_argument("--max-retries", type=int, default=1)

    phase_auto = sub.add_parser("phase-auto", help="Run all leakage-control phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=10)
    phase_auto.add_argument("--max-workers", type=int, default=2)
    phase_auto.add_argument("--min-delta", type=float, default=0.001)
    phase_auto.add_argument("--max-retries", type=int, default=1)

    phase_diag = sub.add_parser("phase-diagnostics", help="Print phase diagnostics")
    phase_diag.add_argument("--round", type=int, default=None)
    phase_diag.add_argument("--phase", type=int, required=True, choices=[1, 2])

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
