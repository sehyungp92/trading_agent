"""Standalone CLI for IARIC phased auto-optimization."""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.auto.phase_gates import evaluate_gate
from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
from backtests.shared.auto.phase_state import save_phase_state
from backtests.shared.auto.round_manager import RoundManager

from .plugin import IARICPullbackPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUND_MANAGER = RoundManager("stock", "iaric")
ROUND_NAME_BY_ROUND = {
    1: "v4r1",
    2: "v5r1",
    3: "v5r2",
}
ROUND_NAME_CHOICES = ["auto", "r4", "r5", "v2r1", "v2r2", "v2r3", "v2r4", "v3r1", "v4r1", "v5r1", "v5r2"]


def _round_name_for_args(args: argparse.Namespace, *, for_write: bool) -> str:
    requested = getattr(args, "round_name", "auto")
    if requested != "auto":
        return requested
    requested_round = getattr(args, "round", None)
    if requested_round is not None:
        return ROUND_NAME_BY_ROUND.get(int(requested_round), "v5r2")
    latest = ROUND_MANAGER.get_latest_round()
    inferred_round = max(latest, 1) if not for_write else latest + 1
    return ROUND_NAME_BY_ROUND.get(inferred_round, "v5r2")


def _previous_round_name(round_num: int) -> str:
    return ROUND_NAME_BY_ROUND.get(round_num - 1, "v5r2")


def _previous_round_num_phases(round_name: str) -> int:
    return 4 if round_name == "v5r2" else 5


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    round_name = _round_name_for_args(args, for_write=for_write)
    plugin = IARICPullbackPlugin(
        data_dir=Path(args.data_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        profile=args.profile,
        round_name=round_name,
    )
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if round_num > 1:
        previous_name = _previous_round_name(round_num)
        previous_provenance = IARICPullbackPlugin(
            data_dir=Path(args.data_dir),
            start_date=args.start_date,
            end_date=args.end_date,
            initial_equity=args.equity,
            max_workers=getattr(args, "max_workers", None),
            profile=args.profile,
            num_phases=_previous_round_num_phases(previous_name),
            round_name=previous_name,
        ).build_provenance()
        plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(
            round_num,
            current_provenance=previous_provenance,
        )
        plugin.previous_round_provenance = previous_provenance
    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        round_name=f"iaric_{round_name}_{args.profile}",
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 0),
        round_manager=ROUND_MANAGER,
        round_num=round_num,
    )


def cmd_phase_run(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.load_state()
    missing = [phase for phase in range(1, args.phase) if phase not in state.completed_phases]
    if missing:
        print(
            f"Cannot run phase {args.phase} yet. Missing earlier phases: {missing}. "
            "Run phase-auto or complete prior phases first.",
            file=sys.stderr,
        )
        sys.exit(1)

    state = runner.run_phase(args.phase, state)
    result = state.phase_results.get(args.phase, {})
    print(f"IARIC phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print(f"IARIC phased auto-optimization complete ({runner.plugin.profile}, {runner.plugin._round_name}).")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False)
    state = runner.load_state()
    if args.phase not in state.phase_results:
        print(f"Phase {args.phase} has not been completed yet.")
        return

    phase_mutations = dict(getattr(runner.plugin, "initial_mutations", None) or {})
    phase_mutations.update(_mutations_through_phase(state, args.phase))
    metrics = runner.plugin.compute_final_metrics(phase_mutations)
    spec = runner.plugin.get_phase_spec(args.phase, state)
    gate = evaluate_gate(spec.gate_criteria_fn(metrics))
    state.record_gate(args.phase, {
        "passed": gate.passed,
        "criteria": [criterion.__dict__ for criterion in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": list(gate.recommendations),
    })
    save_phase_state(state, runner.state_path)

    print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for criterion in gate.criteria:
        marker = "[PASS]" if criterion.passed else "[FAIL]"
        print(f"  {marker} {criterion.name}: {criterion.actual:.4f} (target {criterion.target:.4f})")
    if not gate.passed:
        print(f"Failure category: {gate.failure_category}")
        for recommendation in gate.recommendations:
            print(f"  - {recommendation}")


def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    _, round_dir = ROUND_MANAGER.resolve_round(getattr(args, "round", None), for_write=False)
    diag_path = round_dir / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.")
        return
    print(diag_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="iaric-auto", description="IARIC phased auto-optimization")
    sub = parser.add_subparsers(dest="command")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", default="backtests/stock/data/raw")
        command.add_argument("--start-date", "--start", dest="start_date", default="2024-01-01")
        command.add_argument("--end-date", "--end", dest="end_date", default="2026-03-01")
        command.add_argument("--equity", type=float, default=10_000.0)
        command.add_argument("--profile", choices=["mainline", "aggressive"], default="mainline")
        command.add_argument("--round-name", choices=ROUND_NAME_CHOICES, default="auto")
        command.add_argument("--round", type=int, default=None)

    phase_run = sub.add_parser("phase-run", help="Run a single IARIC phase")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    phase_run.add_argument("--max-rounds", type=int, default=24)
    phase_run.add_argument("--max-workers", type=int, default=4)
    phase_run.add_argument("--min-delta", type=float, default=0.001)

    phase_auto = sub.add_parser("phase-auto", help="Run all IARIC phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=24)
    phase_auto.add_argument("--max-workers", type=int, default=4)
    phase_auto.add_argument("--min-delta", type=float, default=0.001)
    phase_auto.add_argument("--max-retries", type=int, default=0)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed IARIC phase gate")
    add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])

    phase_diag = sub.add_parser("phase-diagnostics", help="Print IARIC phase diagnostics")
    phase_diag.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    phase_diag.add_argument("--round", type=int, default=None)

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
