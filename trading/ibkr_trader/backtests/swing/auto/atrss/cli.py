"""ATRSS phased auto-optimization CLI."""
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
from backtests.swing.auto.atrss.plugin import ATRSSPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUND_MANAGER = RoundManager("swing", "atrss")


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    plugin = ATRSSPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        mode=getattr(args, "mode", "synchronized"),
        candidate_profile=getattr(args, "candidate_profile", "auto"),
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
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.005),
        max_retries=getattr(args, "max_retries", 2),
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


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print("ATRSS auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    _, round_dir = ROUND_MANAGER.resolve_round(getattr(args, "round", None), for_write=False)
    diag_path = round_dir / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.")
        return
    print(diag_path.read_text(encoding="utf-8"))


def cmd_status(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False)
    state = runner.load_state()
    print(f"Completed phases: {state.completed_phases}")
    print(f"Cumulative mutations ({len(state.cumulative_mutations)}):")
    for key, value in sorted(state.cumulative_mutations.items()):
        print(f"  {key}: {value}")
    for phase in sorted(state.phase_results.keys()):
        result = state.phase_results[phase]
        accepted = len(result.get("kept_features", []))
        score = result.get("final_score", 0.0)
        print(f"  Phase {phase}: {accepted} accepted, score={score:.4f}")


def _add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--data-dir", default="backtests/swing/data/raw")
    command.add_argument("--equity", type=float, default=10_000.0)
    command.add_argument("--max-workers", type=int, default=None)
    command.add_argument("--max-rounds", type=int, default=None)
    command.add_argument("--min-delta", type=float, default=0.005)
    command.add_argument("--max-retries", type=int, default=2)
    command.add_argument("--round", type=int, default=None)
    command.add_argument("--mode", choices=["independent", "synchronized"], default="synchronized")
    command.add_argument("--candidate-profile", choices=["auto", "alpha", "risk"], default="auto")


def main() -> None:
    parser = argparse.ArgumentParser(prog="atrss-auto", description="ATRSS R1 phased auto-optimization")
    sub = parser.add_subparsers(dest="command")

    phase_run = sub.add_parser("phase-run", help="Run a single ATRSS phase")
    _add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])

    phase_auto = sub.add_parser("phase-auto", help="Run all ATRSS phases")
    _add_common(phase_auto)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed ATRSS phase gate")
    _add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])

    phase_diag = sub.add_parser("phase-diagnostics", help="Print ATRSS phase diagnostics")
    phase_diag.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])
    phase_diag.add_argument("--round", type=int, default=None)

    status = sub.add_parser("status", help="Show ATRSS optimization status")
    _add_common(status)

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
