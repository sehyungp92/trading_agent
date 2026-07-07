"""VdubusNQ phased auto-optimization CLI."""
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
from backtests.shared.auto.phase_state import save_phase_state
from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
from backtests.shared.auto.round_manager import RoundManager
from backtests.momentum.auto.vdubus.plugin import VdubusPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose engine logging in main process
logging.getLogger("strategies.momentum.vdub").setLevel(logging.WARNING)
logging.getLogger("backtests.momentum.engine.vdubus_engine").setLevel(logging.WARNING)

ROUND_MANAGER = RoundManager("momentum", "vdubus")
PHASE_CHOICES = list(range(1, VdubusPlugin.num_phases + 1))


def _build_runner(
    args: argparse.Namespace,
    *,
    for_write: bool = True,
    allow_selection_drift: bool = False,
) -> PhaseRunner:
    plugin = VdubusPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        num_phases=VdubusPlugin.num_phases,
    )
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if round_num > 1:
        if for_write:
            plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(
                round_num,
                current_provenance=plugin.build_provenance(),
            )
        else:
            plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(round_num)
    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.003),
        max_retries=getattr(args, "max_retries", 2),
        round_manager=ROUND_MANAGER,
        round_num=round_num,
        allow_selection_drift=allow_selection_drift,
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
    print("VdubusNQ auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False)
    state = runner.load_state()
    if args.phase not in state.phase_results:
        print(f"Phase {args.phase} has not been completed yet.")
        return

    plugin = runner.plugin
    phase_mutations = dict(getattr(runner.plugin, "initial_mutations", None) or {})
    phase_mutations.update(_mutations_through_phase(state, args.phase))
    metrics = plugin.compute_final_metrics(phase_mutations)
    spec = plugin.get_phase_spec(args.phase, state)
    gate = evaluate_gate(spec.gate_criteria_fn(metrics))
    state.record_gate(args.phase, {
        "passed": gate.passed,
        "criteria": [c.__dict__ for c in gate.criteria],
        "failure_category": gate.failure_category,
        "recommendations": list(gate.recommendations),
    })
    save_phase_state(state, runner.state_path)

    print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for c in gate.criteria:
        marker = "[PASS]" if c.passed else "[FAIL]"
        print(f"  {marker} {c.name}: {c.actual:.4f} (target {c.target:.4f})")
    if not gate.passed:
        print(f"Failure category: {gate.failure_category}")
        for r in gate.recommendations:
            print(f"  - {r}")


def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    _, round_dir = ROUND_MANAGER.resolve_round(getattr(args, "round", None), for_write=False)
    diag_path = round_dir / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.")
        return
    print(diag_path.read_text(encoding="utf-8"))


def cmd_final_diagnostics(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False, allow_selection_drift=True)
    state = runner.load_state()
    runner.run_end_of_round(state)
    print("VdubusNQ final diagnostics regenerated.")
    print(f"Diagnostics: {runner.output_dir / 'round_final_diagnostics.txt'}")
    print(f"Evaluation:  {runner.output_dir / 'round_evaluation.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vdubus-auto", description="VdubusNQ phased auto-optimization")
    sub = parser.add_subparsers(dest="command")

    def add_common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--data-dir", default="backtests/momentum/data/raw")
        cmd.add_argument("--equity", type=float, default=10_000.0)
        cmd.add_argument("--round", type=int, default=None)

    phase_run = sub.add_parser("phase-run", help="Run a single VdubusNQ phase")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=PHASE_CHOICES)
    phase_run.add_argument("--max-rounds", type=int, default=50)
    phase_run.add_argument("--max-workers", type=int, default=None)
    phase_run.add_argument("--min-delta", type=float, default=0.003)

    phase_auto = sub.add_parser("phase-auto", help="Run all VdubusNQ phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=50)
    phase_auto.add_argument("--max-workers", type=int, default=None)
    phase_auto.add_argument("--min-delta", type=float, default=0.003)
    phase_auto.add_argument("--max-retries", type=int, default=2)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed phase gate")
    add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=PHASE_CHOICES)

    phase_diag = sub.add_parser("phase-diagnostics", help="Print phase diagnostics")
    phase_diag.add_argument("--phase", type=int, required=True, choices=PHASE_CHOICES)
    phase_diag.add_argument("--round", type=int, default=None)

    final_diag = sub.add_parser("final-diagnostics", help="Regenerate latest VdubusNQ round diagnostics")
    add_common(final_diag)

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    elif args.command == "final-diagnostics":
        cmd_final_diagnostics(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
