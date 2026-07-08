"""Standalone CLI for ALCB phased auto-optimization.

Usage::

    python -m backtests.stock.auto.alcb phase-auto --max-workers 2 -v
    python -m backtests.stock.auto.alcb phase-run --phase 1
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_runner(args: argparse.Namespace, *, for_write: bool = True):
    from backtests.shared.auto.phase_runner import PhaseRunner
    from backtests.shared.auto.round_manager import RoundManager

    from .phase_candidates import sanitize_round2_seed
    from .plugin import ALCBP16Plugin

    experiment_names = set(args.experiments) if getattr(args, "experiments", None) else None
    plugin = ALCBP16Plugin(
        data_dir=Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        experiment_names=experiment_names,
    )

    round_manager = None
    round_num = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        round_manager = RoundManager("stock", "alcb")
        round_num, output_dir = round_manager.resolve_round(
            args.round,
            for_write=for_write,
            expected_phases=plugin.num_phases if for_write else None,
        )
        if round_num == 2:
            plugin.initial_mutations = sanitize_round2_seed(
                round_manager.get_previous_mutations(
                    round_num,
                    current_provenance=plugin.build_provenance(),
                )
            )
        elif round_num > 2:
            plugin.initial_mutations = round_manager.get_previous_mutations(
                round_num,
                current_provenance=plugin.build_provenance(),
            )

    return PhaseRunner(
        plugin=plugin,
        output_dir=Path(output_dir),
        round_name="alcb",
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 0),
        round_manager=round_manager,
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
    print(f"ALCB phase {args.phase} complete.")
    print(f"Score: {result.get('base_score', 0.0):.4f} -> {result.get('final_score', 0.0):.4f}")
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print("ALCB phased auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    from backtests.shared.auto.phase_gates import evaluate_gate
    from backtests.shared.auto.phase_runner import _mutations_through_phase
    from backtests.shared.auto.phase_state import save_phase_state

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


def cmd_phase_diagnostics(args: argparse.Namespace) -> None:
    from backtests.shared.auto.round_manager import RoundManager

    if args.output_dir:
        diag_path = Path(args.output_dir) / f"phase_{args.phase}_diagnostics.txt"
    else:
        round_manager = RoundManager("stock", "alcb")
        _, round_dir = round_manager.resolve_round(args.round, for_write=False)
        diag_path = round_dir / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.")
        return
    print(diag_path.read_text(encoding="utf-8"))


def _add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("-v", "--verbose", action="store_true")
    command.add_argument("--data-dir", default="backtests/stock/data/raw")
    command.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults to centralized round output.",
    )
    command.add_argument("--start", default="2024-01-01")
    command.add_argument("--end", default="2026-03-01")
    command.add_argument("--equity", type=float, default=10_000.0)
    command.add_argument("--max-workers", type=int, default=2)
    command.add_argument("--max-rounds", type=int, default=None)
    command.add_argument("--min-delta", type=float, default=0.001)
    command.add_argument("--max-retries", type=int, default=0)
    command.add_argument("--round", type=int, default=None)
    command.add_argument("--experiments", nargs="*", help="Filter to specific experiment names")


def _build_standard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alcb-auto",
        description="ALCB phased auto-optimization (6 phases, targeted alpha extraction scoring)",
    )
    sub = parser.add_subparsers(dest="command")

    phase_run = sub.add_parser("phase-run", help="Run a single ALCB phase")
    _add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=list(range(1, 9)))

    phase_auto = sub.add_parser("phase-auto", help="Run all ALCB phases")
    _add_common(phase_auto)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed ALCB phase gate")
    _add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=list(range(1, 9)))

    phase_diag = sub.add_parser("phase-diagnostics", help="Print ALCB phase diagnostics")
    phase_diag.add_argument("--phase", type=int, required=True, choices=list(range(1, 9)))
    phase_diag.add_argument("--output-dir", default="")
    phase_diag.add_argument("--round", type=int, default=None)

    return parser


def main() -> None:
    parser = _build_standard_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return

    _setup_logging(getattr(args, "verbose", False))

    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)


if __name__ == "__main__":
    main()
