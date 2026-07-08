"""Helix phased auto-optimization CLI."""
from __future__ import annotations

import argparse
import io
import json
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
from backtests.swing.auto.helix.plugin import HelixPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUND_MANAGER = RoundManager("swing", "helix")


def _load_seed_mutations(seed: str | None) -> tuple[dict, Path | None]:
    if not seed:
        return {}, None
    seed_path = Path(seed)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed config not found: {seed_path}")
    payload = json.loads(seed_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        if isinstance(payload.get("mutations"), dict):
            return dict(payload["mutations"]), seed_path
        if isinstance(payload.get("cumulative_mutations"), dict):
            return dict(payload["cumulative_mutations"]), seed_path
        return dict(payload), seed_path
    raise TypeError(f"Unexpected seed config payload in {seed_path}")


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    seed_mutations, seed_path = _load_seed_mutations(getattr(args, "seed", None))
    plugin = HelixPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=args.equity,
        max_workers=getattr(args, "max_workers", None),
        start_date=getattr(args, "start_date", None),
        end_date=getattr(args, "end_date", None),
    )
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if seed_path is not None:
        plugin.initial_mutations = seed_mutations
        plugin.initial_mutations_source = seed_path
    elif round_num > 1:
        plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(
            round_num,
            current_provenance=plugin.build_provenance(),
        )
    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.001),
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


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print("Helix auto-optimization complete.")
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
    parser = argparse.ArgumentParser(prog="helix-auto", description="Helix phased auto-optimization")
    sub = parser.add_subparsers(dest="command")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-dir", default="backtests/swing/data/raw")
        command.add_argument("--equity", type=float, default=10_000.0)
        command.add_argument("--start-date", default=None)
        command.add_argument("--end-date", default=None)
        command.add_argument("--round", type=int, default=None)
        command.add_argument("--seed", default=None, help="Optimized config JSON to use as the cumulative starting point")

    phase_run = sub.add_parser("phase-run", help="Run a single Helix phase")
    add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    phase_run.add_argument("--max-rounds", type=int, default=20)
    phase_run.add_argument("--max-workers", type=int, default=None)
    phase_run.add_argument("--min-delta", type=float, default=0.001)

    phase_auto = sub.add_parser("phase-auto", help="Run all Helix phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=20)
    phase_auto.add_argument("--max-workers", type=int, default=None)
    phase_auto.add_argument("--min-delta", type=float, default=0.001)
    phase_auto.add_argument("--max-retries", type=int, default=2)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed phase gate")
    add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])

    phase_diag = sub.add_parser("phase-diagnostics", help="Print phase diagnostics")
    phase_diag.add_argument("--round", type=int, default=None)
    phase_diag.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])

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
