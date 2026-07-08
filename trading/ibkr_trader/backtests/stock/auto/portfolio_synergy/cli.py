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
from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
from backtests.shared.auto.phase_state import save_phase_state
from backtests.shared.auto.round_manager import RoundManager

from .phase_candidates import BLOCKED_ALPHA_ROUND3_PROFILE, DEFAULT_PROFILE, INITIAL_EQUITY
from .plugin import StockPortfolioSynergyPlugin
from .round_design import write_round_design

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROUND_MANAGER = RoundManager("stock", "portfolio_synergy")


def _round_label(round_num: int) -> str:
    return f"round_{round_num}_dynamic_stock_synergy"


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        if isinstance(payload.get("mutations"), dict):
            return dict(payload["mutations"])
        if isinstance(payload.get("cumulative_mutations"), dict):
            return dict(payload["cumulative_mutations"])
        return dict(payload)
    raise TypeError(f"Expected a JSON object in {path}")


def _existing_run_spec_baseline(round_dir: Path) -> dict | None:
    path = ROUND_MANAGER.run_spec_path(round_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    baseline = payload.get("baseline_mutations")
    return dict(baseline) if isinstance(baseline, dict) else None


def _build_runner(args: argparse.Namespace, *, for_write: bool = True) -> PhaseRunner:
    resolve_for_write = for_write and getattr(args, "start_phase", None) is None
    plugin = StockPortfolioSynergyPlugin(
        data_dir=Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=float(args.equity),
        max_workers=getattr(args, "max_workers", 1),
        round_profile=getattr(args, "profile", DEFAULT_PROFILE),
    )
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=resolve_for_write,
        expected_phases=plugin.num_phases if resolve_for_write else None,
    )
    round_label = _round_label(round_num)
    plugin.diagnostic_round_label = round_label
    provenance = plugin.build_provenance()
    baseline_source = None
    baseline_config = getattr(args, "baseline_config", None)
    if baseline_config:
        if getattr(args, "start_phase", None) is not None:
            raise ValueError("--baseline-config can only be used when starting a round from phase 1.")
        baseline_path = Path(baseline_config)
        plugin.initial_mutations = _load_config(baseline_path)
        baseline_source = baseline_path
    elif round_num > 1:
        existing_baseline = _existing_run_spec_baseline(round_dir)
        if existing_baseline is not None:
            ROUND_MANAGER.write_run_spec(
                round_dir,
                round_num,
                plugin.name,
                provenance=provenance,
            )
            plugin.initial_mutations = existing_baseline
        else:
            plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(
                round_num,
                current_provenance=provenance,
            )
    if baseline_source is not None:
        plugin.initial_mutations_source = baseline_source

    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        round_name=round_label,
        max_rounds=getattr(args, "max_rounds", None),
        min_delta=getattr(args, "min_delta", 0.003),
        max_retries=getattr(args, "max_retries", 0),
        max_diagnostic_retries=getattr(args, "max_diagnostic_retries", 0),
        round_manager=ROUND_MANAGER,
        round_num=round_num,
        allow_selection_drift=bool(baseline_config),
    )


def cmd_design(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    paths = write_round_design(output_dir, repo_root=Path.cwd())
    print("Stock portfolio synergy round design written.")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases(start_phase=getattr(args, "start_phase", None))
    print("Stock portfolio synergy auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


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
    state.record_gate(
        args.phase,
        {
            "passed": gate.passed,
            "criteria": [criterion.__dict__ for criterion in gate.criteria],
            "failure_category": gate.failure_category,
            "recommendations": list(gate.recommendations),
        },
    )
    save_phase_state(state, runner.state_path)

    print(f"Phase {args.phase} gate: {'PASSED' if gate.passed else 'FAILED'}")
    for criterion in gate.criteria:
        marker = "[PASS]" if criterion.passed else "[FAIL]"
        print(f"  {marker} {criterion.name}: {criterion.actual:.4f} (target {criterion.target:.4f})")


def cmd_status(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False)
    state = runner.load_state()
    print(f"Completed phases: {state.completed_phases}")
    print(f"Cumulative mutations ({len(state.cumulative_mutations)}):")
    for key, value in sorted(state.cumulative_mutations.items()):
        print(f"  {key}: {value}")
    for phase in sorted(state.phase_results):
        result = state.phase_results[phase]
        print(
            f"  Phase {phase}: {len(result.get('kept_features', []))} accepted, "
            f"score={result.get('final_score', 0.0):.4f}"
        )


def _add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--data-dir", default="backtests/stock/data/raw")
    command.add_argument("--start", default="2024-01-01")
    command.add_argument("--end", default="2026-03-01")
    command.add_argument("--equity", type=float, default=INITIAL_EQUITY)
    command.add_argument("--max-workers", type=int, default=1)
    command.add_argument("--max-rounds", type=int, default=None)
    command.add_argument("--min-delta", type=float, default=0.003)
    command.add_argument("--max-retries", type=int, default=0)
    command.add_argument("--max-diagnostic-retries", type=int, default=0)
    command.add_argument("--round", type=int, default=None)
    command.add_argument(
        "--profile",
        choices=(DEFAULT_PROFILE, BLOCKED_ALPHA_ROUND3_PROFILE),
        default=DEFAULT_PROFILE,
    )
    command.add_argument("--baseline-config", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stock-portfolio-synergy-auto",
        description="Two-sleeve stock portfolio synergy phased auto-optimization",
    )
    sub = parser.add_subparsers(dest="command")

    design = sub.add_parser("design", help="Write the phase-auto round design artifacts")
    design.add_argument("--output-dir", default="backtests/output/stock/portfolio_synergy/round_1_design")

    phase_auto = sub.add_parser("phase-auto", help="Run all stock portfolio synergy phases")
    _add_common(phase_auto)
    phase_auto.add_argument("--start-phase", type=int, default=None, choices=range(1, 8))

    phase_run = sub.add_parser("phase-run", help="Run one stock portfolio synergy phase")
    _add_common(phase_run)
    phase_run.add_argument("--phase", type=int, required=True, choices=range(1, 8))

    phase_gate = sub.add_parser("phase-gate", help="Check a completed stock portfolio synergy phase gate")
    _add_common(phase_gate)
    phase_gate.add_argument("--phase", type=int, required=True, choices=range(1, 8))

    status = sub.add_parser("status", help="Show portfolio synergy optimization status")
    _add_common(status)

    args = parser.parse_args()
    if args.command == "design":
        cmd_design(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
