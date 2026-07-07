"""Crisis detection phased auto-optimization CLI.

Usage:
    python -m backtests.regime.auto.crisis phase-auto
    python -m backtests.regime.auto.crisis phase-run --phase 1
    python -m backtests.regime.auto.crisis phase-gate --phase 1
    python -m backtests.regime.auto.crisis phase-diagnostics --phase 1
    python -m backtests.regime.auto.crisis validate-robustness
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
    )

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.auto.phase_gates import evaluate_gate
from backtests.shared.auto.phase_state import save_phase_state
from backtests.shared.auto.phase_runner import PhaseRunner, _mutations_through_phase
from backtests.shared.auto.round_manager import RoundManager
from backtests.regime.auto.crisis.plugin import CrisisPlugin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROUND_MANAGER = RoundManager("regime", "crisis")


def _build_runner(
    args: argparse.Namespace, *, for_write: bool = True,
) -> PhaseRunner:
    plugin = CrisisPlugin(data_dir=Path(args.data_dir))
    round_num, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=for_write,
        expected_phases=plugin.num_phases if for_write else None,
    )
    if round_num > 1:
        plugin.initial_mutations = ROUND_MANAGER.get_previous_mutations(round_num)
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
    print(
        f"Score: {result.get('base_score', 0.0):.4f} -> "
        f"{result.get('final_score', 0.0):.4f}"
    )
    print(f"Accepted: {len(result.get('kept_features', []))}")


def cmd_phase_auto(args: argparse.Namespace) -> None:
    runner = _build_runner(args)
    state = runner.run_all_phases()
    print("Crisis detection auto-optimization complete.")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Final mutations: {len(state.cumulative_mutations)}")


def cmd_phase_gate(args: argparse.Namespace) -> None:
    runner = _build_runner(args, for_write=False)
    state = runner.load_state()
    if args.phase not in state.phase_results:
        print(f"Phase {args.phase} has not been completed yet.")
        return

    plugin = runner.plugin
    phase_mutations = _mutations_through_phase(state, args.phase)
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
    _, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None), for_write=False,
    )
    diag_path = round_dir / f"phase_{args.phase}_diagnostics.txt"
    if not diag_path.exists():
        print(f"No diagnostics found at {diag_path}.")
        return
    print(diag_path.read_text(encoding="utf-8"))


def cmd_validate_robustness(args: argparse.Namespace) -> None:
    """Run post-optimization robustness validation using fast vectorized path."""
    from .worker import init_worker, _fast_evaluate, _INTEGER_PARAMS

    init_worker(str(args.data_dir))

    # Load optimized mutations
    round_num = getattr(args, "round", None)
    if round_num is None:
        round_num = ROUND_MANAGER.get_latest_round()
    mutations = ROUND_MANAGER.get_previous_mutations(round_num + 1)  # loads round_num's config

    perturbation = getattr(args, "perturbation", 0.10)

    float_params = [
        k for k, v in mutations.items()
        if k not in _INTEGER_PARAMS and isinstance(v, (int, float))
    ]

    scenarios: list[tuple[str, dict]] = []
    for key in float_params:
        for direction, factor in [("plus", 1 + perturbation), ("minus", 1 - perturbation)]:
            perturbed = dict(mutations)
            perturbed[key] = mutations[key] * factor
            scenarios.append((f"{key}_{direction}_{perturbation:.0%}", perturbed))
    for direction, factor in [("all_plus", 1 + perturbation), ("all_minus", 1 - perturbation)]:
        perturbed = {
            k: (v * factor if k in float_params else v)
            for k, v in mutations.items()
        }
        scenarios.append((direction, perturbed))

    passed = 0
    failed_scenarios: list[tuple[str, object]] = []
    for name, perturbed_muts in scenarios:
        try:
            metrics = _fast_evaluate(perturbed_muts)
            detected = int(metrics.get("crises_detected", 0))
            if detected >= 7:
                passed += 1
            else:
                failed_scenarios.append((name, detected))
        except Exception as e:
            failed_scenarios.append((name, f"error: {e}"))

    total = len(scenarios)
    print(f"\nRobustness Validation: {passed}/{total} scenarios passed")
    print(f"Perturbation: +/-{perturbation:.0%}")
    if failed_scenarios:
        print(f"\nFailed scenarios ({len(failed_scenarios)}):")
        for name, detail in failed_scenarios:
            print(f"  {name}: {detail}")
    else:
        print("All scenarios detect 7/7 crises. Parameter stability confirmed.")


def cmd_event_chronology(args: argparse.Namespace) -> None:
    """Generate event-level channel chronology diagnostics for the detector."""
    import pandas as pd

    from backtests.regime.crisis_validation import (
        build_event_channel_chronology,
        run_crisis_detector,
    )

    data_path = Path(args.data_dir)
    market_df = pd.read_parquet(data_path / "market_df.parquet")
    strat_ret_df = pd.read_parquet(data_path / "strat_ret_df.parquet")
    logging.getLogger("regime.crisis.hysteresis").setLevel(logging.WARNING)
    alerts = run_crisis_detector(market_df, strat_ret_df)
    chronology = build_event_channel_chronology(alerts)

    _, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=False,
    )
    out_path = Path(args.out) if args.out else round_dir / "event_channel_chronology.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chronology, indent=2), encoding="utf-8")

    print(f"Event channel chronology written to {out_path}")
    for name, item in chronology.items():
        latency = item.get("latency_days")
        bottleneck = item.get("bottleneck_channel") or "n/a"
        print(
            f"  {name}: detected={item.get('detected')} "
            f"latency={latency}d bottleneck={bottleneck}"
        )


def _economic_inputs(args: argparse.Namespace):
    import pandas as pd

    from backtests.regime.auto.crisis.economic import (
        DEFAULT_SIGNALS_PATH,
        build_base_portfolios,
        load_replay_inputs,
    )
    from backtests.regime.crisis_validation import run_crisis_detector

    signals_path = Path(args.signals_path) if args.signals_path else DEFAULT_SIGNALS_PATH
    market_df, strat_ret_df, signals_df = load_replay_inputs(
        Path(args.data_dir),
        signals_path=signals_path,
    )
    logging.getLogger("regime.crisis.hysteresis").setLevel(logging.WARNING)
    alerts = run_crisis_detector(market_df, strat_ret_df)
    base_portfolios = build_base_portfolios(strat_ret_df, signals_df)
    cash_returns = strat_ret_df.get("CASH", pd.Series(0.0, index=strat_ret_df.index))
    return alerts, base_portfolios, cash_returns


def _sleeve_economic_inputs(args: argparse.Namespace):
    import pandas as pd

    from backtests.regime.auto.crisis.economic import (
        DEFAULT_SIGNALS_PATH,
        load_replay_inputs,
    )
    from backtests.regime.crisis_validation import run_crisis_detector

    signals_path = Path(args.signals_path) if args.signals_path else DEFAULT_SIGNALS_PATH
    market_df, strat_ret_df, _ = load_replay_inputs(
        Path(args.data_dir),
        signals_path=signals_path,
    )
    logging.getLogger("regime.crisis.hysteresis").setLevel(logging.WARNING)
    alerts = run_crisis_detector(market_df, strat_ret_df)
    cash_returns = strat_ret_df.get("CASH", pd.Series(0.0, index=strat_ret_df.index))
    return alerts, strat_ret_df, cash_returns


def cmd_economic_evaluate(args: argparse.Namespace) -> None:
    """Evaluate fixed crisis action scenarios against portfolio economics."""
    from backtests.regime.auto.crisis.economic import (
        build_economic_report,
        evaluate_standard_scenarios,
    )

    alerts, base_portfolios, cash_returns = _economic_inputs(args)
    scenarios = evaluate_standard_scenarios(
        alerts_df=alerts,
        base_portfolios=base_portfolios,
        cash_returns=cash_returns,
    )
    result = {
        "baseline_policy": None,
        "optimized_policy": None,
        "base_score": scenarios["C_current_live"]["score"],
        "optimized_score": scenarios["C_current_live"]["score"],
        "accepted": [],
        "rounds": [],
        "standard_scenarios": scenarios,
    }

    _, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=False,
    )
    out_path = Path(args.out) if args.out else round_dir / "economic_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = out_path.with_suffix(".md")
    report_path.write_text(build_economic_report(result), encoding="utf-8")

    print(f"Economic evaluation written to {out_path}")
    print(f"Economic report written to {report_path}")
    for name, item in scenarios.items():
        print(
            f"  {name}: score={item['score']:.6f} "
            f"action_days={item['action_day_share']:.1%} "
            f"avg_exposure={item['avg_exposure']:.3f}"
        )


def cmd_economic_optimize(args: argparse.Namespace) -> None:
    """Run greedy action-layer optimization with detector thresholds frozen."""
    from backtests.regime.auto.crisis.economic import (
        build_economic_report,
        optimize_action_policy,
    )

    alerts, base_portfolios, cash_returns = _economic_inputs(args)
    result = optimize_action_policy(
        alerts_df=alerts,
        base_portfolios=base_portfolios,
        cash_returns=cash_returns,
        min_delta=args.min_delta,
        max_rounds=args.max_rounds,
    )

    _, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=False,
    )
    out_path = Path(args.out) if args.out else round_dir / "economic_optimization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = out_path.with_suffix(".md")
    report_path.write_text(build_economic_report(result), encoding="utf-8")

    print(f"Economic optimization written to {out_path}")
    print(f"Economic report written to {report_path}")
    print(f"Base score: {result['base_score']:.6f}")
    print(f"Optimized score: {result['optimized_score']:.6f}")
    print(f"Accepted: {result['accepted'] or 'none'}")


def cmd_sleeve_optimize(args: argparse.Namespace) -> None:
    """Run sleeve-aware action-layer optimization with thresholds frozen."""
    from backtests.regime.auto.crisis.economic import (
        build_sleeve_economic_report,
        optimize_sleeve_action_policy,
    )

    alerts, strat_ret_df, cash_returns = _sleeve_economic_inputs(args)
    result = optimize_sleeve_action_policy(
        alerts_df=alerts,
        strat_ret_df=strat_ret_df,
        cash_returns=cash_returns,
    )

    _, round_dir = ROUND_MANAGER.resolve_round(
        getattr(args, "round", None),
        for_write=False,
    )
    out_path = Path(args.out) if args.out else round_dir / "sleeve_economic_optimization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = out_path.with_suffix(".md")
    report_path.write_text(build_sleeve_economic_report(result), encoding="utf-8")

    print(f"Sleeve economic optimization written to {out_path}")
    print(f"Sleeve economic report written to {report_path}")
    print(f"Base score: {result['base_score']:.6f}")
    print(f"Optimized score: {result['optimized_score']:.6f}")
    print(f"Accepted: {result['accepted'] or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="crisis-auto",
        description="Crisis detection threshold auto-optimization",
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--data-dir", default="backtests/regime/data/raw")
        cmd.add_argument("--round", type=int, default=None)

    phase_run = sub.add_parser("phase-run", help="Run a single phase")
    add_common(phase_run)
    phase_run.add_argument(
        "--phase", type=int, required=True, choices=[1, 2, 3, 4],
    )
    phase_run.add_argument("--max-rounds", type=int, default=50)
    phase_run.add_argument("--min-delta", type=float, default=0.005)

    phase_auto = sub.add_parser("phase-auto", help="Run all 4 phases")
    add_common(phase_auto)
    phase_auto.add_argument("--max-rounds", type=int, default=50)
    phase_auto.add_argument("--min-delta", type=float, default=0.005)
    phase_auto.add_argument("--max-retries", type=int, default=2)

    phase_gate = sub.add_parser("phase-gate", help="Check a completed phase gate")
    add_common(phase_gate)
    phase_gate.add_argument(
        "--phase", type=int, required=True, choices=[1, 2, 3, 4],
    )

    phase_diag = sub.add_parser(
        "phase-diagnostics", help="Print phase diagnostics",
    )
    phase_diag.add_argument(
        "--phase", type=int, required=True, choices=[1, 2, 3, 4],
    )
    phase_diag.add_argument("--round", type=int, default=None)

    robustness = sub.add_parser(
        "validate-robustness", help="Post-optimization robustness check",
    )
    add_common(robustness)
    robustness.add_argument("--perturbation", type=float, default=0.10)

    chronology = sub.add_parser(
        "event-chronology",
        help="Write event-level channel chronology diagnostics",
    )
    add_common(chronology)
    chronology.add_argument("--out", default=None)

    economic_eval = sub.add_parser(
        "economic-evaluate",
        help="Evaluate crisis action policies using portfolio economics",
    )
    add_common(economic_eval)
    economic_eval.add_argument("--signals-path", default=None)
    economic_eval.add_argument("--out", default=None)

    economic_opt = sub.add_parser(
        "economic-optimize",
        help="Optimize crisis action policy using portfolio economics",
    )
    add_common(economic_opt)
    economic_opt.add_argument("--signals-path", default=None)
    economic_opt.add_argument("--out", default=None)
    economic_opt.add_argument("--min-delta", type=float, default=0.001)
    economic_opt.add_argument("--max-rounds", type=int, default=8)

    sleeve_opt = sub.add_parser(
        "sleeve-optimize",
        help="Optimize sleeve-aware crisis action policy using portfolio economics",
    )
    add_common(sleeve_opt)
    sleeve_opt.add_argument("--signals-path", default=None)
    sleeve_opt.add_argument("--out", default=None)

    args = parser.parse_args()
    if args.command == "phase-run":
        cmd_phase_run(args)
    elif args.command == "phase-auto":
        cmd_phase_auto(args)
    elif args.command == "phase-gate":
        cmd_phase_gate(args)
    elif args.command == "phase-diagnostics":
        cmd_phase_diagnostics(args)
    elif args.command == "validate-robustness":
        cmd_validate_robustness(args)
    elif args.command == "event-chronology":
        cmd_event_chronology(args)
    elif args.command == "economic-evaluate":
        cmd_economic_evaluate(args)
    elif args.command == "economic-optimize":
        cmd_economic_optimize(args)
    elif args.command == "sleeve-optimize":
        cmd_sleeve_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
