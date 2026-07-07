"""Run TPC round 8 train-only phased auto-optimisation."""
from __future__ import annotations

import argparse
from pathlib import Path

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.round_manager import RoundManager

from .round8_candidates import ROUND8_STRUCTURAL_SEED
from .round8_plugin import ROUND8_HARD_REJECTS, ROUND8_SCORING_WEIGHTS, Round8TPCPlugin


def main() -> None:
    parser = argparse.ArgumentParser(prog="tpc-round8-alpha")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--round", type=int, default=8)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default="2025-11-01")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--start-phase", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.004)
    parser.add_argument("--max-retries", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manager = RoundManager("swing", "tpc")
    plugin = Round8TPCPlugin(
        data_dir,
        initial_equity=args.equity,
        max_workers=args.max_workers,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if args.start_phase is not None:
        if args.round is None:
            raise ValueError("--start-phase requires --round so the resume target is explicit.")
        round_num = args.round
        round_dir = manager.round_path(round_num)
        if not round_dir.exists():
            raise FileNotFoundError(f"Cannot resume round {round_num}; directory does not exist: {round_dir}")
    else:
        round_num, round_dir = manager.resolve_round(args.round, for_write=True, expected_phases=plugin.num_phases)

    provenance = plugin.build_provenance()
    previous_mutations = manager.get_previous_mutations(
        round_num,
        current_provenance=provenance,
    )
    initial_mutations = dict(previous_mutations)
    initial_mutations.update(ROUND8_STRUCTURAL_SEED)
    plugin.initial_mutations = initial_mutations

    manager.write_run_spec(
        round_dir,
        round_num,
        plugin.name,
        description=(
            "Round 8 train-only optimisation: round-7 baseline plus additive 4h trend, "
            "completed 30m scaled pullback, 15m confirmation, true MA-transition filters, "
            "and EMA20 reclaim-geometry context research."
        ),
        scoring_weights=ROUND8_SCORING_WEIGHTS,
        baseline_mutations=initial_mutations,
        baseline_source=manager.optimized_config_path(manager.round_path(round_num - 1)),
        provenance=provenance,
        provenance_status="complete",
    )

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        round_name="round_8_tpc_30m_pullback_ma_transition_context",
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        max_retries=args.max_retries,
        round_manager=manager,
        round_num=round_num,
    )
    state = runner.run_all_phases(start_phase=args.start_phase)
    metrics = plugin.compute_final_metrics(state.cumulative_mutations)
    print(f"TPC round {round_num} complete at {round_dir}")
    print(
        "Final train metrics: "
        f"net={metrics.get('net_return_pct', 0):+.2f}%, "
        f"avgR={metrics.get('avg_r', 0):+.3f}, "
        f"trades={metrics.get('total_trades', 0):.0f}, "
        f"$PF={metrics.get('dollar_profit_factor', 0):.2f}, "
        f"DD={metrics.get('max_dd_pct', 0):.2f}%"
    )
    print(f"Score components ({len(ROUND8_SCORING_WEIGHTS)}): {', '.join(ROUND8_SCORING_WEIGHTS)}")
    print(f"Hard rejects: {len(ROUND8_HARD_REJECTS)} guardrails")


if __name__ == "__main__":
    main()
