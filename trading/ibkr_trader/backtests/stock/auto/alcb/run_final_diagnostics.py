from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.shared.auto.phase_state import load_phase_state
from backtests.shared.auto.round_manager import RoundManager
from backtests.stock.auto.alcb.plugin import ALCBP16Plugin


ROUND_MANAGER = RoundManager("stock", "alcb")

FLOOR_NAME_MAP = {
    "expectancy_dollar": "min_expectancy_dollar",
    "expected_total_r": "min_expected_total_r",
    "inv_dd": "min_inv_dd",
    "max_drawdown_pct": "max_dd_pct",
    "net_profit": "min_net_profit",
    "profit_factor": "min_pf",
    "trades_per_month": "min_trades_per_month",
}


def _hydrate_final_phase_runtime_context(plugin: ALCBP16Plugin, state) -> int:
    final_phase = max(state.completed_phases) if state.completed_phases else plugin.num_phases
    # Warm replay data first because the initial source-fingerprint load clears
    # transient runtime context caches, including phase diagnostics metadata.
    plugin._replay_bundle()
    phase_result = state.phase_results.get(final_phase, {})
    phase_gate = state.phase_gate_results.get(final_phase, {})
    plugin._phase_runtime_context[final_phase] = {
        "base_metrics": dict(phase_result.get("final_metrics", {})),
        "hard_rejects": {
            FLOOR_NAME_MAP.get(criterion["name"], criterion["name"]): criterion["target"]
            for criterion in phase_gate.get("criteria", [])
            if isinstance(criterion, dict) and "name" in criterion and "target" in criterion
        },
    }
    return final_phase


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate the ALCB final diagnostics from saved phase state.")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--phase-state", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--data-dir", default="backtests/stock/data/raw")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    round_num, round_dir = ROUND_MANAGER.resolve_round(args.round, for_write=False)
    phase_state_path = Path(args.phase_state) if args.phase_state else ROUND_MANAGER.phase_state_path(round_dir)
    round_output_path = ROUND_MANAGER.diagnostics_path(round_dir)
    output_path = Path(args.output) if args.output else round_output_path
    round_output_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_phase_state(phase_state_path)
    plugin = ALCBP16Plugin(
        Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        initial_equity=float(args.equity),
        max_workers=max(1, int(args.max_workers)),
    )
    provenance = plugin.build_provenance()
    _hydrate_final_phase_runtime_context(plugin, state)
    diagnostics_text = plugin.render_final_diagnostics_text(state)
    round_output_path.write_text(diagnostics_text, encoding="utf-8")
    if output_path != round_output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(diagnostics_text, encoding="utf-8")

    final_phase = max(state.completed_phases) if state.completed_phases else plugin.num_phases
    final_metrics = dict(state.phase_results.get(final_phase, {}).get("final_metrics", {}))
    ROUND_MANAGER.write_run_spec(
        round_dir,
        round_num,
        "alcb",
        description="Final diagnostics replay",
        baseline_mutations=(
            ROUND_MANAGER.get_previous_mutations(round_num, current_provenance=provenance)
            if round_num > 1 else {}
        ),
        provenance=provenance,
        provenance_status="complete",
    )
    ROUND_MANAGER.write_run_summary(
        round_dir,
        state.cumulative_mutations,
        final_metrics,
        state.completed_phases,
        round_num=round_num,
        source_diagnostics=round_output_path,
        source_phase_state=phase_state_path,
        provenance=provenance,
        provenance_status="complete",
    )
    ROUND_MANAGER.write_optimized_config(round_dir, state.cumulative_mutations)
    ROUND_MANAGER.append_to_manifest(
        round_num,
        state.cumulative_mutations,
        final_metrics,
        provenance=provenance,
        provenance_status="complete",
    )
    print(f"Saved final diagnostics to {round_output_path}")


if __name__ == "__main__":
    main()
