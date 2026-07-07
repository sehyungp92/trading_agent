from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.momentum.auto.nqdtc.plugin import NQDTCPlugin
from backtests.shared.auto.phase_state import PhaseState, load_phase_state
from backtests.shared.auto.round_manager import RoundManager


ROUND_MANAGER = RoundManager("momentum", "nqdtc")


def render_final_diagnostics_text(
    state: PhaseState,
    *,
    data_dir: Path,
    equity: float,
    max_workers: int,
) -> tuple[str, dict[str, float]]:
    plugin = NQDTCPlugin(
        data_dir=data_dir,
        initial_equity=equity,
        max_workers=max_workers,
        num_phases=max(state.completed_phases, default=5),
    )
    try:
        final_metrics = dict(plugin.compute_final_metrics(state.cumulative_mutations))
        artifacts = plugin.build_end_of_round_artifacts(state)
        return artifacts.final_diagnostics_text, final_metrics
    finally:
        plugin.close_pool()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate NQDTC final diagnostics from saved phase state.")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--phase-state", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    round_num, round_dir = ROUND_MANAGER.resolve_round(args.round, for_write=False)
    phase_state_path = Path(args.phase_state) if args.phase_state else ROUND_MANAGER.phase_state_path(round_dir)
    output_path = Path(args.output) if args.output else ROUND_MANAGER.diagnostics_path(round_dir)

    state = load_phase_state(phase_state_path)
    diagnostics_text, final_metrics = render_final_diagnostics_text(
        state,
        data_dir=Path(args.data_dir),
        equity=float(args.equity),
        max_workers=max(1, int(args.max_workers)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(diagnostics_text, encoding="utf-8")

    if not args.output:
        provenance = NQDTCPlugin(
            data_dir=Path(args.data_dir),
            initial_equity=float(args.equity),
            max_workers=max(1, int(args.max_workers)),
            num_phases=max(state.completed_phases, default=5),
        ).build_provenance()
        ROUND_MANAGER.write_run_summary(
            round_dir,
            state.cumulative_mutations,
            final_metrics,
            state.completed_phases,
            round_num=round_num,
            source_diagnostics=output_path,
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

    print(f"Saved final diagnostics to {output_path}")


if __name__ == "__main__":
    main()
