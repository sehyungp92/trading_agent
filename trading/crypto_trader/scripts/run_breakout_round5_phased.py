"""Run canonical breakout round 2 from the relabelled round-1 baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_trader.cli import _configure_logging, _update_rounds_manifest
from crypto_trader.optimize.breakout_round3_pre_round1 import SYMBOLS, build_backtest_config
from crypto_trader.optimize.breakout_round4_trade_frequency import run_greedy_without_pruning
from crypto_trader.optimize.breakout_round5_phased import (
    IMMUTABLE_HARD_REJECTS,
    IMMUTABLE_SCORING_CEILINGS,
    IMMUTABLE_SCORING_WEIGHTS,
    PHASE_GATE_CRITERIA,
    BreakoutRound5PhasedPlugin,
    load_breakout_strategy,
)
import crypto_trader.optimize.phase_runner as phase_runner_module
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    _configure_logging()
    log = structlog.get_logger("scripts.breakout_round2_phased")

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / "breakout"
    round_dir = output_dir / "round_2"
    round_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "round_1" / "optimized_config.json"
    base_config = load_breakout_strategy(baseline_path)
    bt_cfg, window_meta = build_backtest_config(data_dir)

    plugin = BreakoutRound5PhasedPlugin(
        backtest_config=bt_cfg,
        base_config=base_config,
        data_dir=data_dir,
        max_workers=2,
    )
    phase_runner_module.run_greedy = run_greedy_without_pruning
    runner = PhaseRunner(plugin, round_dir)
    state = PhaseState.load_or_create(round_dir / "phase_state.json")

    run_spec = {
        "description": (
            "Breakout round 2 phased auto run seeded from breakout round_1 and "
            "optimized for stronger signal discrimination, higher capture "
            "efficiency, and controlled trade-frequency expansion."
        ),
        "symbols": list(SYMBOLS),
        "max_workers": 2,
        "contract_hash": runner.contract.get("contract_hash", ""),
        "profile_hash": runner.contract.get("profile_hash", ""),
        "contract": runner.contract,
        "window": window_meta,
        "immutable_scoring_weights": IMMUTABLE_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": IMMUTABLE_SCORING_CEILINGS,
        "hard_rejects": {
            key: {"operator": op, "threshold": threshold}
            for key, (op, threshold) in IMMUTABLE_HARD_REJECTS.items()
        },
        "phase_gate_criteria": {
            str(phase): [
                {
                    "metric": criterion.metric,
                    "operator": criterion.operator,
                    "threshold": criterion.threshold,
                }
                for criterion in criteria
            ]
            for phase, criteria in PHASE_GATE_CRITERIA.items()
        },
        "baseline_strategy": base_config.to_dict(),
        "baseline_source": str(baseline_path),
        "phase_objectives": {
            "1": "Improve signal discrimination and remove persistently weak symbol/direction segments.",
            "2": "Tighten entry quality, especially retest logic, without collapsing trade count.",
            "3": "Increase realized capture and reduce giveback through faster management.",
            "4": "Expand trade count only through structural, testable changes.",
            "5": "Scale returns through sizing only after alpha quality is improved.",
            "6": "Conservative finetune around accepted mutations.",
        },
        "score_notes": [
            "Coverage is co-primary with returns, but leverage cannot dominate because edge, capture, and Sharpe remain material.",
            "Capture efficiency is explicitly scored to reduce the current winner giveback problem.",
            "Risk stays a guardrail; phase gates tighten as the run progresses.",
            "The run starts from the round-1 optimized config and uses the maximum common BTC/ETH/SOL window for comparability.",
        ],
        "assumptions": [
            "Retained BTC, ETH, and SOL only.",
            "Used max_workers=2 for reproducibility and resource control.",
            "Resumes from an existing round_2 phase_state.json if present.",
        ],
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    log.info(
        "breakout.round2_phased.start",
        output_dir=str(round_dir),
        baseline=str(baseline_path),
        start_date=window_meta["start_date"],
        end_date=window_meta["end_date"],
        workers=2,
    )
    runner.run_all_phases(state)

    final_phase = max(state.phase_metrics) if state.phase_metrics else None
    final_metrics = state.phase_metrics.get(final_phase) if final_phase else None
    final_mutations = (
        state.phase_results.get(final_phase, {}).get("final_mutations", {})
        if final_phase
        else {}
    )
    if final_mutations and state.cumulative_mutations != final_mutations:
        state.cumulative_mutations = dict(final_mutations)
        state.save(round_dir / "phase_state.json")
        runner._save_optimized_config(state)

    _update_rounds_manifest(output_dir, 2, final_mutations, final_metrics)

    log.info(
        "breakout.round2_phased.complete",
        output_dir=str(round_dir),
        final_phase=final_phase,
        final_mutations=len(final_mutations),
        total_trades=(final_metrics or {}).get("total_trades"),
        net_return_pct=(final_metrics or {}).get("net_return_pct"),
        sharpe_ratio=(final_metrics or {}).get("sharpe_ratio"),
        exit_efficiency=(final_metrics or {}).get("exit_efficiency"),
    )


if __name__ == "__main__":
    main()
