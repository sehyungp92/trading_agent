"""Run momentum round 3 focused on profit-lock and exit architecture."""

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
from crypto_trader.optimize.breakout_round4_trade_frequency import run_greedy_without_pruning
from crypto_trader.optimize.momentum_round3_exit_phased import (
    IMMUTABLE_HARD_REJECTS,
    IMMUTABLE_SCORING_CEILINGS,
    IMMUTABLE_SCORING_WEIGHTS,
    PHASE_GATE_CRITERIA,
    MomentumRound3ExitPhasedPlugin,
    load_momentum_strategy,
)
from crypto_trader.optimize.momentum_round4_union import SYMBOLS, build_backtest_config
import crypto_trader.optimize.phase_runner as phase_runner_module
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    _configure_logging()
    log = structlog.get_logger("scripts.momentum_round3_exit_phased")

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / "momentum"
    round_dir = output_dir / "round_3"
    round_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "round_2" / "optimized_config.json"
    base_config = load_momentum_strategy(baseline_path)
    bt_cfg, window_meta = build_backtest_config(data_dir)

    plugin = MomentumRound3ExitPhasedPlugin(
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
            "Momentum round 3 phased auto run seeded from momentum round_2 and "
            "focused specifically on profit-lock and exit architecture. The "
            "search emphasizes proof-lock, failure-to-follow-through control, "
            "state-specific management, peak-MFE retrace exits, runner-only "
            "MFE-aware trailing regimes, small-scale partial exits, and "
            "calibration of the existing reversal/structure exits."
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
            "1": "Reduce full-loss reversals after early proof using proof-lock, family-scoped failure-to-follow-through logic, and coarse retrace locks.",
            "2": "Find a runner-specific, peak-aware trailing regime that captures more MFE without collapsing the >5-bar winner profile.",
            "3": "Test small, generalizable partial-exit and BE structures that monetize proof without chopping runners.",
            "4": "Calibrate the reversal and structure exits that already contribute disproportionate winner capture.",
            "5": "Conservatively finetune accepted exit and trail numerics only.",
        },
        "score_notes": [
            "Capture is the dominant score dimension for this round because round 2 still monetized less than half of available MFE.",
            "The hard rejects preserve trade count, entry quality, and drawdown so exit changes cannot win by simply suppressing exposure.",
            "Signal, entry, session, and risk surfaces are intentionally frozen so this round isolates repeatable monetization improvements.",
            "The run starts from the round-2 optimized config and uses the maximum common BTC/ETH/SOL window for comparability.",
        ],
        "assumptions": [
            "Retained BTC, ETH, and SOL only.",
            "Used max_workers=2 for reproducibility and resource control.",
            "Applied structural exit features before the run: proof-lock, family-scoped failure-to-follow-through, peak-MFE retrace exits, runner-only trail overrides, configurable reversal/structure thresholds, and MFE-aware trailing.",
            "Resumes from an existing round_3 phase_state.json if present.",
        ],
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    log.info(
        "momentum.round3_exit_phased.start",
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

    _update_rounds_manifest(output_dir, 3, final_mutations, final_metrics)

    log.info(
        "momentum.round3_exit_phased.complete",
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
