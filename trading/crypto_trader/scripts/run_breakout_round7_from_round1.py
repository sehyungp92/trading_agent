"""Run the next breakout phased auto round from the live round_1 baseline."""

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
from crypto_trader.optimize.breakout_round7_from_round1 import (
    PHASE_NAMES,
    ROUND7_HARD_REJECTS,
    ROUND7_IMMUTABLE_SCORING_CEILINGS,
    ROUND7_IMMUTABLE_SCORING_WEIGHTS,
    ROUND7_PHASE_GATE_CRITERIA,
    BreakoutRound7FromRound1Plugin,
    build_backtest_config,
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
    log = structlog.get_logger("scripts.breakout_round7_from_round1")

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / "breakout"
    round_dir = output_dir / "round_2"
    round_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "round_1" / "optimized_config.json"
    base_config = load_breakout_strategy(baseline_path)
    bt_cfg, window_meta = build_backtest_config(data_dir)

    plugin = BreakoutRound7FromRound1Plugin(
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
            "Breakout round 2 phased auto run seeded from the live breakout round_1 "
            "optimized config and focused on re-earning the revalidated alpha branch, "
            "tightening signal discrimination, improving retest entry quality, and "
            "capturing more of each move before attempting structural frequency gains."
        ),
        "symbols": list(bt_cfg.symbols),
        "max_workers": 2,
        "contract_hash": runner.contract.get("contract_hash", ""),
        "profile_hash": runner.contract.get("profile_hash", ""),
        "contract": runner.contract,
        "window": window_meta,
        "immutable_scoring_weights": ROUND7_IMMUTABLE_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": ROUND7_IMMUTABLE_SCORING_CEILINGS,
        "hard_rejects": {
            key: {"operator": op, "threshold": threshold}
            for key, (op, threshold) in ROUND7_HARD_REJECTS.items()
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
            for phase, criteria in ROUND7_PHASE_GATE_CRITERIA.items()
        },
        "baseline_strategy": base_config.to_dict(),
        "baseline_source": str(baseline_path),
        "phase_objectives": {
            "1": "Re-earn the strongest revalidated alpha branch that was stripped out of the cleaned seed.",
            "2": "Increase discrimination so low-quality body structures and weak symbol pockets are rejected earlier.",
            "3": "Improve retest entry quality so extra trades come from cleaner continuation behavior.",
            "4": "Reduce giveback and monetize more of the move with faster management and early profit locks.",
            "5": "Seek additional trade count only through structural changes that are still plausibly real alpha.",
            "6": "Narrowly finetune accepted non-risk parameters without turning the round into leverage optimisation.",
        },
        "score_notes": [
            "Coverage and returns are co-primary because the goal is to improve both expected return and trading frequency.",
            "Capture and entry quality are explicitly scored so the round cannot win by adding noisy trades with poor monetization.",
            "Risk and Calmar remain guardrails, but risk-tier mutations are excluded from the search so score gains reflect strategy edge.",
            "This run must be launched from a real script file on Windows so 2-worker multiprocessing can spawn correctly.",
        ],
        "assumptions": [
            "Retained BTC, ETH, and SOL only.",
            "Used the maximum common BTC/ETH/SOL period detected from candles and funding data.",
            "Kept max_workers=2 as requested.",
            "Resumes from an existing round_2 phase_state.json if present.",
        ],
        "phase_names": PHASE_NAMES,
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    log.info(
        "breakout.round2_post_round1.start",
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
        "breakout.round2_post_round1.complete",
        output_dir=str(round_dir),
        final_phase=final_phase,
        final_mutations=len(final_mutations),
        total_trades=(final_metrics or {}).get("total_trades"),
        net_return_pct=(final_metrics or {}).get("net_return_pct"),
        profit_factor=(final_metrics or {}).get("profit_factor"),
        sharpe_ratio=(final_metrics or {}).get("sharpe_ratio"),
        exit_efficiency=(final_metrics or {}).get("exit_efficiency"),
    )


if __name__ == "__main__":
    main()
