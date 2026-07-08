"""Run canonical breakout round 3 from the relabelled round-2 baseline."""

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
from crypto_trader.optimize.breakout_round6_phased import (
    ROUND6_HARD_REJECTS,
    ROUND6_IMMUTABLE_SCORING_CEILINGS,
    ROUND6_IMMUTABLE_SCORING_WEIGHTS,
    ROUND6_PHASE_GATE_CRITERIA,
    BreakoutRound6PhasedPlugin,
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
    log = structlog.get_logger("scripts.breakout_round3_phased")

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / "breakout"
    round_dir = output_dir / "round_3"
    round_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "round_2" / "optimized_config.json"
    base_config = load_breakout_strategy(baseline_path)
    bt_cfg, window_meta = build_backtest_config(data_dir)

    plugin = BreakoutRound6PhasedPlugin(
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
            "Breakout round 3 phased auto run seeded from breakout round_2 and "
            "optimized for higher monetization of existing alpha, a selective "
            "supplemental relaxed-body entry branch, and targeted structural "
            "frequency expansion without broad quality-destroying loosening."
        ),
        "symbols": list(SYMBOLS),
        "max_workers": 2,
        "contract_hash": runner.contract.get("contract_hash", ""),
        "profile_hash": runner.contract.get("profile_hash", ""),
        "contract": runner.contract,
        "window": window_meta,
        "immutable_scoring_weights": ROUND6_IMMUTABLE_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": ROUND6_IMMUTABLE_SCORING_CEILINGS,
        "hard_rejects": {
            key: {"operator": op, "threshold": threshold}
            for key, (op, threshold) in ROUND6_HARD_REJECTS.items()
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
            for phase, criteria in ROUND6_PHASE_GATE_CRITERIA.items()
        },
        "baseline_strategy": base_config.to_dict(),
        "baseline_source": str(baseline_path),
        "phase_objectives": {
            "1": "Harvest more realized return from the current winners before altering signal extraction.",
            "2": "Add a lower-risk supplemental entry branch only in the strongest symbol-direction pockets.",
            "3": "Expand trade count through the validated structural zone-age improvement instead of global loosening.",
            "4": "Pressure-test stricter retest and model2 controls so added trades remain real alpha rather than noise.",
            "5": "Narrowly finetune accepted non-risk parameters while avoiding pure leverage optimization.",
        },
        "score_notes": [
            "Coverage and returns are co-primary because the goal is to improve both expected return and trading frequency.",
            "Profit factor and Sharpe remain material, but their ceilings are widened so a few concentrated winners do not saturate the score.",
            "Capture is retained as a lighter guardrail because exit_efficiency improved less reliably than headline PnL in the validation sweeps.",
            "Finetuning excludes risk-tier and branch risk-scale mutations to keep the round focused on real alpha extraction rather than leverage.",
            "The run starts from the round-2 optimized config and uses the maximum common BTC/ETH/SOL window for comparability.",
        ],
        "assumptions": [
            "Retained BTC, ETH, and SOL only.",
            "Used max_workers=2 for reproducibility and resource control.",
            "Resumes from an existing round_3 phase_state.json if present.",
        ],
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    log.info(
        "breakout.round3_phased.start",
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
        "breakout.round3_phased.complete",
        output_dir=str(round_dir),
        final_phase=final_phase,
        final_mutations=len(final_mutations),
        total_trades=(final_metrics or {}).get("total_trades"),
        net_return_pct=(final_metrics or {}).get("net_return_pct"),
        profit_factor=(final_metrics or {}).get("profit_factor"),
        sharpe_ratio=(final_metrics or {}).get("sharpe_ratio"),
    )


if __name__ == "__main__":
    main()
