"""Run the next breakout phased auto round from the live round_2 baseline."""

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
from crypto_trader.optimize.breakout_round8_from_round2 import (
    PHASE_NAMES,
    ROUND8_HARD_REJECTS,
    ROUND8_IMMUTABLE_SCORING_CEILINGS,
    ROUND8_IMMUTABLE_SCORING_WEIGHTS,
    ROUND8_PHASE_GATE_CRITERIA,
    BreakoutRound8FromRound2Plugin,
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
    log = structlog.get_logger("scripts.breakout_round8_from_round2")

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / "breakout"
    round_dir = output_dir / "round_3"
    round_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "round_2" / "optimized_config.json"
    base_config = load_breakout_strategy(baseline_path)
    bt_cfg, window_meta = build_backtest_config(data_dir)

    plugin = BreakoutRound8FromRound2Plugin(
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
            "Breakout round 3 phased auto run seeded from the live breakout "
            "round_2 optimized config and focused on broad-only alpha "
            "extraction: relaxed-body branch auditing, stronger signal "
            "discrimination, cleaner retest entries, better failed-breakout "
            "handling, and structural trade-frequency gains that do not rely "
            "on symbol-specific pruning."
        ),
        "symbols": list(bt_cfg.symbols),
        "max_workers": 2,
        "contract_hash": runner.contract.get("contract_hash", ""),
        "profile_hash": runner.contract.get("profile_hash", ""),
        "contract": runner.contract,
        "window": window_meta,
        "immutable_scoring_weights": ROUND8_IMMUTABLE_SCORING_WEIGHTS,
        "immutable_scoring_ceilings": ROUND8_IMMUTABLE_SCORING_CEILINGS,
        "hard_rejects": {
            key: {"operator": op, "threshold": threshold}
            for key, (op, threshold) in ROUND8_HARD_REJECTS.items()
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
            for phase, criteria in ROUND8_PHASE_GATE_CRITERIA.items()
        },
        "baseline_strategy": base_config.to_dict(),
        "baseline_source": str(baseline_path),
        "excluded_search_space": [
            "All symbol-direction filters",
            "All relaxed-body direction filters",
            "Risk sizing and leverage mutations",
            "Session, funding, and OI filters",
            "Hour-of-day or day-of-week filters",
        ],
        "phase_objectives": {
            "1": "Verify whether the relaxed-body branch is broad alpha, broad noise, or only useful with stricter universal thresholds.",
            "2": "Reject weak low-conviction breakouts with stronger universal body, volume, and minimum-quality filters.",
            "3": "Tighten context so broad-side trades require more real trend support instead of permissive countertrend participation.",
            "4": "Improve retest execution so added trades come from cleaner continuation behavior and tighter confirmation.",
            "5": "Reduce favorable-excursion reversals through universal failed-breakout handling and early monetization.",
            "6": "Add trades only through broader structure formation or guarded reentry rather than looser raw signal standards.",
            "7": "Finetune only accepted non-risk parameters with narrow 5 percent perturbations.",
        },
        "score_notes": [
            "Coverage and returns remain the primary objectives because the user wants both higher expected returns and more trading opportunities.",
            "Capture and entry quality are explicitly scored so the round cannot win by adding noisy trades with poor monetization.",
            "Hard rejects and phase gates enforce minimum expectancy, exit efficiency, and drawdown discipline to reduce small-sample overfit.",
            "No asset-specific pruning is allowed in this round; broad durability is favored over selective cherry-picking.",
            "This run is launched from a real script file on Windows so 2-worker multiprocessing can spawn correctly.",
        ],
        "assumptions": [
            "Used output/breakout/round_2/optimized_config.json as the starting baseline.",
            "Kept BTC, ETH, and SOL only, with both directions available throughout the search.",
            "Used the maximum common BTC/ETH/SOL period detected from candles and funding data.",
            "Kept max_workers=2 as requested.",
            "Resumes from an existing output/breakout/round_3/phase_state.json if present.",
        ],
        "phase_names": PHASE_NAMES,
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    log.info(
        "breakout.round3_from_round2.start",
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
        "breakout.round3_from_round2.complete",
        output_dir=str(round_dir),
        final_phase=final_phase,
        final_mutations=len(final_mutations),
        total_trades=(final_metrics or {}).get("total_trades"),
        net_return_pct=(final_metrics or {}).get("net_return_pct"),
        profit_factor=(final_metrics or {}).get("profit_factor"),
        expectancy_r=(final_metrics or {}).get("expectancy_r"),
        sharpe_ratio=(final_metrics or {}).get("sharpe_ratio"),
        exit_efficiency=(final_metrics or {}).get("exit_efficiency"),
    )


if __name__ == "__main__":
    main()
