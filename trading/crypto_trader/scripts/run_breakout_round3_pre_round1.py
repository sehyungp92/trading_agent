"""Run breakout round 3 from the reconstructed pre-round-1 baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_trader.cli import _update_rounds_manifest, _configure_logging
from crypto_trader.optimize.breakout_round3_pre_round1 import (
    BreakoutRound3PreRound1Plugin,
    IMMUTABLE_HARD_REJECTS,
    IMMUTABLE_SCORING_CEILINGS,
    IMMUTABLE_SCORING_WEIGHTS,
    SYMBOLS,
    build_backtest_config,
    build_pre_round1_config,
)
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    _configure_logging()
    log = structlog.get_logger("scripts.breakout_round3_pre_round1")

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / "breakout"
    round_dir = output_dir / "round_3"
    round_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_pre_round1_config()
    bt_cfg, window_meta = build_backtest_config(data_dir)

    plugin = BreakoutRound3PreRound1Plugin(
        backtest_config=bt_cfg,
        base_config=base_config,
        data_dir=data_dir,
        max_workers=2,
    )
    runner = PhaseRunner(plugin, round_dir)
    state = PhaseState(_path=round_dir / "phase_state.json")

    run_spec = {
        "description": "Breakout round 3 seeded from reconstructed pre-round-1 baseline",
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
        "baseline_strategy": base_config.to_dict(),
        "baseline_notes": [
            "Reverted documented baked round-1/round-2 settings to the earliest recoverable breakout baseline.",
            "Used implementation-plan risk/session defaults where current strategy defaults were later risk-sweep values.",
            "Dates use the maximum common BTC/ETH/SOL candle+funding window available in local data.",
            "Finetune preserves integers to avoid the historical float slice-index failure on fields like profile.lookback_bars.",
        ],
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    log.info(
        "breakout.round3_pre_round1.start",
        output_dir=str(round_dir),
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
        "breakout.round3_pre_round1.complete",
        output_dir=str(round_dir),
        final_phase=final_phase,
        final_mutations=len(final_mutations),
        total_trades=(final_metrics or {}).get("total_trades"),
        net_return_pct=(final_metrics or {}).get("net_return_pct"),
        sharpe_ratio=(final_metrics or {}).get("sharpe_ratio"),
    )


if __name__ == "__main__":
    main()
