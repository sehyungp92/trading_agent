"""Run portfolio round-2 phased auto optimization through PhaseRunner."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.optimize.portfolio_round2_phased import (
    DEV_END,
    DEV_START,
    FULL_END,
    FULL_START,
    HARD_REJECTS,
    HOLDOUT_END,
    HOLDOUT_START,
    MAX_SCORE_COMPONENTS,
    PHASE_OBJECTIVES,
    SCORING_WEIGHTS,
    SYMBOLS,
    PortfolioRound2PhasedPlugin,
)


ROUND_NAME = "portfolio_round2_phased_auto"
MAX_WORKERS = 2


def _configure_logging() -> None:
    logging.basicConfig(level=logging.ERROR)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> None:
    _configure_logging()

    data_dir = ROOT / "data"
    round_dir = ROOT / "output" / "portfolio" / "round_2"
    portfolio_config_path = ROOT / "config" / "portfolio_config.json"
    round_dir.mkdir(parents=True, exist_ok=True)

    plugin = PortfolioRound2PhasedPlugin(
        root=ROOT,
        data_dir=data_dir,
        portfolio_config_path=portfolio_config_path,
        max_workers=MAX_WORKERS,
    )
    runner = PhaseRunner(
        plugin,
        round_dir,
        round_name=ROUND_NAME,
        min_delta=0.003,
        max_retries=0,
        max_diagnostic_retries=0,
        contract=plugin.contract,
        validation_mode="dev",
    )
    state = PhaseState.load_or_create(round_dir / "phase_state.json")

    run_spec = {
        "description": (
            "Portfolio round 2 phased auto run using the latest optimized "
            "momentum, trend, and breakout configs as the starting baseline."
        ),
        "symbols": list(SYMBOLS),
        "max_workers": MAX_WORKERS,
        "framework": "crypto_trader.optimize.phase_runner.PhaseRunner",
        "plugin": "crypto_trader.optimize.portfolio_round2_phased.PortfolioRound2PhasedPlugin",
        "baseline_portfolio_config": str(portfolio_config_path),
        "development_window": {"start": str(DEV_START), "end": str(DEV_END)},
        "holdout_window": {
            "start": str(HOLDOUT_START),
            "end": str(HOLDOUT_END),
            "usage": "validation only; excluded from optimization score",
        },
        "full_window": {"start": str(FULL_START), "end": str(FULL_END)},
        "score_component_limit": MAX_SCORE_COMPONENTS,
        "immutable_score_weights": dict(SCORING_WEIGHTS),
        "hard_rejects": {
            metric: {"operator": op, "threshold": threshold}
            for metric, (op, threshold) in HARD_REJECTS.items()
        },
        "phase_objectives": {str(k): v for k, v in PHASE_OBJECTIVES.items()},
        "contract": plugin.contract,
        "notes": [
            "Strategy mutations are evaluated with real portfolio backtests on the development window.",
            "Portfolio structural mutations use deployable PortfolioConfig fields only.",
            "The post-2026-04-20 holdout is reported after each phase but never included in score.",
            "Scoring has seven components: return, frequency, edge quality, capture, drawdown resilience, rule efficiency, and strategy balance.",
        ],
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    print("Portfolio phase-auto round 2")
    print(f"Output: {round_dir}")
    print(f"Framework: PhaseRunner + PortfolioRound2PhasedPlugin")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Development window: {DEV_START} to {DEV_END}")
    print(f"Holdout window: {HOLDOUT_START} to {HOLDOUT_END} (validation only)")

    runner.run_all_phases(state)

    payload = plugin.save_recommended_artifacts(state.cumulative_mutations, round_dir)
    final_metrics = payload["metrics"]["full"]
    dev_metrics = payload["metrics"]["development"]
    holdout_metrics = payload["metrics"]["holdout"]

    print("Portfolio round 2 complete")
    print(
        "Development: "
        f"trades={dev_metrics['total_trades']:.0f}, "
        f"return={dev_metrics['net_return_pct']:.2f}%, "
        f"PF={dev_metrics['profit_factor']:.2f}, "
        f"DD={dev_metrics['max_drawdown_pct']:.2f}%"
    )
    print(
        "Holdout: "
        f"trades={holdout_metrics['total_trades']:.0f}, "
        f"return={holdout_metrics['net_return_pct']:.2f}%, "
        f"PF={holdout_metrics['profit_factor']:.2f}, "
        f"DD={holdout_metrics['max_drawdown_pct']:.2f}%"
    )
    print(
        "Full: "
        f"trades={final_metrics['total_trades']:.0f}, "
        f"return={final_metrics['net_return_pct']:.2f}%, "
        f"PF={final_metrics['profit_factor']:.2f}, "
        f"DD={final_metrics['max_drawdown_pct']:.2f}%"
    )
    print(f"Immutable score: {payload['immutable_score']:.4f}")


if __name__ == "__main__":
    main()
