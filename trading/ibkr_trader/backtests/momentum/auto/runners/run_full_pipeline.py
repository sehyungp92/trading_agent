"""Full momentum auto pipeline: hierarchical optimization.

Usage:
    cd trading
    python -u backtests/momentum/auto/run_full_pipeline.py
    python -u backtests/momentum/auto/run_full_pipeline.py --phase strategy-greedy
    python -u backtests/momentum/auto/run_full_pipeline.py --phase portfolio-greedy

Five phases (hierarchical — strategies first, then portfolio):
  Phase 1: Run all ~420 experiments (strategy-level + portfolio-level)
  Phase 2: Per-strategy greedy (optimize NQDTC/Vdubus independently, 1 engine each)
  Phase 3: Portfolio greedy (optimize portfolio rules on top of optimized strategies)
  Phase 4: Diagnostics + comparison vs v6 baseline
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Project root
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# Install momentum aliases before any backtest imports
from backtests.momentum.auto.harness import MomentumAutoHarness
from backtests.momentum.auto.experiments import build_experiment_queue
from backtests.momentum.auto.greedy_optimize import (
    PORTFOLIO_CANDIDATES,
    convert_experiment_to_portfolio_mutation,
    run_greedy,
    run_strategy_greedy,
    save_result,
)
from backtests.momentum.auto.scoring import composite_score, extract_metrics

_DEFAULT_EQUITY = 10_000.0
_DEFAULT_DATA_DIR = ROOT / "backtests" / "momentum" / "data" / "raw"
OUTPUT_DIR = ROOT / "backtests" / "momentum" / "auto" / "output"

# Module-level state set by main() for use by phase functions
EQUITY = _DEFAULT_EQUITY
DATA_DIR = _DEFAULT_DATA_DIR

# Valid phase names
_PHASES = (
    "experiments",       # Phase 1 only
    "strategy-greedy",   # Phase 2 only
    "portfolio-greedy",  # Phase 3 only
    "greedy",            # Phases 2+3
    "diagnostics",       # Phase 4 only
    "full",              # All phases
)


def main(
    phase: str = "full",
    strategy_filter: str = "all",
    resume: bool = True,
    max_workers: int | None = None,
    equity: float | None = None,
    data_dir: str | Path | None = None,
    experiment_ids: list[str] | None = None,
    skip_robustness: bool = True,
):
    """Run the full pipeline or a specific phase.

    Args:
        phase: One of: experiments, strategy-greedy, portfolio-greedy,
               greedy (both), diagnostics, full (all)
        strategy_filter: "nqdtc", "vdubus", "portfolio", or "all"
        resume: Skip completed experiments on restart
        max_workers: Parallel worker count
        equity: Initial equity (default: 10_000)
        data_dir: Path to raw data directory
        experiment_ids: Specific experiment IDs to run (None = all)
        skip_robustness: Skip robustness checks for faster scan
    """
    global EQUITY, DATA_DIR
    EQUITY = equity or _DEFAULT_EQUITY
    DATA_DIR = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Phase 1: Run all experiments
    if phase in ("experiments", "full"):
        _phase_1_experiments(strategy_filter, resume, experiment_ids,
                            skip_robustness, max_workers)

    # Phase 2: Per-strategy greedy
    if phase in ("strategy-greedy", "greedy", "full"):
        _phase_2_strategy_greedy(max_workers)

    # Phase 3: Portfolio greedy
    if phase in ("portfolio-greedy", "greedy", "full"):
        _phase_3_portfolio_greedy(max_workers)

    # Phase 4: Diagnostics
    if phase in ("diagnostics", "full"):
        _phase_4_diagnostics()

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE in {total_time:.0f}s ({total_time / 60:.1f}m)")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Phase 1: Experiments
# ---------------------------------------------------------------------------

def _phase_1_experiments(
    strategy_filter: str,
    resume: bool,
    experiment_ids: list[str] | None = None,
    skip_robustness: bool = True,
    max_workers: int | None = None,
) -> None:
    print("=" * 70)
    print("PHASE 1: Running all experiments")
    print("=" * 70)

    n_workers = max_workers or 1
    t0 = time.time()
    harness = MomentumAutoHarness(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        initial_equity=EQUITY,
        max_workers=n_workers,
    )
    harness.run_all(
        strategy_filter=strategy_filter,
        experiment_ids=experiment_ids,
        skip_robustness=skip_robustness,
        resume=resume,
    )
    phase1_time = time.time() - t0
    print(f"\nPhase 1 complete in {phase1_time:.0f}s ({phase1_time / 60:.1f}m)")


# ---------------------------------------------------------------------------
# Phase 2: Per-strategy greedy (1 engine per eval = fast)
# ---------------------------------------------------------------------------

def _phase_2_strategy_greedy(max_workers: int | None) -> None:
    print(f"\n{'=' * 70}")
    print("PHASE 2: Per-strategy greedy optimization")
    print("=" * 70)

    from backtests.momentum.auto.results_tracker import ResultsTracker

    tracker = ResultsTracker(OUTPUT_DIR)
    results = tracker.load_all()
    experiments_map = {e.id: e for e in build_experiment_queue("all")}

    # Group positive experiments by strategy (exclude portfolio)
    strategy_candidates: dict[str, list[tuple[str, dict]]] = {
        "nqdtc": [], "vdubus": [],
    }
    seen: dict[str, set] = {s: set() for s in strategy_candidates}

    for r in results:
        if r.delta_pct <= 0 or r.status in ("CRASH", "UNWIRED"):
            continue
        if r.strategy not in strategy_candidates:
            continue
        exp = experiments_map.get(r.experiment_id)
        if not exp or not exp.mutations:
            continue
        # Deduplicate
        muts_key = json.dumps(exp.mutations, sort_keys=True, default=str)
        if muts_key in seen[r.strategy]:
            continue
        seen[r.strategy].add(muts_key)
        strategy_candidates[r.strategy].append((r.experiment_id, exp.mutations))

    for s, cands in strategy_candidates.items():
        print(f"  {s}: {len(cands)} positive-delta candidates")

    # Run greedy for each strategy
    strategy_optimal: dict[str, dict] = {}
    n_workers_val = max_workers or 3

    for strategy in ("nqdtc", "vdubus"):
        cands = strategy_candidates[strategy]
        if not cands:
            print(f"\n  [{strategy.upper()}] No positive candidates, skipping")
            strategy_optimal[strategy] = {}
            continue

        result = run_strategy_greedy(
            strategy=strategy,
            candidates=cands,
            initial_equity=EQUITY,
            data_dir=DATA_DIR,
            max_workers=n_workers_val,
            verbose=True,
        )

        strategy_optimal[strategy] = result.accepted_mutations
        save_result(result, OUTPUT_DIR / f"greedy_strategy_{strategy}.json")

    # Save combined strategy-optimal mutations
    combined_path = OUTPUT_DIR / "strategy_optimal_mutations.json"
    combined_path.write_text(json.dumps(strategy_optimal, indent=2, default=str))
    print(f"\n  Strategy-optimal mutations saved to: {combined_path}")

    # Show summary
    total_accepted = sum(len(m) for m in strategy_optimal.values())
    print(f"  Total strategy-level mutations accepted: {total_accepted}")


# ---------------------------------------------------------------------------
# Phase 3: Portfolio greedy (builds on optimized strategies)
# ---------------------------------------------------------------------------

def _phase_3_portfolio_greedy(max_workers: int | None) -> None:
    print(f"\n{'=' * 70}")
    print("PHASE 3: Portfolio-level greedy optimization")
    print("=" * 70)

    # Load strategy-optimal mutations from Phase 2
    strat_path = OUTPUT_DIR / "strategy_optimal_mutations.json"
    if strat_path.exists():
        strategy_optimal = json.loads(strat_path.read_text())
        print(f"  Loaded strategy-optimal mutations from Phase 2")
    else:
        strategy_optimal = {}
        print("  No strategy-optimal mutations found, using defaults")

    # Convert strategy-level mutations to portfolio-level format
    base_mutations: dict = {}
    for strategy, muts in strategy_optimal.items():
        if muts:
            portfolio_muts = convert_experiment_to_portfolio_mutation(strategy, muts)
            base_mutations.update(portfolio_muts)

    if base_mutations:
        print(f"  Base mutations from strategy greedy: {len(base_mutations)} keys")
        for key, val in sorted(base_mutations.items()):
            print(f"    {key}: {val}")

    # Build portfolio candidates: default PORTFOLIO_CANDIDATES + positive portfolio experiments
    from backtests.momentum.auto.results_tracker import ResultsTracker

    tracker = ResultsTracker(OUTPUT_DIR)
    results = tracker.load_all()
    experiments_map = {e.id: e for e in build_experiment_queue("all")}

    # Collect positive portfolio experiments
    portfolio_positives = []
    seen_mutations = set()
    for r in results:
        if r.strategy != "portfolio":
            continue
        if r.delta_pct <= 0 or r.status in ("CRASH", "UNWIRED"):
            continue
        exp = experiments_map.get(r.experiment_id)
        if not exp or not exp.mutations:
            continue
        muts_key = json.dumps(exp.mutations, sort_keys=True, default=str)
        if muts_key in seen_mutations:
            continue
        seen_mutations.add(muts_key)
        portfolio_positives.append((r.experiment_id, exp.mutations))

    print(f"  Positive portfolio experiments: {len(portfolio_positives)}")

    # Combine: PORTFOLIO_CANDIDATES + positive portfolio experiments
    all_candidates = list(PORTFOLIO_CANDIDATES) + portfolio_positives
    print(f"  Total portfolio candidates: {len(all_candidates)}")

    # Save candidates for inspection
    candidates_path = OUTPUT_DIR / "greedy_portfolio_candidates.json"
    candidates_path.write_text(json.dumps(
        [{"name": n, "mutations": m} for n, m in all_candidates],
        indent=2, default=str,
    ))

    # Run portfolio greedy with strategy-optimal base
    result = run_greedy(
        candidates=all_candidates,
        initial_equity=EQUITY,
        data_dir=DATA_DIR,
        max_workers=max_workers or 3,
        base_mutations=base_mutations if base_mutations else None,
        verbose=True,
    )

    save_result(result, OUTPUT_DIR / "greedy_portfolio_optimal.json")


# ---------------------------------------------------------------------------
# Phase 4: Diagnostics & comparison
# ---------------------------------------------------------------------------

def _phase_4_diagnostics() -> None:
    print(f"\n{'=' * 70}")
    print("PHASE 4: Diagnostics & v6 baseline comparison")
    print("=" * 70)

    # Load greedy result
    greedy_path = OUTPUT_DIR / "greedy_portfolio_optimal.json"
    if not greedy_path.exists():
        print("  No greedy result found, skipping diagnostics")
        return

    greedy_data = json.loads(greedy_path.read_text())
    optimal_mutations = greedy_data.get("accepted_mutations", {})

    print(f"  Optimal mutations: {len(optimal_mutations)} keys")
    print(f"  Greedy score: {greedy_data.get('final_score', 0):.6f}")
    print(f"  Baseline score: {greedy_data.get('baseline_score', 0):.6f}")
    improvement = greedy_data.get("improvement_pct", 0)
    print(f"  Improvement: {improvement:+.2f}%")

    # Run full diagnostics on optimal config
    try:
        _run_optimal_diagnostics(optimal_mutations)
    except Exception:
        print(f"  Diagnostics failed:\n{traceback.format_exc()}")

    # Save experiment summary
    _save_experiment_summary()


def _run_optimal_diagnostics(mutations: dict) -> None:
    """Run the optimal portfolio config and generate full diagnostics."""
    from backtests.momentum.auto.config_mutator import (
        extract_passthrough_mutations,
        mutate_nqdtc_config,
        mutate_portfolio_config,
        mutate_vdubus_config,
    )
    from backtests.momentum.cli import (
        _load_nqdtc_data,
        _load_vdubus_data,
    )
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.config_portfolio import PortfolioBacktestConfig
    from backtests.momentum.config_vdubus import VdubusBacktestConfig
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
    from backtests.momentum.engine.portfolio_engine import PortfolioBacktester
    from backtests.momentum.engine.vdubus_engine import VdubusEngine

    print("  Loading data for diagnostics...")
    nqdtc_data = _load_nqdtc_data("NQ", DATA_DIR)
    vdubus_data = _load_vdubus_data("NQ", DATA_DIR)

    # Build configs with optimal mutations
    portfolio_cfg = PortfolioBacktestConfig()
    portfolio_cfg = mutate_portfolio_config(portfolio_cfg, mutations)

    nqdtc_muts = extract_passthrough_mutations(mutations, "nqdtc")
    vdubus_muts = extract_passthrough_mutations(mutations, "vdubus")

    # Run engines
    nqdtc_cfg = NQDTCBacktestConfig(symbols=["MNQ"], initial_equity=EQUITY, fixed_qty=10)
    if nqdtc_muts:
        nqdtc_cfg = mutate_nqdtc_config(nqdtc_cfg, nqdtc_muts)
    engine = NQDTCEngine(symbol="MNQ", bt_config=nqdtc_cfg)
    nqdtc_result = engine.run(
        nqdtc_data["five_min_bars"], nqdtc_data["thirty_min"], nqdtc_data["hourly"],
        nqdtc_data["four_hour"], nqdtc_data["daily"],
        nqdtc_data["thirty_min_idx_map"], nqdtc_data["hourly_idx_map"],
        nqdtc_data["four_hour_idx_map"], nqdtc_data["daily_idx_map"],
        daily_es=nqdtc_data.get("daily_es"),
        daily_es_idx_map=nqdtc_data.get("daily_es_idx_map"),
    )

    from backtests.momentum.config_vdubus import VdubusAblationFlags
    vdubus_cfg = VdubusBacktestConfig(
        initial_equity=EQUITY, fixed_qty=10,
        flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
    )
    if vdubus_muts:
        vdubus_cfg = mutate_vdubus_config(vdubus_cfg, vdubus_muts)
    engine = VdubusEngine(symbol="NQ", bt_config=vdubus_cfg)
    vdubus_result = engine.run(
        vdubus_data["bars_15m"], vdubus_data.get("bars_5m"), vdubus_data["hourly"],
        vdubus_data["daily_es"], vdubus_data["hourly_idx_map"],
        vdubus_data["daily_es_idx_map"], vdubus_data.get("five_to_15_idx_map"),
    )

    # Portfolio simulation
    backtester = PortfolioBacktester(portfolio_cfg)
    result = backtester.run(nqdtc_trades=nqdtc_result.trades, vdubus_trades=vdubus_result.trades)

    # Compute metrics
    eq = result.equity_curve
    ts = np.array([dt.timestamp() for dt in result.equity_timestamps]) if result.equity_timestamps else np.array([])
    init_eq = portfolio_cfg.portfolio.initial_equity
    metrics = extract_metrics(result.trades, eq, ts, init_eq)

    # Print diagnostics
    lines = [
        "=" * 70,
        "OPTIMAL PORTFOLIO DIAGNOSTICS",
        "=" * 70,
        f"Total trades:      {metrics.total_trades}",
        f"Win rate:          {metrics.win_rate:.1%}",
        f"Profit factor:     {metrics.profit_factor:.2f}",
        f"Net profit:        ${metrics.net_profit:,.0f}",
        f"Max drawdown:      {metrics.max_drawdown_pct:.2%} (${metrics.max_drawdown_dollar:,.0f})",
        f"Sharpe:            {metrics.sharpe:.2f}",
        f"Calmar:            {metrics.calmar:.2f}",
        f"Expectancy:        {metrics.expectancy:.2f}R (${metrics.expectancy_dollar:.0f})",
        f"Trades/month:      {metrics.trades_per_month:.1f}",
        "",
        f"Per-strategy breakdown:",
        f"  NQDTC:  {len(nqdtc_result.trades)} trades",
        f"  Vdubus: {len(vdubus_result.trades)} trades",
        f"  Portfolio (after rules): {len(result.trades)} trades",
        f"  Blocked: {len(result.blocked_trades)} trades",
        "",
    ]

    # Rule impact
    if result.rule_blocks:
        lines.append("Rule blocks:")
        for rule, count in sorted(result.rule_blocks.items(), key=lambda x: -x[1]):
            blocked_pnl = result.rule_blocked_pnl.get(rule, 0)
            lines.append(f"  {rule}: {count} blocks (${blocked_pnl:+,.0f} blocked PnL)")

    diag_text = "\n".join(lines)
    print(diag_text)

    diag_path = OUTPUT_DIR / "greedy_optimal_diagnostics.txt"
    diag_path.write_text(diag_text, encoding="utf-8")
    print(f"\n  Diagnostics saved to: {diag_path}")


def _save_experiment_summary() -> None:
    """Save a summary of all experiment results as JSON."""
    from backtests.momentum.auto.results_tracker import ResultsTracker

    tracker = ResultsTracker(OUTPUT_DIR)
    results = tracker.load_all()

    summary = {
        "total": len(results),
        "by_status": {},
        "by_strategy": {},
        "by_type": {},
        "top_improvements": [],
    }

    for r in results:
        summary["by_status"][r.status] = summary["by_status"].get(r.status, 0) + 1
        summary["by_strategy"][r.strategy] = summary["by_strategy"].get(r.strategy, 0) + 1
        summary["by_type"][r.type] = summary["by_type"].get(r.type, 0) + 1

    # Top 20 improvements
    positive = sorted(
        [r for r in results if r.delta_pct > 0 and r.status != "CRASH"],
        key=lambda x: -x.delta_pct,
    )[:20]
    summary["top_improvements"] = [
        {
            "id": r.experiment_id,
            "strategy": r.strategy,
            "type": r.type,
            "delta_pct": r.delta_pct,
            "score": r.experiment_score,
            "status": r.status,
        }
        for r in positive
    ]

    path = OUTPUT_DIR / "experiment_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Momentum auto pipeline")
    parser.add_argument(
        "--phase",
        choices=list(_PHASES),
        default="full",
    )
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--equity", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--experiment-ids", nargs="*", default=None)
    parser.add_argument("--skip-robustness", action="store_true", default=True)
    args = parser.parse_args()

    main(
        phase=args.phase,
        strategy_filter=args.strategy,
        resume=not args.no_resume,
        max_workers=args.max_workers,
        equity=args.equity,
        data_dir=args.data_dir,
        experiment_ids=args.experiment_ids,
        skip_robustness=args.skip_robustness,
    )
