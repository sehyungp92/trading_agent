"""Greedy forward selection for optimal momentum portfolio config.

Algorithm:
  1. Start from baseline (v6) portfolio config
  2. Each round, try adding each candidate mutation
  3. Pick the candidate that gives highest portfolio score improvement
  4. Accept if improvement >= min_delta, else stop
  5. Record round-by-round history

Worker processes run the active engines independently, then feed trade lists
to the post-hoc PortfolioBacktester.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level worker state (initialized once per process)
_worker_nqdtc_data: dict | None = None
_worker_vdubus_data: dict | None = None
_worker_equity: float = 10_000.0


def _init_worker(data_dir_str: str, equity: float) -> None:
    """Initialize worker process with shared data."""
    global _worker_nqdtc_data, _worker_vdubus_data, _worker_equity

    from backtests.momentum.cli import (
        _load_nqdtc_data,
        _load_vdubus_data,
    )

    data_dir = Path(data_dir_str)
    _worker_nqdtc_data = _load_nqdtc_data("NQ", data_dir)
    _worker_vdubus_data = _load_vdubus_data("NQ", data_dir)
    _worker_equity = equity


def _worker_score(mutations: dict) -> tuple[float, bool, str]:
    """Score a portfolio config in a worker process.

    Runs active engines independently, then feeds trade lists to
    PortfolioBacktester for post-hoc simulation.

    Returns:
        (score, rejected, reject_reason)
    """
    from backtests.momentum.auto.config_mutator import (
        extract_passthrough_mutations,
        mutate_nqdtc_config,
        mutate_portfolio_config,
        mutate_vdubus_config,
    )
    from backtests.momentum.auto.scoring import composite_score, extract_metrics
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.config_portfolio import PortfolioBacktestConfig
    from backtests.momentum.config_vdubus import VdubusBacktestConfig
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
    from backtests.momentum.engine.portfolio_engine import PortfolioBacktester
    from backtests.momentum.engine.vdubus_engine import VdubusEngine

    try:
        # Build portfolio config
        portfolio_cfg = PortfolioBacktestConfig()
        portfolio_cfg = mutate_portfolio_config(portfolio_cfg, mutations)

        # Extract per-strategy mutations
        nqdtc_muts = extract_passthrough_mutations(mutations, "nqdtc")
        vdubus_muts = extract_passthrough_mutations(mutations, "vdubus")

        # Run NQDTC
        nqdtc_trades = []
        if portfolio_cfg.run_nqdtc and _worker_nqdtc_data:
            cfg = NQDTCBacktestConfig(symbols=["MNQ"], initial_equity=_worker_equity, fixed_qty=10)
            if nqdtc_muts:
                cfg = mutate_nqdtc_config(cfg, nqdtc_muts)
            d = _worker_nqdtc_data
            engine = NQDTCEngine(symbol="MNQ", bt_config=cfg)
            result = engine.run(
                d["five_min_bars"], d["thirty_min"], d["hourly"], d["four_hour"], d["daily"],
                d["thirty_min_idx_map"], d["hourly_idx_map"], d["four_hour_idx_map"], d["daily_idx_map"],
                daily_es=d.get("daily_es"), daily_es_idx_map=d.get("daily_es_idx_map"),
            )
            nqdtc_trades = result.trades

        # Run Vdubus
        vdubus_trades = []
        if portfolio_cfg.run_vdubus and _worker_vdubus_data:
            from backtests.momentum.config_vdubus import VdubusAblationFlags
            cfg = VdubusBacktestConfig(
                initial_equity=_worker_equity, fixed_qty=10,
                flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
            )
            if vdubus_muts:
                cfg = mutate_vdubus_config(cfg, vdubus_muts)
            d = _worker_vdubus_data
            engine = VdubusEngine(symbol="NQ", bt_config=cfg)
            result = engine.run(
                d["bars_15m"], d.get("bars_5m"), d["hourly"], d["daily_es"],
                d["hourly_idx_map"], d["daily_es_idx_map"], d.get("five_to_15_idx_map"),
            )
            vdubus_trades = result.trades

        # Portfolio simulation
        backtester = PortfolioBacktester(portfolio_cfg)
        result = backtester.run(nqdtc_trades=nqdtc_trades, vdubus_trades=vdubus_trades)

        eq = result.equity_curve
        ts = np.array([dt.timestamp() for dt in result.equity_timestamps]) if result.equity_timestamps else np.array([])
        init_eq = portfolio_cfg.portfolio.initial_equity

        metrics = extract_metrics(result.trades, eq, ts, init_eq)
        score = composite_score(metrics, init_eq, strategy="portfolio", equity_curve=eq)

        return score.total, score.rejected, score.reject_reason

    except Exception as exc:
        logger.error("Worker failed: %s", exc)
        return 0.0, True, str(exc)


# ---------------------------------------------------------------------------
# Strategy-level worker (single engine, used for per-strategy greedy)
# ---------------------------------------------------------------------------

_strat_worker_data: dict | None = None
_strat_worker_strategy: str = ""
_strat_worker_equity: float = 10_000.0


def _init_strategy_worker(strategy: str, data_dir_str: str, equity: float) -> None:
    """Initialize worker for single-strategy greedy (loads only 1 dataset)."""
    global _strat_worker_data, _strat_worker_strategy, _strat_worker_equity

    from backtests.momentum.cli import (
        _load_nqdtc_data,
        _load_vdubus_data,
    )

    _strat_worker_strategy = strategy
    _strat_worker_equity = equity
    data_dir = Path(data_dir_str)

    if strategy == "nqdtc":
        _strat_worker_data = _load_nqdtc_data("NQ", data_dir)
    elif strategy == "vdubus":
        _strat_worker_data = _load_vdubus_data("NQ", data_dir)


def _strategy_worker_score(mutations: dict) -> tuple[float, bool, str]:
    """Score a single-strategy config. Runs only 1 engine."""
    from backtests.momentum.auto.config_mutator import (
        mutate_nqdtc_config,
        mutate_vdubus_config,
    )
    from backtests.momentum.auto.scoring import composite_score, extract_metrics
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.config_vdubus import VdubusBacktestConfig
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
    from backtests.momentum.engine.vdubus_engine import VdubusEngine

    strategy = _strat_worker_strategy
    eq = _strat_worker_equity
    d = _strat_worker_data

    try:
        if strategy == "nqdtc":
            cfg = NQDTCBacktestConfig(symbols=["MNQ"], initial_equity=eq, fixed_qty=10)
            if mutations:
                cfg = mutate_nqdtc_config(cfg, mutations)
            engine = NQDTCEngine(symbol="MNQ", bt_config=cfg)
            r = engine.run(
                d["five_min_bars"], d["thirty_min"], d["hourly"], d["four_hour"], d["daily"],
                d["thirty_min_idx_map"], d["hourly_idx_map"],
                d["four_hour_idx_map"], d["daily_idx_map"],
                daily_es=d.get("daily_es"), daily_es_idx_map=d.get("daily_es_idx_map"),
            )
            trades, ecurve, ts = r.trades, r.equity_curve, r.timestamps

        elif strategy == "vdubus":
            from backtests.momentum.config_vdubus import VdubusAblationFlags
            cfg = VdubusBacktestConfig(
                initial_equity=eq, fixed_qty=10,
                flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
            )
            if mutations:
                cfg = mutate_vdubus_config(cfg, mutations)
            engine = VdubusEngine(symbol="NQ", bt_config=cfg)
            r = engine.run(
                d["bars_15m"], d.get("bars_5m"), d["hourly"], d["daily_es"],
                d["hourly_idx_map"], d["daily_es_idx_map"], d.get("five_to_15_idx_map"),
            )
            trades, ecurve, ts = r.trades, r.equity_curve, r.time_series
        else:
            return 0.0, True, f"Unknown strategy: {strategy}"

        metrics = extract_metrics(trades, ecurve, ts, eq)
        score = composite_score(metrics, eq, strategy=strategy, equity_curve=ecurve)
        return score.total, score.rejected, score.reject_reason

    except Exception as exc:
        logger.error("Strategy worker failed (%s): %s", strategy, exc)
        return 0.0, True, str(exc)


# ---------------------------------------------------------------------------
# Greedy result types
# ---------------------------------------------------------------------------

@dataclass
class GreedyRound:
    """One round of greedy selection."""
    round_num: int
    candidate_id: str
    candidate_mutations: dict
    score_before: float
    score_after: float
    delta: float
    accepted: bool


@dataclass
class GreedyResult:
    """Final result of greedy optimization."""
    baseline_score: float
    final_score: float
    accepted_mutations: dict = field(default_factory=dict)
    rounds: list[GreedyRound] = field(default_factory=list)
    total_candidates_tested: int = 0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def run_greedy(
    candidates: list[tuple[str, dict]],
    initial_equity: float = 10_000.0,
    data_dir: Path | None = None,
    max_workers: int | None = None,
    max_rounds: int = 50,
    min_delta: float = 0.001,
    base_mutations: dict | None = None,
    verbose: bool = True,
) -> GreedyResult:
    """Run greedy forward selection on portfolio config.

    Args:
        candidates: List of (name, mutations_dict) to try.
        initial_equity: Starting equity.
        data_dir: Path to bar data.
        max_workers: Pool size (None = cpu_count).
        max_rounds: Stop after this many rounds.
        min_delta: Minimum score improvement to accept.
        base_mutations: Pre-applied mutations (e.g., from per-strategy greedy).
            These are always included and form the baseline.
        verbose: Print progress.

    Returns:
        GreedyResult with optimal mutations and round-by-round history.
    """
    if data_dir is None:
        data_dir = Path("backtests/momentum/data/raw")

    t0 = time.time()
    n_workers = max_workers or max(1, mp.cpu_count() - 1)

    if verbose:
        print(f"Starting greedy optimization with {len(candidates)} candidates, "
              f"{n_workers} workers")

    # Compute baseline
    base_muts = dict(base_mutations) if base_mutations else {}

    if verbose:
        print("Computing baseline...")
        if base_muts:
            print(f"  (with {len(base_muts)} pre-applied strategy mutations)")

    pool = mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(str(data_dir), initial_equity),
    )

    try:
        baseline_score = pool.apply(_worker_score, (base_muts,))
        baseline_score = baseline_score[0]  # (score, rejected, reason)
        if verbose:
            print(f"Baseline score: {baseline_score:.6f}")

        current_mutations: dict = dict(base_muts)
        current_score = baseline_score
        rounds: list[GreedyRound] = []
        total_tested = 0
        remaining = list(candidates)

        for round_num in range(1, max_rounds + 1):
            if not remaining:
                if verbose:
                    print(f"Round {round_num}: no candidates remaining")
                break

            if verbose:
                print(f"\nRound {round_num}: testing {len(remaining)} candidates...")

            # Build mutation sets: current + each candidate
            tasks = []
            for name, muts in remaining:
                combined = {**current_mutations, **muts}
                tasks.append((name, muts, combined))

            # Score all candidates in parallel
            combined_mutations_list = [t[2] for t in tasks]
            results = pool.map(_worker_score, combined_mutations_list)
            total_tested += len(results)

            # Find best
            best_idx = -1
            best_score = current_score
            best_delta = 0.0

            for i, (score, rejected, reason) in enumerate(results):
                if not rejected and score > best_score:
                    best_score = score
                    best_idx = i
                    best_delta = score - current_score

            if best_idx >= 0 and best_delta >= min_delta:
                name, muts, combined = tasks[best_idx]
                current_mutations = combined
                current_score = best_score

                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidate_id=name,
                    candidate_mutations=muts,
                    score_before=current_score - best_delta,
                    score_after=current_score,
                    delta=best_delta,
                    accepted=True,
                ))

                # Remove accepted candidate from future rounds
                remaining = [(n, m) for n, m in remaining if n != name]

                if verbose:
                    print(f"  ACCEPTED: {name} (score {current_score:.6f}, "
                          f"delta +{best_delta:.6f})")
            else:
                if verbose:
                    print(f"  No improvement >= {min_delta:.4f}, stopping.")
                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidate_id="NONE",
                    candidate_mutations={},
                    score_before=current_score,
                    score_after=current_score,
                    delta=0.0,
                    accepted=False,
                ))
                break

    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - t0

    result = GreedyResult(
        baseline_score=baseline_score,
        final_score=current_score,
        accepted_mutations=current_mutations,
        rounds=rounds,
        total_candidates_tested=total_tested,
        elapsed_seconds=elapsed,
    )

    if verbose:
        _print_summary(result, baseline_score)

    return result


def _print_summary(result: GreedyResult, baseline_score: float) -> None:
    """Print summary of greedy optimization."""
    print(f"\n{'=' * 60}")
    print("GREEDY OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Baseline score:  {baseline_score:.6f}")
    print(f"Final score:     {result.final_score:.6f}")
    if baseline_score > 0:
        print(f"Improvement:     {(result.final_score - baseline_score) / baseline_score:+.2%}")
    print(f"Accepted rounds: {sum(1 for r in result.rounds if r.accepted)}")
    print(f"Total tested:    {result.total_candidates_tested}")
    print(f"Elapsed:         {result.elapsed_seconds:.0f}s ({result.elapsed_seconds / 60:.1f}m)")

    if result.rounds:
        print(f"\nAccepted mutations:")
        for r in result.rounds:
            if r.accepted:
                print(f"  Round {r.round_num}: {r.candidate_id} (+{r.delta:.6f})")


def run_strategy_greedy(
    strategy: str,
    candidates: list[tuple[str, dict]],
    initial_equity: float = 10_000.0,
    data_dir: Path | None = None,
    max_workers: int | None = None,
    max_rounds: int = 50,
    min_delta: float = 0.001,
    verbose: bool = True,
) -> GreedyResult:
    """Greedy forward selection for a single strategy (1 engine per eval).

    Much faster than portfolio greedy since each candidate only runs 1 engine.
    """
    if data_dir is None:
        data_dir = Path("backtests/momentum/data/raw")

    t0 = time.time()
    n_workers = max_workers or max(1, mp.cpu_count() - 1)

    if verbose:
        print(f"  [{strategy.upper()}] Greedy: {len(candidates)} candidates, "
              f"{n_workers} workers")

    pool = mp.Pool(
        processes=n_workers,
        initializer=_init_strategy_worker,
        initargs=(strategy, str(data_dir), initial_equity),
    )

    try:
        baseline_score = pool.apply(_strategy_worker_score, ({},))[0]
        if verbose:
            print(f"  [{strategy.upper()}] Baseline: {baseline_score:.6f}")

        current_mutations: dict = {}
        current_score = baseline_score
        rounds: list[GreedyRound] = []
        total_tested = 0
        remaining = list(candidates)

        for round_num in range(1, max_rounds + 1):
            if not remaining:
                break

            if verbose:
                print(f"  [{strategy.upper()}] Round {round_num}: "
                      f"{len(remaining)} candidates...")

            tasks = []
            for name, muts in remaining:
                combined = {**current_mutations, **muts}
                tasks.append((name, muts, combined))

            combined_list = [t[2] for t in tasks]
            results = pool.map(_strategy_worker_score, combined_list)
            total_tested += len(results)

            best_idx = -1
            best_score = current_score
            best_delta = 0.0

            for i, (score, rejected, reason) in enumerate(results):
                if not rejected and score > best_score:
                    best_score = score
                    best_idx = i
                    best_delta = score - current_score

            if best_idx >= 0 and best_delta >= min_delta:
                name, muts, combined = tasks[best_idx]
                current_mutations = combined
                current_score = best_score
                rounds.append(GreedyRound(
                    round_num=round_num, candidate_id=name,
                    candidate_mutations=muts,
                    score_before=current_score - best_delta,
                    score_after=current_score,
                    delta=best_delta, accepted=True,
                ))
                remaining = [(n, m) for n, m in remaining if n != name]
                if verbose:
                    print(f"  [{strategy.upper()}] ACCEPTED: {name} "
                          f"(score {current_score:.6f}, +{best_delta:.6f})")
            else:
                if verbose:
                    print(f"  [{strategy.upper()}] No improvement, stopping.")
                rounds.append(GreedyRound(
                    round_num=round_num, candidate_id="NONE",
                    candidate_mutations={}, score_before=current_score,
                    score_after=current_score, delta=0.0, accepted=False,
                ))
                break
    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    result = GreedyResult(
        baseline_score=baseline_score,
        final_score=current_score,
        accepted_mutations=current_mutations,
        rounds=rounds,
        total_candidates_tested=total_tested,
        elapsed_seconds=elapsed,
    )

    if verbose:
        accepted = sum(1 for r in rounds if r.accepted)
        imp = ((current_score - baseline_score) / baseline_score * 100
               if baseline_score > 0 else 0.0)
        print(f"  [{strategy.upper()}] Done: {baseline_score:.6f} -> "
              f"{current_score:.6f} ({imp:+.1f}%), "
              f"{accepted} accepted, {elapsed:.0f}s")

    return result


def save_result(result: GreedyResult, output_path: Path) -> None:
    """Serialize greedy result to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    data = {
        "baseline_score": result.baseline_score,
        "final_score": result.final_score,
        "accepted_mutations": result.accepted_mutations,
        "improvement_pct": (
            (result.final_score - result.baseline_score) / result.baseline_score * 100
            if result.baseline_score > 0 else 0.0
        ),
        "rounds": [
            {
                "round": r.round_num,
                "candidate": r.candidate_id,
                "mutations": r.candidate_mutations,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "delta": r.delta,
                "accepted": r.accepted,
            }
            for r in result.rounds
        ],
        "total_tested": result.total_candidates_tested,
        "elapsed_seconds": result.elapsed_seconds,
    }

    output_path.write_text(json.dumps(data, indent=2, default=_serialize))
    print(f"\nGreedy result saved to: {output_path}")


# ---------------------------------------------------------------------------
# Default portfolio candidates (common mutations to try)
# ---------------------------------------------------------------------------

PORTFOLIO_CANDIDATES: list[tuple[str, dict]] = [
    # Heat / position caps
    ("heat_cap_3.0", {"portfolio.heat_cap_R": 3.0}),
    ("heat_cap_4.0", {"portfolio.heat_cap_R": 4.0}),
    ("dir_cap_3.0", {"portfolio.directional_cap_R": 3.0}),
    ("max_pos_2", {"portfolio.max_total_positions": 2}),
    ("max_pos_4", {"portfolio.max_total_positions": 4}),

    # Daily/weekly stops
    ("daily_stop_1.0", {"portfolio.portfolio_daily_stop_R": 1.0}),
    ("daily_stop_2.5", {"portfolio.portfolio_daily_stop_R": 2.5}),
    ("weekly_stop_8", {"portfolio.portfolio_weekly_stop_R": 8.0}),
    ("weekly_stop_0", {"portfolio.portfolio_weekly_stop_R": 0.0}),

    # Cross-strategy rules
    ("nqdtc_dir_filter", {"portfolio.nqdtc_direction_filter_enabled": True}),
    ("nqdtc_oppose_0.25", {"portfolio.nqdtc_oppose_size_mult": 0.25}),

    # Per-strategy risk
    ("nqdtc_risk_0.010", {"portfolio.strategies[1].base_risk_pct": 0.010}),
    ("vdubus_risk_0.010", {"portfolio.strategies[0].base_risk_pct": 0.010}),

    # DD tiers
    ("dd_aggressive", {"portfolio.dd_tiers": ((0.05, 1.0), (0.08, 0.50), (0.12, 0.25), (1.0, 0.0))}),
    ("dd_relaxed", {"portfolio.dd_tiers": ((0.12, 1.0), (0.18, 0.50), (0.22, 0.25), (1.0, 0.0))}),
    ("dd_none", {"portfolio.dd_tiers": ((1.0, 1.0),)}),

    # NQDTC continuation
    ("nqdtc_cont_0.85", {"portfolio.strategies[1].continuation_size_mult": 0.85}),
    ("nqdtc_reversal_only", {"portfolio.strategies[1].reversal_only": True}),
]


def convert_experiment_to_portfolio_mutation(
    strategy: str,
    mutations: dict,
) -> dict:
    """Convert strategy-level experiment mutations to portfolio-level mutations.

    Maps:
      nqdtc flags.X     -> nqdtc_flags.X
      nqdtc param_overrides.Y -> nqdtc_param.Y
      vdubus flags.X    -> vdubus_flags.X
      vdubus param_overrides.Y -> vdubus_param.Y
      portfolio.*        -> pass through unchanged
    """
    portfolio_mutations: dict[str, Any] = {}

    for key, value in mutations.items():
        if strategy == "portfolio":
            portfolio_mutations[key] = value
        elif key.startswith("flags."):
            field_name = key[len("flags."):]
            portfolio_mutations[f"{strategy}_flags.{field_name}"] = value
        elif key.startswith("param_overrides."):
            param_key = key[len("param_overrides."):]
            portfolio_mutations[f"{strategy}_param.{param_key}"] = value
        elif key.startswith("slippage."):
            # Slippage mutations don't map to portfolio level
            pass
        else:
            # Top-level strategy mutations -> prefix with strategy
            portfolio_mutations[f"{strategy}_{key}"] = value

    return portfolio_mutations
