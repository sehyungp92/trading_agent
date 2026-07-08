"""Greedy forward selection for optimal regime MetaConfig.

Algorithm:
  1. Load cached data (one-time)
  2. Score baseline MetaConfig() → base_score
  3. For each round: try each remaining candidate merged with accepted mutations
  4. Accept best if delta >= min_delta
  5. Stop when no improvement

Uses multiprocessing with module-level worker state (same pattern as momentum).
Tiered evaluation: non-HMM candidates reuse cached HMM models for ~2.3× speedup.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level worker state (initialized once per process)
_worker_macro_df = None
_worker_market_df = None
_worker_strat_ret_df = None
_worker_sim_cfg = None
_worker_growth_feature = None
_worker_inflation_feature = None
_worker_phase = None  # Phase number for phase-aware scoring (None = legacy)

# HMM cache state (per-worker, built lazily)
_worker_hmm_cache = None
_worker_cache_key = None

# Signal cache state (per-worker, built lazily — crisis, scanner, covariance)
_worker_signal_cache = None
_worker_signal_cache_key = None


def _init_worker(
    data_dir_str: str,
    initial_equity: float,
    rebalance_cost_bps: float,
    growth_feature: str,
    inflation_feature: str,
    phase: int = 0,
) -> None:
    """Initialize worker process with shared data."""
    global _worker_macro_df, _worker_market_df, _worker_strat_ret_df
    global _worker_sim_cfg, _worker_growth_feature, _worker_inflation_feature
    global _worker_phase

    from backtests.regime._aliases import install
    install()

    from backtests.regime.config import RegimeBacktestConfig
    from backtests.regime.data.downloader import load_cached_data

    data_dir = Path(data_dir_str)
    _worker_macro_df, _worker_market_df, _worker_strat_ret_df = load_cached_data(data_dir)
    _worker_sim_cfg = RegimeBacktestConfig(
        initial_equity=initial_equity,
        rebalance_cost_bps=rebalance_cost_bps,
        data_dir=data_dir,
    )
    _worker_growth_feature = growth_feature
    _worker_inflation_feature = inflation_feature
    _worker_phase = phase if phase > 0 else None


def _worker_score(mutations: dict) -> tuple[float, bool, str]:
    """Score a MetaConfig via full engine run (HMM fitting included).

    Returns:
        (score, rejected, reject_reason)
    """
    try:
        from regime.config import MetaConfig
        from regime.engine import run_signal_engine

        from backtests.regime.auto.config_mutator import mutate_meta_config
        from backtests.regime.engine.portfolio_sim import simulate_portfolio

        cfg = mutate_meta_config(MetaConfig(), mutations)

        signals = run_signal_engine(
            macro_df=_worker_macro_df,
            strat_ret_df=_worker_strat_ret_df,
            market_df=_worker_market_df,
            growth_feature=_worker_growth_feature,
            inflation_feature=_worker_inflation_feature,
            cfg=cfg,
        )

        result = simulate_portfolio(signals, _worker_strat_ret_df, _worker_sim_cfg)

        if _worker_phase is not None:
            from backtests.regime.auto.phase_scoring import (
                compute_regime_stats,
                get_phase_scorer,
            )
            scorer = get_phase_scorer(_worker_phase)
            regime_stats = compute_regime_stats(signals, L_max=cfg.L_max)
            score = scorer(result.metrics, regime_stats)
        else:
            from backtests.regime.auto.scoring import composite_score
            score = composite_score(result.metrics)

        return score.total, score.rejected, score.reject_reason

    except Exception as exc:
        logger.error("Worker failed: %s", exc)
        return 0.0, True, f"CRASH: {exc}"


def _worker_score_cached(args: tuple) -> tuple[float, bool, str]:
    """Score a MetaConfig using cached HMM models (skips HMM fitting).

    Args:
        args: (mutations_dict, base_mutations_for_cache)

    Returns:
        (score, rejected, reject_reason)
    """
    mutations, base_muts = args
    global _worker_hmm_cache, _worker_cache_key
    global _worker_signal_cache, _worker_signal_cache_key

    try:
        from regime.config import MetaConfig

        from backtests.regime.auto.config_mutator import mutate_meta_config
        from backtests.regime.engine.cached_engine import (
            build_hmm_cache,
            build_signal_cache,
            hmm_cache_key,
            run_from_cache,
            signal_cache_key,
        )
        from backtests.regime.engine.portfolio_sim import simulate_portfolio

        base_cfg = mutate_meta_config(MetaConfig(), base_muts)

        # Build/update HMM cache lazily (once per worker per HMM-affecting change)
        needed_key = hmm_cache_key(base_muts)
        if _worker_hmm_cache is None or _worker_cache_key != needed_key:
            _worker_hmm_cache = build_hmm_cache(
                _worker_macro_df, _worker_strat_ret_df, _worker_market_df,
                _worker_growth_feature, _worker_inflation_feature, base_cfg,
            )
            _worker_cache_key = needed_key
            _worker_signal_cache = None  # invalidate when HMM changes

        # Build/update signal cache lazily (crisis, scanner, covariance)
        needed_sig_key = signal_cache_key(base_cfg)
        if _worker_signal_cache is None or _worker_signal_cache_key != needed_sig_key:
            _worker_signal_cache = build_signal_cache(
                _worker_hmm_cache, _worker_strat_ret_df, _worker_market_df, base_cfg,
            )
            _worker_signal_cache_key = needed_sig_key

        cfg = mutate_meta_config(MetaConfig(), mutations)
        signals = run_from_cache(
            _worker_hmm_cache, _worker_strat_ret_df, _worker_market_df, cfg,
            signal_cache=_worker_signal_cache,
        )

        result = simulate_portfolio(signals, _worker_strat_ret_df, _worker_sim_cfg)

        if _worker_phase is not None:
            from backtests.regime.auto.phase_scoring import (
                compute_regime_stats,
                get_phase_scorer,
            )
            scorer = get_phase_scorer(_worker_phase)
            regime_stats = compute_regime_stats(signals, L_max=cfg.L_max)
            score = scorer(result.metrics, regime_stats)
        else:
            from backtests.regime.auto.scoring import composite_score
            score = composite_score(result.metrics)

        return score.total, score.rejected, score.reject_reason

    except Exception as exc:
        logger.error("Cached worker failed: %s", exc)
        return 0.0, True, f"CRASH: {exc}"


def _worker_final_metrics(mutations: dict) -> dict:
    """Run final evaluation in worker and return metrics dict."""
    try:
        from dataclasses import asdict

        from regime.config import MetaConfig
        from regime.engine import run_signal_engine

        from backtests.regime.auto.config_mutator import mutate_meta_config
        from backtests.regime.engine.portfolio_sim import simulate_portfolio

        cfg = mutate_meta_config(MetaConfig(), mutations)
        signals = run_signal_engine(
            macro_df=_worker_macro_df,
            strat_ret_df=_worker_strat_ret_df,
            market_df=_worker_market_df,
            growth_feature=_worker_growth_feature,
            inflation_feature=_worker_inflation_feature,
            cfg=cfg,
        )
        result = simulate_portfolio(signals, _worker_strat_ret_df, _worker_sim_cfg)
        return asdict(result.metrics)
    except Exception as exc:
        logger.warning("Failed to get final metrics: %s", exc)
        return {}


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
    final_metrics: dict = field(default_factory=dict)
    max_rounds: int = 50


def _serialize_value(v: Any) -> Any:
    """Make values JSON-serializable."""
    if isinstance(v, (tuple, list)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _default_workers() -> int:
    """Conservative default worker count (Windows multiprocessing is fragile)."""
    n_cpu = mp.cpu_count()
    if platform.system() == "Windows":
        return min(max(1, n_cpu // 3), 4)
    return max(1, n_cpu - 1)


# ---------------------------------------------------------------------------
# Timeout-aware parallel evaluation
# ---------------------------------------------------------------------------

def _map_with_timeout(
    pool: mp.Pool,
    func,
    args_list: list,
    timeout: float,
    names: list[str],
    verbose: bool = True,
) -> tuple[list[tuple[float, bool, str]], set[str]]:
    """Evaluate candidates with per-item timeout via apply_async.

    Returns:
        (results, crashed_names) where crashed_names contains candidates
        that timed out or raised pool-level exceptions. These should be
        permanently pruned since they indicate hung or broken workers.
    """
    if not args_list:
        return [], set()

    async_results = [pool.apply_async(func, (arg,)) for arg in args_list]
    results: list[tuple[float, bool, str]] = []
    crashed: set[str] = set()
    t0 = time.time()

    for i, ar in enumerate(async_results):
        name = names[i]
        try:
            score, rejected, reason = ar.get(timeout=timeout)
            elapsed_i = time.time() - t0
            if rejected:
                logger.info("  [%d/%d] %-35s REJECTED (%s)  [%.0fs]",
                            i + 1, len(args_list), name, reason, elapsed_i)
                if reason.startswith("CRASH:"):
                    crashed.add(name)
            else:
                logger.info("  [%d/%d] %-35s score=%.4f  [%.0fs]",
                            i + 1, len(args_list), name, score, elapsed_i)
            results.append((score, rejected, reason))
        except mp.TimeoutError:
            elapsed_i = time.time() - t0
            reason = f"TIMEOUT after {int(timeout)}s"
            logger.warning("  [%d/%d] %-35s %s  [%.0fs]",
                           i + 1, len(args_list), name, reason, elapsed_i)
            if verbose:
                print(f"    TIMEOUT: {name} (>{int(timeout)}s)", flush=True)
            results.append((0.0, True, reason))
            crashed.add(name)
        except Exception as exc:
            elapsed_i = time.time() - t0
            reason = f"CRASH: {exc}"
            logger.warning("  [%d/%d] %-35s POOL ERROR: %s  [%.0fs]",
                           i + 1, len(args_list), name, exc, elapsed_i)
            results.append((0.0, True, reason))
            crashed.add(name)

    elapsed = time.time() - t0
    if verbose and len(args_list) > 0:
        print(f"    {len(args_list)} candidates in {elapsed:.0f}s"
              f"{f' ({len(crashed)} crashed)' if crashed else ''}",
              flush=True)
    return results, crashed


def _score_mutation_set(
    pool: mp.Pool,
    mutations: dict,
    timeout: float,
) -> tuple[float, bool, str]:
    """Score a single merged mutation set with timeout handling."""
    async_result = pool.apply_async(_worker_score, (mutations,))
    try:
        return async_result.get(timeout=timeout)
    except mp.TimeoutError:
        return (0.0, True, f"TIMEOUT after {int(timeout)}s")
    except Exception as exc:
        return (0.0, True, f"CRASH: {exc}")


def _merged_mutations(base_muts: dict, accepted_sequence: list[tuple[str, dict]]) -> dict:
    """Reconstruct the merged mutation dict from ordered accepted mutations."""
    merged = dict(base_muts)
    for _, muts in accepted_sequence:
        merged.update(muts)
    return merged


def _rollback_last_mutations(
    pool: mp.Pool,
    base_muts: dict,
    accepted_sequence: list[tuple[str, dict]],
    current_score: float,
    min_delta: float,
    timeout: float,
    verbose: bool = True,
) -> tuple[list[tuple[str, dict]], float, list[GreedyRound]]:
    """Re-evaluate the tail of the accepted set and drop harmful mutations."""
    rollback_rounds: list[GreedyRound] = []

    for _ in range(2):
        tail = accepted_sequence[-2:]
        if not tail:
            break

        variants = []
        for remove_name, _ in tail:
            variant = [
                (name, muts)
                for name, muts in accepted_sequence
                if name != remove_name
            ]
            variants.append((remove_name, variant))

        best_name = None
        best_sequence = accepted_sequence
        best_score = current_score

        for remove_name, variant_sequence in variants:
            merged = _merged_mutations(base_muts, variant_sequence)
            score, rejected, reason = _score_mutation_set(pool, merged, timeout)
            if rejected:
                logger.info(
                    "  Rollback candidate %s rejected (%s)",
                    remove_name,
                    reason,
                )
                continue
            delta = score - current_score
            logger.info(
                "  Rollback candidate drop=%s score=%.4f delta=%+.4f",
                remove_name,
                score,
                delta,
            )
            if delta >= min_delta and score > best_score:
                best_name = remove_name
                best_sequence = variant_sequence
                best_score = score

        if best_name is None:
            break

        removed_mutation = next(muts for name, muts in accepted_sequence if name == best_name)
        accepted_sequence = best_sequence
        rollback_rounds.append(
            GreedyRound(
                round_num=len(accepted_sequence) + len(rollback_rounds) + 1,
                candidate_id=f"rollback_remove_{best_name}",
                candidate_mutations={k: _serialize_value(v) for k, v in removed_mutation.items()},
                score_before=current_score,
                score_after=best_score,
                delta=best_score - current_score,
                accepted=True,
            )
        )
        current_score = best_score
        logger.info(
            "  ROLLBACK REMOVED: %s | score=%.4f",
            best_name,
            current_score,
        )
        if verbose:
            print(
                f"  ROLLBACK REMOVED: {best_name}  "
                f"score={current_score:.4f}",
                flush=True,
            )

    return accepted_sequence, current_score, rollback_rounds


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def run_greedy(
    candidates: list[tuple[str, dict]],
    data_dir: Path | None = None,
    initial_equity: float = 100_000.0,
    rebalance_cost_bps: float = 5.0,
    growth_feature: str = "GROWTH",
    inflation_feature: str = "INFLATION",
    max_workers: int | None = None,
    max_rounds: int = 50,
    min_delta: float = 0.001,
    prune_threshold: float = -0.10,
    base_mutations: dict | None = None,
    verbose: bool = True,
    phase: int = 0,
    candidate_timeout: float = 600.0,
) -> GreedyResult:
    """Run greedy forward selection on MetaConfig parameters.

    Uses tiered evaluation: candidates that don't affect HMM fitting reuse
    cached HMM models, providing ~2-3× speedup per round.

    Args:
        candidates: List of (name, mutations_dict) to try.
        data_dir: Path to cached parquet data.
        initial_equity: Starting equity.
        rebalance_cost_bps: Per-unit turnover cost.
        growth_feature: Column name for growth in macro_df.
        inflation_feature: Column name for inflation in macro_df.
        max_workers: Pool size (None = auto, conservative on Windows).
        max_rounds: Stop after this many rounds.
        min_delta: Minimum score improvement to accept.
        prune_threshold: Remove candidates with delta below this (default -0.10).
            Also prunes worker failures (rejected/exceptions). Set to None to disable.
        base_mutations: Pre-applied mutations forming the baseline.
        verbose: Print progress.

    Returns:
        GreedyResult with optimal mutations and round-by-round history.
    """
    from backtests.regime.engine.cached_engine import mutations_affect_hmm

    if data_dir is None:
        data_dir = Path("backtests/regime/data/raw")

    t0 = time.time()
    n_workers = max_workers or _default_workers()
    base_muts = dict(base_mutations) if base_mutations else {}

    logger.info("Starting greedy optimization: %d candidates, %d workers, "
                "max_rounds=%d, min_delta=%.4f, phase=%d",
                len(candidates), n_workers, max_rounds, min_delta, phase)
    logger.info("Base mutations: %s", base_muts)
    if verbose:
        print(f"Starting greedy optimization", flush=True)
        print(f"  Candidates: {len(candidates)}", flush=True)
        print(f"  Workers: {n_workers}", flush=True)
        print(f"  Max rounds: {max_rounds}", flush=True)
        print(f"  Min delta: {min_delta}", flush=True)

    # Create pool once, reuse for baseline + all rounds + final metrics
    initargs = (str(data_dir), initial_equity, rebalance_cost_bps,
                growth_feature, inflation_feature, phase)

    accepted_muts = dict(base_muts)
    accepted_sequence: list[tuple[str, dict]] = []
    current_score = 0.0
    remaining = list(candidates)
    rounds = []
    total_tested = 0

    pool = mp.Pool(n_workers, initializer=_init_worker, initargs=initargs,
                   maxtasksperchild=20)
    _needs_pool_recreate = False
    try:
        # Score baseline with timeout
        _bl_ar = pool.apply_async(_worker_score, (base_muts,))
        try:
            baseline_result = _bl_ar.get(timeout=candidate_timeout)
        except mp.TimeoutError:
            logger.error("Baseline timed out after %ds", int(candidate_timeout))
            baseline_result = (0.0, True, f"TIMEOUT after {int(candidate_timeout)}s")
            pool.terminate()
            pool.join()
            pool = mp.Pool(n_workers, initializer=_init_worker, initargs=initargs,
                           maxtasksperchild=20)
        baseline_score = baseline_result[0]
        _bl_rejected = baseline_result[1]
        _bl_reason = baseline_result[2]
        _bl_status = f"REJECTED ({_bl_reason})" if _bl_rejected else "OK"
        logger.info("Baseline score: %.4f [%s]", baseline_score, _bl_status)

        if verbose:
            print(f"  Baseline score: {baseline_score:.4f} [{_bl_status}]", flush=True)

        current_score = baseline_score

        for round_num in range(1, max_rounds + 1):
            if not remaining:
                if verbose:
                    print(f"\nNo candidates remaining. Stopping.", flush=True)
                break

            # Split candidates into HMM-affecting (full run) and cacheable (fast)
            hmm_tasks = []
            fast_tasks = []
            for name, muts in remaining:
                merged = {**accepted_muts, **muts}
                if mutations_affect_hmm(muts):
                    hmm_tasks.append((name, muts, merged))
                else:
                    fast_tasks.append((name, muts, merged))

            n_hmm = len(hmm_tasks)
            n_fast = len(fast_tasks)
            logger.info("Round %d: %d candidates (%d HMM + %d cached)",
                        round_num, n_hmm + n_fast, n_hmm, n_fast)
            if verbose:
                print(f"\n--- Round {round_num} ({n_hmm + n_fast} candidates: "
                      f"{n_hmm} full + {n_fast} cached) ---", flush=True)

            # Evaluate HMM-affecting candidates (full engine) with timeout
            hmm_results = []
            hmm_crashed: set[str] = set()
            if hmm_tasks:
                t_hmm = time.time()
                hmm_results, hmm_crashed = _map_with_timeout(
                    pool, _worker_score,
                    [t[2] for t in hmm_tasks],
                    timeout=candidate_timeout,
                    names=[t[0] for t in hmm_tasks],
                    verbose=verbose,
                )
                hmm_elapsed = time.time() - t_hmm
                logger.info("  HMM evals: %d in %.0fs (%d crashed)",
                            len(hmm_tasks), hmm_elapsed, len(hmm_crashed))
                if verbose:
                    print(f"  HMM evals: {len(hmm_tasks)} in "
                          f"{hmm_elapsed:.0f}s", flush=True)
                if hmm_crashed:
                    _needs_pool_recreate = True

            # Evaluate non-HMM candidates (cached engine — ~6-10× faster each)
            fast_results = []
            fast_crashed: set[str] = set()
            if fast_tasks:
                t_fast = time.time()
                fast_args = [(t[2], accepted_muts) for t in fast_tasks]
                fast_results, fast_crashed = _map_with_timeout(
                    pool, _worker_score_cached,
                    fast_args,
                    timeout=max(candidate_timeout // 2, 300),
                    names=[t[0] for t in fast_tasks],
                    verbose=verbose,
                )
                fast_elapsed = time.time() - t_fast
                logger.info("  Cached evals: %d in %.0fs (%d crashed)",
                            len(fast_tasks), fast_elapsed, len(fast_crashed))
                if verbose:
                    print(f"  Cached evals: {len(fast_tasks)} in "
                          f"{fast_elapsed:.0f}s", flush=True)
                if fast_crashed:
                    _needs_pool_recreate = True

            # Combine results
            all_tasks = hmm_tasks + fast_tasks
            all_results = list(hmm_results) + list(fast_results)
            total_tested += len(all_tasks)

            # Log per-candidate scores
            for i, (score_i, rejected_i, reason_i) in enumerate(all_results):
                name_i = all_tasks[i][0]
                delta_i = score_i - current_score
                if rejected_i:
                    logger.debug("  %-40s REJECTED (%s)", name_i, reason_i)
                else:
                    logger.debug("  %-40s score=%.4f delta=%+.4f",
                                 name_i, score_i, delta_i)

            # Find best
            best_idx = -1
            best_score = current_score
            best_delta = 0.0

            for i, (score, rejected, reason) in enumerate(all_results):
                if rejected:
                    continue
                delta = score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_score = score
                    best_idx = i

            # Prune crashed candidates (timeouts + exceptions) — permanent removal
            all_crashed = hmm_crashed | fast_crashed
            for i, (score, rejected, reason) in enumerate(all_results):
                name_i = all_tasks[i][0]
                if rejected and (reason.startswith("CRASH:") or
                                 reason.startswith("TIMEOUT")):
                    all_crashed.add(name_i)
            if all_crashed:
                before = len(remaining)
                remaining = [(n, m) for n, m in remaining
                             if n not in all_crashed]
                logger.info("  Pruned %d crashed candidates: %s",
                            before - len(remaining), sorted(all_crashed))
                if verbose:
                    print(f"  PRUNED {len(all_crashed)} crashed/timed-out "
                          f"candidates", flush=True)

            # Prune candidates that scored very negatively (but NOT hard-rejected
            # candidates — they may pass after an accepted mutation shifts the
            # baseline, e.g. a candidate rejected for "<3 regimes" could work
            # after another candidate fixes regime collapse).
            if prune_threshold is not None:
                prune_names = set()
                for i, (score, rejected, reason) in enumerate(all_results):
                    name_i = all_tasks[i][0]
                    if not rejected:
                        delta_i = score - current_score
                        if delta_i < prune_threshold:
                            prune_names.add(name_i)

                if prune_names:
                    logger.info("  Pruned %d candidates (delta < %.2f): %s",
                                len(prune_names), prune_threshold,
                                sorted(prune_names))
                    if verbose:
                        print(f"  PRUNED {len(prune_names)} candidates "
                              f"(delta < {prune_threshold})", flush=True)
                remaining = [(n, m) for n, m in remaining
                             if n not in prune_names]

            if best_idx >= 0 and best_delta >= min_delta:
                name, muts, _ = all_tasks[best_idx]
                accepted_sequence.append((name, muts))
                accepted_muts = _merged_mutations(base_muts, accepted_sequence)

                round_rec = GreedyRound(
                    round_num=round_num,
                    candidate_id=name,
                    candidate_mutations={k: _serialize_value(v) for k, v in muts.items()},
                    score_before=current_score,
                    score_after=best_score,
                    delta=best_delta,
                    accepted=True,
                )
                rounds.append(round_rec)

                logger.info("  ACCEPTED: %s | score=%.4f -> %.4f (delta=+%.4f) | "
                            "mutations=%s",
                            name, current_score, best_score, best_delta, muts)
                if verbose:
                    print(f"  ACCEPTED: {name}  "
                          f"score={best_score:.4f}  delta=+{best_delta:.4f}", flush=True)

                current_score = best_score
                remaining = [(n, m) for n, m in remaining if n != name]
            else:
                logger.info("  No improvement >= %.4f. Stopping. Best delta=%.4f",
                            min_delta, best_delta)
                if verbose:
                    print(f"  No improvement >= {min_delta}. Stopping.", flush=True)
                break

            # Recreate pool if any workers hung during this round
            if _needs_pool_recreate:
                logger.info("  Recreating worker pool (cleaning up hung workers)")
                if verbose:
                    print("  Recreating worker pool...", flush=True)
                pool.terminate()
                pool.join()
                pool = mp.Pool(n_workers, initializer=_init_worker,
                               initargs=initargs, maxtasksperchild=20)
                _needs_pool_recreate = False

        accepted_sequence, current_score, rollback_rounds = _rollback_last_mutations(
            pool=pool,
            base_muts=base_muts,
            accepted_sequence=accepted_sequence,
            current_score=current_score,
            min_delta=min_delta,
            timeout=candidate_timeout,
            verbose=verbose,
        )
        if rollback_rounds:
            rounds.extend(rollback_rounds)
            accepted_muts = _merged_mutations(base_muts, accepted_sequence)

        # Get final metrics via a worker (avoids importing regime in main process)
        _fm_ar = pool.apply_async(_worker_final_metrics, (accepted_muts,))
        try:
            final_result = _fm_ar.get(timeout=candidate_timeout)
        except (mp.TimeoutError, Exception) as exc:
            logger.warning("Final metrics eval failed: %s", exc)
            final_result = {}
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0
    final_metrics = final_result if isinstance(final_result, dict) else {}

    greedy_result = GreedyResult(
        baseline_score=baseline_score,
        final_score=current_score,
        accepted_mutations={k: _serialize_value(v) for k, v in accepted_muts.items()},
        rounds=rounds,
        total_candidates_tested=total_tested,
        elapsed_seconds=elapsed,
        final_metrics=final_metrics,
        max_rounds=max_rounds,
    )

    logger.info("Greedy complete: baseline=%.4f final=%.4f (+%.4f) "
                "rounds=%d tested=%d elapsed=%.0fs",
                baseline_score, current_score, current_score - baseline_score,
                len(rounds), total_tested, elapsed)
    logger.info("Final accepted mutations: %s", accepted_muts)
    if verbose:
        print(f"\n=== Greedy Optimization Complete ===", flush=True)
        print(f"  Baseline: {baseline_score:.4f}", flush=True)
        print(f"  Final:    {current_score:.4f}  (+{current_score - baseline_score:.4f})", flush=True)
        print(f"  Rounds:   {len(rounds)}", flush=True)
        print(f"  Tested:   {total_tested}", flush=True)
        print(f"  Elapsed:  {elapsed:.0f}s", flush=True)
        print(f"  Accepted: {list(accepted_muts.keys())}", flush=True)

    return greedy_result


def save_greedy_result(result: GreedyResult, path: Path) -> None:
    """Save greedy result to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "baseline_score": result.baseline_score,
        "final_score": result.final_score,
        "accepted_mutations": result.accepted_mutations,
        "total_candidates_tested": result.total_candidates_tested,
        "elapsed_seconds": result.elapsed_seconds,
        "final_metrics": result.final_metrics,
        "rounds": [
            {
                "round_num": r.round_num,
                "candidate_id": r.candidate_id,
                "candidate_mutations": r.candidate_mutations,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "delta": r.delta,
                "accepted": r.accepted,
            }
            for r in result.rounds
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved greedy result to {path}", flush=True)
