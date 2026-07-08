"""Parallel candidate evaluation with per-worker data caching."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.metrics import metrics_to_dict
from crypto_trader.backtest.runner import run
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations
from crypto_trader.optimize.scoring import composite_score
from crypto_trader.optimize.types import Experiment, ScoredCandidate

log = structlog.get_logger("optimize.parallel")

# ── Worker process globals (set once per worker via _init_worker) ──────

_worker_stores: dict[tuple[str, tuple[str, ...], tuple[str, ...]], _CachedStore] = {}
_worker_store: _CachedStore | None = None  # compatibility alias for tests/helpers


class _CachedStore:
    """In-memory store that pre-loads all DataFrames from a real store.

    Implements the same load_candles/load_funding interface as ParquetStore,
    backed by a dict instead of disk I/O.  HistoricalFeed._load_all() calls
    store.load_candles(coin, interval) → returns cached DataFrame.
    The date-range filtering in _load_all() creates a copy, so the cache
    is never mutated.
    """

    def __init__(self, store, symbols: list[str], timeframes: list[str]):
        self._candles: dict[tuple[str, str], Any] = {}
        self._funding: dict[str, Any] = {}
        for sym in symbols:
            for tf in timeframes:
                self._candles[(sym, tf)] = store.load_candles(sym, tf)
            self._funding[sym] = store.load_funding(sym)

    def load_candles(self, coin: str, interval: str):
        return self._candles.get((coin, interval))

    def load_funding(self, coin: str):
        return self._funding.get(coin)


def _cache_key(
    data_dir_str: str,
    symbols: list[str] | tuple[str, ...],
    timeframes: list[str] | tuple[str, ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return (
        str(Path(data_dir_str).resolve()),
        tuple(str(symbol).upper() for symbol in symbols),
        tuple(str(timeframe) for timeframe in timeframes),
    )


def _init_worker(data_dir_str: str, symbols: list[str], timeframes: list[str]) -> None:
    """Called once per worker process.  Pre-loads all data into _CachedStore."""
    global _worker_store

    # Workers don't inherit structlog config on Windows spawn — configure here.
    # WARNING level: main process already logs experiment progress at INFO.
    # Only reconfigure when actually in a worker process (not sequential fallback).
    import multiprocessing
    if multiprocessing.current_process().name != "MainProcess":
        import structlog as _structlog
        _structlog.configure(
            processors=[
                _structlog.processors.add_log_level,
                _structlog.processors.TimeStamper(fmt="iso"),
                _structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=_structlog.make_filtering_bound_logger(30),  # WARNING
        )

    from crypto_trader.data.store import ParquetStore

    key = _cache_key(data_dir_str, symbols, timeframes)
    if key not in _worker_stores:
        real_store = ParquetStore(base_dir=Path(data_dir_str))
        _worker_stores[key] = _CachedStore(real_store, symbols, timeframes)
    _worker_store = _worker_stores[key]


def _deserialize_config(config_dict: dict, strategy_type: str):
    """Deserialize config dict to the appropriate config class."""
    if strategy_type == "trend":
        from crypto_trader.strategy.trend.config import TrendConfig
        return TrendConfig.from_dict(config_dict)
    elif strategy_type == "breakout":
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        return BreakoutConfig.from_dict(config_dict)
    else:
        from crypto_trader.strategy.momentum.config import MomentumConfig
        return MomentumConfig.from_dict(config_dict)


def _evaluate_single(args: tuple) -> tuple[int, ScoredCandidate]:
    """Evaluate one candidate in a worker process.

    Module-level function (not a closure) for Windows ``spawn`` picklability.
    """
    (
        idx,
        exp_name,
        exp_mutations,
        merged_mutations,
        base_config_dict,
        bt_config_dict,
        scoring_weights,
        hard_rejects,
        strategy_type,
        ceilings,
        cache_key,
    ) = args

    try:
        store = _worker_stores.get(cache_key)
        if store is None:
            raise RuntimeError(f"worker cache is not initialized for {cache_key!r}")
        base_config = _deserialize_config(base_config_dict, strategy_type)
        config = apply_mutations(base_config, merged_mutations)

        bt_config = BacktestConfig(**bt_config_dict)
        bt_result = run(config, bt_config, store=store, strategy_type=strategy_type)
        metrics = metrics_to_dict(bt_result.metrics)
        score, rejected, reason = composite_score(
            metrics, scoring_weights, hard_rejects, ceilings=ceilings,
        )

        return idx, ScoredCandidate(
            experiment=Experiment(name=exp_name, mutations=exp_mutations),
            score=score,
            metrics=metrics,
            rejected=rejected,
            reject_reason=reason,
        )
    except Exception as e:
        return idx, ScoredCandidate(
            experiment=Experiment(name=exp_name, mutations=exp_mutations),
            score=0.0,
            metrics={},
            rejected=True,
            reject_reason=f"Exception: {e}",
        )


def _bt_config_to_dict(bt: BacktestConfig) -> dict:
    """Serialize BacktestConfig to a picklable dict."""
    return {
        "symbols": bt.symbols,
        "start_date": bt.start_date,
        "end_date": bt.end_date,
        "initial_equity": bt.initial_equity,
        "taker_fee_bps": bt.taker_fee_bps,
        "maker_fee_bps": bt.maker_fee_bps,
        "slippage_bps": bt.slippage_bps,
        "spread_bps": bt.spread_bps,
        "train_pct": bt.train_pct,
        "apply_funding": bt.apply_funding,
        "warmup_days": bt.warmup_days,
    }


def _worker_exception_candidate(item: tuple, error: Exception) -> tuple[int, ScoredCandidate]:
    exp_name = item[1]
    exp_mutations = item[2]
    return (
        item[0],
        ScoredCandidate(
            experiment=Experiment(name=exp_name, mutations=exp_mutations),
            score=0.0,
            metrics={},
            rejected=True,
            reject_reason=f"Worker exception: {error}",
        ),
    )


def _retry_worker_errors(
    failed_items: list[tuple],
    *,
    data_dir: Path,
    symbols: list[str],
    timeframes: list[str],
    phase: int,
) -> list[tuple[int, ScoredCandidate]]:
    """Retry infrastructure failures in fresh one-candidate worker pools.

    ``_evaluate_single`` catches ordinary strategy/config exceptions inside the
    worker. Exceptions reaching ``future.result()`` are therefore executor-level
    failures such as a broken process pool. Retrying them in isolated workers
    prevents unrelated pending candidates from being scored as false rejects.
    """
    retry_results: list[tuple[int, ScoredCandidate]] = []
    for retry_idx, item in enumerate(failed_items, start=1):
        exp_name = item[1]
        log.info(
            "experiment.worker_retry",
            phase=phase,
            progress=f"{retry_idx}/{len(failed_items)}",
            name=exp_name,
            workers=1,
        )
        try:
            with ProcessPoolExecutor(
                max_workers=1,
                initializer=_init_worker,
                initargs=(str(data_dir), symbols, timeframes),
            ) as retry_pool:
                future = retry_pool.submit(_evaluate_single, item)
                idx, sc = future.result()
        except Exception as e:
            log.warning("experiment.worker_retry_failed", name=exp_name, error=str(e))
            retry_results.append(_worker_exception_candidate(item, e))
            continue

        log.info(
            "experiment.worker_retry_complete",
            phase=phase,
            progress=f"{retry_idx}/{len(failed_items)}",
            name=sc.experiment.name,
            score=f"{sc.score:.4f}",
            rejected=sc.rejected,
        )
        retry_results.append((idx, sc))

    return retry_results


def evaluate_parallel(
    candidates: list[Experiment],
    current_mutations: dict[str, Any],
    cumulative_mutations: dict[str, Any],
    base_config,
    backtest_config: BacktestConfig,
    data_dir: Path,
    scoring_weights: dict[str, float],
    hard_rejects: dict[str, tuple[str, float]],
    phase: int,
    max_workers: int | None = None,
    strategy_type: str = "momentum",
    ceilings: dict[str, float] | None = None,
) -> list[ScoredCandidate]:
    """Evaluate candidates in parallel using ProcessPoolExecutor.

    Falls back to sequential evaluation for max_workers=1 or single candidate.
    """
    if not candidates:
        return []

    # Determine worker count
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    base_config_dict = base_config.to_dict()
    bt_config_dict = _bt_config_to_dict(backtest_config)
    symbols = backtest_config.symbols or base_config.symbols
    if strategy_type == "trend":
        timeframes = ["15m", "1h", "1d"]
    elif strategy_type == "breakout":
        timeframes = ["30m", "4h"]
    else:
        timeframes = ["15m", "1h", "4h"]
    cache_key = _cache_key(str(data_dir), symbols, timeframes)

    # Build work items — base merge computed once (same for all candidates)
    base_merged = merge_mutations(cumulative_mutations, current_mutations)
    work_items = []
    for idx, experiment in enumerate(candidates):
        merged = merge_mutations(base_merged, experiment.mutations)
        work_items.append((
            idx,
            experiment.name,
            experiment.mutations,
            merged,
            base_config_dict,
            bt_config_dict,
            scoring_weights,
            hard_rejects,
            strategy_type,
            ceilings,
            cache_key,
        ))

    # Sequential fallback
    if max_workers <= 1 or len(candidates) == 1:
        log.info("evaluate.sequential", candidates=len(candidates), phase=phase)
        # Initialise worker store on first call (reuses across candidates & rounds)
        global _worker_store
        if cache_key not in _worker_stores:
            _init_worker(str(data_dir), symbols, timeframes)
        if cache_key in _worker_stores:
            _worker_store = _worker_stores[cache_key]
        results = []
        for item in work_items:
            log.info(
                "experiment.evaluate",
                phase=phase,
                progress=f"{item[0] + 1}/{len(candidates)}",
                name=item[1],
            )
            _, sc = _evaluate_single(item)
            results.append((item[0], sc))

        results.sort(key=lambda x: x[0])
        return [sc for _, sc in results]

    # Parallel execution
    log.info(
        "evaluate.parallel",
        candidates=len(candidates),
        workers=max_workers,
        phase=phase,
    )

    results: list[tuple[int, ScoredCandidate]] = []
    failed_items: list[tuple] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(data_dir), symbols, timeframes),
    ) as pool:
        futures = {pool.submit(_evaluate_single, item): item for item in work_items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                idx, sc = future.result()
                log.info(
                    "experiment.complete",
                    phase=phase,
                    progress=f"{len(results) + 1}/{len(candidates)}",
                    name=sc.experiment.name,
                    score=f"{sc.score:.4f}",
                    rejected=sc.rejected,
                )
                results.append((idx, sc))
            except Exception as e:
                exp_name = item[1]
                log.warning("experiment.worker_error", name=exp_name, error=str(e))
                failed_items.append(item)

    if failed_items:
        results.extend(
            _retry_worker_errors(
                failed_items,
                data_dir=Path(data_dir),
                symbols=list(symbols),
                timeframes=list(timeframes),
                phase=phase,
            )
        )

    results.sort(key=lambda x: x[0])
    return [sc for _, sc in results]
