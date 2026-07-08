from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .phase_state import PhaseState
from .types import Experiment, GreedyResult, GreedyRound, ScoredCandidate

DEFAULT_AUTO_WORKERS = 3
DEFAULT_POOL_HEARTBEAT_SECONDS = 150.0
DEFAULT_POOL_TIMEOUT_PER_CANDIDATE_SECONDS = 300.0
DEFAULT_POOL_MIN_TIMEOUT_SECONDS = 300.0
DEFAULT_POOL_PROGRESS_MILESTONES = 10
MIN_POOL_PROGRESS_STEP = 5
_FALLBACK_EXCEPTIONS = (
    PermissionError,
    OSError,
    EOFError,
    BrokenPipeError,
    RuntimeError,
    TimeoutError,
)


def deserialize_experiments(raw: list[dict[str, Any]] | None) -> list[Experiment]:
    experiments: list[Experiment] = []
    for item in raw or []:
        if isinstance(item, Experiment):
            experiments.append(item)
        elif isinstance(item, dict):
            experiments.append(Experiment(name=item.get("name", ""), mutations=item.get("mutations", {})))
    return experiments


def greedy_result_to_dict(greedy_result: GreedyResult) -> dict[str, Any]:
    return {
        "base_score": greedy_result.base_score,
        "final_score": greedy_result.final_score,
        "accepted_count": greedy_result.accepted_count,
        "total_candidates": greedy_result.total_candidates,
        "kept_features": greedy_result.kept_features,
        "rounds": [asdict(round_result) for round_result in greedy_result.rounds],
        "final_metrics": greedy_result.final_metrics,
    }


def greedy_result_from_state(
    state: PhaseState,
    *,
    phase: int,
    final_metrics: dict[str, float],
) -> GreedyResult:
    phase_result = state.phase_results.get(phase, {})
    rounds = [GreedyRound(**round_result) for round_result in phase_result.get("rounds", [])]
    total_candidates = phase_result.get("total_candidates")
    if total_candidates is None and rounds:
        total_candidates = max(round_result.candidates_tested for round_result in rounds)
    if total_candidates is None:
        total_candidates = 0
    return GreedyResult(
        base_score=float(phase_result.get("base_score", 0.0)),
        final_score=float(phase_result.get("final_score", 0.0)),
        final_mutations=dict(phase_result.get("final_mutations", state.cumulative_mutations)),
        kept_features=list(phase_result.get("kept_features", [])),
        rounds=rounds,
        final_metrics=final_metrics,
        total_candidates=int(total_candidates),
        accepted_count=int(phase_result.get("accepted_count", len(phase_result.get("kept_features", [])))),
        elapsed_seconds=float(phase_result.get("elapsed_seconds", 0.0)),
    )


def seen_experiment_names(state: PhaseState) -> set[str]:
    seen: set[str] = set()
    for phase_result in state.phase_results.values():
        seen.update(name for name in phase_result.get("kept_features", []) if name)
        for item in phase_result.get("suggested_experiments", []):
            if isinstance(item, Experiment):
                if item.name:
                    seen.add(item.name)
            elif isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    seen.add(name)
    return seen


def resolve_worker_processes(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))

    cpu_count = max(1, os.cpu_count() or DEFAULT_AUTO_WORKERS)
    return min(DEFAULT_AUTO_WORKERS, cpu_count)


def create_process_pool(
    max_workers: int | None,
    *,
    initializer=None,
    initargs: tuple[Any, ...] = (),
    logger: logging.Logger | None = None,
    description: str = "parallel batch",
) -> mp.pool.Pool:
    processes = resolve_worker_processes(max_workers)
    context = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
    if logger is not None:
        logger.info(
            "Starting %s pool with %d worker(s) via %s.",
            description,
            processes,
            context.get_start_method(),
        )
    return context.Pool(
        processes=processes,
        initializer=initializer,
        initargs=tuple(initargs),
    )


def shutdown_process_pool(pool: mp.pool.Pool | None, *, force: bool = False) -> None:
    if pool is None:
        return
    try:
        if force:
            pool.terminate()
        else:
            pool.close()
        pool.join()
    except Exception:
        if not force:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass


def pool_map_with_heartbeat(
    pool: mp.pool.Pool,
    worker_fn,
    args: list[Any],
    *,
    description: str = "parallel batch",
    logger: logging.Logger | None = None,
    heartbeat_seconds: float = DEFAULT_POOL_HEARTBEAT_SECONDS,
    per_candidate_timeout_seconds: float = DEFAULT_POOL_TIMEOUT_PER_CANDIDATE_SECONDS,
    minimum_timeout_seconds: float = DEFAULT_POOL_MIN_TIMEOUT_SECONDS,
    chunksize: int | None = None,
):
    if not args:
        return []

    pending = {
        index: pool.apply_async(worker_fn, (arg,))
        for index, arg in enumerate(args)
    }
    results: list[Any | None] = [None] * len(args)
    total = len(args)
    timeout_seconds = max(
        float(minimum_timeout_seconds),
        total * float(per_candidate_timeout_seconds),
    )
    poll_interval = min(max(float(heartbeat_seconds) / 3.0, 1.0), 5.0)
    next_heartbeat = time.monotonic() + max(1.0, float(heartbeat_seconds))
    started_at = time.monotonic()
    completed = 0
    progress_step = _progress_step(total)
    next_progress_at = progress_step

    while completed < total:
        elapsed = time.monotonic() - started_at
        total_workers, alive_workers, dead_workers = _pool_worker_snapshot(pool)
        if dead_workers:
            worker_summary = ", ".join(
                f"pid={pid} exitcode={exitcode}" for pid, exitcode in dead_workers
            )
            raise RuntimeError(f"{description} worker exited unexpectedly: {worker_summary}")
        if pending and total_workers > 0 and alive_workers <= 0:
            raise RuntimeError(f"{description} lost all worker processes before completing any pending jobs.")
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"{description} exceeded timeout after {elapsed:.0f}s "
                f"while evaluating {total} candidate(s)."
            )
        progressed = False
        for index, async_result in list(pending.items()):
            if not async_result.ready():
                continue
            results[index] = async_result.get()
            del pending[index]
            completed += 1
            progressed = True

        if logger is not None and progressed and (completed >= next_progress_at or completed == total):
            logger.info(
                "%s progress: %d/%d completed (%.0f%%, %.0fs elapsed%s).",
                description,
                completed,
                total,
                (completed / total) * 100.0 if total else 100.0,
                elapsed,
                _eta_suffix(elapsed, completed, total),
            )
            while next_progress_at <= completed:
                next_progress_at += progress_step
            next_heartbeat = time.monotonic() + max(1.0, float(heartbeat_seconds))
        elif logger is not None and time.monotonic() >= next_heartbeat:
            logger.info(
                "%s still running after %.0fs (%d/%d completed, %d/%d worker(s) alive%s).",
                description,
                elapsed,
                completed,
                total,
                alive_workers,
                total_workers or alive_workers,
                _eta_suffix(elapsed, completed, total),
            )
            next_heartbeat = time.monotonic() + max(1.0, float(heartbeat_seconds))
        if completed < total and not progressed:
            time.sleep(poll_interval)

    if any(result is None for result in results):
        raise RuntimeError(f"{description} completed with missing worker results.")
    return [result for result in results]


def mutation_signature(mutations: dict[str, Any]) -> str:
    encoded = json.dumps(
        mutations,
        sort_keys=True,
        default=_signature_default,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clone_scored_candidate(result: ScoredCandidate, *, name: str | None = None) -> ScoredCandidate:
    return ScoredCandidate(
        name=name or result.name,
        score=float(result.score),
        rejected=bool(result.rejected),
        reject_reason=str(result.reject_reason or ""),
        metrics=dict(result.metrics or {}),
    )


class CachedBatchEvaluator:
    def __init__(
        self,
        delegate,
        *,
        cache: dict[str, ScoredCandidate] | None = None,
        seed_results: dict[str, ScoredCandidate] | None = None,
        signature_prefix: str = "",
        metrics_cache: dict[str, dict[str, float]] | None = None,
        max_batch_size: int | None = None,
    ):
        self._delegate = delegate
        self._cache = cache if cache is not None else {}
        self._signature_prefix = signature_prefix
        self._metrics_cache = metrics_cache
        self._max_batch_size = max_batch_size if max_batch_size and max_batch_size > 0 else None
        self._progress_callback = None
        for key, result in (seed_results or {}).items():
            self._cache[self._cache_key(key)] = clone_scored_candidate(result)
            if self._metrics_cache is not None and result.metrics:
                self._metrics_cache[key] = dict(result.metrics)

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        if not candidates:
            return []

        results: list[ScoredCandidate | None] = [None] * len(candidates)
        pending_candidates: list[Experiment] = []
        pending_keys: list[str] = []
        key_to_indexes: dict[str, list[int]] = {}

        for index, candidate in enumerate(candidates):
            merged = dict(current_mutations)
            merged.update(candidate.mutations)
            key = mutation_signature(merged)
            cached = self._cache.get(self._cache_key(key))
            if cached is not None:
                results[index] = clone_scored_candidate(cached, name=candidate.name)
                continue
            key_to_indexes.setdefault(key, []).append(index)
            if len(key_to_indexes[key]) == 1:
                pending_keys.append(key)
                pending_candidates.append(candidate)

        cached_count = len(candidates) - len(pending_candidates)
        self._emit_progress(
            {
                "event": "batch_start",
                "total_candidates": len(candidates),
                "pending_candidates": len(pending_candidates),
                "cached_candidates": cached_count,
                "completed_candidates": cached_count,
                "batch_size": self._max_batch_size or len(pending_candidates) or len(candidates),
            }
        )

        if pending_candidates:
            batch_size = self._max_batch_size or len(pending_candidates)
            total_chunks = int(math.ceil(len(pending_candidates) / batch_size))
            completed_pending = 0
            for chunk_index, start in enumerate(range(0, len(pending_candidates), batch_size), start=1):
                chunk_candidates = pending_candidates[start:start + batch_size]
                chunk_keys = pending_keys[start:start + batch_size]
                self._emit_progress(
                    {
                        "event": "chunk_start",
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "chunk_size": len(chunk_candidates),
                        "chunk_candidate_names": [candidate.name for candidate in chunk_candidates],
                        "total_candidates": len(candidates),
                        "pending_candidates": len(pending_candidates),
                        "cached_candidates": cached_count,
                        "completed_candidates": cached_count + completed_pending,
                    }
                )
                scored = self._delegate(chunk_candidates, current_mutations)
                if len(scored) != len(chunk_candidates):
                    raise RuntimeError(
                        f"Batch evaluator returned {len(scored)} results for {len(chunk_candidates)} candidates."
                    )
                for key, prototype, result in zip(chunk_keys, chunk_candidates, scored, strict=True):
                    template = clone_scored_candidate(result, name=prototype.name)
                    self._cache[self._cache_key(key)] = template
                    if self._metrics_cache is not None and result.metrics:
                        self._metrics_cache[key] = dict(result.metrics)
                    for index in key_to_indexes.get(key, []):
                        results[index] = clone_scored_candidate(template, name=candidates[index].name)
                completed_pending += len(chunk_candidates)
                rejected_count = sum(1 for item in scored if item.rejected)
                self._emit_progress(
                    {
                        "event": "chunk_complete",
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "chunk_size": len(chunk_candidates),
                        "chunk_candidate_names": [candidate.name for candidate in chunk_candidates],
                        "rejected_count": rejected_count,
                        "valid_count": len(chunk_candidates) - rejected_count,
                        "total_candidates": len(candidates),
                        "pending_candidates": len(pending_candidates),
                        "cached_candidates": cached_count,
                        "completed_candidates": cached_count + completed_pending,
                    }
                )

        self._emit_progress(
            {
                "event": "batch_complete",
                "total_candidates": len(candidates),
                "pending_candidates": len(pending_candidates),
                "cached_candidates": cached_count,
                "completed_candidates": len(candidates),
            }
        )

        return [result for result in results if result is not None]

    def close(self) -> None:
        close_fn = getattr(self._delegate, "close", None)
        if callable(close_fn):
            close_fn()

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback
        set_delegate_callback = getattr(self._delegate, "set_progress_callback", None)
        if callable(set_delegate_callback):
            set_delegate_callback(callback)

    def _cache_key(self, mutation_key: str) -> str:
        if not self._signature_prefix:
            return mutation_key
        return f"{self._signature_prefix}:{mutation_key}"

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if callable(self._progress_callback):
            self._progress_callback(payload)


class ResilientBatchEvaluator:
    def __init__(
        self,
        preferred_factory: Callable[[], Any],
        fallback_factory: Callable[[], Any],
        *,
        description: str = "batch evaluator",
        logger: logging.Logger | None = None,
        retryable_exceptions: tuple[type[BaseException], ...] = _FALLBACK_EXCEPTIONS,
    ):
        self._preferred_factory = preferred_factory
        self._fallback_factory = fallback_factory
        self._description = description
        self._logger = logger or logging.getLogger(__name__)
        self._retryable_exceptions = retryable_exceptions
        self._using_fallback = False
        self._delegate = self._build_delegate()

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        try:
            return self._delegate(candidates, current_mutations)
        except self._retryable_exceptions as exc:
            self._switch_to_fallback(exc)
            return self._delegate(candidates, current_mutations)

    def close(self) -> None:
        self._close_delegate(force=False)

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def _build_delegate(self):
        try:
            return self._preferred_factory()
        except self._retryable_exceptions as exc:
            self._using_fallback = True
            self._logger.warning(
                "%s initialization failed; falling back to local execution: %s",
                self._description,
                exc,
            )
            return self._fallback_factory()

    def _switch_to_fallback(self, exc: BaseException) -> None:
        if self._using_fallback:
            raise exc
        self._using_fallback = True
        self._logger.warning(
            "%s failed during parallel execution; retrying locally: %s",
            self._description,
            exc,
        )
        self._close_delegate(force=True)
        self._delegate = self._fallback_factory()

    def _close_delegate(self, *, force: bool) -> None:
        delegate = getattr(self, "_delegate", None)
        if delegate is None:
            return
        try:
            terminate_fn = getattr(delegate, "terminate", None)
            if force and callable(terminate_fn):
                terminate_fn()
                return
            close_fn = getattr(delegate, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            self._delegate = None


class SharedPoolBatchEvaluator:
    """Reusable pool-backed evaluator with heartbeat polling and safe fallback hooks."""

    def __init__(
        self,
        pool: mp.pool.Pool,
        *,
        worker_fn,
        build_args: Callable[[list[Experiment], dict[str, Any]], list[Any]],
        on_terminate: Callable[[], None] | None,
        on_close: Callable[[], None] | None = None,
        description: str,
        logger: logging.Logger | None = None,
        heartbeat_seconds: float = DEFAULT_POOL_HEARTBEAT_SECONDS,
        per_candidate_timeout_seconds: float = DEFAULT_POOL_TIMEOUT_PER_CANDIDATE_SECONDS,
        minimum_timeout_seconds: float = DEFAULT_POOL_MIN_TIMEOUT_SECONDS,
        chunksize: int | None = None,
    ):
        self._pool = pool
        self._worker_fn = worker_fn
        self._build_args = build_args
        self._on_terminate = on_terminate
        self._on_close = on_close
        self._description = description
        self._logger = logger
        self._heartbeat_seconds = heartbeat_seconds
        self._per_candidate_timeout_seconds = per_candidate_timeout_seconds
        self._minimum_timeout_seconds = minimum_timeout_seconds
        self._chunksize = chunksize

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        args = self._build_args(candidates, current_mutations)
        return pool_map_with_heartbeat(
            self._pool,
            self._worker_fn,
            args,
            description=self._description,
            logger=self._logger,
            heartbeat_seconds=self._heartbeat_seconds,
            per_candidate_timeout_seconds=self._per_candidate_timeout_seconds,
            minimum_timeout_seconds=self._minimum_timeout_seconds,
            chunksize=self._chunksize,
        )

    def close(self) -> None:
        if callable(self._on_close):
            self._on_close()

    def terminate(self) -> None:
        if callable(self._on_terminate):
            self._on_terminate()


def _pool_worker_snapshot(pool: mp.pool.Pool) -> tuple[int, int, list[tuple[int | None, int]]]:
    workers = list(getattr(pool, "_pool", []) or [])
    alive_workers = 0
    dead_workers: list[tuple[int | None, int]] = []
    for worker in workers:
        is_alive = False
        try:
            is_alive = bool(worker.is_alive())
        except Exception:
            is_alive = False
        if is_alive:
            alive_workers += 1
            continue
        exitcode = getattr(worker, "exitcode", None)
        if exitcode not in (None, 0):
            dead_workers.append((getattr(worker, "pid", None), int(exitcode)))
    return len(workers), alive_workers, dead_workers


def _signature_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _progress_step(total: int) -> int:
    return max(MIN_POOL_PROGRESS_STEP, int(math.ceil(total / DEFAULT_POOL_PROGRESS_MILESTONES)))


def _eta_suffix(elapsed_seconds: float, completed: int, total: int) -> str:
    if completed <= 0 or completed >= total:
        return ""
    avg_seconds = elapsed_seconds / completed
    remaining_seconds = avg_seconds * (total - completed)
    return f", ETA ~{remaining_seconds:.0f}s"
