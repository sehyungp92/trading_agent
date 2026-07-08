from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
import sys
import time as _time
from pathlib import Path
from typing import Any

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.provenance import AutoRunProvenance, build_phase_auto_provenance
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    greedy_result_from_state,
    mutation_signature,
    resolve_worker_processes,
    shutdown_process_pool,
)
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate

from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .phase_scoring import (
    IMMUTABLE_SCORING_WEIGHTS,
    PHASE_SCORING_WEIGHTS,
    merge_alcb_metrics,
    score_alcb_phase,
)
from .time_utils import hydrate_time_mutations

logger = logging.getLogger(__name__)

_ALCB_HEARTBEAT_SECONDS = 150.0
_ALCB_PER_CANDIDATE_TIMEOUT_SECONDS = 450.0
_ALCB_MINIMUM_TIMEOUT_SECONDS = 3600.0
_ALCB_MAX_EVAL_BATCH_SIZE = 2


_COMMON_HARD_REJECTS = {
    "min_expected_total_r": 118.0,
    "min_net_profit": 9000.0,
    "min_trades_per_month": 19.4,
    "min_pf": 1.85,
    "min_expectancy_dollar": 17.00,
    "max_dd_pct": 0.055,
}

PHASE_STATIC_HARD_REJECTS: dict[int, dict[str, float]] = {
    phase: dict(_COMMON_HARD_REJECTS)
    for phase in PHASE_FOCUS
}
PHASE_STATIC_HARD_REJECTS[2].update({
    "min_mfe_capture_efficiency": 0.74,
    "min_flow_mfe_exit_inverse": 0.76,
    "min_short_hold_24_drag_inverse": 0.32,
})
PHASE_STATIC_HARD_REJECTS[3].update({
    "min_mfe_capture_efficiency": 0.74,
    "min_flow_mfe_exit_inverse": 0.76,
    "min_short_hold_24_drag_inverse": 0.32,
})
PHASE_STATIC_HARD_REJECTS[4].update({
    "min_late_entry_quality": 0.50,
})
PHASE_STATIC_HARD_REJECTS[5].update({
    "min_mfe_capture_efficiency": 0.70,
    "min_flow_mfe_exit_inverse": 0.65,
    "min_short_hold_24_drag_inverse": 0.15,
})
PHASE_STATIC_HARD_REJECTS[6].update({
    "min_mfe_capture_efficiency": 0.70,
    "min_flow_mfe_exit_inverse": 0.65,
    "min_short_hold_24_drag_inverse": 0.15,
})
PHASE_STATIC_HARD_REJECTS[7].update({
    "min_mfe_capture_efficiency": 0.74,
    "min_flow_mfe_exit_inverse": 0.76,
    "min_short_hold_24_drag_inverse": 0.32,
})
PHASE_STATIC_HARD_REJECTS[8].update({
    "min_mfe_capture_efficiency": 0.72,
    "min_short_hold_24_drag_inverse": 0.30,
    "min_signal_quality": 0.55,
    "min_timing_quality": 0.55,
})

PHASE_MAX_ROUNDS = {1: 7, 2: 8, 3: 8, 4: 6, 5: 6, 6: 7, 7: 7, 8: 6}
PHASE_PRUNE_THRESHOLDS = {1: 0.035, 2: 0.035, 3: 0.035, 4: 0.030, 5: 0.030, 6: 0.025, 7: 0.030, 8: 0.025}
PHASE_MIN_EFFECTIVE_DELTA = {1: 0.0012, 2: 0.0012, 3: 0.0012, 4: 0.0010, 5: 0.0010, 6: 0.0008, 7: 0.0010, 8: 0.0008}

PHASE_GATE_RATIOS: dict[int, dict[str, float]] = {
    1: {"expected_total_r": 0.94, "net_profit": 0.93, "trades_per_month": 0.90, "profit_factor": 0.94, "expectancy_dollar": 0.90},
    2: {"expected_total_r": 0.95, "net_profit": 0.94, "trades_per_month": 0.92, "profit_factor": 0.94, "expectancy_dollar": 0.90},
    3: {"expected_total_r": 0.94, "net_profit": 0.92, "trades_per_month": 0.88, "profit_factor": 0.92, "expectancy_dollar": 0.86},
    4: {"expected_total_r": 0.95, "net_profit": 0.94, "trades_per_month": 0.98, "profit_factor": 0.92, "expectancy_dollar": 0.88},
    5: {"expected_total_r": 0.94, "net_profit": 0.92, "trades_per_month": 0.88, "profit_factor": 0.92, "expectancy_dollar": 0.86},
    6: {"expected_total_r": 0.97, "net_profit": 0.95, "trades_per_month": 0.95, "profit_factor": 0.95, "expectancy_dollar": 0.92},
    7: {"expected_total_r": 0.97, "net_profit": 0.95, "trades_per_month": 0.95, "profit_factor": 0.95, "expectancy_dollar": 0.92},
    8: {"expected_total_r": 0.97, "net_profit": 0.95, "trades_per_month": 0.98, "profit_factor": 0.95, "expectancy_dollar": 0.92},
}

ULTIMATE_TARGETS = {
    "expectancy_dollar": 22.0,
    "expected_total_r": 140.0,
    "trades_per_month": 21.5,
    "profit_factor": 2.18,
    "signal_quality": 0.66,
    "timing_quality": 0.64,
    "profit_protection": 0.72,
    "short_hold_24_drag_inverse": 0.48,
    "flow_mfe_exit_inverse": 0.86,
    "mfe_capture_efficiency": 0.82,
    "sizing_alignment": 1.0,
    "inv_dd": 0.78,
}


class _LocalBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        start_date: str,
        end_date: str,
        initial_equity: float,
        phase: int,
        hard_rejects: dict[str, float] | None,
        scoring_weights: dict[str, float] | None,
    ):
        from .worker import init_worker

        init_worker(
            str(data_dir),
            start_date,
            end_date,
            initial_equity,
            phase,
            hard_rejects,
            scoring_weights,
        )
        self._phase = phase
        self._hard_rejects = hard_rejects or {}
        self._scoring_weights = scoring_weights or {}

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        return [
            score_candidate((
                candidate.name, candidate.mutations, current_mutations,
                self._phase, self._hard_rejects, self._scoring_weights,
            ))
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


class _ThreadBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        start_date: str,
        end_date: str,
        initial_equity: float,
        phase: int,
        hard_rejects: dict[str, float] | None,
        scoring_weights: dict[str, float] | None,
        *,
        max_workers: int | None,
        logger: logging.Logger | None = None,
        description: str = "thread batch",
        heartbeat_seconds: float = 150.0,
        per_candidate_timeout_seconds: float = 300.0,
        minimum_timeout_seconds: float = 300.0,
    ):
        from .worker import init_worker

        init_worker(
            str(data_dir),
            start_date,
            end_date,
            initial_equity,
            phase,
            hard_rejects,
            scoring_weights,
        )
        self._phase = phase
        self._hard_rejects = hard_rejects or {}
        self._scoring_weights = scoring_weights or {}
        self._max_workers = resolve_worker_processes(max_workers)
        self._logger = logger
        self._description = description
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._per_candidate_timeout_seconds = float(per_candidate_timeout_seconds)
        self._minimum_timeout_seconds = float(minimum_timeout_seconds)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=f"alcb-phase-{phase}",
        )

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        from .worker import score_candidate

        if not candidates:
            return []

        args = [
            (
                candidate.name,
                candidate.mutations,
                current_mutations,
                self._phase,
                self._hard_rejects,
                self._scoring_weights,
            )
            for candidate in candidates
        ]
        results: list[ScoredCandidate | None] = [None] * len(args)
        futures = {
            self._executor.submit(score_candidate, arg): index
            for index, arg in enumerate(args)
        }
        total = len(args)
        completed = 0
        started_at = _time.monotonic()
        timeout_seconds = max(
            self._minimum_timeout_seconds,
            total * self._per_candidate_timeout_seconds,
        )
        poll_interval = min(max(self._heartbeat_seconds / 3.0, 1.0), 5.0)
        next_heartbeat = started_at + max(1.0, self._heartbeat_seconds)
        progress_step = max(5, (total + 9) // 10)
        next_progress_at = progress_step

        try:
            while futures:
                elapsed = _time.monotonic() - started_at
                if elapsed >= timeout_seconds:
                    raise TimeoutError(
                        f"{self._description} exceeded timeout after {elapsed:.0f}s "
                        f"while evaluating {total} candidate(s)."
                    )

                done, _ = concurrent.futures.wait(
                    tuple(futures),
                    timeout=poll_interval,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                progressed = False
                for future in done:
                    index = futures.pop(future)
                    results[index] = future.result()
                    completed += 1
                    progressed = True

                if self._logger is not None and progressed and (completed >= next_progress_at or completed == total):
                    self._logger.info(
                        "%s progress: %d/%d completed (%.0f%%, %.0fs elapsed%s).",
                        self._description,
                        completed,
                        total,
                        (completed / total) * 100.0 if total else 100.0,
                        elapsed,
                        self._eta_suffix(elapsed, completed, total),
                    )
                    while next_progress_at <= completed:
                        next_progress_at += progress_step
                    next_heartbeat = _time.monotonic() + max(1.0, self._heartbeat_seconds)
                elif self._logger is not None and _time.monotonic() >= next_heartbeat:
                    alive_threads = min(len(futures), self._max_workers)
                    self._logger.info(
                        "%s still running after %.0fs (%d/%d completed, %d/%d worker thread(s) active%s).",
                        self._description,
                        elapsed,
                        completed,
                        total,
                        alive_threads,
                        self._max_workers,
                        self._eta_suffix(elapsed, completed, total),
                    )
                    next_heartbeat = _time.monotonic() + max(1.0, self._heartbeat_seconds)
        except Exception:
            for future in futures:
                future.cancel()
            raise

        if any(result is None for result in results):
            raise RuntimeError(f"{self._description} completed with missing thread results.")
        return [result for result in results if result is not None]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _eta_suffix(elapsed_seconds: float, completed: int, total: int) -> str:
        if completed <= 0 or completed >= total:
            return ""
        avg_seconds = elapsed_seconds / completed
        remaining_seconds = avg_seconds * (total - completed)
        return f", ETA ~{remaining_seconds:.0f}s"


class ALCBP16Plugin:
    name = "alcb"
    num_phases = len(PHASE_FOCUS)
    initial_mutations = dict(BASE_MUTATIONS)
    ultimate_targets = ULTIMATE_TARGETS
    immutable_scoring_weights = dict(IMMUTABLE_SCORING_WEIGHTS)

    def __init__(
        self,
        data_dir: Path,
        *,
        start_date: str = "2024-01-01",
        end_date: str = "2026-03-01",
        initial_equity: float = 10_000.0,
        max_workers: int | None = 2,
        experiment_names: set[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.experiment_names = set(experiment_names or [])
        self._cached_bundle = None
        self._config_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        # Reserved for CachedBatchEvaluator's raw mutation_signature keys.
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._cache_source_fingerprint: str = ""
        self._last_context: dict[str, Any] = {}
        self._phase_runtime_context: dict[int, dict[str, Any]] = {}
        self._shared_pool: mp.Pool | None = None
        self._provenance: AutoRunProvenance | None = None

    def build_provenance(self) -> AutoRunProvenance:
        if self._provenance is None:
            repo_root = Path(__file__).resolve().parents[4]
            self._provenance = build_phase_auto_provenance(
                self.name,
                repo_root=repo_root,
                code_dirs=(
                    Path(__file__).resolve().parent,
                    repo_root / "strategies/stock/alcb",
                ),
                code_paths=(
                    repo_root / "backtests/stock/engine/alcb_engine.py",
                    repo_root / "backtests/stock/config_alcb.py",
                    repo_root / "backtests/stock/auto/config_mutator.py",
                    repo_root / "backtests/stock/data/replay_cache.py",
                    repo_root / "strategies/stock/alcb/core/logic.py",
                    repo_root / "strategies/stock/alcb/artifact_store.py",
                ),
                data_dir=self.data_dir,
                selection_context={
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "initial_equity": self.initial_equity,
                    "num_phases": self.num_phases,
                    "experiment_names": sorted(self.experiment_names),
                    "phase_scoring_weights": PHASE_SCORING_WEIGHTS,
                    "phase_hard_rejects": PHASE_STATIC_HARD_REJECTS,
                    "phase_focus": PHASE_FOCUS,
                    "ultimate_targets": ULTIMATE_TARGETS,
                    "round_baseline_policy": "run_spec.baseline_mutations",
                },
            )
        return self._provenance

    def _replay_bundle(self):
        from backtests.stock.data.replay_cache import load_research_replay_bundle

        bundle = load_research_replay_bundle(self.data_dir)
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._config_cache.clear()
            self._evaluation_cache.clear()
            self._last_context = {}
            self._phase_runtime_context.clear()
            self.close_pool()
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        return bundle

    def _replay_data_fingerprint(self) -> str:
        return self._replay_bundle().cache_source_fingerprint

    def _metrics_cache_key(
        self,
        mutations: dict[str, Any],
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        effective_start = start_date or self.start_date
        effective_end = end_date or self.end_date
        return build_cache_key(
            "alcb_p18_real_alpha.metrics",
            source_fingerprint=self._replay_data_fingerprint(),
            mutations=mutations,
            extra={
                "start_date": effective_start,
                "end_date": effective_end,
                "initial_equity": self.initial_equity,
            },
        )

    def _get_or_create_pool(self) -> mp.Pool:
        if self._shared_pool is None:
            from .worker import init_worker

            self._shared_pool = create_process_pool(
                self.max_workers,
                initializer=init_worker,
                initargs=(
                    str(self.data_dir),
                    self.start_date,
                    self.end_date,
                    self.initial_equity,
                ),
            )
        return self._shared_pool

    def close_pool(self) -> None:
        shutdown_process_pool(self._shared_pool)
        self._shared_pool = None

    def _destroy_pool(self) -> None:
        shutdown_process_pool(self._shared_pool, force=True)
        self._shared_pool = None

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        focus, focus_metrics = PHASE_FOCUS[phase]
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                experiment_filter=self.experiment_names or None,
            )
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics, current_phase=phase: self._gate_criteria(current_phase, metrics),
            scoring_weights=dict(PHASE_SCORING_WEIGHTS.get(phase, IMMUTABLE_SCORING_WEIGHTS)),
            hard_rejects=PHASE_STATIC_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=PHASE_MIN_EFFECTIVE_DELTA.get(phase, 0.001),
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=PHASE_MAX_ROUNDS.get(phase, len(candidates)),
            prune_threshold=PHASE_PRUNE_THRESHOLDS.get(phase, 0.04),
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        t0 = _time.time()
        base_metrics = self._run_config(cumulative_mutations, store_context=False)["metrics"]
        logger.info("Parent baseline evaluated in %.1fs", _time.time() - t0)
        resolved_hard_rejects = self._resolve_phase_hard_rejects(phase, base_metrics, hard_rejects or {})
        baseline_result = self._seed_result_for_metrics(
            "__baseline__",
            phase,
            base_metrics,
            resolved_hard_rejects,
            scoring_weights,
        )
        baseline_key = mutation_signature(cumulative_mutations)
        self._metrics_cache[baseline_key] = dict(base_metrics)
        self._phase_runtime_context[phase] = {
            "base_metrics": base_metrics,
            "hard_rejects": resolved_hard_rejects,
        }
        evaluation_key = build_cache_key(
            "alcb_p18_real_alpha.evaluation",
            source_fingerprint=self._replay_data_fingerprint(),
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": resolved_hard_rejects,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "initial_equity": self.initial_equity,
            },
        )

        def local_factory():
            return _LocalBatchEvaluator(
                self.data_dir,
                self.start_date,
                self.end_date,
                self.initial_equity,
                phase,
                resolved_hard_rejects,
                scoring_weights,
            )

        if self.max_workers == 1:
            evaluator = local_factory()
        elif sys.platform == "win32":
            logger.info(
                "Using Windows thread evaluator for %s phase %d with max_workers=%s.",
                self.name, phase, self.max_workers,
            )
            evaluator = _ThreadBatchEvaluator(
                self.data_dir,
                self.start_date,
                self.end_date,
                self.initial_equity,
                phase,
                resolved_hard_rejects,
                scoring_weights,
                max_workers=self.max_workers,
                logger=logger,
                description=f"{self.name} phase {phase}",
                heartbeat_seconds=_ALCB_HEARTBEAT_SECONDS,
                per_candidate_timeout_seconds=_ALCB_PER_CANDIDATE_TIMEOUT_SECONDS,
                minimum_timeout_seconds=_ALCB_MINIMUM_TIMEOUT_SECONDS,
            )
        elif not _supports_spawn():
            evaluator = local_factory()
        else:
            def pool_factory():
                pool = self._get_or_create_pool()
                from .worker import score_candidate

                return SharedPoolBatchEvaluator(
                    pool,
                    worker_fn=score_candidate,
                    build_args=lambda candidates, current_mutations: [
                        (
                            candidate.name,
                            candidate.mutations,
                            current_mutations,
                            phase,
                            resolved_hard_rejects,
                            scoring_weights,
                        )
                        for candidate in candidates
                    ],
                    on_terminate=self._destroy_pool,
                    description=f"{self.name} phase {phase}",
                    logger=logger,
                    heartbeat_seconds=_ALCB_HEARTBEAT_SECONDS,
                    per_candidate_timeout_seconds=_ALCB_PER_CANDIDATE_TIMEOUT_SECONDS,
                    minimum_timeout_seconds=_ALCB_MINIMUM_TIMEOUT_SECONDS,
                )

            evaluator = ResilientBatchEvaluator(
                preferred_factory=pool_factory,
                fallback_factory=local_factory,
                description=f"{self.name} phase {phase} evaluator",
                logger=logger,
            )
        return CachedBatchEvaluator(
            evaluator,
            cache=self._evaluation_cache,
            seed_results={baseline_key: baseline_result},
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
            max_batch_size=_ALCB_MAX_EVAL_BATCH_SIZE,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        metrics_key = self._metrics_cache_key(mutations)
        if self._last_context.get("metrics_cache_key") == metrics_key:
            return dict(self._last_context["metrics"])
        return self._run_config(mutations, store_context=True)["metrics"]

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        del state
        runtime = self._phase_runtime_context.get(phase, {})
        return _build_phase_snapshot(
            phase,
            PHASE_FOCUS[phase][0],
            metrics,
            greedy_result,
            base_metrics=runtime.get("base_metrics", {}),
            hard_rejects=runtime.get("hard_rejects", {}),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from backtests.stock.analysis.alcb_diagnostics import alcb_full_diagnostic
        from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis

        snapshot = self.run_phase_diagnostics(phase, state, metrics, greedy_result)
        expected_signature = mutation_signature(greedy_result.final_mutations)
        if self._last_context.get("mutation_signature") != expected_signature:
            self._run_config(greedy_result.final_mutations, store_context=True, collect_diagnostics=True)
        trades = self._last_context.get("trades")
        config = self._last_context.get("config")
        if not trades or config is None:
            return snapshot

        max_positions = int(config.param_overrides.get("max_positions", 10))
        return "\n\n".join([
            snapshot,
            alcb_full_diagnostic(
                trades,
                shadow_tracker=self._last_context.get("shadow_tracker"),
                daily_selections=self._last_context.get("daily_selections"),
            ),
            qe_replacement_analysis(trades, max_positions=max_positions),
        ])

    def _build_final_round_context(self, state: PhaseState) -> tuple[int, dict[str, Any], dict[str, float], Any]:
        final_phase = max(state.completed_phases) if state.completed_phases else self.num_phases
        final_ctx = self._run_config(state.cumulative_mutations, store_context=True, collect_diagnostics=True)
        final_metrics = final_ctx["metrics"]
        final_greedy = greedy_result_from_state(state, phase=final_phase, final_metrics=final_metrics)
        return final_phase, final_ctx, final_metrics, final_greedy

    def render_final_diagnostics_text(self, state: PhaseState) -> str:
        final_phase, _final_ctx, final_metrics, final_greedy = self._build_final_round_context(state)
        return self.run_enhanced_diagnostics(final_phase, state, final_metrics, final_greedy)

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        from backtests.stock.analysis.alcb_qe_replacement import qe_replacement_analysis

        final_phase, final_ctx, final_metrics, final_greedy = self._build_final_round_context(state)
        base_mutations = dict(getattr(self, "initial_mutations", BASE_MUTATIONS) or {})
        base_ctx = self._run_config(base_mutations, store_context=False, collect_diagnostics=True)
        base_metrics = base_ctx["metrics"]
        final_trades = final_ctx["trades"]
        final_diagnostics_text = self.run_enhanced_diagnostics(final_phase, state, final_metrics, final_greedy)

        dimension_reports = {
            "expected_return": "\n".join([
                f"Net profit {base_metrics['net_profit']:+.2f} -> {final_metrics['net_profit']:+.2f}.",
                f"Expectancy $ {base_metrics['expectancy_dollar']:+.2f} -> {final_metrics['expectancy_dollar']:+.2f}.",
                f"Expected total R {base_metrics['expected_total_r']:+.2f} -> {final_metrics['expected_total_r']:+.2f}.",
                f"Profit factor {base_metrics['profit_factor']:.3f} -> {final_metrics['profit_factor']:.3f}.",
            ]),
            "frequency": "\n".join([
                f"Total trades {int(base_metrics['total_trades'])} -> {int(final_metrics['total_trades'])}.",
                f"Trades/month {base_metrics['trades_per_month']:.2f} -> {final_metrics['trades_per_month']:.2f}.",
                f"Avg hold hours {base_metrics['avg_hold_hours']:.2f} -> {final_metrics['avg_hold_hours']:.2f}.",
            ]),
            "entry_shape": "\n".join([
                f"Signal quality {base_metrics.get('signal_quality', 0.0):.3f} -> {final_metrics.get('signal_quality', 0.0):.3f}.",
                f"Timing quality {base_metrics.get('timing_quality', 0.0):.3f} -> {final_metrics.get('timing_quality', 0.0):.3f}.",
                f"Extended AVWAP inverse {base_metrics.get('extended_avwap_inverse', 0.0):.3f} -> {final_metrics.get('extended_avwap_inverse', 0.0):.3f}.",
                f"Bar-9 inverse {base_metrics.get('bar9_inverse', 0.0):.3f} -> {final_metrics.get('bar9_inverse', 0.0):.3f}.",
                f"Late-entry quality {base_metrics.get('late_entry_quality', 0.0):.3f} -> {final_metrics.get('late_entry_quality', 0.0):.3f}.",
                f"Score monotonicity {base_metrics.get('score_monotonicity', 0.0):.3f} -> {final_metrics.get('score_monotonicity', 0.0):.3f}.",
            ]),
            "management": "\n".join([
                f"MFE capture eff {base_metrics.get('mfe_capture_efficiency', 0.0):.3f} -> {final_metrics.get('mfe_capture_efficiency', 0.0):.3f}.",
                f"Profit protection {base_metrics.get('profit_protection', 0.0):.3f} -> {final_metrics.get('profit_protection', 0.0):.3f}.",
                f"Short-hold <=24 bars R {base_metrics.get('short_hold_24_total_r', 0.0):+.2f} -> {final_metrics.get('short_hold_24_total_r', 0.0):+.2f}.",
                f"FLOW/MFE exit inverse {base_metrics.get('flow_mfe_exit_inverse', 0.0):.3f} -> {final_metrics.get('flow_mfe_exit_inverse', 0.0):.3f}.",
                f"Sizing alignment {base_metrics.get('sizing_alignment', 0.0):.3f} -> {final_metrics.get('sizing_alignment', 0.0):.3f}.",
                f"Long-hold capture {base_metrics.get('long_hold_capture', 0.0):.3f} -> {final_metrics.get('long_hold_capture', 0.0):.3f}.",
            ]),
            "risk": "\n".join([
                f"Max DD {base_metrics['max_drawdown_pct']:.1%} -> {final_metrics['max_drawdown_pct']:.1%}.",
                f"Sharpe {base_metrics['sharpe']:.2f} -> {final_metrics['sharpe']:.2f}.",
                f"Inv DD {base_metrics['inv_dd']:.3f} -> {final_metrics['inv_dd']:.3f}.",
            ]),
        }
        extra_sections = {
            "qe_replacement": qe_replacement_analysis(
                final_trades,
                max_positions=int(final_ctx["config"].param_overrides.get("max_positions", 10)),
            ),
        }
        overall_verdict = (
            f"Final {self.num_phases}-phase bundle finished at "
            f"expectancy ${final_metrics['expectancy_dollar']:+.2f}, expected_total_r {final_metrics['expected_total_r']:+.2f}, "
            f"{final_metrics['trades_per_month']:.2f} trades/month, PF {final_metrics['profit_factor']:.3f}, "
            f"signal_quality {final_metrics.get('signal_quality', 0.0):.3f}, "
            f"timing_quality {final_metrics.get('timing_quality', 0.0):.3f}, "
            f"and max DD {final_metrics['max_drawdown_pct']:.1%}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=overall_verdict,
            extra_sections=extra_sections,
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        base_metrics = self._phase_runtime_context.get(phase, {}).get("base_metrics", {})
        gaps: list[str] = []
        if phase == 1:
            if metrics.get("signal_quality", 0.0) < max(0.50, base_metrics.get("signal_quality", 0.0) * 0.92):
                gaps.append("Score-component sizing is not improving discrimination between stronger and weaker breakouts.")
            if metrics.get("score_monotonicity", 0.0) < max(0.42, base_metrics.get("score_monotonicity", 0.0) * 0.90):
                gaps.append("The completed-bar score shape is still not monotonic enough.")
            if metrics["expected_total_r"] < base_metrics.get("expected_total_r", 0.0) * 0.94:
                gaps.append("Signal discrimination is trimming too much total alpha.")
        elif phase == 2:
            if metrics.get("short_hold_24_drag_inverse", 0.0) < max(0.15, base_metrics.get("short_hold_24_drag_inverse", 0.0) * 0.80):
                gaps.append("The early-failure protective stop is not neutralizing the <=24-bar loser pocket enough.")
            if metrics.get("profit_protection", 0.0) < max(0.45, base_metrics.get("profit_protection", 0.0) * 0.85):
                gaps.append("Early-failure protection is not improving aggregate profit protection.")
            if metrics.get("mfe_capture_efficiency", 0.0) < max(0.70, base_metrics.get("mfe_capture_efficiency", 0.0) * 0.90):
                gaps.append("Early stop tightening is giving up too much winner MFE capture.")
        elif phase == 3:
            if metrics.get("flow_mfe_exit_inverse", 0.0) < max(0.65, base_metrics.get("flow_mfe_exit_inverse", 0.0) * 0.85):
                gaps.append("FLOW_REVERSAL and MFE_CONVICTION exits are still too negative.")
            if metrics.get("profit_protection", 0.0) < max(0.45, base_metrics.get("profit_protection", 0.0) * 0.85):
                gaps.append("Exit management still is not improving aggregate profit protection.")
            if metrics.get("long_hold_capture", 0.0) < max(0.75, base_metrics.get("long_hold_capture", 0.0) * 0.85):
                gaps.append("Profit-retention changes are clipping too much runner alpha.")
        elif phase == 4:
            if metrics.get("trades_per_month", 0.0) < base_metrics.get("trades_per_month", 0.0) * 0.98:
                gaps.append("The selective late-window extension is not recovering enough acceptable frequency.")
            if metrics.get("late_entry_quality", 0.0) < max(0.50, base_metrics.get("late_entry_quality", 0.0) * 0.80):
                gaps.append("The recovered late-window entries are not high quality enough.")
            if metrics["expected_total_r"] < base_metrics.get("expected_total_r", 0.0) * 0.95:
                gaps.append("The frequency-recovery phase is giving back too much total alpha.")
        elif phase == 5:
            if metrics.get("extended_avwap_inverse", 0.0) < max(0.30, base_metrics.get("extended_avwap_inverse", 0.0) * 0.90):
                gaps.append("Entry geometry still is not cleaning up extended entries enough.")
            if metrics.get("timing_quality", 0.0) < max(0.40, base_metrics.get("timing_quality", 0.0) * 0.88):
                gaps.append("Entry geometry changes are hurting timing quality.")
        elif phase == 6:
            if metrics["expected_total_r"] < base_metrics.get("expected_total_r", 0.0) * 0.97:
                gaps.append("Synthesis is not preserving the alpha gains from the earlier targeted fixes.")
            if metrics["trades_per_month"] < base_metrics.get("trades_per_month", 0.0) * 0.95:
                gaps.append("Synthesis is sacrificing too much throughput.")
            if metrics.get("sizing_alignment", 0.0) < max(0.80, base_metrics.get("sizing_alignment", 0.0) * 0.90):
                gaps.append("Position sizing still allocates too much risk to losing trades versus winners.")
        return gaps

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        del state, greedy_result
        base_metrics = self._phase_runtime_context.get(phase, {}).get("base_metrics", {})
        focus_metrics = PHASE_FOCUS[phase][1]
        deltas = {
            metric_name: {
                "base": float(base_metrics.get(metric_name, 0.0)),
                "final": float(metrics.get(metric_name, 0.0)),
                "delta": float(metrics.get(metric_name, 0.0)) - float(base_metrics.get(metric_name, 0.0)),
            }
            for metric_name in focus_metrics
        }
        core_metrics = [
            "net_profit",
            "expectancy_dollar",
            "expected_total_r",
            "trades_per_month",
            "profit_factor",
            "max_drawdown_pct",
            "signal_quality",
            "timing_quality",
            "profit_protection",
            "short_hold_24_drag_inverse",
            "flow_mfe_exit_inverse",
            "mfe_capture_efficiency",
            "sizing_alignment",
            "extended_avwap_inverse",
            "bar9_inverse",
            "late_entry_quality",
            "score_monotonicity",
        ]
        return {
            "focus_deltas": deltas,
            "core": {
                metric_name: float(metrics.get(metric_name, 0.0))
                for metric_name in core_metrics
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        focus_deltas = extra.get("focus_deltas", {})
        delta_line = ", ".join(
            f"{metric}={values['base']:.3f}->{values['final']:.3f} ({values['delta']:+.3f})"
            for metric, values in focus_deltas.items()
        )
        core = extra.get("core", {})
        core_line = ", ".join(
            [
                f"net_profit={core.get('net_profit', 0.0):+.2f}",
                f"expectancy_dollar={core.get('expectancy_dollar', 0.0):+.2f}",
                f"expected_total_r={core.get('expected_total_r', 0.0):+.2f}",
                f"trades_per_month={core.get('trades_per_month', 0.0):.2f}",
                f"profit_factor={core.get('profit_factor', 0.0):.3f}",
                f"max_drawdown_pct={core.get('max_drawdown_pct', 0.0):.2%}",
                f"signal_quality={core.get('signal_quality', 0.0):.3f}",
                f"timing_quality={core.get('timing_quality', 0.0):.3f}",
                f"profit_protection={core.get('profit_protection', 0.0):.3f}",
                f"short24_inv={core.get('short_hold_24_drag_inverse', 0.0):.3f}",
                f"flow_mfe_inv={core.get('flow_mfe_exit_inverse', 0.0):.3f}",
                f"mfe_capture_eff={core.get('mfe_capture_efficiency', 0.0):.3f}",
                f"sizing_alignment={core.get('sizing_alignment', 0.0):.3f}",
                f"extended_avwap_inverse={core.get('extended_avwap_inverse', 0.0):.3f}",
                f"bar9_inverse={core.get('bar9_inverse', 0.0):.3f}",
                f"late_entry_quality={core.get('late_entry_quality', 0.0):.3f}",
                f"score_monotonicity={core.get('score_monotonicity', 0.0):.3f}",
            ]
        )
        return [f"Focus deltas: {delta_line}", f"Core: {core_line}"]

    def _gate_criteria(self, phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
        base_metrics = self._phase_runtime_context.get(phase, {}).get("base_metrics", {})
        ratios = PHASE_GATE_RATIOS.get(phase, PHASE_GATE_RATIOS[max(PHASE_GATE_RATIOS)])
        dd_budget = self._drawdown_budget(phase, base_metrics)

        criteria = [
            self._min_gate(
                "expected_total_r",
                metrics["expected_total_r"],
                max(_COMMON_HARD_REJECTS["min_expected_total_r"], base_metrics.get("expected_total_r", 0.0) * ratios["expected_total_r"]),
            ),
            self._min_gate(
                "net_profit",
                metrics["net_profit"],
                max(_COMMON_HARD_REJECTS["min_net_profit"], base_metrics.get("net_profit", 0.0) * ratios["net_profit"]),
            ),
            self._min_gate(
                "trades_per_month",
                metrics["trades_per_month"],
                max(_COMMON_HARD_REJECTS["min_trades_per_month"], base_metrics.get("trades_per_month", 0.0) * ratios["trades_per_month"]),
            ),
            self._min_gate(
                "profit_factor",
                metrics["profit_factor"],
                max(_COMMON_HARD_REJECTS["min_pf"], base_metrics.get("profit_factor", 0.0) * ratios["profit_factor"]),
            ),
            self._min_gate(
                "expectancy_dollar",
                metrics["expectancy_dollar"],
                max(_COMMON_HARD_REJECTS["min_expectancy_dollar"], base_metrics.get("expectancy_dollar", 0.0) * ratios["expectancy_dollar"]),
            ),
            self._max_gate("max_drawdown_pct", metrics["max_drawdown_pct"], dd_budget),
        ]

        if phase == 1:
            criteria.extend([
                self._min_gate(
                    "signal_quality",
                    metrics.get("signal_quality", 0.0),
                    max(0.50, base_metrics.get("signal_quality", 0.0) * 0.92),
                ),
                self._min_gate(
                    "score_monotonicity",
                    metrics.get("score_monotonicity", 0.0),
                    max(0.42, base_metrics.get("score_monotonicity", 0.0) * 0.90),
                ),
                self._min_gate(
                    "sizing_alignment",
                    metrics.get("sizing_alignment", 0.0),
                    max(0.80, base_metrics.get("sizing_alignment", 0.0) * 0.90),
                ),
            ])
        elif phase in (2, 3, 7):
            criteria.extend([
                self._min_gate(
                    "profit_protection",
                    metrics.get("profit_protection", 0.0),
                    max(0.45, base_metrics.get("profit_protection", 0.0) * 0.85),
                ),
                self._min_gate(
                    "short_hold_24_drag_inverse",
                    metrics.get("short_hold_24_drag_inverse", 0.0),
                    max(0.15, base_metrics.get("short_hold_24_drag_inverse", 0.0) * 0.80),
                ),
                self._min_gate(
                    "mfe_capture_efficiency",
                    metrics.get("mfe_capture_efficiency", 0.0),
                    max(0.70, base_metrics.get("mfe_capture_efficiency", 0.0) * 0.90),
                ),
            ])
            if phase == 3:
                criteria.append(
                    self._min_gate(
                        "long_hold_capture",
                        metrics.get("long_hold_capture", 0.0),
                        max(0.75, base_metrics.get("long_hold_capture", 0.0) * 0.85),
                    )
                )
        elif phase == 4:
            criteria.extend([
                self._min_gate(
                    "late_entry_quality",
                    metrics.get("late_entry_quality", 0.0),
                    max(0.50, base_metrics.get("late_entry_quality", 0.0) * 0.80),
                ),
                self._min_gate(
                    "signal_quality",
                    metrics.get("signal_quality", 0.0),
                    max(0.50, base_metrics.get("signal_quality", 0.0) * 0.88),
                ),
            ])
        elif phase == 5:
            criteria.extend([
                self._min_gate(
                    "extended_avwap_inverse",
                    metrics.get("extended_avwap_inverse", 0.0),
                    max(0.30, base_metrics.get("extended_avwap_inverse", 0.0) * 0.90),
                ),
                self._min_gate(
                    "timing_quality",
                    metrics.get("timing_quality", 0.0),
                    max(0.40, base_metrics.get("timing_quality", 0.0) * 0.88),
                ),
            ])
        else:
            criteria.extend([
                self._min_gate(
                    "signal_quality",
                    metrics.get("signal_quality", 0.0),
                    max(0.50, base_metrics.get("signal_quality", 0.0) * 0.92),
                ),
                self._min_gate(
                    "sizing_alignment",
                    metrics.get("sizing_alignment", 0.0),
                    max(0.80, base_metrics.get("sizing_alignment", 0.0) * 0.90),
                ),
                self._min_gate(
                    "profit_protection",
                    metrics.get("profit_protection", 0.0),
                    max(0.45, base_metrics.get("profit_protection", 0.0) * 0.85),
                ),
            ])
        return criteria

    def _resolve_phase_hard_rejects(
        self,
        phase: int,
        base_metrics: dict[str, float],
        hard_rejects: dict[str, float],
    ) -> dict[str, float]:
        ratios = PHASE_GATE_RATIOS.get(phase, PHASE_GATE_RATIOS[max(PHASE_GATE_RATIOS)])
        resolved = dict(PHASE_STATIC_HARD_REJECTS.get(phase, {}))
        resolved.update(hard_rejects or {})

        self._raise_floor(resolved, "min_expected_total_r", base_metrics.get("expected_total_r", 0.0), ratios["expected_total_r"], minimum=_COMMON_HARD_REJECTS["min_expected_total_r"])
        self._raise_floor(resolved, "min_net_profit", base_metrics.get("net_profit", 0.0), ratios["net_profit"], minimum=_COMMON_HARD_REJECTS["min_net_profit"])
        self._raise_floor(resolved, "min_trades_per_month", base_metrics.get("trades_per_month", 0.0), ratios["trades_per_month"], minimum=_COMMON_HARD_REJECTS["min_trades_per_month"])
        self._raise_floor(resolved, "min_pf", base_metrics.get("profit_factor", 0.0), ratios["profit_factor"], minimum=_COMMON_HARD_REJECTS["min_pf"])
        self._raise_floor(resolved, "min_expectancy_dollar", base_metrics.get("expectancy_dollar", 0.0), ratios["expectancy_dollar"], minimum=_COMMON_HARD_REJECTS["min_expectancy_dollar"])

        if phase == 1:
            self._raise_floor(resolved, "min_signal_quality", base_metrics.get("signal_quality", 0.0), 0.92, minimum=0.50)
            self._raise_floor(resolved, "min_score_monotonicity", base_metrics.get("score_monotonicity", 0.0), 0.90, minimum=0.42)
            self._raise_floor(resolved, "min_sizing_alignment", base_metrics.get("sizing_alignment", 0.0), 0.90, minimum=0.80)
        if phase in (2, 3, 7):
            self._raise_floor(resolved, "min_profit_protection", base_metrics.get("profit_protection", 0.0), 0.85, minimum=0.45)
            self._raise_floor(resolved, "min_short_hold_24_drag_inverse", base_metrics.get("short_hold_24_drag_inverse", 0.0), 0.80, minimum=0.15)
            self._raise_floor(resolved, "min_flow_mfe_exit_inverse", base_metrics.get("flow_mfe_exit_inverse", 0.0), 0.85, minimum=0.65)
            self._raise_floor(resolved, "min_mfe_capture_efficiency", base_metrics.get("mfe_capture_efficiency", 0.0), 0.90, minimum=0.70)
        if phase == 3:
            self._raise_floor(resolved, "min_long_hold_capture", base_metrics.get("long_hold_capture", 0.0), 0.85, minimum=0.75)
        if phase == 4:
            self._raise_floor(resolved, "min_late_entry_quality", base_metrics.get("late_entry_quality", 0.0), 0.80, minimum=0.50)
            self._raise_floor(resolved, "min_signal_quality", base_metrics.get("signal_quality", 0.0), 0.88, minimum=0.50)
        if phase >= 5:
            self._raise_floor(resolved, "min_extended_avwap_inverse", base_metrics.get("extended_avwap_inverse", 0.0), 0.90, minimum=0.30)
            self._raise_floor(resolved, "min_timing_quality", base_metrics.get("timing_quality", 0.0), 0.88, minimum=0.40)
        if phase == 6:
            self._raise_floor(resolved, "min_profit_protection", base_metrics.get("profit_protection", 0.0), 0.85, minimum=0.45)
            self._raise_floor(resolved, "min_short_hold_24_drag_inverse", base_metrics.get("short_hold_24_drag_inverse", 0.0), 0.80, minimum=0.15)
            self._raise_floor(resolved, "min_flow_mfe_exit_inverse", base_metrics.get("flow_mfe_exit_inverse", 0.0), 0.85, minimum=0.65)
            self._raise_floor(resolved, "min_mfe_capture_efficiency", base_metrics.get("mfe_capture_efficiency", 0.0), 0.90, minimum=0.70)
            self._raise_floor(resolved, "min_signal_quality", base_metrics.get("signal_quality", 0.0), 0.92, minimum=0.50)
            self._raise_floor(resolved, "min_sizing_alignment", base_metrics.get("sizing_alignment", 0.0), 0.90, minimum=0.80)

        resolved["max_dd_pct"] = min(
            float(resolved.get("max_dd_pct", _COMMON_HARD_REJECTS["max_dd_pct"])),
            self._drawdown_budget(phase, base_metrics),
        )
        return resolved

    def _ensure_replay(self):
        return self._replay_bundle().data

    def _run_config(
        self,
        mutations: dict[str, Any],
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        store_context: bool = False,
        collect_diagnostics: bool = False,
    ) -> dict[str, Any]:
        from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
        from backtests.stock.auto.config_mutator import mutate_alcb_config
        from backtests.stock.auto.scoring import extract_metrics
        from backtests.stock.config_alcb import ALCBBacktestConfig
        from backtests.stock.engine.alcb_engine import ALCBIntradayEngine

        mutations = hydrate_time_mutations(mutations)
        effective_start = start_date or self.start_date
        effective_end = end_date or self.end_date
        diagnostics_enabled = bool(collect_diagnostics or store_context)
        signature = mutation_signature(mutations)
        replay_bundle = self._replay_bundle()
        replay = replay_bundle.data
        data_fingerprint = replay_bundle.cache_source_fingerprint
        metrics_key = self._metrics_cache_key(mutations, start_date=effective_start, end_date=effective_end)
        cache_key = (data_fingerprint, signature, effective_start, effective_end, diagnostics_enabled)
        cached = self._config_cache.get(cache_key)
        if cached is None and not diagnostics_enabled:
            cached = self._config_cache.get((data_fingerprint, signature, effective_start, effective_end, True))
        if cached is not None:
            if store_context:
                self._last_context = cached
            return cached

        config = mutate_alcb_config(
            ALCBBacktestConfig(
                start_date=effective_start,
                end_date=effective_end,
                initial_equity=self.initial_equity,
                tier=2,
                data_dir=self.data_dir,
            ),
            mutations,
        )
        engine = ALCBIntradayEngine(config, replay)
        shadow_tracker = ALCBShadowTracker() if (collect_diagnostics or store_context) else None
        if shadow_tracker is not None:
            engine.shadow_tracker = shadow_tracker
        result = engine.run()
        perf = extract_metrics(result.trades, result.equity_curve, result.timestamps, self.initial_equity)
        metrics = merge_alcb_metrics(perf, result.trades)
        context = {
            "mutation_signature": signature,
            "metrics": metrics,
            "trades": result.trades,
            "replay": replay,
            "daily_selections": result.daily_selections,
            "shadow_tracker": shadow_tracker,
            "config": config,
            "metrics_cache_key": metrics_key,
            "cache_source_fingerprint": data_fingerprint,
        }
        self._config_cache[cache_key] = context
        if diagnostics_enabled:
            self._config_cache[(data_fingerprint, signature, effective_start, effective_end, False)] = context
        if store_context:
            self._last_context = context
        return context

    def _seed_result_for_metrics(
        self,
        name: str,
        phase: int,
        metrics: dict[str, float],
        hard_rejects: dict[str, float],
        scoring_weights: dict[str, float] | None,
    ) -> ScoredCandidate:
        from .worker import phase_reject_reason

        reject_reason = phase_reject_reason(metrics, hard_rejects, phase=phase)
        if reject_reason:
            return ScoredCandidate(
                name=name,
                score=0.0,
                rejected=True,
                reject_reason=reject_reason,
                metrics=dict(metrics),
            )
        return ScoredCandidate(
            name=name,
            score=score_alcb_phase(phase, metrics, scoring_weights),
            metrics=dict(metrics),
        )

    @staticmethod
    def _min_gate(name: str, actual: float, target: float, *, strict: bool = False) -> GateCriterion:
        passed = float(actual) > float(target) if strict else float(actual) >= float(target)
        return GateCriterion(name, float(target), float(actual), passed)

    @staticmethod
    def _max_gate(name: str, actual: float, target: float) -> GateCriterion:
        return GateCriterion(name, float(target), float(actual), float(actual) <= float(target))

    @staticmethod
    def _raise_floor(
        resolved: dict[str, float],
        key: str,
        base_value: float,
        ratio: float,
        *,
        minimum: float = 0.0,
    ) -> None:
        floor = max(float(minimum), float(base_value) * float(ratio)) if float(base_value) > 0.0 else float(minimum)
        resolved[key] = max(float(resolved.get(key, floor)), floor)

    @staticmethod
    def _drawdown_budget(phase: int, base_metrics: dict[str, float]) -> float:
        base_dd = float(base_metrics.get("max_drawdown_pct", 0.0))
        multiplier = {1: 1.22, 2: 1.18, 3: 1.15, 4: 1.10, 5: 1.08, 6: 1.10, 7: 1.08, 8: 1.08}.get(phase, 1.12)
        static_cap = {1: 0.055, 2: 0.055, 3: 0.054, 4: 0.052, 5: 0.052, 6: 0.055, 7: 0.052, 8: 0.052}.get(phase, 0.055)
        floor = {1: 0.047, 2: 0.047, 3: 0.046, 4: 0.045, 5: 0.045, 6: 0.047, 7: 0.045, 8: 0.045}.get(phase, 0.045)
        dynamic = max(base_dd * multiplier, base_dd + 0.010, floor)
        return min(static_cap, dynamic)


def _build_phase_snapshot(
    phase: int,
    focus: str,
    metrics: dict[str, float],
    greedy_result,
    *,
    base_metrics: dict[str, float],
    hard_rejects: dict[str, float],
) -> str:
    floors = ", ".join(
        f"{key}={value:.4f}" if "dd" not in key else f"{key}={value:.2%}"
        for key, value in sorted(hard_rejects.items())
    )
    focus_deltas = ", ".join(
        f"{metric}={base_metrics.get(metric, 0.0):+.3f}->{metrics.get(metric, 0.0):+.3f}"
        for metric in PHASE_FOCUS[phase][1]
    )
    return "\n".join([
        "=" * 70,
        f"ALCB ROUND-3 RESIDUAL ALPHA PHASE {phase} SNAPSHOT",
        "=" * 70,
        f"Focus: {focus}",
        f"Score {greedy_result.base_score:.4f} -> {greedy_result.final_score:.4f} with {greedy_result.accepted_count} accepted mutations.",
        "",
        (
            f"Core: trades={int(metrics['total_trades'])}, net_profit={metrics['net_profit']:+.2f}, "
            f"expectancy_dollar={metrics['expectancy_dollar']:+.2f}, expected_total_r={metrics['expected_total_r']:+.2f}, "
            f"pf={metrics['profit_factor']:.3f}, dd={metrics['max_drawdown_pct']:.1%}, trades/month={metrics['trades_per_month']:.2f}"
        ),
        (
            f"Quality: signal_quality={metrics.get('signal_quality', 0.0):.3f}, "
            f"timing_quality={metrics.get('timing_quality', 0.0):.3f}, "
            f"extended_avwap_inverse={metrics.get('extended_avwap_inverse', 0.0):.3f}, "
            f"bar9_inverse={metrics.get('bar9_inverse', 0.0):.3f}, "
            f"late_entry_quality={metrics.get('late_entry_quality', 0.0):.3f}, "
            f"score_monotonicity={metrics.get('score_monotonicity', 0.0):.3f}"
        ),
        (
            f"Management: profit_protection={metrics.get('profit_protection', 0.0):.3f}, "
            f"short24_inv={metrics.get('short_hold_24_drag_inverse', 0.0):.3f}, "
            f"flow_mfe_inv={metrics.get('flow_mfe_exit_inverse', 0.0):.3f}, "
            f"mfe_capture_eff={metrics.get('mfe_capture_efficiency', 0.0):.3f}, "
            f"sizing_alignment={metrics.get('sizing_alignment', 0.0):.3f}, "
            f"long_hold_capture={metrics.get('long_hold_capture', 0.0):.3f}, "
            f"inv_dd={metrics.get('inv_dd', 0.0):.3f}"
        ),
        f"Focus metric deltas: {focus_deltas}",
        f"Hard floors: {floors}",
    ])


def _supports_spawn() -> bool:
    if sys.platform != "win32":
        return True
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", "")
    return bool(main_path) and not str(main_path).startswith("<")
