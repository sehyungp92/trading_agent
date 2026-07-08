"""Helix leakage-control phased optimizer plugin."""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    deserialize_experiments,
    mutation_signature,
    seen_experiment_names,
    shutdown_process_pool,
)
from backtests.shared.auto.types import Experiment, GateCriterion
from backtests.swing.auto.helix.plugin import HelixPlugin
from backtests.swing.auto.helix.scoring import extract_helix_metrics

from .phase_candidates import get_phase_candidates
from .scoring import DEFAULT_HARD_REJECTS, IMMUTABLE_SCORE_WEIGHTS, extract_leakage_metrics

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("RTS_PROTECTIVE_GUARDS", ["right_then_leak_ratio", "net_return_pct", "tail_pct"]),
    2: ("FAILURE_CONFIRMATION + EXIT FINETUNE", ["right_then_loss_r", "profit_factor", "total_trades"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 700.0,
    "profit_factor": 4.0,
    "total_trades": 425.0,
    "right_then_leak_ratio": 0.20,
    "right_then_loss_r": 55.0,
    "tail_pct": 0.66,
    "max_r_dd": 8.0,
}


class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ):
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker

        init_worker(str(self._data_dir), self._initial_equity)
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate

        return [
            score_candidate((c.name, c.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects))
            for c in candidates
        ]

    def close(self) -> None:
        pass


class HelixLeakageControlPlugin(HelixPlugin):
    name = "helix"
    num_phases = 2
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 2,
    ):
        super().__init__(data_dir, initial_equity=initial_equity, max_workers=max_workers, num_phases=2)
        self._pool: mp.Pool | None = None

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics, _p=phase: self._gate_criteria(_p, metrics),
            scoring_weights=IMMUTABLE_SCORE_WEIGHTS,
            hard_rejects=dict(DEFAULT_HARD_REJECTS),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.05,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=None,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=10,
            prune_threshold=0.05,
            reject_streak_limit=2,
        )

    def _ensure_pool(self) -> None:
        if self._pool is not None and not self._pool_dirty:
            return
        if self._pool is not None:
            shutdown_process_pool(self._pool, force=True)
        from .worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity),
            description="Helix leakage-control fast screen",
        )
        self._pool_dirty = False

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        del cumulative_mutations
        evaluation_key = build_cache_key(
            "swing.helix.leakage_control.fast_screen_sync_gate.v2",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": IMMUTABLE_SCORE_WEIGHTS,
                "hard_rejects": hard_rejects or {},
            },
        )

        def make_parallel():
            self._ensure_pool()
            from .worker import score_candidate

            return SharedPoolBatchEvaluator(
                self._pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: [
                    (candidate.name, candidate.mutations, current_mutations, phase, scoring_weights, hard_rejects)
                    for candidate in candidates
                ],
                on_terminate=self._on_pool_terminate,
                description=f"Helix leakage phase {phase}",
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                phase,
                scoring_weights,
                hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"Helix leakage phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
            max_batch_size=4,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        cached = self._metrics_cache.get(sig)
        if cached is not None and self._last_metrics_sig == sig and self._last_context.get("all_trades"):
            return dict(cached)

        from backtests.swing.auto.helix.config_mutator import mutate_helix_config
        from backtests.swing.config_helix import HelixBacktestConfig
        from backtests.swing.engine.helix_portfolio_engine import run_helix_synchronized

        base_config = HelixBacktestConfig(initial_equity=self.initial_equity, data_dir=self.data_dir)
        config = mutate_helix_config(base_config, mutations)
        result = run_helix_synchronized(self._replay_bundle().data, config)
        metrics = extract_helix_metrics(result, self.initial_equity)

        all_trades = []
        for symbol_result in result.symbol_results.values():
            all_trades.extend(symbol_result.trades)
        leakage = extract_leakage_metrics(metrics, all_trades)

        self._last_context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics,
            "all_trades": all_trades,
        }
        metrics_dict = asdict(metrics)
        metrics_dict.update(leakage)
        self._metrics_cache[sig] = metrics_dict
        self._last_metrics_sig = sig
        self._last_metrics_result = metrics_dict
        return metrics_dict

    def suggest_experiments(
        self,
        phase: int,
        metrics: dict[str, float],
        weaknesses: list[str],
        state: PhaseState,
    ) -> list[Experiment]:
        del weaknesses
        tested = seen_experiment_names(state)
        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name not in tested:
                tested.add(name)
                suggestions.append(Experiment(name=name, mutations=mutations))

        if phase == 1 and metrics.get("right_then_leak_ratio", 1.0) > 0.23:
            add("sug_rts_065_balanced", {
                "param_overrides.RTS_GUARD_MFE_R": 0.65,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.40,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.0,
                "param_overrides.RTS_GUARD_MIN_BARS": 6,
                "param_overrides.RTS_GUARD_FADE_BARS": 1,
                "param_overrides.RTS_GUARD_MAX_MFE_R": 1.50,
            })
        if phase == 2 and metrics.get("right_then_loss_r", 999.0) > 65.0:
            add("sug_class_d_bail_12_m025", {
                "param_overrides.CLASS_D_BAIL_BARS": 12,
                "param_overrides.CLASS_D_BAIL_R_THRESH": -0.25,
            })
        return suggestions

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        del phase, state, greedy_result
        return {
            "right_then_lost_count": metrics.get("right_then_lost_count", 0.0),
            "right_then_loss_r": metrics.get("right_then_loss_r", 0.0),
            "right_then_leak_r": metrics.get("right_then_leak_r", 0.0),
            "right_then_leak_ratio": metrics.get("right_then_leak_ratio", 0.0),
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        return [
            (
                "Right-then-stopped leakage: "
                f"n={extra.get('right_then_lost_count', 0):.0f}, "
                f"loss={extra.get('right_then_loss_r', 0.0):.1f}R, "
                f"leak={extra.get('right_then_leak_r', 0.0):.1f}R, "
                f"ratio={extra.get('right_then_leak_ratio', 0.0):.3f}"
            )
        ]

    def close_pool(self) -> None:
        shutdown_process_pool(self._pool)
        self._pool = None

    def _gate_criteria(self, phase: int, metrics: dict[str, float]) -> list[GateCriterion]:
        leak_ratio_target = 0.225
        loss_r_target = 58.0
        return [
            GateCriterion("hard_net_return_pct", 650.0, metrics.get("net_return_pct", 0.0), metrics.get("net_return_pct", 0.0) >= 650.0),
            GateCriterion("hard_profit_factor", 3.75, metrics.get("profit_factor", 0.0), metrics.get("profit_factor", 0.0) >= 3.75),
            GateCriterion("hard_total_trades", 409.0, metrics.get("total_trades", 0.0), metrics.get("total_trades", 0.0) >= 409.0),
            GateCriterion("hard_tail_pct", 0.80, metrics.get("tail_pct", 0.0), metrics.get("tail_pct", 0.0) >= 0.80),
            GateCriterion("right_then_leak_ratio", leak_ratio_target, metrics.get("right_then_leak_ratio", 999.0), metrics.get("right_then_leak_ratio", 999.0) <= leak_ratio_target),
            GateCriterion("right_then_loss_r", loss_r_target, metrics.get("right_then_loss_r", 999.0), metrics.get("right_then_loss_r", 999.0) <= loss_r_target),
        ]
