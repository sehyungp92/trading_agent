"""Helix alpha expansion plugin.

This round keeps trading-logic changes out of the optimizer and tests only
existing shared Helix config/flag levers. Candidate screening uses the fast
independent replay path; phase gates and final diagnostics are synchronized.
"""
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
from backtests.swing.auto.helix.scoring import HelixMetrics, extract_helix_metrics

from .phase_candidates import get_phase_candidates
from .scoring import DEFAULT_HARD_REJECTS, IMMUTABLE_SCORE_WEIGHTS, AlphaExpansionScore, alpha_expansion_score

ROUND_1_BASELINE_MUTATIONS: dict[str, Any] = {
    "flags.disable_class_a": True,
    "param_overrides.ADD_1H_R": 0.9,
    "param_overrides.ADD_4H_R": 0.4,
    "param_overrides.ADD_RISK_FRAC": 1.008,
    "param_overrides.ADX_UPPER_GATE": 40,
    "param_overrides.BE_ATR1H_OFFSET": 0.24,
    "param_overrides.CLASS_B_BAIL_BARS": 10,
    "param_overrides.CLASS_B_MOM_LOOKBACK": 5,
    "param_overrides.PARTIAL_2P5_FRAC": 0.72,
    "param_overrides.TRAIL_STALL_ONSET": 6,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("SIGNAL_DISCRIMINATION + COVERAGE", ["profit_factor", "total_trades", "net_return_pct"]),
    2: ("ENTRY_FILL_RATE + STOP_GEOMETRY", ["total_trades", "net_return_pct", "max_r_dd"]),
    3: ("ADD_ONS + PARTIAL_REALIZATION", ["net_return_pct", "tail_pct", "exit_efficiency"]),
    4: ("EXIT_LEAK_CONTROL + LOCAL_FINETUNE", ["waste_ratio", "exit_efficiency", "calmar_r"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 260.0,
    "profit_factor": 3.0,
    "max_r_dd": 8.0,
    "exit_efficiency": 0.55,
    "waste_ratio": 0.75,
    "tail_pct": 0.65,
    "total_trades": 430.0,
}


def score_phase_metrics(
    phase: int,
    metrics: HelixMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> AlphaExpansionScore:
    del phase, weight_overrides
    return alpha_expansion_score(metrics, hard_rejects=hard_rejects)


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


class HelixAlphaExpansionPlugin(HelixPlugin):
    name = "helix"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = ROUND_1_BASELINE_MUTATIONS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 2,
        *,
        num_phases: int = 4,
    ):
        super().__init__(data_dir, initial_equity=initial_equity, max_workers=max_workers, num_phases=num_phases)
        self._pool: mp.Pool | None = None

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        p3_metrics = state.get_phase_metrics(3) if phase == 4 else None
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        ]

        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics, _p=phase, _p3=p3_metrics: self._gate_criteria(_p, metrics, _p3),
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
            max_rounds=20,
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
            description="Helix alpha expansion fast screen",
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
            "swing.helix.alpha_expansion.fast_screen_sync_gate",
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
                description=f"Helix fast-screen phase {phase}",
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir,
                self.initial_equity,
                phase,
                scoring_weights,
                hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"Helix fast-screen phase {phase}")
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
        for sr in result.symbol_results.values():
            all_trades.extend(sr.trades)

        self._last_context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics,
            "all_trades": all_trades,
        }
        metrics_dict = asdict(metrics)
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
        suggestions = super().suggest_experiments(phase, metrics, weaknesses, state)
        tested = seen_experiment_names(state)
        tested.update(item.name for item in suggestions)

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name not in tested:
                suggestions.append(Experiment(name=name, mutations=mutations))
                tested.add(name)

        trades = float(metrics.get("total_trades", 0.0))
        if trades < 360 and phase <= 2:
            add("sug_alpha_ttl_1h_10", {"param_overrides.TTL_1H_HOURS": 10})
            add("sug_alpha_extreme_vol_off", {"param_overrides.EXTREME_VOL_PCT": 999.0})
        if metrics.get("waste_ratio", 1.0) < 0.65 and phase >= 3:
            add("sug_alpha_stale_1h_26", {"param_overrides.STALE_1H_BARS": 26})
            add("sug_alpha_be_offset_030", {"param_overrides.BE_ATR1H_OFFSET": 0.30})
        if metrics.get("tail_pct", 0.0) < 0.60 and phase >= 3:
            add("sug_alpha_partial_light_late", {
                "param_overrides.PARTIAL_2P5_FRAC": 0.55,
                "param_overrides.R_PARTIAL_2P5": 2.75,
            })
        return suggestions

    def _gate_criteria(
        self,
        phase: int,
        metrics: dict[str, float],
        p3_metrics: dict[str, float] | None = None,
    ) -> list[GateCriterion]:
        from . import phase_gates

        if phase == 1:
            return phase_gates.gate_criteria_phase_1(metrics)
        if phase == 2:
            return phase_gates.gate_criteria_phase_2(metrics)
        if phase == 3:
            return phase_gates.gate_criteria_phase_3(metrics)
        if phase == 4:
            return phase_gates.gate_criteria_phase_4(metrics, p3_metrics)
        return []
