"""Helix phased auto-optimization plugin implementing StrategyPlugin protocol."""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backtests.shared.auto.cache_keys import build_cache_key
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.provenance import AutoRunProvenance, build_phase_auto_provenance
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    deserialize_experiments,
    greedy_result_from_state,
    greedy_result_to_dict,
    mutation_signature,
    seen_experiment_names,
    shutdown_process_pool,
)
from backtests.shared.auto.types import (
    EndOfRoundArtifacts,
    Experiment,
    GateCriterion,
    GateResult,
    GreedyResult,
    ScoredCandidate,
)

from .phase_candidates import get_phase_candidates
from .scoring import HelixCompositeScore, HelixMetrics, composite_score

# ---------------------------------------------------------------------------
# Phase-specific scoring weights
# ---------------------------------------------------------------------------

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,  # use defaults
    2: {
        "net_profit": 0.34,
        "win_rate": 0.23,
        "frequency": 0.08,
        "winning_trades": 0.16,
        "pf": 0.10,
        "exit_quality": 0.05,
        "inv_dd": 0.04,
    },
    3: {
        "net_profit": 0.42,
        "win_rate": 0.17,
        "frequency": 0.06,
        "winning_trades": 0.12,
        "pf": 0.11,
        "exit_quality": 0.08,
        "inv_dd": 0.04,
    },
    4: {
        "net_profit": 0.44,
        "win_rate": 0.16,
        "frequency": 0.06,
        "winning_trades": 0.11,
        "pf": 0.11,
        "exit_quality": 0.08,
        "inv_dd": 0.04,
    },
    5: {
        "net_profit": 0.46,
        "win_rate": 0.15,
        "frequency": 0.05,
        "winning_trades": 0.10,
        "pf": 0.11,
        "exit_quality": 0.09,
        "inv_dd": 0.04,
    },
    6: {
        "net_profit": 0.46,
        "win_rate": 0.15,
        "frequency": 0.05,
        "winning_trades": 0.10,
        "pf": 0.11,
        "exit_quality": 0.09,
        "inv_dd": 0.04,
    },
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 300, "min_win_rate": 50.0, "min_winning_trades": 150.0, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80, "min_side_pf": 0.80},
    2: {"min_trades": 300, "min_win_rate": 50.0, "min_winning_trades": 150.0, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80, "min_side_pf": 0.80},
    3: {"min_trades": 300, "min_win_rate": 50.0, "min_winning_trades": 150.0, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80, "min_side_pf": 0.80},
    4: {"min_trades": 300, "min_win_rate": 50.0, "min_winning_trades": 150.0, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80, "min_side_pf": 0.80},
    5: {"min_trades": 300, "min_win_rate": 50.0, "min_winning_trades": 150.0, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80, "min_side_pf": 0.80},
    6: {"min_trades": 300, "min_win_rate": 50.0, "min_winning_trades": 150.0, "min_pf": 1.2, "max_r_dd": 25.0, "min_tail_pct": 0.30, "min_regime_pf": 0.80, "min_side_pf": 0.80},
}

ABSOLUTE_HARD_REJECT_FLOORS = {
    "min_trades": 260.0,
    "min_pf": 1.05,
    "min_tail_pct": 0.20,
    "min_regime_pf": 0.65,
    "min_side_pf": 0.65,
    "min_win_rate": 30.0,
    "min_winning_trades": 90.0,
    "max_r_dd": 25.0,
}

BASELINE_HARD_REJECT_RATIOS = {
    "min_trades": 1.0,
    "min_pf": 0.95,
    "min_tail_pct": 0.75,
    "min_regime_pf": 0.95,
    "min_side_pf": 0.85,
    "min_win_rate": 0.99,
    "min_winning_trades": 1.0,
    "max_r_dd": 1.25,
}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("SIGNAL_PRUNING + ENTRY_GATES", ["win_rate", "total_trades", "profit_factor", "min_side_pf"]),
    2: ("EXIT_LEAKAGE + STALE", ["win_rate", "winning_trades", "net_return_pct", "total_trades"]),
    3: ("STOP_TRAILING + PAYOFF", ["net_return_pct", "total_r", "win_rate", "profit_factor"]),
    4: ("VOLATILITY + ADDON", ["net_return_pct", "total_trades", "profit_factor", "max_r_dd"]),
    5: ("EXIT-SENSITIVE FINETUNE", ["net_return_pct", "total_r", "win_rate", "winning_trades"]),
    6: ("SIZING + REMAINING FINETUNE", ["calmar_r", "net_return_pct", "win_rate", "total_trades"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 250.0,
    "win_rate": 55.0,
    "profit_factor": 3.5,
    "max_r_dd": 8.0,
    "exit_efficiency": 0.65,
    "waste_ratio": 0.85,
    "tail_pct": 0.85,
    "min_side_pf": 3.5,
    "total_trades": 420.0,
    "winning_trades": 230.0,
}

_GATE_TO_SCORING = {
    "net_return_pct": "net_profit",
    "profit_factor": "pf",
    "total_trades": "frequency",
    "calmar_r": "net_profit",
    "max_r_dd": "inv_dd",
    "exit_efficiency": "exit_quality",
    "waste_ratio": "exit_quality",
    "tail_pct": "exit_quality",
    "win_rate": "win_rate",
    "winning_trades": "winning_trades",
    "min_side_pf": "side_quality",
}


def score_phase_metrics(
    phase: int,
    metrics: HelixMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> HelixCompositeScore:
    """Score metrics with phase-specific weights and hard rejects."""
    weights = PHASE_WEIGHTS.get(phase)
    if weight_overrides:
        base = dict(weights or {})
        base.update(weight_overrides)
        total = sum(base.values())
        weights = {key: value / total for key, value in base.items()} if total > 0 else base
    return composite_score(metrics, weights, hard_rejects=hard_rejects)


def _score_result_from_metrics(
    name: str,
    phase: int,
    metrics_dict: dict[str, float],
    scoring_weights: dict[str, float] | None,
    hard_rejects: dict[str, float] | None,
) -> ScoredCandidate:
    metrics = _metrics_from_dict(metrics_dict)
    score = score_phase_metrics(
        phase,
        metrics,
        weight_overrides=scoring_weights,
        hard_rejects=hard_rejects,
    )
    return ScoredCandidate(
        name=name,
        score=0.0 if score.rejected else score.total,
        rejected=score.rejected,
        reject_reason=score.reject_reason,
        metrics=dict(metrics_dict),
    )


class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._start_date = start_date
        self._end_date = end_date
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker
        init_worker(str(self._data_dir), self._initial_equity, self._start_date, self._end_date)
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


# ---------------------------------------------------------------------------
# HelixPlugin
# ---------------------------------------------------------------------------

class HelixPlugin:
    name = "helix"
    num_phases = 6
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 6,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        if not 1 <= num_phases <= 6:
            raise ValueError(f"HelixPlugin supports 1-6 phases, got {num_phases}.")
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self.start_date = start_date
        self.end_date = end_date
        self._last_context: dict[str, Any] = {}
        # Persistent pool -- created lazily, reused across phases
        self._pool: mp.Pool | None = None
        self._pool_dirty = False
        # Data cache for compute_final_metrics (avoids reloading)
        self._cached_bundle = None
        # Metrics memoization
        self._last_metrics_sig: str | None = None
        self._last_metrics_result: dict[str, float] | None = None
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._cache_source_fingerprint: str = ""
        self._phase_runtime_context: dict[int, dict[str, Any]] = {}
        self._round_artifact_output_dir: Path | None = None
        self._round_artifact_state_path: Path | None = None
        self._round_artifact_round_num: int | None = None
        self._provenance: AutoRunProvenance | None = None

    def build_provenance(self) -> AutoRunProvenance:
        if self._provenance is None:
            repo_root = Path(__file__).resolve().parents[4]
            self._provenance = build_phase_auto_provenance(
                self.name,
                repo_root=repo_root,
                code_dirs=(
                    Path(__file__).resolve().parent,
                    repo_root / "strategies/swing/akc_helix",
                ),
                code_paths=(
                    repo_root / "backtests/swing/engine/helix_engine.py",
                    repo_root / "backtests/swing/engine/helix_portfolio_engine.py",
                    repo_root / "backtests/swing/config_helix.py",
                    repo_root / "backtests/swing/data/replay_cache.py",
                ),
                data_dir=self.data_dir,
                selection_context={
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "initial_equity": self.initial_equity,
                    "num_phases": self.num_phases,
                    "phase_weights": PHASE_WEIGHTS,
                    "phase_hard_rejects": PHASE_HARD_REJECTS,
                    "phase_focus": PHASE_FOCUS,
                    "ultimate_targets": ULTIMATE_TARGETS,
                    "round_baseline_policy": "run_spec.baseline_mutations",
                },
            )
        return self._provenance

    def set_round_artifact_context(
        self,
        *,
        output_dir: Path,
        state_path: Path,
        round_num: int | None = None,
    ) -> None:
        self._round_artifact_output_dir = Path(output_dir)
        self._round_artifact_state_path = Path(state_path)
        self._round_artifact_round_num = round_num

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase in (5, 6) else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        ]

        # Later gates need the prior phase for no-regression checks.
        prior_metrics = state.get_phase_metrics(phase - 1) if phase >= 4 else None

        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics, _p=phase, _prior=prior_metrics: self._gate_criteria(_p, metrics, _prior),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=PHASE_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.01,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=20,
            prune_threshold=0.05,
        )

    # ------------------------------------------------------------------
    # Persistent pool management
    # ------------------------------------------------------------------

    def _ensure_pool(self) -> None:
        """Create the worker pool lazily; reuse across phases."""
        if self._pool is not None and not self._pool_dirty:
            return
        if self._pool is not None:
            shutdown_process_pool(self._pool, force=True)
        from .worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity, self.start_date, self.end_date),
        )
        self._pool_dirty = False

    def _on_pool_terminate(self) -> None:
        """Called by _SharedPoolBatchEvaluator.terminate() on error."""
        self._pool_dirty = True

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        base_metrics = self.compute_final_metrics(cumulative_mutations)
        resolved_hard_rejects = self._resolve_phase_hard_rejects(phase, base_metrics, hard_rejects or {})
        baseline_key = mutation_signature(cumulative_mutations)
        baseline_result = _score_result_from_metrics(
            "__baseline__",
            phase,
            base_metrics,
            scoring_weights,
            resolved_hard_rejects,
        )
        self._phase_runtime_context[phase] = {
            "base_metrics": dict(base_metrics),
            "hard_rejects": dict(resolved_hard_rejects),
        }

        evaluation_key = build_cache_key(
            "swing.helix.evaluation",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": resolved_hard_rejects,
                "initial_equity": self.initial_equity,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
        )

        def make_parallel():
            self._ensure_pool()
            from .worker import score_candidate

            return SharedPoolBatchEvaluator(
                self._pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current_mutations: [
                    (candidate.name, candidate.mutations, current_mutations, phase, scoring_weights, resolved_hard_rejects)
                    for candidate in candidates
                ],
                on_terminate=self._on_pool_terminate,
                description=f"Helix phase {phase}",
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, resolved_hard_rejects,
                start_date=self.start_date,
                end_date=self.end_date,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"Helix phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            seed_results={baseline_key: baseline_result},
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def _resolve_phase_hard_rejects(
        self,
        phase: int,
        base_metrics: dict[str, float],
        hard_rejects: dict[str, float],
    ) -> dict[str, float]:
        """Resolve hard rejects as guardrails relative to the current phase baseline.

        The optimizer should not rubber-stamp poor candidates, but it also
        must not reject the whole cumulative seed because a corrected metric is
        a few basis points below an old static threshold.
        """

        resolved = dict(PHASE_HARD_REJECTS.get(phase, {}))
        resolved.update(hard_rejects or {})

        def relax_min(reject_key: str, metric_key: str) -> None:
            if reject_key not in resolved:
                return
            static_threshold = float(resolved[reject_key])
            base_value = float(base_metrics.get(metric_key, 0.0) or 0.0)
            ratio = float(BASELINE_HARD_REJECT_RATIOS[reject_key])
            absolute = float(ABSOLUTE_HARD_REJECT_FLOORS[reject_key])
            threshold = max(absolute, min(static_threshold, base_value * ratio))
            if reject_key == "min_trades":
                resolved[reject_key] = int(round(threshold))
            else:
                resolved[reject_key] = threshold

        relax_min("min_trades", "total_trades")
        relax_min("min_pf", "profit_factor")
        relax_min("min_tail_pct", "tail_pct")
        relax_min("min_regime_pf", "min_regime_pf")
        relax_min("min_side_pf", "min_side_pf")
        relax_min("min_win_rate", "win_rate")
        if "min_winning_trades" in resolved:
            static_threshold = float(resolved["min_winning_trades"])
            base_trades = float(base_metrics.get("total_trades", 0.0) or 0.0)
            base_wr = float(base_metrics.get("win_rate", 0.0) or 0.0)
            base_winners = base_trades * base_wr / 100.0
            ratio = float(BASELINE_HARD_REJECT_RATIOS["min_winning_trades"])
            absolute = float(ABSOLUTE_HARD_REJECT_FLOORS["min_winning_trades"])
            resolved["min_winning_trades"] = max(absolute, min(static_threshold, base_winners * ratio))

        if "max_r_dd" in resolved:
            static_threshold = float(resolved["max_r_dd"])
            base_dd = float(base_metrics.get("max_r_dd", 0.0) or 0.0)
            ratio = float(BASELINE_HARD_REJECT_RATIOS["max_r_dd"])
            baseline_budget = max(12.0, base_dd * ratio, base_dd + 2.0)
            resolved["max_r_dd"] = min(static_threshold, baseline_budget)

        return resolved

    def should_adopt_failed_gate(
        self,
        *,
        phase: int,
        base_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        greedy_result: GreedyResult,
        gate_result: GateResult,
    ) -> tuple[bool, str]:
        """Adopt Pareto-style improvements even when stretch phase gates fail.

        Gates define the destination. They should not discard a mutation that
        improves the return/win-rate/good-trade frontier without materially
        damaging the rest of the strategy.
        """
        if greedy_result.accepted_count <= 0:
            return False, "gate_failed_no_accepted_mutations"

        base = _metrics_from_dict(base_metrics)
        candidate = _metrics_from_dict(candidate_metrics)
        base_winners = _winning_trades(base)
        candidate_winners = _winning_trades(candidate)

        improvements = []
        if candidate.net_return_pct >= base.net_return_pct + 0.25:
            improvements.append("net_return_pct")
        if candidate.total_r >= base.total_r + 0.50:
            improvements.append("total_r")
        if candidate.win_rate >= base.win_rate + 0.25:
            improvements.append("win_rate")
        if candidate.total_trades >= base.total_trades + 1:
            improvements.append("total_trades")
        if candidate_winners >= base_winners + 1.0:
            improvements.append("winning_trades")
        if candidate.profit_factor >= base.profit_factor + 0.02:
            improvements.append("profit_factor")

        if not improvements:
            return False, "gate_failed_no_frontier_metric_improved"

        harms = _material_harms(base, candidate)
        if harms:
            return False, f"gate_failed_material_harm:{','.join(harms)}"

        return True, f"incremental_frontier_improvement:{','.join(improvements)}"

    def _replay_bundle(self):
        from backtests.swing.config_helix import HelixBacktestConfig
        from backtests.swing.data.replay_cache import load_helix_replay_bundle

        base_config = HelixBacktestConfig(
            initial_equity=self.initial_equity,
            data_dir=self.data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
            track_shadows=False,
        )
        bundle = load_helix_replay_bundle(
            base_config.symbols,
            base_config.data_dir,
            start_date=base_config.start_date,
            end_date=base_config.end_date,
        )
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._evaluation_cache.clear()
            self._last_context = {}
            self._last_metrics_sig = None
            self._last_metrics_result = None
            self.close_pool()
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        return bundle

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        cached = self._metrics_cache.get(sig)
        if cached is not None and self._last_context.get("mutation_signature") == sig:
            self._last_metrics_sig = sig
            self._last_metrics_result = dict(cached)
            return dict(cached)

        from backtests.swing.config_helix import HelixBacktestConfig
        from backtests.swing.engine.helix_portfolio_engine import run_helix_independent
        from .config_mutator import mutate_helix_config
        from .scoring import extract_helix_metrics

        base_config = HelixBacktestConfig(
            initial_equity=self.initial_equity,
            data_dir=self.data_dir,
            start_date=self.start_date,
            end_date=self.end_date,
            track_shadows=False,
        )
        config = mutate_helix_config(base_config, mutations)
        result = run_helix_independent(self._replay_bundle().data, config)
        metrics = extract_helix_metrics(result, self.initial_equity)

        all_trades = []
        for sr in result.symbol_results.values():
            all_trades.extend(sr.trades)

        self._last_context = {
            "mutation_signature": sig,
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

    def run_phase_diagnostics(self, phase: int, state: PhaseState,
                               metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        metrics_obj = _metrics_from_dict(metrics)
        return generate_phase_diagnostics(
            phase=phase,
            metrics=metrics_obj,
            greedy_result=greedy_result_to_dict(greedy_result),
            state_dict=asdict(state),
            all_trades=self._last_context.get("all_trades"),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState,
                                  metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        metrics_obj = _metrics_from_dict(metrics)
        return generate_phase_diagnostics(
            phase=phase,
            metrics=metrics_obj,
            greedy_result=greedy_result_to_dict(greedy_result),
            state_dict=asdict(state),
            all_trades=self._last_context.get("all_trades"),
            force_all_modules=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        metrics_obj = _metrics_from_dict(metrics)
        final_diagnostics_text = self._build_full_round_diagnostics(state)

        dimension_reports = {
            "signal_quality": (
                f"PF={metrics_obj.profit_factor:.2f}, Bull PF={metrics_obj.bull_pf:.2f}, "
                f"Bear PF={metrics_obj.bear_pf:.2f}, "
                f"Long/Short PF={metrics_obj.long_pf:.2f}/{metrics_obj.short_pf:.2f}. "
                f"Total trades: {metrics_obj.total_trades}."
            ),
            "exit_management": (
                f"Exit efficiency={metrics_obj.exit_efficiency:.3f}, "
                f"waste ratio={metrics_obj.waste_ratio:.3f}. "
                f"Stale R={metrics_obj.stale_r:.1f}, short-hold R={metrics_obj.short_hold_r:.1f}."
            ),
            "tail_preservation": (
                f"Big winners (>=3R) = {metrics_obj.tail_pct:.1%} of gross win R. "
                f"Big winner R={metrics_obj.big_winner_r:.1f}."
            ),
            "risk_management": (
                f"Max R DD={metrics_obj.max_r_dd:.2f}, Sharpe={metrics_obj.sharpe:.2f}, "
                f"Calmar(R)={metrics_obj.calmar_r:.2f}."
            ),
        }

        overall_verdict = (
            f"Exit efficiency {'meets' if metrics_obj.exit_efficiency >= 0.35 else 'below'} target "
            f"({metrics_obj.exit_efficiency:.3f} vs 0.35 target). "
            f"Waste ratio {'good' if metrics_obj.waste_ratio >= 0.60 else 'needs work'} "
            f"({metrics_obj.waste_ratio:.3f}). "
            f"Net return {metrics_obj.net_return_pct:.1f}% with PF {metrics_obj.profit_factor:.2f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=overall_verdict,
        )

    def _build_full_round_diagnostics(self, state: PhaseState) -> str:
        """Generate the archive-style synchronized full diagnostics report.

        Phase diagnostics are useful while selecting candidates, but the
        round-level diagnostics artifact is expected to be the comprehensive
        synchronized fee-net report produced by swing.analysis.helix_full_diagnostics.
        """

        script = Path(__file__).resolve().parents[2] / "analysis" / "helix_full_diagnostics.py"
        if not script.exists():
            metrics = self.compute_final_metrics(state.cumulative_mutations)
            final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
            return self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        with tempfile.TemporaryDirectory(prefix="helix_full_diag_") as tmp:
            tmp_dir = Path(tmp)
            output_path = tmp_dir / "round_final_diagnostics.txt"
            state_path = self._round_artifact_state_path
            if state_path is None or not state_path.exists():
                state_path = tmp_dir / "phase_state.json"
                state_path.write_text(
                    json.dumps(asdict(state), indent=2, default=str),
                    encoding="utf-8",
                )
            round_label = self._round_label_for_state(state, self._round_artifact_round_num)
            cmd = [
                sys.executable,
                str(script),
                "--state-path",
                str(state_path),
                "--phase-result",
                "current",
                "--output",
                str(output_path),
                "--title",
                f"HELIX {round_label.upper()} FULL DIAGNOSTICS (SYNCHRONIZED / FEE-NET)",
                "--lineage-label",
                f"Helix {round_label}",
                "--equity",
                str(float(self.initial_equity)),
            ]
            if self.start_date:
                cmd.extend(["--start-date", str(self.start_date)])
            if self.end_date:
                cmd.extend(["--end-date", str(self.end_date)])

            subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parents[4],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return output_path.read_text(encoding="utf-8")

    @staticmethod
    def _round_label_for_state(state: PhaseState, round_num: int | None = None) -> str:
        if round_num is not None:
            return f"Round {round_num}"
        match = re.search(r"\bRound\s+(\d+)\b", state.round_name or "", flags=re.IGNORECASE)
        if match:
            return f"Round {match.group(1)}"
        return "Current Round"

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        from .phase_diagnostics import get_diagnostic_gaps
        return get_diagnostic_gaps(phase, _metrics_from_dict(metrics))

    def suggest_experiments(
        self,
        phase: int,
        metrics: dict[str, float],
        weaknesses: list[str],
        state: PhaseState,
    ) -> list[Experiment]:
        m = _metrics_from_dict(metrics)
        tested = seen_experiment_names(state)
        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name not in tested:
                tested.add(name)
                suggestions.append(Experiment(name=name, mutations=mutations))

        if m.exit_efficiency < 0.25 and phase <= 2:
            add("sug_trail_base_30", {"param_overrides.TRAIL_BASE": 3.0})
            add("sug_fade_floor_10", {"param_overrides.TRAIL_FADE_FLOOR": 1.0})
            add("sug_stall_onset_5", {"param_overrides.TRAIL_STALL_ONSET": 5})
            add("sug_rts_guard_mfe075_be", {
                "param_overrides.RTS_GUARD_MFE_R": 0.75,
                "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.45,
                "param_overrides.RTS_GUARD_FLOOR_R": 0.0,
            })

        if m.waste_ratio < 0.50:
            if phase <= 2:
                add("sug_stale_1h_20", {"param_overrides.STALE_1H_BARS": 20})
                add("sug_bail_d_6", {"param_overrides.CLASS_D_BAIL_BARS": 6})
                add("sug_early_stale_15", {"param_overrides.EARLY_STALE_BARS": 15})

        if m.min_regime_pf < 1.2:
            add("sug_prune_class_a", {"flags.disable_class_a": True})

        if m.min_side_pf < 1.2 and phase <= 1:
            add("sug_d_short_adx_24", {"param_overrides.CLASS_D_SHORT_MIN_ADX": 24.0})
            add("sug_d_hist_sign", {"param_overrides.CLASS_D_HIST_SIGN_GATE": True})

        if m.tail_pct < 0.40 and phase >= 2:
            add("sug_partial_later", {"param_overrides.R_PARTIAL_2P5": 3.0})
            add("sug_no_partial_2p5", {"flags.disable_partial_2p5r": True})

        if m.profit_factor < 1.7 and phase <= 2:
            add("sug_adx_cap_40", {"param_overrides.ADX_UPPER_GATE": 40})
            add("sug_div_mag_floor_08", {"param_overrides.DIV_MAG_FLOOR": 0.08})

        if m.max_r_dd > 15 and phase >= 3:
            add("sug_emergency_neg15", {"param_overrides.EMERGENCY_STOP_R": -1.5})
            add("sug_circuit_daily_neg20", {"param_overrides.DAILY_STOP_R": -2.0})

        return suggestions

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        base_weights = dict(current_weights or PHASE_WEIGHTS.get(phase) or {})
        if not base_weights:
            return None

        boosted = False
        for criterion in gate_result.criteria:
            if criterion.passed:
                continue
            scoring_key = _GATE_TO_SCORING.get(criterion.name.removeprefix("hard_"))
            if scoring_key and scoring_key in base_weights:
                base_weights[scoring_key] *= 1.5
                boosted = True

        for metric_name, progress in analysis.goal_progress.items():
            if progress.get("pct_of_target", 0) < 40:
                scoring_key = _GATE_TO_SCORING.get(metric_name)
                if scoring_key and scoring_key in base_weights:
                    base_weights[scoring_key] *= 1.3
                    boosted = True

        if not boosted:
            return None

        total = sum(base_weights.values())
        return {k: v / total for k, v in base_weights.items()} if total > 0 else base_weights

    def build_analysis_extra(self, phase: int, metrics: dict[str, float],
                              state: PhaseState, greedy_result) -> dict[str, Any]:
        """Per-class R attribution for targeted insight."""
        all_trades = self._last_context.get("all_trades", [])
        class_r: dict[str, float] = {}
        for t in all_trades:
            cls = getattr(t, "setup_class", "?")
            class_r[cls] = class_r.get(cls, 0.0) + t.r_multiple
        return {"class_attribution": class_r}

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines = []
        class_r = extra.get("class_attribution", {})
        if class_r:
            parts = [f"{cls}={r:+.1f}R" for cls, r in sorted(class_r.items())]
            lines.append(f"Class R attribution: {', '.join(parts)}")
        return lines

    def close_pool(self) -> None:
        """Called in finally block by PhaseRunner.run_all_phases()."""
        shutdown_process_pool(self._pool)
        self._pool = None

    # ------------------------------------------------------------------
    # Gate criteria
    # ------------------------------------------------------------------

    def _gate_criteria(self, phase: int, metrics: dict[str, float],
                        prior_metrics: dict[str, float] | None = None) -> list[GateCriterion]:
        from . import phase_gates
        if phase == 1:
            return phase_gates.gate_criteria_phase_1(metrics)
        elif phase == 2:
            return phase_gates.gate_criteria_phase_2(metrics)
        elif phase == 3:
            return phase_gates.gate_criteria_phase_3(metrics)
        elif phase == 4:
            return phase_gates.gate_criteria_phase_4(metrics, prior_metrics)
        elif phase == 5:
            return phase_gates.gate_criteria_phase_5(metrics, prior_metrics)
        elif phase == 6:
            return phase_gates.gate_criteria_phase_6(metrics, prior_metrics)
        return []


def _metrics_from_dict(metrics: dict[str, float]) -> HelixMetrics:
    fields = HelixMetrics.__dataclass_fields__
    kwargs = {key: metrics.get(key, 0.0) for key in fields}
    kwargs["total_trades"] = int(kwargs.get("total_trades", 0))
    return HelixMetrics(**kwargs)


def _winning_trades(metrics: HelixMetrics) -> float:
    return float(metrics.winning_trades or (metrics.total_trades * metrics.win_rate / 100.0))


def _material_harms(base: HelixMetrics, candidate: HelixMetrics) -> list[str]:
    harms: list[str] = []
    base_winners = _winning_trades(base)
    candidate_winners = _winning_trades(candidate)

    if candidate.net_return_pct < base.net_return_pct - 0.50:
        harms.append("net_return_pct")
    if candidate.total_r < base.total_r - 0.75:
        harms.append("total_r")
    if candidate.win_rate < base.win_rate - 0.25:
        harms.append("win_rate")
    if candidate.total_trades < base.total_trades:
        harms.append("total_trades")
    if candidate_winners < base_winners - 0.50:
        harms.append("winning_trades")
    if candidate.profit_factor < base.profit_factor * 0.99:
        harms.append("profit_factor")
    if candidate.max_r_dd > max(base.max_r_dd + 0.50, base.max_r_dd * 1.05):
        harms.append("max_r_dd")
    if candidate.min_side_pf and base.min_side_pf and candidate.min_side_pf < base.min_side_pf * 0.98:
        harms.append("min_side_pf")
    if candidate.tail_pct and base.tail_pct and candidate.tail_pct < base.tail_pct - 0.03:
        harms.append("tail_pct")
    return harms
