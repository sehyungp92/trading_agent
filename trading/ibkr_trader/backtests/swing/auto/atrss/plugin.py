"""ATRSS phased auto-optimization plugin.

Implements the StrategyPlugin protocol for the shared PhaseRunner framework.

R9 mode: 4 phases with synchronized execution for honest baselines.
  Phase 1: Structural fixes (synchronized throughout)
  Phase 2: Exit cleanup (independent screen + synchronized gate)
  Phase 3: Signal & filtering (independent screen + synchronized gate)
  Phase 4: Fine-tune (synchronized throughout)
"""
from __future__ import annotations

import logging
import multiprocessing as mp
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
    shutdown_process_pool,
)
from backtests.shared.auto.types import (
    EndOfRoundArtifacts,
    Experiment,
    GateCriterion,
)

from .phase_analyzer import get_diagnostic_gaps, suggest_experiments
from .phase_candidates import (
    get_phase_candidates,
    get_r9_phase_candidates,
    get_risk_allocation_phase_candidates,
)
from .phase_gates import gate_criteria_for_phase, risk_allocation_gate_criteria_for_phase
from .phase_scoring import (
    PHASE_FOCUS,
    PHASE_HARD_REJECTS,
    RISK_ALLOCATION_PHASE_FOCUS,
    RISK_ALLOCATION_PHASE_HARD_REJECTS,
    RISK_ALLOCATION_ULTIMATE_TARGETS,
    PHASE_WEIGHTS,
    ULTIMATE_TARGETS,
)
from .scoring import ATRSSMetrics, extract_atrss_metrics

logger = logging.getLogger(__name__)

_SYMBOL_EXPERIMENT_UNIVERSE = ["USO"]


def _metrics_from_dict(d: dict[str, float]) -> ATRSSMetrics:
    """Reconstruct ATRSSMetrics from a flat dict (safe for missing keys)."""
    return ATRSSMetrics(**{k: v for k, v in d.items() if k in ATRSSMetrics.__dataclass_fields__})


class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        mode: str = "independent",
        symbols: list[str] | None = None,
        data_symbols: list[str] | None = None,
        scoring_profile: str | None = None,
    ):
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._mode = mode
        self._symbols = symbols
        self._data_symbols = data_symbols
        self._scoring_profile = scoring_profile
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker
        init_worker(str(self._data_dir), self._initial_equity,
                    mode=self._mode, symbols=self._symbols,
                    data_symbols=self._data_symbols,
                    scoring_profile=self._scoring_profile)
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate
        results = []
        total = len(candidates)
        for i, c in enumerate(candidates):
            if total > 5 and (i + 1) % 5 == 0:
                logger.info("  Sequential eval: %d/%d candidates scored", i + 1, total)
            results.append(
                score_candidate((c.name, c.mutations, current_mutations,
                                 self._phase, self._scoring_weights, self._hard_rejects))
            )
        return results

    def close(self) -> None:
        pass


class ATRSSPlugin:
    """ATRSS phased auto-optimization plugin (duck-typing StrategyPlugin)."""

    name = "atrss"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations: dict[str, Any] | None = None

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = None,
        mode: str = "independent",
        symbols: list[str] | None = None,
        candidate_profile: str = "auto",
    ):
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.mode = mode
        self.symbols = symbols or ["QQQ", "GLD"]
        self.data_symbols = self._data_symbol_universe(self.symbols)
        self.candidate_profile = candidate_profile
        self._active_candidate_profile = "alpha"
        self._scoring_profile = "r9_synchronized" if mode == "synchronized" else "r1_independent"
        # Persistent pool -- created lazily, reused across phases
        self._pool: mp.Pool | None = None
        # Data cache for compute_final_metrics (avoids reloading parquets)
        self._cached_bundle: Any | None = None
        # Cross-phase caches: evaluation cache is namespaced by signature_prefix
        # so stale scores from a prior phase (different weights/rejects) are never hit.
        # Metrics cache stores raw metrics (settings-independent) for compute_final_metrics reuse.
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._cache_source_fingerprint: str = ""
        self._provenance: AutoRunProvenance | None = None

    # -- PhaseSpec -------------------------------------------------------------

    def build_provenance(self) -> AutoRunProvenance:
        if self._provenance is None:
            repo_root = Path(__file__).resolve().parents[4]
            self._provenance = build_phase_auto_provenance(
                self.name,
                repo_root=repo_root,
                code_dirs=(
                    Path(__file__).resolve().parent,
                    repo_root / "strategies/swing/atrss",
                ),
                code_paths=(
                    repo_root / "backtests/swing/data/replay_cache.py",
                    repo_root / "backtests/swing/analysis/atrss_full_diagnostics.py",
                ),
                data_dir=self.data_dir,
                selection_context={
                    "initial_equity": self.initial_equity,
                    "mode": self.mode,
                    "candidate_profile": self.candidate_profile,
                    "scoring_profile": self._scoring_profile,
                    "symbols": self.symbols,
                    "data_symbols": self.data_symbols,
                    "num_phases": self.num_phases,
                    "phase_weights": PHASE_WEIGHTS,
                    "phase_focus": PHASE_FOCUS,
                    "risk_phase_focus": RISK_ALLOCATION_PHASE_FOCUS,
                    "phase_hard_rejects": PHASE_HARD_REJECTS,
                    "risk_phase_hard_rejects": RISK_ALLOCATION_PHASE_HARD_REJECTS,
                    "round_baseline_policy": "run_spec.baseline_mutations",
                },
            )
        return self._provenance

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        active_profile = self._set_active_candidate_profile(self._candidate_profile_for_state(state))
        focus_table = RISK_ALLOCATION_PHASE_FOCUS if active_profile == "risk" else PHASE_FOCUS
        focus, focus_metrics = focus_table[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))

        if self.mode == "synchronized" and active_profile == "risk":
            raw_candidates = get_risk_allocation_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        elif self.mode == "synchronized":
            raw_candidates = get_r9_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        else:
            raw_candidates = get_phase_candidates(
                phase,
                prior_mutations=state.cumulative_mutations if phase == 4 else None,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        candidates = [Experiment(name=n, mutations=m) for n, m in raw_candidates]
        hard_reject_table = (
            RISK_ALLOCATION_PHASE_HARD_REJECTS
            if active_profile == "risk" else PHASE_HARD_REJECTS
        )
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics, state),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=hard_reject_table.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.005,
                diagnostic_gap_fn=get_diagnostic_gaps,
                suggest_experiments_fn=suggest_experiments,
            ),
            max_rounds=None,
        )

    def _gate_criteria(
        self, phase: int, metrics: dict[str, float], state: PhaseState,
    ) -> list[GateCriterion]:
        m = _metrics_from_dict(metrics)
        prior_phase_metrics = None
        if phase > 1:
            prior_phase_metrics = state.phase_results.get(phase - 1, {}).get("final_metrics")
        if self._active_candidate_profile == "risk":
            return risk_allocation_gate_criteria_for_phase(phase, m, prior_phase_metrics)
        return gate_criteria_for_phase(phase, m, prior_phase_metrics)

    # -- Evaluator factory -----------------------------------------------------

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        # Settings-aware cache key: same mutations scored with different
        # weights/rejects (different phase or scoring retry) get distinct entries.
        evaluation_key = build_cache_key(
            "swing.atrss.evaluation",
            source_fingerprint=self._ensure_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
                "scoring_profile": self._scoring_profile,
                "scoring_weights": scoring_weights or {},
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
                    (
                        candidate.name,
                        candidate.mutations,
                        current_mutations,
                        phase,
                        scoring_weights,
                        hard_rejects,
                    )
                    for candidate in candidates
                ],
                on_terminate=self._destroy_pool,
                description=f"ATRSS phase {phase}",
                logger=logger,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
                mode=self.mode, symbols=self.symbols, data_symbols=self.data_symbols,
                scoring_profile=self._scoring_profile,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"ATRSS phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    # -- Pool lifecycle --------------------------------------------------------

    def _ensure_pool(self) -> None:
        """Create the worker pool lazily; reuse across phases."""
        if self._pool is not None:
            return
        from .worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(
                str(self.data_dir),
                self.initial_equity,
                self.mode,
                self.symbols,
                self.data_symbols,
                self._scoring_profile,
            ),
            logger=logger,
            description=f"{self.name} evaluation",
        )
        logger.info(
            "Using mode=%s, symbols=%s, data_symbols=%s, scoring_profile=%s",
            self.mode, self.symbols, self.data_symbols, self._scoring_profile,
        )

    def _destroy_pool(self) -> None:
        """Force-kill the pool (called on worker errors via terminate())."""
        shutdown_process_pool(self._pool, force=True)
        self._pool = None

    def close_pool(self) -> None:
        """Gracefully shut down the persistent worker pool.

        Called by PhaseRunner.run_all_phases() at the end of all phases.
        """
        shutdown_process_pool(self._pool)
        self._pool = None

    # -- Metrics ---------------------------------------------------------------

    def _ensure_bundle(self):
        """Load and cache PortfolioData for compute_final_metrics."""
        from backtests.swing.data.replay_cache import load_atrss_replay_bundle

        bundle = load_atrss_replay_bundle(self.data_dir, symbols=tuple(self.data_symbols))
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._evaluation_cache.clear()
            self.close_pool()
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        return bundle

    @staticmethod
    def _data_symbol_universe(symbols: list[str]) -> list[str]:
        ordered = list(symbols)
        for sym in _SYMBOL_EXPERIMENT_UNIVERSE:
            if sym not in ordered:
                ordered.append(sym)
        return ordered

    def _candidate_profile_for_state(self, state: PhaseState | None = None) -> str:
        if self.candidate_profile in {"alpha", "risk"}:
            return self.candidate_profile
        if self.mode != "synchronized":
            return "alpha"
        mutations: dict[str, Any] = {}
        if self.initial_mutations:
            mutations.update(self.initial_mutations)
        if state and state.cumulative_mutations:
            mutations.update(state.cumulative_mutations)
        risk_markers = {
            "fixed_qty",
            "param_overrides.fixed_qty_addon_b",
            "param_overrides.fixed_qty_regime_scaling",
            "param_overrides.base_risk_pct",
            "param_overrides.dynamic_risk_strong_trend_mult",
            "param_overrides.dynamic_risk_weak_trend_mult",
        }
        if any(key in mutations for key in risk_markers):
            return "risk"
        # The current alpha-optimized baseline is the handoff into the
        # risk-allocation round.
        if "param_overrides.adx_on" in mutations and "param_overrides.recovery_tolerance_atr_trend" in mutations:
            return "risk"
        return "alpha"

    def _set_active_candidate_profile(self, profile: str) -> str:
        if profile not in {"alpha", "risk"}:
            profile = "alpha"
        if profile != self._active_candidate_profile:
            self.close_pool()
        self._active_candidate_profile = profile
        if self.mode == "synchronized":
            self._scoring_profile = "r11_risk_allocation" if profile == "risk" else "r9_synchronized"
        else:
            self._scoring_profile = "r1_independent"
        self.ultimate_targets = (
            RISK_ALLOCATION_ULTIMATE_TARGETS if profile == "risk" else ULTIMATE_TARGETS
        )
        return profile

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        """Compute final metrics -- always uses synchronized mode for honest results."""
        sig = mutation_signature(mutations)

        # Check cross-phase metrics cache (raw metrics are settings-independent)
        cached = self._metrics_cache.get(sig)
        if cached:
            return dict(cached)

        from backtests.swing.config import AblationFlags, BacktestConfig, SlippageConfig
        from backtests.swing.engine.portfolio_engine import run_independent, run_synchronized
        from backtests.swing.auto.config_mutator import mutate_atrss_config

        base_config = BacktestConfig(
            symbols=self.symbols,
            initial_equity=self.initial_equity,
            fixed_qty=10,
            data_dir=self.data_dir,
            slippage=SlippageConfig(commission_per_contract=1.00),
            flags=AblationFlags(stall_exit=False),
        )
        config = mutate_atrss_config(base_config, mutations)

        # Always use synchronized for final metrics (honest gate checks)
        if self.mode == "synchronized":
            result = run_synchronized(self._ensure_bundle().data, config)
        else:
            result = run_independent(self._ensure_bundle().data, config)

        metrics = extract_atrss_metrics(result, self.initial_equity)
        result_dict = asdict(metrics)
        # Use R-based calmar as primary (CAGR/MaxDD% is unreliable for R-based equity curves)
        result_dict["calmar"] = result_dict["calmar_r"]
        self._metrics_cache[sig] = result_dict
        return result_dict

    # -- Diagnostics -----------------------------------------------------------

    def run_phase_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase=phase,
            metrics=_metrics_from_dict(metrics),
            greedy_result=greedy_result_to_dict(greedy_result),
        )

    def run_enhanced_diagnostics(
        self,
        phase: int,
        state: PhaseState,
        metrics: dict[str, float],
        greedy_result,
    ) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase=phase,
            metrics=_metrics_from_dict(metrics),
            greedy_result=greedy_result_to_dict(greedy_result),
            force_all_phases=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        m = _metrics_from_dict(metrics)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(
            self.num_phases, state, metrics, final_greedy,
        )

        dimension_reports = {
            "signal_extraction": (
                f"PF={m.profit_factor:.2f}, WR={m.win_rate:.1%}, "
                f"{m.total_trades} trades ({m.trades_per_month:.1f}/month)."
            ),
            "exit_management": (
                f"MFE capture={m.mfe_capture:.3f}, total R={m.total_r:+.1f}, "
                f"avg R={m.avg_r:+.3f}."
            ),
            "risk_management": (
                f"Max DD={m.max_dd_pct:.2%}, Calmar R={m.calmar_r:.1f}, "
                f"Sharpe={m.sharpe:.2f}."
            ),
        }
        mode_label = "sync" if self.mode == "synchronized" else "indep"
        overall_verdict = (
            f"ATRSS ({mode_label}): {m.total_trades} trades, PF={m.profit_factor:.2f}, "
            f"+{m.total_r:.1f}R, DD={m.max_dd_pct:.2%}, "
            f"Calmar R={m.calmar_r:.1f}, MFE capture={m.mfe_capture:.3f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports=dimension_reports,
            overall_verdict=overall_verdict,
        )
