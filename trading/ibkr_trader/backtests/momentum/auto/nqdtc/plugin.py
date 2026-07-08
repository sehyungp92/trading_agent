"""NQDTC phased auto-optimization plugin.

Implements the StrategyPlugin protocol for the shared PhaseRunner framework.
5 phases targeting regime filtering, signal quality, timing/exit,
fine-tuning, and evidence-guided frequency expansion with no-regression
protection.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import MISSING, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scoring import NQDTCCompositeScore, NQDTCMetrics

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.cache_keys import build_cache_key
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
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, PhaseDecision

import logging

from .phase_candidates import get_phase_candidates
from .phase_gates import gate_criteria_for_phase

_seq_log = logging.getLogger("nqdtc.sequential")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Immutable scoring weights (7 components)
# ---------------------------------------------------------------------------

IMMUTABLE_SCORE_WEIGHTS: dict[str, float] = {
    "returns": 0.22,
    "pf": 0.12,
    "expectancy": 0.14,
    "frequency": 0.18,
    "risk": 0.10,
    "exit_capture": 0.16,
    "stability": 0.08,
}

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    phase: dict(IMMUTABLE_SCORE_WEIGHTS) for phase in range(1, 6)
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        "max_dd_pct": 0.24,
        "min_trades": 70,
        "min_pf": 1.45,
        "min_avg_r": 0.20,
        "min_capture": 0.28,
        "min_net_return_pct": 110.0,
        "min_robust_net_return_pct": 80.0,
        "max_largest_win_pnl_share": 0.42,
    },
    2: {
        "max_dd_pct": 0.24,
        "min_trades": 70,
        "min_pf": 1.45,
        "min_avg_r": 0.20,
        "min_capture": 0.30,
        "min_net_return_pct": 110.0,
        "min_robust_net_return_pct": 80.0,
        "max_largest_win_pnl_share": 0.42,
    },
    3: {
        "max_dd_pct": 0.26,
        "min_trades": 75,
        "min_pf": 1.40,
        "min_avg_r": 0.16,
        "min_capture": 0.28,
        "min_net_return_pct": 100.0,
        "min_robust_net_return_pct": 70.0,
        "max_largest_win_pnl_share": 0.45,
    },
    4: {
        "max_dd_pct": 0.26,
        "min_trades": 75,
        "min_pf": 1.40,
        "min_avg_r": 0.16,
        "min_capture": 0.28,
        "min_net_return_pct": 100.0,
        "min_robust_net_return_pct": 70.0,
        "max_largest_win_pnl_share": 0.45,
    },
    5: {
        "max_dd_pct": 0.26,
        "min_trades": 80,
        "min_pf": 1.40,
        "min_avg_r": 0.16,
        "min_capture": 0.28,
        "min_net_return_pct": 100.0,
        "min_robust_net_return_pct": 70.0,
        "max_largest_win_pnl_share": 0.45,
    },
}

PHASE_FOCUS = {
    1: ("Exit Monetization", ["capture_ratio", "tp2_hit_rate", "net_return_pct"]),
    2: ("Conditional Signal Discrimination", ["profit_factor", "avg_r", "total_trades"]),
    3: ("Entry Diversification", ["total_trades", "net_return_pct", "profit_factor"]),
    4: ("Frequency And Interaction Fine-Tune", ["total_trades", "net_return_pct", "robust_net_return_pct"]),
    5: ("Evidence-Guided Frequency Expansion", ["total_trades", "net_return_pct", "profit_factor"]),
}

ULTIMATE_TARGETS = {
    "net_return_pct": 260.0,
    "robust_net_return_pct": 185.0,
    "profit_factor": 1.75,
    "max_dd_pct": 0.20,
    "calmar": 8.0,
    "total_trades": 130.0,
    "capture_ratio": 0.48,
    "avg_r": 0.38,
    "win_rate": 0.56,
    "sharpe": 1.8,
    "sortino": 5.0,
}


def score_phase_metrics(
    phase: int,
    metrics: NQDTCMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> NQDTCCompositeScore:
    from .scoring import composite_score

    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    weights = weight_overrides or PHASE_WEIGHTS.get(phase)
    return composite_score(metrics, weights, hard_rejects=rejects)


def _format_end_date(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        initial_equity: float,
        end_date: datetime | None,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ):
        self._data_dir = data_dir
        self._initial_equity = initial_equity
        self._end_date = end_date
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker
        init_worker(str(self._data_dir), self._initial_equity, _format_end_date(self._end_date))
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate
        results = []
        total = len(candidates)
        for i, c in enumerate(candidates, 1):
            r = score_candidate((c.name, c.mutations, current_mutations, self._phase, self._scoring_weights, self._hard_rejects))
            tag = f"score={r.score:.4f}" if not r.rejected else f"REJECTED({r.reject_reason})"
            _seq_log.info("[%d/%d] %s -- %s", i, total, c.name, tag)
            results.append(r)
        return results

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

_INITIAL_MUTATIONS = {"flags.max_loss_cap": True, "flags.max_stop_width": True}


class NQDTCPlugin:
    name = "nqdtc"
    num_phases = 5
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = _INITIAL_MUTATIONS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 5,
        end_date: datetime | None = None,
    ):
        if not 1 <= num_phases <= max(PHASE_FOCUS):
            raise ValueError(f"NQDTCPlugin supports 1-{max(PHASE_FOCUS)} phases, got {num_phases}.")
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self.end_date = end_date
        self._cached_bundle = None
        self._last_context: dict[str, Any] = {}
        self._pool: mp.Pool | None = None
        self._final_metrics_cache: dict[str, dict[str, Any]] = {}
        self._evaluation_cache: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._cache_source_fingerprint: str = ""
        self._provenance: AutoRunProvenance | None = None

    def build_provenance(self) -> AutoRunProvenance:
        if self._provenance is None:
            repo_root = Path(__file__).resolve().parents[4]
            self._provenance = build_phase_auto_provenance(
                self.name,
                repo_root=repo_root,
                code_dirs=(Path(__file__).resolve().parent,),
                code_paths=(
                    repo_root / "backtests/momentum/engine/nqdtc_engine.py",
                    repo_root / "backtests/momentum/engine/sim_broker.py",
                    repo_root / "backtests/momentum/config_nqdtc.py",
                    repo_root / "backtests/momentum/auto/config_mutator.py",
                    repo_root / "backtests/momentum/data/replay_cache.py",
                ),
                data_dir=self.data_dir,
                selection_context={
                    "initial_equity": self.initial_equity,
                    "end_date": _format_end_date(self.end_date),
                    "num_phases": self.num_phases,
                    "phase_weights": PHASE_WEIGHTS,
                    "phase_hard_rejects": PHASE_HARD_REJECTS,
                    "phase_focus": PHASE_FOCUS,
                    "ultimate_targets": ULTIMATE_TARGETS,
                    "round_baseline_policy": "run_spec.baseline_mutations",
                },
            )
        return self._provenance

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(prior_phase.get("suggested_experiments", []))
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                state.cumulative_mutations,
                suggested_experiments=[(e.name, e.mutations) for e in suggested] or None,
            )
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(phase, metrics, state),
            scoring_weights=PHASE_WEIGHTS.get(phase),
            hard_rejects=PHASE_HARD_REJECTS.get(phase, {}),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.005,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
                decide_action_fn=self.decide_phase_action,
            ),
            max_rounds=50,
            prune_threshold=0.05,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        evaluation_extra = {
            "phase": phase,
            "scoring_weights": scoring_weights or {},
            "hard_rejects": hard_rejects or {},
        }
        if self.end_date is not None:
            evaluation_extra["end_date"] = _format_end_date(self.end_date)
        evaluation_key = build_cache_key(
            "nqdtc.evaluation",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra=evaluation_extra,
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
                on_terminate=self._destroy_pool,
                description=f"nqdtc phase {phase}",
                logger=logger,
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, self.end_date, phase,
                scoring_weights, hard_rejects,
            )

        # On Windows (spawn), mp.Pool re-imports everything + reloads data per
        # worker -- extremely slow and often hangs.  Skip parallel entirely when
        # max_workers <= 1 to avoid the overhead.
        if self.max_workers is not None and self.max_workers <= 1:
            raw = make_sequential()
        else:
            raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"nqdtc phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def _replay_bundle(self):
        from backtests.momentum.auto.nqdtc.worker import load_worker_data

        bundle = load_worker_data("NQ", self.data_dir)
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._evaluation_cache.clear()
            self._final_metrics_cache.clear()
            self._last_context = {}
            self.close_pool()
            self._cache_source_fingerprint = bundle.cache_source_fingerprint
        self._cached_bundle = bundle
        return bundle

    # -- Pool lifecycle --------------------------------------------------------

    def _ensure_pool(self) -> None:
        """Create the worker pool lazily; reuse across phases."""
        if self._pool is not None:
            return
        from .worker import init_worker

        self._pool = create_process_pool(
            self.max_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.initial_equity, _format_end_date(self.end_date)),
            logger=logger,
            description=f"{self.name} evaluation",
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

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
        from backtests.momentum.data.replay_cache import replay_engine_kwargs
        from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
        from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
        from backtests.momentum.auto.nqdtc.scoring import extract_nqdtc_metrics

        replay_bundle = self._replay_bundle()
        metrics_sig = mutation_signature(mutations)
        if self._last_context.get("mutation_signature") == metrics_sig:
            return dict(self._last_context["metrics"])

        final_extra = {"initial_equity": self.initial_equity}
        if self.end_date is not None:
            final_extra["end_date"] = _format_end_date(self.end_date)
        cache_key = build_cache_key(
            "nqdtc.final_metrics",
            source_fingerprint=replay_bundle.cache_source_fingerprint,
            mutations=mutations,
            extra=final_extra,
        )
        cached = self._final_metrics_cache.get(cache_key)
        if cached is not None:
            self._last_context = cached["context"]
            return dict(cached["metrics"])

        config = mutate_nqdtc_config(
            NQDTCBacktestConfig(
                initial_equity=self.initial_equity,
                data_dir=self.data_dir,
                fixed_qty=10,
                end_date=self.end_date,
            ),
            mutations,
        )
        engine = NQDTCEngine("MNQ", config)
        result = engine.run(**replay_engine_kwargs(replay_bundle))
        metrics = extract_nqdtc_metrics(
            result.trades,
            list(result.equity_curve),
            list(result.timestamps),
            self.initial_equity,
        )
        metrics_dict = asdict(metrics)
        self._last_context = {
            "mutation_signature": metrics_sig,
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": dict(metrics_dict),
            "trades": result.trades,
            "cache_key": cache_key,
        }
        self._metrics_cache[metrics_sig] = dict(metrics_dict)
        self._final_metrics_cache[cache_key] = {
            "metrics": dict(metrics_dict),
            "context": self._last_context,
        }
        return metrics_dict

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            None,
            self._last_context.get("trades"),
        )

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result) -> str:
        from .phase_diagnostics import generate_phase_diagnostics

        return generate_phase_diagnostics(
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            None,
            self._last_context.get("trades"),
            force_all_modules=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        m = _metrics_from_dict(metrics)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        extraction = (
            f"Net return {m.net_return_pct:.1f}% with {m.total_trades} trades, "
            f"robust return {m.robust_net_return_pct:.1f}%, "
            f"PF={m.profit_factor:.2f}, DD={m.max_dd_pct:.1%}. "
            "The round prioritized MFE monetization, contextual signal filters, "
            "and entry diversification over broad gate relaxation."
        )
        discrimination = (
            f"Win rate {m.win_rate:.1%}, avg R={m.avg_r:.3f}, "
            f"capture ratio {m.capture_ratio:.2f}; Range trades are "
            f"{m.range_regime_pct:.1%} of the book and ETH shorts remain "
            f"blocked ({m.eth_short_trades} trades)."
        )
        entry = (
            "Entry experiments targeted A-latch, B-range sweep, and guarded "
            "C-continuation paths so frequency gains had an explicit mechanism "
            "and live/backtest parity hook."
        )
        management = (
            f"Calmar={m.calmar:.2f}, Sharpe={m.sharpe:.2f}, "
            f"Sortino={m.sortino:.2f}; burst trades are {m.burst_trade_pct:.1%} "
            "with MFE ratchets and cooldown interactions available as selected "
            "trade-management controls."
        )
        exits = (
            f"TP1 hit rate {m.tp1_hit_rate:.1%}, TP2 hit rate {m.tp2_hit_rate:.1%}, "
            f"avg hold {m.avg_hold_hours:.1f}h, largest win share {m.largest_win_pnl_share:.1%}."
        )
        timing = (
            f"Burst trade pct {m.burst_trade_pct:.1%}, "
            f"ETH short WR {m.eth_short_wr:.1%} ({m.eth_short_trades} trades), "
            f"Range regime {m.range_regime_pct:.1%} of trades."
        )
        overall = (
            f"NQDTC optimized to {m.net_return_pct:.1f}% return, PF={m.profit_factor:.2f}, "
            f"DD={m.max_dd_pct:.1%}, {m.total_trades} trades. "
            f"Robust return {m.robust_net_return_pct:.1f}%, "
            f"capture ratio {'improved to' if m.capture_ratio > 0.33 else 'at'} {m.capture_ratio:.2f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports={
                "signal_extraction": extraction,
                "signal_discrimination": discrimination,
                "entry_mechanism": entry,
                "trade_management": management,
                "exit_mechanism": exits,
            },
            overall_verdict=overall,
            extra_sections={"timing": timing},
        )

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
        seen = seen_experiment_names(state)
        suggestions: list[Experiment] = []

        def add(name: str, mutations: dict[str, Any]) -> None:
            if name in seen:
                return
            seen.add(name)
            suggestions.append(Experiment(name=name, mutations=mutations))

        weakness_text = " ".join(weaknesses).lower()

        context = {
            "param_overrides.WEAK_SCORE_BAND_FILTER_ENABLED": True,
            "param_overrides.WEAK_SCORE_BAND_MAX_BOX_WIDTH": 225.0,
            "param_overrides.WEAK_SCORE_BAND_MIN_RVOL": 1.75,
            "param_overrides.WIDE_BOX_SCORE_FILTER_ENABLED": True,
            "param_overrides.WIDE_BOX_MIN_WIDTH": 275.0,
            "param_overrides.WIDE_BOX_MIN_SCORE": 3.0,
            "param_overrides.WIDE_BOX_MIN_RVOL": 1.75,
        }

        if m.profit_factor < 1.55 or m.avg_r < 0.25 or "profit_factor" in weakness_text:
            add("suggest_score_context_full", context)
            add("suggest_aligned_score_3_context", {
                **context,
                "param_overrides.BLOCK_ALIGNED_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
            })

        if m.max_dd_pct > 0.22 or "drawdown" in weakness_text:
            add("suggest_mfe_tiers_fast_lock", {
                "param_overrides.MFE_RATCHET_TIERS_ENABLED": True,
                "param_overrides.MFE_RATCHET_T1_R": 1.75,
                "param_overrides.MFE_RATCHET_T1_LOCK_R": 0.75,
                "param_overrides.MFE_RATCHET_T2_R": 2.75,
                "param_overrides.MFE_RATCHET_T2_LOCK_R": 1.30,
                "param_overrides.MFE_RATCHET_T3_R": 3.75,
                "param_overrides.MFE_RATCHET_T3_LOCK_R": 1.90,
            })
            add("suggest_max_stop_width_175", {"param_overrides.MAX_STOP_WIDTH_PTS": 175})

        if m.capture_ratio < 0.45 or m.tp2_hit_rate == 0:
            add("suggest_tp2_degraded_only", {
                "param_overrides.TP1_R": 1.2,
                "param_overrides.TP1_PARTIAL_PCT": 0.45,
                "param_overrides.TP2_R": 2.50,
                "param_overrides.TP2_PARTIAL_PCT": 0.15,
                "param_overrides.TP1_ONLY_CAP_MODE": "degraded_only",
            })
            add("suggest_mfe_tiers_balanced", {
                "param_overrides.MFE_RATCHET_TIERS_ENABLED": True,
                "param_overrides.MFE_RATCHET_T1_R": 2.0,
                "param_overrides.MFE_RATCHET_T1_LOCK_R": 0.8,
                "param_overrides.MFE_RATCHET_T2_R": 3.0,
                "param_overrides.MFE_RATCHET_T2_LOCK_R": 1.35,
                "param_overrides.MFE_RATCHET_T3_R": 4.0,
                "param_overrides.MFE_RATCHET_T3_LOCK_R": 2.0,
            })

        # Burst clustering
        if m.burst_trade_pct > 0.15 or "burst" in weakness_text:
            add("suggest_cooldown_45m", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 45})
            add("suggest_loss_streak_skip_12", {"param_overrides.LOSS_STREAK_SKIP_BARS": 12})

        # ETH shorts
        if m.eth_short_wr < 0.40 and m.eth_short_trades > 30:
            add("suggest_block_eth_shorts", {"flags.block_eth_shorts": True})
            add("suggest_eth_short_half", {"param_overrides.ETH_SHORT_SIZE_MULT": 0.50})

        if m.total_trades < 110:
            add("suggest_b_range_p85", {
                "param_overrides.B_ALLOW_RANGE": True,
                "param_overrides.B_MIN_DISP_Q": 0.85,
            })
            add("suggest_c_cont_mfe_0.50", {
                "flags.entry_c_continuation": True,
                "param_overrides.C_CONT_ENTRY_ENABLED": True,
                "param_overrides.C_CONT_MFE_GATE_R": 0.50,
            })
            add("suggest_cooldown_15_context", {
                **context,
                "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 15,
            })

        if m.largest_win_pnl_share > 0.30 or m.robust_net_return_pct < 220:
            add("suggest_outlier_guard_context", context)
            add("suggest_outlier_guard_mfe", {
                "param_overrides.MFE_RATCHET_TIERS_ENABLED": True,
                "param_overrides.MFE_RATCHET_T1_R": 2.0,
                "param_overrides.MFE_RATCHET_T1_LOCK_R": 0.8,
                "param_overrides.MFE_RATCHET_T2_R": 3.0,
                "param_overrides.MFE_RATCHET_T2_LOCK_R": 1.35,
                "param_overrides.MFE_RATCHET_T3_R": 4.0,
                "param_overrides.MFE_RATCHET_T3_LOCK_R": 2.0,
            })

        return suggestions

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        # The round-2 score is intentionally immutable so experiments are
        # compared against one stable objective rather than moving goalposts.
        return dict(PHASE_WEIGHTS.get(phase) or IMMUTABLE_SCORE_WEIGHTS)

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        m = _metrics_from_dict(metrics)
        return {
            "session_direction": {
                "eth_short_wr": m.eth_short_wr,
                "eth_short_trades": m.eth_short_trades,
                "range_regime_pct": m.range_regime_pct,
            },
            "exit_efficiency": {
                "capture_ratio": m.capture_ratio,
                "tp1_hit_rate": m.tp1_hit_rate,
                "tp2_hit_rate": m.tp2_hit_rate,
            },
            "return_robustness": {
                "robust_net_return_pct": m.robust_net_return_pct,
                "largest_win_return_pct": m.largest_win_return_pct,
                "largest_win_pnl_share": m.largest_win_pnl_share,
                "largest_winner_r": m.largest_winner_r,
            },
            "clustering": {
                "burst_trade_pct": m.burst_trade_pct,
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        sd = extra.get("session_direction", {})
        if sd:
            lines.append(
                f"Session/direction: ETH short WR={sd.get('eth_short_wr', 0):.1%} "
                f"({sd.get('eth_short_trades', 0)} trades), "
                f"Range regime={sd.get('range_regime_pct', 0):.1%}"
            )
        ee = extra.get("exit_efficiency", {})
        if ee:
            lines.append(
                f"Exit efficiency: capture={ee.get('capture_ratio', 0):.2f}, "
                f"TP1={ee.get('tp1_hit_rate', 0):.1%}, TP2={ee.get('tp2_hit_rate', 0):.1%}"
            )
        rr = extra.get("return_robustness", {})
        if rr:
            lines.append(
                f"Return robustness: robust={rr.get('robust_net_return_pct', 0):.1f}%, "
                f"largest_win_share={rr.get('largest_win_pnl_share', 0):.1%}, "
                f"largest_win={rr.get('largest_winner_r', 0):.2f}R"
            )
        cl = extra.get("clustering", {})
        if cl:
            lines.append(f"Clustering: burst_trade_pct={cl.get('burst_trade_pct', 0):.1%}")
        return lines

    def decide_phase_action(
        self,
        phase: int,
        metrics: dict[str, float],
        state: PhaseState,
        greedy_result,
        gate_result,
        current_weights: dict[str, float] | None,
        analysis,
        max_scoring_retries: int,
        max_diagnostic_retries: int,
    ) -> PhaseDecision | None:
        return None  # Use default framework logic

    def _gate_criteria(self, phase: int, metrics: dict[str, float], state: PhaseState) -> list[GateCriterion]:
        m = _metrics_from_dict(metrics)
        prior = state.get_phase_metrics(phase - 1) if phase > 1 else None
        return gate_criteria_for_phase(phase, m, prior)


def _metrics_from_dict(metrics: dict[str, float]) -> NQDTCMetrics:
    from .scoring import NQDTCMetrics

    fields = NQDTCMetrics.__dataclass_fields__
    payload = {}
    for key, field_info in fields.items():
        if key in metrics:
            val = metrics[key]
            # Ensure int fields get ints
            if field_info.type == "int" or (hasattr(field_info, 'default') and isinstance(field_info.default, int)):
                val = int(val)
            payload[key] = val
        elif field_info.default is not MISSING:
            payload[key] = field_info.default
        elif field_info.default_factory is not MISSING:
            payload[key] = field_info.default_factory()
    return NQDTCMetrics(**payload)
