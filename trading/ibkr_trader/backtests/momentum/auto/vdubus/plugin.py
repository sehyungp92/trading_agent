"""VdubusNQ phased auto-optimization plugin.

Implements the StrategyPlugin protocol for the shared PhaseRunner framework.
Round 2 uses six phases: runner protection, execution recovery, entry
precision, alpha expansion, session/regime composition, and interactions.

Key design: one immutable score across all phases. Phase-specific control is
limited to hard rejects and candidate families so the optimizer cannot chase a
different objective in each phase.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import MISSING, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scoring import VdubusCompositeScore, VdubusMetrics

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
from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, PhaseDecision

from .phase_candidates import get_phase_candidates
from .phase_gates import gate_criteria_for_phase

# ---------------------------------------------------------------------------
# Immutable scoring weights -- same objective in every phase
# ---------------------------------------------------------------------------

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"max_dd_pct": 0.26, "min_trades": 120, "min_pf": 1.55, "min_avg_r": 0.18, "min_tpm": 3.5},
    2: {"max_dd_pct": 0.26, "min_trades": 120, "min_pf": 1.55, "min_avg_r": 0.18, "min_tpm": 3.5},
    3: {"max_dd_pct": 0.25, "min_trades": 120, "min_pf": 1.55, "min_avg_r": 0.18, "min_tpm": 3.5},
    4: {"max_dd_pct": 0.27, "min_trades": 120, "min_pf": 1.45, "min_avg_r": 0.15, "min_tpm": 3.5},
    5: {"max_dd_pct": 0.25, "min_trades": 120, "min_pf": 1.55, "min_avg_r": 0.18, "min_tpm": 3.5},
    6: {"max_dd_pct": 0.25, "min_trades": 120, "min_pf": 1.55, "min_avg_r": 0.18, "min_tpm": 3.5},
}

PHASE_FOCUS = {
    1: ("Blocked-Gate Alpha Conversion", ["total_trades", "trades_per_month", "avg_r", "profit_factor"]),
    2: ("No-Signal Continuation Entries", ["total_trades", "trades_per_month", "avg_r", "profit_factor"]),
    3: ("Entry Frequency & Fill Mechanics", ["total_trades", "trades_per_month", "avg_r", "profit_factor"]),
    4: ("Exit Capture & Slow-Death Rescue", ["capture_ratio", "stale_exit_pct", "fast_death_pct", "profit_factor"]),
    5: ("Session & Cohort Composition", ["evening_avg_r", "evening_trade_pct", "max_dd_pct", "r_calmar"]),
    6: ("Structural Interactions & Fine Tune", ["r_calmar", "avg_r", "capture_ratio", "trades_per_month"]),
}

ULTIMATE_TARGETS = {
    "profit_factor": 2.4,
    "max_dd_pct": 0.15,
    "max_r_drawdown": 5.5,
    "r_calmar": 6.0,
    "r_per_month": 3.0,
    "total_trades": 220.0,
    "capture_ratio": 0.60,
    "sharpe": 2.0,
    "trades_per_month": 7.0,
    "avg_r": 0.40,
}


def score_phase_metrics(
    phase: int,
    metrics: VdubusMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> VdubusCompositeScore:
    from .scoring import VdubusCompositeScore, composite_score

    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    if metrics.total_trades < rejects.get("min_trades", 40):
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_too_few_trades ({metrics.total_trades})")
    if metrics.max_dd_pct > rejects.get("max_dd_pct", 0.30):
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_max_dd ({metrics.max_dd_pct:.2%})")
    if "min_pf" in rejects and metrics.profit_factor < rejects["min_pf"]:
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_pf ({metrics.profit_factor:.2f})")
    if "min_avg_r" in rejects and metrics.avg_r < rejects["min_avg_r"]:
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_avg_r ({metrics.avg_r:.3f})")
    if "min_tpm" in rejects and metrics.trades_per_month < rejects["min_tpm"]:
        return VdubusCompositeScore(rejected=True, reject_reason=f"phase{phase}_low_frequency ({metrics.trades_per_month:.2f}/mo)")

    weights = PHASE_WEIGHTS.get(phase)
    if weight_overrides:
        base = dict(weights or {})
        base.update(weight_overrides)
        total = sum(base.values())
        weights = {k: v / total for k, v in base.items()} if total > 0 else base
    return composite_score(metrics, weights)


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


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

_INITIAL_MUTATIONS = {"flags.plus_1r_partial": False, "param_overrides.CHOP_THRESHOLD": 40}


class VdubusPlugin:
    name = "vdubus"
    num_phases = 6
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = _INITIAL_MUTATIONS

    def __init__(
        self,
        data_dir: Path,
        initial_equity: float = 10_000.0,
        max_workers: int | None = 3,
        *,
        num_phases: int = 6,
    ):
        if not 1 <= num_phases <= max(PHASE_FOCUS):
            raise ValueError(f"VdubusPlugin supports 1-{max(PHASE_FOCUS)} phases, got {num_phases}.")
        self.data_dir = Path(data_dir)
        self.initial_equity = initial_equity
        self.max_workers = max_workers
        self.num_phases = num_phases
        self._last_context: dict[str, Any] = {}
        self._pool: mp.Pool | None = None
        self._last_metrics_sig: str = ""
        self._last_metrics_result: dict[str, float] | None = None
        self._cached_bundle = None
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
                    repo_root / "backtests/momentum/engine/vdubus_engine.py",
                    repo_root / "backtests/momentum/engine/sim_broker.py",
                    repo_root / "backtests/momentum/config_vdubus.py",
                    repo_root / "backtests/momentum/auto/config_mutator.py",
                    repo_root / "backtests/momentum/data/replay_cache.py",
                ),
                data_dir=self.data_dir,
                selection_context={
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
            prune_threshold=0.06,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        evaluation_key = build_cache_key(
            "momentum.vdub.evaluation",
            source_fingerprint=self._replay_bundle().cache_source_fingerprint,
            extra={
                "phase": phase,
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
                    (candidate.name, candidate.mutations, current_mutations, phase, scoring_weights, hard_rejects)
                    for candidate in candidates
                ],
                on_terminate=self._destroy_pool,
                description=f"vdubus phase {phase}",
            )

        def make_sequential():
            return _SequentialBatchEvaluator(
                self.data_dir, self.initial_equity, phase,
                scoring_weights, hard_rejects,
            )

        raw = ResilientBatchEvaluator(make_parallel, make_sequential, description=f"vdubus phase {phase}")
        return CachedBatchEvaluator(
            raw,
            cache=self._evaluation_cache,
            signature_prefix=evaluation_key,
            metrics_cache=self._metrics_cache,
        )

    def _replay_bundle(self):
        from backtests.momentum.data.replay_cache import load_vdub_replay_bundle

        # Final-metrics and diagnostics flows require the 5m surface used by the
        # live-aligned replay engine, so keep the optimizer bundle as the full
        # superset rather than a narrower candidate-only variant.
        bundle = load_vdub_replay_bundle("NQ", self.data_dir, include_5m=True)
        if self._cache_source_fingerprint != bundle.cache_source_fingerprint:
            self._metrics_cache.clear()
            self._evaluation_cache.clear()
            self._last_context = {}
            self._last_metrics_sig = ""
            self._last_metrics_result = None
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
            initargs=(str(self.data_dir), self.initial_equity),
        )

    def _destroy_pool(self) -> None:
        """Force-kill the pool (called on worker errors via terminate())."""
        shutdown_process_pool(self._pool, force=True)
        self._pool = None

    def close_pool(self) -> None:
        """Gracefully shut down the persistent worker pool."""
        shutdown_process_pool(self._pool)
        self._pool = None

    # -- Metrics ---------------------------------------------------------------

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        sig = mutation_signature(mutations)
        cached = self._metrics_cache.get(sig)
        if cached is not None:
            return dict(cached)

        from backtests.momentum.config_vdubus import VdubusAblationFlags, VdubusBacktestConfig
        from backtests.momentum.engine.vdubus_engine import VdubusEngine
        from backtests.momentum.auto.config_mutator import mutate_vdubus_config
        from backtests.momentum.auto.vdubus.scoring import extract_vdubus_metrics

        config = mutate_vdubus_config(
            VdubusBacktestConfig(
                initial_equity=self.initial_equity,
                data_dir=self.data_dir,
                fixed_qty=10,
                flags=VdubusAblationFlags(heat_cap=False, viability_filter=False),
            ),
            mutations,
        )
        engine = VdubusEngine("NQ", config)
        result = engine.run(**self._replay_bundle().data)
        metrics = extract_vdubus_metrics(
            result.trades,
            list(result.equity_curve),
            list(result.time_series),
            self.initial_equity,
        )
        self._last_context = {
            "mutations": dict(mutations),
            "config": config,
            "result": result,
            "metrics": metrics,
            "trades": result.trades,
        }
        metrics_dict = asdict(metrics)
        self._metrics_cache[sig] = metrics_dict
        self._last_metrics_sig = sig
        self._last_metrics_result = metrics_dict
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
        from .scoring import composite_score

        metrics = self.compute_final_metrics(state.cumulative_mutations)
        m = _metrics_from_dict(metrics)
        score = composite_score(m)
        final_greedy = greedy_result_from_state(state, phase=self.num_phases, final_metrics=metrics)
        final_diagnostics_text = self.run_enhanced_diagnostics(self.num_phases, state, metrics, final_greedy)

        extraction = (
            f"Fixed-qty headline return is {m.net_return_pct:.1f}% with {m.total_trades} trades, "
            f"PF={m.profit_factor:.2f}, fixed-qty DD={m.max_dd_pct:.1%}; "
            f"deployable comparison should use {m.total_r:.1f} total R and "
            f"{m.r_per_month:.2f} R/month."
        )
        discrimination = (
            f"Win rate {m.win_rate:.1%}, avg R={m.avg_r:.3f}, "
            f"capture ratio {m.capture_ratio:.2f}."
        )
        management = (
            f"R-Calmar={m.r_calmar:.2f}, max R drawdown={m.max_r_drawdown:.2f}R, "
            f"Sharpe={m.sharpe:.2f}, Sortino={m.sortino:.2f}."
        )
        exits = (
            f"Stale exit pct {m.stale_exit_pct:.1%}, multi-session {m.multi_session_pct:.1%}, "
            f"avg hold {m.avg_hold_hours:.1f}h."
        )
        timing = (
            f"Fast deaths {m.fast_death_pct:.1%}, "
            f"evening trades {m.evening_trade_pct:.1%} (avgR={m.evening_avg_r:+.3f}), "
            f"trades/month {m.trades_per_month:.1f}."
        )
        overall = (
            f"VdubusNQ finishes the round at {m.r_per_month:.2f} R/month "
            f"({m.total_r:.1f} total R, {m.avg_r:.3f} avgR, {m.total_trades} trades) "
            f"with R-normalized score {score.total:.4f}. Fixed-qty return is "
            f"{m.net_return_pct:.1f}%, but that should be treated as a sizing diagnostic, "
            f"not the like-for-like profitability basis. Capture ratio "
            f"{'improved to' if m.capture_ratio > 0.47 else 'is'} {m.capture_ratio:.2f}."
        )
        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics_text,
            dimension_reports={
                "signal_extraction": (
                    f"{extraction} Normalized simple return would be "
                    f"{m.norm_return_25bp_pct:.1f}% at 0.25% risk/R, "
                    f"{m.norm_return_50bp_pct:.1f}% at 0.50% risk/R, and "
                    f"{m.norm_return_100bp_pct:.1f}% at 1.00% risk/R over the sample. "
                    f"The fixed-qty run implies about ${m.fixed_dollars_per_r:,.0f} per R, "
                    "which is why raw return is not comparable to dynamically sized strategies."
                ),
                "signal_discrimination": (
                    f"{discrimination} Negative evening flow is contained by the current "
                    f"caps: {m.evening_trade_pct:.1%} evening trades at {m.evening_avg_r:+.3f} avgR."
                ),
                "entry_mechanism": timing,
                "trade_management": management,
                "exit_mechanism": exits,
            },
            overall_verdict=overall,
            extra_sections={
                "score_basis": (
                    "Current diagnostics use the active seven-component R-normalized scorer: "
                    "R/month, PF, R-Calmar, inverse R drawdown, MFE capture, frequency, and trade-R Sharpe. "
                    "Phase progression scores remain the stored scores from the completed optimization run."
                ),
            },
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

        # Capture ratio low -- trail issues
        if m.capture_ratio < 0.45:
            add("suggest_be_wide_45", {"flags.plus_1r_partial": True, "param_overrides.TRAIL_MULT_BASE": 4.5})
            add("suggest_mfe_exempt_040", {"flags.stale_mfe_exempt": True, "param_overrides.STALE_MFE_EXEMPT_R": 0.40})
            add("suggest_late_trail_default", {"flags.late_trail": True})

        # High stale rate
        if m.stale_exit_pct > 0.40 or "stale" in weakness_text:
            add("suggest_stale_bars_10", {"param_overrides.STALE_BARS_15M": 10})
            add("suggest_mfe_exempt_default", {"flags.stale_mfe_exempt": True})

        # Fast deaths
        if m.fast_death_pct > 0.20 or "fast death" in weakness_text:
            add("suggest_disable_early_kill", {"flags.early_kill": False})
            add("suggest_vwap_cap_065", {"param_overrides.VWAP_CAP_CORE": 0.65})
            add("suggest_eqs_rth_1", {"flags.entry_quality_gate": True, "param_overrides.EQS_MIN_RTH": 1})

        # High drawdown
        if m.max_dd_pct > 0.15 or "drawdown" in weakness_text:
            add("suggest_lower_risk_008", {"param_overrides.BASE_RISK_PCT": 0.008})

        # Low trade count/frequency
        if m.total_trades < 180 or m.trades_per_month < 6.0 or "trade count" in weakness_text:
            add("suggest_relax_floor_015", {"param_overrides.FLOOR_PCT": 0.15})
            add("suggest_ttl_5", {"param_overrides.TTL_BARS": 5})
            add("suggest_enable_type_b", {"param_overrides.USE_TYPE_B": True})

        return suggestions

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        # Round 2 deliberately keeps one immutable objective across phases.
        return None

    def build_analysis_extra(self, phase: int, metrics: dict[str, float], state: PhaseState, greedy_result) -> dict[str, Any]:
        m = _metrics_from_dict(metrics)
        return {
            "exit_profile": {
                "stale_exit_pct": m.stale_exit_pct,
                "capture_ratio": m.capture_ratio,
                "multi_session_pct": m.multi_session_pct,
            },
            "entry_quality": {
                "fast_death_pct": m.fast_death_pct,
                "evening_trade_pct": m.evening_trade_pct,
                "evening_avg_r": m.evening_avg_r,
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        ep = extra.get("exit_profile", {})
        if ep:
            lines.append(
                f"Exit profile: stale={ep.get('stale_exit_pct', 0):.1%}, "
                f"capture={ep.get('capture_ratio', 0):.2f}, "
                f"multi-session={ep.get('multi_session_pct', 0):.1%}"
            )
        eq = extra.get("entry_quality", {})
        if eq:
            lines.append(
                f"Entry quality: fast_deaths={eq.get('fast_death_pct', 0):.1%}, "
                f"evening={eq.get('evening_trade_pct', 0):.1%} (avgR={eq.get('evening_avg_r', 0):+.3f})"
            )
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


def _metrics_from_dict(metrics: dict[str, float]) -> VdubusMetrics:
    from .scoring import VdubusMetrics

    fields = VdubusMetrics.__dataclass_fields__
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
    return VdubusMetrics(**payload)
