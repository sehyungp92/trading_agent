"""Crisis detection phased auto-optimization plugin.

Implements the StrategyPlugin protocol for the shared PhaseRunner framework.
4 phases targeting VIX/credit, yield/correlation, conjunction/hysteresis,
and fine-tuning with progressive FP-rate tightening.
"""
from __future__ import annotations

import logging
from dataclasses import MISSING, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scoring import CrisisCompositeScore, CrisisMetrics

from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.plugin_utils import (
    deserialize_experiments,
    greedy_result_from_state,
    greedy_result_to_dict,
    mutation_signature,
    seen_experiment_names,
)
from backtests.shared.auto.types import (
    EndOfRoundArtifacts,
    Experiment,
    GateCriterion,
    PhaseDecision,
)

from .phase_candidates import get_phase_candidates
from .phase_gates import gate_criteria_for_phase

logger = logging.getLogger(__name__)
_seq_log = logging.getLogger("crisis.sequential")

# ---------------------------------------------------------------------------
# Immutable scoring weights (hard detection + early advisory/action layer)
# ---------------------------------------------------------------------------

IMMUTABLE_SCORE_WEIGHTS: dict[str, float] = {
    # Final structural pass: prioritize earlier hard WARNING confirmation while
    # still enforcing FP, recovery, and coverage through hard phase gates.
    "detection_speed": 0.40,
    "early_action_speed": 0.08,
    "fp_control": 0.18,
    "coverage": 0.14,
    "severity": 0.08,
    "stability": 0.06,
    "calibration": 0.01,
    "recovery_speed": 0.04,
    "preaction_quality": 0.01,
}

PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    phase: dict(IMMUTABLE_SCORE_WEIGHTS) for phase in range(1, 5)
}

PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {
        # Phase 1: Recovery architecture -- structural discovery.
        "max_warning_fp_rate": 0.15,
        "max_crisis_fp_rate": 0.08,
        "min_crises_detected": 7,
        "max_avg_latency": 30.0,
        "max_avg_recovery_days": 40.0,
        "max_advisory_fp_rate": 0.60,
        "max_preaction_fp_rate": 0.25,
    },
    2: {
        # Phase 2: Threshold re-optimization.
        "max_warning_fp_rate": 0.10,
        "max_crisis_fp_rate": 0.05,
        "min_crises_detected": 7,
        "max_avg_latency": 25.0,
        "max_avg_recovery_days": 25.0,
        "max_advisory_fp_rate": 0.55,
        "max_preaction_fp_rate": 0.20,
    },
    3: {
        # Phase 3: Correlation + yield + conjunction -- tighten.
        "max_warning_fp_rate": 0.07,
        "max_crisis_fp_rate": 0.03,
        "min_crises_detected": 7,
        "max_avg_latency": 22.0,
        "max_avg_recovery_days": 15.0,
        "max_advisory_fp_rate": 0.52,
        "max_preaction_fp_rate": 0.16,
    },
    4: {
        # Phase 4: Fine-tuning -- final targets.
        "max_warning_fp_rate": 0.06,
        "max_crisis_fp_rate": 0.025,
        "min_crises_detected": 7,
        "max_avg_latency": 20.0,
        "max_avg_recovery_days": 10.0,
        "max_advisory_fp_rate": 0.50,
        "max_preaction_fp_rate": 0.12,
    },
}

PHASE_FOCUS = {
    1: ("Recovery + Early Action Architecture", ["avg_action_latency", "pct_preaction_watch", "avg_recovery_days"]),
    2: ("Threshold Re-optimization", ["warning_fp_rate", "avg_latency", "avg_action_latency"]),
    3: ("Correlation + Yield + Conjunction", ["warning_fp_rate", "preaction_fp_rate", "avg_action_latency"]),
    4: ("Fine-Tuning", ["avg_latency", "avg_action_latency", "preaction_fp_rate"]),
}

ULTIMATE_TARGETS = {
    "avg_latency": 16.0,
    "avg_action_latency": 12.0,
    "avg_advisory_latency": 8.0,
    "warning_fp_rate": 0.04,
    "crisis_fp_rate": 0.015,
    "advisory_fp_rate": 0.45,
    "preaction_fp_rate": 0.10,
    "pct_preaction_watch": 0.05,
    "crises_detected": 7.0,
    "transitions_per_year": 20.0,
    "avg_recovery_days": 10.0,
}


def score_phase_metrics(
    phase: int,
    metrics: CrisisMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
) -> CrisisCompositeScore:
    from .scoring import composite_score

    rejects = hard_rejects or PHASE_HARD_REJECTS.get(phase, {})
    weights = PHASE_WEIGHTS.get(phase)
    return composite_score(metrics, weights, hard_rejects=rejects)


# ---------------------------------------------------------------------------
# Sequential batch evaluator (no multiprocessing needed -- ~2ms per candidate)
# ---------------------------------------------------------------------------

class _SequentialBatchEvaluator:
    def __init__(
        self,
        data_dir: Path,
        phase: int,
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
    ):
        self._data_dir = data_dir
        self._phase = phase
        self._scoring_weights = scoring_weights
        self._hard_rejects = hard_rejects
        self._initialised = False

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        from .worker import init_worker
        init_worker(str(self._data_dir))
        self._initialised = True

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        self._ensure_init()
        from .worker import score_candidate
        results = []
        total = len(candidates)
        for i, c in enumerate(candidates, 1):
            r = score_candidate((
                c.name, c.mutations, current_mutations,
                self._phase, self._scoring_weights, self._hard_rejects,
            ))
            tag = (
                f"score={r.score:.4f}"
                if not r.rejected
                else f"REJECTED({r.reject_reason})"
            )
            _seq_log.info("[%d/%d] %s -- %s", i, total, c.name, tag)
            results.append(r)
        return results

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class CrisisPlugin:
    name = "crisis"
    num_phases = 4
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = None  # start from hand-calibrated defaults

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._last_context: dict[str, Any] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        focus, focus_metrics = PHASE_FOCUS[phase]
        prior_phase = state.phase_results.get(phase - 1, {}) if phase > 1 else {}
        suggested = deserialize_experiments(
            prior_phase.get("suggested_experiments", [])
        )
        candidates = [
            Experiment(name=name, mutations=mutations)
            for name, mutations in get_phase_candidates(
                phase,
                state.cumulative_mutations,
                suggested_experiments=(
                    [(e.name, e.mutations) for e in suggested] or None
                ),
            )
        ]
        return PhaseSpec(
            focus=focus,
            candidates=candidates,
            gate_criteria_fn=lambda metrics: self._gate_criteria(
                phase, metrics, state,
            ),
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
        return _SequentialBatchEvaluator(
            self.data_dir, phase, scoring_weights, hard_rejects,
        )

    # -- Metrics ---------------------------------------------------------------

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        import regime.crisis.config as C
        import pandas as pd
        from backtests.regime.crisis_validation import run_crisis_detector
        from .scoring import extract_crisis_metrics
        from .worker import _INTEGER_PARAMS

        sig = mutation_signature(mutations)
        cached = self._metrics_cache.get(sig)
        if cached is not None:
            return dict(cached)

        market_df = pd.read_parquet(self.data_dir / "market_df.parquet")
        strat_ret_df = pd.read_parquet(self.data_dir / "strat_ret_df.parquet")

        # Snapshot originals
        originals = {
            k: getattr(C, k)
            for k in dir(C)
            if k.isupper() and not k.startswith("ALERT") and hasattr(C, k)
        }

        # Patch config
        for k, v in mutations.items():
            if hasattr(C, k):
                setattr(C, k, int(v) if k in _INTEGER_PARAMS else float(v))

        try:
            alerts_df = run_crisis_detector(market_df, strat_ret_df)
            metrics = extract_crisis_metrics(alerts_df)
            metrics_dict = asdict(metrics)
            self._metrics_cache[sig] = dict(metrics_dict)
            self._last_context = {
                "mutation_signature": sig,
                "mutations": dict(mutations),
                "metrics": dict(metrics_dict),
            }
            return metrics_dict
        finally:
            for k, v in originals.items():
                setattr(C, k, v)

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
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
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
            phase,
            _metrics_from_dict(metrics),
            greedy_result_to_dict(greedy_result),
            force_all_modules=True,
        )

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        metrics = self.compute_final_metrics(state.cumulative_mutations)
        m = _metrics_from_dict(metrics)
        final_greedy = greedy_result_from_state(
            state, phase=self.num_phases, final_metrics=metrics,
        )
        final_diagnostics = self.run_enhanced_diagnostics(
            self.num_phases, state, metrics, final_greedy,
        )

        detection = (
            f"Detects {m.crises_detected}/{m.total_crises} labeled crises with avg latency "
            f"{m.avg_latency:.1f}d (max {m.max_latency:.1f}d). "
            f"Corrections: {m.corrections_detected}/{m.total_corrections} "
            f"(avg latency {m.correction_avg_latency:.1f}d)."
        )
        early = (
            f"External advisory avg latency {m.avg_advisory_latency:.1f}d; "
            f"portfolio action avg latency {m.avg_action_latency:.1f}d. "
            f"Pre-action WATCH {m.pct_preaction_watch:.1%}, "
            f"pre-action FP {m.preaction_fp_rate:.2%}."
        )
        fp = (
            f"WARNING FP rate {m.warning_fp_rate:.2%}, "
            f"CRISIS FP rate {m.crisis_fp_rate:.2%}, "
            f"external advisory FP {m.advisory_fp_rate:.2%}."
        )
        stability = (
            f"{m.transitions_per_year:.0f} transitions/year, "
            f"elevated {m.pct_warning + m.pct_crisis:.1%} of time."
        )
        recovery = (
            f"Avg recovery: {m.avg_recovery_days:.1f}d elevated post-crisis "
            f"(max {m.max_recovery_days:.1f}d)."
        )
        overall = (
            f"Crisis detection optimized: {m.crises_detected}/{m.total_crises} crises detected, "
            f"{m.corrections_detected}/{m.total_corrections} corrections detected, "
            f"avg latency {m.avg_latency:.1f}d, "
            f"action latency {m.avg_action_latency:.1f}d, "
            f"WARNING FP {m.warning_fp_rate:.2%}, CRISIS FP {m.crisis_fp_rate:.2%}, "
            f"pre-action FP {m.preaction_fp_rate:.2%}, "
            f"avg recovery {m.avg_recovery_days:.1f}d."
        )

        return EndOfRoundArtifacts(
            final_diagnostics_text=final_diagnostics,
            dimension_reports={
                "detection_speed": detection,
                "early_action": early,
                "false_positive_control": fp,
                "stability": stability,
                "recovery": recovery,
            },
            overall_verdict=overall,
        )

    # -- Analysis callbacks ----------------------------------------------------

    def get_diagnostic_gaps(
        self, phase: int, metrics: dict[str, float],
    ) -> list[str]:
        from .phase_diagnostics import get_diagnostic_gaps

        return get_diagnostic_gaps(phase, _metrics_from_dict(metrics))

    def suggest_experiments(
        self,
        phase: int,
        metrics: dict[str, float],
        weaknesses: list[str],
        state: PhaseState,
    ) -> list[Experiment]:
        from .phase_diagnostics import suggest_experiments as _suggest

        seen = seen_experiment_names(state)
        raw = _suggest(phase, _metrics_from_dict(metrics), weaknesses)
        return [
            Experiment(name=name, mutations=muts)
            for name, muts in raw
            if name not in seen
        ]

    def redesign_scoring_weights(
        self,
        phase: int,
        current_weights: dict[str, float] | None,
        analysis,
        gate_result,
    ) -> dict[str, float] | None:
        return dict(PHASE_WEIGHTS.get(phase) or IMMUTABLE_SCORE_WEIGHTS)

    def build_analysis_extra(
        self,
        phase: int,
        metrics: dict[str, float],
        state: PhaseState,
        greedy_result,
    ) -> dict[str, Any]:
        m = _metrics_from_dict(metrics)
        return {
            "detection": {
                "crises_detected": m.crises_detected,
                "total_crises": m.total_crises,
                "avg_latency": m.avg_latency,
                "avg_action_latency": m.avg_action_latency,
                "avg_advisory_latency": m.avg_advisory_latency,
                "gfc_latency": m.gfc_latency,
                "covid_latency": m.covid_latency,
            },
            "corrections": {
                "corrections_detected": m.corrections_detected,
                "total_corrections": m.total_corrections,
                "correction_avg_latency": m.correction_avg_latency,
            },
            "fp_rates": {
                "warning": m.warning_fp_rate,
                "crisis": m.crisis_fp_rate,
                "advisory": m.advisory_fp_rate,
                "preaction": m.preaction_fp_rate,
            },
            "distribution": {
                "pct_normal": m.pct_normal,
                "pct_watch": m.pct_watch,
                "pct_warning": m.pct_warning,
                "pct_crisis": m.pct_crisis,
                "pct_advisory_watch": m.pct_advisory_watch,
                "pct_preaction_watch": m.pct_preaction_watch,
            },
            "recovery": {
                "avg_recovery_days": m.avg_recovery_days,
                "max_recovery_days": m.max_recovery_days,
            },
        }

    def format_analysis_extra(self, extra: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        d = extra.get("detection", {})
        if d:
            total = d.get("total_crises", 7)
            lines.append(
                f"Detection: {d.get('crises_detected', 0)}/{total} crises, "
                f"hard/action/advisory latency "
                f"{d.get('avg_latency', 0):.1f}d/"
                f"{d.get('avg_action_latency', 0):.1f}d/"
                f"{d.get('avg_advisory_latency', 0):.1f}d"
            )
        corr = extra.get("corrections", {})
        if corr:
            tc = corr.get("total_corrections", 2)
            lines.append(
                f"Corrections: {corr.get('corrections_detected', 0)}/{tc}, "
                f"avg latency {corr.get('correction_avg_latency', 0):.1f}d"
            )
        fp = extra.get("fp_rates", {})
        if fp:
            lines.append(
                f"FP rates: WARNING={fp.get('warning', 0):.2%}, "
                f"CRISIS={fp.get('crisis', 0):.2%}, "
                f"ADVISORY={fp.get('advisory', 0):.2%}, "
                f"PREACTION={fp.get('preaction', 0):.2%}"
            )
        dist = extra.get("distribution", {})
        if dist:
            lines.append(
                f"Distribution: NORMAL={dist.get('pct_normal', 0):.1%}, "
                f"WATCH={dist.get('pct_watch', 0):.1%}, "
                f"WARNING={dist.get('pct_warning', 0):.1%}, "
                f"CRISIS={dist.get('pct_crisis', 0):.1%}, "
                f"PREACTION_WATCH={dist.get('pct_preaction_watch', 0):.1%}"
            )
        rec = extra.get("recovery", {})
        if rec:
            lines.append(
                f"Recovery: avg {rec.get('avg_recovery_days', 0):.1f}d, "
                f"max {rec.get('max_recovery_days', 0):.1f}d post-crisis elevated"
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

    def close_pool(self) -> None:
        pass  # No pool to close (sequential evaluator)

    def _gate_criteria(
        self,
        phase: int,
        metrics: dict[str, float],
        state: PhaseState,
    ) -> list[GateCriterion]:
        m = _metrics_from_dict(metrics)
        prior = state.get_phase_metrics(phase - 1) if phase > 1 else None
        return gate_criteria_for_phase(phase, m, prior)


def _metrics_from_dict(metrics: dict[str, float]) -> CrisisMetrics:
    from .scoring import CrisisMetrics

    fields = CrisisMetrics.__dataclass_fields__
    payload = {}
    for key, field_info in fields.items():
        if key in metrics:
            val = metrics[key]
            if field_info.type == "int" or (
                hasattr(field_info, "default")
                and isinstance(field_info.default, int)
            ):
                val = int(val)
            payload[key] = val
        elif field_info.default is not MISSING:
            payload[key] = field_info.default
        elif field_info.default_factory is not MISSING:
            payload[key] = field_info.default_factory()
    return CrisisMetrics(**payload)
