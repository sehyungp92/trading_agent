from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation import build_end_of_round_report
from .greedy_optimizer import run_greedy
from .phase_analyzer import analyze_phase
from .phase_gates import evaluate_gate
from .phase_logging import PhaseLogger
from .phase_state import PhaseState, _utc_now_iso, load_phase_state, save_phase_state
from .plugin import StrategyPlugin
from .provenance import (
    AutoRunProvenance,
    ProvenanceValidationError,
    ProvenanceValidationResult,
    build_fallback_provenance,
    coerce_provenance,
)
from .round_manager import RoundManager
from .types import GateResult, GreedyResult, PhaseAnalysis

_PROVENANCE_STATUS_COMPLETE = "complete"
_PROVENANCE_STATUS_FALLBACK_INCOMPLETE = "fallback_incomplete"

_STRICT_PROVENANCE_SURFACES = {
    ("momentum", "nqdtc"),
    ("momentum", "vdubus"),
    ("momentum", "portfolio_synergy"),
    ("stock", "alcb"),
    ("stock", "iaric"),
    ("stock", "portfolio_synergy"),
    ("swing", "tpc"),
    ("swing", "atrss"),
    ("swing", "helix"),
    ("swing", "portfolio_synergy"),
}


def _to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _analysis_action_allowed(
    analysis: PhaseAnalysis,
    state: PhaseState,
    phase: int,
    *,
    max_scoring_retries: int,
    max_diagnostic_retries: int,
) -> bool:
    if analysis.recommendation == "improve_scoring":
        return state.scoring_retries.get(phase, 0) < max_scoring_retries
    if analysis.recommendation == "improve_diagnostics":
        return state.diagnostic_retries.get(phase, 0) < max_diagnostic_retries
    return True


class PhaseRunner:
    def __init__(
        self,
        plugin: StrategyPlugin,
        output_dir: Path,
        round_name: str = "",
        *,
        max_rounds: int | None = None,
        min_delta: float = 0.001,
        max_retries: int = 2,
        max_diagnostic_retries: int = 1,
        round_manager: RoundManager | None = None,
        round_num: int | None = None,
        allow_selection_drift: bool = False,
    ):
        self.plugin = plugin
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_name = round_name
        self.max_rounds = max_rounds
        self.min_delta = min_delta
        self.max_retries = max_retries
        self.max_diagnostic_retries = max_diagnostic_retries
        self._round_manager = round_manager
        self._round_num = round_num
        self._provenance: AutoRunProvenance | None = None
        self._provenance_status: str | None = None
        self._provenance_validation: ProvenanceValidationResult | None = None
        self._allow_selection_drift = allow_selection_drift
        self.state_path = (
            self._round_manager.phase_state_path(self.output_dir)
            if self._round_manager else self.output_dir / "phase_state.json"
        )
        self.phase_logger = PhaseLogger(self.output_dir, round_name=round_name)

    def _ensure_round_spec(self, state: PhaseState) -> None:
        if not self._round_manager or self._round_num is None:
            return
        provenance = self._current_provenance()
        self._validate_provenance_before_round_start(provenance)
        previous_validation_provenance = self._previous_round_validation_provenance(provenance)

        baseline_mutations = getattr(self.plugin, "initial_mutations", None)
        if baseline_mutations is None:
            if self._round_num > 1:
                baseline_mutations = self._round_manager.get_previous_mutations(
                    self._round_num,
                    current_provenance=previous_validation_provenance,
                )
            else:
                baseline_mutations = {}
        baseline_source = getattr(self.plugin, "initial_mutations_source", None)
        scoring_weights = getattr(self.plugin, "score_weights", None)
        if scoring_weights is None:
            scoring_weights = getattr(self.plugin, "scoring_weights", None)
        self._round_manager.write_run_spec(
            self.output_dir,
            self._round_num,
            self.plugin.name,
            description=self.round_name or f"Round {self._round_num}",
            scoring_weights=dict(scoring_weights or {}),
            baseline_mutations=dict(baseline_mutations),
            baseline_source=baseline_source,
            execution_context=_plugin_execution_context(self.plugin),
            provenance=provenance,
            provenance_status=self._provenance_status,
        )

    def load_state(self) -> PhaseState:
        state = load_phase_state(self.state_path)
        initial_mutations = getattr(self.plugin, "initial_mutations", None)
        if initial_mutations and not state.cumulative_mutations and not state.phase_results:
            state.cumulative_mutations = dict(initial_mutations)
        if self.round_name and not state.round_name:
            state.round_name = self.round_name
        return state

    def run_phase(self, phase: int, state: PhaseState | None = None) -> PhaseState:
        state = state or self.load_state()
        self._ensure_round_spec(state)
        state = self._prepare_state_for_phase(state, phase)
        phase_log = self.phase_logger.get_phase_logger(phase)
        phase_base_mutations = dict(state.cumulative_mutations)
        state.start_phase(phase)
        save_phase_state(state, self.state_path)

        spec = self.plugin.get_phase_spec(phase, state)
        phase_candidates = _dedupe_experiments(spec.candidates)
        scoring_weights = spec.scoring_weights
        force_all_diagnostics = False
        greedy_result: GreedyResult | None = None
        metrics: dict[str, float] | None = None
        gate_result: GateResult | None = None

        def emit_progress(status: str, extra: dict[str, Any] | None = None) -> None:
            self.phase_logger.update_progress(
                phase,
                _live_progress_summary(
                    state,
                    phase,
                    status=status,
                    focus=spec.focus,
                    candidate_count=len(phase_candidates),
                    extra=extra,
                ),
            )

        def greedy_progress(payload: dict[str, Any]) -> None:
            event = str(payload.get("event", "greedy_update"))
            status = "greedy_running"
            if event == "baseline_complete":
                status = "greedy_baseline_complete"
            elif event == "round_complete" and payload.get("stop_reason"):
                status = "greedy_stopping"
            elif event == "batch_complete":
                status = "greedy_batch_complete"
            emit_progress(
                status,
                {
                    "last_event": event,
                    "greedy_progress": payload,
                },
            )

        phase_log.info("Starting phase %d with %d candidates", phase, len(phase_candidates))
        self.phase_logger.log_activity(
            phase,
            "phase_start",
            {
                "focus": spec.focus,
                "candidate_count": len(phase_candidates),
                "timestamp": _utc_now_iso(),
            },
        )
        emit_progress("phase_started")

        while True:
            if greedy_result is None:
                self.phase_logger.log_activity(
                    phase,
                    "greedy_start",
                    {
                        "candidate_count": len(phase_candidates),
                        "scoring_retry": state.scoring_retries.get(phase, 0),
                        "diagnostic_retry": state.diagnostic_retries.get(phase, 0),
                    },
                )
                emit_progress(
                    "evaluating_baseline",
                    {
                        "scoring_retry": state.scoring_retries.get(phase, 0),
                        "diagnostic_retry": state.diagnostic_retries.get(phase, 0),
                    },
                )
                evaluate_batch = self.plugin.create_evaluate_batch(
                    phase,
                    phase_base_mutations,
                    scoring_weights=scoring_weights,
                    hard_rejects=spec.hard_rejects,
                )
                set_progress_callback = getattr(evaluate_batch, "set_progress_callback", None)
                if callable(set_progress_callback):
                    set_progress_callback(greedy_progress)
                checkpoint_path = self.output_dir / f"phase_{phase}_greedy_checkpoint.json"
                greedy_result = run_greedy(
                    phase_candidates,
                    phase_base_mutations,
                    evaluate_batch,
                    max_rounds=self.max_rounds or spec.max_rounds or len(phase_candidates),
                    min_delta=self.min_delta,
                    prune_threshold=spec.prune_threshold if spec.prune_threshold is not None else 0.05,
                    reject_streak_limit=spec.reject_streak_limit if spec.reject_streak_limit is not None else 1,
                    checkpoint_path=checkpoint_path,
                    checkpoint_context={
                        "phase": phase,
                        "scoring_weights": scoring_weights or {},
                        "hard_rejects": spec.hard_rejects,
                    },
                    logger=phase_log,
                    progress_callback=greedy_progress,
                )
                self.phase_logger.save_phase_output(phase, "greedy_raw", _to_dict(greedy_result))
                try:
                    metrics = self.plugin.compute_final_metrics(greedy_result.final_mutations)
                except Exception:
                    phase_log.exception("compute_final_metrics failed for phase %d", phase)
                    raise
                if metrics is None:
                    raise ValueError(f"compute_final_metrics returned None for phase {phase}")
                greedy_result.final_metrics = metrics
                self.phase_logger.save_phase_output(phase, "greedy", _to_dict(greedy_result))
                self.phase_logger.log_activity(
                    phase,
                    "greedy_complete",
                    {
                        "base_score": greedy_result.base_score,
                        "final_score": greedy_result.final_score,
                        "accepted_count": greedy_result.accepted_count,
                    },
                )
                emit_progress(
                    "greedy_complete",
                    {
                        "base_score": greedy_result.base_score,
                        "final_score": greedy_result.final_score,
                        "accepted_count": greedy_result.accepted_count,
                        "kept_features": list(greedy_result.kept_features),
                    },
                )

                criteria = spec.gate_criteria_fn(metrics)
                gate_result = evaluate_gate(criteria, greedy_result)
                state.record_gate(phase, _gate_to_dict(gate_result))
                save_phase_state(state, self.state_path)
                self.phase_logger.log_activity(
                    phase,
                    "gate_check",
                    {
                        "passed": gate_result.passed,
                        "failure_category": gate_result.failure_category,
                    },
                )
                emit_progress(
                    "gate_passed" if gate_result.passed else "gate_failed",
                    {
                        "gate_passed": gate_result.passed,
                        "failure_category": gate_result.failure_category,
                    },
                )

            assert metrics is not None
            assert gate_result is not None
            assert greedy_result is not None

            try:
                diagnostics_text = (
                    self.plugin.run_enhanced_diagnostics(phase, state, metrics, greedy_result)
                    if force_all_diagnostics
                    else self.plugin.run_phase_diagnostics(phase, state, metrics, greedy_result)
                )
            except Exception:
                phase_log.exception("Diagnostics failed for phase %d", phase)
                diagnostics_text = f"[DIAGNOSTICS FAILED] See phase_{phase}.log for traceback."
            diagnostics_kind = "diagnostics_enhanced" if force_all_diagnostics else "diagnostics"
            self.phase_logger.save_phase_output(phase, diagnostics_kind, diagnostics_text)
            self.phase_logger.log_activity(
                phase,
                "diagnostics_run",
                {"enhanced": force_all_diagnostics},
            )
            emit_progress(
                "diagnostics_running" if force_all_diagnostics else "diagnostics_complete",
                {"enhanced_diagnostics": force_all_diagnostics},
            )

            policy = spec.analysis_policy
            if force_all_diagnostics:
                policy = replace(policy, diagnostic_gap_fn=lambda current_phase, current_metrics: [])
            analysis = analyze_phase(
                phase,
                greedy_result,
                metrics,
                state,
                gate_result,
                ultimate_targets=self.plugin.ultimate_targets,
                policy=policy,
                current_weights=scoring_weights,
                max_scoring_retries=self.max_retries,
                max_diagnostic_retries=self.max_diagnostic_retries,
            )
            if not _analysis_action_allowed(
                analysis,
                state,
                phase,
                max_scoring_retries=self.max_retries,
                max_diagnostic_retries=self.max_diagnostic_retries,
            ):
                analysis.recommendation = "advance"
                if analysis.suggested_experiments:
                    analysis.recommendation_reason = "Retry budget exhausted; carry suggested experiments into the next phase."
                else:
                    analysis.recommendation_reason = "Retry budget exhausted; advance with current best mutations."
            self.phase_logger.save_phase_output(phase, "analysis", _analysis_to_dict(analysis))
            self.phase_logger.log_activity(
                phase,
                "analysis_complete",
                {
                    "recommendation": analysis.recommendation,
                    "reason": analysis.recommendation_reason,
                },
            )
            emit_progress(
                f"analysis_{analysis.recommendation}",
                {
                    "recommendation": analysis.recommendation,
                    "recommendation_reason": analysis.recommendation_reason,
                    "scoring_assessment": analysis.scoring_assessment,
                },
            )

            if analysis.recommendation == "improve_scoring" and state.scoring_retries.get(phase, 0) < self.max_retries:
                state.increment_retry(phase)
                state.increment_scoring_retry(phase)
                save_phase_state(state, self.state_path)
                scoring_weights = analysis.scoring_weight_overrides or scoring_weights
                if analysis.suggested_experiments:
                    phase_candidates = _dedupe_experiments([*phase_candidates, *analysis.suggested_experiments])
                greedy_result = None
                metrics = None
                gate_result = None
                force_all_diagnostics = False
                self.phase_logger.log_activity(
                    phase,
                    "decision_improve_scoring",
                    {
                        "retry": state.scoring_retries.get(phase, 0),
                        "weight_overrides": scoring_weights or {},
                        "candidate_count": len(phase_candidates),
                    },
                )
                emit_progress(
                    "retrying_scoring",
                    {
                        "recommendation_reason": analysis.recommendation_reason,
                        "scoring_weight_overrides": scoring_weights or {},
                        "candidate_count": len(phase_candidates),
                    },
                )
                continue

            if analysis.recommendation == "improve_diagnostics" and state.diagnostic_retries.get(phase, 0) < self.max_diagnostic_retries:
                state.increment_retry(phase)
                state.increment_diagnostic_retry(phase)
                save_phase_state(state, self.state_path)
                force_all_diagnostics = True
                self.phase_logger.log_activity(
                    phase,
                    "decision_improve_diagnostics",
                    {"retry": state.diagnostic_retries.get(phase, 0)},
                )
                emit_progress(
                    "retrying_diagnostics",
                    {
                        "recommendation_reason": analysis.recommendation_reason,
                    },
                )
                continue

            adopt_phase_mutations = gate_result.passed
            adoption_reason = "gate_passed" if gate_result.passed else "gate_failed"
            if not adopt_phase_mutations and greedy_result.accepted_count > 0:
                should_adopt_failed_gate = getattr(self.plugin, "should_adopt_failed_gate", None)
                if callable(should_adopt_failed_gate):
                    phase_base_metrics = self.plugin.compute_final_metrics(phase_base_mutations)
                    adoption_decision = should_adopt_failed_gate(
                        phase=phase,
                        base_metrics=phase_base_metrics,
                        candidate_metrics=metrics,
                        greedy_result=greedy_result,
                        gate_result=gate_result,
                    )
                    if isinstance(adoption_decision, tuple):
                        adopt_phase_mutations = bool(adoption_decision[0])
                        adoption_reason = str(adoption_decision[1])
                    else:
                        adopt_phase_mutations = bool(adoption_decision)
                        adoption_reason = (
                            "incremental_improvement_without_material_harm"
                            if adopt_phase_mutations else "gate_failed"
                        )
            adopted_final_mutations = dict(greedy_result.final_mutations)
            adopted_final_metrics = metrics
            adopted_final_score = greedy_result.final_score
            adopted_kept_features = list(greedy_result.kept_features)
            adopted_accepted_count = greedy_result.accepted_count

            if not adopt_phase_mutations:
                adopted_final_mutations = dict(phase_base_mutations)
                adopted_final_metrics = self.plugin.compute_final_metrics(adopted_final_mutations)
                adopted_final_score = greedy_result.base_score
                adopted_kept_features = []
                adopted_accepted_count = 0

            missing = object()
            phase_new_mutations = {
                key: value
                for key, value in adopted_final_mutations.items()
                if phase_base_mutations.get(key, missing) != value
            }
            phase_result = {
                "focus": spec.focus,
                "base_mutations": dict(phase_base_mutations),
                "final_mutations": dict(adopted_final_mutations),
                "base_score": greedy_result.base_score,
                "final_score": adopted_final_score,
                "kept_features": adopted_kept_features,
                "rounds": [_to_dict(round_result) for round_result in greedy_result.rounds],
                "final_metrics": adopted_final_metrics,
                "total_candidates": greedy_result.total_candidates,
                "accepted_count": adopted_accepted_count,
                "elapsed_seconds": greedy_result.elapsed_seconds,
                "suggested_experiments": [_to_dict(experiment) for experiment in analysis.suggested_experiments],
                "analysis": _analysis_to_dict(analysis),
                "new_mutations": phase_new_mutations,
                "applied_phase_mutations": adopt_phase_mutations,
                "adoption_reason": adoption_reason,
                "attempted_final_mutations": dict(greedy_result.final_mutations),
                "attempted_final_score": greedy_result.final_score,
                "attempted_final_metrics": metrics,
                "attempted_kept_features": list(greedy_result.kept_features),
                "attempted_accepted_count": greedy_result.accepted_count,
            }
            state.advance_phase(phase, phase_new_mutations, phase_result)
            state.record_gate(phase, _gate_to_dict(gate_result))
            save_phase_state(state, self.state_path)
            completed_summary = _progress_summary(state, phase)
            completed_summary["focus"] = spec.focus
            completed_summary["candidate_count"] = len(phase_candidates)
            self.phase_logger.update_progress(phase, completed_summary)
            self.phase_logger.log_activity(
                phase,
                "decision_advance",
                {
                    "reason": analysis.recommendation_reason,
                    "gate_passed": gate_result.passed,
                    "applied_phase_mutations": adopt_phase_mutations,
                    "adoption_reason": adoption_reason,
                    "new_mutation_count": len(phase_new_mutations),
                },
            )
            phase_log.info(
                "Phase %d complete: %.4f -> %.4f (%d accepted)",
                phase,
                greedy_result.base_score,
                greedy_result.final_score,
                greedy_result.accepted_count,
            )
            if phase == self.plugin.num_phases:
                self.run_end_of_round(state)
            return state

    def run_all_phases(self, start_phase: int | None = None) -> PhaseState:
        state = self.load_state()
        if self.round_name:
            state.round_name = self.round_name
        save_phase_state(state, self.state_path)

        backup_label = self.round_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.phase_logger.backup_state(self.state_path, backup_label)

        self._ensure_round_spec(state)

        if start_phase is None:
            start_phase = max(state.completed_phases) + 1 if state.completed_phases else 1

        try:
            for phase in range(start_phase, self.plugin.num_phases + 1):
                state = self.run_phase(phase, state)

            report_path = self.output_dir / "round_evaluation.txt"
            if not report_path.exists():
                self.run_end_of_round(state)
        finally:
            # Allow plugins with persistent pools to clean up
            close_pool = getattr(self.plugin, "close_pool", None)
            if callable(close_pool):
                close_pool()
        return state

    def _prepare_state_for_phase(self, state: PhaseState, phase: int) -> PhaseState:
        stale_phases = _phases_at_or_after(state, phase)
        if stale_phases:
            rerun_label = f"pre_phase_{phase}_rerun_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.phase_logger.backup_state(self.state_path, rerun_label)
            self.phase_logger.clear_generated_outputs(phase)

            for container in (
                state.phase_results,
                state.phase_gate_results,
                state.retry_count,
                state.scoring_retries,
                state.diagnostic_retries,
                state.phase_timestamps,
            ):
                for stale_phase in stale_phases:
                    container.pop(stale_phase, None)

            state.completed_phases = [completed for completed in state.completed_phases if completed < phase]
            state.current_phase = max(state.completed_phases, default=0)
            state.cumulative_mutations = self._base_mutations_for_phase(state, phase)
            self.phase_logger.prune_progress(set(state.completed_phases), current_phase=state.current_phase)
            save_phase_state(state, self.state_path)
            self.phase_logger.log_activity(
                phase,
                "phase_reset",
                {
                    "discarded_phases": stale_phases,
                    "resuming_from_phase": phase,
                },
            )
        else:
            state.cumulative_mutations = self._base_mutations_for_phase(state, phase)
        return state

    def _base_mutations_for_phase(self, state: PhaseState, phase: int) -> dict[str, Any]:
        base_mutations: dict[str, Any] = {}
        initial_mutations = getattr(self.plugin, "initial_mutations", None)
        if initial_mutations:
            base_mutations.update(initial_mutations)
        base_mutations.update(_mutations_through_phase(state, phase - 1))
        return base_mutations

    def run_end_of_round(self, state: PhaseState) -> str:
        set_round_artifact_context = getattr(self.plugin, "set_round_artifact_context", None)
        if callable(set_round_artifact_context):
            set_round_artifact_context(
                output_dir=self.output_dir,
                state_path=self.state_path,
                round_num=self._round_num,
            )
        artifacts = self.plugin.build_end_of_round_artifacts(state)
        diagnostics_path = (
            self._round_manager.diagnostics_path(self.output_dir)
            if self._round_manager else self.output_dir / "round_final_diagnostics.txt"
        )
        evaluation_path = (
            self._round_manager.evaluation_path(self.output_dir)
            if self._round_manager else self.output_dir / "round_evaluation.txt"
        )
        diagnostics_path.write_text(artifacts.final_diagnostics_text, encoding="utf-8")
        report = build_end_of_round_report(self.plugin.name, state, artifacts)
        evaluation_path.write_text(report, encoding="utf-8")
        save_phase_state(state, self.state_path)

        final_metrics = dict(self.plugin.compute_final_metrics(state.cumulative_mutations) or {})

        if self._round_manager and self._round_num is not None:
            provenance = self._current_provenance()
            self._validate_provenance_before_round_start(provenance)
            self._round_manager.write_run_summary(
                self.output_dir,
                state.cumulative_mutations,
                final_metrics,
                state.completed_phases,
                round_num=self._round_num,
                provenance=provenance,
                provenance_status=self._provenance_status,
                provenance_validation=self._provenance_validation,
            )
            self._round_manager.write_optimized_config(self.output_dir, state.cumulative_mutations)
            self._round_manager.append_to_manifest(
                self._round_num,
                state.cumulative_mutations,
                final_metrics,
                provenance=provenance,
                provenance_status=self._provenance_status,
            )

        self.phase_logger.log_activity(
            max(state.completed_phases) if state.completed_phases else 0,
            "end_of_round",
            {"completed_phases": state.completed_phases},
        )
        return report

    def _current_provenance(self) -> AutoRunProvenance:
        if self._provenance is not None:
            return self._provenance

        build_provenance = getattr(self.plugin, "build_provenance", None)
        if callable(build_provenance):
            provenance = coerce_provenance(build_provenance())
            if provenance is None:
                raise ValueError(f"{self.plugin.name}.build_provenance() returned no provenance.")
            self._provenance = provenance
            self._provenance_status = getattr(self.plugin, "provenance_status", None) or _PROVENANCE_STATUS_COMPLETE
            return self._provenance

        context = _plugin_execution_context(self.plugin)
        self._provenance = build_fallback_provenance(
            plugin_name=self.plugin.name,
            execution_context=context,
            shared_auto_dir=Path(__file__).resolve().parent,
        )
        self._provenance_status = _PROVENANCE_STATUS_FALLBACK_INCOMPLETE
        return self._provenance

    def _validate_provenance_before_round_start(self, provenance: AutoRunProvenance) -> None:
        if not self._round_manager or self._round_num is None:
            return

        surface = (self._round_manager.family, self._round_manager.strategy)
        if self._provenance_status == _PROVENANCE_STATUS_FALLBACK_INCOMPLETE and surface in _STRICT_PROVENANCE_SURFACES:
            raise RuntimeError(
                f"{self._round_manager.family}/{self._round_manager.strategy} requires complete provenance before "
                "new rounds can be accepted; implement build_provenance() for this plugin."
            )

        if self._round_num <= 1 or self._provenance_validation is not None:
            return

        validation_provenance = self._previous_round_validation_provenance(provenance)
        result = self._round_manager.validate_previous_round_provenance(
            self._round_num,
            validation_provenance,
            allow_diagnostics_only_drift=True,
        )
        self._provenance_validation = result
        if not result.valid:
            if self._allow_selection_drift:
                self._provenance_status = "selection_drift_accepted"
                return
            raise ProvenanceValidationError(result)

    def _previous_round_validation_provenance(self, provenance: AutoRunProvenance) -> AutoRunProvenance:
        override = getattr(self.plugin, "previous_round_provenance", None)
        if override is None:
            return provenance
        validation_provenance = coerce_provenance(override)
        if validation_provenance is None:
            raise ValueError(f"{self.plugin.name}.previous_round_provenance could not be coerced to provenance.")
        return validation_provenance


def _gate_to_dict(gate_result: GateResult) -> dict:
    return {
        "passed": gate_result.passed,
        "criteria": [_to_dict(criterion) for criterion in gate_result.criteria],
        "failure_category": gate_result.failure_category,
        "recommendations": list(gate_result.recommendations),
    }


def _analysis_to_dict(analysis: PhaseAnalysis) -> dict:
    return {
        "phase": analysis.phase,
        "goal_progress": analysis.goal_progress,
        "strengths": analysis.strengths,
        "weaknesses": analysis.weaknesses,
        "scoring_assessment": analysis.scoring_assessment,
        "diagnostic_gaps": analysis.diagnostic_gaps,
        "suggested_experiments": [_to_dict(experiment) for experiment in analysis.suggested_experiments],
        "recommendation": analysis.recommendation,
        "recommendation_reason": analysis.recommendation_reason,
        "report": analysis.report,
        "scoring_weight_overrides": analysis.scoring_weight_overrides,
        "extra": analysis.extra,
    }


def _plugin_execution_context(plugin: StrategyPlugin) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for attr in ("data_dir", "initial_equity", "start_date", "end_date", "max_workers"):
        if hasattr(plugin, attr):
            value = getattr(plugin, attr)
            context[attr] = str(value) if isinstance(value, Path) else value
    return context


def _progress_summary(state: PhaseState, phase: int) -> dict:
    result = state.phase_results.get(phase, {})
    gate = state.phase_gate_results.get(phase, {})
    return {
        "status": "completed",
        "updated_at": _utc_now_iso(),
        "completed_phases": state.completed_phases,
        "current_phase": state.current_phase,
        "total_mutations": len(state.cumulative_mutations),
        "base_score": result.get("base_score", 0.0),
        "final_score": result.get("final_score", 0.0),
        "kept_features": result.get("kept_features", []),
        "gate_passed": gate.get("passed", False),
        "failure_category": gate.get("failure_category"),
        "scoring_retries": state.scoring_retries.get(phase, 0),
        "diagnostic_retries": state.diagnostic_retries.get(phase, 0),
    }


def _live_progress_summary(
    state: PhaseState,
    phase: int,
    *,
    status: str,
    focus: str,
    candidate_count: int,
    extra: dict[str, Any] | None = None,
) -> dict:
    summary = {
        "status": status,
        "updated_at": _utc_now_iso(),
        "completed_phases": state.completed_phases,
        "current_phase": state.current_phase,
        "phase": phase,
        "focus": focus,
        "candidate_count": candidate_count,
        "phase_started_at": state.phase_timestamps.get(phase, {}).get("started"),
        "scoring_retries": state.scoring_retries.get(phase, 0),
        "diagnostic_retries": state.diagnostic_retries.get(phase, 0),
        "total_mutations": len(state.cumulative_mutations),
    }
    if extra:
        summary.update(extra)
    return summary


def _mutations_through_phase(state: PhaseState, phase: int) -> dict[str, Any]:
    """Accumulate mutations from phases 1..phase using new_mutations (preferred).

    Falls back to computing delta from final_mutations - base_mutations for
    legacy state files that lack new_mutations.
    """
    if phase <= 0:
        return {}

    mutations: dict[str, Any] = {}
    for phase_num in sorted(state.phase_results):
        if phase_num > phase:
            break
        result = state.phase_results[phase_num]
        new = result.get("new_mutations")
        if new is not None:
            mutations.update(new)
        else:
            final = result.get("final_mutations", {})
            base = result.get("base_mutations", {})
            missing = object()
            for key, value in final.items():
                if base.get(key, missing) != value:
                    mutations[key] = value
    return mutations


def _phases_at_or_after(state: PhaseState, phase: int) -> list[int]:
    phases = set()
    for container in (
        state.phase_results,
        state.phase_gate_results,
        state.retry_count,
        state.scoring_retries,
        state.diagnostic_retries,
        state.phase_timestamps,
    ):
        phases.update(key for key in container if key >= phase)
    phases.update(completed for completed in state.completed_phases if completed >= phase)
    if state.current_phase >= phase:
        phases.add(state.current_phase)
    return sorted(phases)


def _dedupe_experiments(experiments: list) -> list:
    deduped = []
    seen: set[str] = set()
    for experiment in experiments:
        if experiment.name in seen:
            continue
        seen.add(experiment.name)
        deduped.append(experiment)
    return deduped
