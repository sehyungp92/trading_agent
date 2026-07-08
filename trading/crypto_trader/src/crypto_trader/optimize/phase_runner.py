"""PhaseRunner — orchestrates phased optimization with retry loops."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.profiles import LIVE_PARITY_PROFILE
from crypto_trader.optimize.contracts import (
    build_optimization_contract,
    phase_checkpoint_context,
    run_optimization_preflight,
)
from crypto_trader.optimize.evaluation import build_end_of_round_report
from crypto_trader.optimize.greedy_optimizer import run_greedy
from crypto_trader.optimize.phase_analyzer import analyze_phase
from crypto_trader.optimize.phase_gates import (
    evaluate_gate,
    suggest_scoring_adjustment,
)
from crypto_trader.optimize.phase_logging import PhaseLogger
from crypto_trader.optimize.phase_state import PhaseState, _atomic_write_json
from crypto_trader.optimize.plugin import StrategyPlugin
from crypto_trader.optimize.types import Experiment

log = structlog.get_logger("optimize.runner")


def _dedupe(experiments: list[Experiment]) -> list[Experiment]:
    """Deduplicate experiments by name, preserving order."""
    seen: set[str] = set()
    result = []
    for e in experiments:
        if e.name not in seen:
            seen.add(e.name)
            result.append(e)
    return result


class PhaseRunner:
    """Orchestrates phased optimization."""

    def __init__(
        self,
        plugin: StrategyPlugin,
        output_dir: Path,
        *,
        round_name: str = "",
        min_delta: float = 0.001,
        max_retries: int = 2,
        max_diagnostic_retries: int = 1,
        contract: dict[str, Any] | None = None,
        validation_mode: str = "strict",
    ) -> None:
        if validation_mode not in {"strict", "fast", "dev"}:
            raise ValueError("validation_mode must be one of: strict, fast, dev")
        self.plugin = plugin
        self.output_dir = output_dir
        self.logger = PhaseLogger(output_dir)
        self.round_name = round_name
        self.min_delta = min_delta
        self.max_retries = max_retries
        self.max_diagnostic_retries = max_diagnostic_retries
        self.validation_mode = validation_mode
        self.contract = contract or self._build_default_contract()
        if contract:
            self._preflight_contract(self.contract)

    def _infer_strategy_type(self) -> str | None:
        base_config = getattr(self.plugin, "base_config", None)
        text = " ".join(
            str(value).lower()
            for value in (
                getattr(self.plugin, "name", ""),
                self.plugin.__class__.__name__,
                self.plugin.__class__.__module__,
                getattr(base_config.__class__, "__name__", ""),
                getattr(base_config.__class__, "__module__", ""),
            )
        )
        for strategy in ("trend", "breakout", "momentum"):
            if strategy in text:
                return strategy
        return None

    def _build_default_contract(self) -> dict[str, Any]:
        """Build and preflight a contract when a plugin exposes standard fields."""
        backtest_config = getattr(self.plugin, "backtest_config", None)
        base_config = getattr(self.plugin, "base_config", None)
        data_dir = getattr(self.plugin, "data_dir", None)
        strategy_type = self._infer_strategy_type()
        if not isinstance(backtest_config, BacktestConfig):
            return {}
        if base_config is None or data_dir is None or strategy_type is None:
            return {}

        contract = build_optimization_contract(
            strategy_type=strategy_type,
            strategy_config=base_config,
            backtest_config=backtest_config,
            data_dir=Path(data_dir),
            profile=LIVE_PARITY_PROFILE,
            plugin=self.plugin,
        )
        self._preflight_contract(contract)
        return contract

    def _preflight_contract(self, contract: dict[str, Any]) -> None:
        backtest_config = getattr(self.plugin, "backtest_config", None)
        data_dir = getattr(self.plugin, "data_dir", None)
        if not isinstance(backtest_config, BacktestConfig) or data_dir is None:
            return
        run_optimization_preflight(
            contract=contract,
            backtest_config=backtest_config,
            data_dir=Path(data_dir),
            output_dir=self.output_dir,
            profile=LIVE_PARITY_PROFILE,
            validation_mode=self.validation_mode,
        )

    def _ensure_state_contract(self, state: PhaseState) -> None:
        if self.contract:
            state.ensure_contract(self.contract, strict=self.validation_mode == "strict")

    def load_state(self) -> PhaseState:
        """Load state from output_dir or create fresh."""
        state_path = self.output_dir / "phase_state.json"
        state = PhaseState.load_or_create(state_path)
        if self.round_name:
            state.round_name = self.round_name
        self._ensure_state_contract(state)
        # Apply initial mutations if fresh state
        if not state.completed_phases and not state.cumulative_mutations:
            initial = self.plugin.initial_mutations
            if initial:
                state.cumulative_mutations.update(initial)
        return state

    def _prepare_state_for_phase(
        self, state: PhaseState, phase: int
    ) -> PhaseState:
        """Detect and roll back stale phases when re-running from a specific phase.

        If phase N was already completed, removes data for phases >= N,
        backs up state, and re-derives cumulative mutations.
        """
        stale = [p for p in state.completed_phases if p >= phase]
        if not stale:
            return state

        state_path = self.output_dir / "phase_state.json"
        self.logger.backup_state(state_path, f"before_rollback_p{phase}")

        rolled_back = state.rollback_to_phase(phase)
        if rolled_back:
            log.info(
                "phase.rollback",
                target=phase,
                rolled_back=rolled_back,
            )
            self.logger.clear_generated_outputs(phase)
            state.save(state_path)

        return state

    def run_all_phases(self, state: PhaseState) -> PhaseState:
        """Run all phases sequentially."""
        state_path = self.output_dir / "phase_state.json"
        self._ensure_state_contract(state)
        state.save(state_path)

        # Backup state at start
        self.logger.backup_state(state_path, "round_start")

        try:
            for phase in range(1, self.plugin.num_phases + 1):
                if phase in state.completed_phases:
                    log.info("phase.skip_completed", phase=phase)
                    continue
                self.run_phase(phase, state)

            # End-of-round report after final phase
            report_text = self.run_end_of_round(state)
            if report_text:
                log.info("round.complete", report_length=len(report_text))
        finally:
            # Cleanup plugin pool if it has a close method
            close_fn = getattr(self.plugin, "close", None)
            if callable(close_fn):
                close_fn()
            self.logger.close()

        return state

    def run_phase(self, phase: int, state: PhaseState) -> PhaseState:
        """Run a single optimization phase with retry loop.

        Control flow:
        1. Prepare state, get phase spec
        2. Run greedy optimizer
        3. Compute final metrics via walk-forward
        4. Evaluate gate criteria
        5. Analyze results and decide: advance, retry scoring, or retry diagnostics
        """
        # 1. Prepare — handle re-runs of completed phases
        self._ensure_state_contract(state)
        if phase in state.completed_phases:
            self._prepare_state_for_phase(state, phase)

        state.start_phase(phase)
        state_path = self.output_dir / "phase_state.json"
        state.save(state_path)

        spec = self.plugin.get_phase_spec(phase, state)

        # Deduplicate candidates by name
        candidates = _dedupe(spec.candidates)

        self.logger.log_phase_start(phase, spec.name, len(candidates))

        scoring_weights = dict(spec.scoring_weights)
        greedy_result = None
        metrics: dict[str, float] | None = None
        checkpoint_path = self.output_dir / f"phase_{phase}_greedy_checkpoint.json"
        force_all_diagnostics = False

        # Get retry limits from spec policy
        max_scoring = spec.analysis_policy.max_scoring_retries
        max_diag = spec.analysis_policy.max_diagnostic_retries

        # 2. Retry loop
        while True:
            # a. Run greedy if needed
            if greedy_result is None:
                checkpoint_context = phase_checkpoint_context(
                    self.contract,
                    phase=phase,
                    spec=spec,
                    scoring_weights=scoring_weights,
                    validation_mode=self.validation_mode,
                )
                evaluate_fn = self.plugin.create_evaluate_batch(
                    phase,
                    state.cumulative_mutations,
                    scoring_weights=scoring_weights,
                    hard_rejects=spec.hard_rejects,
                )
                greedy_result = run_greedy(
                    candidates,
                    state.cumulative_mutations,
                    evaluate_fn,
                    min_delta=spec.min_delta,
                    max_rounds=spec.max_rounds or 20,
                    prune_threshold=spec.prune_threshold or 0.05,
                    checkpoint_path=checkpoint_path,
                    checkpoint_context=checkpoint_context,
                )

                # Log experiment results
                for sc in greedy_result.accepted_experiments:
                    self.logger.log_experiment_result(
                        phase, sc.experiment.name, sc.score, accepted=True,
                    )
                for sc in greedy_result.rejected_experiments:
                    self.logger.log_experiment_result(
                        phase, sc.experiment.name, sc.score, accepted=False,
                        rejected=sc.rejected, reject_reason=sc.reject_reason,
                    )

                # Compute final OOS metrics via walk-forward
                final_validation = {"status": "passed"}
                try:
                    metrics = self.plugin.compute_final_metrics(
                        greedy_result.final_mutations
                    )
                except Exception as exc:
                    log.exception("phase.walk_forward_failed", phase=phase)
                    final_validation = {
                        "status": "failed",
                        "error": str(exc),
                        "mode": self.validation_mode,
                    }
                    if self.validation_mode == "strict":
                        state.mark_phase_invalid(
                            phase,
                            reason="final_validation_failed",
                            error=str(exc),
                            metadata={"contract_hash": self.contract.get("contract_hash", "")},
                        )
                        state.save(state_path)
                        self.logger.save_phase_output(
                            phase,
                            "validation_failure",
                            final_validation,
                        )
                        raise RuntimeError(
                            f"Final validation failed for phase {phase}; strict mode refuses fallback metrics."
                        ) from exc
                    final_validation["status"] = "fallback"
                    final_validation["fallback_source"] = "greedy_in_sample"
                    metrics = (
                        greedy_result.accepted_experiments[-1].metrics
                        if greedy_result.accepted_experiments
                        else {}
                    )
                greedy_result.final_metrics = metrics

                # Save greedy output
                self.logger.save_phase_output(phase, "greedy", {
                    "accepted": [sc.experiment.name for sc in greedy_result.accepted_experiments],
                    "rejected": [sc.experiment.name for sc in greedy_result.rejected_experiments],
                    "final_score": greedy_result.final_score,
                    "base_score": greedy_result.base_score,
                    "rounds": len(greedy_result.rounds),
                    "elapsed_seconds": greedy_result.elapsed_seconds,
                    "contract_hash": self.contract.get("contract_hash", ""),
                    "final_validation": final_validation,
                })

            # b. Gate criteria
            gate_criteria = (
                spec.gate_criteria_fn(metrics)
                if spec.gate_criteria_fn
                else spec.gate_criteria
            )

            # c. Evaluate gate
            gate_result = evaluate_gate(gate_criteria, greedy_result)
            self.logger.log_gate_result(
                phase, gate_result.passed, gate_result.failure_reasons,
                failure_category=gate_result.failure_category,
            )

            # Record gate result
            state.record_gate(phase, {
                "passed": gate_result.passed,
                "failure_reasons": gate_result.failure_reasons,
                "failure_category": gate_result.failure_category,
            })

            # d. Diagnostics
            if force_all_diagnostics:
                diag_text = self.plugin.run_enhanced_diagnostics(
                    phase, state, metrics, greedy_result,
                )
            else:
                diag_text = self.plugin.run_phase_diagnostics(
                    phase, state, metrics, greedy_result,
                )
            self.logger.save_phase_output(phase, "diagnostics", diag_text)

            # e. Analysis
            analysis = analyze_phase(
                phase,
                greedy_result,
                metrics,
                state,
                gate_result,
                ultimate_targets=self.plugin.ultimate_targets,
                policy=spec.analysis_policy,
                current_weights=scoring_weights,
                max_scoring_retries=max_scoring,
                max_diagnostic_retries=max_diag,
            )

            self.logger.log_analysis(phase, analysis.recommendation, analysis.summary)
            self.logger.save_phase_output(phase, "analysis", analysis.report or analysis.summary)

            # f. Check retry budget
            scoring_used = state.scoring_retries.get(phase, 0)
            diag_used = state.diagnostic_retries.get(phase, 0)
            budget_exhausted = (
                scoring_used >= max_scoring and diag_used >= max_diag
            )
            if budget_exhausted and analysis.recommendation != "advance":
                log.info("phase.budget_exhausted", phase=phase)
                analysis.recommendation = "advance"

            # g. Act on recommendation
            if analysis.recommendation == "advance":
                break
            elif analysis.recommendation == "improve_scoring":
                count = state.increment_scoring_retry(phase)
                self.logger.log_retry(phase, "scoring", count)

                # Use weight overrides from analysis if available
                if analysis.scoring_weight_overrides:
                    scoring_weights = analysis.scoring_weight_overrides
                else:
                    scoring_weights = suggest_scoring_adjustment(
                        gate_result, scoring_weights
                    )

                # Add suggested experiments to candidate pool
                if analysis.suggested_experiments:
                    candidates = _dedupe([*candidates, *analysis.suggested_experiments])

                greedy_result = None
                metrics = None
                continue
            elif analysis.recommendation == "improve_diagnostics":
                count = state.increment_diagnostic_retry(phase)
                self.logger.log_retry(phase, "diagnostics", count)
                force_all_diagnostics = True
                continue

        # 3. Accept phase — build result dict
        new_mutations = {
            k: v for k, v in greedy_result.final_mutations.items()
            if k not in state.cumulative_mutations
            or state.cumulative_mutations[k] != v
        }

        phase_result = {
            "focus": spec.focus or spec.name,
            "base_mutations": dict(state.cumulative_mutations),
            "final_mutations": greedy_result.final_mutations,
            "base_score": greedy_result.base_score,
            "final_score": greedy_result.final_score,
            "kept_features": greedy_result.kept_features,
            "rounds": [
                {"round_num": r.round_num, "best_name": r.best_name,
                 "best_score": r.best_score, "kept": r.kept}
                for r in greedy_result.rounds
            ],
            "final_metrics": metrics,
            "final_validation": final_validation,
            "contract_hash": self.contract.get("contract_hash", ""),
            "contract": self.contract,
            "accepted_count": greedy_result.accepted_count,
            "new_mutations": new_mutations,
            "suggested_experiments": [
                e.name for e in analysis.suggested_experiments
            ] if analysis.suggested_experiments else [],
        }

        state.advance_phase(phase, greedy_result.final_mutations, phase_result)
        state.complete_phase(phase)
        state.save(state_path)

        self.logger.log_phase_end(
            phase,
            spec.name,
            accepted=len(greedy_result.accepted_experiments),
            final_score=greedy_result.final_score,
            metrics=metrics,
        )

        # Update progress
        self.logger.update_progress(phase, {
            "name": spec.name,
            "accepted": len(greedy_result.accepted_experiments),
            "final_score": greedy_result.final_score,
            "gate_passed": gate_result.passed,
            "contract_hash": self.contract.get("contract_hash", ""),
        })

        return state

    def _save_optimized_config(self, state: PhaseState) -> Path | None:
        """Apply cumulative mutations to base config and save as JSON.

        Returns the path to the saved config, or None if plugin has no base_config.
        """
        base_config = getattr(self.plugin, "base_config", None)
        if base_config is None:
            return None

        from crypto_trader.optimize.config_mutator import apply_mutations

        optimized = apply_mutations(base_config, state.cumulative_mutations)
        config_dict = {
            "strategy": optimized.to_dict(),
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "contract_hash": self.contract.get("contract_hash", ""),
                "profile_hash": self.contract.get("profile_hash", ""),
                "strategy_config_hash": self.contract.get("strategy_config_hash", ""),
                "portfolio_config_hash": self.contract.get("portfolio_config_hash", ""),
                "data_window": self.contract.get("data_window", {}),
                "data_fingerprint": self.contract.get("data_fingerprint", {}),
                "symbols": self.contract.get("symbols", []),
                "required_timeframes": self.contract.get("required_timeframes", []),
                "contract": self.contract,
            },
        }

        out_path = self.output_dir / "optimized_config.json"
        _atomic_write_json(config_dict, out_path)
        log.info("round.saved_optimized_config", path=str(out_path))
        return out_path

    def run_end_of_round(self, state: PhaseState) -> str:
        """Build and save end-of-round evaluation report."""
        artifacts = self.plugin.build_end_of_round_artifacts(state)

        report_text = build_end_of_round_report(
            self.plugin.name, state, artifacts,
        )

        # Save outputs
        diag_path = self.output_dir / "round_final_diagnostics.txt"
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(artifacts.final_diagnostics_text or "(no diagnostics)")

        eval_path = self.output_dir / "round_evaluation.txt"
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Save optimized config
        self._save_optimized_config(state)

        return report_text
