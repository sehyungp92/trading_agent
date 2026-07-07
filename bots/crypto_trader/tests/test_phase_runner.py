"""Tests for phase_runner — mock 2-phase plugin, verify state transitions."""

from __future__ import annotations

import json
from typing import Any

import pytest

import crypto_trader.optimize.phase_runner as phase_runner_module
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.optimize.types import (
    EndOfRoundArtifacts,
    Experiment,
    GateCriterion,
    GreedyResult,
    PhaseAnalysisPolicy,
    PhaseSpec,
    ScoredCandidate,
)


class MockPlugin:
    """Minimal plugin for testing phase runner."""

    def __init__(self, *, gate_pass: bool = True, num_phases: int = 2):
        self._gate_pass = gate_pass
        self._num_phases_val = num_phases
        self.evaluate_calls = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def num_phases(self) -> int:
        return self._num_phases_val

    @property
    def ultimate_targets(self) -> dict[str, float]:
        return {"total_trades": 50.0, "profit_factor": 2.0}

    @property
    def initial_mutations(self) -> dict[str, Any]:
        return {}

    def get_phase_spec(self, phase: int, state: Any) -> PhaseSpec:
        candidates = [
            Experiment(f"P{phase}_A", {f"p{phase}.a": 1}),
            Experiment(f"P{phase}_B", {f"p{phase}.b": 2}),
        ]

        gate_criteria = [
            GateCriterion("total_trades", ">=", 10.0),
        ]

        return PhaseSpec(
            phase_num=phase,
            name=f"Phase {phase}",
            candidates=candidates,
            scoring_weights={"coverage": 0.5, "risk": 0.5},
            hard_rejects={"max_drawdown_pct": ("<=", 50.0)},
            gate_criteria=gate_criteria,
            analysis_policy=PhaseAnalysisPolicy(
                max_scoring_retries=1, max_diagnostic_retries=0
            ),
            focus=f"Phase {phase}",
        )

    def create_evaluate_batch(
        self, phase, cumulative_mutations, scoring_weights, hard_rejects,
    ):
        self.evaluate_calls += 1

        def evaluate_fn(candidates, current_mutations):
            results = []
            for i, exp in enumerate(candidates):
                # Baseline gets a lower score so experiments show improvement
                if exp.name == "__baseline__":
                    score = 0.1
                else:
                    score = 0.8 - i * 0.1
                results.append(ScoredCandidate(
                    experiment=exp,
                    score=score,
                    metrics={
                        "total_trades": 50.0,
                        "max_drawdown_pct": 15.0,
                        "profit_factor": 2.0,
                    },
                ))
            return results

        return evaluate_fn

    def compute_final_metrics(self, mutations):
        return {
            "total_trades": 50.0,
            "max_drawdown_pct": 15.0,
            "profit_factor": 2.0,
            "win_rate": 55.0,
            "sharpe_ratio": 1.5,
            "calmar_ratio": 3.0,
        }

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        return "mock diagnostics"

    def run_enhanced_diagnostics(self, phase, state, metrics, greedy_result):
        return "mock enhanced diagnostics"

    def build_end_of_round_artifacts(self, state):
        return EndOfRoundArtifacts(
            final_diagnostics_text="mock diagnostics",
            dimension_reports={"Signal": "mock signal report"},
            overall_verdict="mock verdict",
        )


class FailingGatePlugin(MockPlugin):
    """Plugin that returns metrics failing the gate, triggering retries."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diagnostics_calls = 0
        self.enhanced_diagnostics_calls = 0

    def compute_final_metrics(self, mutations):
        return {
            "total_trades": 5.0,  # Below gate threshold of 10
            "max_drawdown_pct": 15.0,
            "profit_factor": 2.0,
            "win_rate": 55.0,
            "sharpe_ratio": 1.5,
            "calmar_ratio": 3.0,
        }

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        self.diagnostics_calls += 1
        return "diagnostics text"

    def run_enhanced_diagnostics(self, phase, state, metrics, greedy_result):
        self.diagnostics_calls += 1
        self.enhanced_diagnostics_calls += 1
        return "enhanced diagnostics text"


class FinalMetricsFailurePlugin(MockPlugin):
    def compute_final_metrics(self, mutations):
        raise RuntimeError("walk-forward unavailable")


class TestPhaseRunner:
    def test_run_single_phase(self, tmp_path):
        plugin = MockPlugin()
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        assert 1 in state.completed_phases
        assert state.current_phase == 2
        assert len(state.cumulative_mutations) > 0
        assert (tmp_path / "phase_state.json").exists()

    def test_run_all_phases(self, tmp_path):
        plugin = MockPlugin(num_phases=2)
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_all_phases(state)

        assert state.completed_phases == [1, 2]
        assert state.current_phase == 3

    def test_skips_completed_phases(self, tmp_path):
        plugin = MockPlugin(num_phases=2)
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()
        state.completed_phases = [1]
        state.current_phase = 2

        runner.run_all_phases(state)

        assert state.completed_phases == [1, 2]
        # Should only have evaluated once (for phase 2)
        assert plugin.evaluate_calls == 1

    def test_activity_log_created(self, tmp_path):
        plugin = MockPlugin()
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        log_path = tmp_path / "phase_activity_log.jsonl"
        assert log_path.exists()
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) > 0

    def test_phase_metrics_stored(self, tmp_path):
        plugin = MockPlugin()
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        assert 1 in state.phase_metrics

    def test_scoring_retry_on_gate_failure(self, tmp_path):
        """Gate failure triggers scoring retry before force-advancing."""
        plugin = FailingGatePlugin()
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        # Should have retried: 1 initial + 1 scoring retry = 2 evaluate calls
        assert plugin.evaluate_calls == 2
        # Phase still advances (budget exhausted)
        assert 1 in state.completed_phases

    def test_budget_exhaustion_forces_advance(self, tmp_path):
        """When all retries are exhausted, phase is force-advanced."""
        plugin = FailingGatePlugin()
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        # Phase completes despite gate failure
        assert 1 in state.completed_phases
        assert state.current_phase == 2

    def test_diagnostic_retry(self, tmp_path):
        """Diagnostic retry runs enhanced diagnostics."""
        plugin = FailingGatePlugin()
        # Policy: 0 scoring retries, 1 diagnostic retry
        orig_get_spec = plugin.get_phase_spec

        def custom_spec(phase, state):
            spec = orig_get_spec(phase, state)
            return PhaseSpec(
                phase_num=spec.phase_num,
                name=spec.name,
                candidates=spec.candidates,
                scoring_weights=spec.scoring_weights,
                hard_rejects=spec.hard_rejects,
                gate_criteria=spec.gate_criteria,
                analysis_policy=PhaseAnalysisPolicy(
                    max_scoring_retries=0, max_diagnostic_retries=1
                ),
            )

        plugin.get_phase_spec = custom_spec
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        # 1 evaluate call (no scoring retries)
        assert plugin.evaluate_calls == 1
        # Diagnostics ran at least twice (initial + enhanced)
        assert plugin.diagnostics_calls >= 2
        assert 1 in state.completed_phases

    def test_end_of_round_report(self, tmp_path):
        """run_all_phases generates end-of-round report."""
        plugin = MockPlugin(num_phases=1)
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_all_phases(state)

        eval_path = tmp_path / "round_evaluation.txt"
        assert eval_path.exists()
        with open(eval_path) as f:
            content = f.read()
        assert "mock" in content.lower()

    def test_progress_json_updated(self, tmp_path):
        """Phase completion updates progress.json."""
        plugin = MockPlugin(num_phases=1)
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        progress_path = tmp_path / "progress.json"
        assert progress_path.exists()

    def test_gate_result_recorded(self, tmp_path):
        """Gate result is stored in state."""
        plugin = MockPlugin()
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        assert 1 in state.phase_gate_results
        assert "passed" in state.phase_gate_results[1]

    def test_phase_output_files_created(self, tmp_path):
        """Per-phase output files are created."""
        plugin = MockPlugin()
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        runner.run_phase(1, state)

        assert (tmp_path / "phase_1_greedy.json").exists()
        assert (tmp_path / "phase_1_diagnostics.txt").exists()
        assert (tmp_path / "phase_1_analysis.txt").exists()

    def test_strict_final_validation_failure_marks_invalid_and_raises(self, tmp_path):
        plugin = FinalMetricsFailurePlugin()
        runner = PhaseRunner(
            plugin,
            tmp_path,
            contract={"contract_hash": "strict_hash"},
            validation_mode="strict",
        )
        state = PhaseState()

        with pytest.raises(RuntimeError, match="strict mode refuses fallback"):
            runner.run_phase(1, state)

        assert state.invalid_phases[1]["reason"] == "final_validation_failed"
        assert state.invalid_phases[1]["metadata"]["contract_hash"] == "strict_hash"
        assert 1 not in state.completed_phases

    def test_fast_validation_failure_uses_annotated_fallback(self, tmp_path):
        plugin = FinalMetricsFailurePlugin()
        runner = PhaseRunner(plugin, tmp_path, validation_mode="fast")
        state = PhaseState()

        runner.run_phase(1, state)

        assert 1 in state.completed_phases
        result = state.phase_results[1]
        assert result["final_validation"]["status"] == "fallback"
        assert result["final_validation"]["fallback_source"] == "greedy_in_sample"

    def test_checkpoint_context_includes_contract_hash(self, tmp_path, monkeypatch):
        plugin = MockPlugin()
        captured: list[dict] = []

        def fake_run_greedy(*args, **kwargs):
            captured.append(json.loads(kwargs["checkpoint_context"]))
            return GreedyResult(
                accepted_experiments=[],
                rejected_experiments=[],
                final_mutations={},
                final_score=0.25,
                base_score=0.1,
                accepted_count=0,
            )

        monkeypatch.setattr(phase_runner_module, "run_greedy", fake_run_greedy)
        runner = PhaseRunner(plugin, tmp_path, contract={"contract_hash": "hash_a"})
        runner.run_phase(1, PhaseState())

        assert captured[0]["contract_hash"] == "hash_a"

    def test_rerun_completed_phase(self, tmp_path):
        """Re-running a completed phase rolls back stale data."""
        plugin = MockPlugin(num_phases=2)
        runner = PhaseRunner(plugin, tmp_path)
        state = PhaseState()

        # Complete both phases
        runner.run_all_phases(state)
        assert state.completed_phases == [1, 2]
        first_mutations = dict(state.cumulative_mutations)

        # Re-run phase 2 — should rollback phase 2 and re-run
        runner.run_phase(2, state)
        assert 2 in state.completed_phases
        assert state.current_phase == 3
