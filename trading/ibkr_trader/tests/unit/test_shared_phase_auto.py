from pathlib import Path
from types import SimpleNamespace

import pytest

import backtests.shared.auto.plugin_utils as plugin_utils_module
import backtests.shared.auto.phase_runner as phase_runner_module
import backtests.stock.analysis.iaric_pullback_diagnostics as pullback_diagnostics_module
import backtests.stock.analysis.iaric_pullback_round_diagnostics as pullback_round_module
import backtests.stock.auto.iaric.plugin as pullback_plugin_module
import backtests.stock.cli as stock_cli_module
from backtests.momentum.auto.downturn.plugin import DownturnPlugin
from backtests.shared.auto.phase_analyzer import analyze_phase
from backtests.shared.auto.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    mutation_signature,
    pool_map_with_heartbeat,
)
from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.round_manager import RoundManager
from backtests.shared.auto.types import (
    EndOfRoundArtifacts,
    Experiment,
    GateCriterion,
    GateResult,
    GreedyResult,
    PhaseDecision,
    ScoredCandidate,
)
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin


def _greedy_result(*, base_score: float, final_score: float, accepted_count: int = 1) -> GreedyResult:
    return GreedyResult(
        base_score=base_score,
        final_score=final_score,
        final_mutations={},
        kept_features=["kept"] if accepted_count else [],
        rounds=[],
        final_metrics={},
        total_candidates=1,
        accepted_count=accepted_count,
        elapsed_seconds=0.0,
    )


def test_analyze_phase_respects_phase_specific_min_score_delta():
    gate_result = GateResult(
        passed=False,
        criteria=(GateCriterion("avg_r", 1.0, 0.7, False),),
    )
    greedy_result = _greedy_result(base_score=100.0, final_score=100.5)
    metrics = {"avg_r": 0.7}

    effective = analyze_phase(
        1,
        greedy_result,
        metrics,
        PhaseState(),
        gate_result,
        ultimate_targets={"avg_r": 1.0},
        policy=PhaseAnalysisPolicy(focus_metrics=["avg_r"], min_effective_score_delta_pct=0.0),
    )
    marginal = analyze_phase(
        1,
        greedy_result,
        metrics,
        PhaseState(),
        gate_result,
        ultimate_targets={"avg_r": 1.0},
        policy=PhaseAnalysisPolicy(focus_metrics=["avg_r"], min_effective_score_delta_pct=0.01),
    )

    assert effective.scoring_assessment == "EFFECTIVE"
    assert marginal.scoring_assessment == "MARGINAL"
    assert marginal.recommendation == "improve_scoring"


def test_dual_track_worker_split_balances_total_budget():
    assert stock_cli_module._split_dual_track_workers(None) == (None, None)
    assert stock_cli_module._split_dual_track_workers(1) == (1, 1)
    assert stock_cli_module._split_dual_track_workers(4) == (2, 2)
    assert stock_cli_module._split_dual_track_workers(5) == (3, 2)


def test_analyze_phase_policy_decision_overrides_shared_fallback():
    def decide_action(*args, **kwargs):
        return PhaseDecision(
            action="advance",
            reason="custom phase logic prefers advancing",
            extra_suggested_experiments=[Experiment("extra_probe", {"foo": 1})],
        )

    analysis = analyze_phase(
        1,
        _greedy_result(base_score=10.0, final_score=11.0),
        {"avg_r": 0.4},
        PhaseState(),
        GateResult(passed=False, criteria=(GateCriterion("avg_r", 1.0, 0.4, False),)),
        ultimate_targets={"avg_r": 1.0},
        policy=PhaseAnalysisPolicy(
            focus_metrics=["avg_r"],
            diagnostic_gap_fn=lambda phase, metrics: ["missing detail"],
            decide_action_fn=decide_action,
        ),
    )

    assert analysis.recommendation == "advance"
    assert analysis.recommendation_reason == "custom phase logic prefers advancing"
    assert [experiment.name for experiment in analysis.suggested_experiments] == ["extra_probe"]


def test_analyze_phase_ignores_infeasible_policy_decision():
    state = PhaseState(scoring_retries={1: 2})

    analysis = analyze_phase(
        1,
        _greedy_result(base_score=10.0, final_score=11.0),
        {"avg_r": 0.4},
        state,
        GateResult(passed=False, criteria=(GateCriterion("avg_r", 1.0, 0.4, False),)),
        ultimate_targets={"avg_r": 1.0},
        policy=PhaseAnalysisPolicy(
            focus_metrics=["avg_r"],
            decide_action_fn=lambda *args, **kwargs: PhaseDecision(
                action="improve_scoring",
                reason="force another score retry",
            ),
        ),
        max_scoring_retries=2,
    )

    assert analysis.recommendation == "advance"
    assert "Retry budget exhausted" in analysis.recommendation_reason


def test_analyze_phase_prefers_diagnostics_for_diagnostic_needed_failures():
    analysis = analyze_phase(
        1,
        _greedy_result(base_score=10.0, final_score=11.5),
        {"avg_r": 0.86},
        PhaseState(),
        GateResult(
            passed=False,
            criteria=(GateCriterion("avg_r", 1.0, 0.86, False),),
            failure_category="diagnostic_needed",
        ),
        ultimate_targets={"avg_r": 1.0},
        policy=PhaseAnalysisPolicy(focus_metrics=["avg_r"]),
    )

    assert analysis.scoring_assessment == "MARGINAL"
    assert analysis.recommendation == "improve_diagnostics"


def test_analyze_phase_marks_focus_metrics_far_from_target_as_misaligned_without_prior_metrics():
    analysis = analyze_phase(
        1,
        _greedy_result(base_score=100.0, final_score=103.0),
        {"profit_factor": 1.0, "net_return_pct": 3.0, "bear_alpha_pct": 2.0},
        PhaseState(),
        GateResult(
            passed=False,
            criteria=(GateCriterion("profit_factor", 2.0, 1.0, False),),
            failure_category="scoring_ineffective",
        ),
        ultimate_targets={"profit_factor": 3.0, "net_return_pct": 10.0, "bear_alpha_pct": 20.0},
        policy=PhaseAnalysisPolicy(
            focus_metrics=["profit_factor", "net_return_pct", "bear_alpha_pct"],
            min_effective_score_delta_pct=0.01,
        ),
    )

    assert analysis.scoring_assessment == "MISALIGNED"
    assert analysis.recommendation == "improve_scoring"


def test_cached_batch_evaluator_reuses_seeded_and_duplicate_mutation_results():
    class _Delegate:
        def __init__(self):
            self.calls = []

        def __call__(self, candidates, current_mutations):
            self.calls.append([candidate.name for candidate in candidates])
            results = []
            for candidate in candidates:
                merged = dict(current_mutations)
                merged.update(candidate.mutations)
                results.append(
                    ScoredCandidate(
                        name=candidate.name,
                        score=float(len(merged)),
                        metrics={"merged_size": float(len(merged))},
                    )
                )
            return results

    delegate = _Delegate()
    evaluator = CachedBatchEvaluator(
        delegate,
        seed_results={
            mutation_signature({"base": 1}): ScoredCandidate(
                name="baseline",
                score=1.0,
                metrics={"merged_size": 1.0},
            )
        },
    )

    results = evaluator(
        [
            Experiment("baseline_alias", {}),
            Experiment("dup_a", {"x": 1}),
            Experiment("dup_b", {"x": 1}),
            Experiment("other", {"y": 2}),
        ],
        {"base": 1},
    )

    assert delegate.calls == [["dup_a", "other"]]
    assert [result.name for result in results] == ["baseline_alias", "dup_a", "dup_b", "other"]
    assert results[0].score == 1.0
    assert results[1].score == results[2].score == 2.0
    assert results[1].metrics == results[2].metrics


def test_resilient_batch_evaluator_falls_back_to_local_after_parallel_permission_error():
    class _BrokenDelegate:
        def __init__(self):
            self.calls = 0
            self.terminated = False

        def __call__(self, candidates, current_mutations):
            del candidates, current_mutations
            self.calls += 1
            raise PermissionError("access denied")

        def terminate(self):
            self.terminated = True

    class _LocalDelegate:
        def __init__(self):
            self.calls = 0

        def __call__(self, candidates, current_mutations):
            del current_mutations
            self.calls += 1
            return [ScoredCandidate(name=candidate.name, score=5.0, metrics={"ok": 1.0}) for candidate in candidates]

        def close(self):
            return None

    broken = _BrokenDelegate()
    local = _LocalDelegate()
    evaluator = ResilientBatchEvaluator(
        preferred_factory=lambda: broken,
        fallback_factory=lambda: local,
        description="test evaluator",
    )

    first = evaluator([Experiment("candidate", {"x": 1})], {})
    second = evaluator([Experiment("candidate2", {"y": 2})], {})

    assert broken.calls == 1
    assert broken.terminated
    assert local.calls == 2
    assert [result.name for result in first] == ["candidate"]
    assert [result.name for result in second] == ["candidate2"]


def test_resilient_batch_evaluator_falls_back_to_local_after_parallel_timeout():
    class _BrokenDelegate:
        def __init__(self):
            self.calls = 0
            self.terminated = False

        def __call__(self, candidates, current_mutations):
            del candidates, current_mutations
            self.calls += 1
            raise TimeoutError("stalled worker pool")

        def terminate(self):
            self.terminated = True

    class _LocalDelegate:
        def __init__(self):
            self.calls = 0

        def __call__(self, candidates, current_mutations):
            del current_mutations
            self.calls += 1
            return [ScoredCandidate(name=candidate.name, score=3.0) for candidate in candidates]

    broken = _BrokenDelegate()
    local = _LocalDelegate()
    evaluator = ResilientBatchEvaluator(
        preferred_factory=lambda: broken,
        fallback_factory=lambda: local,
        description="timeout evaluator",
    )

    result = evaluator([Experiment("candidate", {"x": 1})], {})

    assert broken.calls == 1
    assert broken.terminated
    assert local.calls == 1
    assert [item.name for item in result] == ["candidate"]


def test_shared_pool_batch_evaluator_closes_owned_pool():
    class _AsyncResult:
        def ready(self):
            return True

        def get(self):
            return "ok"

    class _Pool:
        def __init__(self):
            self.calls = []
            self._pool = []

        def apply_async(self, worker_fn, args):
            self.calls.append((worker_fn, args))
            return _AsyncResult()

    pool = _Pool()
    close_calls = {"count": 0}
    worker_fn = object()
    evaluator = SharedPoolBatchEvaluator(
        pool,
        worker_fn=worker_fn,
        build_args=lambda candidates, current_mutations: [(c.name, current_mutations) for c in candidates],
        on_close=lambda: close_calls.__setitem__("count", close_calls["count"] + 1),
        on_terminate=None,
        description="owned pool batch",
    )

    result = evaluator([Experiment("candidate", {"x": 1})], {})
    evaluator.close()

    assert result == ["ok"]
    assert pool.calls == [(worker_fn, (("candidate", {}),))]
    assert close_calls["count"] == 1


def test_pool_map_with_heartbeat_times_out_on_stalled_pool(monkeypatch):
    clock = {"t": 0.0}

    class _AsyncResult:
        def ready(self):
            return False

    class _Pool:
        def __init__(self):
            self._pool = []

        def apply_async(self, worker_fn, args):
            del worker_fn, args
            return _AsyncResult()

    monkeypatch.setattr(plugin_utils_module.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(plugin_utils_module.time, "sleep", lambda seconds: clock.__setitem__("t", clock["t"] + seconds))

    with pytest.raises(TimeoutError, match="stalled batch exceeded timeout"):
        pool_map_with_heartbeat(
            _Pool(),
            object(),
            [("candidate",)],
            description="stalled batch",
            heartbeat_seconds=1.0,
            per_candidate_timeout_seconds=2.0,
            minimum_timeout_seconds=2.0,
        )


def test_shared_pool_batch_evaluator_detects_dead_workers(monkeypatch):
    clock = {"t": 0.0}

    class _DeadWorker:
        pid = 1234
        exitcode = 1

        def is_alive(self):
            return False

    class _AsyncResult:
        def ready(self):
            return False

    class _Pool:
        def __init__(self):
            self._pool = [_DeadWorker()]

        def apply_async(self, worker_fn, args):
            del worker_fn, args
            return _AsyncResult()

    monkeypatch.setattr(plugin_utils_module.time, "monotonic", lambda: clock["t"])

    evaluator = SharedPoolBatchEvaluator(
        _Pool(),
        worker_fn=object(),
        build_args=lambda candidates, current_mutations: [(c.name, current_mutations) for c in candidates],
        on_terminate=None,
        description="dead worker batch",
    )

    with pytest.raises(RuntimeError, match="worker exited unexpectedly"):
        evaluator([Experiment("candidate", {"x": 1})], {})


def test_pool_map_with_heartbeat_logs_grouped_progress(monkeypatch):
    clock = {"t": 0.0}
    messages: list[str] = []

    class _Logger:
        def info(self, message, *args):
            messages.append(message % args)

    class _AsyncResult:
        def __init__(self, ready_at: float, value: str):
            self._ready_at = ready_at
            self._value = value

        def ready(self):
            return clock["t"] >= self._ready_at

        def get(self):
            return self._value

    class _Worker:
        def is_alive(self):
            return True

        exitcode = None
        pid = 1000

    class _Pool:
        def __init__(self):
            self._pool = [_Worker(), _Worker()]
            self._submitted = 0

        def apply_async(self, worker_fn, args):
            del worker_fn
            ready_at = 0.0 if self._submitted < 6 else 1.0
            result = _AsyncResult(ready_at, args[0][0])
            self._submitted += 1
            return result

    monkeypatch.setattr(plugin_utils_module.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(plugin_utils_module.time, "sleep", lambda seconds: clock.__setitem__("t", clock["t"] + seconds))

    results = pool_map_with_heartbeat(
        _Pool(),
        object(),
        [(f"cand_{index}",) for index in range(12)],
        description="grouped batch",
        logger=_Logger(),
        heartbeat_seconds=5.0,
        per_candidate_timeout_seconds=10.0,
        minimum_timeout_seconds=10.0,
    )

    progress_messages = [message for message in messages if "progress:" in message]
    assert results == [f"cand_{index}" for index in range(12)]
    assert len(progress_messages) == 2
    assert "6/12 completed" in progress_messages[0]
    assert "12/12 completed" in progress_messages[1]


def test_phase_runner_reruns_greedy_with_phase_specific_weights_and_candidates(tmp_path, monkeypatch):
    class _DummyPlugin:
        name = "dummy"
        num_phases = 1
        ultimate_targets = {"avg_r": 1.0}

        def __init__(self):
            self.weight_calls = []

        def get_phase_spec(self, phase, state):
            return PhaseSpec(
                focus="test",
                candidates=[Experiment("seed", {"_seed": 0})],
                gate_criteria_fn=lambda metrics: [GateCriterion("avg_r", 1.0, metrics["avg_r"], metrics["avg_r"] >= 1.0)],
                scoring_weights={"quality": 0.5, "risk": 0.5},
                hard_rejects={},
                analysis_policy=PhaseAnalysisPolicy(
                    focus_metrics=["avg_r"],
                    min_effective_score_delta_pct=0.01,
                    suggest_experiments_fn=lambda phase, metrics, weaknesses, state: [Experiment("followup", {"x": 1})],
                    redesign_scoring_weights_fn=lambda phase, current_weights, analysis, gate_result: {"quality": 1.0},
                ),
                max_rounds=2,
                prune_threshold=0.0,
            )

        def create_evaluate_batch(self, phase, cumulative_mutations, *, scoring_weights=None, hard_rejects=None):
            self.weight_calls.append(dict(scoring_weights or {}))
            return lambda candidates, current_mutations: []

        def compute_final_metrics(self, mutations):
            return {"avg_r": 0.25}

        def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
            return "baseline"

        def run_enhanced_diagnostics(self, phase, state, metrics, greedy_result):
            return "enhanced"

        def build_end_of_round_artifacts(self, state):
            return EndOfRoundArtifacts(
                final_diagnostics_text="final diag",
                dimension_reports={"signal_extraction": "ok"},
                overall_verdict="done",
            )

    plugin = _DummyPlugin()
    candidate_counts = []
    greedy_calls = {"count": 0}
    gate_calls = {"count": 0}

    def fake_run_greedy(candidates, base_mutations, evaluate_batch, **kwargs):
        greedy_calls["count"] += 1
        candidate_counts.append(len(candidates))
        if greedy_calls["count"] == 1:
            return _greedy_result(base_score=100.0, final_score=100.5)
        return _greedy_result(base_score=100.0, final_score=102.0)

    def fake_evaluate_gate(criteria, greedy_result=None):
        gate_calls["count"] += 1
        return GateResult(
            passed=gate_calls["count"] > 1,
            criteria=tuple(criteria),
        )

    monkeypatch.setattr(phase_runner_module, "run_greedy", fake_run_greedy)
    monkeypatch.setattr(phase_runner_module, "evaluate_gate", fake_evaluate_gate)

    runner = PhaseRunner(plugin=plugin, output_dir=Path(tmp_path), max_retries=2)
    state = runner.run_phase(1, runner.load_state())

    assert state.completed_phases == [1]
    assert candidate_counts == [1, 2]
    assert plugin.weight_calls == [{"quality": 0.5, "risk": 0.5}, {"quality": 1.0}]


def test_phase_runner_writes_enhanced_diagnostics_and_final_round_artifacts(tmp_path, monkeypatch):
    class _DummyPlugin:
        name = "dummy"
        num_phases = 1
        ultimate_targets = {"avg_r": 1.0}

        def get_phase_spec(self, phase, state):
            return PhaseSpec(
                focus="test",
                candidates=[],
                gate_criteria_fn=lambda metrics: [GateCriterion("avg_r", 1.0, metrics["avg_r"], False)],
                scoring_weights={},
                hard_rejects={},
                analysis_policy=PhaseAnalysisPolicy(
                    focus_metrics=[],
                    diagnostic_gap_fn=lambda phase, metrics: ["need more diagnostics"],
                ),
                max_rounds=1,
                prune_threshold=0.0,
            )

        def create_evaluate_batch(self, phase, cumulative_mutations, *, scoring_weights=None, hard_rejects=None):
            return lambda candidates, current_mutations: []

        def compute_final_metrics(self, mutations):
            return {"avg_r": 0.2}

        def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
            return "baseline diag"

        def run_enhanced_diagnostics(self, phase, state, metrics, greedy_result):
            return "enhanced diag"

        def build_end_of_round_artifacts(self, state):
            return EndOfRoundArtifacts(
                final_diagnostics_text="final diag",
                dimension_reports={"signal_extraction": "ok"},
                overall_verdict="done",
            )

    monkeypatch.setattr(
        phase_runner_module,
        "run_greedy",
        lambda *args, **kwargs: _greedy_result(base_score=1.0, final_score=2.0),
    )

    runner = PhaseRunner(plugin=_DummyPlugin(), output_dir=Path(tmp_path), max_diagnostic_retries=1)
    runner.run_phase(1, runner.load_state())

    assert (Path(tmp_path) / "phase_1_diagnostics.txt").read_text(encoding="utf-8") == "baseline diag"
    assert (Path(tmp_path) / "phase_1_diagnostics_enhanced.txt").read_text(encoding="utf-8") == "enhanced diag"
    assert (Path(tmp_path) / "round_evaluation.txt").exists()
    assert (Path(tmp_path) / "round_final_diagnostics.txt").read_text(encoding="utf-8") == "final diag"


def test_phase_runner_writes_run_spec_for_incremental_round_runs(tmp_path, monkeypatch):
    class _DummyPlugin:
        name = "dummy"
        num_phases = 1
        initial_mutations = {"flags.seed": True}
        ultimate_targets = {"avg_r": 1.0}

        def get_phase_spec(self, phase, state):
            return PhaseSpec(
                focus="test",
                candidates=[],
                gate_criteria_fn=lambda metrics: [GateCriterion("avg_r", 1.0, metrics["avg_r"], True)],
                scoring_weights={},
                hard_rejects={},
                analysis_policy=PhaseAnalysisPolicy(focus_metrics=[]),
                max_rounds=1,
                prune_threshold=0.0,
            )

        def create_evaluate_batch(self, phase, cumulative_mutations, *, scoring_weights=None, hard_rejects=None):
            return lambda candidates, current_mutations: []

        def compute_final_metrics(self, mutations):
            return {"avg_r": 1.2}

        def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
            return "phase diag"

        def build_end_of_round_artifacts(self, state):
            return EndOfRoundArtifacts(
                final_diagnostics_text="final diag",
                dimension_reports={"signal_extraction": "ok"},
                overall_verdict="done",
            )

    monkeypatch.setattr(
        phase_runner_module,
        "run_greedy",
        lambda *args, **kwargs: _greedy_result(base_score=1.0, final_score=2.0),
    )

    round_manager = RoundManager("momentum", "sample", base_dir=Path(tmp_path) / "output")
    round_dir = round_manager.get_round_dir(1)
    runner = PhaseRunner(
        plugin=_DummyPlugin(),
        output_dir=round_dir,
        round_manager=round_manager,
        round_num=1,
    )

    runner.run_phase(1, runner.load_state())

    run_spec = (round_dir / "run_spec.json").read_text(encoding="utf-8")
    assert '"round": 1' in run_spec
    assert '"baseline_mutations": {' in run_spec
    assert '"flags.seed": true' in run_spec


def test_downturn_policy_forces_rescoring_on_negative_correction_pnl():
    plugin = DownturnPlugin(Path("."))
    plugin._last_context = {
        "trades": [
            SimpleNamespace(pnl=-10.0, in_correction_window=True),
            SimpleNamespace(pnl=25.0, in_correction_window=False),
        ]
    }
    state = PhaseState()
    spec = plugin.get_phase_spec(2, state)
    policy = PhaseAnalysisPolicy(
        focus_metrics=spec.analysis_policy.focus_metrics,
        min_effective_score_delta_pct=spec.analysis_policy.min_effective_score_delta_pct,
        diagnostic_gap_fn=lambda phase, metrics: [],
        suggest_experiments_fn=lambda phase, metrics, weaknesses, state: [],
        redesign_scoring_weights_fn=spec.analysis_policy.redesign_scoring_weights_fn,
        build_extra_analysis_fn=lambda phase, metrics, state, greedy_result: {
            "engine_health": {},
            "correction_attribution": {"correction_pnl": -10.0, "non_correction_pnl": 25.0, "ratio": -0.67},
        },
        format_extra_analysis_fn=spec.analysis_policy.format_extra_analysis_fn,
        decide_action_fn=spec.analysis_policy.decide_action_fn,
    )

    analysis = analyze_phase(
        2,
        _greedy_result(base_score=1.0, final_score=1.4),
        {
            "correction_alpha_pct": 5.0,
            "total_trades": 20.0,
            "exit_efficiency": 0.22,
            "profit_factor": 1.25,
            "max_dd_pct": 0.20,
            "calmar": 0.7,
            "sharpe": 0.4,
            "signal_to_entry_ratio": 0.15,
        },
        state,
        GateResult(passed=False, criteria=(GateCriterion("correction_alpha_pct", 10.0, 5.0, False),)),
        ultimate_targets=plugin.ultimate_targets,
        policy=policy,
        current_weights=spec.scoring_weights,
    )

    assert analysis.recommendation == "improve_scoring"
    assert analysis.scoring_assessment == "MISALIGNED"
    assert analysis.scoring_weight_overrides is not None


def test_iaric_pullback_builds_final_diagnostics_artifact(monkeypatch):
    plugin = IARICPullbackPlugin(Path("."))
    state = PhaseState(
        cumulative_mutations={"final": 1},
        phase_results={
            4: {"final_mutations": {"phase4": 1}},
            5: {
                "base_score": 1.0,
                "final_score": 2.0,
                "final_mutations": {"final": 1},
                "kept_features": ["carry5"],
                "rounds": [],
                "accepted_count": 1,
            },
        },
    )

    monkeypatch.setattr(pullback_plugin_module, "_bucket_lines", lambda *args, **kwargs: ["  bucket line"])
    monkeypatch.setattr(pullback_plugin_module, "_weekday_lines", lambda *args, **kwargs: ["  weekday line"])
    monkeypatch.setattr(pullback_plugin_module, "_exit_mix_lines", lambda *args, **kwargs: ["  exit line"])
    monkeypatch.setattr(pullback_plugin_module, "_best_variant", lambda rows: "best variant")
    monkeypatch.setattr(
        pullback_diagnostics_module,
        "compute_pullback_diagnostic_snapshot",
        lambda *args, **kwargs: {
            "shadow": {"shadow": {"avg_r": 0.1}},
            "funnel": {"accept_rate": 0.25},
            "selection": {"entered_avg_r": 0.2, "skipped_avg_shadow_r": 0.05},
            "entry_timing": [{"label": "open", "avg_r": 0.2}],
            "carry_funnel": {"flow_ok": 3, "profitable": 2},
            "exit_frontier": [{"label": "frontier", "avg_r": 0.3}],
        },
    )
    monkeypatch.setattr(
        pullback_round_module,
        "build_pullback_round_comparison_report",
        lambda *args, **kwargs: "round comparison",
    )
    monkeypatch.setattr(
        IARICPullbackPlugin,
        "run_enhanced_diagnostics",
        lambda self, phase, state, metrics, greedy_result: "final full diagnostics",
    )
    monkeypatch.setattr(IARICPullbackPlugin, "_run_ablation_suite", lambda self, state, **kwargs: ["  ablation line"])
    monkeypatch.setattr(IARICPullbackPlugin, "_run_temporal_walkforward", lambda self, mutations: ["  wf line"])

    final_metrics = {
        "avg_r": 0.24,
        "expected_total_r": 31.2,
        "profit_factor": 2.5,
        "sharpe": 1.6,
        "max_drawdown_pct": 0.04,
        "managed_exit_share": 0.65,
        "eod_flatten_share": 0.35,
        "total_trades": 130.0,
        "rsi_depth_edge": 0.10,
        "trend_band_edge": 0.06,
        "late_rank_edge": 0.05,
        "mean_entry_rank": 6.0,
        "carry_trade_share": 0.20,
        "carry_avg_r": 0.30,
        "stop_hit_total_r": -2.0,
        "stop_hit_avg_r": -0.5,
        "positive_eod_share": 0.60,
    }
    base_metrics = {
        **final_metrics,
        "avg_r": 0.18,
        "expected_total_r": 23.4,
        "profit_factor": 2.2,
        "managed_exit_share": 0.25,
        "eod_flatten_share": 0.75,
        "rsi_depth_edge": 0.08,
        "trend_band_edge": 0.03,
        "late_rank_edge": 0.03,
    }

    contexts = {
        (("final", 1),): {"metrics": final_metrics, "trades": [], "replay": object(), "candidate_ledger": {}, "funnel_counters": {}, "rejection_log": [], "shadow_outcomes": [], "selection_attribution": {}, "daily_selections": {}},
        tuple(sorted(plugin.initial_mutations.items())): {"metrics": base_metrics, "trades": [], "replay": object(), "candidate_ledger": {}, "funnel_counters": {}, "rejection_log": [], "shadow_outcomes": [], "selection_attribution": {}, "daily_selections": {}},
        (("phase4", 1),): {"metrics": final_metrics, "trades": [], "replay": object(), "candidate_ledger": {}, "funnel_counters": {}, "rejection_log": [], "shadow_outcomes": [], "selection_attribution": {}, "daily_selections": {}},
    }

    def fake_run_config(self, mutations, *, start_date=None, end_date=None, store_context=False, collect_diagnostics=False):
        key = tuple(sorted((mutations or {}).items()))
        context = contexts[key]
        if store_context:
            self._last_context = context
        return context

    monkeypatch.setattr(IARICPullbackPlugin, "_run_config", fake_run_config)

    artifacts = plugin.build_end_of_round_artifacts(state)

    assert artifacts.final_diagnostics_text == "final full diagnostics"
    assert set(artifacts.dimension_reports) == {
        "signal_extraction",
        "signal_discrimination",
        "entry_mechanism",
        "trade_management",
        "exit_mechanism",
    }
    assert "robustness" in artifacts.extra_sections
    assert "round_comparison" in artifacts.extra_sections
