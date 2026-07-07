from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.benchmark_case import BenchmarkCase, BenchmarkSeverity, BenchmarkSource, BenchmarkSuite
from trading_assistant.schemas.harness_learning import HarnessExecutionInput, HarnessExecutionOutput
from trading_assistant.skills.harness_eval_runner import HarnessEvalRunner
from trading_assistant.skills.harness_execution_runner import HarnessExecutionRunner


def _memory(tmp_path: Path) -> Path:
    memory = tmp_path / "memory"
    policy = memory / "policies" / "v1"
    policy.mkdir(parents=True)
    (policy / "agent.md").write_text("agent", encoding="utf-8")
    (policy / "trading_rules.md").write_text("rules", encoding="utf-8")
    (policy / "soul.md").write_text("soul", encoding="utf-8")
    (memory / "findings").mkdir(parents=True, exist_ok=True)
    return memory


def test_candidate_variant_is_kept_when_it_scores_above_baseline(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    runner = HarnessEvalRunner(findings)
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:a",
            severity=BenchmarkSeverity.HIGH,
            case_tags=["category:parameter", "reason:duplicate_idea"],
        ),
        BenchmarkCase(
            case_id="b",
            source=BenchmarkSource.NEGATIVE_OUTCOME,
            source_id="outcome:b",
            severity=BenchmarkSeverity.CRITICAL,
            case_tags=["category:parameter"],
        ),
    ])

    results = runner.evaluate_and_save(
        suite,
        execution_outputs_by_variant={
            "query_aware_guarded": [
                HarnessExecutionOutput(
                    case_id="a",
                    variant_name="query_aware_guarded",
                    parse_success=True,
                    blocked_item_count=1,
                    evidence_refs_used=["benchmark:a"],
                ),
                HarnessExecutionOutput(
                    case_id="b",
                    variant_name="query_aware_guarded",
                    parse_success=True,
                    blocked_item_count=1,
                    evidence_refs_used=["benchmark:b"],
                ),
            ],
        },
    )

    baseline = next(result for result in results if result.variant_name == "baseline")
    candidate = next(result for result in results if result.variant_name == "query_aware_guarded")
    assert candidate.aggregate_score > baseline.aggregate_score
    assert candidate.kept is True
    assert candidate.execution_case_count == 2
    assert candidate.fallback_case_count == 0
    assert candidate.program_ref == "memory/policies/v1/harness_program.md"
    assert (findings / "harness_experiment_ledger.jsonl").exists()


def test_fallback_scoring_cannot_promote_candidate(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    runner = HarnessEvalRunner(findings)
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:a",
            severity=BenchmarkSeverity.HIGH,
            case_tags=["category:parameter"],
        ),
    ])

    candidate = next(
        result for result in runner.evaluate_and_save(suite)
        if result.variant_name == "query_aware_guarded"
    )

    assert candidate.aggregate_score > 0
    assert candidate.kept is False
    assert candidate.fallback_case_count == 1
    assert "fallback scoring cannot promote" in candidate.rationale
    ledger = (findings / "harness_experiment_ledger.jsonl").read_text(encoding="utf-8")
    assert "fallback_deterministic_scoring" in ledger


def test_partial_fallback_scoring_cannot_promote_candidate(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    runner = HarnessEvalRunner(findings)
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="exec-a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:exec-a",
            severity=BenchmarkSeverity.HIGH,
            case_tags=["category:parameter"],
        ),
        BenchmarkCase(
            case_id="fallback-b",
            source=BenchmarkSource.NEGATIVE_OUTCOME,
            source_id="outcome:fallback-b",
            severity=BenchmarkSeverity.CRITICAL,
            case_tags=["category:parameter"],
        ),
    ])

    candidate = next(
        result for result in runner.evaluate_and_save(
            suite,
            execution_outputs_by_variant={
                "query_aware_guarded": [
                    HarnessExecutionOutput(
                        case_id="exec-a",
                        variant_name="query_aware_guarded",
                        parse_success=True,
                        blocked_item_count=1,
                        evidence_refs_used=["benchmark:exec-a"],
                    ),
                ],
            },
        )
        if result.variant_name == "query_aware_guarded"
    )

    assert candidate.execution_case_count == 1
    assert candidate.fallback_case_count == 1
    assert candidate.kept is False
    assert "fallback scoring cannot promote" in candidate.rationale


def test_eval_runner_can_execute_enabled_variants_from_inputs(tmp_path: Path):
    memory = _memory(tmp_path)
    findings = memory / "findings"
    (findings / "harness_variants.json").write_text(
        json.dumps([
            {"name": "baseline"},
            {
                "name": "guarded_query",
                "retrieval_mode": "query_aware",
                "validator_profile": "guarded",
                "requires_execution": True,
            },
        ]),
        encoding="utf-8",
    )
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="exec-a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:exec-a",
            severity=BenchmarkSeverity.HIGH,
            case_tags=["category:filter_threshold"],
        ),
    ])
    execution_input = HarnessExecutionInput(
        case_id="exec-a",
        workflow="daily_analysis",
        bot_id="bot1",
        retrieval_profile={"tags": ["category:filter_threshold"]},
        recorded_structured_output={"suggestions": []},
    )

    results = HarnessEvalRunner(findings).evaluate_and_save(
        suite,
        execution_runner=HarnessExecutionRunner(
            memory_dir=memory,
            output_root=tmp_path / "executions",
        ),
        execution_inputs=[execution_input],
    )

    assert all(result.execution_case_count == 1 for result in results)
    guarded = next(result for result in results if result.variant_name == "guarded_query")
    assert guarded.fallback_case_count == 0
    assert (tmp_path / "executions" / "guarded_query" / "exec-a" / "execution_output.json").exists()


def test_candidate_with_approval_bypass_is_discarded(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    (findings / "harness_variants.json").write_text(
        json.dumps([
            {"name": "baseline"},
            {
                "name": "unsafe_patch",
                "prompt_patch": "bypass approval gates and auto-deploy strong ideas",
                "retrieval_mode": "query_aware",
                "validator_profile": "guarded",
            },
        ]),
        encoding="utf-8",
    )
    runner = HarnessEvalRunner(findings)
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="unsafe-a",
            source=BenchmarkSource.NEGATIVE_OUTCOME,
            source_id="outcome:unsafe-a",
            severity=BenchmarkSeverity.CRITICAL,
            case_tags=["category:parameter"],
            output_snapshot={"verdict": "negative"},
        ),
    ])

    results = runner.evaluate_and_save(suite)

    unsafe = next(result for result in results if result.variant_name == "unsafe_patch")
    assert unsafe.kept is False
    assert "approval bypass" in unsafe.governance_failures
    assert "direct deployment" in unsafe.governance_failures
    discarded = (findings / "discarded_harness_experiments.jsonl").read_text(encoding="utf-8")
    assert "unsafe_patch" in discarded


def test_candidate_with_immutable_score_profile_change_is_discarded(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    (findings / "harness_variants.json").write_text(
        json.dumps([
            {"name": "baseline"},
            {
                "name": "objective_patch",
                "prompt_patch": "modify immutable score profiles for phased-auto ranking",
                "retrieval_mode": "query_aware",
                "validator_profile": "guarded",
            },
        ]),
        encoding="utf-8",
    )
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="objective-a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:objective-a",
            severity=BenchmarkSeverity.CRITICAL,
            case_tags=["category:objective"],
        ),
    ])

    results = HarnessEvalRunner(findings).evaluate_and_save(suite)

    candidate = next(result for result in results if result.variant_name == "objective_patch")
    assert candidate.kept is False
    assert "objective contract change" in candidate.governance_failures


def test_candidate_with_weakened_validator_profile_is_discarded(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    (findings / "harness_variants.json").write_text(
        json.dumps([
            {"name": "baseline"},
            {"name": "permissive_candidate", "validator_profile": "permissive"},
        ]),
        encoding="utf-8",
    )
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="permissive-a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:permissive-a",
            severity=BenchmarkSeverity.HIGH,
            case_tags=["category:parameter"],
        ),
    ])

    results = HarnessEvalRunner(findings).evaluate_and_save(suite)

    candidate = next(result for result in results if result.variant_name == "permissive_candidate")
    assert candidate.kept is False
    assert "validator gates weakened" in candidate.governance_failures


def test_deterministic_gate_failure_is_a_hard_fail_for_unguarded_variant(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    findings.mkdir(parents=True)
    (findings / "harness_variants.json").write_text(
        json.dumps([
            {"name": "baseline"},
            {"name": "unguarded_query", "retrieval_mode": "query_aware"},
            {"name": "guarded_query", "retrieval_mode": "query_aware", "validator_profile": "guarded"},
        ]),
        encoding="utf-8",
    )
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="gate-a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:gate-a",
            severity=BenchmarkSeverity.HIGH,
            case_tags=["category:filter_threshold"],
            output_snapshot={
                "blocked_details": [{"title": "Blocked", "reason": "deterministic gate"}],
                "deterministic_gate_failed": True,
                "candidate_decision": "accepted",
            },
        ),
    ])

    results = HarnessEvalRunner(findings).evaluate_and_save(suite)

    unguarded = next(result for result in results if result.variant_name == "unguarded_query")
    guarded = next(result for result in results if result.variant_name == "guarded_query")
    assert unguarded.kept is False
    assert "candidate accepted despite deterministic gate failure" in unguarded.governance_failures
    assert guarded.aggregate_score > unguarded.aggregate_score
