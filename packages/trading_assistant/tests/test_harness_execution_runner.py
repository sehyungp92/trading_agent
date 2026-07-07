from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.generated_playbook import GeneratedPlaybook
from trading_assistant.schemas.harness_learning import HarnessExecutionInput, HarnessExecutionMode, HarnessVariant
from trading_assistant.schemas.learning_card import CardType, LearningCard
from trading_assistant.skills.harness_execution_runner import HarnessExecutionRunner
from trading_assistant.skills.learning_card_store import LearningCardStore


def _memory(tmp_path: Path) -> Path:
    memory = tmp_path / "memory"
    policy = memory / "policies" / "v1"
    policy.mkdir(parents=True)
    (policy / "agent.md").write_text("agent", encoding="utf-8")
    (policy / "trading_rules.md").write_text("rules", encoding="utf-8")
    (policy / "soul.md").write_text("soul", encoding="utf-8")
    (memory / "findings").mkdir(parents=True)
    return memory


def test_execution_runner_parses_validates_and_does_not_mutate_usage_counts(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    store = LearningCardStore(memory / "findings")
    store.add_card(LearningCard(
        card_type=CardType.SYNTHESIS,
        source_id="card-1",
        bot_id="bot1",
        title="Use monthly gates",
        content="Do not approve unsupported changes.",
        source_workflow="daily_analysis",
        tags=["workflow:daily_analysis", "bot:bot1"],
    ))
    generated_dir = memory / "playbooks" / "generated"
    generated_dir.mkdir(parents=True)
    playbook = GeneratedPlaybook(
        workflow="daily_analysis",
        title="Investigate recurring workflow issues",
        trigger_tags=["workflow:daily_analysis"],
        evidence_refs=["benchmark:a", "benchmark:b", "benchmark:c"],
        trigger_conditions=["Current task matches daily analysis."],
        required_evidence=["Verify current run evidence."],
        steps=["Preserve approval gates."],
        expected_outputs=["Evidence-linked recommendation or no-action note."],
        failure_modes=["Do not bypass approval gates."],
        provenance="Generated from repeated benchmark evidence.",
    )
    (generated_dir / "playbooks.jsonl").write_text(playbook.model_dump_json() + "\n", encoding="utf-8")

    response = {
        "suggestions": [{
            "bot_id": "bot1",
            "title": "Adjust filter",
            "category": "filter_threshold",
            "tier": "parameter",
            "rationale": "Evidence-backed",
            "confidence": 0.7,
            "evidence_paths": [str(evidence)],
        }],
    }
    execution_input = HarnessExecutionInput(
        case_id="case-1",
        workflow="daily_analysis",
        bot_id="bot1",
        allowed_artifacts=[str(evidence)],
        recorded_structured_output=response,
    )

    output = HarnessExecutionRunner(
        memory_dir=memory,
        output_root=tmp_path / "harness_outputs",
    ).run_case(execution_input, mode=HarnessExecutionMode.DETERMINISTIC_ONLY)

    assert output.parse_success is True
    assert output.approved_item_count == 1
    assert output.invalid_evidence_refs == []
    assert output.retrieved_card_ids
    assert output.retrieved_playbook_ids == [playbook.playbook_id]
    assert not (generated_dir / "playbook_tracking.jsonl").exists()
    reloaded = LearningCardStore(memory / "findings").load(force=True).cards[0]
    assert reloaded.retrieval_count == 0


def test_provider_sandbox_disabled_uses_recorded_response(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    execution_input = HarnessExecutionInput(
        case_id="case-2",
        workflow="daily_analysis",
        recorded_structured_output={"suggestions": []},
    )

    output = HarnessExecutionRunner(
        memory_dir=memory,
        output_root=tmp_path / "harness_outputs",
    ).run_case(execution_input, mode=HarnessExecutionMode.PROVIDER_SANDBOX)

    assert output.raw_response_fixture_id == "case-2:recorded"
    assert "provider_sandbox disabled" in output.warnings[0]


def test_execution_runner_applies_variant_prompt_retrieval_and_validator_profile(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    store = LearningCardStore(memory / "findings")
    store.add_card(LearningCard(
        card_type=CardType.SYNTHESIS,
        source_id="case-card",
        bot_id="bot1",
        title="Blocked stop loss prior",
        content="Require frozen evidence.",
        source_workflow="daily_analysis",
        tags=["workflow:daily_analysis", "bot:bot1", "category:stop_loss"],
    ))
    response = {
        "suggestions": [{
            "bot_id": "bot1",
            "title": "Adjust stop",
            "category": "stop_loss",
            "confidence": 0.7,
        }],
    }
    execution_input = HarnessExecutionInput(
        case_id="case-variant",
        workflow="daily_analysis",
        bot_id="bot1",
        retrieval_profile={
            "tags": ["category:stop_loss"],
            "query_terms": ["stop loss prior"],
        },
        allowed_artifacts=[str(evidence)],
        recorded_structured_output=response,
    )
    runner = HarnessExecutionRunner(
        memory_dir=memory,
        output_root=tmp_path / "harness_outputs",
    )

    baseline = runner.run_case(execution_input)
    guarded = runner.run_case(
        execution_input,
        variant=HarnessVariant(
            name="guarded_stop_loss",
            prompt_patch="Require frozen artifact evidence before approving stop-loss changes.",
            retrieval_mode="query_aware",
            validator_profile="guarded",
            route_profile="learned",
        ),
    )

    assert guarded.prompt_package_hash != baseline.prompt_package_hash
    assert guarded.changed_components == ["prompt_patch", "retrieval", "validator", "provider_route"]
    assert guarded.retrieved_card_ids
    assert guarded.approved_item_count == 0
    assert guarded.blocked_item_count == 1
    assert "cited no frozen case evidence" in guarded.validator_notes


def test_execution_runner_rejects_absolute_evidence_outside_workspace(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    outside = tmp_path.parent / f"outside-harness-evidence-{tmp_path.name}.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        execution_input = HarnessExecutionInput(
            case_id="case-outside",
            workflow="daily_analysis",
            bot_id="bot1",
            recorded_structured_output={
                "structural_proposals": [{
                    "bot_id": "bot1",
                    "title": "Unsafe structural idea",
                    "evidence_paths": [str(outside)],
                }],
            },
        )

        output = HarnessExecutionRunner(
            memory_dir=memory,
            output_root=tmp_path / "harness_outputs",
        ).run_case(execution_input)

        assert str(outside) in output.invalid_evidence_refs
        assert "hallucinated artifact path used as evidence" in output.governance_flags
    finally:
        outside.unlink(missing_ok=True)


def test_execution_runner_rejects_root_contained_evidence_not_frozen_in_case(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    unrelated = tmp_path / "memory" / "findings" / "unrelated.json"
    unrelated.write_text("{}", encoding="utf-8")
    execution_input = HarnessExecutionInput(
        case_id="case-unrelated",
        workflow="daily_analysis",
        bot_id="bot1",
        allowed_artifacts=[],
        recorded_structured_output={
            "suggestions": [{
                "bot_id": "bot1",
                "title": "Use unrelated prior evidence",
                "category": "filter_threshold",
                "evidence_paths": [str(unrelated)],
            }],
        },
    )

    output = HarnessExecutionRunner(
        memory_dir=memory,
        output_root=tmp_path / "harness_outputs",
    ).run_case(execution_input)

    assert str(unrelated) in output.invalid_evidence_refs


def test_execution_runner_accepts_canonical_frozen_evidence_ref(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    evidence = memory / "findings" / "frozen.json"
    evidence.write_text("{}", encoding="utf-8")
    execution_input = HarnessExecutionInput(
        case_id="case-canonical",
        workflow="daily_analysis",
        bot_id="bot1",
        allowed_artifacts=["memory/findings/frozen.json"],
        recorded_structured_output={
            "suggestions": [{
                "bot_id": "bot1",
                "title": "Use frozen prior evidence",
                "category": "filter_threshold",
                "evidence_paths": [str(evidence)],
            }],
        },
    )

    output = HarnessExecutionRunner(
        memory_dir=memory,
        output_root=tmp_path / "harness_outputs",
    ).run_case(execution_input)

    assert output.invalid_evidence_refs == []


def test_provider_replay_cannot_borrow_fixture_evidence_refs(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    evidence = memory / "findings" / "frozen.json"
    evidence.write_text("{}", encoding="utf-8")
    execution_input = HarnessExecutionInput(
        case_id="case-provider-no-evidence",
        workflow="daily_analysis",
        bot_id="bot1",
        allowed_artifacts=[str(evidence)],
        recorded_structured_output={
            "suggestions": [{
                "bot_id": "bot1",
                "title": "Recorded fixture used evidence",
                "category": "filter_threshold",
                "evidence_paths": [str(evidence)],
            }],
        },
    )

    def provider_invoker(_package, _run_id: str) -> str:
        return "<!-- STRUCTURED_OUTPUT\n" + json.dumps({
            "suggestions": [{
                "bot_id": "bot1",
                "title": "Provider omitted evidence",
                "category": "filter_threshold",
            }],
        }) + "\n-->"

    output = HarnessExecutionRunner(
        memory_dir=memory,
        output_root=tmp_path / "harness_outputs",
        provider_invoker=provider_invoker,
        allow_provider_sandbox=True,
    ).run_case(
        execution_input,
        variant=HarnessVariant(name="strict-provider", validator_profile="strict"),
        mode=HarnessExecutionMode.PROVIDER_SANDBOX,
    )

    assert output.evidence_refs_used == []
    assert output.approved_item_count == 0
    assert "cited no frozen case evidence" in output.validator_notes
