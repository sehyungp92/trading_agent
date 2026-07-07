from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.schemas.strategy_change_ledger import StrategyChangeRecord, StrategyChangeRecordType
from trading_assistant.skills.run_recall_summarizer import RunRecallSummarizer
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger


def test_focused_recall_loads_existing_cards_with_provenance(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    findings = memory / "findings"
    findings.mkdir(parents=True)
    evidence = findings / "harness_eval_results.jsonl"
    evidence.write_text("{}", encoding="utf-8")
    (findings / "focused_recall_cards.jsonl").write_text(
        json.dumps({
            "title": "Harness review: unsafe_patch",
            "content": json.dumps({"governance_failures": ["approval bypass"]}),
            "source_workflow": "harness_meta_learning",
            "evidence_summary": "memory/findings/harness_eval_results.jsonl",
            "tags": ["workflow:harness_meta_learning"],
        }) + "\n",
        encoding="utf-8",
    )

    cards = RunRecallSummarizer(memory).summarize(workflow="harness_meta_learning")

    assert len(cards) == 1
    assert cards[0].validator_gate_status == "blocked_or_warning"
    assert cards[0].evidence_paths == ["memory/findings/harness_eval_results.jsonl"]


def test_context_builder_prefers_focused_recall_over_similar_snippets(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    (memory / "policies" / "v1").mkdir(parents=True)
    findings = memory / "findings"
    findings.mkdir(parents=True)
    evidence = findings / "learning_reviews.jsonl"
    evidence.write_text("{}", encoding="utf-8")
    (findings / "focused_recall_cards.jsonl").write_text(
        json.dumps({
            "run_id": "run-1",
            "workflow": "daily_analysis",
            "proposal_or_finding_summary": "Prior blocked idea",
            "evidence_paths": ["memory/findings/learning_reviews.jsonl"],
            "how_this_matters_now": "Do not repeat without monthly evidence.",
        }) + "\n",
        encoding="utf-8",
    )

    class RunIndexStub:
        def search(self, **_: object) -> list[dict]:
            return [{"run_id": "snippet", "agent_type": "daily_analysis", "snippet": "raw"}]

    package = ContextBuilder(memory, run_index=RunIndexStub()).base_package(agent_type="daily_analysis")

    assert "focused_run_recall" in package.data
    assert "similar_past_runs" not in package.data


def test_focused_recall_does_not_attach_unrelated_latest_outcome(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    findings = memory / "findings"
    findings.mkdir(parents=True)
    evidence = findings / "learning_reviews.jsonl"
    evidence.write_text("{}", encoding="utf-8")
    (findings / "focused_recall_cards.jsonl").write_text(
        json.dumps({
            "run_id": "run-old",
            "workflow": "daily_analysis",
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "proposal_or_finding_summary": "Prior idea without measured outcome",
            "evidence_paths": ["memory/findings/learning_reviews.jsonl"],
            "how_this_matters_now": "Verify first.",
        }) + "\n",
        encoding="utf-8",
    )
    (findings / "monthly_outcomes.jsonl").write_text(
        json.dumps({
            "run_id": "different-run",
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "verdict": "keep",
            "run_month": "2026-04",
        }) + "\n",
        encoding="utf-8",
    )

    cards = RunRecallSummarizer(memory).summarize(
        workflow="daily_analysis",
        bot_id="bot1",
        strategy_id="strat1",
    )

    assert len(cards) == 1
    assert cards[0].outcome_status == ""


def test_focused_recall_reads_strategy_change_ledger_projection(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    findings = memory / "findings"
    findings.mkdir(parents=True)
    evidence = findings / "learning_reviews.jsonl"
    evidence.write_text("{}", encoding="utf-8")
    record = StrategyChangeRecord(
        bot_id="bot1",
        strategy_id="strat1",
        record_type=StrategyChangeRecordType.PROPOSED_CHANGE,
        approval_request_id="req-123",
        decision_reason="candidate ready",
        run_id="monthly-bot1-strat1-2026-04",
        run_month="2026-04",
        evidence_paths=[str(evidence)],
        monthly_status="approval_ready",
    )
    StrategyChangeLedger(findings).record(record)
    (findings / "focused_recall_cards.jsonl").write_text(
        json.dumps({
            "run_id": "monthly-bot1-strat1-2026-04",
            "workflow": "monthly_validation",
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "proposal_or_finding_summary": "req-123 candidate",
            "evidence_paths": ["memory/findings/learning_reviews.jsonl"],
            "how_this_matters_now": "Check lifecycle status before repeating.",
        }) + "\n",
        encoding="utf-8",
    )

    cards = RunRecallSummarizer(memory).summarize(
        workflow="monthly_validation",
        bot_id="bot1",
        strategy_id="strat1",
    )

    assert len(cards) == 1
    assert cards[0].approval_status == "approval_request_linked"
