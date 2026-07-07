from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.skills.learning_review_orchestrator import LearningReviewOrchestrator
from trading_assistant.skills.provider_route_scorer import ProviderRouteScorer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_learning_review_writes_focused_recall_cards(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "harness_eval_results.jsonl", [{
        "variant_name": "unsafe_patch",
        "kept": False,
        "per_metric": {"validator_gate_fidelity": 0.3},
        "governance_failures": ["approval bypass"],
        "recorded_at": "2026-05-05T12:00:00+00:00",
    }])
    _write_jsonl(findings / "provider_route_scores.jsonl", [{
        "workflow": "monthly_model_review",
        "provider": "codex_pro",
        "model": "gpt-5.4",
        "benchmark_quality": 0.4,
        "recorded_at": "2026-05-05T12:00:00+00:00",
    }])

    review = LearningReviewOrchestrator(findings).run(
        week_start="2026-05-04",
        week_end="2026-05-10",
    )

    assert len(review["actions"]) == 2
    assert (findings / "learning_reviews.jsonl").exists()
    assert (findings / "focused_recall_cards.jsonl").exists()
    cards = (findings / "learning_cards.jsonl").read_text(encoding="utf-8")
    assert "unsafe_patch" in cards
    assert "monthly_model_review" in cards


def test_llm_review_mode_falls_back_to_deterministic_artifact_review(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"

    review = LearningReviewOrchestrator(findings, review_mode="llm_review").run(
        week_start="2026-05-04",
        week_end="2026-05-10",
    )

    assert review["actions"] == []
    assert review["review_mode"] == "deterministic"
    assert review["requested_review_mode"] == "llm_review"
    assert "deterministic artifact review" in review["review_notes"][0]


def test_learning_review_can_disable_specific_workflows(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "provider_route_scores.jsonl", [{
        "workflow": "daily_analysis",
        "provider": "codex_pro",
        "model": "gpt-5.4",
        "benchmark_quality": 0.2,
        "sample_count": 6,
        "recorded_at": "2026-05-05T12:00:00+00:00",
    }])

    review = LearningReviewOrchestrator(
        findings,
        disabled_workflows=["daily_analysis"],
    ).run(
        week_start="2026-05-04",
        week_end="2026-05-10",
    )

    assert review["actions"] == []
    assert not (findings / "focused_recall_cards.jsonl").exists()


def test_learning_review_picks_up_fresh_provider_scores_from_scorer(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "provider_benchmark_results.jsonl", [{
        "workflow": "monthly_model_review",
        "provider": "codex_pro",
        "model": "gpt-5.4",
        "benchmark_score": 0.3,
        "sample_count": 12,
    }])
    ProviderRouteScorer(findings).recompute()
    today = datetime.now(timezone.utc).date().isoformat()

    review = LearningReviewOrchestrator(findings).run(
        week_start=today,
        week_end=today,
    )

    assert any(action["type"] == "provider_route_watch" for action in review["actions"])


def test_learning_review_reads_run_artifacts(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    run_dir = tmp_path / "runs" / "daily-run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        "agent_type": "daily_analysis",
        "bot_id": "bot1",
        "started_at": "2026-05-05T12:00:00+00:00",
    }), encoding="utf-8")
    (run_dir / "response.md").write_text("report", encoding="utf-8")
    (run_dir / "validator_notes.md").write_text("1 suggestion blocked", encoding="utf-8")

    review = LearningReviewOrchestrator(findings, runs_dir=tmp_path / "runs").run(
        week_start="2026-05-04",
        week_end="2026-05-10",
    )

    action_types = {action["type"] for action in review["actions"]}
    assert "run_parse_missing" in action_types
    assert "validator_block_recall" in action_types
    cards = (findings / "learning_cards.jsonl").read_text(encoding="utf-8")
    assert "daily-run-1" in cards


def test_learning_review_reads_monthly_approval_and_model_artifacts(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    artifact_root = tmp_path / "artifacts" / "monthly-run-1"
    artifact_root.mkdir(parents=True)
    run_dir = tmp_path / "runs" / "monthly-run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        "workflow": "monthly_validation",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "started_at": "2026-05-05T12:00:00+00:00",
        "artifact_root": str(artifact_root),
        "_learning_card_ids": ["card-1"],
        "_generated_playbook_ids": ["playbook-1"],
    }), encoding="utf-8")
    (run_dir / "validator_notes.md").write_text("1 candidate blocked", encoding="utf-8")
    (artifact_root / "selected_candidates.json").write_text(json.dumps([{
        "candidate_id": "cand-1",
        "candidate_attempt_id": "attempt-1",
        "source_weekly_signal_ids": ["weekly-1"],
    }]), encoding="utf-8")
    (artifact_root / "model_review_validation.json").write_text(json.dumps({
        "valid": False,
        "issues": [{"message": "invented evidence"}],
    }), encoding="utf-8")
    (artifact_root / "monthly_approval_packet.json").write_text(json.dumps({
        "candidate_id": "cand-1",
        "machine_readable_payload": {
            "model_review_validation": {"valid": False},
        },
        "rollback_plan": "Restore prior config.",
        "evidence_paths": [str(artifact_root / "selected_candidates.json")],
    }), encoding="utf-8")
    _write_jsonl(findings / "approvals.jsonl", [{
        "request_id": "req-1",
        "monthly_run_id": "monthly-run-1",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "status": "PENDING",
        "created_at": "2026-05-05T12:00:00+00:00",
        "approval_packet_path": str(artifact_root / "monthly_approval_packet.json"),
    }])

    review = LearningReviewOrchestrator(findings, runs_dir=tmp_path / "runs").run(
        week_start="2026-05-04",
        week_end="2026-05-10",
    )

    action_types = {action["type"] for action in review["actions"]}
    assert "monthly_candidate_recall" in action_types
    assert "monthly_model_review_issue" in action_types
    assert "monthly_approval_packet_issue" in action_types
    assert "retrieval_context_warning" in action_types
    assert "approval_lifecycle_recall" in action_types


def test_llm_review_accepts_only_structured_evidence_backed_actions(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    evidence = findings / "learning_reviews.jsonl"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")

    def reviewer(payload: dict) -> dict:
        assert "allowed_action_types" in payload
        return {
            "actions": [
                {
                    "type": "focused_recall",
                    "source": "structured_learning_review",
                    "summary": "Keep this validated pattern visible.",
                    "evidence_paths": ["memory/findings/learning_reviews.jsonl"],
                },
                {
                    "type": "write_policy",
                    "evidence_paths": ["memory/findings/learning_reviews.jsonl"],
                },
            ],
        }

    review = LearningReviewOrchestrator(
        findings,
        review_mode="llm_review",
        structured_reviewer=reviewer,
    ).run(week_start="2026-05-04", week_end="2026-05-10")

    assert review["review_mode"] == "llm_review"
    assert any(action["source"] == "structured_learning_review" for action in review["actions"])
    assert any("disallowed action type" in note for note in review["review_notes"])


def test_learning_review_is_bounded_to_requested_week(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "harness_eval_results.jsonl", [
        {
            "variant_name": "old_patch",
            "kept": False,
            "per_metric": {"validator_gate_fidelity": 0.2},
            "recorded_at": "2026-04-01T12:00:00+00:00",
        },
        {
            "variant_name": "current_patch",
            "kept": False,
            "per_metric": {"validator_gate_fidelity": 0.2},
            "recorded_at": "2026-05-05T12:00:00+00:00",
        },
    ])

    review = LearningReviewOrchestrator(findings).run(
        week_start="2026-05-04",
        week_end="2026-05-10",
    )

    variants = {action.get("variant_name") for action in review["actions"]}
    assert "current_patch" in variants
    assert "old_patch" not in variants
