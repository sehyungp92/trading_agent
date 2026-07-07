from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.analysis.context_sources.policy_memory import PolicyMemorySource
from trading_assistant.analysis.evidence_memory import EvidenceMemory


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_finding_memory_filters_by_bot_and_temporal_decay(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "findings" / "corrections.jsonl",
        [
            {"id": "old", "bot_id": "bot_a", "created_at": "2026-02-01T00:00:00+00:00"},
            {"id": "other", "bot_id": "bot_b", "created_at": "2026-06-29T00:00:00+00:00"},
            {"id": "global", "created_at": "2026-06-28T00:00:00+00:00"},
            {"id": "recent", "bot_id": "bot_a", "created_at": "2026-06-30T00:00:00+00:00"},
        ],
    )

    corrections = EvidenceMemory(tmp_path).findings.load_corrections(
        "bot_a",
        max_age_days=30,
        as_of=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )

    assert [item["id"] for item in corrections] == ["recent", "global"]


def test_policy_memory_reports_prompt_file_provenance(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policies" / "v1"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agent.md").write_text("agent policy", encoding="utf-8")
    (policy_dir / "soul.md").write_text("soul policy", encoding="utf-8")

    source = PolicyMemorySource(tmp_path)

    assert "agent policy" in source.build_system_prompt()
    assert source.context_files() == [
        str(policy_dir / "agent.md"),
        str(policy_dir / "soul.md"),
    ]


def test_finding_memory_filters_inactive_strategy_failures(tmp_path: Path) -> None:
    class _Registry:
        def is_active(self, strategy_id: str) -> bool:
            return strategy_id == "live_strategy"

    _write_jsonl(
        tmp_path / "findings" / "failure-log.jsonl",
        [
            {"id": "old_strategy", "strategy_id": "retired_strategy"},
            {"id": "live_strategy", "strategy_id": "live_strategy"},
        ],
    )

    failures = EvidenceMemory(
        tmp_path,
        strategy_registry=_Registry(),
    ).findings.load_failure_log()

    assert [item["id"] for item in failures] == ["live_strategy"]


def test_finding_memory_work_log_preserves_evidence_provenance(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "findings" / "loop_run_ledger.jsonl",
        [{
            "loop_run_id": "run-1",
            "loop_id": "daily_analysis",
            "status": "completed",
            "completed_at": "2026-06-30T00:00:00+00:00",
            "summary": "done",
            "evidence_paths": ["memory/findings/evidence.json"],
            "approval_packet_paths": ["memory/artifacts/approval.json"],
        }],
    )

    entries = EvidenceMemory(tmp_path).findings.load_recent_work_log_entries(
        agent_type="daily_analysis",
    )

    assert entries[0]["evidence_paths"] == ["memory/findings/evidence.json"]
    assert entries[0]["approval_packet_paths"] == ["memory/artifacts/approval.json"]


def test_finding_memory_outcomes_split_quality_and_keep_latest(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "findings" / "outcomes.jsonl",
        [
            {
                "suggestion_id": "s1",
                "measurement_quality": "low",
                "recorded_at": "2026-06-28T00:00:00+00:00",
            },
            {
                "suggestion_id": "s1",
                "measurement_quality": "high",
                "recorded_at": "2026-06-30T00:00:00+00:00",
            },
        ],
    )

    reliable, low_quality = EvidenceMemory(tmp_path).findings.load_outcome_measurements()

    assert [item["suggestion_id"] for item in reliable] == ["s1"]
    assert low_quality == []
