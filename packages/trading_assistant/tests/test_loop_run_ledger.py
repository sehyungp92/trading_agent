from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading_assistant.orchestrator.scheduled_runs import ScheduledRunStore
from trading_assistant.skills.loop_run_ledger import (
    LoopRunLedgerProjector,
    LoopRunLedgerStore,
    RuntimeLoopProjectionWriter,
)
from trading_assistant.skills.work_log_projector import WorkLogProjector


@pytest.mark.asyncio
async def test_loop_projection_reads_scheduled_runs_without_mutating_state(tmp_path: Path) -> None:
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 6, 2, 3, 0, tzinfo=timezone.utc)
        await store.mark_started("monthly_validation", "bot:bot1", scheduled_for)
        await store.mark_failed(
            "monthly_validation",
            "bot:bot1",
            scheduled_for,
            error="deployment metadata is shadow",
        )
        before_status = await store.get_status("monthly_validation", "bot:bot1", scheduled_for)
        before_count = await store.count_runs()
        records = await store.list_recent_records(limit=5)

        entry = LoopRunLedgerProjector().project_scheduled_run(
            records[0],
            loop_id="monthly_validation",
            task=SimpleNamespace(
                id="task-1",
                retries=2,
                run_folder=str(tmp_path / "runs" / "task-1"),
                result_summary="monthly validation blocked",
                error="",
            ),
            run_metadata={
                "run_id": "monthly-bot1-strat1-2026-05",
                "provider": "codex_pro",
                "model": "gpt-5",
                "duration_ms": 1200,
                "cost_usd": 0.12,
                "bot_id": "bot1",
                "strategy_id": "strat1",
            },
            monthly_artifacts={
                "artifact_index": str(tmp_path / "artifact_index.json"),
                "runner_observability": str(tmp_path / "runner_observability.json"),
                "candidate_gate_report": str(tmp_path / "candidate_gate_report.json"),
                "approval_packet": str(tmp_path / "approval_packet_cand.json"),
            },
        )

        assert entry.loop_id == "monthly_validation"
        assert entry.status == "failed"
        assert entry.task_id == "task-1"
        assert entry.agent_run_id == "monthly-bot1-strat1-2026-05"
        assert "deployment metadata is shadow" in entry.blocking_reasons
        assert entry.approval_packet_paths == [str(tmp_path / "approval_packet_cand.json")]
        assert await store.get_status("monthly_validation", "bot:bot1", scheduled_for) == before_status
        assert await store.count_runs() == before_count
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_runtime_projection_writes_ledger_and_work_log_on_final_status(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    writer = RuntimeLoopProjectionWriter(memory_dir)
    store = ScheduledRunStore(
        str(tmp_path / "scheduled_runs.db"),
        final_status_observer=writer.project_record,
    )
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 6, 2, 3, 0, tzinfo=timezone.utc)
        await store.mark_started("monthly_validation", "bot:bot1", scheduled_for)
        await store.mark_failed(
            "monthly_validation",
            "bot:bot1",
            scheduled_for,
            error="deployment metadata is shadow",
        )

        ledger_entries = LoopRunLedgerStore(
            memory_dir / "findings" / "loop_run_ledger.jsonl"
        ).read()
        work_log = (memory_dir / "work_log.md").read_text(encoding="utf-8")

        assert len(ledger_entries) == 1
        assert ledger_entries[0].loop_id == "monthly_validation"
        assert ledger_entries[0].status == "failed"
        assert "deployment metadata is shadow" in ledger_entries[0].blocking_reasons
        assert "monthly_validation - failed" in work_log
        assert "deployment metadata is shadow" in work_log
    finally:
        await store.close()


def test_work_log_is_generated_from_ledger_projection(tmp_path: Path) -> None:
    ledger_path = tmp_path / "memory" / "findings" / "loop_run_ledger.jsonl"
    work_log_path = tmp_path / "memory" / "work_log.md"
    projector = LoopRunLedgerProjector()
    record = SimpleNamespace(
        job_key="monthly_validation",
        scope_key="bot:bot1",
        scheduled_for=datetime(2026, 6, 2, 3, 0, tzinfo=timezone.utc),
        status="failed",
        started_at="2026-06-02T03:00:00+00:00",
        finished_at="2026-06-02T03:05:00+00:00",
        error="shadow metadata",
    )
    entry = projector.project_scheduled_run(
        record,
        loop_id="monthly_validation",
        monthly_artifacts={"artifact_index": "artifact_index.json"},
    )
    LoopRunLedgerStore(ledger_path).write_projection([entry])

    projected = WorkLogProjector(ledger_path, work_log_path).project(limit=20)
    text = work_log_path.read_text(encoding="utf-8")

    assert projected[0].loop_id == "monthly_validation"
    assert "Generated from `memory/findings/loop_run_ledger.jsonl`" in text
    assert "shadow metadata" in text
    assert "artifact_index.json" in text


def test_projection_write_merges_existing_entries_by_run_id(tmp_path: Path) -> None:
    from trading_assistant.schemas.loop_run_ledger import LoopRunLedgerEntry

    ledger = LoopRunLedgerStore(tmp_path / "loop_run_ledger.jsonl")
    original = LoopRunLedgerEntry(
        loop_run_id="same-run",
        loop_id="daily_analysis",
        job_key="daily_analysis",
        status="started",
    )
    retained = LoopRunLedgerEntry(
        loop_run_id="other-run",
        loop_id="weekly_summary",
        job_key="weekly_analysis",
        status="completed",
    )
    replacement = original.model_copy(update={"status": "completed", "summary": "daily done"})

    ledger.write_projection([original, retained])
    ledger.write_projection([replacement])
    entries = {entry.loop_run_id: entry for entry in ledger.read()}

    assert entries["same-run"].status == "completed"
    assert entries["same-run"].summary == "daily done"
    assert entries["other-run"].status == "completed"


def test_loop_run_ledger_read_fails_on_malformed_rows(tmp_path: Path) -> None:
    from trading_assistant.schemas.loop_run_ledger import LoopRunLedgerEntry

    ledger_path = tmp_path / "loop_run_ledger.jsonl"
    valid = LoopRunLedgerEntry(
        loop_run_id="valid-run",
        loop_id="daily_analysis",
        job_key="daily_analysis",
        status="completed",
    )
    ledger_path.write_text(
        valid.model_dump_json() + "\n{bad-json}\n",
        encoding="utf-8",
    )
    store = LoopRunLedgerStore(ledger_path)

    with pytest.raises(ValueError, match="invalid loop-run ledger record"):
        store.read()
    assert [entry.loop_run_id for entry in store.read(strict=False)] == ["valid-run"]


def test_repeated_blockers_group_by_loop_bot_and_strategy(tmp_path: Path) -> None:
    raw = {
        "loop_id": "monthly_validation",
        "job_key": "monthly_validation",
        "scope_key": "bot:bot1",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "status": "failed",
        "blocking_reasons": ["shadow metadata"],
    }
    entries = [
        LoopRunLedgerStore(tmp_path / "unused.jsonl").read(),
    ]
    assert entries == [[]]
    from trading_assistant.schemas.loop_run_ledger import LoopRunLedgerEntry

    grouped = LoopRunLedgerProjector().repeated_blockers([
        LoopRunLedgerEntry.model_validate(raw),
        LoopRunLedgerEntry.model_validate({**raw, "blocking_reasons": ["shadow metadata", "missing parity"]}),
    ])

    assert grouped[("monthly_validation", "bot1", "strat1")] == [
        "shadow metadata",
        "missing parity",
    ]
