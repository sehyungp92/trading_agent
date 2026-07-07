"""Tests for MemoryConsolidator scheduling (A0)."""
from __future__ import annotations

import json

import pytest

from trading_assistant.orchestrator.app import create_app
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


@pytest.fixture
def app_with_tmp(tmp_path):
    config = AppConfig(
        data_dir=str(tmp_path),
        bot_ids=["bot1"],
        allow_unauthenticated_local=True,
    )
    return create_app(db_dir=str(tmp_path), config=config)


def test_consolidator_instantiated_in_create_app(app_with_tmp):
    assert hasattr(app_with_tmp.state, "consolidator")
    assert app_with_tmp.state.consolidator is not None


def test_scheduler_includes_memory_consolidation_job():
    config = SchedulerConfig()

    async def noop():
        pass

    jobs = create_scheduler_jobs(
        config=config,
        worker_fn=noop,
        monitoring_fn=noop,
        relay_fn=noop,
        memory_consolidation_fn=noop,
    )
    names = [j["name"] for j in jobs]
    assert "memory_consolidation" in names


def test_scheduler_memory_consolidation_job_is_weekly_cron():
    config = SchedulerConfig()

    async def noop():
        pass

    jobs = create_scheduler_jobs(
        config=config,
        worker_fn=noop,
        monitoring_fn=noop,
        relay_fn=noop,
        memory_consolidation_fn=noop,
    )
    job = next(j for j in jobs if j["name"] == "memory_consolidation")
    assert job["trigger"] == "cron"
    assert job["day_of_week"] == "sun"
    assert job["hour"] == 9


@pytest.mark.asyncio
async def test_consolidation_function_calls_rebuild_index(tmp_path):
    from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

    findings_dir = tmp_path / "memory" / "findings"
    findings_dir.mkdir(parents=True)
    consolidator = MemoryConsolidator(findings_dir=findings_dir, base_dir=tmp_path)
    consolidator.rebuild_index()

    index_path = tmp_path / "memory" / "index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert "last_consolidated" in data


@pytest.mark.asyncio
async def test_consolidation_conditionally_calls_consolidate(tmp_path):
    from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

    findings_dir = tmp_path / "memory" / "findings"
    findings_dir.mkdir(parents=True)

    # Write > threshold entries
    corrections = findings_dir / "corrections.jsonl"
    lines = [json.dumps({"bot_id": f"b{i}", "error_type": "test"}) for i in range(150)]
    corrections.write_text("\n".join(lines))

    consolidator = MemoryConsolidator(findings_dir=findings_dir, base_dir=tmp_path, threshold=100)
    result = consolidator.consolidate("corrections.jsonl")

    assert result is not None
    assert (findings_dir / "patterns_consolidated.md").exists()


def test_graceful_when_findings_dir_missing(tmp_path):
    from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator(
        findings_dir=tmp_path / "nonexistent",
        base_dir=tmp_path,
    )
    index = consolidator.rebuild_index()
    assert index.total_findings == 0
