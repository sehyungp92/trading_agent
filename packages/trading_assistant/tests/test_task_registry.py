
import pytest

from trading_assistant.orchestrator.task_registry import TaskRegistry
from trading_assistant.schemas.tasks import TaskRecord, TaskStatus


@pytest.fixture
async def registry(tmp_db_path) -> TaskRegistry:
    r = TaskRegistry(db_path=str(tmp_db_path))
    await r.initialize()
    return r


class TestTaskRegistry:
    async def test_create_and_get(self, registry: TaskRegistry):
        task = TaskRecord(
            id="daily-report-2026-03-01",
            type="daily_analysis",
            agent="claude-code",
            context_files=["memory/policies/v1/trading_rules.md"],
            run_folder="runs/2026-03-01/daily-report/",
            source_event_id="event-1",
            source_action_type="spawn_daily_analysis",
            subagent_id="daily-analysis-abc",
        )
        await registry.create(task)
        retrieved = await registry.get("daily-report-2026-03-01")

        assert retrieved is not None
        assert retrieved.id == "daily-report-2026-03-01"
        assert retrieved.status == TaskStatus.PENDING
        assert retrieved.source_event_id == "event-1"
        assert retrieved.source_action_type == "spawn_daily_analysis"
        assert retrieved.subagent_id == "daily-analysis-abc"

    async def test_update_status(self, registry: TaskRegistry):
        task = TaskRecord(id="t1", type="test", agent="test")
        await registry.create(task)
        await registry.update_status("t1", TaskStatus.RUNNING)

        retrieved = await registry.get("t1")
        assert retrieved is not None
        assert retrieved.status == TaskStatus.RUNNING

    async def test_complete_with_result(self, registry: TaskRegistry):
        task = TaskRecord(id="t2", type="test", agent="test")
        await registry.create(task)
        await registry.complete("t2", result_summary="Report generated successfully")

        retrieved = await registry.get("t2")
        assert retrieved is not None
        assert retrieved.status == TaskStatus.COMPLETED
        assert retrieved.result_summary == "Report generated successfully"

    async def test_fail_with_retry(self, registry: TaskRegistry):
        task = TaskRecord(id="t3", type="test", agent="test", max_retries=3)
        await registry.create(task)
        await registry.fail("t3", error="Timeout")

        retrieved = await registry.get("t3")
        assert retrieved is not None
        assert retrieved.status == TaskStatus.PENDING  # retryable, back to pending
        assert retrieved.retries == 1

    async def test_fail_exhausts_retries(self, registry: TaskRegistry):
        task = TaskRecord(id="t4", type="test", agent="test", max_retries=1)
        await registry.create(task)
        await registry.fail("t4", error="Timeout")

        retrieved = await registry.get("t4")
        assert retrieved is not None
        assert retrieved.status == TaskStatus.FAILED
        assert retrieved.retries == 1

    async def test_list_by_status(self, registry: TaskRegistry):
        await registry.create(TaskRecord(id="a1", type="test", agent="test"))
        await registry.create(TaskRecord(id="a2", type="test", agent="test"))
        await registry.update_status("a1", TaskStatus.RUNNING)

        running = await registry.list_by_status(TaskStatus.RUNNING)
        assert len(running) == 1
        assert running[0].id == "a1"

    async def test_list_by_source_event(self, registry: TaskRegistry):
        await registry.create(TaskRecord(
            id="linked",
            type="daily_analysis",
            agent="subagent_manager",
            source_event_id="source-1",
            source_action_type="spawn_daily_analysis",
            subagent_id="linked",
        ))

        linked = await registry.list_by_source_event("source-1")
        assert len(linked) == 1
        assert linked[0].id == "linked"

    async def test_retryable_linked_tasks_and_mark_running(self, registry: TaskRegistry):
        await registry.create(TaskRecord(
            id="linked-retry",
            type="daily_analysis",
            agent="subagent_manager",
            status=TaskStatus.RUNNING,
            max_retries=2,
            source_event_id="source-retry",
            source_action_type="spawn_daily_analysis",
            subagent_id="agent-1",
        ))
        await registry.fail("linked-retry", error="boom")

        retryable = await registry.list_retryable_linked()
        assert [task.id for task in retryable] == ["linked-retry"]

        await registry.mark_running("linked-retry", subagent_id="agent-2")
        task = await registry.get("linked-retry")
        assert task is not None
        assert task.status == TaskStatus.RUNNING
        assert task.subagent_id == "agent-2"

    async def test_find_stale_tasks(self, registry: TaskRegistry):
        task = TaskRecord(id="stale1", type="test", agent="test")
        await registry.create(task)
        await registry.update_status("stale1", TaskStatus.RUNNING)

        # Stale = running longer than timeout_seconds
        stale = await registry.find_stale(timeout_seconds=0)  # 0 = everything is stale
        assert len(stale) == 1
        assert stale[0].id == "stale1"
