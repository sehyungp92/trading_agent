"""Runtime-owned worker dispatch wiring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.orchestrator.subagent import CapacityExceeded
from trading_assistant.schemas.tasks import TaskRecord, TaskStatus


@dataclass(frozen=True)
class WorkerDispatch:
    reconcile_linked_subagent_tasks: Any
    spawn_or_raise: Any


def wire_worker_dispatch(runtime: Any) -> WorkerDispatch:
    """Attach worker handlers and expose retry reconciliation."""
    worker = runtime.worker
    handlers = runtime.handlers
    daily_analysis_loop = runtime.daily_analysis_loop
    weekly_analysis_loop = runtime.weekly_analysis_loop
    monthly_validation_loop = runtime.monthly_validation_loop
    registry = runtime.registry
    queue = runtime.queue
    brain = runtime.brain
    event_stream = runtime.event_stream
    subagent_mgr = runtime.subagent_mgr

    worker.on_alert = handlers.handle_alert
    worker.on_heartbeat = handlers.handle_heartbeat
    worker.on_notification = handlers.handle_notification
    worker.on_feedback = handlers.handle_feedback

    async def _spawn_or_raise(
        agent_type: str,
        action: Action,
        coro,
        *,
        task: TaskRecord | None = None,
    ):
        running = len(subagent_mgr.get_running())
        if running >= subagent_mgr.max_concurrent:
            raise CapacityExceeded(
                f"No subagent slot for {agent_type} "
                f"({running}/{subagent_mgr.max_concurrent} in use)"
            )

        planned_agent_id = subagent_mgr.make_agent_id(agent_type)
        if task is None:
            task_id = planned_agent_id
            await registry.create(TaskRecord(
                id=task_id,
                type=agent_type,
                agent="subagent_manager",
                status=TaskStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
                max_retries=2,
                source_event_id=action.event_id,
                source_action_type=action.type.value,
                subagent_id=planned_agent_id,
            ))
        else:
            task_id = task.id
            await registry.mark_running(task_id, subagent_id=planned_agent_id)

        async def _record_terminal(info):
            if info.status == "completed":
                await registry.complete(task_id, result_summary="subagent completed")
            else:
                await registry.fail(task_id, error=info.error or info.status)
            event_stream.broadcast("subagent_terminal", {
                "subagent_id": info.id,
                "task_id": task_id,
                "agent_type": info.agent_type,
                "status": info.status,
                "source_event_id": action.event_id,
                "source_action_type": action.type.value,
                "error": info.error,
            })

        agent_id = await subagent_mgr.spawn(
            agent_type,
            coro,
            agent_id=planned_agent_id,
            on_terminal=_record_terminal,
        )
        if agent_id is None:
            running = len(subagent_mgr.get_running())
            if task is None:
                await registry.fail(task_id, error="subagent spawn rejected at capacity")
            else:
                await registry.update_status(task_id, TaskStatus.PENDING)
            raise CapacityExceeded(
                f"No subagent slot for {agent_type} "
                f"({running}/{subagent_mgr.max_concurrent} in use)"
            )
        return agent_id

    def _long_handler_for_action(action: Action):
        if action.type == ActionType.SPAWN_TRIAGE:
            return "triage", lambda a=action: handlers.handle_triage(a)
        if action.type == ActionType.SPAWN_DAILY_ANALYSIS:
            return "daily_analysis", lambda a=action: daily_analysis_loop.handle(a)
        if action.type == ActionType.SPAWN_WEEKLY_SUMMARY:
            return "weekly_analysis", lambda a=action: weekly_analysis_loop.handle(a)
        if action.type == ActionType.SPAWN_MONTHLY_VALIDATION:
            return "monthly_validation", lambda a=action: monthly_validation_loop.handle(a)
        return None

    async def _reconcile_linked_subagent_tasks(limit: int = 20) -> dict[str, int]:
        retried = 0
        failed = 0
        skipped_capacity = 0
        for task in await registry.list_retryable_linked(limit=limit):
            source_event = await queue.get(task.source_event_id)
            if source_event is None:
                await registry.fail(task.id, error="source event not found for retry")
                failed += 1
                continue

            actions = brain.decide(source_event)
            action = next(
                (candidate for candidate in actions if candidate.type.value == task.source_action_type),
                None,
            )
            if action is None:
                await registry.fail(task.id, error="source action not reconstructed for retry")
                failed += 1
                continue

            target = _long_handler_for_action(action)
            if target is None:
                await registry.fail(task.id, error=f"no retry handler for {task.source_action_type}")
                failed += 1
                continue

            agent_type, handler_coro = target
            try:
                await _spawn_or_raise(agent_type, action, handler_coro, task=task)
                retried += 1
            except CapacityExceeded:
                skipped_capacity += 1
                break
            except Exception as exc:
                await registry.fail(task.id, error=str(exc))
                failed += 1
        return {"retried": retried, "failed": failed, "skipped_capacity": skipped_capacity}

    worker.on_triage = lambda action: _spawn_or_raise(
        "triage", action, lambda a=action: handlers.handle_triage(a))
    worker.on_daily_analysis = lambda action: _spawn_or_raise(
        "daily_analysis", action, lambda a=action: daily_analysis_loop.handle(a))
    worker.on_weekly_analysis = lambda action: _spawn_or_raise(
        "weekly_analysis", action, lambda a=action: weekly_analysis_loop.handle(a))
    worker.on_monthly_validation = lambda action: _spawn_or_raise(
        "monthly_validation", action, lambda a=action: monthly_validation_loop.handle(a))

    return WorkerDispatch(
        reconcile_linked_subagent_tasks=_reconcile_linked_subagent_tasks,
        spawn_or_raise=_spawn_or_raise,
    )
