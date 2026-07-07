from __future__ import annotations
import asyncio
import inspect
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class CapacityExceeded(Exception):
    """Raised when SubagentManager has no available slots for a new spawn."""


@dataclass
class SubagentInfo:
    """Metadata about a running or recent terminal subagent."""
    id: str
    agent_type: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    status: str = "running"
    error: str = ""
    task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self.task is not None and not self.task.done()

class SubagentManager:
    """Manages background async tasks with concurrency limits."""

    def __init__(self, max_concurrent: int = 3, terminal_history_limit: int = 100) -> None:
        self._max_concurrent = max_concurrent
        self._agents: dict[str, SubagentInfo] = {}
        self._terminal_history: deque[SubagentInfo] = deque(maxlen=max(0, terminal_history_limit))
        self._terminal_callbacks: dict[str, Callable[[SubagentInfo], object]] = {}

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def _finalize_done_agents(self) -> None:
        for agent_id, info in list(self._agents.items()):
            if info.task is not None and info.task.done():
                self._finalize_agent(agent_id, info.task)

    def get_running(self) -> list[SubagentInfo]:
        """Get all currently running subagents."""
        self._finalize_done_agents()
        return [a for a in self._agents.values() if a.is_running]

    def get_all(self) -> list[SubagentInfo]:
        """Get running subagents plus bounded recent terminal summaries."""
        self._finalize_done_agents()
        return [*self._agents.values(), *self._terminal_history]

    @staticmethod
    def make_agent_id(agent_type: str) -> str:
        return f"{agent_type}-{uuid.uuid4().hex[:8]}"

    async def spawn(
        self,
        agent_type: str,
        coro: Callable[[], Awaitable[object]],
        *,
        agent_id: str | None = None,
        on_terminal: Callable[[SubagentInfo], object] | None = None,
    ) -> str | None:
        """Spawn a new subagent. Returns agent_id or None if at capacity."""
        running = self.get_running()
        if len(running) >= self._max_concurrent:
            logger.warning(
                "Cannot spawn %s: %d/%d slots used",
                agent_type, len(running), self._max_concurrent,
            )
            return None

        agent_id = agent_id or self.make_agent_id(agent_type)
        task = asyncio.create_task(self._run_agent(agent_id, coro))
        info = SubagentInfo(id=agent_id, agent_type=agent_type, task=task)
        self._agents[agent_id] = info
        if on_terminal is not None:
            self._terminal_callbacks[agent_id] = on_terminal
        task.add_done_callback(lambda done_task, aid=agent_id: self._finalize_agent(aid, done_task))
        logger.info("Spawned subagent %s (type=%s)", agent_id, agent_type)
        return agent_id

    def _finalize_agent(self, agent_id: str, task: asyncio.Task) -> None:
        info = self._agents.pop(agent_id, None)
        if info is None:
            return

        info.finished_at = datetime.now(timezone.utc)
        if task.cancelled():
            info.status = "cancelled"
        else:
            exc = task.exception()
            if exc is not None:
                info.status = "failed"
                info.error = str(exc)
            else:
                info.status = "completed"
        info.task = None

        if self._terminal_history.maxlen != 0:
            self._terminal_history.appendleft(info)

        callback = self._terminal_callbacks.pop(agent_id, None)
        if callback is None:
            return
        try:
            result = callback(info)
        except Exception:
            logger.exception("Subagent terminal callback failed for %s", agent_id)
            return
        if inspect.isawaitable(result):
            asyncio.create_task(self._run_terminal_callback(agent_id, result))

    async def _run_terminal_callback(self, agent_id: str, awaitable: Awaitable[object]) -> None:
        try:
            await awaitable
        except Exception:
            logger.exception("Subagent terminal callback failed for %s", agent_id)

    async def _run_agent(self, agent_id: str, coro: Callable[[], Awaitable[object]]) -> object:
        """Wrapper that logs completion/failure."""
        try:
            result = await coro()
            logger.info("Subagent %s completed", agent_id)
            return result
        except asyncio.CancelledError:
            logger.info("Subagent %s cancelled", agent_id)
            raise
        except Exception:
            logger.exception("Subagent %s failed", agent_id)
            raise

    async def cancel(self, agent_id: str) -> bool:
        """Cancel a running subagent. Returns True if cancelled."""
        info = self._agents.get(agent_id)
        if info is None or not info.is_running:
            return False
        info.task.cancel()
        try:
            await info.task
        except (asyncio.CancelledError, Exception):
            pass
        return True

    async def cancel_all(self) -> int:
        """Cancel all running subagents. Returns count cancelled."""
        cancelled = 0
        for info in self.get_running():
            if await self.cancel(info.id):
                cancelled += 1
        return cancelled
