"""Plugin/hook system for message transformation (M2)."""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from trading_assistant.schemas.notifications import NotificationPayload

logger = logging.getLogger(__name__)


@runtime_checkable
class MessageHook(Protocol):
    """Protocol for message transformation hooks."""

    def transform(self, payload: NotificationPayload) -> NotificationPayload: ...


class HookPipeline:
    """Runs a chain of hooks on outgoing messages."""

    def __init__(self) -> None:
        self._hooks: list[MessageHook] = []

    def add(self, hook: MessageHook) -> None:
        self._hooks.append(hook)

    @property
    def hooks(self) -> list[MessageHook]:
        return self._hooks

    def run(self, payload: NotificationPayload) -> NotificationPayload:
        """Run all hooks in order. Each hook receives the output of the previous."""
        result = payload
        for hook in self._hooks:
            try:
                result = hook.transform(result)
            except Exception:
                logger.exception("Hook %s failed, skipping", hook.__class__.__name__)
        return result
