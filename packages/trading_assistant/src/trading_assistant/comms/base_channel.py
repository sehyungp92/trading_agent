# comms/base_channel.py
"""Base channel with lifecycle management and retry semantics."""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseChannel(ABC):
    """Abstract base class for communication channels with lifecycle and retry."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0) -> None:
        self._is_running = False
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._consecutive_failures = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @abstractmethod
    async def _start(self) -> None:
        """Channel-specific startup logic."""

    @abstractmethod
    async def _stop(self) -> None:
        """Channel-specific shutdown logic."""

    @abstractmethod
    async def _send(self, *args, **kwargs) -> object:
        """Channel-specific send logic."""

    async def start(self) -> None:
        await self._start()
        self._is_running = True
        self._consecutive_failures = 0

    async def stop(self) -> None:
        await self._stop()
        self._is_running = False

    async def restart(self) -> None:
        if self._is_running:
            await self.stop()
        await self.start()

    async def send_with_retry(self, *args, **kwargs) -> object:
        """Send with exponential backoff retry. Returns the send result."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                result = await self._send(*args, **kwargs)
                self._consecutive_failures = 0
                return result
            except Exception as exc:
                last_exc = exc
                self._consecutive_failures += 1
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        "%s send attempt %d/%d failed: %s (retrying in %.1fs)",
                        self.__class__.__name__, attempt + 1, self._max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
        logger.error(
            "%s send failed after %d attempts: %s",
            self.__class__.__name__, self._max_retries, last_exc,
        )
        raise last_exc  # type: ignore[misc]
