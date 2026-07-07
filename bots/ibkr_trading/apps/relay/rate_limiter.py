"""Per-bot rate limiter for the relay service."""
from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    """Simple sliding-window rate limiter per bot_id."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, bot_id: str) -> bool:
        """Check if a request from bot_id is allowed."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune old entries
        self._requests[bot_id] = [
            t for t in self._requests[bot_id] if t > cutoff
        ]

        if len(self._requests[bot_id]) >= self.max_requests:
            return False

        self._requests[bot_id].append(now)
        return True

    def remaining(self, bot_id: str) -> int:
        """Return how many requests remain in the current window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._requests[bot_id] = [
            t for t in self._requests[bot_id] if t > cutoff
        ]
        return max(0, self.max_requests - len(self._requests[bot_id]))
