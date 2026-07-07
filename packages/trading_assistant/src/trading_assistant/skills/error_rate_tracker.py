# skills/error_rate_tracker.py
"""Error rate tracker — sliding window frequency detection.

Tracks error counts per bot within a configurable time window.
When errors exceed the threshold (default: 3/hour), the bot is flagged
as having "repeated errors", which promotes severity to HIGH.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta

from trading_assistant.schemas.bug_triage import ErrorEvent


class ErrorRateTracker:
    """In-memory sliding window error rate tracker."""

    def __init__(self, window_seconds: int = 3600, threshold: int = 3) -> None:
        self._window = timedelta(seconds=window_seconds)
        self._threshold = threshold
        self._timestamps: dict[str, list[datetime]] = defaultdict(list)

    def record(self, event: ErrorEvent) -> None:
        """Record an error event's timestamp for the given bot."""
        self._timestamps[event.bot_id].append(event.timestamp)

    def is_repeated(self, bot_id: str) -> bool:
        """Check if a bot has exceeded the error threshold within the window."""
        return self.get_rate(bot_id) >= self._threshold

    def get_rate(self, bot_id: str) -> int:
        """Get the count of errors within the current window for a bot."""
        self._prune(bot_id)
        return len(self._timestamps.get(bot_id, []))

    def clear(self, bot_id: str) -> None:
        """Clear all tracked errors for a bot."""
        self._timestamps.pop(bot_id, None)

    def _prune(self, bot_id: str) -> None:
        """Remove timestamps outside the sliding window."""
        if bot_id not in self._timestamps:
            return
        cutoff = datetime.now(timezone.utc) - self._window
        self._timestamps[bot_id] = [
            ts for ts in self._timestamps[bot_id] if ts > cutoff
        ]
