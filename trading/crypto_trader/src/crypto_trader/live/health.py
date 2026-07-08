"""Health monitoring for live trading engine."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog

log = structlog.get_logger()


class HealthMonitor:
    """Monitors live trading engine health.

    Tracks:
    - Heartbeat (alive confirmation)
    - Stale data detection (no new bars for 2x expected interval)
    - Exchange connectivity
    - Error counts and rate limiting
    """

    def __init__(
        self,
        expected_bar_interval_sec: float = 900.0,  # M15 = 15 min
        stale_multiplier: float = 2.0,
    ) -> None:
        self._expected_interval = expected_bar_interval_sec
        self._stale_threshold = expected_bar_interval_sec * stale_multiplier

        self._last_bar_time: float = time.monotonic()
        self._last_heartbeat: float = time.monotonic()
        self._error_count: int = 0
        self._consecutive_errors: int = 0
        self._total_polls: int = 0
        self._start_time: float = time.monotonic()
        self._tf_last_bar: dict[tuple[str, str], float] = {}

    def on_bar_received(
        self, symbol: str | None = None, tf: str | None = None,
    ) -> None:
        """Record that a new bar was received, optionally per (symbol, tf)."""
        self._last_bar_time = time.monotonic()
        self._consecutive_errors = 0
        if symbol and tf:
            self._tf_last_bar[(symbol, tf)] = time.monotonic()

    def on_error(self, error_type: str = "unknown") -> None:
        """Record an error occurrence."""
        self._error_count += 1
        self._consecutive_errors += 1
        log.warning(
            "health.error",
            type=error_type,
            consecutive=self._consecutive_errors,
            total=self._error_count,
        )

    def on_poll(self) -> None:
        """Record a poll cycle."""
        self._total_polls += 1

    def is_stale(self) -> bool:
        """Check if data is stale (no bars for too long)."""
        elapsed = time.monotonic() - self._last_bar_time
        return elapsed > self._stale_threshold

    def should_reconnect(self, max_consecutive: int = 5) -> bool:
        """Check if we should attempt reconnection."""
        return self._consecutive_errors >= max_consecutive

    def heartbeat(self) -> None:
        """Emit a heartbeat log."""
        now = time.monotonic()
        uptime = now - self._start_time
        since_bar = now - self._last_bar_time

        log.info(
            "health.heartbeat",
            uptime_sec=round(uptime),
            since_last_bar_sec=round(since_bar),
            total_polls=self._total_polls,
            total_errors=self._error_count,
            stale=self.is_stale(),
        )
        self._last_heartbeat = now

    def get_status(self) -> dict:
        """Get current health status as dict."""
        now = time.monotonic()
        return {
            "uptime_sec": round(now - self._start_time),
            "since_last_bar_sec": round(now - self._last_bar_time),
            "is_stale": self.is_stale(),
            "total_polls": self._total_polls,
            "total_errors": self._error_count,
            "consecutive_errors": self._consecutive_errors,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_tf_last_bar(self) -> dict[tuple[str, str], float]:
        """Return a copy of per-(symbol, tf) last-bar timestamps."""
        return dict(self._tf_last_bar)

    def get_stale_feeds(
        self,
        expected_intervals: dict[str, float],
        multiplier: float = 2.0,
    ) -> list[tuple[str, str, float]]:
        """Return list of (symbol, tf, elapsed_sec) for stale feeds."""
        now = time.monotonic()
        stale = []
        for (sym, tf), last in self._tf_last_bar.items():
            expected = expected_intervals.get(tf, 900.0)
            elapsed = now - last
            if elapsed > expected * multiplier:
                stale.append((sym, tf, elapsed))
        return stale

    def get_backoff_delay(self, base: float = 1.0, max_delay: float = 60.0) -> float:
        """Compute exponential backoff delay based on consecutive errors."""
        delay = base * (2 ** min(self._consecutive_errors, 6))
        return min(delay, max_delay)
