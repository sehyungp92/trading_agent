"""End-to-end delivery latency tracker.

Maintains a sliding window of latency observations per bot_id.
Computes p50, p95, max percentiles for monitoring and alerting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class LatencyStats:
    """Computed latency percentiles for a set of observations."""
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0
    sample_count: int = 0


@dataclass
class _Observation:
    monotonic_time: float
    latency_seconds: float


class LatencyTracker:
    """In-memory sliding window latency tracker.

    Stores per-bot latency observations and computes percentile stats.
    Observations older than ``window_seconds`` are pruned on access.
    """

    def __init__(self, window_seconds: float = 86400.0) -> None:
        self._window = window_seconds
        self._observations: dict[str, list[_Observation]] = {}

    def record(
        self,
        bot_id: str,
        exchange_timestamp: str,
        received_at: str,
    ) -> None:
        """Record a latency observation from timestamp strings.

        Negative latencies (clock skew) are clamped to 0.
        Unparseable timestamps are silently skipped.
        """
        try:
            ex_dt = datetime.fromisoformat(exchange_timestamp)
            rx_dt = datetime.fromisoformat(received_at)
        except (ValueError, TypeError):
            return

        # Make both timezone-aware if naive
        if ex_dt.tzinfo is None:
            ex_dt = ex_dt.replace(tzinfo=timezone.utc)
        if rx_dt.tzinfo is None:
            rx_dt = rx_dt.replace(tzinfo=timezone.utc)

        delta = (rx_dt - ex_dt).total_seconds()
        if delta < 0:
            delta = 0.0

        obs = _Observation(monotonic_time=time.monotonic(), latency_seconds=delta)
        self._observations.setdefault(bot_id, []).append(obs)

    def _prune(self, bot_id: str) -> list[_Observation]:
        """Remove expired observations and return remaining."""
        cutoff = time.monotonic() - self._window
        obs = self._observations.get(bot_id, [])
        fresh = [o for o in obs if o.monotonic_time >= cutoff]
        self._observations[bot_id] = fresh
        return fresh

    @staticmethod
    def _compute_stats(observations: list[_Observation]) -> LatencyStats:
        if not observations:
            return LatencyStats()

        latencies = sorted(o.latency_seconds for o in observations)
        n = len(latencies)

        def percentile(p: float) -> float:
            idx = int(p / 100.0 * (n - 1))
            return latencies[min(idx, n - 1)]

        return LatencyStats(
            p50=percentile(50),
            p95=percentile(95),
            max=latencies[-1],
            sample_count=n,
        )

    def get_stats(self, bot_id: str) -> LatencyStats:
        """Get latency stats for a single bot."""
        obs = self._prune(bot_id)
        return self._compute_stats(obs)

    def get_all_stats(self) -> dict[str, LatencyStats]:
        """Get latency stats for all tracked bots."""
        result: dict[str, LatencyStats] = {}
        for bot_id in list(self._observations):
            obs = self._prune(bot_id)
            if obs:
                result[bot_id] = self._compute_stats(obs)
        return result

    def get_aggregate_stats(self) -> LatencyStats:
        """Get combined latency stats across all bots."""
        all_obs: list[_Observation] = []
        for bot_id in list(self._observations):
            all_obs.extend(self._prune(bot_id))
        return self._compute_stats(all_obs)
