"""Token-bucket rate limiters for IBKR pacing."""
from __future__ import annotations

import asyncio
import time
from enum import Enum


class PacingChannel(Enum):
    ORDERS = "orders"
    MARKET_DATA = "market_data"
    HISTORICAL = "historical"
    GENERAL = "general"


class CongestionError(Exception):
    """Raised when queue depth exceeds the configured congestion threshold."""


class Throttler:
    """Per-channel token-bucket throttler."""

    def __init__(self, orders_per_sec: float = 5.0, messages_per_sec: float = 50.0):
        self._buckets: dict[PacingChannel, _TokenBucket] = {
            PacingChannel.ORDERS: _TokenBucket(orders_per_sec),
            PacingChannel.MARKET_DATA: _TokenBucket(messages_per_sec),
            PacingChannel.HISTORICAL: _TokenBucket(1.0),
            PacingChannel.GENERAL: _TokenBucket(messages_per_sec),
        }
        self._queue_depth: dict[PacingChannel, int] = {channel: 0 for channel in PacingChannel}
        self.congestion_threshold = 20

    async def acquire(self, channel: PacingChannel) -> None:
        """Acquire a pacing token for the given channel.

        Raises:
            CongestionError: If the channel queue depth exceeds ``congestion_threshold``.
                Callers must handle this to implement back-pressure (e.g. skip entry,
                delay retry, or alert the operator).
        """
        self._queue_depth[channel] += 1
        try:
            if self._queue_depth[channel] > self.congestion_threshold:
                raise CongestionError(f"Channel {channel.value} congested")
            await self._buckets[channel].acquire()
        finally:
            self._queue_depth[channel] -= 1

    @property
    def is_congested(self) -> bool:
        return any(depth > self.congestion_threshold for depth in self._queue_depth.values())


class GlobalThrottler:
    """Two-tier throttler with a gateway-wide bucket plus per-channel buckets."""

    def __init__(self, global_msg_per_sec: float = 45.0, orders_per_sec: float = 5.0):
        self._global_bucket = _TokenBucket(global_msg_per_sec)
        self._channel_buckets: dict[PacingChannel, _TokenBucket] = {
            PacingChannel.ORDERS: _TokenBucket(orders_per_sec),
            PacingChannel.MARKET_DATA: _TokenBucket(global_msg_per_sec),
            PacingChannel.HISTORICAL: _TokenBucket(1.0),
            PacingChannel.GENERAL: _TokenBucket(global_msg_per_sec),
        }
        self._queue_depth: dict[PacingChannel, int] = {channel: 0 for channel in PacingChannel}
        self.congestion_threshold = 20

    async def acquire(self, channel: PacingChannel) -> None:
        """Acquire a pacing token (global + channel).

        Raises:
            CongestionError: If the channel queue depth exceeds ``congestion_threshold``.
                Callers must handle this to implement back-pressure (e.g. skip entry,
                delay retry, or alert the operator).
        """
        self._queue_depth[channel] += 1
        try:
            if self._queue_depth[channel] > self.congestion_threshold:
                raise CongestionError(f"Channel {channel.value} congested")
            await self._global_bucket.acquire()
            await self._channel_buckets[channel].acquire()
        finally:
            self._queue_depth[channel] -= 1

    @property
    def is_congested(self) -> bool:
        return any(depth > self.congestion_threshold for depth in self._queue_depth.values())


class _TokenBucket:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        wait = 0.0
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                self.tokens = 0.0
            else:
                self.tokens -= 1.0
        if wait > 0:
            await asyncio.sleep(wait)

