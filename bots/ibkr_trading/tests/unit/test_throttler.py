"""Unit tests for libs.broker_ibkr.throttler — _TokenBucket, Throttler, GlobalThrottler."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from libs.broker_ibkr.throttler import (
    _TokenBucket,
    Throttler,
    GlobalThrottler,
    PacingChannel,
    CongestionError,
)


# ---------------------------------------------------------------------------
# _TokenBucket refill math
# ---------------------------------------------------------------------------
class TestTokenBucket:
    """Verify token-bucket refill and sleep behaviour."""

    @pytest.mark.asyncio
    async def test_refill_after_elapsed_time(self) -> None:
        """Draining all tokens then advancing time should refill proportionally."""
        bucket = _TokenBucket(rate=10.0)

        # Drain all tokens
        for _ in range(10):
            await bucket.acquire()

        assert bucket.tokens < 1.0, "Bucket should be near-empty after draining 10 tokens"

        # Simulate 0.5s passing by rewinding last_refill
        bucket.last_refill = time.monotonic() - 0.5

        # acquire() should refill 0.5 * 10 = 5.0 tokens, consume 1 → 4.0 remain
        # It should NOT need to sleep because 5.0 >= 1.0
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start

        # Should be nearly instant (no sleep)
        assert elapsed < 0.05, f"acquire() slept {elapsed:.3f}s but should have been instant"
        # After refill of 5.0 and consuming 1.0, tokens ≈ 4.0
        assert 3.5 <= bucket.tokens <= 4.5, f"Expected ~4.0 tokens, got {bucket.tokens:.2f}"

    @pytest.mark.asyncio
    async def test_tokens_capped_at_rate(self) -> None:
        """Tokens should never exceed the bucket rate even with large elapsed time."""
        bucket = _TokenBucket(rate=5.0)

        # Simulate 100s passing — should cap at rate (5.0)
        bucket.last_refill = time.monotonic() - 100.0

        await bucket.acquire()

        # After refill to cap (5.0) and consuming 1 → 4.0
        assert bucket.tokens <= 5.0, f"Tokens {bucket.tokens} exceed rate"

    @pytest.mark.asyncio
    async def test_acquire_sleeps_when_empty(self) -> None:
        """When bucket is empty, acquire() should sleep to wait for a token."""
        bucket = _TokenBucket(rate=10.0)

        # Drain all tokens
        for _ in range(10):
            await bucket.acquire()

        # Don't rewind time — bucket is truly empty. Next acquire must sleep.
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start

        # At rate=10, need 0.1s to generate 1 token
        assert elapsed >= 0.05, f"Expected sleep >=0.05s, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Throttler congestion threshold
# ---------------------------------------------------------------------------
class TestThrottlerCongestion:
    """Verify CongestionError when queue depth exceeds threshold."""

    @pytest.mark.asyncio
    async def test_congestion_error_raised_above_threshold(self) -> None:
        """When concurrent waiters exceed congestion_threshold, CongestionError fires."""
        throttler = Throttler(orders_per_sec=1.0, messages_per_sec=1.0)
        throttler.congestion_threshold = 2
        channel = PacingChannel.ORDERS

        # Drain the bucket so acquires block
        bucket = throttler._buckets[channel]
        bucket.tokens = 0.0
        bucket.last_refill = time.monotonic()

        barrier = asyncio.Event()
        errors: list[Exception] = []
        started = asyncio.Event()

        async def blocking_acquire(idx: int) -> None:
            try:
                if idx < 2:
                    # These two will block on the empty bucket (queue depth 1, 2)
                    started.set()
                await throttler.acquire(channel)
            except CongestionError as e:
                errors.append(e)

        # Launch 2 tasks that will block (consuming queue depth)
        tasks = [asyncio.create_task(blocking_acquire(i)) for i in range(2)]
        # Let them enter acquire and block
        await asyncio.sleep(0.02)

        # Now the 3rd concurrent call should exceed threshold=2
        with pytest.raises(CongestionError):
            # Manually bump queue depth to simulate the 2 blocked + this one
            throttler._queue_depth[channel] = 2
            await throttler.acquire(channel)

        # Clean up blocked tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_no_congestion_under_threshold(self) -> None:
        """Acquires within threshold should succeed without CongestionError."""
        throttler = Throttler()
        throttler.congestion_threshold = 100

        # Bucket has plenty of tokens (rate=50 by default for GENERAL)
        await throttler.acquire(PacingChannel.GENERAL)
        # If we get here, no CongestionError was raised

    @pytest.mark.asyncio
    async def test_queue_depth_decrements_after_acquire(self) -> None:
        """Queue depth should return to 0 after a successful acquire."""
        throttler = Throttler()
        channel = PacingChannel.GENERAL

        await throttler.acquire(channel)
        assert throttler._queue_depth[channel] == 0


# ---------------------------------------------------------------------------
# GlobalThrottler two-tier acquire
# ---------------------------------------------------------------------------
class TestGlobalThrottler:
    """Verify GlobalThrottler drains both global and channel buckets."""

    @pytest.mark.asyncio
    async def test_acquire_drains_global_and_channel(self) -> None:
        """A single acquire() should consume from both buckets."""
        gt = GlobalThrottler(global_msg_per_sec=10.0, orders_per_sec=5.0)
        channel = PacingChannel.ORDERS

        global_before = gt._global_bucket.tokens
        channel_before = gt._channel_buckets[channel].tokens

        await gt.acquire(channel)

        # Both buckets should have lost approximately 1 token
        global_after = gt._global_bucket.tokens
        channel_after = gt._channel_buckets[channel].tokens

        global_consumed = global_before - global_after
        channel_consumed = channel_before - channel_after

        # Each should have consumed ~1 token (allow small timing drift)
        assert 0.9 <= global_consumed <= 1.1, f"Global consumed {global_consumed:.2f}"
        assert 0.9 <= channel_consumed <= 1.1, f"Channel consumed {channel_consumed:.2f}"

    @pytest.mark.asyncio
    async def test_global_throttler_congestion(self) -> None:
        """GlobalThrottler should also raise CongestionError above threshold."""
        gt = GlobalThrottler()
        gt.congestion_threshold = 1
        channel = PacingChannel.GENERAL

        # Pre-fill queue depth to exceed threshold
        gt._queue_depth[channel] = 1

        with pytest.raises(CongestionError):
            await gt.acquire(channel)

    @pytest.mark.asyncio
    async def test_is_congested_property(self) -> None:
        """is_congested reflects whether any channel queue is above threshold."""
        gt = GlobalThrottler()
        gt.congestion_threshold = 5

        assert gt.is_congested is False

        gt._queue_depth[PacingChannel.ORDERS] = 6
        assert gt.is_congested is True
