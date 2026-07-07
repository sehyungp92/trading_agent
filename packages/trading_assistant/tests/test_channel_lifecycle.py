"""Tests for channel lifecycle protocol with retry + backoff (H2)."""
from __future__ import annotations

import asyncio
import pytest

from trading_assistant.comms.base_channel import BaseChannel
from trading_assistant.comms.dispatcher import NotificationDispatcher
from trading_assistant.schemas.notifications import NotificationChannel


class MockChannel(BaseChannel):
    """Test channel that tracks calls and can be configured to fail."""

    def __init__(self, fail_count: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fail_count = fail_count
        self._send_calls = 0
        self._started = False
        self._stopped = False

    async def _start(self) -> None:
        self._started = True

    async def _stop(self) -> None:
        self._stopped = True

    async def _send(self, *args, **kwargs) -> str:
        self._send_calls += 1
        if self._send_calls <= self._fail_count:
            raise ConnectionError(f"Simulated failure #{self._send_calls}")
        return "sent"


class TestBaseChannelLifecycle:
    async def test_start_sets_running(self):
        ch = MockChannel()
        assert ch.is_running is False
        await ch.start()
        assert ch.is_running is True

    async def test_stop_clears_running(self):
        ch = MockChannel()
        await ch.start()
        await ch.stop()
        assert ch.is_running is False

    async def test_restart_cycles_lifecycle(self):
        ch = MockChannel()
        await ch.start()
        await ch.restart()
        assert ch.is_running is True
        assert ch._started is True
        assert ch._stopped is True


class TestSendWithRetry:
    async def test_successful_send_no_retry(self):
        ch = MockChannel(base_delay=0.01)
        result = await ch.send_with_retry("hello")
        assert result == "sent"
        assert ch._send_calls == 1
        assert ch.consecutive_failures == 0

    async def test_retry_on_failure_then_succeed(self):
        ch = MockChannel(fail_count=2, base_delay=0.01)
        result = await ch.send_with_retry("hello")
        assert result == "sent"
        assert ch._send_calls == 3  # 2 failures + 1 success
        assert ch.consecutive_failures == 0

    async def test_exhausts_retries_raises(self):
        ch = MockChannel(fail_count=10, max_retries=3, base_delay=0.01)
        with pytest.raises(ConnectionError):
            await ch.send_with_retry("hello")
        assert ch._send_calls == 3
        assert ch.consecutive_failures == 3

    async def test_consecutive_failures_reset_on_success(self):
        ch = MockChannel(fail_count=1, base_delay=0.01)
        await ch.send_with_retry("first")
        assert ch.consecutive_failures == 0
        # Reset for second send
        ch._send_calls = 0
        ch._fail_count = 0
        await ch.send_with_retry("second")
        assert ch.consecutive_failures == 0

    async def test_exponential_backoff_timing(self):
        ch = MockChannel(fail_count=2, max_retries=3, base_delay=0.05)
        start = asyncio.get_event_loop().time()
        await ch.send_with_retry("hello")
        elapsed = asyncio.get_event_loop().time() - start
        # Should have delays of ~0.05 and ~0.10 = ~0.15 total
        assert elapsed >= 0.1


class TestDispatcherWithBaseChannel:
    async def test_dispatcher_uses_send_with_retry_for_base_channels(self):
        from trading_assistant.schemas.notifications import (
            NotificationPayload, NotificationPreferences, ChannelConfig,
        )

        ch = MockChannel(fail_count=1, base_delay=0.01)
        dispatcher = NotificationDispatcher()
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, ch)  # type: ignore[arg-type]

        payload = NotificationPayload(notification_type="test", title="Test")
        prefs = NotificationPreferences(channels=[
            ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True),
        ])
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=12)
        assert len(results) == 1
        assert results[0].success is True

    async def test_channel_health_reporting(self):
        ch = MockChannel()
        await ch.start()
        dispatcher = NotificationDispatcher()
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, ch)  # type: ignore[arg-type]

        health = dispatcher.get_channel_health()
        assert len(health) == 1
        assert health[0].channel == NotificationChannel.TELEGRAM
        assert health[0].is_running is True
        assert health[0].consecutive_failures == 0
