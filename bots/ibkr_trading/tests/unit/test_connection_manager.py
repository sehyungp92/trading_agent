"""Unit tests for libs.broker_ibkr.connection — ConnectionManager."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


def _make_profile() -> MagicMock:
    """Return a mock ConnectionGroupConfig with sensible defaults."""
    profile = MagicMock()
    profile.host = "127.0.0.1"
    profile.port = 4001
    profile.client_id = 1
    profile.readonly = False
    profile.reconnect_max_retries = 3
    profile.reconnect_base_delay_s = 1.0
    profile.reconnect_max_delay_s = 30.0
    return profile


# ---------------------------------------------------------------------------
# Backoff delay calculation
# ---------------------------------------------------------------------------
class TestBackoffDelay:
    """Verify _backoff_delay() jitter and cap behaviour."""

    @patch("libs.broker_ibkr.connection.IB")
    def test_jitter_within_expected_range(self, MockIB: MagicMock) -> None:
        """delay * jitter must lie in [0.5*base_delay, 1.5*base_delay] for retry 1."""
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        mgr = ConnectionManager(profile)
        mgr._retry_count = 1  # first retry → base_delay * 2^0 = 1.0

        results: list[float] = [mgr._backoff_delay() for _ in range(200)]

        # With jitter multiplier in (0.5, 1.5), delay ∈ [0.5, 1.5]
        assert all(0.5 <= d <= 1.5 for d in results), (
            f"Some delays outside [0.5, 1.5]: min={min(results):.4f}, max={max(results):.4f}"
        )

    @patch("libs.broker_ibkr.connection.IB")
    def test_backoff_capped_at_max_delay(self, MockIB: MagicMock) -> None:
        """Even at high retry counts the delay must never exceed 1.5 * max_delay."""
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        profile.reconnect_max_delay_s = 10.0
        mgr = ConnectionManager(profile)
        mgr._retry_count = 100  # absurdly high — should still be capped

        results: list[float] = [mgr._backoff_delay() for _ in range(200)]

        # base * 2^99 is huge, but min(…, max_delay) caps to 10.0
        # then jitter multiplier ∈ (0.5, 1.5) → max possible = 15.0
        assert all(d <= 15.0 for d in results), f"max={max(results):.4f}"
        # lower bound: 0.5 * 10.0 = 5.0
        assert all(d >= 5.0 for d in results), f"min={min(results):.4f}"

    @patch("libs.broker_ibkr.connection.IB")
    def test_exponential_growth_before_cap(self, MockIB: MagicMock) -> None:
        """Retry 2 should have roughly double the midpoint of retry 1."""
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        profile.reconnect_max_delay_s = 1000.0  # high cap so we don't clip
        mgr = ConnectionManager(profile)

        # Midpoint for retry n = base * 2^(n-1) * 1.0  (jitter midpoint)
        mgr._retry_count = 1
        samples_r1 = [mgr._backoff_delay() for _ in range(500)]
        mgr._retry_count = 2
        samples_r2 = [mgr._backoff_delay() for _ in range(500)]

        mean_r1 = sum(samples_r1) / len(samples_r1)
        mean_r2 = sum(samples_r2) / len(samples_r2)

        # ratio should be ~2 (allow 1.5–2.5 for randomness with 500 samples)
        ratio = mean_r2 / mean_r1
        assert 1.5 <= ratio <= 2.5, f"Expected ratio ~2, got {ratio:.2f}"


# ---------------------------------------------------------------------------
# Disconnect cancels reconnect task
# ---------------------------------------------------------------------------
class TestDisconnect:
    """Verify disconnect() tears down reconnect task and unregisters event."""

    @patch("libs.broker_ibkr.connection.IB")
    @pytest.mark.asyncio
    async def test_disconnect_cancels_reconnect_task(self, MockIB: MagicMock) -> None:
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        mgr = ConnectionManager(profile)

        # Create a long-running task that we can cancel
        async def sleep_forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(sleep_forever())
        mgr._reconnect_task = task

        # Mock ib.isConnected so disconnect() doesn't try real IB calls
        mgr._ib.isConnected.return_value = False

        await mgr.disconnect()

        assert task.cancelled(), "Reconnect task should have been cancelled"

    @patch("libs.broker_ibkr.connection.IB")
    @pytest.mark.asyncio
    async def test_disconnect_unregisters_event_handler(self, MockIB: MagicMock) -> None:
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        mgr = ConnectionManager(profile)
        mgr._ib.isConnected.return_value = False

        # Capture the original event mock so we can inspect __isub__ calls on it
        original_event = mgr._ib.disconnectedEvent

        await mgr.disconnect()

        # Python compiles `obj.disconnectedEvent -= handler` as:
        #   obj.disconnectedEvent = obj.disconnectedEvent.__isub__(handler)
        # So __isub__ is called on the *original* event mock with _on_disconnected.
        original_event.__isub__.assert_called_once_with(mgr._on_disconnected)

    @patch("libs.broker_ibkr.connection.IB")
    @pytest.mark.asyncio
    async def test_disconnect_sets_shutting_down(self, MockIB: MagicMock) -> None:
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        mgr = ConnectionManager(profile)
        mgr._ib.isConnected.return_value = False

        await mgr.disconnect()

        assert mgr._shutting_down is True

    @patch("libs.broker_ibkr.connection.IB")
    @pytest.mark.asyncio
    async def test_disconnect_calls_ib_disconnect_when_connected(self, MockIB: MagicMock) -> None:
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        mgr = ConnectionManager(profile)
        mgr._ib.isConnected.return_value = True

        await mgr.disconnect()

        mgr._ib.disconnect.assert_called_once()

    @patch("libs.broker_ibkr.connection.IB")
    @pytest.mark.asyncio
    async def test_disconnect_clears_connected_event(self, MockIB: MagicMock) -> None:
        from libs.broker_ibkr.connection import ConnectionManager

        profile = _make_profile()
        mgr = ConnectionManager(profile)
        mgr._ib.isConnected.return_value = False
        mgr._connected.set()

        await mgr.disconnect()

        assert not mgr._connected.is_set()
