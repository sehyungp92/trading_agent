"""Heartbeat tests across the broker, service, and watchdog layers."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.watchdog.checks import check_heartbeats
from libs.services.heartbeat import emit_family_heartbeats


# ---------------------------------------------------------------------------
# Broker-layer: libs.broker_ibkr.heartbeat.HeartbeatMonitor
# ---------------------------------------------------------------------------
def _make_monitor(interval: float = 0.05, timeout: float = 0.05):
    with patch("libs.broker_ibkr.heartbeat.IB"):
        from libs.broker_ibkr.heartbeat import HeartbeatMonitor

    mock_ib = MagicMock()
    return HeartbeatMonitor(mock_ib, interval_s=interval, timeout_s=timeout)


class TestStatusTransitions:
    @pytest.mark.asyncio
    async def test_alive_to_dead_to_alive(self) -> None:
        monitor = _make_monitor()
        statuses: list[bool] = []
        monitor.on_status_change = lambda alive: statuses.append(alive)

        call_count = 0

        async def flaky_ping() -> int:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                await asyncio.sleep(10)
            return 1

        monitor._ping = flaky_ping  # type: ignore[assignment]

        await monitor.start()
        await asyncio.sleep(0.6)
        await monitor.stop()

        assert len(statuses) >= 2, f"Expected at least 2 transitions, got {statuses}"
        assert statuses[0] is False, f"First transition should be dead, got {statuses}"
        assert statuses[1] is True, f"Second transition should be alive, got {statuses}"

    @pytest.mark.asyncio
    async def test_stays_alive_on_successful_pings(self) -> None:
        monitor = _make_monitor()
        statuses: list[bool] = []
        monitor.on_status_change = lambda alive: statuses.append(alive)
        monitor._ping = AsyncMock(return_value=1)  # type: ignore[assignment]

        await monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()

        assert statuses == [], f"Expected no transitions, got {statuses}"

    @pytest.mark.asyncio
    async def test_stays_dead_on_repeated_failures(self) -> None:
        monitor = _make_monitor()
        statuses: list[bool] = []
        monitor.on_status_change = lambda alive: statuses.append(alive)

        async def always_timeout() -> int:
            await asyncio.sleep(10)
            return 1

        monitor._ping = always_timeout  # type: ignore[assignment]

        await monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()

        assert statuses == [False], f"Expected [False], got {statuses}"


class TestAsyncCallback:
    @pytest.mark.asyncio
    async def test_async_callback_is_awaited(self) -> None:
        monitor = _make_monitor()
        callback_called = asyncio.Event()

        async def async_callback(alive: bool) -> None:
            callback_called.set()

        monitor.on_status_change = async_callback  # type: ignore[assignment]

        async def timeout_ping() -> int:
            await asyncio.sleep(10)
            return 1

        monitor._ping = timeout_ping  # type: ignore[assignment]

        await monitor.start()
        try:
            await asyncio.wait_for(callback_called.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Async callback was never awaited within 1 second")
        finally:
            await monitor.stop()

        assert callback_called.is_set()

    @pytest.mark.asyncio
    async def test_sync_callback_still_works(self) -> None:
        monitor = _make_monitor()
        called_with: list[bool] = []

        def sync_callback(alive: bool) -> None:
            called_with.append(alive)

        monitor.on_status_change = sync_callback

        async def timeout_ping() -> int:
            await asyncio.sleep(10)
            return 1

        monitor._ping = timeout_ping  # type: ignore[assignment]

        await monitor.start()
        await asyncio.sleep(0.4)
        await monitor.stop()

        assert False in called_with, f"Expected False in {called_with}"


# ---------------------------------------------------------------------------
# Service-layer: libs.services.heartbeat.emit_family_heartbeats
# ---------------------------------------------------------------------------
class _FakeHeartbeat:
    def __init__(self) -> None:
        self.strategy_calls: list[str] = []
        self.adapter_calls: list[tuple[str, bool]] = []
        self.slow_strategy_ids: set[str] = set()
        self.slow_adapter = False

    async def strategy_heartbeat(self, **payload) -> None:
        sid = payload["strategy_id"]
        self.strategy_calls.append(sid)
        if sid in self.slow_strategy_ids:
            await asyncio.sleep(0.05)

    async def adapter_heartbeat(self, adapter_id: str, connected: bool, broker: str = "IBKR") -> None:
        self.adapter_calls.append((adapter_id, connected))
        if self.slow_adapter:
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_family_heartbeats_do_not_block_on_slow_strategy() -> None:
    heartbeat = _FakeHeartbeat()
    heartbeat.slow_strategy_ids.add("slow")

    await emit_family_heartbeats(
        heartbeat,
        "family",
        [{"strategy_id": "slow"}, {"strategy_id": "fast"}],
        adapter_connected=True,
        timeout_s=0.01,
    )

    assert set(heartbeat.strategy_calls) == {"slow", "fast"}
    assert heartbeat.adapter_calls == [("family", True)]


@pytest.mark.asyncio
async def test_family_heartbeats_adapter_timeout_is_isolated() -> None:
    heartbeat = _FakeHeartbeat()
    heartbeat.slow_adapter = True

    await emit_family_heartbeats(
        heartbeat,
        "family",
        [{"strategy_id": "fast"}],
        adapter_connected=False,
        timeout_s=0.01,
    )

    assert heartbeat.strategy_calls == ["fast"]
    assert heartbeat.adapter_calls == [("family", False)]


# ---------------------------------------------------------------------------
# Watchdog-layer: apps.watchdog.checks.check_heartbeats
# ---------------------------------------------------------------------------
def _row(strategy_id: str, age: int, status: str = "OK") -> dict:
    return {
        "strategy_id": strategy_id,
        "heartbeat_age_sec": age,
        "health_status": status,
    }


@pytest.mark.asyncio
async def test_single_stale_heartbeat_keeps_individual_alert() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [_row("s1", 181, "STALE"), _row("s2", 5)]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"family"},
        {"s1": "family", "s2": "family"},
    )

    problems = [r for r in results if r.is_problem]
    assert [r.key for r in problems] == ["heartbeat:s1"]


@pytest.mark.asyncio
async def test_three_stale_heartbeats_are_aggregated() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [
        _row("s1", 181, "STALE"),
        _row("s2", 220, "STALE"),
        _row("s3", 260, "STALE"),
    ]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"family"},
        {"s1": "family", "s2": "family", "s3": "family"},
    )

    problems = [r for r in results if r.is_problem]
    assert [r.key for r in problems] == ["heartbeat:systemic"]
    assert "3 strategies stale" in problems[0].detail
    assert "s1, s2, s3" in problems[0].detail


@pytest.mark.asyncio
async def test_inactive_family_heartbeat_is_skipped() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [_row("s1", 300, "STALE")]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"other"},
        {"s1": "family"},
    )

    assert not [r for r in results if r.is_problem]


@pytest.mark.asyncio
async def test_unknown_strategy_heartbeat_is_skipped() -> None:
    pool = AsyncMock()
    pool.fetch.return_value = [_row("US_ORB_v1", 300, "STALE")]

    results = await check_heartbeats(
        pool,
        {"checks": {"heartbeat": {"stale_threshold_sec": 180}}},
        {"stock"},
        {"IARIC_v1": "stock", "ALCB_v1": "stock"},
    )

    assert not [r for r in results if r.is_problem]
