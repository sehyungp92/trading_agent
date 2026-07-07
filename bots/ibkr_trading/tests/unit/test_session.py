"""Unit tests for libs.broker_ibkr.session — UnifiedIBSession."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest
from libs.broker_ibkr.throttler import PacingChannel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_group_config(**overrides) -> MagicMock:
    """Return a mock ConnectionGroupConfig with sensible defaults."""
    cfg = MagicMock()
    cfg.host = overrides.get("host", "127.0.0.1")
    cfg.port = overrides.get("port", 4001)
    cfg.client_id = overrides.get("client_id", 1)
    cfg.readonly = False
    cfg.reconnect_max_retries = 3
    cfg.reconnect_base_delay_s = 1.0
    cfg.reconnect_max_delay_s = 30.0
    cfg.pacing_messages_per_sec = 45.0
    cfg.pacing_orders_per_sec = 5.0
    return cfg


@patch("libs.broker_ibkr.session.FarmMonitor")
@patch("libs.broker_ibkr.session.HeartbeatMonitor")
@patch("libs.broker_ibkr.session.ConnectionManager")
def _make_session(
    MockConnMgr,
    MockHeartbeat,
    MockFarmMon,
    group_configs=None,
    strategy_map=None,
):
    """Create a UnifiedIBSession with mocked dependencies."""
    from libs.broker_ibkr.session import UnifiedIBSession

    if group_configs is None:
        group_configs = {"futures": _mock_group_config(client_id=1)}
    if strategy_map is None:
        strategy_map = {"ATRS_v1": "futures"}

    session = UnifiedIBSession(group_configs, strategy_map)
    return session, MockConnMgr, MockHeartbeat, MockFarmMon


# ---------------------------------------------------------------------------
# KeyError -> ValueError guards
# ---------------------------------------------------------------------------
class TestValueErrorGuards:
    """Verify all public methods raise ValueError (not KeyError) for bad IDs."""

    def test_strategy_group_unknown_raises_valueerror(self) -> None:
        session, *_ = _make_session()

        with pytest.raises(ValueError, match="Unknown strategy_id.*BOGUS"):
            session.strategy_group("BOGUS")

    def test_strategy_group_error_includes_registered_ids(self) -> None:
        session, *_ = _make_session()

        with pytest.raises(ValueError, match="ATRS_v1"):
            session.strategy_group("BOGUS")

    def test_get_ib_unknown_raises_valueerror(self) -> None:
        session, *_ = _make_session()

        with pytest.raises(ValueError, match="Unknown strategy_id.*BOGUS"):
            session.get_ib("BOGUS")

    @pytest.mark.asyncio
    async def test_wait_ready_unknown_group_raises_valueerror(self) -> None:
        session, *_ = _make_session()

        with pytest.raises(ValueError, match="Unknown group_id.*bogus_group"):
            await session.wait_ready("bogus_group")

    def test_wait_ready_error_includes_available_groups(self) -> None:
        session, *_ = _make_session()

        with pytest.raises(ValueError, match="futures"):
            session._get_group("bogus_group")

    def test_register_farm_recovery_callback_unknown_raises_valueerror(self) -> None:
        session, *_ = _make_session()

        with pytest.raises(ValueError, match="Unknown group_id.*bogus"):
            session.register_farm_recovery_callback("bogus", lambda farm: None)

    def test_set_reconnect_callback_unknown_raises_valueerror(self) -> None:
        session, *_ = _make_session()

        with pytest.raises(ValueError, match="Unknown group_id.*bogus"):
            session.set_reconnect_callback("bogus", lambda: None)


# ---------------------------------------------------------------------------
# Partial startup cleanup
# ---------------------------------------------------------------------------
class TestPartialStartupCleanup:
    """When start() fails partway, already-started groups must be cleaned up."""

    @pytest.mark.asyncio
    async def test_partial_startup_calls_stop(self) -> None:
        """If group2 connect() raises, group1's resources should still be stopped."""
        with (
            patch("libs.broker_ibkr.session.ConnectionManager") as MockConnMgr,
            patch("libs.broker_ibkr.session.HeartbeatMonitor") as MockHeartbeat,
            patch("libs.broker_ibkr.session.FarmMonitor") as MockFarmMon,
        ):
            from libs.broker_ibkr.session import UnifiedIBSession

            group_configs = {
                "group1": _mock_group_config(client_id=1),
                "group2": _mock_group_config(client_id=2),
            }
            strategy_map = {"s1": "group1", "s2": "group2"}

            session = UnifiedIBSession(group_configs, strategy_map)

            # group1's connect succeeds, group2's connect raises
            g1_conn = session._groups["group1"].conn
            g2_conn = session._groups["group2"].conn

            # Setup group1 connect to succeed (mock all subsidiary calls)
            g1_conn.connect = AsyncMock()
            g1_conn.wait_until_ready = AsyncMock()
            g1_conn.ib.reqMarketDataType = MagicMock()
            g1_conn.ib.client.getReqId.return_value = 100
            g1_conn.disconnect = AsyncMock()

            # Mock HeartbeatMonitor and FarmMonitor constructed for group1
            mock_hb = MagicMock()
            mock_hb.start = AsyncMock()
            mock_hb.stop = AsyncMock()
            MockHeartbeat.return_value = mock_hb

            mock_fm = MagicMock()
            mock_fm.start = MagicMock()
            mock_fm.stop = MagicMock()
            MockFarmMon.return_value = mock_fm

            # group2 connect raises
            g2_conn.connect = AsyncMock(side_effect=ConnectionError("Gateway down"))
            g2_conn.disconnect = AsyncMock()

            with pytest.raises(ConnectionError, match="Gateway down"):
                await session.start()

            # stop() should have been called, which disconnects group1
            g1_conn.disconnect.assert_awaited()


# ---------------------------------------------------------------------------
# Resilient stop — errors in one group don't block others
# ---------------------------------------------------------------------------
class TestResilientStop:
    """Verify stop() continues through per-group errors."""

    @pytest.mark.asyncio
    async def test_error_in_one_group_does_not_block_others(self) -> None:
        """If grpA's disconnect() raises, grpB should still be disconnected."""
        with (
            patch("libs.broker_ibkr.session.ConnectionManager") as MockConnMgr,
            patch("libs.broker_ibkr.session.HeartbeatMonitor"),
            patch("libs.broker_ibkr.session.FarmMonitor"),
        ):
            # Each call to ConnectionManager() must return a distinct mock
            # so that the two groups don't share the same conn object.
            MockConnMgr.side_effect = lambda cfg: MagicMock()

            from libs.broker_ibkr.session import UnifiedIBSession

            group_configs = {
                "grpA": _mock_group_config(client_id=1),
                "grpB": _mock_group_config(client_id=2),
            }
            strategy_map = {"s1": "grpA"}

            session = UnifiedIBSession(group_configs, strategy_map)

            grp_a = session._groups["grpA"]
            grp_b = session._groups["grpB"]

            # Confirm the two groups have distinct conn objects
            assert grp_a.conn is not grp_b.conn

            # Wire up individual disconnect mocks
            disconnect_a = AsyncMock(side_effect=RuntimeError("socket error"))
            disconnect_b = AsyncMock()

            grp_a.conn.disconnect = disconnect_a
            grp_a.heartbeat = None
            grp_a.farm_monitor = None

            grp_b.conn.disconnect = disconnect_b
            grp_b.heartbeat = None
            grp_b.farm_monitor = None

            # stop() iterates in reverse order and captures the first error
            with pytest.raises(RuntimeError, match="socket error"):
                await session.stop()

            # Both groups should have had disconnect attempted
            disconnect_a.assert_awaited()
            disconnect_b.assert_awaited()

    @pytest.mark.asyncio
    async def test_stop_clears_ready_events(self) -> None:
        """After stop(), all group ready events should be cleared."""
        with (
            patch("libs.broker_ibkr.session.ConnectionManager"),
            patch("libs.broker_ibkr.session.HeartbeatMonitor"),
            patch("libs.broker_ibkr.session.FarmMonitor"),
        ):
            from libs.broker_ibkr.session import UnifiedIBSession

            group_configs = {"g1": _mock_group_config()}
            session = UnifiedIBSession(group_configs, {})

            grp = session._groups["g1"]
            grp.ready.set()
            grp.conn.disconnect = AsyncMock()
            grp.heartbeat = None
            grp.farm_monitor = None

            await session.stop()

            assert not grp.ready.is_set(), "ready event should be cleared after stop"

    @pytest.mark.asyncio
    async def test_stop_calls_heartbeat_and_farm_stop(self) -> None:
        """stop() should call heartbeat.stop() and farm_monitor.stop()."""
        with (
            patch("libs.broker_ibkr.session.ConnectionManager"),
            patch("libs.broker_ibkr.session.HeartbeatMonitor"),
            patch("libs.broker_ibkr.session.FarmMonitor"),
        ):
            from libs.broker_ibkr.session import UnifiedIBSession

            group_configs = {"g1": _mock_group_config()}
            session = UnifiedIBSession(group_configs, {})

            grp = session._groups["g1"]
            grp.conn.disconnect = AsyncMock()

            mock_hb = MagicMock()
            mock_hb.stop = AsyncMock()
            grp.heartbeat = mock_hb

            mock_fm = MagicMock()
            mock_fm.stop = MagicMock()
            grp.farm_monitor = mock_fm

            await session.stop()

            mock_hb.stop.assert_awaited_once()
            mock_fm.stop.assert_called_once()


class TestHistoricalDataWrapper:
    @pytest.mark.asyncio
    async def test_req_historical_data_throttles_and_passes_timeout(self) -> None:
        session, *_ = _make_session()
        session.throttled = AsyncMock()
        session.ib.reqHistoricalDataAsync = AsyncMock(return_value=[object()])

        rows = await session.req_historical_data(
            "contract", "", "1 D", "1 day", "TRADES", False, request_kind="quick",
        )

        assert rows
        session.throttled.assert_awaited_once_with(PacingChannel.HISTORICAL)
        kwargs = session.ib.reqHistoricalDataAsync.await_args.kwargs
        assert kwargs["timeout"] == 15.0

    @pytest.mark.asyncio
    async def test_req_historical_data_opens_breaker_and_skips_refreshes(self) -> None:
        session, *_ = _make_session()
        session.throttled = AsyncMock()

        async def slow_empty(*args, **kwargs):
            await asyncio.sleep(0.02)
            return []

        session.ib.reqHistoricalDataAsync = AsyncMock(side_effect=slow_empty)

        for _ in range(3):
            await session.req_historical_data(
                "contract", "", "1 D", "1 day", "TRADES", False, timeout=0.01,
            )

        assert session.historical_health()["timeout_count"] == 3
        session.ib.reqHistoricalDataAsync.reset_mock()

        rows = await session.req_historical_data(
            "contract", "", "1 D", "1 day", "TRADES", False, timeout=0.01,
        )

        assert rows == []
        session.ib.reqHistoricalDataAsync.assert_not_awaited()
        assert session.historical_health()["skipped_count"] == 1

    @pytest.mark.asyncio
    async def test_req_historical_data_does_not_skip_subscriptions(self) -> None:
        session, *_ = _make_session()
        session._historical_breaker_open_until = 10**9
        session.throttled = AsyncMock()
        session.ib.reqHistoricalDataAsync = AsyncMock(return_value=[object()])

        await session.req_historical_data(
            "contract", "", "1 D", "1 day", "TRADES", False, keepUpToDate=True,
        )

        session.ib.reqHistoricalDataAsync.assert_awaited_once()
        assert session.ib.reqHistoricalDataAsync.await_args.kwargs["timeout"] == 45.0

    @pytest.mark.asyncio
    async def test_req_historical_data_filters_incomplete_intraday_tail_when_requested(self) -> None:
        session, *_ = _make_session()
        session.throttled = AsyncMock()
        session.ib.reqHistoricalDataAsync = AsyncMock(
            return_value=[
                SimpleNamespace(date=datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc)),
                SimpleNamespace(date=datetime(2024, 1, 5, 11, 0, tzinfo=timezone.utc)),
            ]
        )

        rows = await session.req_historical_data(
            "contract",
            "",
            "2 D",
            "1 hour",
            "TRADES",
            False,
            completed_only=True,
            as_of=datetime(2024, 1, 5, 11, 30, tzinfo=timezone.utc),
        )

        assert len(rows) == 1
        assert rows[0].date == datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_req_historical_data_filters_same_day_rth_daily_tail_when_requested(self) -> None:
        session, *_ = _make_session()
        session.throttled = AsyncMock()
        session.ib.reqHistoricalDataAsync = AsyncMock(
            return_value=[
                SimpleNamespace(date="20240104"),
                SimpleNamespace(date="20240105"),
            ]
        )

        rows = await session.req_historical_data(
            "contract",
            "",
            "200 D",
            "1 day",
            "TRADES",
            True,
            completed_only=True,
            as_of=datetime(2024, 1, 5, 15, 30, tzinfo=timezone.utc),
        )

        assert [row.date for row in rows] == ["20240104"]

    @pytest.mark.asyncio
    async def test_req_historical_data_does_not_filter_explicit_enddatetime(self) -> None:
        session, *_ = _make_session()
        session.throttled = AsyncMock()
        session.ib.reqHistoricalDataAsync = AsyncMock(
            return_value=[
                SimpleNamespace(date=datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc)),
                SimpleNamespace(date=datetime(2024, 1, 5, 11, 0, tzinfo=timezone.utc)),
            ]
        )

        rows = await session.req_historical_data(
            "contract",
            "20240105 11:30:00",
            "2 D",
            "1 hour",
            "TRADES",
            False,
            completed_only=True,
            as_of=datetime(2024, 1, 5, 11, 30, tzinfo=timezone.utc),
        )

        assert len(rows) == 2
