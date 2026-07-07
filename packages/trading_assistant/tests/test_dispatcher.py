# tests/test_dispatcher.py
"""Tests for the central notification dispatcher."""
import pytest
from unittest.mock import AsyncMock

from trading_assistant.comms.dispatcher import NotificationDispatcher, ChannelAdapter
from trading_assistant.schemas.notifications import (
    NotificationChannel,
    NotificationPriority,
    NotificationPreferences,
    ChannelConfig,
    NotificationPayload,
)


class TestDispatcherInit:
    def test_register_adapter(self):
        dispatcher = NotificationDispatcher()
        adapter = AsyncMock(spec=ChannelAdapter)
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, adapter)
        assert NotificationChannel.TELEGRAM in dispatcher.adapters


class TestDispatchRouting:
    @pytest.fixture
    def telegram_adapter(self):
        a = AsyncMock(spec=ChannelAdapter)
        a.send = AsyncMock()
        return a

    @pytest.fixture
    def discord_adapter(self):
        a = AsyncMock(spec=ChannelAdapter)
        a.send = AsyncMock()
        return a

    @pytest.fixture
    def prefs(self):
        return NotificationPreferences(
            channels=[
                ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True, chat_id="12345"),
                ChannelConfig(channel=NotificationChannel.DISCORD, enabled=True, chat_id="67890"),
            ]
        )

    @pytest.fixture
    def dispatcher(self, telegram_adapter, discord_adapter):
        d = NotificationDispatcher()
        d.register_adapter(NotificationChannel.TELEGRAM, telegram_adapter)
        d.register_adapter(NotificationChannel.DISCORD, discord_adapter)
        return d

    @pytest.mark.asyncio
    async def test_dispatches_to_all_active_channels(self, dispatcher, prefs, telegram_adapter, discord_adapter):
        payload = NotificationPayload(
            notification_type="daily_report",
            priority=NotificationPriority.NORMAL,
            title="Daily Report",
            body="Content",
        )
        await dispatcher.dispatch(payload, prefs, current_hour_utc=12)
        telegram_adapter.send.assert_called_once()
        discord_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_disabled_channel(self, dispatcher, prefs, telegram_adapter, discord_adapter):
        prefs.channels[1].enabled = False
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.NORMAL,
            title="Test",
            body="Body",
        )
        await dispatcher.dispatch(payload, prefs, current_hour_utc=12)
        telegram_adapter.send.assert_called_once()
        discord_adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_quiet_hours(self, dispatcher, telegram_adapter, discord_adapter):
        prefs = NotificationPreferences(
            channels=[
                ChannelConfig(
                    channel=NotificationChannel.TELEGRAM,
                    quiet_hours_start=22,
                    quiet_hours_end=8,
                ),
                ChannelConfig(channel=NotificationChannel.DISCORD, enabled=True),
            ]
        )
        payload = NotificationPayload(
            notification_type="daily_report",
            priority=NotificationPriority.NORMAL,
            title="Report",
            body="Content",
        )
        await dispatcher.dispatch(payload, prefs, current_hour_utc=3)
        telegram_adapter.send.assert_not_called()
        discord_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_critical_bypasses_quiet_hours(self, dispatcher, telegram_adapter, discord_adapter):
        prefs = NotificationPreferences(
            channels=[
                ChannelConfig(
                    channel=NotificationChannel.TELEGRAM,
                    quiet_hours_start=22,
                    quiet_hours_end=8,
                ),
            ]
        )
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="CRASH",
            body="Bot3 down",
        )
        await dispatcher.dispatch(payload, prefs, current_hour_utc=3)
        telegram_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_adapters_registered(self):
        dispatcher = NotificationDispatcher()
        prefs = NotificationPreferences(
            channels=[ChannelConfig(channel=NotificationChannel.TELEGRAM)]
        )
        payload = NotificationPayload(
            notification_type="alert",
            title="Test",
            body="Body",
        )
        await dispatcher.dispatch(payload, prefs, current_hour_utc=12)

    @pytest.mark.asyncio
    async def test_dispatch_returns_delivery_results(self, dispatcher, prefs, telegram_adapter, discord_adapter):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Report",
            body="Content",
        )
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=12)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_adapter_failure_doesnt_block_others(self, dispatcher, prefs, telegram_adapter, discord_adapter):
        telegram_adapter.send.side_effect = Exception("Network error")
        payload = NotificationPayload(
            notification_type="alert",
            title="Test",
            body="Body",
        )
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=12)
        assert results[0].success is False
        assert results[1].success is True
        discord_adapter.send.assert_called_once()
