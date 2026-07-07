# tests/test_comms_integration.py
"""Integration test for the full Phase 6 communication pipeline."""
import pytest
from unittest.mock import AsyncMock

from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, ActionType
from trading_assistant.orchestrator.worker import Worker
from trading_assistant.skills.proactive_scanner import ProactiveScanner
from trading_assistant.comms.dispatcher import NotificationDispatcher, ChannelAdapter
from trading_assistant.comms.renderer import PlainTextRenderer
from trading_assistant.comms.telegram_renderer import TelegramRenderer
from trading_assistant.comms.telegram_control_surface import ControlSurface
from trading_assistant.comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig
from trading_assistant.schemas.notifications import (
    NotificationChannel,
    NotificationPriority,
    NotificationPreferences,
    ChannelConfig,
    NotificationPayload,
    ControlPanelState,
    BotStatusLine,
)


class TestFullNotificationPipeline:
    @pytest.fixture
    def mock_telegram_adapter(self):
        adapter = AsyncMock(spec=ChannelAdapter)
        adapter.send = AsyncMock()
        return adapter

    @pytest.fixture
    def mock_discord_adapter(self):
        adapter = AsyncMock(spec=ChannelAdapter)
        adapter.send = AsyncMock()
        return adapter

    @pytest.fixture
    def prefs(self):
        return NotificationPreferences(
            channels=[
                ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True, chat_id="12345"),
                ChannelConfig(channel=NotificationChannel.DISCORD, enabled=True, chat_id="67890"),
            ]
        )

    @pytest.fixture
    def dispatcher(self, mock_telegram_adapter, mock_discord_adapter):
        d = NotificationDispatcher()
        d.register_adapter(NotificationChannel.TELEGRAM, mock_telegram_adapter)
        d.register_adapter(NotificationChannel.DISCORD, mock_discord_adapter)
        return d

    @pytest.mark.asyncio
    async def test_critical_error_dispatches_to_all_channels(self, dispatcher, prefs, mock_telegram_adapter, mock_discord_adapter):
        prefs.channels[0].quiet_hours_start = 22
        prefs.channels[0].quiet_hours_end = 8
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Bot3 crash",
            body="RuntimeError in signal handler",
        )
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=3)
        assert all(r.success for r in results)
        mock_telegram_adapter.send.assert_called_once()
        mock_discord_adapter.send.assert_called_once()


class TestProactiveScannerToDispatcher:
    @pytest.mark.asyncio
    async def test_morning_errors_dispatched(self):
        scanner = ProactiveScanner()
        errors = [
            {"bot_id": "bot1", "error_type": "ConnectionError", "message": "Timeout", "severity": "HIGH"},
        ]
        result = scanner.morning_scan(events=[], errors=errors, unusual_losses=[])
        mock_adapter = AsyncMock(spec=ChannelAdapter)
        mock_adapter.send = AsyncMock()
        dispatcher = NotificationDispatcher()
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, mock_adapter)
        prefs = NotificationPreferences(
            channels=[ChannelConfig(channel=NotificationChannel.TELEGRAM)]
        )
        for payload in result.payloads:
            await dispatcher.dispatch(payload, prefs, current_hour_utc=7)
        assert mock_adapter.send.call_count == 1


class TestControlSurfaceIntegration:
    @pytest.mark.asyncio
    async def test_publish_and_update_panel(self):
        config = TelegramBotConfig(token="fake", chat_id="12345")
        adapter = TelegramBotAdapter(config)
        adapter._bot = AsyncMock()
        adapter._bot.send_message = AsyncMock(return_value=AsyncMock(message_id=100))
        adapter._bot.edit_message_text = AsyncMock()
        adapter._bot.pin_chat_message = AsyncMock()
        surface = ControlSurface(adapter=adapter, renderer=TelegramRenderer())
        panel = ControlPanelState(
            date="2026-03-01",
            portfolio_pnl=342.0,
            portfolio_pnl_pct=1.2,
            drawdown_pct=-0.3,
            exposure_pct=47.0,
            bot_statuses=[
                BotStatusLine(bot_id="Bot1", status="green", pnl=210.0, wins=4, losses=1, summary="Strong"),
            ],
        )
        await surface.publish(panel)
        assert surface.current_message_id == 100
        await surface.update_field(alert_count=1, alert_summary="Bot3 volume filter")
        adapter._bot.edit_message_text.assert_called_once()


class TestRenderersProduceValidOutput:
    def test_plain_text_daily(self):
        payload = NotificationPayload(notification_type="daily_report", title="Daily Report", body="Portfolio: +$342")
        text = PlainTextRenderer().render(payload)
        assert len(text) > 0

    def test_telegram_daily(self):
        payload = NotificationPayload(notification_type="daily_report", title="Daily Report", body="Portfolio: +$342")
        text = TelegramRenderer().render(payload)
        assert len(text) > 0

    def test_plain_text_alert(self):
        payload = NotificationPayload(notification_type="alert", priority=NotificationPriority.CRITICAL, title="Crash", body="Details")
        text = PlainTextRenderer().render(payload)
        assert "CRITICAL" in text

    def test_telegram_alert(self):
        payload = NotificationPayload(notification_type="alert", priority=NotificationPriority.CRITICAL, title="Crash", body="Details")
        text = TelegramRenderer().render(payload)
        assert "\U0001f6a8" in text


class TestBrainToWorkerNotificationFlow:
    @pytest.mark.asyncio
    async def test_notification_event_flow(self):
        brain = OrchestratorBrain()
        event = {
            "event_type": "notification_trigger",
            "event_id": "notif-001",
            "bot_id": "system",
            "payload": '{"type": "daily_report"}',
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.SEND_NOTIFICATION
        mock_queue = AsyncMock()
        mock_queue.claim = AsyncMock(return_value=[event])
        mock_queue.ack = AsyncMock()
        mock_registry = AsyncMock()
        worker = Worker(queue=mock_queue, registry=mock_registry, brain=brain)
        handler = AsyncMock()
        worker.on_notification = handler
        await worker.process_batch(limit=1)
        handler.assert_called_once()


class TestNotificationEndpoints:
    @pytest.fixture
    def app(self, tmp_path):
        from trading_assistant.orchestrator.app import create_app
        from trading_assistant.orchestrator.config import AppConfig
        return create_app(
            db_dir=str(tmp_path),
            config=AppConfig(allow_unauthenticated_local=True),
        )

    @pytest.mark.asyncio
    async def test_get_preferences_returns_defaults(self, app):
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await app.state.queue.initialize()
            await app.state.registry.initialize()
            resp = await client.get("/notifications/preferences")
            assert resp.status_code == 200
            data = resp.json()
            assert "channels" in data

    @pytest.mark.asyncio
    async def test_update_preferences(self, app):
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await app.state.queue.initialize()
            await app.state.registry.initialize()
            resp = await client.put("/notifications/preferences", json={
                "channels": [
                    {"channel": "telegram", "enabled": True, "chat_id": "12345"},
                ]
            })
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["channels"]) == 1
            assert data["channels"][0]["chat_id"] == "12345"
