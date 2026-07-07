# tests/test_telegram_control_surface.py
"""Tests for the daily pinned control surface."""
import pytest
from unittest.mock import AsyncMock

from trading_assistant.comms.telegram_control_surface import ControlSurface
from trading_assistant.comms.telegram_renderer import TelegramRenderer
from trading_assistant.comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig
from trading_assistant.schemas.notifications import ControlPanelState


@pytest.fixture
def mock_adapter():
    config = TelegramBotConfig(token="fake", chat_id="12345")
    adapter = TelegramBotAdapter(config)
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=AsyncMock(message_id=100))
    adapter._bot.edit_message_text = AsyncMock()
    adapter._bot.pin_chat_message = AsyncMock()
    return adapter


@pytest.fixture
def surface(mock_adapter):
    return ControlSurface(adapter=mock_adapter, renderer=TelegramRenderer())


class TestControlSurfaceInit:
    def test_no_pinned_message_initially(self, surface):
        assert surface.current_message_id is None
        assert surface.current_date is None


class TestControlSurfacePublish:
    @pytest.mark.asyncio
    async def test_first_publish_creates_and_pins(self, surface, mock_adapter):
        panel = ControlPanelState(date="2026-03-01", portfolio_pnl=100.0)
        await surface.publish(panel)
        assert surface.current_message_id == 100
        assert surface.current_date == "2026-03-01"
        mock_adapter._bot.send_message.assert_called_once()
        mock_adapter._bot.pin_chat_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_publish_same_day_edits_in_place(self, surface, mock_adapter):
        panel1 = ControlPanelState(date="2026-03-01", portfolio_pnl=100.0)
        await surface.publish(panel1)
        panel2 = ControlPanelState(date="2026-03-01", portfolio_pnl=200.0, daily_report_ready=True)
        await surface.publish(panel2)
        assert mock_adapter._bot.send_message.call_count == 1
        assert mock_adapter._bot.edit_message_text.call_count == 1
        assert surface.current_message_id == 100

    @pytest.mark.asyncio
    async def test_new_day_creates_new_message(self, surface, mock_adapter):
        panel1 = ControlPanelState(date="2026-03-01", portfolio_pnl=100.0)
        await surface.publish(panel1)
        mock_adapter._bot.send_message = AsyncMock(return_value=AsyncMock(message_id=200))
        panel2 = ControlPanelState(date="2026-03-02", portfolio_pnl=50.0)
        await surface.publish(panel2)
        assert surface.current_message_id == 200
        assert surface.current_date == "2026-03-02"
        assert mock_adapter._bot.pin_chat_message.call_count == 2


class TestControlSurfaceUpdate:
    @pytest.mark.asyncio
    async def test_update_alert_count(self, surface, mock_adapter):
        panel = ControlPanelState(date="2026-03-01")
        await surface.publish(panel)
        await surface.update_field(alert_count=2, alert_summary="Bot1 crash, Bot3 timeout")
        assert mock_adapter._bot.edit_message_text.call_count == 1

    @pytest.mark.asyncio
    async def test_update_daily_report_ready(self, surface, mock_adapter):
        panel = ControlPanelState(date="2026-03-01")
        await surface.publish(panel)
        await surface.update_field(daily_report_ready=True)
        assert mock_adapter._bot.edit_message_text.call_count == 1

    @pytest.mark.asyncio
    async def test_update_without_publish_is_noop(self, surface, mock_adapter):
        await surface.update_field(alert_count=5)
        mock_adapter._bot.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_monthly_validation_status(self, surface, mock_adapter):
        panel = ControlPanelState(date="2026-03-01")
        await surface.publish(panel)
        await surface.update_field(monthly_validation_status="Bot2 complete")
        assert mock_adapter._bot.edit_message_text.call_count == 1
