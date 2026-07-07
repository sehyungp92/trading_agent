# tests/test_discord_bot.py
"""Tests for Discord bot adapter."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from trading_assistant.comms.discord_bot import DiscordBotAdapter, DiscordBotConfig


class TestDiscordBotConfig:
    def test_defaults(self):
        cfg = DiscordBotConfig(token="fake-token", channel_id=123456)
        assert cfg.token == "fake-token"
        assert cfg.channel_id == 123456


class TestDiscordBotAdapter:
    @pytest.fixture
    def mock_channel(self):
        ch = AsyncMock()
        msg = MagicMock(id=42)
        ch.send = AsyncMock(return_value=msg)
        ch.fetch_message = AsyncMock(return_value=msg)
        msg.pin = AsyncMock()
        msg.create_thread = AsyncMock(return_value=MagicMock(id=99))
        return ch

    @pytest.fixture
    def adapter(self, mock_channel):
        config = DiscordBotConfig(token="fake-token", channel_id=123456)
        a = DiscordBotAdapter(config)
        a._channel = mock_channel
        return a

    @pytest.mark.asyncio
    async def test_send_message(self, adapter, mock_channel):
        msg_id = await adapter.send_message("Hello")
        assert msg_id == 42
        mock_channel.send.assert_called_once_with(content="Hello")

    @pytest.mark.asyncio
    async def test_send_embed(self, adapter, mock_channel):
        embed_dict = {"title": "Test", "description": "Body", "color": 0xFF0000}
        msg_id = await adapter.send_embed(embed_dict)
        assert msg_id == 42
        mock_channel.send.assert_called_once()
        call_kwargs = mock_channel.send.call_args.kwargs
        assert "embed" in call_kwargs

    @pytest.mark.asyncio
    async def test_pin_message(self, adapter, mock_channel):
        await mock_channel.fetch_message(42)
        await adapter.pin_message(42)
        mock_channel.fetch_message.assert_called_with(42)

    @pytest.mark.asyncio
    async def test_create_thread(self, adapter, mock_channel):
        thread_id = await adapter.create_thread(42, "Bot1 Discussion")
        assert thread_id == 99

    @pytest.mark.asyncio
    async def test_send_to_thread(self, adapter, mock_channel):
        thread = MagicMock()
        thread.send = AsyncMock(return_value=MagicMock(id=55))
        mock_channel.fetch_message = AsyncMock(return_value=MagicMock(id=42))
        adapter._threads = {99: thread}
        msg_id = await adapter.send_to_thread(99, "Thread message")
        assert msg_id == 55
