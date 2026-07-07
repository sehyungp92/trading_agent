# tests/test_telegram_handlers.py
"""Tests for Telegram callback query + slash command handlers."""
import pytest
from unittest.mock import AsyncMock

from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter


class TestCallbackRouter:
    @pytest.fixture
    def router(self):
        return TelegramCallbackRouter()

    def test_register_handler(self, router):
        handler = AsyncMock()
        router.register("cmd_daily", handler)
        assert "cmd_daily" in router.handlers

    def test_register_multiple(self, router):
        router.register("cmd_daily", AsyncMock())
        router.register("cmd_weekly", AsyncMock())
        assert len(router.handlers) == 2


class TestCallbackDispatch:
    @pytest.fixture
    def router(self):
        r = TelegramCallbackRouter()
        r.register("cmd_daily", AsyncMock(return_value="Daily report content"))
        r.register("cmd_weekly", AsyncMock(return_value="Weekly report content"))
        r.register("cmd_bot_status", AsyncMock(return_value="All bots OK"))
        return r

    @pytest.mark.asyncio
    async def test_dispatch_known_command(self, router):
        result = await router.dispatch("cmd_daily")
        assert result == "Daily report content"
        router.handlers["cmd_daily"].assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_unknown_command(self, router):
        result = await router.dispatch("cmd_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_passes_context(self, router):
        ctx = {"user_id": "123", "chat_id": "456"}
        await router.dispatch("cmd_daily", context=ctx)
        router.handlers["cmd_daily"].assert_called_once_with(context=ctx)


class TestSlashCommandFallback:
    @pytest.fixture
    def router(self):
        r = TelegramCallbackRouter()
        r.register("cmd_daily", AsyncMock(return_value="Daily"))
        r.register("cmd_weekly", AsyncMock(return_value="Weekly"))
        r.register("cmd_settings", AsyncMock(return_value="Settings"))
        return r

    @pytest.mark.asyncio
    async def test_slash_command_maps_to_callback(self, router):
        result = await router.dispatch_slash("/daily")
        assert result == "Daily"

    @pytest.mark.asyncio
    async def test_slash_command_unknown(self, router):
        result = await router.dispatch_slash("/unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_slash_help_returns_command_list(self, router):
        result = await router.dispatch_slash("/help")
        assert "/daily" in result
        assert "/weekly" in result
        assert "/settings" in result

    @pytest.mark.asyncio
    async def test_settings_slash_command_maps_to_callback(self, router):
        result = await router.dispatch_slash("/settings")
        assert result == "Settings"


class TestApprovalFlow:
    @pytest.fixture
    def router(self):
        r = TelegramCallbackRouter()
        approval_handler = AsyncMock(return_value="Approved 2 items")
        r.register("cmd_approve_all", approval_handler)
        return r

    @pytest.mark.asyncio
    async def test_approve_all_dispatches(self, router):
        result = await router.dispatch("cmd_approve_all")
        assert result == "Approved 2 items"

    @pytest.mark.asyncio
    async def test_approve_with_confirmation_context(self, router):
        ctx = {"confirmed": True}
        await router.dispatch("cmd_approve_all", context=ctx)
        router.handlers["cmd_approve_all"].assert_called_once_with(context=ctx)
