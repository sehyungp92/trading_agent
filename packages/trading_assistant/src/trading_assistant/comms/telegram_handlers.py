# comms/telegram_handlers.py
"""Telegram callback query router and slash command fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

_SLASH_MAP: dict[str, str] = {
    "/daily": "cmd_daily",
    "/weekly": "cmd_weekly",
    "/botstatus": "cmd_bot_status",
    "/bot_status": "cmd_bot_status",
    "/topmissed": "cmd_top_missed",
    "/top_missed": "cmd_top_missed",
    "/openprs": "cmd_open_prs",
    "/open_prs": "cmd_open_prs",
    "/approve": "cmd_approve_all",
    "/pending": "cmd_pending",
    "/settings": "cmd_settings",
}


@dataclass
class TelegramCallbackResponse:
    """Structured callback response for Telegram slash commands and buttons."""

    text: str = ""
    keyboard: list[list[dict]] | None = None
    answer: str | None = None
    edit_message: bool = False


class TelegramCallbackRouter:
    """Routes callback queries and slash commands to handler functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Awaitable[str | TelegramCallbackResponse | None]]] = {}

    @property
    def handlers(self) -> dict[str, Callable[..., Awaitable[str | TelegramCallbackResponse | None]]]:
        return self._handlers

    def register(
        self,
        callback_data: str,
        handler: Callable[..., Awaitable[str | TelegramCallbackResponse | None]],
    ) -> None:
        self._handlers[callback_data] = handler

    async def dispatch(
        self, callback_data: str, context: dict | None = None
    ) -> str | TelegramCallbackResponse | None:
        # Try exact match first
        handler = self._handlers.get(callback_data)
        if handler is not None:
            if context is not None:
                return await handler(context=context)
            return await handler()

        # Try prefix match — extract suffix as positional arg (e.g. request_id)
        for prefix, handler in self._handlers.items():
            if callback_data.startswith(prefix) and prefix != callback_data:
                suffix = callback_data[len(prefix):]
                return await handler(suffix)
        return None

    async def dispatch_slash(
        self, command: str, context: dict | None = None
    ) -> str | TelegramCallbackResponse | None:
        if command == "/help":
            return self._build_help_text()
        callback_data = _SLASH_MAP.get(command)
        if callback_data is None:
            return None
        return await self.dispatch(callback_data, context=context)

    def _build_help_text(self) -> str:
        lines = ["Available commands:"]
        for slash, cb in sorted(_SLASH_MAP.items()):
            if cb in self._handlers:
                lines.append(f"  {slash}")
        return "\n".join(lines)
