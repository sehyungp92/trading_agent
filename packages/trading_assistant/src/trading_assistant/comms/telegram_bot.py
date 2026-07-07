# comms/telegram_bot.py
"""Telegram bot adapter — wraps send/edit/pin operations with lifecycle and retry."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from trading_assistant.comms.base_channel import BaseChannel
from trading_assistant.comms.telegram_handlers import TelegramCallbackResponse
from trading_assistant.orchestrator.input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


@dataclass
class TelegramBotConfig:
    token: str
    chat_id: str
    parse_mode: str = "MarkdownV2"
    # Inbound allowlists (P0-4). If allowed_chat_ids is empty, only the
    # configured chat_id is accepted. allowed_user_ids is optional; when
    # populated, the sender's user.id must also match.
    allowed_chat_ids: set[str] = field(default_factory=set)
    allowed_user_ids: set[int] = field(default_factory=set)

    def is_chat_allowed(self, chat_id: str | int | None) -> bool:
        if chat_id is None:
            return False
        normalized = str(chat_id)
        if self.allowed_chat_ids:
            return normalized in self.allowed_chat_ids
        return normalized == str(self.chat_id)

    def is_user_allowed(self, user_id: int | None) -> bool:
        if not self.allowed_user_ids:
            return True
        return user_id is not None and user_id in self.allowed_user_ids


def _build_inline_keyboard(keyboard: list[list[dict]]) -> dict:
    return {
        "inline_keyboard": [
            [{"text": btn["text"], "callback_data": btn["callback_data"]} for btn in row]
            for row in keyboard
        ]
    }


class TelegramBotAdapter(BaseChannel):
    """Async adapter for Telegram Bot API operations."""

    def __init__(self, config: TelegramBotConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._bot: object | None = None
        self._callback_router = None
        self._polling_task: asyncio.Task | None = None
        self._sanitizer = InputSanitizer()

    def set_callback_router(self, router) -> None:
        """Connect a TelegramCallbackRouter to handle incoming callback queries."""
        self._callback_router = router

    async def _start(self) -> None:
        from telegram import Bot
        self._bot = Bot(token=self._config.token)

    async def _stop(self) -> None:
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        self._bot = None

    async def start_polling(self) -> None:
        """Start polling for incoming updates (callback queries and slash commands).

        Requires a callback router to be set via set_callback_router().
        """
        if self._callback_router is None:
            logger.warning("No callback router set — polling not started")
            return
        self._polling_task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram update polling started")

    async def _poll_loop(self) -> None:
        """Long-poll for Telegram updates and dispatch to callback router."""
        offset = 0
        while True:
            try:
                updates = await self._bot.get_updates(
                    offset=offset, timeout=30,
                    allowed_updates=["callback_query", "message"],
                )
                for update in updates:
                    offset = update.update_id + 1
                    await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Telegram polling error — retrying in 5s")
                await asyncio.sleep(5)

    async def _handle_update(self, update) -> None:
        """Process a single Telegram update.

        Inbound updates are filtered by chat_id / user_id allowlists (P0-4)
        before dispatch. Free-text messages are sanitized via InputSanitizer
        (P0-3) so prompt-injection patterns never reach handlers.
        """
        if update.callback_query is not None:
            query = update.callback_query
            chat = getattr(query.message, "chat", None) if query.message is not None else None
            chat_id = getattr(chat, "id", None)
            user = getattr(query, "from_user", None)
            user_id = getattr(user, "id", None)
            if not self._config.is_chat_allowed(chat_id) or not self._config.is_user_allowed(user_id):
                logger.warning(
                    "Rejecting Telegram callback from unauthorized chat=%s user=%s",
                    chat_id, user_id,
                )
                try:
                    await query.answer(text="Unauthorized")
                except Exception:
                    pass
                return

            callback_data = query.data or ""
            try:
                result = await self._callback_router.dispatch(callback_data)
                response = self._normalize_callback_response(result)
                answer = response.answer or (response.text if response.text else "Done")
                await query.answer(text=answer[:200])
                if response.text:
                    try:
                        if response.edit_message and query.message is not None:
                            await self.edit_message(
                                query.message.message_id,
                                response.text,
                                keyboard=response.keyboard,
                            )
                        else:
                            await self.send_message(response.text, keyboard=response.keyboard)
                    except Exception:
                        logger.warning("Failed to send callback result message")
            except Exception:
                logger.exception("Callback dispatch error for %s", callback_data)
                await query.answer(text="Error processing request")

        elif update.message is not None and update.message.text:
            chat = getattr(update.message, "chat", None)
            chat_id = getattr(chat, "id", None)
            user = getattr(update.message, "from_user", None)
            user_id = getattr(user, "id", None)
            if not self._config.is_chat_allowed(chat_id) or not self._config.is_user_allowed(user_id):
                logger.warning(
                    "Rejecting Telegram message from unauthorized chat=%s user=%s",
                    chat_id, user_id,
                )
                return

            text = update.message.text.strip()
            sanitized = self._sanitizer.sanitize(text, source="telegram")
            if not sanitized.safe:
                logger.warning("Blocked Telegram message: %s", sanitized.reason)
                try:
                    await self.send_message(
                        "Sorry, your message was blocked by the input filter."
                    )
                except Exception:
                    pass
                return

            if text.startswith("/"):
                command = text.split()[0].lower()
                try:
                    result = await self._callback_router.dispatch_slash(command)
                    response = self._normalize_callback_response(result)
                    if response.text:
                        await self.send_message(response.text, keyboard=response.keyboard)
                except Exception:
                    logger.exception("Slash command error for %s", command)

    async def _send(self, text: str, keyboard: list[list[dict]] | None = None) -> int:
        return await self.send_message(text, keyboard=keyboard)

    async def start(self) -> None:
        await super().start()

    async def send_message(self, text: str, keyboard: list[list[dict]] | None = None) -> int:
        kwargs: dict = {"chat_id": self._config.chat_id, "text": text}
        if keyboard:
            kwargs["reply_markup"] = _build_inline_keyboard(keyboard)
        result = await self._bot.send_message(**kwargs)
        return result.message_id

    async def edit_message(self, message_id: int, text: str, keyboard: list[list[dict]] | None = None) -> None:
        kwargs: dict = {"chat_id": self._config.chat_id, "message_id": message_id, "text": text}
        if keyboard:
            kwargs["reply_markup"] = _build_inline_keyboard(keyboard)
        await self._bot.edit_message_text(**kwargs)

    async def pin_message(self, message_id: int) -> None:
        await self._bot.pin_chat_message(
            chat_id=self._config.chat_id, message_id=message_id, disable_notification=True
        )

    async def send_and_pin(self, text: str, keyboard: list[list[dict]] | None = None) -> int:
        msg_id = await self.send_message(text, keyboard=keyboard)
        await self.pin_message(msg_id)
        return msg_id

    def _normalize_callback_response(
        self, result: str | TelegramCallbackResponse | None
    ) -> TelegramCallbackResponse:
        if isinstance(result, TelegramCallbackResponse):
            return result
        if isinstance(result, str):
            return TelegramCallbackResponse(text=result)
        return TelegramCallbackResponse()
