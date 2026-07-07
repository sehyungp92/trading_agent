# comms/telegram_control_surface.py
"""Telegram control surface — one pinned message per day, updated in-place."""
from __future__ import annotations

from trading_assistant.comms.telegram_bot import TelegramBotAdapter
from trading_assistant.comms.telegram_renderer import TelegramRenderer
from trading_assistant.schemas.notifications import ControlPanelState


class ControlSurface:
    """Manages the daily pinned control panel message."""

    def __init__(self, adapter: TelegramBotAdapter, renderer: TelegramRenderer) -> None:
        self._adapter = adapter
        self._renderer = renderer
        self._current_message_id: int | None = None
        self._current_date: str | None = None
        self._current_panel: ControlPanelState | None = None

    @property
    def current_message_id(self) -> int | None:
        return self._current_message_id

    @property
    def current_date(self) -> str | None:
        return self._current_date

    async def publish(self, panel: ControlPanelState) -> None:
        text, keyboard = self._renderer.render_control_panel_with_keyboard(panel)
        self._current_panel = panel
        if self._current_date == panel.date and self._current_message_id is not None:
            await self._adapter.edit_message(self._current_message_id, text, keyboard=keyboard)
        else:
            msg_id = await self._adapter.send_and_pin(text, keyboard=keyboard)
            self._current_message_id = msg_id
            self._current_date = panel.date

    async def update_field(self, **kwargs) -> None:
        if self._current_panel is None:
            return
        allowed_fields = set(ControlPanelState.model_fields)
        updates = {key: value for key, value in kwargs.items() if key in allowed_fields}
        if not updates:
            return
        updated = self._current_panel.model_copy(update=updates)
        self._current_panel = updated
        text, keyboard = self._renderer.render_control_panel_with_keyboard(updated)
        await self._adapter.edit_message(self._current_message_id, text, keyboard=keyboard)
