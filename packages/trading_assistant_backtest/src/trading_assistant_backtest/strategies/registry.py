"""Strategy plugin registry."""

from __future__ import annotations

from typing import Any


class StrategyRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}

    def register(self, plugin: Any) -> None:
        self._plugins[str(plugin.plugin_id)] = plugin

    def get(self, plugin_id: str) -> Any | None:
        return self._plugins.get(plugin_id)
