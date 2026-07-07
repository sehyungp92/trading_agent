"""Base class for diagnostic-only plugins."""

from __future__ import annotations


class DiagnosticOnlyPlugin:
    maturity = "diagnostic"

    def __init__(self, plugin_id: str, strategy_id: str) -> None:
        self.plugin_id = plugin_id
        self.strategy_id = strategy_id
