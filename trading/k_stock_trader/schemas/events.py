"""Compatibility wrapper for legacy ``schemas.events`` imports."""

from __future__ import annotations

from typing import Any

from pydantic import model_validator
from trading_assistant.schemas.events import *  # noqa: F403
from trading_assistant.schemas.events import TradeEvent as _AssistantTradeEvent


class TradeEvent(_AssistantTradeEvent):
    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_k_stock_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for key in ("decision_ref", "action_ref", "portfolio_decision_ref"):
            if normalized.get(key) == "":
                normalized[key] = None
        if normalized.get("commission") is None:
            normalized["commission"] = 0.0
        return normalized
