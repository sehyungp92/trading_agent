from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def family_surface_adapter_name(fixture: Mapping[str, Any]) -> str:
    return {
        "swing": "swing_unified_overlay_replay_adapter",
        "momentum": "momentum_family_portfolio_backtester",
        "stock": "stock_portfolio_replay",
    }.get(_family_name(fixture), "")


def coordinator_class_name(fixture: Mapping[str, Any]) -> str:
    if not (fixture.get("family_config", {}) or {}).get("strategies"):
        return ""
    return {
        "swing": "SwingFamilyCoordinator",
        "momentum": "MomentumFamilyCoordinator",
        "stock": "StockFamilyCoordinator",
    }.get(_family_name(fixture), "")


def _family_name(fixture: Mapping[str, Any]) -> str:
    return str(fixture.get("family", "") or (fixture.get("family_config", {}) or {}).get("family", ""))
