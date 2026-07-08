from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def order_matches(order: Any, match: Mapping[str, Any]) -> bool:
    strategy_id = _order_value(order, "strategy_id")
    symbol = _order_symbol(order)
    role = _enum_text(_order_value(order, "role", "order_role", default="ENTRY"))
    side = _enum_text(_order_value(order, "side", default="BUY"))
    return (
        str(strategy_id) == str(match.get("strategy_id", strategy_id))
        and str(symbol) == str(match.get("symbol", symbol))
        and role.upper() == str(match.get("role", role)).upper()
        and side.upper() == str(match.get("side", side)).upper()
    )


def candidate_key(match: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(match.get("strategy_id", "")),
            str(match.get("symbol", "")),
            str(match.get("role", "ENTRY")).upper(),
            str(match.get("side", "")).upper(),
            str(int(match.get("sequence", 1))),
        ]
    )


def broker_event_key(event: Mapping[str, Any]) -> str:
    if event.get("exec_id"):
        return str(event["exec_id"])
    match = event.get("order_match", {})
    return "|".join(
        [
            str(event.get("event", "fill")).lower(),
            str(match.get("strategy_id", "")),
            str(match.get("symbol", "")),
            str(match.get("role", "")),
            str(match.get("side", "")),
            str(match.get("sequence", 1)),
        ]
    )


def _order_value(order: Any, *names: str, default: Any = "") -> Any:
    if isinstance(order, Mapping):
        for name in names:
            if name in order:
                return order[name]
        return default
    for name in names:
        if hasattr(order, name):
            return getattr(order, name)
    return default


def _order_symbol(order: Any) -> str:
    if isinstance(order, Mapping):
        return str(order.get("symbol") or order.get("instrument_symbol") or "")
    instrument = getattr(order, "instrument", None)
    if instrument is not None and getattr(instrument, "symbol", None):
        return str(instrument.symbol)
    return str(getattr(order, "symbol", getattr(order, "instrument_symbol", "")))


def _enum_text(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
