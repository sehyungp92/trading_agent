from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from libs.oms.models.events import OMSEventType
from libs.oms.models.order import OMSOrder


FamilyResolver = Callable[[str], str]

_EVENTS_TO_COMPARE = {
    "FILL",
    "RISK_DENIAL",
    "ORDER_FILLED",
    "ORDER_CANCELLED",
    "ORDER_REJECTED",
    "ORDER_EXPIRED",
}
_KNOWN_EVENT_TYPES = {item.value for item in OMSEventType} | {"RISK_DENIED"}


def normalize_order_intents(
    orders: Iterable[Any],
    *,
    family_for_strategy: FamilyResolver | None = None,
    instrument_ticks: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    normalized = [
        _normalize_order(order, family_for_strategy, instrument_ticks or {})
        for order in orders
        if order is not None
    ]
    return sorted(normalized, key=lambda row: _json_key(row))


def normalize_oms_events(
    events: Iterable[Any],
    *,
    family_for_strategy: FamilyResolver | None = None,
    instrument_ticks: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        row = _normalize_event(event, family_for_strategy, instrument_ticks or {})
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda row: _json_key(row))


def normalize_trade_ledger(
    rows: Iterable[Any],
    *,
    family_for_strategy: FamilyResolver | None = None,
    instrument_ticks: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    normalized = [
        _normalize_trade(row, family_for_strategy, instrument_ticks or {})
        for row in rows
        if row is not None
    ]
    return sorted(normalized, key=lambda row: _json_key(row))


def normalize_state_snapshot(value: Any) -> dict[str, Any]:
    normalized = _canonical(value)
    return normalized if isinstance(normalized, dict) else {"value": normalized}


def normalize_reason(reason: str | None) -> str:
    if not reason:
        return ""
    text = str(reason).strip()
    lower = text.lower()
    # Portfolio-rule messages include volatile explanatory tails; bucket only
    # the stable rule name so live/replay compare the same denial contract.
    if lower.startswith("portfolio rule:"):
        detail = text.split(":", 2)
        return f"portfolio_rule:{detail[1].strip() if len(detail) > 1 else ''}"
    if ":" in text:
        return text.split(":", 1)[0].strip().lower().replace(" ", "_")
    return lower.replace(" ", "_")


def _normalize_order(
    order: Any,
    family_for_strategy: FamilyResolver | None,
    instrument_ticks: Mapping[str, float],
) -> dict[str, Any]:
    if isinstance(order, OMSOrder):
        strategy_id = order.strategy_id
        symbol = order.instrument.symbol if order.instrument else ""
        side = _enum_value(order.side)
        qty = order.qty
        order_type = _enum_value(order.order_type)
        role = _enum_value(order.role)
        limit_price = order.limit_price
        stop_price = order.stop_price
        tif = order.tif
        client_tag = _stable_client_tag(order.client_order_id, symbol, role)
    else:
        data = _as_mapping(order, category="order")
        strategy_id = str(data.get("strategy_id", ""))
        symbol = str(data.get("symbol") or data.get("contract_symbol") or "")
        side = str(data.get("side") or data.get("action") or "")
        qty = data.get("qty", 0)
        order_type = str(data.get("order_type", ""))
        role = str(data.get("role") or data.get("order_role") or "")
        limit_price = data.get("limit_price")
        stop_price = data.get("stop_price")
        tif = str(data.get("tif", "DAY"))
        client_tag = _stable_client_tag(
            str(data.get("client_order_id") or data.get("client_tag") or ""),
            symbol,
            role,
        )

    tick = instrument_ticks.get(symbol, 0.01)
    return {
        "strategy_id": strategy_id,
        "family": family_for_strategy(strategy_id) if family_for_strategy else "",
        "symbol": symbol,
        "side": side.upper(),
        "qty": _number(qty),
        "order_type": order_type.upper(),
        "tif": tif,
        "limit_price": _price(limit_price, tick),
        "stop_price": _price(stop_price, tick),
        "parent_order_id": "",
        "client_tag": client_tag,
        "order_role": role.upper(),
    }


def _normalize_event(
    event: Any,
    family_for_strategy: FamilyResolver | None,
    instrument_ticks: Mapping[str, float],
) -> dict[str, Any] | None:
    data = _as_mapping(event, category="event")
    event_type = _enum_value(data.get("event_type")).upper()
    if not event_type:
        raise TypeError("unsupported parity event value: missing event_type")
    if event_type not in _KNOWN_EVENT_TYPES:
        raise TypeError(f"unsupported parity event type: {event_type}")
    if event_type not in _EVENTS_TO_COMPARE:
        return None

    payload = data.get("payload")
    if payload is None:
        payload = {}
    elif not isinstance(payload, Mapping):
        raise TypeError(f"unsupported parity event payload value: {type(payload).__name__}")
    strategy_id = str(data.get("strategy_id", "") or payload.get("strategy_id", ""))
    symbol = str(payload.get("symbol", ""))
    tick = instrument_ticks.get(symbol, 0.01)

    row = {
        "event_type": event_type,
        "strategy_id": strategy_id,
        "family": family_for_strategy(strategy_id) if family_for_strategy else "",
        "symbol": symbol,
        "side": str(payload.get("side", "")).upper(),
        "qty": _number(payload.get("qty", payload.get("filled_qty", 0))),
        "price": _price(payload.get("price", payload.get("avg_fill_price")), tick),
        "status": str(payload.get("status", "")).upper(),
        "reason": normalize_reason(str(payload.get("reason") or payload.get("reject_reason") or "")),
        "order_role": str(payload.get("role", "")).upper(),
    }
    row["event_time"] = _timestamp(payload.get("timestamp") or data.get("timestamp"))
    return row


def _normalize_trade(
    row: Any,
    family_for_strategy: FamilyResolver | None,
    instrument_ticks: Mapping[str, float],
) -> dict[str, Any]:
    data = _as_mapping(row, category="trade")
    strategy_id = str(data.get("strategy_id", ""))
    symbol = str(data.get("symbol") or data.get("instrument_symbol") or "")
    tick = instrument_ticks.get(symbol, 0.01)
    return {
        "strategy_id": strategy_id,
        "family": family_for_strategy(strategy_id) if family_for_strategy else str(data.get("family", "")),
        "symbol": symbol,
        "direction": str(data.get("direction", "")).upper(),
        "qty": _number(data.get("qty", data.get("quantity", 0))),
        "entry_time": _timestamp(data.get("entry_time") or data.get("entry_ts")),
        "entry_price": _price(data.get("entry_price"), tick),
        "exit_time": _timestamp(data.get("exit_time") or data.get("exit_ts")),
        "exit_price": _price(data.get("exit_price"), tick),
        "gross_pnl": _money(data.get("gross_pnl", 0.0)),
        "commission": _money(data.get("commission", 0.0)),
        "net_pnl": _money(data.get("net_pnl", 0.0)),
        "exit_reason": str(data.get("exit_reason") or ""),
        "r_multiple": _rounded(data.get("r_multiple")),
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        return _timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TypeError(f"unsupported non-finite parity state decimal: {value!r}")
        return _rounded(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, val in value.items():
            key_s = str(key)
            if key_s in {
                "oms_order_id",
                "broker_order_id",
                "perm_id",
                "broker_order_ref",
                "ref",
                "created_at",
                "submitted_at",
                "acked_at",
                "last_update_at",
                "last_heartbeat_ts",
                "last_error_ts",
            }:
                continue
            if key_s in {"client_order_id", "client_tag"}:
                symbol = str(value.get("symbol") or value.get("instrument_symbol") or "")
                role = str(value.get("role") or value.get("order_role") or "")
                stable = _stable_client_tag(str(val or ""), symbol, role)
                if not stable:
                    continue
                cleaned[key_s] = stable
                continue
            cleaned[key_s] = _canonical(val)
        return {key: cleaned[key] for key in sorted(cleaned)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(item) for item in value]
        if all(isinstance(item, Mapping) for item in items):
            return sorted(items, key=_json_key)
        return sorted(items, key=_json_key) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"unsupported non-finite parity state float: {value!r}")
        return _rounded(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int) or isinstance(value, bool) or value is None:
        return value
    raise TypeError(f"unsupported parity state value: {type(value).__name__}")


def _as_mapping(value: Any, *, category: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"unsupported parity {category} value: {type(value).__name__}")


def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported parity enum value: {type(value).__name__}")


def _stable_client_tag(client_tag: str, symbol: str, role: str) -> str:
    if not client_tag:
        return ""
    lower = client_tag.lower()
    if len(client_tag) >= 12 and any(ch in lower for ch in "-_"):
        suffix = lower.rsplit("-", 1)[-1].rsplit("_", 1)[-1]
        if len(suffix) >= 8 and all(ch in "0123456789abcdef" for ch in suffix):
            return f"{symbol}:{role.upper()}"
    return client_tag


def _timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise TypeError(f"unsupported parity timestamp value: {value!r}") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _price(value: Any, tick: float) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise TypeError(f"unsupported non-finite parity price: {value!r}")
    if tick and tick > 0:
        number = round(round(number / tick) * tick, 10)
    return _rounded(number)


def _money(value: Any) -> float:
    return float(_rounded(value) or 0.0)


def _number(value: Any) -> int | float:
    rounded = _rounded(value)
    if isinstance(rounded, float) and rounded.is_integer():
        return int(rounded)
    return rounded if rounded is not None else 0


def _rounded(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise TypeError(f"unsupported non-finite parity numeric value: {value!r}")
    rounded = round(number, 6)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
