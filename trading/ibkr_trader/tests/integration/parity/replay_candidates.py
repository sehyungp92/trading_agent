from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tests.integration.parity.broker_matching import (
    broker_event_key as _broker_event_key,
    candidate_key as _candidate_key,
    order_matches as _order_matches,
)
from tests.integration.parity.source_inputs import action_order_row, family_resolver, parse_time


@dataclass
class ReplayDecisionTimeline:
    fixture: Mapping[str, Any]
    order_intents: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    strategy_state: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    _applied: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._index_initial_repository_orders()

    def record_actions(self, strategy_id: str, actions: list[Any]) -> None:
        family = family_resolver(self.fixture)(strategy_id)
        for action in actions:
            row = action_order_row(action, strategy_id, family)
            if not row["symbol"] or not row["qty"]:
                continue
            self.order_intents.append(row)
            self.orders.append(
                {
                    "strategy_id": row["strategy_id"],
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "qty": row["qty"],
                    "order_type": row["order_type"],
                    "limit_price": row["limit_price"],
                    "stop_price": row["stop_price"],
                    "role": row["role"],
                    "status": "ROUTED",
                    "filled_qty": 0.0,
                    "remaining_qty": row["qty"],
                    "avg_fill_price": 0.0,
                    "client_tag": row["client_order_id"],
                    "source": "generated",
                    "_action": action,
                }
            )
            self.timeline.append({"type": "action", "strategy_id": strategy_id, "action": action})

    def apply_broker_script(self) -> None:
        for event in self.fixture.get("broker_event_script", []):
            key = _broker_event_key(event)
            if key in self._applied:
                continue
            if str(event.get("event", "fill")).lower() != "fill":
                continue
            order = self._match_order(event.get("order_match", {}))
            if order is None:
                raise AssertionError(f"broker event could not match submitted replay order: {event.get('order_match', {})}")
            self.note_broker_event(order, event)
            self._applied.add(key)

    def note_broker_event(self, order: dict[str, Any], event: Mapping[str, Any]) -> None:
        if str(event.get("event", "fill")).lower() == "fill":
            fill_qty = float(event.get("qty", order["qty"]))
            order["status"] = "FILLED"
            order["filled_qty"] = fill_qty
            order["remaining_qty"] = max(0.0, float(order["qty"]) - fill_qty)
            order["avg_fill_price"] = float(event.get("price", order.get("limit_price") or order.get("stop_price") or 0.0))
        self.timeline.append({"type": "broker_event", "event": event})

    def _index_initial_repository_orders(self) -> None:
        initial = self.fixture.get("initial_repository_state", {}) or {}
        for item in initial.get("orders", []) or []:
            self.orders.append(
                {
                    "strategy_id": str(item.get("strategy_id", "")),
                    "symbol": str(item.get("symbol") or item.get("instrument_symbol") or ""),
                    "side": str(item.get("side", "")).upper(),
                    "qty": item.get("qty", 0),
                    "order_type": str(item.get("order_type", "")).upper(),
                    "role": str(item.get("role", item.get("order_role", ""))).upper(),
                    "status": str(item.get("status", "CREATED")).upper(),
                    "filled_qty": float(item.get("filled_qty", 0.0) or 0.0),
                    "remaining_qty": float(item.get("remaining_qty", item.get("qty", 0.0)) or 0.0),
                    "avg_fill_price": float(item.get("avg_fill_price", 0.0) or 0.0),
                    "client_tag": str(item.get("client_order_id") or item.get("client_tag") or ""),
                    "source": "initial_repository",
                }
            )

    def _match_order(self, match: Mapping[str, Any]) -> dict[str, Any] | None:
        sequence = int(match.get("sequence", 1))
        matches = [order for order in self.orders if _order_matches(order, match)]
        if len(matches) < sequence:
            return None
        if len(matches) > sequence and "sequence" not in match:
            raise AssertionError(f"broker event matched multiple replay orders: {match}")
        return matches[sequence - 1]


def _entry_candidate_specs(fixture: Mapping[str, Any], out: ReplayDecisionTimeline) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    sequence_by_match: dict[tuple[str, str, str, str], int] = {}
    for order in _generated_entry_orders(out):
        match_key = (
            str(order.get("strategy_id", "")),
            str(order.get("symbol", "")),
            str(order.get("role", "ENTRY")).upper(),
            str(order.get("side", "")).upper(),
        )
        sequence_by_match[match_key] = sequence_by_match.get(match_key, 0) + 1
        order_match = {
            "strategy_id": match_key[0],
            "symbol": match_key[1],
            "role": match_key[2],
            "side": match_key[3],
            "sequence": sequence_by_match[match_key],
        }
        event = _broker_fill_for_order(fixture, out, order)
        entry_price = _entry_price_for_candidate(fixture, order, event)
        stop_price = _protective_stop_price(out, order, entry_price)
        entry_time = parse_time(event.get("timestamp")) if event else parse_time(fixture["clock_start"])
        order_qty = int(float(order.get("qty", 1) or 1))
        fill_qty = int(float((event or {}).get("qty", order_qty) or order_qty))
        candidates.append(
            {
                "order": order,
                "event": event,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "qty": max(order_qty, 1),
                "fill_qty": max(fill_qty, 1),
                "commission": float((event or {}).get("commission", 0.0) or 0.0),
                "action": order.get("_action"),
                "order_match": order_match,
                "candidate_key": _candidate_key(order_match),
            }
        )
    return candidates


def _generated_entry_orders(out: ReplayDecisionTimeline) -> list[dict[str, Any]]:
    return [
        order
        for order in out.orders
        if order.get("source") == "generated"
        and str(order.get("role", "")).upper() == "ENTRY"
    ]


def _broker_fill_for_order(
    fixture: Mapping[str, Any],
    out: ReplayDecisionTimeline,
    order: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for event in fixture.get("broker_event_script", []) or []:
        if str(event.get("event", "fill")).lower() != "fill":
            continue
        matched = out._match_order(event.get("order_match", {}))
        if matched is order:
            return event
    return None


def _entry_price_for_candidate(
    fixture: Mapping[str, Any],
    order: Mapping[str, Any],
    event: Mapping[str, Any] | None,
) -> float:
    if event is not None and event.get("price") not in (None, ""):
        return float(event["price"])
    for key in ("limit_price", "stop_price", "price"):
        value = order.get(key)
        if value not in (None, ""):
            return float(value)
    return _last_source_close(fixture, str(order.get("symbol", "")), 1.0)


def _protective_stop_price(out: ReplayDecisionTimeline, order: Mapping[str, Any], entry_price: float) -> float:
    side = "SELL" if str(order.get("side", "")).upper() == "BUY" else "BUY"
    for candidate in out.orders:
        if (
            str(candidate.get("strategy_id")) == str(order.get("strategy_id"))
            and str(candidate.get("symbol")) == str(order.get("symbol"))
            and str(candidate.get("role", "")).upper() == "STOP"
            and str(candidate.get("side", "")).upper() == side
        ):
            price = candidate.get("stop_price") or candidate.get("limit_price")
            if price not in (None, ""):
                return float(price)
    return entry_price - 1.0 if side == "SELL" else entry_price + 1.0


def _last_source_close(fixture: Mapping[str, Any], symbol: str, default: float) -> float:
    closes = [
        float(row["close"])
        for row in fixture.get("bars", [])
        if str(row.get("symbol", "")).upper() == symbol.upper() and row.get("close") is not None
    ]
    return closes[-1] if closes else float(default)








entry_candidate_specs = _entry_candidate_specs
generated_entry_orders = _generated_entry_orders
order_matches = _order_matches
candidate_key = _candidate_key
broker_event_key = _broker_event_key
