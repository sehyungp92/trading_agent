"""Small helpers for carrying OMS fill refs into trade completion logs."""
from __future__ import annotations

from typing import Any, Mapping


def _first(*values: Any) -> str:
    for value in values:
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def fill_runtime_refs(
    oms_order_id: str = "",
    payload: Mapping[str, Any] | None = None,
    *,
    fill_qty: Any = None,
    is_exit: bool = False,
) -> dict[str, Any]:
    """Build facade kwargs that preserve order, fill, intent, and risk joins."""
    data = dict(payload or {})
    order_id = _first(oms_order_id, data.get("oms_order_id"), data.get("order_id"))
    fill_id = _first(
        data.get("fill_id"),
        data.get("exec_id"),
        data.get("execution_id"),
        data.get("broker_fill_id"),
    )
    qty = fill_qty if fill_qty is not None else data.get("qty")
    portfolio_ref = _first(data.get("portfolio_decision_ref"), data.get("portfolio_rule_trace_id"))
    risk_ref = _first(data.get("risk_decision_ref"))
    refs: dict[str, Any] = {
        "fill_order_id": order_id,
        "fill_qty": qty,
        "runtime_join_refs": {
            "order": {
                "order_id": order_id,
                "oms_order_id": order_id,
                "client_order_id": data.get("client_order_id", ""),
            },
            "fill": {
                "fill_id": fill_id,
                "exec_id": data.get("exec_id", fill_id),
                "qty": qty,
                "price": data.get("price"),
                "timestamp": data.get("timestamp"),
            },
            "intent": {"intent_id": data.get("intent_id", "")},
            "risk_decision": {"event_id": risk_ref},
            "portfolio_rule": {"event_id": portfolio_ref},
        },
    }
    if fill_id:
        refs["exit_fill_id" if is_exit else "fill_id"] = fill_id
    if data.get("intent_id"):
        refs["intent_id"] = data["intent_id"]
    if risk_ref:
        refs["risk_decision_ref"] = risk_ref
    if portfolio_ref:
        refs["portfolio_decision_ref"] = portfolio_ref
    if data.get("action_ref"):
        refs["action_ref"] = data["action_ref"]
    return refs
