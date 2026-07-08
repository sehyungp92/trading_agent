"""Trade completion enrichment from decision, order, fill, and runtime refs."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value


def _hash(prefix: str, value: Any, length: int = 16) -> str:
    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:length]}"


def _first(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _nested_first(payload: Mapping[str, Any], *paths: tuple[str, str]) -> str:
    for root, key in paths:
        value = payload.get(root)
        if isinstance(value, Mapping) and value.get(key) not in (None, "", [], {}):
            return str(value[key])
    return ""


def _collect_ids(payload: Mapping[str, Any], fields: tuple[str, ...]) -> list[str]:
    ids: list[str] = []
    for field in fields:
        value = payload.get(field)
        if isinstance(value, (list, tuple, set)):
            ids.extend(str(item) for item in value if item not in (None, ""))
        elif value not in (None, ""):
            ids.append(str(value))
    for root in ("entry_fill_details", "exit_fill_details", "runtime_join_refs"):
        nested = payload.get(root)
        if not isinstance(nested, Mapping):
            continue
        for field in fields:
            value = nested.get(field)
            if isinstance(value, (list, tuple, set)):
                ids.extend(str(item) for item in value if item not in (None, ""))
            elif value not in (None, ""):
                ids.append(str(value))
    return sorted(dict.fromkeys(ids))


def enrich_trade_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach stable diagnostic joins to completed trade payloads.

    The function is deliberately schema-tolerant because stock, momentum, and
    swing logs use slightly different names for the same runtime refs.
    """
    result = dict(payload)
    runtime_refs = result.get("runtime_join_refs")
    runtime_refs = dict(runtime_refs) if isinstance(runtime_refs, Mapping) else {}

    if not result.get("decision_ref"):
        result["decision_ref"] = (
            _first(result, "decision_id", "entry_signal_id", "signal_id")
            or _nested_first(result, ("event_metadata", "decision_ref"), ("execution_timestamps", "decision_ref"))
        )
    if not result.get("action_ref"):
        result["action_ref"] = (
            _first(result, "action_id")
            or _nested_first(result, ("event_metadata", "event_id"), ("execution_timestamps", "action_ref"))
        )
    if not result.get("portfolio_decision_ref"):
        result["portfolio_decision_ref"] = (
            _first(result, "portfolio_rule_trace_id", "risk_decision_ref")
            or _nested_first(runtime_refs, ("portfolio_rule", "event_id"), ("risk_decision", "event_id"))
        )
    if not result.get("intent_id"):
        result["intent_id"] = (
            _nested_first(result, ("execution_timestamps", "intent_id"), ("execution_timeline", "intent_id"))
            or _nested_first(runtime_refs, ("intent", "intent_id"))
        )

    order_ids = _collect_ids(
        result,
        (
            "order_ids",
            "order_id",
            "fill_order_id",
            "entry_order_id",
            "exit_order_id",
            "oms_order_id",
            "client_order_id",
        ),
    )
    fill_ids = _collect_ids(
        result,
        (
            "fill_ids",
            "fill_id",
            "exec_id",
            "entry_fill_id",
            "exit_fill_id",
            "broker_fill_id",
        ),
    )
    if order_ids:
        result["order_ids"] = order_ids
    if fill_ids:
        result["fill_ids"] = fill_ids

    artifact_source = copy.deepcopy(result)
    artifact_source.pop("artifact_hash", None)
    artifact_source.pop("resource_plan_hash", None)
    if not result.get("artifact_hash"):
        result["artifact_hash"] = _hash("trade_artifact_", artifact_source)
    if not result.get("resource_plan_hash"):
        result["resource_plan_hash"] = _hash(
            "resource_plan_",
            {
                "sizing_inputs": result.get("sizing_inputs"),
                "portfolio_state_at_entry": result.get("portfolio_state_at_entry"),
                "strategy_params_at_entry": result.get("strategy_params_at_entry"),
                "config_version": result.get("config_version"),
                "portfolio_config_version": result.get("portfolio_config_version"),
                "risk_config_version": result.get("risk_config_version"),
            },
        )
    return result
