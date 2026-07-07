"""Common event contract helpers for payloads, sidecar envelopes, and startup events."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .lineage import LineageContext, lineage_to_payload, redact_config, stable_hash


COMMON_LINEAGE_FIELDS = (
    "scope",
    "strategy_id",
    "family_id",
    "portfolio_id",
    "account_alias",
    "strategy_version",
    "config_version",
    "portfolio_config_version",
    "risk_config_version",
    "allocation_version",
    "strategy_registry_version",
    "deployment_id",
    "parameter_set_id",
    "experiment_id",
    "variant_id",
    "code_sha",
    "trace_id",
    "schema_version",
)
ASSISTANT_TRACE_FIELDS = (
    "weekly_signal_ids",
    "source_weekly_signal_ids",
    "monthly_search_brief_id",
    "suggestion_id",
    "suggestion_ids",
    "proposal_id",
    "proposal_ids",
    "source_proposal_ids",
    "candidate_id",
    "candidate_ids",
    "hypothesis_id",
    "hypothesis_ids",
    "experiment_id",
    "strategy_change_record_id",
    "strategy_change_record_ids",
    "monthly_outcome_id",
)

REQUIRED_SECTION_6_2_FIELDS = (
    "schema_version",
    "event_type",
    "scope",
    "bot_id",
    "strategy_id",
    "family_id",
    "portfolio_id",
    "account_alias",
    "strategy_version",
    "config_version",
    "portfolio_config_version",
    "risk_config_version",
    "allocation_version",
    "strategy_registry_version",
    "deployment_id",
    "parameter_set_id",
    "code_sha",
    "trace_id",
)

_REQUIRED_COMMON_FIELDS = REQUIRED_SECTION_6_2_FIELDS

_SCOPE_BY_EVENT = {
    "admin_correction": "oms",
    "allocation_drift": "portfolio",
    "allocation_freeze": "portfolio",
    "allocation_unfreeze": "portfolio",
    "allocation_snapshot": "portfolio",
    "coordinator_action": "family",
    "correlation_snapshot": "portfolio",
    "decision_event": "strategy",
    "deployment": "strategy",
    "drift_assignment": "portfolio",
    "family_daily_snapshot": "family",
    "inferred_fill": "oms",
    "portfolio_rule_check": "portfolio",
    "portfolio_snapshot": "portfolio",
    "position_snapshot": "portfolio",
    "reconciliation_alert": "oms",
    "risk_decision": "oms",
    "risk_denial": "oms",
    "risk_halt": "oms",
    "sector_exposure": "portfolio",
}

_SCHEMA_BY_EVENT = {
    "admin_correction": "admin_correction_v1",
    "allocation_drift": "allocation_drift_v1",
    "allocation_freeze": "allocation_freeze_v1",
    "allocation_unfreeze": "allocation_unfreeze_v1",
    "allocation_snapshot": "allocation_snapshot_v1",
    "config_snapshot": "config_snapshot_v1",
    "coordinator_action": "coordinator_action_v1",
    "daily_snapshot": "daily_snapshot_v2",
    "decision_event": "decision_event_v1",
    "deployment": "deployment_v1",
    "drift_assignment": "drift_assignment_v1",
    "error": "error_event_v2",
    "filter_decision": "filter_decision_v2",
    "heartbeat": "heartbeat_v2",
    "indicator_snapshot": "indicator_snapshot_v2",
    "inferred_fill": "inferred_fill_v1",
    "market_snapshot": "market_snapshot_v2",
    "missed_opportunity": "missed_opportunity_v2",
    "order": "order_event_v2",
    "orderbook_context": "orderbook_context_v2",
    "portfolio_rule_check": "portfolio_rule_check_v2",
    "portfolio_snapshot": "portfolio_snapshot_v1",
    "position_snapshot": "position_snapshot_v1",
    "post_exit": "post_exit_v1",
    "process_quality": "process_quality_v1",
    "parameter_change": "parameter_change_v1",
    "reconciliation_alert": "reconciliation_alert_v1",
    "risk_decision": "risk_decision_v1",
    "risk_denial": "risk_denial_v2",
    "risk_halt": "risk_halt_v1",
    "stop_adjustment": "stop_adjustment_v1",
    "trade": "trade_event_v2",
    "trade_entry": "trade_event_v2",
}


def event_scope(event_type: str, default: str = "strategy") -> str:
    return _SCOPE_BY_EVENT.get(event_type, default)


def event_schema_version(event_type: str) -> str:
    return _SCHEMA_BY_EVENT.get(event_type, f"{event_type}_v1")


def _non_empty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _normalise_parameter_set_id(value: Any) -> str:
    if not _non_empty(value):
        return ""
    text = str(value)
    return text if text.startswith("param_") else f"param_{text}"


def _allows_empty_required_field(payload: Mapping[str, Any], field: str) -> bool:
    if field not in {"strategy_id", "strategy_version"}:
        return False
    scope = str(payload.get("scope") or "")
    event_type = str(payload.get("event_type") or "")
    halt_scope = str(payload.get("halt_scope") or "")
    return (
        scope in {"portfolio", "family"}
        or (event_type == "risk_halt" and halt_scope == "portfolio")
    )


def _has_required_field(payload: Mapping[str, Any], field: str) -> bool:
    if field not in payload:
        return False
    if _allows_empty_required_field(payload, field):
        return payload.get(field) is not None
    return _non_empty(payload.get(field))


def _nested_lineage(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("lineage")
    return dict(value) if isinstance(value, Mapping) else {}


def _event_id(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("event_metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("event_id"):
        return str(metadata["event_id"])
    for key in ("event_id", "trade_id", "order_id", "snapshot_id", "date"):
        if payload.get(key):
            return str(payload[key])
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_lineage(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract lineage-like fields from a raw payload for sidecar envelopes."""
    extracted = _nested_lineage(payload)
    metadata = payload.get("event_metadata", {})
    if isinstance(metadata, Mapping):
        for field in (
            "bot_id",
            "strategy_id",
            "family_id",
            "portfolio_id",
            "trace_id",
            "schema_version",
            "payload_key",
        ):
            value = metadata.get(field)
            if _non_empty(value):
                extracted.setdefault(field, value)
    for field in ("bot_id", *COMMON_LINEAGE_FIELDS, "param_set_id"):
        if field in payload and _non_empty(payload.get(field)):
            extracted[field] = payload[field]
    if not extracted.get("parameter_set_id") and payload.get("param_set_id"):
        param = str(payload["param_set_id"])
        extracted["parameter_set_id"] = param if param.startswith("param_") else f"param_{param}"
    return extracted


def merge_lineage(
    payload: Mapping[str, Any],
    lineage: LineageContext | Mapping[str, Any] | None,
    scope: str,
    schema_version: str,
    event_type: str,
) -> dict[str, Any]:
    """Return a payload with top-level and nested lineage fields.

    Existing event aliases are preserved. Missing fields are visible through
    ``lineage_gap``/``lineage_missing_fields`` but never raise, keeping
    instrumentation fail-open.
    """
    result = dict(payload)
    merged_lineage = extract_lineage(result)
    merged_lineage.update({k: v for k, v in lineage_to_payload(lineage).items() if _non_empty(v)})

    metadata = result.get("event_metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("bot_id"):
        merged_lineage.setdefault("bot_id", metadata.get("bot_id"))
    if result.get("bot_id"):
        merged_lineage.setdefault("bot_id", result.get("bot_id"))

    result["event_type"] = event_type
    result["scope"] = scope
    result["schema_version"] = schema_version

    for field in COMMON_LINEAGE_FIELDS:
        if field in {"scope", "schema_version"}:
            merged_value = result[field]
        else:
            merged_value = merged_lineage.get(field)
        if field == "strategy_id" and _allows_empty_required_field(result, field) and field in result:
            continue
        if _non_empty(merged_value) and not _non_empty(result.get(field)):
            result[field] = merged_value

    for field in ASSISTANT_TRACE_FIELDS:
        merged_value = merged_lineage.get(field)
        if _non_empty(merged_value) and not _non_empty(result.get(field)):
            result[field] = merged_value

    if _non_empty(merged_lineage.get("bot_id")) and not _non_empty(result.get("bot_id")):
        result["bot_id"] = merged_lineage["bot_id"]
    if not _non_empty(result.get("trace_id")):
        result["trace_id"] = stable_hash("trace_", {"event_id": _event_id(result), "event_type": event_type})
    canonical_param = _normalise_parameter_set_id(result.get("parameter_set_id"))
    if not canonical_param:
        canonical_param = _normalise_parameter_set_id(result.get("param_set_id"))
    if canonical_param:
        result["parameter_set_id"] = canonical_param
        result["param_set_id"] = canonical_param

    for field in REQUIRED_SECTION_6_2_FIELDS:
        if _allows_empty_required_field(result, field) and result.get(field) is None:
            result[field] = ""

    lineage_block = {field: result.get(field, "") for field in REQUIRED_SECTION_6_2_FIELDS}
    if _non_empty(merged_lineage.get("bot_id")) and not _non_empty(lineage_block.get("bot_id")):
        lineage_block["bot_id"] = merged_lineage["bot_id"]
    for field in ASSISTANT_TRACE_FIELDS:
        if _non_empty(result.get(field)):
            lineage_block[field] = result[field]
    result["lineage"] = lineage_block

    gaps = [field for field in _REQUIRED_COMMON_FIELDS if not _has_required_field(result, field)]
    if gaps:
        missing = sorted(set(gaps))
        result["lineage_gap"] = True
        result["lineage_missing_fields"] = missing
        result["lineage_gaps"] = missing
    else:
        result.pop("lineage_gap", None)
        result.pop("lineage_missing_fields", None)
        result.pop("lineage_gaps", None)

    return result


def enrich_payload(
    payload: Mapping[str, Any],
    *,
    lineage: LineageContext | Mapping[str, Any] | None,
    event_type: str,
    scope: str | None = None,
    schema_version: str | None = None,
) -> dict[str, Any]:
    return merge_lineage(
        payload,
        lineage,
        scope or event_scope(event_type),
        schema_version or event_schema_version(event_type),
        event_type,
    )


def enrich_envelope(event: Mapping[str, Any], raw_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Duplicate payload lineage onto a sidecar event envelope."""
    result = dict(event)
    lineage = extract_lineage(raw_payload)
    for field in COMMON_LINEAGE_FIELDS:
        if field in lineage and _non_empty(lineage[field]):
            result[field] = lineage[field]
    if lineage.get("payload_key"):
        result.setdefault("payload_key", lineage["payload_key"])
    if lineage.get("bot_id"):
        result.setdefault("bot_id", lineage["bot_id"])
    return result


def append_jsonl_event(data_dir: str | Path, subdir: str, prefix: str, payload: Mapping[str, Any]) -> Path:
    out_dir = Path(data_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = out_dir / f"{prefix}_{today}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
    return path


def write_deployment_event(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    status: str = "startup",
    details: Mapping[str, Any] | None = None,
) -> Path:
    payload = enrich_payload(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "details": dict(details or {}),
        },
        lineage=lineage,
        event_type="deployment",
        scope=event_scope("deployment"),
    )
    return append_jsonl_event(data_dir, "deployments", "deployments", payload)


def write_config_snapshot(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    effective_config: Mapping[str, Any] | None = None,
    source: str = "startup",
) -> Path:
    payload = enrich_payload(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "effective_config": redact_config(dict(effective_config or {})),
        },
        lineage=lineage,
        event_type="config_snapshot",
        scope=event_scope("config_snapshot"),
    )
    return append_jsonl_event(data_dir, "config_snapshots", "config_snapshots", payload)


def write_allocation_snapshot(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    allocation_state: Mapping[str, Any] | None = None,
    positions: list[Mapping[str, Any]] | None = None,
    source: str = "startup",
) -> Path:
    try:
        from libs.oms.instrumentation.allocation_snapshot import build_allocation_snapshot

        state = dict(allocation_state or {})
        payload = build_allocation_snapshot(
            lineage=lineage,
            positions=positions or state.get("positions") or [],
            targets=state.get("targets", state),
            raw_nav=state.get("raw_nav"),
            allocated_nav=state.get("allocated_nav"),
            source=source,
            metadata=state,
        )
    except Exception:
        payload = enrich_payload(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "allocation_state": redact_config(dict(allocation_state or {})),
            },
            lineage=lineage,
            event_type="allocation_snapshot",
            scope="portfolio",
        )
    return append_jsonl_event(data_dir, "allocations", "allocations", payload)


def write_portfolio_snapshot(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    portfolio_state: Mapping[str, Any] | None = None,
    positions: list[Mapping[str, Any]] | None = None,
    source: str = "startup",
) -> Path:
    try:
        from libs.oms.instrumentation.portfolio_snapshot import build_portfolio_snapshot

        state = dict(portfolio_state or {})
        payload = build_portfolio_snapshot(
            lineage=lineage,
            positions=positions or state.get("positions") or [],
            portfolio_risk=state,
            account_state=state,
            source=source,
            reconciliation_status=f"{source}_snapshot",
        )
    except Exception:
        payload = enrich_payload(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "portfolio_state": redact_config(dict(portfolio_state or {})),
            },
            lineage=lineage,
            event_type="portfolio_snapshot",
            scope="portfolio",
        )
    return append_jsonl_event(data_dir, "portfolio", "portfolio_snapshots", payload)


def write_position_snapshot(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    positions: list[Mapping[str, Any]] | None = None,
    source: str = "startup",
) -> Path:
    try:
        from libs.oms.instrumentation.position_snapshot import build_position_snapshot

        position_list = [dict(item) for item in list(positions or []) if isinstance(item, Mapping)]
        if not position_list:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return Path(data_dir) / "positions" / f"positions_{today}.jsonl"
        last_path: Path | None = None
        for position in position_list:
            payload = build_position_snapshot(position, lineage=lineage, source=source)
            last_path = append_jsonl_event(data_dir, "positions", "positions", payload)
        return last_path or Path(data_dir) / "positions" / f"positions_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    except Exception:
        payload = enrich_payload(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "positions": redact_config(list(positions or [])),
            },
            lineage=lineage,
            event_type="position_snapshot",
            scope="portfolio",
        )
        return append_jsonl_event(data_dir, "positions", "positions", payload)


def write_error_event(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    component: str,
    method: str,
    message: str,
    error_type: str = "",
    context: Mapping[str, Any] | None = None,
    exc: BaseException | None = None,
    severity: str = "medium",
    category: str = "instrumentation",
    source_file: str = "",
    source_line: int = 0,
) -> Path:
    ts = datetime.now(timezone.utc)
    stack_trace = ""
    if exc is not None:
        error_type = error_type or type(exc).__name__
        stack_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if not source_file or not source_line:
            tb = exc.__traceback__
            last_tb = None
            while tb is not None:
                last_tb = tb
                tb = tb.tb_next
            if last_tb is not None:
                source_file = source_file or str(last_tb.tb_frame.f_code.co_filename)
                source_line = source_line or int(last_tb.tb_lineno)
    payload = enrich_payload(
        {
            "timestamp": ts.isoformat(),
            "component": component,
            "method": method,
            "error_type": error_type or "InstrumentationError",
            "message": str(message),
            "stack_trace": stack_trace,
            "source_file": source_file,
            "source_line": source_line,
            "severity": str(severity or "medium").lower(),
            "category": str(category or "instrumentation").lower(),
            "context": redact_config(dict(context or {})),
        },
        lineage=lineage,
        event_type="error",
        scope="strategy",
    )
    return append_jsonl_event(data_dir, "errors", "instrumentation_errors", payload)


def write_risk_halt_event(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    reason: str,
    strategy_id: str = "",
    halt_scope: str = "",
    source: str = "",
    timestamp: Any = None,
    details: Mapping[str, Any] | None = None,
) -> Path:
    ts = timestamp or datetime.now(timezone.utc)
    ts_value = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    detail_payload = dict(details or {})
    resolved_scope = str(
        halt_scope
        or detail_payload.get("halt_scope")
        or detail_payload.get("scope")
        or ("strategy" if strategy_id else "portfolio")
    )
    resolved_source = str(source or detail_payload.get("source") or "oms")
    payload = enrich_payload(
        {
            "timestamp": ts_value,
            "strategy_id": strategy_id,
            "reason": str(reason or ""),
            "halt_scope": resolved_scope,
            "source": resolved_source,
            "details": detail_payload,
        },
        lineage=lineage,
        event_type="risk_halt",
        scope=event_scope("risk_halt"),
    )
    if resolved_scope == "portfolio" and not strategy_id:
        payload["strategy_id"] = ""
        lineage_block = dict(payload.get("lineage") or {})
        lineage_block["strategy_id"] = ""
        payload["lineage"] = lineage_block
    return append_jsonl_event(data_dir, "risk_halts", "risk_halts", payload)


def write_startup_events(
    data_dir: str | Path,
    lineage: LineageContext | Mapping[str, Any] | None,
    *,
    effective_config: Mapping[str, Any] | None = None,
    allocation_state: Mapping[str, Any] | None = None,
    portfolio_state: Mapping[str, Any] | None = None,
    positions: list[Mapping[str, Any]] | None = None,
    bridge_id: str = "",
    runtime_entrypoint: str = "",
    runtime_started_at_utc: str = "",
    portfolio_rules_config: Any = None,
) -> None:
    effective = _startup_effective_config(
        lineage,
        effective_config or {},
        portfolio_rules_config=portfolio_rules_config,
    )
    write_deployment_event(data_dir, lineage, status="startup")
    write_config_snapshot(data_dir, lineage, effective_config=effective, source="startup")
    write_allocation_snapshot(
        data_dir,
        lineage,
        allocation_state=allocation_state or {},
        positions=positions or [],
        source="startup",
    )
    write_portfolio_snapshot(
        data_dir,
        lineage,
        portfolio_state=portfolio_state or {},
        positions=positions or [],
        source="startup",
    )
    write_position_snapshot(data_dir, lineage, positions=positions or [], source="startup")
    try:
        from .deployment_metadata import write_deployment_metadata

        write_deployment_metadata(
            data_dir,
            lineage,
            bridge_id=bridge_id,
            effective_config=effective,
            runtime_entrypoint=runtime_entrypoint,
            runtime_started_at_utc=runtime_started_at_utc,
        )
    except Exception:
        pass


def _startup_effective_config(
    lineage: LineageContext | Mapping[str, Any] | None,
    runtime_config: Mapping[str, Any],
    *,
    portfolio_rules_config: Any = None,
) -> dict[str, Any]:
    result = dict(runtime_config)
    lineage_payload = lineage_to_payload(lineage)
    strategy_id = str(lineage_payload.get("strategy_id") or result.get("strategy_id") or "")
    family_id = str(lineage_payload.get("family_id") or result.get("family_id") or "")
    try:
        from .config_snapshot import (
            build_effective_portfolio_config,
            build_effective_risk_config,
            build_effective_strategy_config,
        )

        if strategy_id:
            result.setdefault(
                "effective_strategy_config",
                build_effective_strategy_config(strategy_id, runtime_config=runtime_config),
            )
        result.setdefault(
            "effective_portfolio_config",
            build_effective_portfolio_config(
                family_id=family_id,
                portfolio_rules_config=portfolio_rules_config,
            ),
        )
        result.setdefault(
            "effective_risk_config",
            build_effective_risk_config(
                family_id,
                portfolio_rules_config=portfolio_rules_config,
            ),
        )
    except Exception:
        pass
    return result


def write_decision_event(
    data_dir: str | Path,
    decision: Mapping[str, Any] | Any,
    *,
    lineage: LineageContext | Mapping[str, Any] | None = None,
) -> Path:
    if dataclasses.is_dataclass(decision) and not isinstance(decision, type):
        payload = dataclasses.asdict(decision)
    elif hasattr(decision, "to_dict"):
        payload = decision.to_dict()
    else:
        payload = dict(decision)
    payload = _json_ready(payload)
    payload = enrich_payload(
        payload,
        lineage=lineage,
        event_type="decision_event",
        scope=event_scope("decision_event"),
    )
    return append_jsonl_event(data_dir, "decisions", "decisions", payload)


def write_strategy_decision_event(
    data_dir: str | Path,
    *,
    code: str,
    strategy_id: str,
    details: Mapping[str, Any] | None = None,
    exchange_timestamp: Any = None,
    lineage: LineageContext | Mapping[str, Any] | None = None,
) -> Path:
    details_payload = dict(details or {})
    timestamp = exchange_timestamp or datetime.now(timezone.utc)
    if hasattr(timestamp, "isoformat"):
        ts_value = timestamp.isoformat()
    else:
        ts_value = str(timestamp)
    payload = {
        "code": code,
        "ts": ts_value,
        "strategy_id": strategy_id,
        "symbol": details_payload.get("pair") or details_payload.get("symbol") or "",
        "timeframe": details_payload.get("timeframe") or "",
        "details": details_payload,
        "state_ref": details_payload.get("state_ref") or "",
        "emitted_actions": list(details_payload.get("emitted_actions") or []),
        "bar_id": details_payload.get("bar_id") or "",
        "decision_kind": details_payload.get("decision_kind") or str(code).split(":", 1)[0].lower(),
    }
    return write_decision_event(data_dir, payload, lineage=lineage)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_ready(item) for item in value)
    return value
