"""Portfolio rule event contract helpers."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import compute_risk_config_version, redact_config, stable_hash


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    return value


def _public_plain(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, Mapping):
        return {
            str(k): _public_plain(v)
            for k, v in value.items()
            if not str(k).startswith("_")
        }
    if isinstance(value, list):
        return [_public_plain(item) for item in value]
    return value


def build_portfolio_rule_event(
    event: Mapping[str, Any],
    *,
    portfolio_rules_config: Any = None,
    request_context: Mapping[str, Any] | None = None,
    lineage: Any = None,
) -> dict[str, Any]:
    """Build a lossless portfolio_rule_check event while preserving old aliases."""
    raw = dict(event)
    context = dict(request_context or {})
    rule_name = str(raw.get("rule") or raw.get("rule_name") or "unknown")
    approved = bool(raw.get("approved", True))
    raw_rule_mult = raw.get("size_multiplier", 1.0)
    rule_mult = 1.0 if raw_rule_mult is None else float(raw_rule_mult)
    result = "block" if not approved else ("scale" if rule_mult != 1.0 else "pass")

    requested = dict(context.get("requested_sizing") or {})
    raw_current_mult = context.get("current_size_multiplier", 1.0)
    current_mult = 1.0 if raw_current_mult is None else float(raw_current_mult)
    cumulative_mult = current_mult * rule_mult
    requested_qty = int(requested.get("qty") or 0)
    requested_risk_r = float(requested.get("risk_R") or 0.0)
    requested_risk_dollars = float(requested.get("risk_dollars") or 0.0)
    approved_qty = 0 if not approved else _adjusted_qty(requested_qty, cumulative_mult)
    approved_risk_r = (
        requested_risk_r * (approved_qty / requested_qty)
        if requested_qty > 0 else requested_risk_r * cumulative_mult
    )
    approved_risk_dollars = (
        requested_risk_dollars * (approved_qty / requested_qty)
        if requested_qty > 0 else requested_risk_dollars * cumulative_mult
    )

    approved_sizing = {
        "size_multiplier": cumulative_mult if approved else 0.0,
        "rule_size_multiplier": rule_mult if approved else 0.0,
        "qty": approved_qty,
        "risk_R": approved_risk_r if approved else 0.0,
        "risk_dollars": approved_risk_dollars if approved else 0.0,
    }

    details = dict(raw.get("details") or {})
    if raw.get("denial_reason"):
        details["reason"] = raw["denial_reason"]
    if raw.get("symbol"):
        details["blocked_symbol"] = raw["symbol"]
    for key in (
        "strategy_id",
        "direction",
        "symbol",
        "size_multiplier",
        "drawdown_pct",
    ):
        if key in raw:
            details[key] = raw[key]

    reason = str(raw.get("denial_reason") or details.get("reason") or "")
    check_sequence = int(raw.get("check_sequence") or context.get("check_sequence") or 0)
    trace_id = str(raw.get("trace_id") or context.get("trace_id") or "")
    rule_trace_id = str(raw.get("rule_trace_id") or "") or stable_hash(
        "rule_trace_",
        {
            "trace_id": trace_id,
            "rule_name": rule_name,
            "strategy_id": raw.get("strategy_id") or context.get("strategy_id", ""),
            "symbol": raw.get("symbol") or context.get("symbol", ""),
            "direction": raw.get("direction") or context.get("direction", ""),
            "sequence": check_sequence,
        },
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": stable_hash(
            "prule_",
            {
                "rule_trace_id": rule_trace_id,
                "rule_name": rule_name,
                "result": result,
                "check_sequence": check_sequence,
            },
        ),
        "rule_name": rule_name,
        "result": result,
        "action": raw.get("action") or details.get("action") or result,
        "reason": reason,
        "trace_id": trace_id,
        "rule_trace_id": rule_trace_id,
        "check_sequence": check_sequence,
        "signal_id": raw.get("signal_id") or context.get("signal_id", ""),
        "bar_id": raw.get("bar_id") or context.get("bar_id", ""),
        "exchange_timestamp": raw.get("exchange_timestamp") or context.get("exchange_timestamp", ""),
        "details": details,
        "strategy_id": raw.get("strategy_id") or context.get("strategy_id", ""),
        "direction": raw.get("direction") or context.get("direction", ""),
        "symbol": raw.get("symbol") or context.get("symbol", ""),
        "requested_qty": requested_qty,
        "approved_qty": approved_qty,
        "requested_risk_R": requested_risk_r,
        "approved_risk_R": approved_sizing["risk_R"],
        "requested_risk_dollars": requested_risk_dollars,
        "approved_risk_dollars": approved_sizing["risk_dollars"],
        "size_multiplier_before": current_mult,
        "size_multiplier_after": approved_sizing["size_multiplier"],
        "requested_sizing": requested,
        "approved_sizing": approved_sizing,
        "threshold": redact_config(_public_plain(portfolio_rules_config or {})),
        "thresholds": redact_config(_public_plain(portfolio_rules_config or {})),
        "state_before": redact_config(context.get("state_before", {})),
        "state_after": redact_config(
            {
                "approved": approved,
                "result": result,
                "current_size_multiplier": cumulative_mult if approved else 0.0,
            }
        ),
        "raw_rule_event": redact_config(raw),
        "portfolio_rule_config_version": compute_risk_config_version({}, portfolio_rules_config, {}),
    }
    return enrich_payload(
        payload,
        lineage=lineage,
        event_type="portfolio_rule_check",
        scope="portfolio",
    )


def _adjusted_qty(qty: int, size_multiplier: float) -> int:
    if qty <= 0 or size_multiplier <= 0:
        return 0
    return max(1, int(qty * size_multiplier))
