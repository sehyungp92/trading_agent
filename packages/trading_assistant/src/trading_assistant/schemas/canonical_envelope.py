"""Shared helpers for canonical bot relay envelopes.

The bot repos can put lineage and join keys on the relay envelope while the
assistant queue persists only the payload text. These helpers keep the payload
as the durable carrier without changing the queue schema.
"""

from __future__ import annotations

from typing import Any


CANONICAL_ENVELOPE_PAYLOAD_FIELDS = (
    "event_id",
    "schema_version",
    "scope",
    "bot_id",
    "event_type",
    "priority",
    "payload_key",
    "payload_hash",
    "family_id",
    "portfolio_id",
    "account_alias",
    "strategy_id",
    "assistant_strategy_id",
    "symbol",
    "exchange",
    "asset_class",
    "currency",
    "timezone",
    "exchange_timestamp",
    "local_timestamp",
    "data_source",
    "data_source_id",
    "logical_event_id",
    "revision",
    "supersedes_event_id",
    "bar_id",
    "trace_id",
    "event_ref",
    "decision_id",
    "entry_decision_id",
    "exit_decision_id",
    "decision_ref",
    "action_ref",
    "provisional_order_ref",
    "portfolio_decision_ref",
    "portfolio_rule_event_id",
    "risk_decision_id",
    "intent_id",
    "idempotency_key",
    "order_id",
    "order_ids",
    "entry_order_id",
    "exit_order_id",
    "entry_order_event_refs",
    "exit_order_event_refs",
    "client_order_id",
    "client_order_ids",
    "broker_order_id",
    "original_order_id",
    "oms_order_id",
    "kis_order_id",
    "kis_order_date",
    "kis_exec_id",
    "fill_id",
    "fill_ids",
    "trade_id",
    "strategy_version",
    "config_version",
    "portfolio_config_version",
    "risk_config_version",
    "allocation_version",
    "strategy_registry_version",
    "deployment_id",
    "parameter_set_id",
    "weekly_signal_id",
    "weekly_signal_ids",
    "source_weekly_signal_ids",
    "monthly_search_brief_id",
    "proposal_id",
    "proposal_ids",
    "suggestion_id",
    "suggestion_ids",
    "hypothesis_id",
    "hypothesis_ids",
    "strategy_change_record_id",
    "strategy_change_record_ids",
    "monthly_outcome_id",
    "assistant_lineage",
    "experiment_id",
    "variant_id",
    "signal_generation_version",
    "code_sha",
    "artifact_hash",
    "source_artifact_hash",
    "source_fingerprint",
    "candidate_hash",
    "resource_plan_hash",
    "kis_resource_plan_hash",
    "portfolio_policy_hash",
    "state_hash",
    "plan_hash",
    "snapshot_id",
    "runtime_join",
    "join_completeness",
    "lineage",
    "source",
    "source_stream",
)

PASSIVE_ENVELOPE_FIELDS = frozenset({
    "event_id",
    "bot_id",
    "event_type",
    "exchange_timestamp",
    "received_at",
})


def has_canonical_envelope_context(event: dict[str, Any]) -> bool:
    """Return true when an envelope has more than queue bookkeeping fields."""
    return any(
        event.get(key) not in (None, "")
        for key in CANONICAL_ENVELOPE_PAYLOAD_FIELDS
        if key not in PASSIVE_ENVELOPE_FIELDS
    )


def merge_envelope_fields_into_payload(
    event: dict[str, Any],
    payload: dict[str, Any],
    *,
    require_context: bool = True,
) -> dict[str, Any]:
    """Copy canonical envelope fields into payload when absent."""
    if require_context and not has_canonical_envelope_context(event):
        return dict(payload)

    merged = dict(payload)
    for key in CANONICAL_ENVELOPE_PAYLOAD_FIELDS:
        value = event.get(key)
        if value not in (None, "") and merged.get(key) in (None, ""):
            merged[key] = value
    return merged
