"""Reconciliation and drift lifecycle event builders."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from libs.instrumentation.event_contract import enrich_payload, event_scope

from ._shared import event_time, plain, stable_payload_hash


_EVENT_BY_ACTION = {
    "admin_correction": "admin_correction",
    "allocation_drift": "allocation_drift",
    "allocation_freeze": "allocation_freeze",
    "allocation_unfreeze": "allocation_unfreeze",
    "drift_assignment": "drift_assignment",
    "inferred_fill": "inferred_fill",
}


def build_reconciliation_event(
    *,
    lineage: Any = None,
    lifecycle_action: str = "reconciliation_alert",
    status: str = "observed",
    phase: str = "",
    source: str = "reconciliation",
    details: Mapping[str, Any] | None = None,
    discrepancies: Sequence[Mapping[str, Any]] | None = None,
    timestamp: Any = None,
) -> dict[str, Any]:
    details_payload = plain(dict(details or {}))
    discrepancy_payload = [plain(d) for d in list(discrepancies or [])]
    event_type = _EVENT_BY_ACTION.get(lifecycle_action, "reconciliation_alert")
    drift_owner = str(
        details_payload.get("strategy_id")
        or details_payload.get("family_id")
        or details_payload.get("allocation_bucket")
        or "_UNKNOWN_"
    )

    payload = {
        "timestamp": event_time(timestamp),
        "event_id": stable_payload_hash(
            "recon_",
            {
                "lifecycle_action": lifecycle_action,
                "phase": phase,
                "status": status,
                "details": details_payload,
                "discrepancies": discrepancy_payload,
            },
        ),
        "lifecycle_action": lifecycle_action,
        "status": status,
        "phase": phase,
        "source": source,
        "drift_owner": drift_owner,
        "allocation_bucket": details_payload.get("allocation_bucket", drift_owner),
        "details": details_payload,
        "discrepancies": discrepancy_payload,
    }
    return enrich_payload(payload, lineage=lineage, event_type=event_type, scope=event_scope(event_type))
