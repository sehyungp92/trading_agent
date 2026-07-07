"""Lifecycle writers for fills, reconciliation, and daily closeout."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from libs.instrumentation.event_contract import append_jsonl_event

from .allocation_snapshot import build_allocation_snapshot
from .family_snapshot import build_family_daily_snapshot
from .portfolio_snapshot import build_portfolio_snapshot
from .position_snapshot import build_position_snapshot
from .reconciliation_event import build_reconciliation_event


def _positions_from_payload(payload: Mapping[str, Any]) -> list[dict]:
    positions = [dict(item) for item in list(payload.get("positions") or []) if isinstance(item, Mapping)]
    current = payload.get("position")
    if isinstance(current, Mapping) and not positions:
        positions = [dict(current)]
    return positions


def write_fill_lifecycle_snapshots(
    data_dir: str | Path,
    lineage: Any,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    positions = _positions_from_payload(payload)
    current_position = dict(payload.get("position") or (positions[0] if positions else {}))
    fill = dict(payload.get("fill") or {})
    source_snapshot_id = str(payload.get("snapshot_id") or fill.get("exec_id") or "")
    timestamp = payload.get("timestamp") or fill.get("timestamp")

    position_snapshot = build_position_snapshot(
        current_position,
        lineage=lineage,
        fill=fill,
        order=payload.get("order") or {},
        source="fill",
        source_snapshot_id=source_snapshot_id,
        timestamp=timestamp,
    )
    portfolio_snapshot = build_portfolio_snapshot(
        lineage=lineage,
        positions=positions,
        portfolio_risk=payload.get("portfolio_risk") or {},
        account_state=payload.get("account_state") or {},
        source="fill",
        reconciliation_status="post_fill_unreconciled",
        timestamp=timestamp,
    )
    allocation_snapshot = build_allocation_snapshot(
        lineage=lineage,
        positions=positions,
        targets=payload.get("allocation_targets") or {},
        raw_nav=payload.get("raw_nav"),
        allocated_nav=payload.get("allocated_nav"),
        source="fill",
        timestamp=timestamp,
        metadata={"fill": fill, "order": payload.get("order") or {}},
    )

    return {
        "position": append_jsonl_event(data_dir, "positions", "positions", position_snapshot),
        "portfolio": append_jsonl_event(data_dir, "portfolio", "portfolio_snapshots", portfolio_snapshot),
        "allocation": append_jsonl_event(data_dir, "allocations", "allocations", allocation_snapshot),
    }


def write_reconciliation_lifecycle_event(
    data_dir: str | Path,
    lineage: Any,
    payload: Mapping[str, Any],
) -> Path:
    event = build_reconciliation_event(
        lineage=lineage,
        lifecycle_action=str(payload.get("lifecycle_action") or "reconciliation_alert"),
        status=str(payload.get("status") or "observed"),
        phase=str(payload.get("phase") or ""),
        source=str(payload.get("source") or "reconciliation"),
        details=dict(payload.get("details") or {}),
        discrepancies=list(payload.get("discrepancies") or []),
        timestamp=payload.get("timestamp"),
    )
    subdir = {
        "admin_correction": "admin_corrections",
        "allocation_drift": "allocation_drift",
        "allocation_freeze": "allocation_drift",
        "allocation_unfreeze": "allocation_drift",
        "drift_assignment": "allocation_drift",
        "inferred_fill": "inferred_fills",
    }.get(event["event_type"], "reconciliation")
    return append_jsonl_event(data_dir, subdir, subdir, event)


def write_daily_reconciliation(
    data_dir: str | Path,
    lineage: Any,
    *,
    date_str: str,
    family_id: str = "",
    daily_snapshot: Mapping[str, Any] | None = None,
    portfolio_state: Mapping[str, Any] | None = None,
    allocation_state: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    positions = list((portfolio_state or {}).get("positions") or [])
    portfolio_snapshot = build_portfolio_snapshot(
        lineage=lineage,
        positions=positions,
        portfolio_risk=portfolio_state or {},
        account_state=portfolio_state or {},
        source="daily_closeout",
        reconciliation_status="daily_closeout",
    )
    allocation_snapshot = build_allocation_snapshot(
        lineage=lineage,
        positions=positions,
        targets=(allocation_state or {}).get("targets", {}),
        raw_nav=(allocation_state or {}).get("raw_nav"),
        allocated_nav=(allocation_state or {}).get("allocated_nav"),
        source="daily_closeout",
        metadata=allocation_state or {},
    )
    family_snapshot = build_family_daily_snapshot(
        data_dir=str(data_dir),
        lineage=lineage,
        date_str=date_str,
        family_id=family_id,
        daily_snapshot=daily_snapshot or {},
        portfolio_snapshot=portfolio_snapshot,
        allocation_snapshot=allocation_snapshot,
        source="daily_closeout",
    )
    recon_event = build_reconciliation_event(
        lineage=lineage,
        lifecycle_action="allocation_unfreeze",
        status="closed",
        phase="daily_closeout",
        source="daily_closeout",
        details={
            "date": date_str,
            "family_id": family_id,
            "family_snapshot_id": family_snapshot.get("snapshot_id"),
            "portfolio_snapshot_id": portfolio_snapshot.get("snapshot_id"),
            "allocation_snapshot_id": allocation_snapshot.get("snapshot_id"),
        },
    )
    return {
        "portfolio": append_jsonl_event(data_dir, "portfolio", "portfolio_snapshots", portfolio_snapshot),
        "allocation": append_jsonl_event(data_dir, "allocations", "allocations", allocation_snapshot),
        "family": append_jsonl_event(data_dir, "family", "family", family_snapshot),
        "reconciliation": append_jsonl_event(data_dir, "reconciliation", "reconciliation", recon_event),
    }
