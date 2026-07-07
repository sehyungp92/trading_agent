"""Shared lineage coverage helpers for event curation and audits."""
from __future__ import annotations

from collections import Counter
from typing import Any


MONTHLY_REQUIRED_LINEAGE_FIELDS = (
    "strategy_version",
    "config_version",
    "deployment_id",
)

LINEAGE_COUNT_FIELDS = (
    "event_id",
    "strategy_version",
    "config_version",
    "deployment_id",
    "parameter_set_id",
    "experiment_id",
    "variant_id",
    "code_sha",
    "signal_id",
    "bar_id",
    "data_source_id",
    "exchange_timestamp",
    "local_timestamp",
    "clock_skew_ms",
)


def _field_value(container: object, field: str) -> object:
    if isinstance(container, dict):
        return container.get(field)
    return getattr(container, field, None)


def _metadata_value(container: object, field: str) -> object:
    for key in ("event_metadata", "metadata"):
        metadata = _field_value(container, key)
        if metadata is None:
            continue
        value = _field_value(metadata, field)
        if value not in (None, ""):
            return value
    return None


def _value_from_container(container: object, field: str) -> object:
    value = _field_value(container, field)
    if value not in (None, ""):
        return value
    return _metadata_value(container, field)


def event_value(event: object, field: str) -> object:
    if isinstance(event, dict):
        payload = event.get("payload", event)
        if isinstance(payload, dict):
            value = _value_from_container(payload, field)
            if value not in (None, ""):
                return value
        return _value_from_container(event, field)
    return _value_from_container(event, field)


def event_strategy_id(event: object) -> str:
    value = event_value(event, "strategy_id")
    return str(value or "")


def build_lineage_summary(
    events: list[object],
    *,
    required_fields: tuple[str, ...] = MONTHLY_REQUIRED_LINEAGE_FIELDS,
) -> dict[str, Any]:
    """Return counts and missing coverage for monthly-relevant lineage."""

    total = len(events)
    field_counts: dict[str, dict[str, int]] = {}
    missing_field_counts: dict[str, int] = {}
    for field in LINEAGE_COUNT_FIELDS:
        counter: Counter[str] = Counter()
        missing = 0
        for event in events:
            value = event_value(event, field)
            if value in (None, ""):
                missing += 1
            else:
                counter[str(value)] += 1
        field_counts[f"{field}_counts"] = dict(sorted(counter.items()))
        missing_field_counts[field] = missing

    required_slots = total * len(required_fields)
    missing_required = sum(missing_field_counts.get(field, 0) for field in required_fields)
    lineage_coverage_ratio = (
        1.0 if required_slots == 0 else max(0.0, 1.0 - (missing_required / required_slots))
    )
    lineage_gap = total > 0 and missing_required > 0

    return {
        **field_counts,
        "missing_field_counts": missing_field_counts,
        "required_fields": list(required_fields),
        "total_events": total,
        "lineage_coverage_ratio": round(lineage_coverage_ratio, 6),
        "lineage_gap": lineage_gap,
    }


def missing_required_lineage_fields(
    event: object,
    *,
    required_fields: tuple[str, ...] = MONTHLY_REQUIRED_LINEAGE_FIELDS,
) -> list[str]:
    return [
        field for field in required_fields
        if event_value(event, field) in (None, "")
    ]
