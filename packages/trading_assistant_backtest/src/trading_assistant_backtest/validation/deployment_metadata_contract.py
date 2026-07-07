"""Approval-grade deployment metadata contract checks."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

APPROVAL_METADATA_SOURCES = {
    "vps_live_bot_runtime_deployment_metadata_v1",
    "live_bot_runtime_deployment_metadata_v1",
}

APPROVAL_EMISSION_ENVIRONMENTS = {
    "vps",
    "paper_vps",
    "live_bot",
    "production_vps",
}

FORBIDDEN_APPROVAL_TOKENS = (
    "local",
    "shadow",
    "snapshot",
    "dry",
    "example",
    "fixture",
    "test",
)


def declared_telemetry_schema_versions(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Return telemetry schema versions declared by runtime metadata.

    ``telemetry_schema_version`` is the legacy singular field. New metadata
    should also include ``telemetry_schema_versions`` so multi-schema contracts
    can be validated without lossy comma-joining.
    """

    values: list[str] = []
    for field in ("telemetry_schema_versions", "telemetry_schemas"):
        values.extend(_schema_values(metadata.get(field)))
    values.extend(_schema_values(metadata.get("telemetry_schema_version")))
    return tuple(dict.fromkeys(values))


def telemetry_schema_contract_errors(
    metadata: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    """Return telemetry-schema mismatches between metadata and contract."""

    required = tuple(dict.fromkeys(_schema_values(contract.get("required_telemetry_schemas"))))
    if not required:
        return []
    declared = declared_telemetry_schema_versions(metadata)
    if not declared:
        return ["deployment metadata missing telemetry_schema_version(s)"]

    declared_set = set(declared)
    required_set = set(required)
    missing = [schema for schema in required if schema not in declared_set]
    unknown = [schema for schema in declared if schema not in required_set]
    errors: list[str] = []
    if missing:
        errors.append(
            "deployment metadata missing required telemetry schema(s): "
            + ", ".join(missing)
        )
    if unknown:
        errors.append(
            "deployment metadata declares telemetry schema(s) not listed in contract: "
            + ", ".join(unknown)
        )
    return errors


def live_deployment_metadata_errors(metadata: dict[str, Any]) -> list[str]:
    """Return blockers that prevent metadata from counting as live-emitted evidence."""

    errors: list[str] = []
    source = _text(metadata.get("metadata_source")).lower()
    environment = _text(metadata.get("emission_environment")).lower()
    context = _text(metadata.get("emission_context")).lower()

    if source not in APPROVAL_METADATA_SOURCES:
        errors.append(
            "metadata_source must be one of "
            + ", ".join(sorted(APPROVAL_METADATA_SOURCES))
        )
    if environment not in APPROVAL_EMISSION_ENVIRONMENTS:
        errors.append(
            "emission_environment must be one of "
            + ", ".join(sorted(APPROVAL_EMISSION_ENVIRONMENTS))
        )
    for label, value in (
        ("metadata_source", source),
        ("emission_environment", environment),
        ("emission_context", context),
    ):
        forbidden = [token for token in FORBIDDEN_APPROVAL_TOKENS if token in value]
        if forbidden:
            errors.append(f"{label} contains non-approval token(s): {', '.join(forbidden)}")

    required = (
        "emitted_at_utc",
        "live_runtime_started_at_utc",
        "runtime_entrypoint",
        "runtime_instance_id",
        "runtime_host_fingerprint",
        "source_control_origin",
        "source_control_commit_sha",
    )
    missing = [field for field in required if not _text(metadata.get(field))]
    if missing:
        errors.append("deployment metadata missing live-emission fields: " + ", ".join(missing))

    for field in ("emitted_at_utc", "live_runtime_started_at_utc"):
        value = _text(metadata.get(field))
        if value and not _is_utc_timestamp(value):
            errors.append(f"{field} must be an ISO-8601 UTC timestamp ending in Z")

    if metadata.get("dry_run") is True:
        errors.append("deployment metadata dry_run=true cannot be approval evidence")
    if metadata.get("source_control_worktree_clean") is not True:
        errors.append("source_control_worktree_clean must be true")

    deployed_sha = _text(metadata.get("deployed_commit_sha"))
    source_sha = _text(metadata.get("source_control_commit_sha"))
    if deployed_sha and source_sha and deployed_sha != source_sha:
        errors.append("source_control_commit_sha does not match deployed_commit_sha")

    repo_url = _text(metadata.get("repo_url"))
    origin = _text(metadata.get("source_control_origin"))
    if repo_url and origin and repo_url != origin:
        errors.append("source_control_origin does not match repo_url")
    if repo_url.startswith("local://") or origin.startswith("local://"):
        errors.append("source_control_origin/repo_url must not be local://")
    errors.extend(_assistant_lineage_errors(metadata))

    return errors


def _assistant_lineage_errors(metadata: dict[str, Any]) -> list[str]:
    lineage = metadata.get("assistant_lineage")
    assistant_driven = str(metadata.get("assistant_driven") or "").lower() in {"1", "true", "yes"}
    if isinstance(lineage, dict):
        assistant_driven = assistant_driven or any(
            _list_value(lineage.get(field))
            for field in (
                "weekly_signal_ids",
                "proposal_ids",
                "suggestion_ids",
                "hypothesis_ids",
                "strategy_change_record_ids",
            )
        ) or bool(_text(lineage.get("monthly_search_brief_id")) or _text(lineage.get("monthly_outcome_id")))
    assistant_driven = assistant_driven or any(
        _list_value(metadata.get(field))
        for field in (
            "source_weekly_signal_ids",
            "weekly_signal_ids",
            "proposal_ids",
            "proposal_id",
            "suggestion_ids",
            "suggestion_id",
            "hypothesis_ids",
            "hypothesis_id",
            "strategy_change_record_ids",
            "strategy_change_record_id",
        )
    )
    if not assistant_driven:
        return []
    if not isinstance(lineage, dict):
        return ["assistant-driven deployment metadata requires assistant_lineage"]
    proposal_ids = _list_value(lineage.get("proposal_ids")) or _list_value(metadata.get("proposal_ids")) or _list_value(metadata.get("proposal_id"))
    errors: list[str] = []
    if not proposal_ids:
        errors.append("assistant-driven deployment metadata requires assistant_lineage.proposal_ids")
    if not _text(lineage.get("deployment_id") or metadata.get("deployment_id")):
        errors.append("assistant-driven deployment metadata requires assistant_lineage.deployment_id")
    return errors


def _list_value(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    return [str(value)]


def _schema_values(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_schema_values(item))
        return values
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _is_utc_timestamp(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00").astimezone(UTC)
    except ValueError:
        return False
    return True
