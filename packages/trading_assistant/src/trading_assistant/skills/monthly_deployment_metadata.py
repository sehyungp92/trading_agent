"""Deployment metadata checks shared by monthly structural approval gates."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest

REQUIRED_DEPLOYMENT_METADATA_FIELDS = (
    "bot_id",
    "strategy_id",
    "repo_url",
    "deployed_commit_sha",
    "config_hash",
    "strategy_version",
    "config_version",
    "telemetry_schema_version",
    "strategy_plugin_contract_path",
    "strategy_plugin_contract_hash",
)

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


def deployment_metadata_errors(
    manifest: MonthlyRunManifest | None,
    *,
    missing_reason: str,
) -> list[str]:
    if manifest is None or not manifest.deployment_metadata_path:
        return [missing_reason]
    path = Path(manifest.deployment_metadata_path)
    if not path.exists() or not path.is_file():
        return ["deployment metadata path is missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"deployment metadata is invalid: {exc}"]
    if not isinstance(payload, dict):
        return ["deployment metadata must be a JSON object"]

    missing = [
        field
        for field in REQUIRED_DEPLOYMENT_METADATA_FIELDS
        if not str(payload.get(field) or "").strip()
    ]
    errors = (
        ["deployment metadata missing required fields: " + ", ".join(missing)]
        if missing else []
    )
    expected = {
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "deployed_commit_sha": manifest.trading_repo_commit_sha,
        "config_hash": manifest.config_hash,
        "strategy_version": manifest.strategy_version,
        "config_version": manifest.config_version,
    }
    for field, value in expected.items():
        if value and str(payload.get(field) or "").strip() != value:
            errors.append(f"deployment metadata {field} does not match run manifest")

    declared_path = Path(str(payload.get("strategy_plugin_contract_path") or ""))
    if not declared_path.is_absolute():
        declared_path = path.parent / declared_path
    if manifest.strategy_plugin_contract_path:
        if declared_path.resolve() != Path(manifest.strategy_plugin_contract_path).resolve():
            errors.append(
                "deployment metadata strategy_plugin_contract_path does not match run manifest"
            )
    if declared_path.exists():
        digest = hashlib.sha256(declared_path.read_bytes()).hexdigest()
        if digest != str(payload.get("strategy_plugin_contract_hash") or "").strip():
            errors.append(
                "deployment metadata strategy_plugin_contract_hash does not match contract artifact"
            )
        errors.extend(_telemetry_schema_errors(payload, declared_path))
    else:
        errors.append("deployment metadata strategy_plugin_contract_path does not exist")
    errors.extend(live_deployment_metadata_errors(payload))
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


def _telemetry_schema_errors(payload: dict, contract_path: Path) -> list[str]:
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"strategy plugin contract is invalid for deployment metadata: {exc}"]
    if not isinstance(contract, dict):
        return ["strategy plugin contract must be a JSON object"]
    required = {
        str(schema).strip()
        for schema in contract.get("required_telemetry_schemas", [])
        if str(schema).strip()
    }
    telemetry_schema = str(payload.get("telemetry_schema_version") or "").strip()
    if required and telemetry_schema not in required:
        return [
            "deployment metadata telemetry_schema_version is not listed in "
            "strategy plugin contract required_telemetry_schemas"
        ]
    return []


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
