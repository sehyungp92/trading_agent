"""Shared relay-ingest evidence validation for approval and shadow reports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading_contracts.relay_acceptance import contains_placeholder

RELAY_EVIDENCE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def validate_relay_ingest_evidence(
    evidence: dict[str, Any],
    *,
    expected_bot_id: str = "",
    deployment_ids: set[str] | None = None,
    runtime_instance_ids: set[str] | None = None,
    deployment_metadata_hashes: set[str] | None = None,
    max_age_seconds: float = RELAY_EVIDENCE_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> list[str]:
    """Return approval-blocking errors for relay ingest evidence."""

    errors: list[str] = []
    deployment_ids = deployment_ids or set()
    runtime_instance_ids = runtime_instance_ids or set()
    deployment_metadata_hashes = deployment_metadata_hashes or set()

    if evidence.get("ok") is not True:
        errors.append("relay ingest evidence is not ok")

    expected_bot_id = str(expected_bot_id or "").strip()
    evidence_bot = str(evidence.get("bot_id") or "").strip()
    if expected_bot_id and not evidence_bot:
        errors.append("relay ingest evidence missing bot_id")
    elif expected_bot_id and evidence_bot != expected_bot_id:
        errors.append(
            f"relay ingest evidence bot_id {evidence_bot!r} does not match {expected_bot_id!r}"
        )

    event_id = str(evidence.get("event_id") or "").strip()
    if not event_id:
        errors.append("relay ingest evidence missing event_id")

    effective_config_hash = str(evidence.get("effective_config_hash") or "").strip()
    if not effective_config_hash:
        errors.append("relay ingest evidence missing effective_config_hash")
    elif contains_placeholder(effective_config_hash):
        errors.append("relay ingest evidence has placeholder effective_config_hash")

    if evidence.get("authenticated") is False:
        errors.append("relay ingest evidence was unauthenticated")
    auth = evidence.get("auth")
    auth_payload = auth if isinstance(auth, dict) else {}
    if any(
        _flag_true(value)
        for value in (
            evidence.get("placeholder_authenticated"),
            auth_payload.get("placeholder_authenticated"),
            auth_payload.get("placeholder_secret"),
            auth_payload.get("secret_placeholder"),
        )
    ):
        errors.append("relay ingest evidence used placeholder HMAC secret")
    if (
        evidence.get("secret_validation_ok") is False
        or auth_payload.get("secret_validation_ok") is False
    ):
        errors.append("relay ingest evidence HMAC secret validation failed")
    validation_errors = evidence.get("secret_validation_errors") or auth_payload.get(
        "secret_validation_errors"
    )
    if validation_errors:
        errors.append("relay ingest evidence HMAC secret validation failed")

    fingerprint = relay_secret_fingerprint(evidence)
    if not fingerprint:
        errors.append("relay ingest evidence missing HMAC secret fingerprint")
    elif contains_placeholder(fingerprint):
        errors.append("relay ingest evidence has placeholder HMAC secret fingerprint")

    deployment_id = str(evidence.get("deployment_id") or "").strip()
    if not deployment_id:
        errors.append("relay ingest evidence missing deployment_id")
    elif deployment_ids and deployment_id not in deployment_ids:
        errors.append("relay ingest evidence deployment_id is not linked to deployment metadata")

    runtime_instance_id = str(evidence.get("runtime_instance_id") or "").strip()
    if not runtime_instance_id:
        errors.append("relay ingest evidence missing runtime_instance_id")
    elif runtime_instance_ids and runtime_instance_id not in runtime_instance_ids:
        errors.append(
            "relay ingest evidence runtime_instance_id is not linked to deployment metadata"
        )

    evidence_hashes = relay_metadata_hashes(evidence)
    if not evidence_hashes:
        errors.append("relay ingest evidence missing deployment metadata hash")
    elif deployment_metadata_hashes and not (evidence_hashes & deployment_metadata_hashes):
        errors.append("relay ingest evidence deployment metadata hash mismatch")

    errors.extend(
        relay_freshness_errors(
            evidence,
            max_age_seconds=max_age_seconds,
            now=now,
        )
    )
    return errors


def relay_secret_fingerprint(evidence: dict[str, Any]) -> str:
    auth = evidence.get("auth")
    if isinstance(auth, dict):
        for key in ("secret_fingerprint", "hmac_secret_fingerprint"):
            value = str(auth.get(key) or "").strip()
            if value:
                return value
    for key in ("secret_fingerprint", "hmac_secret_fingerprint"):
        value = str(evidence.get(key) or "").strip()
        if value:
            return value
    return ""


def relay_metadata_hashes(evidence: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for key in ("deployment_metadata_hash", "deployment_metadata_sha256"):
        value = str(evidence.get(key) or "").strip()
        if value:
            hashes.add(value)
    raw = evidence.get("deployment_metadata_hashes")
    if isinstance(raw, dict):
        hashes.update(str(value).strip() for value in raw.values() if str(value or "").strip())
    elif isinstance(raw, list):
        hashes.update(str(value).strip() for value in raw if str(value or "").strip())
    return hashes


def relay_freshness_errors(
    evidence: dict[str, Any],
    *,
    max_age_seconds: float = RELAY_EVIDENCE_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> list[str]:
    errors: list[str] = []
    if evidence.get("stale") is True or evidence.get("is_stale") is True:
        errors.append("relay ingest evidence is stale")
    freshness = evidence.get("freshness")
    if isinstance(freshness, dict) and freshness.get("ok") is False:
        errors.append("relay ingest evidence freshness is not ok")

    saw_freshness_proof = False
    for source in (evidence, freshness if isinstance(freshness, dict) else {}):
        for key in (
            "max_event_age_seconds",
            "oldest_event_age_seconds",
            "oldest_accepted_event_age_seconds",
        ):
            if key not in source:
                continue
            saw_freshness_proof = True
            try:
                age = float(source[key])
            except (TypeError, ValueError):
                errors.append(f"relay ingest evidence {key} is not numeric")
                continue
            if age > max_age_seconds:
                errors.append(f"relay ingest evidence {key} is stale at {age:.0f}s")

    timestamp = str(evidence.get("observed_at") or evidence.get("generated_at") or "").strip()
    if timestamp:
        saw_freshness_proof = True
        observed_at = _parse_timestamp(timestamp)
        if observed_at is None:
            errors.append("relay ingest evidence freshness timestamp is malformed")
        else:
            age = (
                (_utc_now() if now is None else now.astimezone(UTC)) - observed_at
            ).total_seconds()
            if age > max_age_seconds:
                errors.append(f"relay ingest evidence freshness timestamp is stale at {age:.0f}s")

    if not saw_freshness_proof:
        errors.append("relay ingest evidence missing freshness proof")
    return errors


def _parse_timestamp(value: str) -> datetime | None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _flag_true(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}
