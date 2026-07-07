"""Shared queue event envelope validation for direct and relay ingest."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from trading_assistant.orchestrator.input_sanitizer import InputSanitizer
from trading_assistant.schemas.canonical_envelope import merge_envelope_fields_into_payload

MAX_INGEST_PAYLOAD_BYTES = 256 * 1024
SYSTEM_BOT_IDS = frozenset({"system", "scheduler", "user", "orchestrator"})


class QueueEventValidationError(ValueError):
    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _check_ingest_payload_size(payload: str) -> None:
    payload_bytes = len(payload.encode("utf-8"))
    if payload_bytes > MAX_INGEST_PAYLOAD_BYTES:
        raise QueueEventValidationError(
            f"Payload too large: {payload_bytes} bytes "
            f"(max {MAX_INGEST_PAYLOAD_BYTES})",
            status_code=413,
        )


def normalize_queue_event(
    event: dict,
    allowed_bot_ids: set[str] | None = None,
) -> dict:
    normalized = dict(event)
    missing = [
        field for field in ("event_id", "bot_id", "event_type", "payload")
        if field not in normalized
    ]
    if missing:
        raise QueueEventValidationError(
            f"Missing required event field(s): {', '.join(missing)}",
        )

    for field in ("event_id", "bot_id", "event_type"):
        if not isinstance(normalized[field], str) or not normalized[field].strip():
            raise QueueEventValidationError(f"'{field}' must be a non-empty string")

    bot_id = normalized["bot_id"]
    if allowed_bot_ids is not None:
        permitted = allowed_bot_ids | SYSTEM_BOT_IDS
        if bot_id not in permitted:
            raise QueueEventValidationError(f"Unknown bot_id: {bot_id!r}")

    payload = normalized["payload"]
    if isinstance(payload, dict):
        normalized["payload"] = json.dumps(
            merge_envelope_fields_into_payload(normalized, payload),
            default=str,
        )
    elif isinstance(payload, list):
        normalized["payload"] = json.dumps(payload, default=str)
    elif not isinstance(payload, str):
        raise QueueEventValidationError(
            "'payload' must be a JSON string, object, or array",
        )
    else:
        try:
            parsed_payload = json.loads(payload)
        except json.JSONDecodeError:
            parsed_payload = None
        if isinstance(parsed_payload, dict):
            normalized["payload"] = json.dumps(
                merge_envelope_fields_into_payload(normalized, parsed_payload),
                default=str,
            )

    _check_ingest_payload_size(normalized["payload"])

    if normalized.get("event_type") == "user_feedback":
        try:
            feedback_payload = json.loads(normalized["payload"])
        except json.JSONDecodeError as exc:
            raise QueueEventValidationError(
                "user_feedback payload must be a JSON object",
            ) from exc
        if not isinstance(feedback_payload, dict):
            raise QueueEventValidationError(
                "user_feedback payload must be a JSON object",
            )
        feedback_text = feedback_payload.get("text")
        if feedback_text is None or not str(feedback_text).strip():
            raise QueueEventValidationError(
                "user_feedback payload requires non-empty text",
            )
        sanitized = InputSanitizer().sanitize(str(feedback_text), source="ingest")
        if not sanitized.safe:
            raise QueueEventValidationError(
                f"Feedback rejected: {sanitized.reason}",
            )
        feedback_payload["text"] = sanitized.content
        feedback_payload.setdefault("intent", sanitized.intent)
        normalized["payload"] = json.dumps(feedback_payload, default=str)

    _check_ingest_payload_size(normalized["payload"])

    received_at = normalized.setdefault("received_at", _utc_now().isoformat())
    normalized.setdefault("exchange_timestamp", received_at)
    return normalized
