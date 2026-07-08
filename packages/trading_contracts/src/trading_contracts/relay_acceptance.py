"""Shared relay readiness and signed deployment-heartbeat helpers."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping


PLACEHOLDER_TOKENS = (
    "change-me",
    "changeme",
    "change_me",
    "placeholder",
    "todo",
    "example",
    "your_",
    "your-",
    "workstation_private_ip",
    "workstation-private-ip",
    "<",
    ">",
)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
MIN_HMAC_SECRET_LENGTH = 32
MIN_HMAC_SECRET_UNIQUE_CHARS = 8


@dataclass(frozen=True, slots=True)
class RelayProbeResult:
    ok: bool
    event_id: str = ""
    bot_id: str = ""
    runtime_instance_id: str = ""
    effective_config_hash: str = ""
    deployment_id: str = ""
    source: str = ""
    status_code: int = 0
    accepted: int = 0
    duplicates: int = 0
    health_confirmed: bool = False
    exact_ack_confirmed: bool = False
    secret_fingerprint: str = ""
    observed_at: str = ""
    error: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "ok": self.ok,
            "event_id": self.event_id,
            "bot_id": self.bot_id,
            "runtime_instance_id": self.runtime_instance_id,
            "effective_config_hash": self.effective_config_hash,
            "deployment_id": self.deployment_id,
            "source": self.source,
            "status_code": self.status_code,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "health_confirmed": self.health_confirmed,
            "exact_ack_confirmed": self.exact_ack_confirmed,
            "observed_at": self.observed_at,
            "generated_at": self.observed_at,
            "error": self.error,
        }
        if self.secret_fingerprint:
            payload["authenticated"] = self.accepted + self.duplicates > 0
            payload["auth"] = {
                "secret_fingerprint": self.secret_fingerprint,
                "secret_validation_ok": self.ok,
                "placeholder_authenticated": False,
            }
        if self.observed_at:
            payload["freshness"] = {"ok": self.ok, "max_event_age_seconds": 0}
        return payload


def contains_placeholder(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("${") and lowered.endswith("}"):
        return True
    normalized = lowered.replace(" ", "_")
    return any(token in lowered or token in normalized for token in PLACEHOLDER_TOKENS)


def secret_fingerprint(secret: str) -> str:
    value = str(secret or "").encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def validate_identifier_field(name: str, value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return [f"{name} is required"]
    if contains_placeholder(text):
        return [f"{name} must not contain a placeholder value"]
    return []


def validate_hmac_secret(
    secret: Any,
    *,
    field_name: str = "relay_secret",
    min_length: int = MIN_HMAC_SECRET_LENGTH,
) -> list[str]:
    text = str(secret or "").strip()
    errors: list[str] = []
    if not text:
        return [f"{field_name} is required"]
    if contains_placeholder(text):
        errors.append(f"{field_name} must not contain a placeholder value")
    if len(text) < min_length:
        errors.append(f"{field_name} must be at least {min_length} characters")
    if len(set(text)) < MIN_HMAC_SECRET_UNIQUE_CHARS:
        errors.append(
            f"{field_name} must contain at least {MIN_HMAC_SECRET_UNIQUE_CHARS} distinct characters"
        )
    return errors


def validate_relay_url(
    relay_url: Any,
    *,
    field_name: str = "relay_url",
    allow_loopback: bool = False,
) -> list[str]:
    text = str(relay_url or "").strip()
    if not text:
        return [f"{field_name} is required"]
    errors: list[str] = []
    if contains_placeholder(text):
        errors.append(f"{field_name} must not contain a placeholder value")
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        errors.append(f"{field_name} must be an absolute http(s) URL")
        return errors
    host = (parsed.hostname or "").lower()
    if host in LOOPBACK_HOSTS and not allow_loopback:
        errors.append(f"{field_name} must not target loopback for paper/live remote ingestion")
    return errors


def validate_relay_config(
    *,
    relay_url: Any,
    hmac_secret: Any,
    bot_id: Any,
    allow_loopback: bool = False,
    require: bool = True,
    secret_field_name: str = "relay_secret",
) -> list[str]:
    if not require and not any(str(value or "").strip() for value in (relay_url, hmac_secret, bot_id)):
        return []
    errors: list[str] = []
    errors.extend(validate_identifier_field("bot_id", bot_id))
    errors.extend(validate_relay_url(relay_url, allow_loopback=allow_loopback))
    errors.extend(validate_hmac_secret(hmac_secret, field_name=secret_field_name))
    return errors


def bot_exact_ack_api_key(env: Mapping[str, str]) -> str:
    """Return a bot-side exact-ack key only when explicitly enabled.

    Bot deployment probes are write-side health checks by default. Requiring an
    opt-in avoids putting the assistant read/ack API key on VPS bots.
    """
    enabled = str(env.get("ALLOW_BOT_RELAY_EXACT_ACK") or "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return ""
    return str(env.get("ASSISTANT_RELAY_EXACT_ACK_API_KEY") or "").strip()


def deployment_heartbeat_event(
    *,
    bot_id: str,
    runtime_instance_id: str,
    effective_config_hash: str,
    deployment_id: str,
    source: str,
    event_type: str = "deployment_start",
    event_id: str = "",
    exchange_timestamp: str = "",
) -> dict[str, Any]:
    timestamp = exchange_timestamp or _utc_now()
    payload = {
        "bot_id": bot_id,
        "event_type": event_type,
        "runtime_instance_id": runtime_instance_id,
        "effective_config_hash": effective_config_hash,
        "deployment_id": deployment_id,
        "exchange_timestamp": timestamp,
        "source": source,
        "host_fingerprint": _host_fingerprint(),
    }
    stable_id = event_id or "relay-heartbeat-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    payload["event_id"] = stable_id
    return {
        "event_id": stable_id,
        "bot_id": bot_id,
        "event_type": event_type,
        "payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        "exchange_timestamp": timestamp,
        "priority": 0,
    }


def deployment_heartbeat_envelope(
    *,
    bot_id: str,
    runtime_instance_id: str,
    effective_config_hash: str,
    deployment_id: str,
    source: str,
    event_type: str = "deployment_start",
    event_id: str = "",
    exchange_timestamp: str = "",
) -> dict[str, Any]:
    return {
        "bot_id": bot_id,
        "events": [
            deployment_heartbeat_event(
                bot_id=bot_id,
                runtime_instance_id=runtime_instance_id,
                effective_config_hash=effective_config_hash,
                deployment_id=deployment_id,
                source=source,
                event_type=event_type,
                event_id=event_id,
                exchange_timestamp=exchange_timestamp,
            )
        ],
    }


def canonical_relay_body(envelope: dict[str, Any]) -> bytes:
    return json.dumps(envelope, sort_keys=True).encode("utf-8")


def relay_signature(body: bytes, hmac_secret: str) -> str:
    return hmac.new(str(hmac_secret).encode("utf-8"), body, hashlib.sha256).hexdigest()


def probe_relay_acceptance(
    *,
    relay_url: str,
    hmac_secret: str,
    bot_id: str,
    runtime_instance_id: str,
    effective_config_hash: str,
    deployment_id: str,
    source: str,
    event_type: str = "deployment_start",
    timeout_seconds: float = 10.0,
    use_gzip: bool = False,
    confirm_health: bool = True,
    require_exact_ack: bool = False,
    relay_api_key: str = "",
) -> RelayProbeResult:
    observed_at = _utc_now()
    probe_metadata = {
        "bot_id": bot_id,
        "runtime_instance_id": runtime_instance_id,
        "effective_config_hash": effective_config_hash,
        "deployment_id": deployment_id,
        "source": source,
        "secret_fingerprint": (
            f"hmac-sha256:{secret_fingerprint(hmac_secret)}" if hmac_secret else ""
        ),
        "observed_at": observed_at,
    }
    errors = validate_relay_config(
        relay_url=relay_url,
        hmac_secret=hmac_secret,
        bot_id=bot_id,
        allow_loopback=True,
    )
    if errors:
        return RelayProbeResult(ok=False, error="; ".join(errors), **probe_metadata)

    envelope = deployment_heartbeat_envelope(
        bot_id=bot_id,
        runtime_instance_id=runtime_instance_id,
        effective_config_hash=effective_config_hash,
        deployment_id=deployment_id,
        source=source,
        event_type=event_type,
    )
    event_id = str(envelope["events"][0]["event_id"])
    probe_metadata["event_id"] = event_id
    body = canonical_relay_body(envelope)
    headers = {
        "Content-Type": "application/json",
        "X-Bot-Id": bot_id,
        "X-Signature": relay_signature(body, hmac_secret),
    }
    request_body = gzip.compress(body) if use_gzip else body
    if use_gzip:
        headers["Content-Encoding"] = "gzip"

    try:
        response_payload, status_code = _request_json(
            _events_url(relay_url),
            method="POST",
            body=request_body,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return RelayProbeResult(ok=False, error=str(exc), **probe_metadata)

    accepted = int(response_payload.get("accepted") or 0) if isinstance(response_payload, dict) else 0
    duplicates = int(response_payload.get("duplicates") or 0) if isinstance(response_payload, dict) else 0
    if status_code < 200 or status_code >= 300:
        return RelayProbeResult(
            ok=False,
            status_code=status_code,
            accepted=accepted,
            duplicates=duplicates,
            error=f"relay returned HTTP {status_code}",
            **probe_metadata,
        )
    if accepted + duplicates <= 0:
        return RelayProbeResult(
            ok=False,
            status_code=status_code,
            accepted=accepted,
            duplicates=duplicates,
            error="relay did not accept or deduplicate the heartbeat",
            **probe_metadata,
        )

    health_confirmed = True
    if confirm_health:
        try:
            health_confirmed = _health_mentions_bot(relay_url, bot_id, timeout_seconds)
        except Exception:
            health_confirmed = False

    exact_ack_confirmed = False
    if require_exact_ack:
        if not relay_api_key:
            return RelayProbeResult(
                ok=False,
                status_code=status_code,
                accepted=accepted,
                duplicates=duplicates,
                health_confirmed=health_confirmed,
                error="relay_api_key is required for exact ack confirmation",
                **probe_metadata,
            )
        try:
            exact_ack_confirmed = _exact_pull_and_ack(
                relay_url,
                event_id,
                bot_id,
                relay_api_key,
                timeout_seconds,
            )
        except Exception:
            exact_ack_confirmed = False

    ok = health_confirmed and (exact_ack_confirmed if require_exact_ack else True)
    return RelayProbeResult(
        ok=ok,
        status_code=status_code,
        accepted=accepted,
        duplicates=duplicates,
        health_confirmed=health_confirmed,
        exact_ack_confirmed=exact_ack_confirmed,
        error="" if ok else "relay heartbeat was posted but ingestion confirmation failed",
        **probe_metadata,
    )


def _events_url(relay_url: str) -> str:
    base = str(relay_url or "").strip().rstrip("/")
    return base if base.endswith("/events") else f"{base}/events"


def _base_url(relay_url: str) -> str:
    base = str(relay_url or "").strip().rstrip("/")
    return base.removesuffix("/events")


def _request_json(
    url: str,
    *,
    method: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float,
) -> tuple[Any, int]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw or "{}"), int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {"error": raw}
        return payload, int(exc.code)


def _health_mentions_bot(relay_url: str, bot_id: str, timeout_seconds: float) -> bool:
    payload, status = _request_json(
        f"{_base_url(relay_url)}/health",
        method="GET",
        timeout_seconds=timeout_seconds,
    )
    if status < 200 or status >= 300 or not isinstance(payload, dict):
        return False
    per_bot = payload.get("per_bot_pending")
    last_event = payload.get("last_event_per_bot")
    return (
        isinstance(per_bot, dict)
        and bot_id in per_bot
        or isinstance(last_event, dict)
        and bot_id in last_event
    )


def _exact_pull_and_ack(
    relay_url: str,
    event_id: str,
    bot_id: str,
    relay_api_key: str,
    timeout_seconds: float,
) -> bool:
    query = urllib.parse.urlencode(
        {
            "bot_id": bot_id,
            "priority_first": "true",
            "max_priority": "0",
            "limit": "100",
        }
    )
    payload, status = _request_json(
        f"{_base_url(relay_url)}/events?{query}",
        method="GET",
        headers={"X-Api-Key": relay_api_key},
        timeout_seconds=timeout_seconds,
    )
    if status < 200 or status >= 300 or not isinstance(payload, dict):
        return False
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    if not any(isinstance(event, dict) and event.get("event_id") == event_id for event in events):
        return False
    ack_payload = json.dumps({"event_ids": [event_id]}, sort_keys=True).encode("utf-8")
    ack_response, ack_status = _request_json(
        f"{_base_url(relay_url)}/ack-exact",
        method="POST",
        body=ack_payload,
        headers={"Content-Type": "application/json", "X-Api-Key": relay_api_key},
        timeout_seconds=timeout_seconds,
    )
    return (
        200 <= ack_status < 300
        and isinstance(ack_response, dict)
        and int(ack_response.get("acked_count") or 0) >= 1
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _host_fingerprint() -> str:
    raw = socket.gethostname().encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]
