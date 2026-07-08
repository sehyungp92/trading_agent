from __future__ import annotations

import json

from trading_contracts.relay_acceptance import (
    RelayProbeResult,
    bot_exact_ack_api_key,
    canonical_relay_body,
    contains_placeholder,
    deployment_heartbeat_envelope,
    relay_signature,
    validate_hmac_secret,
    validate_relay_config,
)
from trading_contracts.relay_evidence import validate_relay_ingest_evidence


STRONG_SECRET = "0123456789abcdef0123456789ABCDEF"


def test_relay_config_rejects_placeholders_low_entropy_and_loopback() -> None:
    assert contains_placeholder("${INSTRUMENTATION_HMAC_SECRET}")

    errors = validate_relay_config(
        relay_url="http://127.0.0.1:8000/change-me/events",
        hmac_secret="aaaaaaaa",
        bot_id="bot-1",
        allow_loopback=False,
    )

    assert any("loopback" in error for error in errors)
    assert any("placeholder" in error for error in errors)
    assert any("at least 32" in error for error in errors)
    assert any("distinct" in error for error in errors)


def test_relay_config_allows_explicit_local_direct_loopback() -> None:
    assert validate_relay_config(
        relay_url="http://127.0.0.1:8000/events",
        hmac_secret=STRONG_SECRET,
        bot_id="bot-1",
        allow_loopback=True,
    ) == []


def test_validate_hmac_secret_requires_entropy() -> None:
    errors = validate_hmac_secret("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert any("distinct" in error for error in errors)


def test_deployment_heartbeat_envelope_has_canonical_signature_surface() -> None:
    envelope = deployment_heartbeat_envelope(
        bot_id="crypto-momentum",
        runtime_instance_id="runtime-1",
        effective_config_hash="a" * 64,
        deployment_id="deploy-1",
        source="contract-test",
        exchange_timestamp="2026-07-08T12:00:00Z",
    )

    body = canonical_relay_body(envelope)
    event = envelope["events"][0]
    payload = json.loads(event["payload"])

    assert envelope["bot_id"] == "crypto-momentum"
    assert event["priority"] == 0
    assert event["event_id"].startswith("relay-heartbeat-")
    assert payload["effective_config_hash"] == "a" * 64
    assert payload["runtime_instance_id"] == "runtime-1"
    assert relay_signature(body, STRONG_SECRET) == relay_signature(body, STRONG_SECRET)


def test_relay_probe_result_serializes_verifier_evidence_fields() -> None:
    payload = RelayProbeResult(
        ok=True,
        event_id="relay-heartbeat-1",
        bot_id="ibkr",
        runtime_instance_id="ibkr-runtime-1",
        effective_config_hash="a" * 64,
        deployment_id="ibkr-deploy-1",
        source="unit-test",
        status_code=202,
        accepted=1,
        secret_fingerprint="hmac-sha256:abcdef1234567890",
        observed_at="2026-07-08T12:00:00Z",
    ).to_json_dict()

    assert payload["bot_id"] == "ibkr"
    assert payload["runtime_instance_id"] == "ibkr-runtime-1"
    assert payload["effective_config_hash"] == "a" * 64
    assert payload["deployment_id"] == "ibkr-deploy-1"
    assert payload["auth"]["secret_fingerprint"] == "hmac-sha256:abcdef1234567890"
    assert payload["freshness"] == {"ok": True, "max_event_age_seconds": 0}


def test_bot_exact_ack_api_key_requires_explicit_opt_in() -> None:
    assert bot_exact_ack_api_key({"RELAY_API_KEY": "assistant-read-key"}) == ""
    assert bot_exact_ack_api_key({
        "ALLOW_BOT_RELAY_EXACT_ACK": "1",
        "ASSISTANT_RELAY_EXACT_ACK_API_KEY": "assistant-read-key",
    }) == "assistant-read-key"


def test_relay_ingest_evidence_blocks_stale_placeholder_and_unlinked_metadata() -> None:
    errors = validate_relay_ingest_evidence(
        {
            "ok": True,
            "bot_id": "ibkr",
            "event_id": "relay-heartbeat-1",
            "effective_config_hash": "b" * 64,
            "deployment_id": "deployment-1",
            "runtime_instance_id": "runtime-1",
            "deployment_metadata_hash": "0" * 64,
            "auth": {"secret_fingerprint": "change-me", "secret_placeholder": True},
            "freshness": {"ok": True, "max_event_age_seconds": 999999999},
        },
        expected_bot_id="ibkr",
        deployment_ids={"deployment-1"},
        runtime_instance_ids={"runtime-1"},
        deployment_metadata_hashes={"a" * 64},
    )

    assert "relay ingest evidence has placeholder HMAC secret fingerprint" in errors
    assert "relay ingest evidence used placeholder HMAC secret" in errors
    assert "relay ingest evidence deployment metadata hash mismatch" in errors
    assert any("max_event_age_seconds is stale" in error for error in errors)
