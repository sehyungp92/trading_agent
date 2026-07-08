from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_contracts.relay_acceptance import (
    canonical_relay_body,
    deployment_heartbeat_envelope,
    relay_signature,
)
import trading_assistant.relay_ingress.app as relay_app
from trading_assistant.relay_ingress.app import RelayStore, create_relay_app


def _event(event_id: str, priority: int) -> dict[str, object]:
    return {
        "event_id": event_id,
        "bot_id": "bot-1",
        "event_type": "test",
        "payload": "{}",
        "exchange_timestamp": "2026-05-10T12:00:00+00:00",
        "priority": priority,
    }


def test_assistant_relay_store_priority_first_uses_exact_ack(tmp_path: Path) -> None:
    store = RelayStore(str(tmp_path / "relay.db"))
    store.insert_events([_event("low-old", 4), _event("high-new", 1)])

    assert [event["event_id"] for event in store.get_events()] == ["low-old", "high-new"]
    assert [event["event_id"] for event in store.get_events(priority_first=True, max_priority=1)] == [
        "high-new"
    ]

    assert store.ack_exact(["high-new"]) == 1
    assert [event["event_id"] for event in store.get_events()] == ["low-old"]

    assert store.ack_up_to("low-old") == 1
    assert store.get_events() == []


def test_assistant_relay_ingress_matches_receiver_protocol(tmp_path: Path) -> None:
    app = create_relay_app(
        db_path=str(tmp_path / "relay.db"),
        shared_secrets={"bot1": "secret"},
        api_key="relay-key",
    )
    payload = {
        "bot_id": "bot1",
        "events": [
            {
                "event_id": "evt-low",
                "bot_id": "bot1",
                "event_type": "trade",
                "payload": "{}",
                "priority": "low",
            },
            {
                "event_id": "evt-high",
                "bot_id": "bot1",
                "event_type": "risk",
                "payload": "{}",
                "priority": "high",
            },
        ],
    }
    body = json.dumps(payload, sort_keys=True).encode()
    signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()

    with TestClient(app) as client:
        ingest = client.post(
            "/events",
            content=body,
            headers={"Content-Type": "application/json", "X-Signature": signature},
        )
        assert ingest.status_code == 200
        assert ingest.json() == {"accepted": 2, "duplicates": 0}

        unauthorized = client.get("/events")
        assert unauthorized.status_code == 401

        urgent = client.get(
            "/events",
            params={"priority_first": "true", "max_priority": 1},
            headers={"X-Api-Key": "relay-key"},
        )
        assert urgent.status_code == 200
        urgent_payload = urgent.json()
        assert urgent_payload["delivery_mode"] == "priority_first"
        assert urgent_payload["ack_mode"] == "exact"
        assert [event["event_id"] for event in urgent_payload["events"]] == ["evt-high"]

        ack = client.post(
            "/ack-exact",
            json={"event_ids": ["evt-high"]},
            headers={"X-Api-Key": "relay-key"},
        )
        assert ack.status_code == 200
        assert ack.json()["acked_count"] == 1

        remaining = client.get("/events", headers={"X-Api-Key": "relay-key"})
        assert [event["event_id"] for event in remaining.json()["events"]] == ["evt-low"]


def test_assistant_relay_accepts_signed_deployment_heartbeat_and_exact_ack(
    tmp_path: Path,
) -> None:
    secret = "0123456789abcdef0123456789ABCDEF"
    bot_id = "crypto-momentum"
    app = create_relay_app(
        db_path=str(tmp_path / "relay.db"),
        shared_secrets={bot_id: secret},
        api_key="relay-key",
    )
    envelope = deployment_heartbeat_envelope(
        bot_id=bot_id,
        runtime_instance_id="runtime-1",
        effective_config_hash="a" * 64,
        deployment_id="deploy-1",
        source="assistant-test",
        exchange_timestamp="2026-07-08T12:00:00Z",
    )
    body = canonical_relay_body(envelope)
    event_id = str(envelope["events"][0]["event_id"])

    with TestClient(app) as client:
        ingest = client.post(
            "/events",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": relay_signature(body, secret),
            },
        )
        assert ingest.status_code == 200
        assert ingest.json() == {"accepted": 1, "duplicates": 0}

        pending = client.get(
            "/events",
            params={"priority_first": "true", "max_priority": 0, "bot_id": bot_id},
            headers={"X-Api-Key": "relay-key"},
        )
        assert pending.status_code == 200
        assert [event["event_id"] for event in pending.json()["events"]] == [event_id]

        ack = client.post(
            "/ack-exact",
            json={"event_ids": [event_id]},
            headers={"X-Api-Key": "relay-key"},
        )
        assert ack.status_code == 200
        assert ack.json()["acked_count"] == 1


def test_assistant_relay_api_key_uses_constant_time_compare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _compare(supplied: str, expected: str) -> bool:
        calls.append((supplied, expected))
        return False

    monkeypatch.setattr(relay_app.hmac, "compare_digest", _compare)
    app = create_relay_app(
        db_path=str(tmp_path / "relay.db"),
        shared_secrets={"bot-1": "0123456789abcdef0123456789ABCDEF"},
        api_key="relay-key",
    )

    with TestClient(app) as client:
        response = client.get("/events", headers={"X-Api-Key": "wrong-key"})

    assert response.status_code == 401
    assert calls == [("wrong-key", "relay-key")]


def test_assistant_relay_verifies_gzip_signature_against_uncompressed_body(
    tmp_path: Path,
) -> None:
    secret = "0123456789abcdef0123456789ABCDEF"
    bot_id = "ibkr"
    app = create_relay_app(
        db_path=str(tmp_path / "relay.db"),
        shared_secrets={bot_id: secret},
        api_key="relay-key",
    )
    envelope = deployment_heartbeat_envelope(
        bot_id=bot_id,
        runtime_instance_id="runtime-1",
        effective_config_hash="b" * 64,
        deployment_id="deploy-1",
        source="assistant-test",
        exchange_timestamp="2026-07-08T12:00:00Z",
    )
    body = canonical_relay_body(envelope)

    with TestClient(app) as client:
        ingest = client.post(
            "/events",
            content=gzip.compress(body),
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "X-Signature": relay_signature(body, secret),
            },
        )

    assert ingest.status_code == 200
    assert ingest.json() == {"accepted": 1, "duplicates": 0}


def test_assistant_relay_rejects_placeholder_shared_secret_in_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_RELAY_DEV", raising=False)

    with pytest.raises(RuntimeError, match="invalid paper/live secret"):
        create_relay_app(
            db_path=str(tmp_path / "relay.db"),
            shared_secrets={"bot1": "change-me"},
            api_key="0123456789abcdef0123456789ABCDEF",
        )

    assert os.environ["TRADING_MODE"] == "paper"


def test_assistant_relay_rejects_duplicate_shared_secrets_in_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_RELAY_DEV", raising=False)
    secret = "0123456789abcdef0123456789ABCDEF"

    with pytest.raises(RuntimeError, match="duplicates bot1"):
        create_relay_app(
            db_path=str(tmp_path / "relay.db"),
            shared_secrets={"bot1": secret, "bot2": secret},
            api_key="fedcba9876543210fedcba9876543210",
        )


def test_assistant_relay_treats_production_as_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.setenv("TRADING_ENV", "production")
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_RELAY_DEV", raising=False)

    with pytest.raises(RuntimeError, match="RELAY_API_KEY required"):
        create_relay_app(
            db_path=str(tmp_path / "relay.db"),
            shared_secrets={"bot1": "0123456789abcdef0123456789ABCDEF"},
            api_key="",
        )


def test_assistant_relay_ignores_empty_trading_mode_for_strict_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_MODE", "")
    monkeypatch.setenv("TRADING_ENV", "production")
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_RELAY_DEV", raising=False)

    with pytest.raises(RuntimeError, match="RELAY_API_KEY required"):
        create_relay_app(
            db_path=str(tmp_path / "relay.db"),
            shared_secrets={"bot1": "0123456789abcdef0123456789ABCDEF"},
            api_key="",
        )
