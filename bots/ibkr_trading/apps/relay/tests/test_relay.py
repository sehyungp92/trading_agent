"""Tests for the relay service."""
from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from relay.app import create_relay_app
from relay.auth import HMACAuth
from relay.db.store import EventStore
from relay.rate_limiter import RateLimiter


# --- EventStore tests ---

class TestEventStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test.db")
        self.store = EventStore(db_path=self.db_path)

    def test_insert_and_retrieve(self):
        events = [
            {"event_id": "e1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
            {"event_id": "e2", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ]
        result = self.store.insert_events(events)
        assert result["accepted"] == 2
        assert result["duplicates"] == 0

        fetched = self.store.get_events()
        assert len(fetched) == 2

    def test_duplicate_rejection(self):
        events = [{"event_id": "e1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"}]
        self.store.insert_events(events)
        result = self.store.insert_events(events)
        assert result["accepted"] == 0
        assert result["duplicates"] == 1

    def test_ack_removes_from_pending(self):
        self.store.insert_events([
            {"event_id": "e1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
            {"event_id": "e2", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        count = self.store.ack_up_to("e1")
        assert count == 1
        pending = self.store.get_events()
        assert len(pending) == 1
        assert pending[0]["event_id"] == "e2"

    def test_get_events_with_since(self):
        self.store.insert_events([
            {"event_id": "e1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
            {"event_id": "e2", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
            {"event_id": "e3", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        events = self.store.get_events(since="e1")
        assert len(events) == 2
        assert events[0]["event_id"] == "e2"

    def test_count_pending(self):
        self.store.insert_events([
            {"event_id": "e1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        assert self.store.count_pending() == 1
        self.store.ack_up_to("e1")
        assert self.store.count_pending() == 0

    def test_default_id_ordering_with_priority_opt_in(self):
        """Default reads stay ID ordered for watermark ack; priority is opt-in."""
        events = [
            {"event_id": "low", "bot_id": "bot1", "event_type": "post_exit", "payload": "{}", "priority": 4},
            {"event_id": "high", "bot_id": "bot1", "event_type": "error", "payload": "{}", "priority": 1},
            {"event_id": "mid", "bot_id": "bot1", "event_type": "trade", "payload": "{}", "priority": 3},
        ]
        self.store.insert_events(events)
        fetched = self.store.get_events()
        assert [event["event_id"] for event in fetched] == ["low", "high", "mid"]

        fetched = self.store.get_events(priority_first=True)
        assert fetched[0]["event_id"] == "high"
        assert fetched[1]["event_id"] == "mid"
        assert fetched[2]["event_id"] == "low"

    def test_purge_acked_deletes_old(self):
        from datetime import timedelta
        self.store.insert_events([
            {"event_id": "old1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        self.store.ack_up_to("old1")
        # Manually backdate the received_at to 30 days ago
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        conn.execute("UPDATE events SET received_at = ? WHERE event_id = 'old1'", (old_ts,))
        conn.commit()
        conn.close()
        deleted = self.store.purge_acked(days=7)
        assert deleted == 1

    def test_purge_keeps_recent(self):
        self.store.insert_events([
            {"event_id": "recent1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        self.store.ack_up_to("recent1")
        deleted = self.store.purge_acked(days=7)
        assert deleted == 0

    def test_purge_ignores_unacked(self):
        from datetime import timedelta
        self.store.insert_events([
            {"event_id": "unacked1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        # Backdate but don't ack
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        conn.execute("UPDATE events SET received_at = ? WHERE event_id = 'unacked1'", (old_ts,))
        conn.commit()
        conn.close()
        deleted = self.store.purge_acked(days=7)
        assert deleted == 0

    def test_get_stats_empty_db(self):
        stats = self.store.get_stats()
        assert stats["per_bot_pending"] == {}
        assert stats["last_event_per_bot"] == {}
        assert stats["oldest_pending_age_seconds"] == 0
        assert stats["db_size_bytes"] > 0

    def test_get_stats_with_events(self):
        self.store.insert_events([
            {"event_id": "s1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
            {"event_id": "s2", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
            {"event_id": "s3", "bot_id": "bot2", "event_type": "trade", "payload": "{}"},
        ])
        stats = self.store.get_stats()
        assert stats["per_bot_pending"]["bot1"] == 2
        assert stats["per_bot_pending"]["bot2"] == 1
        assert "bot1" in stats["last_event_per_bot"]
        assert "bot2" in stats["last_event_per_bot"]
        assert stats["oldest_pending_age_seconds"] >= 0
        assert stats["db_size_bytes"] > 0

    def test_purge_stale_unacked_deletes_old(self):
        from datetime import timedelta
        import sqlite3
        self.store.insert_events([
            {"event_id": "stale1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        # Backdate to 10 days ago (no ack — simulates missing consumer)
        conn = sqlite3.connect(self.db_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        conn.execute("UPDATE events SET received_at = ? WHERE event_id = 'stale1'", (old_ts,))
        conn.commit()
        conn.close()
        deleted = self.store.purge_stale_unacked(days=3)
        assert deleted == 1
        assert self.store.count_pending() == 0

    def test_purge_stale_unacked_keeps_recent(self):
        self.store.insert_events([
            {"event_id": "fresh1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        deleted = self.store.purge_stale_unacked(days=3)
        assert deleted == 0
        assert self.store.count_pending() == 1

    def test_purge_stale_unacked_ignores_acked(self):
        from datetime import timedelta
        import sqlite3
        self.store.insert_events([
            {"event_id": "acked1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        self.store.ack_up_to("acked1")
        conn = sqlite3.connect(self.db_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        conn.execute("UPDATE events SET received_at = ? WHERE event_id = 'acked1'", (old_ts,))
        conn.commit()
        conn.close()
        deleted = self.store.purge_stale_unacked(days=3)
        assert deleted == 0

    def test_purge_vacuum_false_skips_vacuum(self):
        """vacuum=False should still delete but not crash."""
        from datetime import timedelta
        import sqlite3
        self.store.insert_events([
            {"event_id": "v1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
        ])
        self.store.ack_up_to("v1")
        conn = sqlite3.connect(self.db_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        conn.execute("UPDATE events SET received_at = ? WHERE event_id = 'v1'", (old_ts,))
        conn.commit()
        conn.close()
        deleted = self.store.purge_acked(days=7, vacuum=False)
        assert deleted == 1

    def test_filter_by_bot_id(self):
        self.store.insert_events([
            {"event_id": "e1", "bot_id": "bot1", "event_type": "trade", "payload": "{}"},
            {"event_id": "e2", "bot_id": "bot2", "event_type": "trade", "payload": "{}"},
        ])
        events = self.store.get_events(bot_id="bot1")
        assert len(events) == 1
        assert events[0]["bot_id"] == "bot1"


# --- HMACAuth tests ---

class TestHMACAuth:
    def test_disabled_when_no_secrets(self):
        auth = HMACAuth()
        assert not auth.enabled
        assert auth.verify(b"anything", "bad", "bot1") is True

    def test_valid_signature(self):
        secret = "test-secret"
        auth = HMACAuth({"bot1": secret})
        body = json.dumps({"test": True}, sort_keys=True).encode()
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert auth.verify(body, sig, "bot1") is True

    def test_invalid_signature(self):
        auth = HMACAuth({"bot1": "real-secret"})
        body = b'{"test": true}'
        assert auth.verify(body, "badsignature", "bot1") is False

    def test_unknown_bot_id(self):
        auth = HMACAuth({"bot1": "secret"})
        assert auth.verify(b"body", "sig", "unknown_bot") is False


# --- RateLimiter tests ---

class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("bot1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("bot1") is True
        assert limiter.is_allowed("bot1") is True
        assert limiter.is_allowed("bot1") is False

    def test_separate_per_bot(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("bot1") is True
        assert limiter.is_allowed("bot2") is True
        assert limiter.is_allowed("bot1") is False

    def test_remaining(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.remaining("bot1") == 5
        limiter.is_allowed("bot1")
        assert limiter.remaining("bot1") == 4


# --- FastAPI integration tests ---

class TestRelayAPI:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test.db")
        self.secret = "test-secret-key"
        app = create_relay_app(
            db_path=self.db_path,
            shared_secrets={"test_bot": self.secret},
        )
        self.client = TestClient(app)

    def _sign_and_post(self, payload: dict) -> "TestClient":
        body = json.dumps(payload, sort_keys=True)
        sig = hmac.new(self.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        return self.client.post(
            "/events",
            content=body,
            headers={"Content-Type": "application/json", "X-Signature": sig},
        )

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ingest_with_valid_signature(self):
        payload = {
            "bot_id": "test_bot",
            "events": [{
                "event_id": "evt-001",
                "bot_id": "test_bot",
                "event_type": "trade",
                "payload": "{}",
                "exchange_timestamp": "2026-03-02T00:00:00Z",
            }],
        }
        resp = self._sign_and_post(payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 1
        assert data["duplicates"] == 0

    def test_ingest_rejects_bad_signature(self):
        payload = {
            "bot_id": "test_bot",
            "events": [],
        }
        resp = self.client.post(
            "/events",
            json=payload,
            headers={"X-Signature": "badsig"},
        )
        assert resp.status_code == 401

    def test_ingest_duplicate_rejected(self):
        payload = {
            "bot_id": "test_bot",
            "events": [{
                "event_id": "evt-dup",
                "bot_id": "test_bot",
                "event_type": "trade",
                "payload": "{}",
            }],
        }
        self._sign_and_post(payload)
        resp = self._sign_and_post(payload)
        assert resp.status_code == 200
        assert resp.json()["duplicates"] == 1

    def test_get_events(self):
        payload = {
            "bot_id": "test_bot",
            "events": [{
                "event_id": "evt-get-1",
                "bot_id": "test_bot",
                "event_type": "trade",
                "payload": "{}",
            }],
        }
        self._sign_and_post(payload)
        resp = self.client.get("/events")
        assert resp.status_code == 200
        events = resp.json()["events"]
        assert len(events) == 1
        assert events[0]["event_id"] == "evt-get-1"

    def test_get_events_with_since(self):
        for i in range(3):
            payload = {
                "bot_id": "test_bot",
                "events": [{
                    "event_id": f"evt-since-{i}",
                    "bot_id": "test_bot",
                    "event_type": "trade",
                    "payload": "{}",
                }],
            }
            self._sign_and_post(payload)

        resp = self.client.get("/events?since=evt-since-0")
        events = resp.json()["events"]
        assert len(events) == 2
        assert events[0]["event_id"] == "evt-since-1"

    def test_ack_events(self):
        payload = {
            "bot_id": "test_bot",
            "events": [
                {"event_id": "evt-ack-1", "bot_id": "test_bot", "event_type": "trade", "payload": "{}"},
                {"event_id": "evt-ack-2", "bot_id": "test_bot", "event_type": "trade", "payload": "{}"},
            ],
        }
        self._sign_and_post(payload)

        resp = self.client.post("/ack", json={"watermark": "evt-ack-1"})
        assert resp.status_code == 200
        assert resp.json()["acked_count"] == 1

        # Verify only unacked events remain
        resp = self.client.get("/events")
        events = resp.json()["events"]
        assert len(events) == 1
        assert events[0]["event_id"] == "evt-ack-2"

    def test_rate_limiting(self):
        # Create app with very low rate limit
        app = create_relay_app(
            db_path=self.db_path,
            shared_secrets={"test_bot": self.secret},
            max_requests_per_minute=2,
        )
        client = TestClient(app)

        payload = {"bot_id": "test_bot", "events": []}
        body = json.dumps(payload, sort_keys=True)
        sig = hmac.new(self.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        headers = {"Content-Type": "application/json", "X-Signature": sig}

        # First two should succeed
        assert client.post("/events", content=body, headers=headers).status_code == 200
        assert client.post("/events", content=body, headers=headers).status_code == 200
        # Third should be rate limited
        assert client.post("/events", content=body, headers=headers).status_code == 429

    def test_admin_purge_endpoint(self):
        resp = self.client.post("/admin/purge?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "deleted" in data

    def test_admin_purge_stale_endpoint(self):
        resp = self.client.post("/admin/purge-stale?days=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "deleted" in data
        assert data["retention_days"] == 3

    def test_ingest_gzip(self):
        """Relay should accept gzip-compressed event batches."""
        import gzip as gzip_mod
        payload = {
            "bot_id": "test_bot",
            "events": [{
                "event_id": "evt-gzip-1",
                "bot_id": "test_bot",
                "event_type": "trade",
                "payload": "{}",
                "exchange_timestamp": "2026-03-02T00:00:00Z",
                "priority": 3,
            }],
        }
        body = json.dumps(payload, sort_keys=True)
        sig = hmac.new(self.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        compressed = gzip_mod.compress(body.encode())
        resp = self.client.post(
            "/events",
            content=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "X-Signature": sig,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 1

    def test_priority_in_event_model(self):
        payload = {
            "bot_id": "test_bot",
            "events": [{
                "event_id": "evt-priority-1",
                "bot_id": "test_bot",
                "event_type": "error",
                "payload": "{}",
                "priority": 1,
            }],
        }
        resp = self._sign_and_post(payload)
        assert resp.status_code == 200

    def test_health_enriched_fields(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        expected_keys = {
            "status", "pending_events", "per_bot_pending",
            "last_event_per_bot", "oldest_pending_age_seconds",
            "db_size_bytes", "uptime_seconds",
        }
        assert set(data.keys()) == expected_keys
        assert isinstance(data["per_bot_pending"], dict)
        assert isinstance(data["last_event_per_bot"], dict)
        assert isinstance(data["oldest_pending_age_seconds"], (int, float))
        assert isinstance(data["db_size_bytes"], int)
        assert isinstance(data["uptime_seconds"], (int, float))

    def test_health_per_bot_counts(self):
        for i, bot in enumerate(["test_bot", "test_bot", "other_bot"]):
            payload = {
                "bot_id": bot,
                "events": [{
                    "event_id": f"evt-hpc-{i}",
                    "bot_id": bot,
                    "event_type": "trade",
                    "payload": "{}",
                }],
            }
            body = json.dumps(payload, sort_keys=True)
            secret = self.secret if bot == "test_bot" else "other-secret"
            sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            self.client.post(
                "/events", content=body,
                headers={"Content-Type": "application/json", "X-Signature": sig},
            )
        resp = self.client.get("/health")
        data = resp.json()
        # test_bot has valid HMAC so its 2 events are accepted;
        # other_bot has wrong secret so its event is rejected (401)
        assert data["per_bot_pending"].get("test_bot", 0) == 2

    def test_health_empty_db_zeros(self):
        resp = self.client.get("/health")
        data = resp.json()
        assert data["per_bot_pending"] == {}
        assert data["oldest_pending_age_seconds"] == 0
        assert data["pending_events"] == 0

    def test_health_uptime_present(self):
        resp = self.client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] >= 0

    def test_empty_events_list(self):
        payload = {"bot_id": "test_bot", "events": []}
        resp = self._sign_and_post(payload)
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 0

    def test_bot_id_mismatch_rejected(self):
        """Events with bot_id differing from envelope should be rejected."""
        payload = {
            "bot_id": "test_bot",
            "events": [{
                "event_id": "evt-mismatch-1",
                "bot_id": "other_bot",
                "event_type": "trade",
                "payload": "{}",
            }],
        }
        resp = self._sign_and_post(payload)
        assert resp.status_code == 400
        assert "doesn't match" in resp.text


class TestRelayAPIKeyAuth:
    """Tests for API key authentication on read/admin endpoints."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test.db")
        self.secret = "test-secret-key"
        self.api_key = "test-read-api-key"
        app = create_relay_app(
            db_path=self.db_path,
            shared_secrets={"test_bot": self.secret},
            api_key=self.api_key,
        )
        self.client = TestClient(app)

    def _sign_and_post(self, payload: dict):
        body = json.dumps(payload, sort_keys=True)
        sig = hmac.new(self.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        return self.client.post(
            "/events",
            content=body,
            headers={"Content-Type": "application/json", "X-Signature": sig},
        )

    def _seed_event(self):
        payload = {
            "bot_id": "test_bot",
            "events": [{"event_id": "evt-auth-1", "bot_id": "test_bot",
                         "event_type": "trade", "payload": "{}"}],
        }
        self._sign_and_post(payload)

    def test_get_events_without_key_returns_401(self):
        resp = self.client.get("/events")
        assert resp.status_code == 401

    def test_get_events_with_wrong_key_returns_401(self):
        resp = self.client.get("/events", headers={"X-Api-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_get_events_with_valid_key(self):
        self._seed_event()
        resp = self.client.get("/events", headers={"X-Api-Key": self.api_key})
        assert resp.status_code == 200
        assert len(resp.json()["events"]) == 1

    def test_ack_without_key_returns_401(self):
        resp = self.client.post("/ack", json={"watermark": "evt-auth-1"})
        assert resp.status_code == 401

    def test_ack_with_valid_key(self):
        self._seed_event()
        resp = self.client.post(
            "/ack",
            json={"watermark": "evt-auth-1"},
            headers={"X-Api-Key": self.api_key},
        )
        assert resp.status_code == 200

    def test_admin_purge_without_key_returns_401(self):
        resp = self.client.post("/admin/purge?days=7")
        assert resp.status_code == 401

    def test_admin_purge_with_valid_key(self):
        resp = self.client.post(
            "/admin/purge?days=7",
            headers={"X-Api-Key": self.api_key},
        )
        assert resp.status_code == 200

    def test_admin_purge_stale_without_key_returns_401(self):
        resp = self.client.post("/admin/purge-stale?days=3")
        assert resp.status_code == 401

    def test_admin_purge_stale_with_valid_key(self):
        resp = self.client.post(
            "/admin/purge-stale?days=3",
            headers={"X-Api-Key": self.api_key},
        )
        assert resp.status_code == 200

    def test_health_no_key_required(self):
        """Health endpoint should remain open regardless of api_key config."""
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ingest_not_affected_by_api_key(self):
        """POST /events uses HMAC auth, not API key."""
        payload = {
            "bot_id": "test_bot",
            "events": [{"event_id": "evt-api-1", "bot_id": "test_bot",
                         "event_type": "trade", "payload": "{}"}],
        }
        resp = self._sign_and_post(payload)
        assert resp.status_code == 200


class TestStringPriorityCoercion:
    """k_stock_trader sends string priorities ('high','normal','low').
    The relay must coerce them to integers."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test.db")
        app = create_relay_app(db_path=self.db_path)
        self.client = TestClient(app)

    def _post(self, payload: dict):
        body = json.dumps(payload, sort_keys=True)
        return self.client.post(
            "/events",
            content=body,
            headers={"Content-Type": "application/json"},
        )

    def test_string_priority_high(self):
        payload = {
            "bot_id": "k_bot",
            "events": [{"event_id": "sp-1", "bot_id": "k_bot",
                         "event_type": "bot_error", "payload": "{}",
                         "priority": "high"}],
        }
        resp = self._post(payload)
        assert resp.status_code == 200
        events = self.client.get("/events").json()["events"]
        assert events[0]["priority"] == 1

    def test_string_priority_normal(self):
        payload = {
            "bot_id": "k_bot",
            "events": [{"event_id": "sp-2", "bot_id": "k_bot",
                         "event_type": "trade", "payload": "{}",
                         "priority": "normal"}],
        }
        resp = self._post(payload)
        assert resp.status_code == 200
        events = self.client.get("/events").json()["events"]
        assert events[0]["priority"] == 3

    def test_string_priority_low(self):
        payload = {
            "bot_id": "k_bot",
            "events": [{"event_id": "sp-3", "bot_id": "k_bot",
                         "event_type": "heartbeat", "payload": "{}",
                         "priority": "low"}],
        }
        resp = self._post(payload)
        assert resp.status_code == 200
        events = self.client.get("/events").json()["events"]
        assert events[0]["priority"] == 4

    def test_string_priority_critical(self):
        payload = {
            "bot_id": "k_bot",
            "events": [{"event_id": "sp-4", "bot_id": "k_bot",
                         "event_type": "error", "payload": "{}",
                         "priority": "critical"}],
        }
        resp = self._post(payload)
        assert resp.status_code == 200
        events = self.client.get("/events").json()["events"]
        assert events[0]["priority"] == 0

    def test_int_priority_still_works(self):
        payload = {
            "bot_id": "k_bot",
            "events": [{"event_id": "sp-5", "bot_id": "k_bot",
                         "event_type": "trade", "payload": "{}",
                         "priority": 2}],
        }
        resp = self._post(payload)
        assert resp.status_code == 200
        events = self.client.get("/events").json()["events"]
        assert events[0]["priority"] == 2

    def test_unknown_string_priority_defaults_to_3(self):
        payload = {
            "bot_id": "k_bot",
            "events": [{"event_id": "sp-6", "bot_id": "k_bot",
                         "event_type": "trade", "payload": "{}",
                         "priority": "medium"}],
        }
        resp = self._post(payload)
        assert resp.status_code == 200
        events = self.client.get("/events").json()["events"]
        assert events[0]["priority"] == 3

    def test_mixed_priority_ordering_is_opt_in(self):
        """Events from different bots sort by priority only when requested."""
        for eid, prio in [("m1", "low"), ("m2", 1), ("m3", "high"), ("m4", "normal")]:
            payload = {
                "bot_id": "k_bot",
                "events": [{"event_id": eid, "bot_id": "k_bot",
                             "event_type": "trade", "payload": "{}",
                             "priority": prio}],
            }
            self._post(payload)
        events = self.client.get("/events").json()["events"]
        assert [e["event_id"] for e in events] == ["m1", "m2", "m3", "m4"]

        resp = self.client.get("/events?priority_first=true")
        payload = resp.json()
        assert payload["delivery_mode"] == "priority_first"
        assert payload["ack_mode"] == "exact"
        events = payload["events"]
        priorities = [e["priority"] for e in events]
        assert priorities == sorted(priorities)
