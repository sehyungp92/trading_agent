"""FastAPI relay service — receives events from sidecar, serves to trading assistant."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from crypto_trader.relay.auth import RelayAuth
from crypto_trader.relay.store import RelayStore

# FastAPI is an optional dependency — only imported when running the relay
try:
    from fastapi import FastAPI, Header, HTTPException, Request, Response
except ImportError:
    FastAPI = None  # type: ignore


def create_app(
    db_path: Path | str = "relay_events.db",
    bot_secrets: dict[str, str] | None = None,
) -> Any:
    """Create the FastAPI relay application.

    Args:
        db_path: Path to SQLite database file.
        bot_secrets: dict of bot_id -> shared_secret for HMAC auth.
    """
    if FastAPI is None:
        raise ImportError("fastapi is required: pip install fastapi uvicorn")

    app = FastAPI(title="Trading Relay", version="1.0.0")
    store = RelayStore(db_path)
    auth = RelayAuth(bot_secrets or {})

    @app.post("/events")
    async def receive_events(request: Request) -> dict:
        """Receive events from sidecar bots."""
        bot_id = request.headers.get("X-Bot-Id", "")
        signature = request.headers.get("X-Signature", "")

        # Read body (handle gzip)
        body = await request.body()
        if request.headers.get("Content-Encoding") == "gzip":
            try:
                body = gzip.decompress(body)
            except Exception:
                raise HTTPException(400, "Invalid gzip body")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON")

        # Verify HMAC
        if not auth.verify(bot_id, payload, signature):
            raise HTTPException(401, "Invalid signature")

        event_type = payload.get("event_type", "unknown")
        events = payload.get("events", [])

        if not isinstance(events, list):
            raise HTTPException(400, "events must be a list")

        inserted = store.insert_events(bot_id, event_type, events)

        return {"status": "ok", "inserted": inserted, "duplicates": len(events) - inserted}

    @app.get("/events")
    async def get_events(
        since: int = 0,
        limit: int = 100,
        bot_id: str | None = None,
    ) -> dict:
        """Get pending events for the trading assistant."""
        events = store.get_events(since_id=since, limit=min(limit, 500), bot_id=bot_id)
        return {"events": events, "count": len(events)}

    @app.post("/ack")
    async def ack_events(request: Request) -> dict:
        """Acknowledge processed events."""
        body = await request.json()
        event_ids = body.get("event_ids", [])

        if not isinstance(event_ids, list):
            raise HTTPException(400, "event_ids must be a list")

        acked = store.ack_events(event_ids)
        return {"status": "ok", "acked": acked}

    @app.get("/health")
    async def health() -> dict:
        """Get relay health stats."""
        return store.get_health()

    @app.post("/admin/purge")
    async def purge(older_than_hours: int = 24) -> dict:
        """Purge old acked events."""
        deleted = store.purge_acked(older_than_hours)
        return {"status": "ok", "deleted": deleted}

    @app.on_event("shutdown")
    async def shutdown_store() -> None:
        store.close()

    return app
