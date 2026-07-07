"""FastAPI relay application that buffers events from trading bots."""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request, Response
from pydantic import BaseModel, field_validator
from starlette.middleware.gzip import GZipMiddleware

from .auth import HMACAuth
from .db.store import EventStore
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# --- Request / Response models ---

_PRIORITY_STRINGS = {"critical": 0, "high": 1, "normal": 3, "low": 4}


class EventIn(BaseModel):
    event_id: str
    bot_id: str
    event_type: str = "unknown"
    payload: str = "{}"
    exchange_timestamp: str = ""
    priority: int = 3

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, v: int | str) -> int:
        """Accept both integer priorities and string labels (high/normal/low)."""
        if isinstance(v, str):
            return _PRIORITY_STRINGS.get(v.lower(), 3)
        return v


class IngestRequest(BaseModel):
    bot_id: str
    events: list[EventIn]


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int


class AckRequest(BaseModel):
    watermark: str


class AckExactRequest(BaseModel):
    event_ids: list[str]


class AckResponse(BaseModel):
    status: str
    watermark: str
    acked_count: int = 0


# --- App factory ---

def create_relay_app(
    db_path: str = "data/relay.db",
    shared_secrets: dict[str, str] | None = None,
    max_requests_per_minute: int = 60,
    api_key: str = "",
) -> FastAPI:
    """Create and configure the relay FastAPI app.

    Args:
        api_key: Shared API key required for GET /events, POST /ack,
                 and POST /admin/purge. Empty string disables auth
                 on these endpoints (dev mode).
    """

    import time as _time
    _start_mono = _time.monotonic()

    trading_env = os.environ.get("TRADING_MODE", os.environ.get("TRADING_ENV", "dev"))
    # RELAY-2: in paper/live, missing API key or empty shared secrets is a
    # hard fail. The previous warn-and-continue path was a silent
    # production-security gap. Operators who genuinely need an
    # unauthenticated relay (local dev only) must opt in explicitly.
    _allow_unauth = os.environ.get("ALLOW_UNAUTHENTICATED_RELAY_DEV") == "1"
    if not api_key and trading_env in ("paper", "live") and not _allow_unauth:
        raise RuntimeError(
            f"RELAY_API_KEY required in {trading_env} mode. Set the env var, "
            f"or set ALLOW_UNAUTHENTICATED_RELAY_DEV=1 for local development."
        )
    elif not api_key:
        logger.warning("No api_key configured; read/admin endpoints are unauthenticated")

    store = EventStore(db_path=db_path)
    auth = HMACAuth(shared_secrets=shared_secrets)
    if not auth.enabled and trading_env in ("paper", "live") and not _allow_unauth:
        raise RuntimeError(
            f"RELAY_SHARED_SECRETS required in {trading_env} mode. Set a JSON "
            f"object mapping bot_id -> secret, or set "
            f"ALLOW_UNAUTHENTICATED_RELAY_DEV=1 for local development."
        )
    limiter = RateLimiter(max_requests=max_requests_per_minute)

    # RELAY-1: stale-unacked window is configurable. Default 14d covers
    # multi-day assistant outages that the previous 3d window would have
    # silently dropped. Acked-purge stays at 7d since acked rows are
    # already replicated downstream.
    _purge_days = int(os.environ.get("RELAY_PURGE_DAYS", "14"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: purge old acked + stale unacked events, single VACUUM
        try:
            purged = store.purge_acked(days=7, vacuum=False)
            purged_stale = store.purge_stale_unacked(days=_purge_days, vacuum=False)
            if purged or purged_stale:
                store.vacuum()
                logger.info(
                    "Startup purge: removed %d acked, %d stale unacked events "
                    "(stale window=%dd)",
                    purged, purged_stale, _purge_days,
                )
        except Exception as e:
            logger.warning("Startup purge failed: %s", e)

        # Periodic daily purge
        async def _periodic_purge():
            while True:
                await asyncio.sleep(86400)  # 24 hours
                try:
                    a = store.purge_acked(days=7, vacuum=False)
                    s = store.purge_stale_unacked(days=_purge_days, vacuum=False)
                    if a or s:
                        store.vacuum()
                except Exception as e:
                    logger.warning("Periodic purge failed: %s", e)

        task = asyncio.create_task(_periodic_purge())
        yield
        task.cancel()

    app = FastAPI(title="Trading Relay", version="1.0.0", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=500)

    def _check_api_key(key: str) -> Response | None:
        """Return a 401 Response if api_key is configured and doesn't match."""
        if api_key and key != api_key:
            return Response(status_code=401, content="Invalid API key")

    @app.post("/events", response_model=IngestResponse)
    async def ingest_events(
        request: Request,
        x_signature: str = Header(default=""),
    ):
        """Receive HMAC-signed event batches from trading bots."""
        raw_body = await request.body()

        # Decompress gzip if Content-Encoding header present
        content_encoding = request.headers.get("content-encoding", "")
        if content_encoding == "gzip":
            try:
                body = gzip.decompress(raw_body)
            except Exception:
                return Response(status_code=400, content="Invalid gzip data")
        else:
            body = raw_body

        # Parse body to get bot_id for auth lookup
        try:
            data = json.loads(body)
            bot_id = data.get("bot_id", "")
        except Exception:
            return Response(status_code=400, content="Invalid JSON")

        # HMAC verification (always against decompressed canonical JSON)
        if auth.enabled and not auth.verify(body, x_signature, bot_id):
            return Response(status_code=401, content="Invalid signature")

        # Rate limiting
        if not limiter.is_allowed(bot_id):
            return Response(
                status_code=429,
                content="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )

        # Parse and store events
        try:
            ingest = IngestRequest(**data)
            # Enforce that per-event bot_id matches envelope bot_id
            for evt in ingest.events:
                if evt.bot_id != ingest.bot_id:
                    return Response(
                        status_code=400,
                        content=f"Event {evt.event_id} bot_id '{evt.bot_id}' "
                                f"doesn't match envelope bot_id '{ingest.bot_id}'",
                    )
            events = [e.model_dump() for e in ingest.events]
            result = store.insert_events(events)
            logger.info(
                "Ingested from %s: %d accepted, %d duplicates",
                bot_id, result["accepted"], result["duplicates"],
            )
            return IngestResponse(**result)
        except Exception as e:
            logger.error("Ingest error: %s", e)
            return Response(status_code=400, content=str(e))

    @app.get("/events", response_model=None)
    async def get_events(
        since: str | None = None,
        limit: int = 100,
        bot_id: str | None = None,
        min_priority: int | None = None,
        max_priority: int | None = None,
        priority_first: bool = False,
        x_api_key: str = Header(default=""),
    ):
        """Pull un-acked events (used by home orchestrator).

        Default delivery is id ordered for watermark ack safety. Use
        `priority_first=true` with `max_priority` and `/ack-exact` to drain
        urgent events first; `min_priority` remains a deprecated alias for
        `max_priority`.
        """
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        events = store.get_events(
            since=since,
            limit=min(limit, 1000),
            bot_id=bot_id,
            min_priority=min_priority,
            max_priority=max_priority,
            priority_first=priority_first,
        )
        return {
            "events": events,
            "delivery_mode": "priority_first" if priority_first else "id",
            "ack_mode": "exact" if priority_first else "watermark",
        }

    @app.post("/ack", response_model=None)
    async def ack_events(req: AckRequest, x_api_key: str = Header(default="")):
        """Acknowledge events up to a watermark (used by home orchestrator)."""
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        count = store.ack_up_to(req.watermark)
        return AckResponse(status="ok", watermark=req.watermark, acked_count=count)

    @app.post("/ack-exact", response_model=None)
    async def ack_events_exact(req: AckExactRequest, x_api_key: str = Header(default="")):
        """Acknowledge exactly the listed event IDs."""
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        count = store.ack_exact(req.event_ids)
        return {"status": "ok", "acked_count": count}

    @app.get("/health")
    async def health():
        """Health check endpoint with enriched stats."""
        pending = store.count_pending()
        stats = store.get_stats()
        uptime = _time.monotonic() - _start_mono
        return {
            "status": "ok",
            "pending_events": pending,
            "per_bot_pending": stats["per_bot_pending"],
            "last_event_per_bot": stats["last_event_per_bot"],
            "oldest_pending_age_seconds": stats["oldest_pending_age_seconds"],
            "db_size_bytes": stats["db_size_bytes"],
            "uptime_seconds": round(uptime, 1),
        }

    @app.post("/admin/purge", response_model=None)
    async def admin_purge(days: int = 7, x_api_key: str = Header(default="")):
        """Purge acked events older than N days."""
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        deleted = store.purge_acked(days=days)
        return {"status": "ok", "deleted": deleted, "retention_days": days}

    @app.post("/admin/purge-stale", response_model=None)
    async def admin_purge_stale(days: int = _purge_days, x_api_key: str = Header(default="")):
        """Purge unacked events older than N days."""
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        deleted = store.purge_stale_unacked(days=days)
        return {"status": "ok", "deleted": deleted, "retention_days": days}

    return app


# Module-level instance for uvicorn (CMD: uvicorn apps.relay.app:app)

app = create_relay_app(
    db_path=os.environ.get("RELAY_DB_PATH", "data/relay.db"),
    shared_secrets=json.loads(os.environ.get("RELAY_SHARED_SECRETS", "{}")),
    api_key=os.environ.get("RELAY_API_KEY", ""),
)
