"""Assistant-owned FastAPI relay ingress for bot event batches."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request, Response
from pydantic import BaseModel, field_validator
from starlette.middleware.gzip import GZipMiddleware
from trading_contracts.relay_acceptance import (
    contains_placeholder,
    secret_fingerprint,
    validate_hmac_secret,
)

logger = logging.getLogger(__name__)

_PRIORITY_STRINGS = {"critical": 0, "high": 1, "normal": 3, "low": 4}
_STRICT_RELAY_ENVS = {"paper", "live", "prod", "production"}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    bot_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    exchange_timestamp TEXT,
    received_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    acked INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 3
);
CREATE INDEX IF NOT EXISTS idx_events_acked ON events(acked);
CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority);
CREATE INDEX IF NOT EXISTS idx_events_bot_id ON events(bot_id);
CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_received ON events(received_at);
"""


class EventIn(BaseModel):
    event_id: str
    bot_id: str
    event_type: str = "unknown"
    payload: str = "{}"
    exchange_timestamp: str = ""
    priority: int = 3

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, value: int | str) -> int:
        if isinstance(value, str):
            return _PRIORITY_STRINGS.get(value.lower(), 3)
        return value


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


class HMACAuth:
    def __init__(self, shared_secrets: dict[str, str] | None = None) -> None:
        self._secrets = shared_secrets or {}
        if not self._secrets:
            logger.warning("No shared relay secrets configured; HMAC auth disabled")

    @property
    def enabled(self) -> bool:
        return bool(self._secrets)

    def verify(self, body: bytes, signature: str, bot_id: str) -> bool:
        if not self.enabled:
            return True
        secret = self._secrets.get(bot_id)
        if not secret:
            logger.warning("Unknown relay bot_id: %s", bot_id)
            return False
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, bot_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        recent = [item for item in self._requests.get(bot_id, []) if item > cutoff]
        if len(recent) >= self._max_requests:
            self._requests[bot_id] = recent
            return False
        recent.append(now)
        self._requests[bot_id] = recent
        return True


class RelayStore:
    def __init__(self, db_path: str = "data/relay.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()

    def insert_events(self, events: list[dict[str, Any]]) -> dict[str, int]:
        accepted = 0
        duplicates = 0
        with self._connect() as conn:
            for event in events:
                try:
                    conn.execute(
                        """
                        INSERT INTO events
                            (event_id, bot_id, event_type, payload,
                             exchange_timestamp, received_at, priority)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event["event_id"],
                            event["bot_id"],
                            event.get("event_type", "unknown"),
                            event.get("payload", "{}"),
                            event.get("exchange_timestamp", ""),
                            datetime.now(timezone.utc).isoformat(),
                            event.get("priority", 3),
                        ),
                    )
                    accepted += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
            conn.commit()
        return {"accepted": accepted, "duplicates": duplicates}

    def get_events(
        self,
        *,
        since: str | None = None,
        limit: int = 100,
        bot_id: str | None = None,
        min_priority: int | None = None,
        max_priority: int | None = None,
        priority_first: bool = False,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            min_id = 0
            if since:
                row = conn.execute("SELECT id FROM events WHERE event_id = ?", (since,)).fetchone()
                min_id = int(row["id"]) if row else 0
            query = "SELECT * FROM events WHERE acked = 0 AND id > ?"
            params: list[Any] = [min_id]
            if bot_id:
                query += " AND bot_id = ?"
                params.append(bot_id)
            effective_max_priority = max_priority if max_priority is not None else min_priority
            if effective_max_priority is not None:
                query += " AND priority <= ?"
                params.append(int(effective_max_priority))
            query += " ORDER BY priority ASC, id ASC LIMIT ?" if priority_first else " ORDER BY id ASC LIMIT ?"
            params.append(limit)
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def ack_up_to(self, watermark_event_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM events WHERE event_id = ?",
                (watermark_event_id,),
            ).fetchone()
            if not row:
                return 0
            cursor = conn.execute(
                "UPDATE events SET acked = 1 WHERE id <= ? AND acked = 0",
                (row["id"],),
            )
            conn.commit()
            return int(cursor.rowcount)

    def ack_exact(self, event_ids: list[str]) -> int:
        if not event_ids:
            return 0
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in event_ids)
            cursor = conn.execute(
                f"UPDATE events SET acked = 1 WHERE event_id IN ({placeholders}) AND acked = 0",
                event_ids,
            )
            conn.commit()
            return int(cursor.rowcount)

    def count_pending(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM events WHERE acked = 0").fetchone()
            return int(row["count"]) if row else 0

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            per_bot = {
                str(row["bot_id"]): int(row["count"])
                for row in conn.execute(
                    "SELECT bot_id, COUNT(*) AS count FROM events "
                    "WHERE acked = 0 GROUP BY bot_id"
                ).fetchall()
            }
            last_event = {
                str(row["bot_id"]): str(row["last_ts"])
                for row in conn.execute(
                    "SELECT bot_id, MAX(received_at) AS last_ts FROM events GROUP BY bot_id"
                ).fetchall()
            }
            row = conn.execute(
                "SELECT MIN(received_at) AS oldest FROM events WHERE acked = 0"
            ).fetchone()
        oldest_age = 0.0
        if row and row["oldest"]:
            oldest_age = (
                datetime.now(timezone.utc) - datetime.fromisoformat(str(row["oldest"]))
            ).total_seconds()
        try:
            db_size = os.path.getsize(self._db_path)
        except OSError:
            db_size = 0
        return {
            "per_bot_pending": per_bot,
            "last_event_per_bot": last_event,
            "oldest_pending_age_seconds": round(max(oldest_age, 0.0), 1),
            "db_size_bytes": db_size,
        }

    def purge_acked(self, *, days: int = 7, vacuum: bool = True) -> int:
        return self._purge(acked=1, days=days, vacuum=vacuum)

    def purge_stale_unacked(self, *, days: int = 14, vacuum: bool = True) -> int:
        return self._purge(acked=0, days=days, vacuum=vacuum)

    def _purge(self, *, acked: int, days: int, vacuum: bool) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM events WHERE acked = ? AND received_at < ?",
                (acked, cutoff),
            )
            deleted = int(cursor.rowcount)
            conn.commit()
            if deleted and vacuum:
                conn.execute("VACUUM")
        return deleted

    def vacuum(self) -> None:
        with self._connect() as conn:
            conn.execute("VACUUM")


def create_relay_app(
    db_path: str = "data/relay.db",
    shared_secrets: dict[str, str] | None = None,
    max_requests_per_minute: int = 60,
    api_key: str = "",
) -> FastAPI:
    start_mono = time.monotonic()
    trading_env = (
        os.environ.get("TRADING_MODE") or os.environ.get("TRADING_ENV") or "dev"
    ).strip().lower()
    allow_unauth = os.environ.get("ALLOW_UNAUTHENTICATED_RELAY_DEV") == "1"
    strict_relay = trading_env in _STRICT_RELAY_ENVS
    if not api_key and strict_relay and not allow_unauth:
        raise RuntimeError(
            f"RELAY_API_KEY required in {trading_env} mode; set "
            "ALLOW_UNAUTHENTICATED_RELAY_DEV=1 only for local development."
        )
    if api_key and strict_relay and not allow_unauth:
        if contains_placeholder(api_key) or len(api_key.strip()) < 16:
            raise RuntimeError(
                f"RELAY_API_KEY must be non-placeholder and at least 16 characters in {trading_env} mode."
            )
    if shared_secrets and strict_relay and not allow_unauth:
        secret_errors: list[str] = []
        for bot_id, secret in shared_secrets.items():
            secret_errors.extend(
                f"{bot_id}: {error}"
                for error in validate_hmac_secret(secret, field_name="relay shared secret")
            )
        seen_secret_owner: dict[str, str] = {}
        for bot_id, secret in shared_secrets.items():
            normalized_secret = str(secret or "").strip()
            if not normalized_secret:
                continue
            previous_bot = seen_secret_owner.get(normalized_secret)
            if previous_bot:
                fingerprint = secret_fingerprint(normalized_secret)
                secret_errors.append(
                    f"{bot_id}: relay shared secret duplicates {previous_bot} "
                    f"(fingerprint={fingerprint})"
                )
            else:
                seen_secret_owner[normalized_secret] = bot_id
        if secret_errors:
            raise RuntimeError(
                f"RELAY_SHARED_SECRETS contains invalid paper/live secret(s): {'; '.join(secret_errors)}"
            )

    store = RelayStore(db_path=db_path)
    auth = HMACAuth(shared_secrets=shared_secrets)
    if not auth.enabled and strict_relay and not allow_unauth:
        raise RuntimeError(
            f"RELAY_SHARED_SECRETS required in {trading_env} mode; set "
            "ALLOW_UNAUTHENTICATED_RELAY_DEV=1 only for local development."
        )
    limiter = RateLimiter(max_requests=max_requests_per_minute)
    purge_days = int(os.environ.get("RELAY_PURGE_DAYS", "14"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        try:
            acked = store.purge_acked(days=7, vacuum=False)
            stale = store.purge_stale_unacked(days=purge_days, vacuum=False)
            if acked or stale:
                store.vacuum()
        except Exception:
            logger.warning("Relay startup purge failed", exc_info=True)

        async def _periodic_purge() -> None:
            while True:
                await asyncio.sleep(86400)
                try:
                    acked = store.purge_acked(days=7, vacuum=False)
                    stale = store.purge_stale_unacked(days=purge_days, vacuum=False)
                    if acked or stale:
                        store.vacuum()
                except Exception:
                    logger.warning("Relay periodic purge failed", exc_info=True)

        task = asyncio.create_task(_periodic_purge())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Trading Assistant Relay Ingress", version="1.0.0", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=500)

    def _check_api_key(key: str) -> Response | None:
        if api_key and not hmac.compare_digest(str(key), str(api_key)):
            return Response(status_code=401, content="Invalid API key")
        return None

    @app.post("/events", response_model=IngestResponse)
    async def ingest_events(request: Request, x_signature: str = Header(default="")):
        raw_body = await request.body()
        if request.headers.get("content-encoding", "") == "gzip":
            try:
                body = gzip.decompress(raw_body)
            except Exception:
                return Response(status_code=400, content="Invalid gzip data")
        else:
            body = raw_body
        try:
            data = json.loads(body)
            bot_id = str(data.get("bot_id", ""))
        except Exception:
            return Response(status_code=400, content="Invalid JSON")
        if auth.enabled and not auth.verify(body, x_signature, bot_id):
            return Response(status_code=401, content="Invalid signature")
        if not limiter.is_allowed(bot_id):
            return Response(
                status_code=429,
                content="Rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        try:
            ingest = IngestRequest(**data)
            for event in ingest.events:
                if event.bot_id != ingest.bot_id:
                    return Response(
                        status_code=400,
                        content=(
                            f"Event {event.event_id} bot_id '{event.bot_id}' "
                            f"doesn't match envelope bot_id '{ingest.bot_id}'"
                        ),
                    )
            return IngestResponse(**store.insert_events([e.model_dump() for e in ingest.events]))
        except Exception as exc:
            logger.error("Relay ingest error: %s", exc)
            return Response(status_code=400, content=str(exc))

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
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        return {
            "events": store.get_events(
                since=since,
                limit=min(limit, 1000),
                bot_id=bot_id,
                min_priority=min_priority,
                max_priority=max_priority,
                priority_first=priority_first,
            ),
            "delivery_mode": "priority_first" if priority_first else "id",
            "ack_mode": "exact" if priority_first else "watermark",
        }

    @app.post("/ack", response_model=None)
    async def ack_events(req: AckRequest, x_api_key: str = Header(default="")):
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        count = store.ack_up_to(req.watermark)
        return AckResponse(status="ok", watermark=req.watermark, acked_count=count)

    @app.post("/ack-exact", response_model=None)
    async def ack_events_exact(req: AckExactRequest, x_api_key: str = Header(default="")):
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        return {"status": "ok", "acked_count": store.ack_exact(req.event_ids)}

    @app.get("/health")
    async def health():
        stats = store.stats()
        return {
            "status": "ok",
            "pending_events": store.count_pending(),
            "per_bot_pending": stats["per_bot_pending"],
            "last_event_per_bot": stats["last_event_per_bot"],
            "oldest_pending_age_seconds": stats["oldest_pending_age_seconds"],
            "db_size_bytes": stats["db_size_bytes"],
            "uptime_seconds": round(time.monotonic() - start_mono, 1),
        }

    @app.post("/admin/purge", response_model=None)
    async def admin_purge(days: int = 7, x_api_key: str = Header(default="")):
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        return {"status": "ok", "deleted": store.purge_acked(days=days), "retention_days": days}

    @app.post("/admin/purge-stale", response_model=None)
    async def admin_purge_stale(days: int = purge_days, x_api_key: str = Header(default="")):
        denied = _check_api_key(x_api_key)
        if denied:
            return denied
        return {
            "status": "ok",
            "deleted": store.purge_stale_unacked(days=days),
            "retention_days": days,
        }

    return app


app = create_relay_app(
    db_path=os.environ.get("RELAY_DB_PATH", "data/relay.db"),
    shared_secrets=json.loads(os.environ.get("RELAY_SHARED_SECRETS", "{}")),
    api_key=os.environ.get("RELAY_API_KEY", ""),
)
