"""VPS Receiver — pulls events from assistant-local relay ingress into the local event queue.

Protocol:
  1. GET /events?since=<watermark> from relay
  2. Store into local EventQueue (dedup handled by queue)
  3. POST /ack with new watermark to relay
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from trading_assistant.orchestrator.db.queue import EventQueue
from trading_assistant.orchestrator.event_validation import (
    QueueEventValidationError,
    normalize_queue_event,
)

logger = logging.getLogger(__name__)


class VPSReceiver:
    def __init__(
        self,
        relay_url: str,
        local_queue: EventQueue,
        watermark_key: str = "relay",
        timeout: float = 30.0,
        *,
        api_key: str = "",
        latency_tracker=None,
        allowed_bot_ids: set[str] | None = None,
        priority_first: bool = True,
        urgent_max_priority: int = 1,
        _client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._relay_url = relay_url
        self._queue = local_queue
        self._watermark_key = watermark_key
        self._timeout = timeout
        self._api_key = api_key
        self._client_factory = _client_factory
        self._latency_tracker = latency_tracker
        self._consecutive_failures: int = 0
        self._allowed_bot_ids = set(allowed_bot_ids) if allowed_bot_ids is not None else None
        self._priority_first = priority_first
        self._urgent_max_priority = urgent_max_priority
        self._priority_first_supported: bool | None = None
        self._last_pull_full_page = False

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def is_healthy(self) -> bool:
        return self._consecutive_failures == 0

    def _make_client(self) -> httpx.AsyncClient:
        if self._client_factory:
            return self._client_factory()
        headers = {}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        return httpx.AsyncClient(base_url=self._relay_url, timeout=self._timeout, headers=headers)

    async def pull_and_store(self, limit: int = 100) -> int:
        """Pull new events from relay, store locally, ack on relay. Returns count pulled."""
        self._last_pull_full_page = False
        watermark = await self._queue.get_watermark(self._watermark_key)

        async with self._make_client() as client:
            total_inserted = 0
            if self._priority_first and self._priority_first_supported is not False:
                priority_params: dict = {
                    "limit": limit,
                    "priority_first": "true",
                    "max_priority": self._urgent_max_priority,
                }
                priority_resp = await client.get("/events", params=priority_params)
                priority_resp.raise_for_status()
                priority_payload = priority_resp.json()
                if self._is_priority_first_exact_response(priority_payload):
                    self._priority_first_supported = True
                    priority_events = priority_payload.get("events", [])
                    if priority_events:
                        priority_result = await self._ingest_and_ack(
                            client,
                            priority_events,
                            ack_mode="exact",
                        )
                        total_inserted += priority_result["inserted"]
                        if priority_result["raw_count"] >= limit:
                            self._last_pull_full_page = True
                            return total_inserted
                else:
                    self._priority_first_supported = False
                    logger.info(
                        "Relay did not advertise priority-first exact ack; "
                        "using watermark delivery"
                    )

            params: dict = {"limit": limit}
            if watermark:
                params["since"] = watermark

            resp = await client.get("/events", params=params)
            resp.raise_for_status()

            events = resp.json().get("events", [])
            if not events:
                return total_inserted

            normal_result = await self._ingest_and_ack(
                client,
                events,
                ack_mode="watermark",
            )
            total_inserted += normal_result["inserted"]
            self._last_pull_full_page = normal_result["raw_count"] >= limit
            return total_inserted

    async def _ingest_and_ack(
        self,
        client: Any,
        events: list[Any],
        *,
        ack_mode: str,
    ) -> dict[str, int]:
        now = datetime.now(timezone.utc).isoformat()
        normalized_events: list[dict] = []
        normalized_raw_events: list[object] = []
        normalized_raw_event_ids: list[str] = []
        all_raw_event_ids: list[str] = []
        quarantined = 0
        last_event_id = ""

        for raw_event in events:
            raw_event_id = raw_event.get("event_id", "") if isinstance(raw_event, dict) else ""
            if isinstance(raw_event_id, str) and raw_event_id:
                all_raw_event_ids.append(raw_event_id)
                last_event_id = raw_event_id
            try:
                event = self._normalize_relay_event(raw_event, received_at=now)
            except QueueEventValidationError as exc:
                quarantined += 1
                await self._quarantine_event(raw_event_id, raw_event, exc.detail)
                logger.warning("Quarantined relay event %r: %s", raw_event_id, exc.detail)
                continue

            if self._latency_tracker:
                ex_ts = event.get("exchange_timestamp", "")
                rx_ts = event.get("received_at", "")
                if ex_ts and rx_ts:
                    self._latency_tracker.record(event.get("bot_id", "unknown"), ex_ts, rx_ts)
            normalized_events.append(event)
            normalized_raw_events.append(raw_event)
            normalized_raw_event_ids.append(raw_event_id if isinstance(raw_event_id, str) else "")

        classifications = await self._enqueue_classified(normalized_events)
        for raw_event_id, raw_event, classification in zip(
            normalized_raw_event_ids,
            normalized_raw_events,
            classifications,
            strict=True,
        ):
            await self._record_classification(
                raw_event_id=raw_event_id,
                event_id=classification.event_id,
                classification=classification.classification,
                payload=raw_event,
            )
        inserted = sum(1 for item in classifications if item.classification == "enqueued")
        duplicates = sum(1 for item in classifications if item.classification == "duplicate")
        logger.info(
            "Pulled %d relay events (%d new, %d dup, %d quarantined)",
            len(events),
            inserted,
            duplicates,
            quarantined,
        )

        if ack_mode == "exact":
            if all_raw_event_ids:
                ack_resp = await client.post("/ack-exact", json={"event_ids": all_raw_event_ids})
                ack_resp.raise_for_status()
        elif last_event_id:
            ack_resp = await client.post("/ack", json={"watermark": last_event_id})
            ack_resp.raise_for_status()
            await self._queue.update_watermark(self._watermark_key, last_event_id)
        return {"inserted": inserted, "raw_count": len(events)}

    @staticmethod
    def _is_priority_first_exact_response(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        return (
            payload.get("delivery_mode") == "priority_first"
            and payload.get("ack_mode") == "exact"
        )

    async def _enqueue_classified(self, normalized_events: list[dict]) -> list[Any]:
        if hasattr(self._queue, "enqueue_batch_classified"):
            return await self._queue.enqueue_batch_classified(normalized_events)
        result = await self._queue.enqueue_batch(normalized_events)
        inserted = int(getattr(result, "inserted", 0) or 0)
        duplicates = int(getattr(result, "duplicates", 0) or 0)
        classifications: list[Any] = []
        for index, event in enumerate(normalized_events):
            classification = "enqueued" if index < inserted else "duplicate"
            if duplicates <= 0 and index >= inserted:
                classification = "duplicate"
            classifications.append(
                _Classification(
                    event_id=str(event.get("event_id") or ""),
                    classification=classification,
                )
            )
        return classifications

    async def _quarantine_event(self, raw_event_id: Any, payload: Any, reason: str) -> None:
        if hasattr(self._queue, "quarantine_relay_event"):
            await self._queue.quarantine_relay_event(
                source=self._watermark_key,
                raw_event_id=raw_event_id if isinstance(raw_event_id, str) else "",
                reason=reason,
                payload=payload,
            )
        await self._record_classification(
            raw_event_id=raw_event_id if isinstance(raw_event_id, str) else "",
            event_id=raw_event_id if isinstance(raw_event_id, str) else "",
            classification="quarantined",
            payload=payload,
            reason=reason,
        )

    async def _record_classification(
        self,
        *,
        raw_event_id: str,
        event_id: str,
        classification: str,
        payload: Any,
        reason: str = "",
    ) -> None:
        if not hasattr(self._queue, "record_relay_ingest_classification"):
            return
        kwargs = {
            "source": self._watermark_key,
            "raw_event_id": raw_event_id,
            "event_id": event_id,
            "classification": classification,
            "payload": payload,
        }
        if reason:
            kwargs["reason"] = reason
        await self._queue.record_relay_ingest_classification(**kwargs)

    def _normalize_relay_event(self, event: dict, *, received_at: str) -> dict:
        if not isinstance(event, dict):
            raise QueueEventValidationError("relay event must be a JSON object")

        normalized = dict(event)
        normalized.setdefault("received_at", received_at)

        payload = normalized.get("payload", {})
        payload_obj = payload
        if isinstance(payload, str):
            try:
                payload_obj = json.loads(payload)
            except json.JSONDecodeError:
                payload_obj = payload

        normalized.setdefault(
            "exchange_timestamp",
            self._extract_exchange_timestamp(payload_obj) or received_at,
        )
        return normalize_queue_event(
            normalized,
            allowed_bot_ids=self._allowed_bot_ids,
        )

    @staticmethod
    def _extract_exchange_timestamp(payload: object) -> str:
        if not isinstance(payload, dict):
            return ""
        for key in ("exchange_timestamp", "timestamp", "period_end"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for meta_key in ("metadata", "event_metadata"):
            meta = payload.get(meta_key)
            if isinstance(meta, dict):
                value = meta.get("exchange_timestamp")
                if isinstance(value, str) and value:
                    return value
        return ""

    async def poll(self) -> int:
        """Pull with retry-safe error handling. Returns events pulled, 0 on failure."""
        try:
            pulled = await self.pull_and_store()
            self._consecutive_failures = 0
            return pulled
        except Exception as exc:
            self._consecutive_failures += 1
            delay = min(2 ** self._consecutive_failures, 300)
            logger.warning(
                "Relay poll failed (attempt %d, next backoff %ds): %s",
                self._consecutive_failures, delay, exc,
            )
            return 0

    async def drain(self, batch_size: int = 100, max_batches: int = 100) -> int:
        """Pull all pending events from relay. For startup catch-up."""
        total = 0
        for _ in range(max_batches):
            try:
                pulled = await self.pull_and_store(limit=batch_size)
                self._consecutive_failures = 0
            except Exception as exc:
                self._consecutive_failures += 1
                logger.warning(
                    "Relay drain failed (attempt %d): %s",
                    self._consecutive_failures,
                    exc,
                )
                break
            total += pulled
            if pulled < batch_size and not self._last_pull_full_page:
                break
        if total > 0:
            logger.info("Startup drain: pulled %d events from relay", total)
        return total


class _Classification:
    def __init__(self, *, event_id: str, classification: str) -> None:
        self.event_id = event_id
        self.classification = classification
