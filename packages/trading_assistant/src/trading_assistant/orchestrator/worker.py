"""Worker — consumes events from the queue, routes through the brain, executes actions.

The worker is the bridge between the event queue and the rest of the system.
It pulls pending events, asks the OrchestratorBrain what to do, and dispatches.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable

from trading_assistant.orchestrator.conversation_tracker import ConversationTracker
from trading_assistant.orchestrator.db.queue import EventQueue
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain, Action, ActionType
from trading_assistant.orchestrator.subagent import CapacityExceeded
from trading_assistant.orchestrator.task_registry import TaskRegistry
from trading_assistant.orchestrator.tz_utils import bot_trading_date
from trading_assistant.schemas.bot_config import BotConfig
from trading_assistant.skills.lineage_utils import missing_required_lineage_fields

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        queue: EventQueue,
        registry: TaskRegistry,
        brain: OrchestratorBrain,
        event_stream: EventStream | None = None,
        conversation_tracker: ConversationTracker | None = None,
        raw_data_dir: Path | None = None,
        bot_configs: dict[str, BotConfig] | None = None,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._brain = brain
        self._event_stream: EventStream | None = event_stream
        self._conversation_tracker: ConversationTracker | None = conversation_tracker
        self._raw_data_dir: Path | None = raw_data_dir
        self._bot_configs = bot_configs or {}

        # Pluggable handlers — set these to hook into the action pipeline
        self.on_alert: Callable[[Action], Awaitable[None]] | None = None
        self.on_heartbeat: Callable[[Action], Awaitable[None]] | None = None
        self.on_triage: Callable[[Action], Awaitable[None]] | None = None
        self.on_daily_analysis: Callable[[Action], Awaitable[None]] | None = None
        self.on_weekly_analysis: Callable[[Action], Awaitable[None]] | None = None
        self.on_monthly_validation: Callable[[Action], Awaitable[None]] | None = None
        self.on_notification: Callable[[Action], Awaitable[None]] | None = None
        self.on_feedback: Callable[[Action], Awaitable[None]] | None = None

        self.daily_queue_counts: dict[str, int] = {}
        self.weekly_queue_counts: dict[str, int] = {}

    def _emit(self, event_type: str, data: dict | None = None) -> None:
        """Broadcast an event to the SSE stream (if attached)."""
        event_stream = getattr(self, "_event_stream", None)
        if event_stream:
            event_stream.broadcast(event_type, data)

    async def drain(self, batch_size: int = 200, time_budget_seconds: float = 30.0) -> int:
        """Process pending events until the queue is empty or the time budget expires.

        P1-2: replaces the old 10-events-per-minute ceiling. Returns the
        cumulative count of events processed during this call.
        """
        deadline = time.monotonic() + max(0.0, float(time_budget_seconds))
        total = 0
        while True:
            processed = await self.process_batch(limit=batch_size)
            total += processed
            if processed == 0:
                break
            if time.monotonic() >= deadline:
                break
        return total

    async def process_batch(self, limit: int = 10) -> int:
        """Process up to `limit` pending events. Returns count processed."""
        events = await self._queue.claim(limit=limit)
        if not events:
            return 0

        processed = 0
        for event in events:
            event_id = event.get("event_id", "")
            chain_id = event.get("chain_id", "")

            # Begin or extend conversation chain for loop protection
            if self._conversation_tracker:
                if chain_id:
                    ok = self._conversation_tracker.extend_chain(chain_id, event_id)
                    if not ok:
                        logger.warning(
                            "Loop detected for chain %s on event %s — skipping",
                            chain_id, event_id,
                        )
                        await self._queue.ack(event_id)
                        processed += 1
                        continue
                else:
                    chain = self._conversation_tracker.begin_chain(event_id)
                    chain_id = chain.chain_id

            try:
                self._emit("event_processing_start", {"event_id": event_id})
                actions = self._brain.decide(event)
                for action in actions:
                    action.chain_id = chain_id
                    await self._dispatch(action)
                await self._queue.ack(event_id)
                processed += 1
                self._emit("event_processing_complete", {"event_id": event_id})
            except CapacityExceeded as exc:
                # Transient back-pressure: requeue without consuming a retry slot.
                logger.warning("Subagent capacity for event %s — requeueing: %s", event_id, exc)
                self._emit(
                    "event_requeued_capacity",
                    {"event_id": event_id, "reason": str(exc)},
                )
                await self._queue.requeue(event_id)
            except Exception as exc:
                logger.exception("Failed to process event %s", event_id)
                self._emit("event_processing_error", {"event_id": event_id, "error": str(exc)})
                is_dead = await self._queue.nack(event_id, str(exc))
                if is_dead:
                    logger.error(
                        "Event %s moved to dead-letter after exhausting retries", event_id,
                    )

        return processed

    def _record_queued_event(self, bot_id: str, action_type: ActionType, event_type: str = "") -> None:
        """Track event counts for QUEUE_FOR_DAILY/WEEKLY actions."""
        if action_type == ActionType.QUEUE_FOR_DAILY:
            self.daily_queue_counts[bot_id] = self.daily_queue_counts.get(bot_id, 0) + 1
        elif action_type == ActionType.QUEUE_FOR_WEEKLY:
            self.weekly_queue_counts[bot_id] = self.weekly_queue_counts.get(bot_id, 0) + 1

    def _persist_raw_event(self, action: Action) -> None:
        """Persist event payload to raw JSONL file for later curated data building."""
        if self._raw_data_dir is None:
            return
        details = action.details or {}
        event_type = details.get("event_type", "")
        if not event_type:
            return

        payload = self._normalize_payload(details)
        # Ensure bot_id survives envelope stripping
        if isinstance(payload, dict) and "bot_id" not in payload:
            payload["bot_id"] = action.bot_id
        self._emit_lineage_missing_if_needed(action.bot_id, event_type, payload)
        date = self._resolve_raw_event_date(action.bot_id, details, payload)
        bot_dir = self._raw_data_dir / date / action.bot_id
        bot_dir.mkdir(parents=True, exist_ok=True)
        out_path = bot_dir / f"{event_type}.jsonl"
        try:
            line = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            logger.warning("Failed to persist raw event %s for %s", event_type, action.bot_id)

    def _emit_lineage_missing_if_needed(self, bot_id: str, event_type: str, payload: object) -> None:
        if event_type not in {"trade", "missed_opportunity"}:
            return
        missing = missing_required_lineage_fields(payload)
        if not missing:
            return
        severity = "blocking" if {"strategy_version", "config_version", "deployment_id"} & set(missing) else "warning"
        self._emit(
            "lineage_missing",
            {
                "bot_id": bot_id,
                "event_type": event_type,
                "missing_fields": missing,
                "severity": severity,
            },
        )

    def _resolve_raw_event_date(self, bot_id: str, details: dict, payload: object) -> str:
        if isinstance(payload, dict):
            payload_date = payload.get("date")
            if isinstance(payload_date, str):
                try:
                    datetime.strptime(payload_date, "%Y-%m-%d")
                    return payload_date
                except ValueError:
                    pass

        exchange_timestamp = self._parse_timestamp(details.get("exchange_timestamp"))
        if exchange_timestamp is not None:
            bot_config = self._bot_configs.get(bot_id)
            if bot_config is not None:
                return bot_trading_date(bot_config.timezone, exchange_timestamp)
            return exchange_timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")

        logger.warning(
            "No payload date or exchange_timestamp for bot %s — falling back to current UTC date",
            bot_id,
        )
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _normalize_payload(self, details: dict) -> object:
        payload = details.get("payload", details)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return payload

        if isinstance(payload, dict):
            normalized = dict(payload)
            if details.get("event_type") and "event_type" not in normalized:
                normalized["event_type"] = details["event_type"]
            if details.get("exchange_timestamp") and "exchange_timestamp" not in normalized:
                normalized["exchange_timestamp"] = details["exchange_timestamp"]
            return normalized

        return payload

    @staticmethod
    def _parse_timestamp(raw: object) -> datetime | None:
        if not isinstance(raw, str) or not raw:
            return None
        candidate = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def get_and_reset_daily_counts(self) -> dict[str, int]:
        """Get accumulated daily event counts and reset. Called by daily analysis trigger."""
        counts = dict(self.daily_queue_counts)
        self.daily_queue_counts = {}
        return counts

    def get_and_reset_weekly_counts(self) -> dict[str, int]:
        """Get accumulated weekly event counts and reset. Called by weekly analysis trigger."""
        counts = dict(self.weekly_queue_counts)
        self.weekly_queue_counts = {}
        return counts

    async def _dispatch(self, action: Action) -> None:
        """Route an action to the appropriate handler."""
        if action.type == ActionType.ALERT_IMMEDIATE:
            if self.on_alert:
                await self.on_alert(action)
            else:
                logger.warning("ALERT (no handler): %s — %s", action.bot_id, action.details)

        elif action.type == ActionType.SPAWN_TRIAGE:
            if self.on_triage:
                await self.on_triage(action)
            else:
                logger.info("TRIAGE needed: %s — %s", action.bot_id, action.details)

        elif action.type == ActionType.UPDATE_HEARTBEAT:
            if self.on_heartbeat:
                await self.on_heartbeat(action)
            else:
                logger.debug("Heartbeat: %s", action.bot_id)

        elif action.type == ActionType.SPAWN_DAILY_ANALYSIS:
            if self.on_daily_analysis:
                await self.on_daily_analysis(action)
            else:
                logger.info("Daily analysis triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.SPAWN_WEEKLY_SUMMARY:
            if self.on_weekly_analysis:
                await self.on_weekly_analysis(action)
            else:
                logger.info("Weekly analysis triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.SPAWN_MONTHLY_VALIDATION:
            if self.on_monthly_validation:
                await self.on_monthly_validation(action)
            else:
                logger.info("Monthly validation triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.SEND_NOTIFICATION:
            if self.on_notification:
                await self.on_notification(action)
            else:
                logger.info("Notification triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.PROCESS_FEEDBACK:
            if self.on_feedback:
                await self.on_feedback(action)
            else:
                logger.info("Feedback received but no handler set: %s", action.event_id)

        elif action.type == ActionType.QUEUE_FOR_DAILY:
            self._record_queued_event(action.bot_id, action.type)
            self._persist_raw_event(action)
            logger.debug("Queued for daily: %s", action.event_id)

        elif action.type == ActionType.QUEUE_FOR_WEEKLY:
            self._record_queued_event(action.bot_id, action.type)
            logger.debug("Queued for weekly: %s", action.event_id)

        elif action.type == ActionType.LOG_UNKNOWN:
            logger.warning("Unknown event type from %s: %s", action.bot_id, action.event_id)
