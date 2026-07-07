"""Shared helpers for prompt evidence sources."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.orchestrator.jsonl_store import read_jsonl_tail

logger = logging.getLogger(__name__)

FINDINGS_MAX_AGE_DAYS = 90
FINDINGS_MAX_ENTRIES = 50


def safe_jsonl(path: Path, *, max_records: int = 1000) -> list[dict]:
    if not path.exists():
        return []
    try:
        return read_jsonl_tail(path, max_records=max_records)
    except OSError:
        logger.warning("Could not read %s", path)
        return []


def parse_timestamp(entry: dict) -> datetime | None:
    for key in ("timestamp", "created_at", "updated_at", "recorded_at", "measured_at", "date"):
        value = entry.get(key)
        if value and isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except (TypeError, ValueError):
                continue
            if "T" not in value and len(value) == 10:
                parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
            return parsed
    return None


def filter_by_bot(entries: list[dict], bot_id: str) -> list[dict]:
    if not bot_id:
        return entries
    return [
        entry for entry in entries
        if not (entry_bot := entry.get("bot_id", "") or entry.get("target_id", ""))
        or bot_id in entry_bot
    ]


def filter_inactive_strategies(entries: list[dict], registry: object | None) -> list[dict]:
    is_active = getattr(registry, "is_active", None)
    if not callable(is_active):
        return entries
    return [
        entry for entry in entries
        if not (strategy_id := entry.get("strategy_id")) or is_active(strategy_id)
    ]


def apply_temporal_window(
    entries: list[dict],
    *,
    max_age_days: int = FINDINGS_MAX_AGE_DAYS,
    max_entries: int = FINDINGS_MAX_ENTRIES,
    now: datetime | None = None,
) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    with_ts: list[tuple[datetime, dict]] = []
    without_ts: list[dict] = []
    for entry in entries:
        ts = parse_timestamp(entry)
        if ts is None:
            without_ts.append(entry)
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        cutoff_date = cutoff.date() - (
            timedelta(days=1) if _has_date_only_timestamp(entry) else timedelta()
        )
        if ts.date() >= cutoff_date:
            with_ts.append((ts, entry))
    with_ts.sort(key=lambda item: _decay_score(item[0], now), reverse=True)
    return ([entry for _, entry in with_ts] + without_ts)[:max_entries]


def _has_date_only_timestamp(entry: dict) -> bool:
    value = entry.get("date")
    return isinstance(value, str) and len(value) == 10 and "T" not in value


def _decay_score(ts: datetime, now: datetime) -> float:
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 2.0 ** (-age_days / 14.0)
