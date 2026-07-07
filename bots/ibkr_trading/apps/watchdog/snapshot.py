"""Periodic strategy_state snapshot writer.

Inserts a row into strategy_heartbeat_history every ~5 minutes for each
strategy currently visible in v_strategy_health. The
v_daily_strategy_activity view (libs/oms/persistence/postgres.py) reads
this history to compute bars_processed deltas and denial totals across a
session, which the quiet-day classifier (apps/watchdog/checks.py
::check_daily_classification) consumes.

The writer is INSERT-only and never crashes the watchdog loop on a DB
outage — it logs the failure and skips the cycle. State resets on
watchdog restart; the next snapshot rebuilds from scratch.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg

logger = logging.getLogger("watchdog.snapshot")


async def capture_snapshot(
    pool: asyncpg.Pool,
    active_families: set[str],
    strategy_family_map: dict[str, str],
) -> int:
    """Capture one snapshot row per active strategy.

    Returns the number of rows inserted (0 on failure or no active
    strategies). Reads from v_strategy_health + last_decision_details
    JSONB and INSERTs into strategy_heartbeat_history.
    """
    if not active_families:
        # Off-hours: skip the DB roundtrip. The classifier ignores days with
        # no active-family snapshots anyway, so capturing here is wasted I/O.
        return 0

    try:
        rows = await pool.fetch(
            "SELECT strategy_id, mode, last_decision_code, last_decision_details, "
            "       last_seen_bar_ts "
            "FROM v_strategy_health"
        )
    except Exception as exc:
        logger.warning("Snapshot read failed: %s", exc)
        return 0

    captured_at = datetime.now(timezone.utc)
    payloads: list[tuple[Any, ...]] = []
    for row in rows:
        sid = row["strategy_id"]
        family = strategy_family_map.get(sid)
        # Skip strategies we have no family mapping for, or whose family is
        # not currently active. Inactive strategies don't need a
        # bars-processed delta because the classifier skips them anyway.
        if not family or family not in active_families:
            continue

        details = _coerce_details(row["last_decision_details"])
        liveness = details.get("liveness") or {}
        oms_health = details.get("oms_health") or {}
        bars_processed = _safe_int(liveness.get("bars_processed"))
        consecutive = _safe_int(oms_health.get("consecutive_denials"))
        denials_today = _safe_int(oms_health.get("denied"))

        payloads.append((
            captured_at,
            sid,
            row["mode"],
            row["last_decision_code"],
            bars_processed,
            row["last_seen_bar_ts"],
            consecutive,
            denials_today,
        ))

    if not payloads:
        return 0

    try:
        await pool.executemany(
            """
            INSERT INTO strategy_heartbeat_history (
                captured_at, strategy_id, mode, last_decision_code,
                bars_processed, last_seen_bar_ts,
                consecutive_denials, denials_today
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (captured_at, strategy_id) DO NOTHING
            """,
            payloads,
        )
    except Exception as exc:
        logger.warning("Snapshot write failed: %s", exc)
        return 0

    logger.debug("Captured snapshot for %d strategies", len(payloads))
    return len(payloads)


def _coerce_details(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
