"""Daily quiet-day classifier and snapshot writer.

Covers:
  * ``classify_daily_activity`` pure function decision tree.
  * ``check_daily_classification`` flagging missing-history strategies as DEAD.
  * ``capture_snapshot`` row construction from v_strategy_health.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.watchdog.checks import check_daily_classification, classify_daily_activity
from apps.watchdog.snapshot import capture_snapshot


_NOW = datetime(2026, 5, 10, 20, 0, tzinfo=timezone.utc)


def test_classify_active_when_trades_present() -> None:
    label = classify_daily_activity(
        bars=120, trades=3, denials=0, last_bar_ts=_NOW - timedelta(minutes=5),
        family_disconnect_count=0, now=_NOW,
    )
    assert label == "ACTIVE"


def test_classify_normal_quiet_when_alive_no_signals() -> None:
    label = classify_daily_activity(
        bars=78, trades=0, denials=0, last_bar_ts=_NOW - timedelta(minutes=5),
        family_disconnect_count=0, now=_NOW,
    )
    assert label == "NORMAL_QUIET"


def test_classify_blocked_when_denials_with_no_fills() -> None:
    label = classify_daily_activity(
        bars=78, trades=0, denials=4, last_bar_ts=_NOW - timedelta(minutes=5),
        family_disconnect_count=0, now=_NOW,
    )
    assert label == "BLOCKED"


def test_classify_dead_when_no_bars() -> None:
    label = classify_daily_activity(
        bars=0, trades=0, denials=0, last_bar_ts=None,
        family_disconnect_count=0, now=_NOW,
    )
    assert label == "DEAD"


def test_classify_broker_down_when_dead_with_disconnects() -> None:
    label = classify_daily_activity(
        bars=0, trades=0, denials=0, last_bar_ts=None,
        family_disconnect_count=2, now=_NOW,
    )
    assert label == "BROKER_DOWN"


def test_classify_dead_when_last_bar_too_old() -> None:
    label = classify_daily_activity(
        bars=12, trades=0, denials=0,
        last_bar_ts=_NOW - timedelta(hours=10),  # > 8h threshold
        family_disconnect_count=0, now=_NOW,
        session_start_threshold_hours=8.0,
    )
    assert label == "DEAD"


def test_classify_active_takes_precedence_over_disconnect() -> None:
    """ACTIVE wins even if family had a disconnect earlier today."""
    label = classify_daily_activity(
        bars=120, trades=2, denials=1, last_bar_ts=_NOW - timedelta(minutes=2),
        family_disconnect_count=3, now=_NOW,
    )
    assert label == "ACTIVE"


@pytest.mark.asyncio
async def test_check_daily_classification_marks_missing_history_as_dead() -> None:
    """A strategy in an active family with NO row in v_daily_strategy_activity is DEAD."""
    pool = AsyncMock()
    # No rows at all in v_daily_strategy_activity (e.g., snapshot writer never ran today)
    pool.fetch = AsyncMock(side_effect=[
        [],   # v_daily_strategy_activity query
        [],   # adapter_state query
    ])

    config: dict = {"checks": {"daily_classification": {}}}
    family_map = {"ATRSS": "swing", "TPC": "swing"}
    active = {"swing"}

    results = await check_daily_classification(pool, config, active, family_map)
    keys = sorted(r.key for r in results)
    assert keys == ["daily_class:ATRSS", "daily_class:TPC"]
    assert all(r.is_problem for r in results)
    assert all("DEAD" in r.detail for r in results)


@pytest.mark.asyncio
async def test_check_daily_classification_labels_active_strategy() -> None:
    """A strategy with trades > 0 is ACTIVE and not a problem."""
    last_bar = _NOW - timedelta(minutes=2)
    activity_row = {
        "strategy_id": "ATRSS",
        "family_id": "swing",
        "bars": 80,
        "denials": 0,
        "trades": 2,
        "last_bar_ts": last_bar,
    }
    adapter_row = {"adapter_id": "swing", "disconnect_count_24h": 0}

    pool = AsyncMock()
    pool.fetch = AsyncMock(side_effect=[[activity_row], [adapter_row]])
    config: dict = {"checks": {"daily_classification": {}}}
    family_map = {"ATRSS": "swing"}

    results = await check_daily_classification(pool, config, {"swing"}, family_map)
    assert len(results) == 1
    assert results[0].key == "daily_class:ATRSS"
    assert "ACTIVE" in results[0].detail
    assert results[0].is_problem is False


@pytest.mark.asyncio
async def test_check_daily_classification_skips_inactive_family() -> None:
    """Strategies whose family is not currently active are skipped (no row emitted)."""
    pool = AsyncMock()
    pool.fetch = AsyncMock(side_effect=[[], []])
    family_map = {"NQDTC_v2.1": "momentum"}
    results = await check_daily_classification(
        pool, {"checks": {}}, active_families=set(), strategy_family_map=family_map,
    )
    assert results == []


@pytest.mark.asyncio
async def test_capture_snapshot_inserts_one_row_per_active_strategy() -> None:
    """Snapshot pulls bars_processed/oms_health from JSONB and writes one row per active strategy."""
    captured_rows: list[tuple] = []

    async def _executemany(_sql: str, payloads: list[tuple]) -> None:
        captured_rows.extend(payloads)

    last_bar = _NOW - timedelta(minutes=15)
    health_rows = [
        {
            "strategy_id": "ATRSS",
            "mode": "RUNNING",
            "last_decision_code": "EVALUATED_NO_SIGNAL",
            "last_decision_details": json.dumps({
                "liveness": {"bars_processed": 87},
                "oms_health": {"consecutive_denials": 0, "denied": 1, "submitted": 12},
            }),
            "last_seen_bar_ts": last_bar,
        },
        {
            "strategy_id": "Inactive_Strat",
            "mode": "RUNNING",
            "last_decision_code": "IDLE",
            "last_decision_details": "{}",
            "last_seen_bar_ts": None,
        },
    ]

    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=health_rows)
    pool.executemany = AsyncMock(side_effect=_executemany)

    inserted = await capture_snapshot(
        pool, active_families={"swing"}, strategy_family_map={"ATRSS": "swing"},
    )
    assert inserted == 1
    assert len(captured_rows) == 1
    row = captured_rows[0]
    # Order: captured_at, sid, mode, decision, bars, last_bar, consec, denied
    assert row[1] == "ATRSS"
    assert row[2] == "RUNNING"
    assert row[3] == "EVALUATED_NO_SIGNAL"
    assert row[4] == 87
    assert row[5] == last_bar
    assert row[6] == 0
    assert row[7] == 1


@pytest.mark.asyncio
async def test_capture_snapshot_handles_dict_decision_details() -> None:
    """asyncpg can return JSONB as dict OR str. Both must work."""
    captured_rows: list[tuple] = []

    async def _executemany(_sql: str, payloads: list[tuple]) -> None:
        captured_rows.extend(payloads)

    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[{
        "strategy_id": "TPC",
        "mode": "RUNNING",
        "last_decision_code": "IDLE",
        "last_decision_details": {  # dict, not str
            "liveness": {"bars_processed": 10},
            "oms_health": {"consecutive_denials": 0, "denied": 0},
        },
        "last_seen_bar_ts": _NOW,
    }])
    pool.executemany = AsyncMock(side_effect=_executemany)

    inserted = await capture_snapshot(
        pool, active_families={"swing"}, strategy_family_map={"TPC": "swing"},
    )
    assert inserted == 1
    assert captured_rows[0][4] == 10  # bars_processed


@pytest.mark.asyncio
async def test_capture_snapshot_swallows_db_failure() -> None:
    """A read or write failure must not crash the watchdog cycle."""
    pool = MagicMock()
    pool.fetch = AsyncMock(side_effect=RuntimeError("connection lost"))
    pool.executemany = AsyncMock()

    inserted = await capture_snapshot(
        pool, active_families={"swing"}, strategy_family_map={"ATRSS": "swing"},
    )
    assert inserted == 0
    pool.executemany.assert_not_awaited()
