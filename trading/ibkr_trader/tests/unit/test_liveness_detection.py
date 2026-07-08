"""Tests for liveness detection (Phase 1-4 of silent failure detection)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1-3. Engine liveness_payload tests
# ---------------------------------------------------------------------------

class _StubEngine:
    """Minimal engine stub that mirrors the common liveness pattern."""

    def __init__(self):
        self._bars_processed: int = 0
        self._symbol_last_bar_ts: dict[str, datetime] = {}

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._bars_processed,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
        }


def test_liveness_payload_fresh_engine():
    """1. liveness_payload returns correct structure on fresh engine."""
    engine = _StubEngine()
    payload = engine.liveness_payload()
    assert payload == {"bars_processed": 0, "symbol_freshness": {}}


def test_liveness_payload_counter_increments():
    """2. bars_processed counter increments after bar processing."""
    engine = _StubEngine()
    engine._bars_processed += 1
    engine._bars_processed += 1
    assert engine.liveness_payload()["bars_processed"] == 2


def test_liveness_payload_symbol_freshness():
    """3. symbol_freshness updates per-symbol."""
    engine = _StubEngine()
    ts1 = datetime(2026, 5, 3, 14, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 5, 3, 14, 5, tzinfo=timezone.utc)
    engine._symbol_last_bar_ts["QQQ"] = ts1
    engine._symbol_last_bar_ts["GLD"] = ts2

    payload = engine.liveness_payload()
    assert payload["symbol_freshness"]["QQQ"] == ts1.isoformat()
    assert payload["symbol_freshness"]["GLD"] == ts2.isoformat()


# ---------------------------------------------------------------------------
# 4-9. Watchdog check_liveness tests
# ---------------------------------------------------------------------------

def _make_row(sid, bars_processed, symbol_freshness=None, decision="NO_SIGNAL",
              oms_health=None, last_rebalance_date=None):
    """Build a mock DB row mimicking strategy_state table."""
    liveness = {"bars_processed": bars_processed}
    if symbol_freshness is not None:
        liveness["symbol_freshness"] = symbol_freshness
    else:
        liveness["symbol_freshness"] = {}
    if last_rebalance_date is not None:
        liveness["last_rebalance_date"] = last_rebalance_date
    details = {"liveness": liveness}
    if oms_health is not None:
        details["oms_health"] = oms_health
    return {
        "strategy_id": sid,
        "last_decision_details": details,
        "last_decision_code": decision,
        "last_seen_bar_ts": datetime.now(timezone.utc),
    }


@pytest.fixture
def liveness_config():
    return {
        "checks": {
            "liveness": {
                "enabled": True,
                "symbol_stale_thresholds": {
                    "swing": 5400,
                    "momentum": 1200,
                    "stock": 900,
                },
                "stalled_cycles": 3,
            }
        }
    }


@pytest.fixture
def family_map():
    return {
        "ATRSS": "swing",
        "Helix_v40": "momentum",
        "OVERLAY": "swing",
        "ALCB_T2": "stock",
    }


@pytest.mark.asyncio
async def test_check_liveness_ok_when_counter_increases(liveness_config, family_map):
    """4. Returns OK when bars_processed increases between cycles."""
    from apps.watchdog.checks import check_liveness

    rows = [_make_row("ATRSS", bars_processed=10)]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {"ATRSS": 5}  # was 5, now 10 -> increasing
    stalled: dict[str, int] = {}
    active = {"swing"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    ok_results = [r for r in results if not r.is_problem and "stalled" in r.key]
    assert len(ok_results) == 1
    assert "OK" in ok_results[0].detail
    assert prev_bars["ATRSS"] == 10


@pytest.mark.asyncio
async def test_check_liveness_critical_when_stalled(liveness_config, family_map):
    """5. Returns CRITICAL when bars_processed unchanged for N cycles."""
    from apps.watchdog.checks import check_liveness

    rows = [_make_row("ATRSS", bars_processed=42)]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {"ATRSS": 42}  # same as current
    stalled: dict[str, int] = {"ATRSS": 2}  # already 2, will become 3 == threshold
    active = {"swing"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    problems = [r for r in results if r.is_problem and "stalled" in r.key]
    assert len(problems) == 1
    assert "engine stalled" in problems[0].detail
    assert stalled["ATRSS"] == 3


@pytest.mark.asyncio
async def test_check_liveness_symbol_stale(liveness_config, family_map):
    """6. Returns CRITICAL when symbol bar age exceeds threshold."""
    from apps.watchdog.checks import check_liveness

    stale_ts = (datetime.now(timezone.utc) - timedelta(minutes=100)).isoformat()
    rows = [_make_row("ATRSS", bars_processed=10,
                       symbol_freshness={"QQQ": stale_ts})]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {}
    stalled: dict[str, int] = {}
    active = {"swing"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    sym_problems = [r for r in results if r.is_problem and "sym:" in r.key]
    assert len(sym_problems) == 1
    assert "QQQ" in sym_problems[0].detail
    assert "bar stale" in sym_problems[0].detail


@pytest.mark.asyncio
async def test_check_liveness_skips_inactive_family(liveness_config, family_map):
    """7. Skips inactive families (off-hours)."""
    from apps.watchdog.checks import check_liveness

    rows = [_make_row("ATRSS", bars_processed=42)]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {"ATRSS": 42}
    stalled: dict[str, int] = {"ATRSS": 5}
    active: set[str] = set()  # no active families

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    assert len(results) == 0  # nothing checked


@pytest.mark.asyncio
async def test_check_liveness_handles_missing_liveness_key(liveness_config, family_map):
    """8. Handles missing liveness key gracefully."""
    from apps.watchdog.checks import check_liveness

    rows = [{
        "strategy_id": "ATRSS",
        "last_decision_details": {"some_other_key": 123},
        "last_decision_code": "NO_SIGNAL",
        "last_seen_bar_ts": datetime.now(timezone.utc),
    }]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {}
    stalled: dict[str, int] = {}
    active = {"swing"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    assert len(results) == 0  # skipped gracefully


@pytest.mark.asyncio
async def test_check_liveness_overlay_rebalance_check(liveness_config, family_map):
    """9. Overlay checks rebalance date, not bar freshness."""
    from apps.watchdog.checks import check_liveness

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = [_make_row("OVERLAY", bars_processed=5,
                       last_rebalance_date=today)]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {}
    stalled: dict[str, int] = {}
    active = {"swing"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    rebal_results = [r for r in results if "rebalance" in r.key]
    assert len(rebal_results) == 1
    assert not rebal_results[0].is_problem
    assert "rebalance OK" in rebal_results[0].detail


@pytest.mark.asyncio
async def test_check_liveness_overlay_missed_rebalance(liveness_config, family_map):
    """9b. Overlay flags WARNING when rebalance didn't run today."""
    from apps.watchdog.checks import check_liveness

    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = [_make_row("OVERLAY", bars_processed=5,
                       last_rebalance_date=yesterday)]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {}
    stalled: dict[str, int] = {}
    active = {"swing"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    rebal_problems = [r for r in results if r.is_problem and "rebalance" in r.key]
    assert len(rebal_problems) == 1
    assert "rebalance not run today" in rebal_problems[0].detail


# ---------------------------------------------------------------------------
# 10. OMS consecutive_denials triggers WARNING
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_liveness_oms_denials_warning(liveness_config, family_map):
    """10. OMS consecutive_denials > 5 triggers problem alert."""
    from apps.watchdog.checks import check_liveness

    fresh_ts = datetime.now(timezone.utc).isoformat()
    rows = [_make_row("ALCB_T2", bars_processed=100,
                       symbol_freshness={"AAPL": fresh_ts},
                       oms_health={
                           "submitted": 20,
                           "accepted": 13,
                           "denied": 7,
                           "consecutive_denials": 7,
                       })]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {}
    stalled: dict[str, int] = {}
    active = {"stock"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    oms_problems = [r for r in results if r.is_problem and "oms:" in r.key]
    assert len(oms_problems) == 1
    assert "execution blocked" in oms_problems[0].detail
    assert "7 consecutive" in oms_problems[0].detail


# ---------------------------------------------------------------------------
# 11. Recovery: alert clears when counter resumes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_liveness_recovery_after_stall(liveness_config, family_map):
    """11. Stalled counter resets to 0 when bars_processed resumes."""
    from apps.watchdog.checks import check_liveness

    rows = [_make_row("Helix_v40", bars_processed=50)]
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows)

    prev_bars: dict[str, int] = {"Helix_v40": 45}  # was 45, now 50 -> increasing
    stalled: dict[str, int] = {"Helix_v40": 4}  # was stalled for 4 cycles
    active = {"momentum"}

    results = await check_liveness(pool, liveness_config, active, family_map, prev_bars, stalled)
    ok_results = [r for r in results if not r.is_problem and "stalled" in r.key]
    assert len(ok_results) == 1
    assert stalled["Helix_v40"] == 0  # counter reset


# ---------------------------------------------------------------------------
# OMS Service intent tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oms_service_intent_tracking():
    """Verify OMSService tracks intent submissions, acceptances, and denials."""
    from libs.oms.models.intent import IntentReceipt, IntentResult

    handler = AsyncMock()
    bus = MagicMock()
    reconciler = MagicMock()

    from libs.oms.services.oms_service import OMSService
    svc = OMSService(handler, bus, reconciler)
    svc._ready.set()

    # Simulate ACCEPTED
    handler.submit.return_value = IntentReceipt(
        result=IntentResult.ACCEPTED, intent_id="i1", oms_order_id="o1"
    )
    await svc.submit_intent(MagicMock())
    assert svc._intents_submitted == 1
    assert svc._intents_accepted == 1
    assert svc._intents_denied == 0
    assert svc._consecutive_denials == 0
    assert svc._last_accepted_ts is not None

    # Simulate DENIED
    handler.submit.return_value = IntentReceipt(
        result=IntentResult.DENIED, intent_id="i2", denial_reason="risk"
    )
    await svc.submit_intent(MagicMock())
    assert svc._intents_submitted == 2
    assert svc._intents_denied == 1
    assert svc._consecutive_denials == 1

    # Another DENIED
    await svc.submit_intent(MagicMock())
    assert svc._consecutive_denials == 2

    # ACCEPTED resets consecutive_denials
    handler.submit.return_value = IntentReceipt(
        result=IntentResult.ACCEPTED, intent_id="i4", oms_order_id="o2"
    )
    await svc.submit_intent(MagicMock())
    assert svc._consecutive_denials == 0
    assert svc._intents_accepted == 2
