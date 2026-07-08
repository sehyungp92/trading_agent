from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from libs.oms.persistence.postgres import PgStore
from libs.oms.persistence.schema import StrategyStateRow
from strategies.stock.instrumentation.src.facade import InstrumentationKit


class _CapturePool:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def execute(self, *args):
        self.calls.append(args)


@pytest.mark.asyncio
async def test_strategy_heartbeat_preserves_existing_decision_fields() -> None:
    pool = _CapturePool()
    store = PgStore(pool)

    await store.upsert_strategy_state(StrategyStateRow(strategy_id="TEST"))

    sql = pool.calls[0][0]
    assert "NULLIF(EXCLUDED.last_decision_code, '')" in sql
    assert "strategy_state.last_decision_code" in sql
    assert "EXCLUDED.last_decision_details IS DISTINCT FROM '{}'::jsonb" in sql
    assert pool.calls[0][6] is None
    assert pool.calls[0][7] == "{}"


@pytest.mark.asyncio
async def test_record_strategy_decision_updates_decision_without_heartbeat() -> None:
    pool = _CapturePool()
    store = PgStore(pool)
    seen_at = datetime.now(timezone.utc)

    await store.record_strategy_decision(
        "TEST",
        "BLOCKED:spread_gate",
        details={"pair": "QQQ"},
        last_seen_bar_ts=seen_at,
    )

    sql = pool.calls[0][0]
    assert "last_decision_code = EXCLUDED.last_decision_code" in sql
    assert "last_heartbeat_ts =" not in sql
    assert pool.calls[0][1:] == (
        "TEST",
        "BLOCKED:spread_gate",
        '{"pair": "QQQ"}',
        seen_at,
    )


@pytest.mark.asyncio
async def test_instrumentation_indicator_decision_records_strategy_state() -> None:
    calls = []

    class Store:
        async def record_strategy_decision(self, *args, **kwargs):
            calls.append((args, kwargs))

    class Manager:
        _pg_store = Store()
        _strategy_id = "ALCB_v1"
        _config = {}

    kit = InstrumentationKit(Manager(), strategy_type="strategy_alcb")
    seen_at = datetime.now(timezone.utc)

    kit.on_indicator_snapshot(
        pair="QQQ",
        indicators={},
        signal_name="alcb_decision",
        signal_strength=1.0,
        decision="ready",
        strategy_type="strategy_alcb",
        exchange_timestamp=seen_at,
    )
    await asyncio.sleep(0)

    assert calls == [
        (
            ("ALCB_v1", "alcb_decision:ready"),
            {
                "details": {
                    "pair": "QQQ",
                    "signal_name": "alcb_decision",
                    "signal_strength": 1.0,
                    "strategy_type": "strategy_alcb",
                    "bar_id": None,
                    "context": {},
                },
                "last_seen_bar_ts": seen_at,
            },
        )
    ]
