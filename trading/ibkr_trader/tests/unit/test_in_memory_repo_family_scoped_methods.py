from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from libs.oms.models.instrument import Instrument
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType, RiskContext
from libs.oms.models.position import Position
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.persistence.schema import (
    RiskDailyPortfolioRow,
    RiskDailyStrategyRow,
    StrategyStateRow,
    TradeRow,
)


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_family_scoped_risk_methods_filter_positions_and_pending_orders() -> None:
    repo = InMemoryRepository()
    await repo.save_position(_position("S1", "QQQ", 10, risk_dollars=100.0, risk_r=2.0))
    await repo.save_position(_position("S2", "QQQ", -5, risk_dollars=80.0, risk_r=1.5))
    await repo.save_position(_position("OTHER", "QQQ", 7, risk_dollars=999.0, risk_r=9.0))
    await repo.save_order(_entry_order("O1", "S1", "NQ", OrderSide.BUY, 2, 40.0, remaining_qty=1))
    await repo.save_order(_entry_order("O2", "S2", "MNQ", OrderSide.SELL, 3, 60.0))
    await repo.save_order(_entry_order("O3", "OTHER", "NQ", OrderSide.BUY, 9, 999.0))

    assert [p.strategy_id for p in await repo.get_positions_for_strategies(["S1", "S2"])] == ["S1", "S2"]
    assert await repo.get_pending_entry_risk_R_for_strategies(["S1", "S2"], 20.0) == pytest.approx(4.0)
    assert await repo.get_directional_risk_R("LONG") == pytest.approx(11.0)
    assert await repo.get_directional_risk_R_for_strategies("LONG", ["S1", "S2"]) == pytest.approx(2.0)
    assert await repo.get_directional_risk_dollars_for_strategies("LONG", ["S1", "S2"]) == pytest.approx(120.0)
    assert await repo.get_directional_risk_dollars_for_strategies("SHORT", ["S1", "S2"]) == pytest.approx(140.0)
    assert await repo.get_sibling_positions_for_symbol(["S2"], "QQQ") is True
    assert await repo.get_sibling_positions_for_symbol(["S2"], "AAPL") is False
    assert await repo.get_open_position_count_for_strategies(["S1", "S2"]) == 4
    assert await repo.get_symbol_open_risk_dollars_for_strategies(["S1", "S2"], "QQQ") == pytest.approx(180.0)
    assert await repo.get_symbols_open_risk_dollars_for_strategies(["S1", "S2"], ["QQQ", "NQ"]) == pytest.approx(180.0)
    assert await repo.get_active_risk_dollars_for_strategies(["S1", "S2"]) == pytest.approx(260.0)
    assert await repo.get_family_aggregate_mnq_eq(["S1", "S2"]) == 28


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_in_memory_repo_supports_risk_daily_trade_and_signal_surfaces() -> None:
    repo = InMemoryRepository()
    trade_day = date(2026, 5, 20)
    await repo.upsert_risk_daily_strategy(
        RiskDailyStrategyRow(trade_date=trade_day, strategy_id="S1", family_id="stock", daily_realized_r=Decimal("1.25"))
    )
    await repo.upsert_risk_daily_strategy(
        RiskDailyStrategyRow(trade_date=trade_day, strategy_id="S2", family_id="stock", daily_realized_r=Decimal("-0.25"))
    )
    await repo.upsert_risk_daily_portfolio(
        RiskDailyPortfolioRow(trade_date=trade_day, family_id="stock", daily_realized_r=Decimal("1.0"))
    )

    rows = await repo.get_risk_daily_strategies_for_date(trade_day, strategy_ids=["S2"])
    assert [row.strategy_id for row in rows] == ["S2"]
    assert await repo.get_risk_daily_strategy_totals(trade_day, trade_day, strategy_ids=["S1"]) == {
        "total_r": Decimal("1.25"),
        "total_usd": Decimal("0"),
    }
    await repo.halt_strategy("S1", "test halt", trade_day)
    assert (await repo.get_risk_daily_strategy("S1", trade_day)).halted is True
    await repo.halt_portfolio("portfolio halt", trade_day, family_id="stock")
    assert (await repo.get_risk_daily_portfolio(trade_day, "stock")).halt_reason == "portfolio halt"

    now = datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc)
    await repo.save_trade(_trade("T1", "S1", now - timedelta(minutes=10), exit_ts=now, realized_r=Decimal("0.5")))
    await repo.save_trade(_trade("T2", "S1", now - timedelta(minutes=5), exit_ts=None))
    assert await repo.get_completed_trade_counts_for_strategies(["S1", "S2"]) == {"S1": 1}
    assert await repo.get_recent_strategy_r_multiples("S1", 5) == [0.5]
    assert [trade.trade_id for trade in await repo.get_open_trades()] == ["T2"]

    await repo.upsert_strategy_signal("S1", "LONG", now)
    await repo.upsert_strategy_signal("S1", "SHORT", now + timedelta(minutes=5))
    await repo.update_chop_score("S1", 3)
    signal = await repo.get_strategy_signal("S1")
    assert signal["daily_entry_count"] == 2
    assert signal["last_direction"] == "SHORT"
    assert signal["chop_score"] == 3


@pytest.mark.asyncio
@pytest.mark.parity_smoke
async def test_in_memory_strategy_state_updates_match_postgres_non_creation_semantics() -> None:
    repo = InMemoryRepository()
    heartbeat = datetime(2026, 5, 20, 13, 55, tzinfo=timezone.utc)
    await repo.upsert_strategy_state(StrategyStateRow(strategy_id="S1", last_heartbeat_ts=heartbeat))

    await repo.record_strategy_decision(
        "S1",
        "ENTRY_BLOCKED",
        {"reason": "fixture"},
        last_seen_bar_ts=datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc),
    )
    state = (await repo.get_strategy_states())[0]

    assert state.last_heartbeat_ts == heartbeat
    assert state.last_decision_code == "ENTRY_BLOCKED"
    assert state.last_seen_bar_ts == datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc)

    await repo.update_chop_score("UNKNOWN", 5)
    assert await repo.get_strategy_signal("UNKNOWN") is None


@pytest.mark.asyncio
@pytest.mark.parity_smoke
@pytest.mark.skipif(
    not os.getenv("PARITY_POSTGRES_DSN"),
    reason="set PARITY_POSTGRES_DSN to run InMemoryRepository against a real PgStore fixture",
)
async def test_in_memory_family_scoped_methods_match_postgres_fixture() -> None:
    import asyncpg

    from libs.oms.persistence.postgres import PgStore

    prefix = f"PARITY_{uuid4().hex[:8]}"
    strategy_ids = [f"{prefix}_S1", f"{prefix}_S2"]
    other_strategy = f"{prefix}_OTHER"
    all_strategy_ids = [*strategy_ids, other_strategy]
    pool = await asyncpg.create_pool(os.environ["PARITY_POSTGRES_DSN"], min_size=1, max_size=1)
    store = PgStore(pool)
    memory = InMemoryRepository()

    try:
        await store.init_schema()
        await _seed_memory_family_fixture(memory, strategy_ids, other_strategy)
        await _seed_postgres_family_fixture(pool, strategy_ids, other_strategy)

        assert await memory.get_directional_risk_R_for_strategies("LONG", strategy_ids) == pytest.approx(
            await store.get_directional_risk_R_for_strategies("LONG", strategy_ids)
        )
        assert await memory.get_directional_risk_dollars_for_strategies("LONG", strategy_ids) == pytest.approx(
            await store.get_directional_risk_dollars_for_strategies("LONG", strategy_ids)
        )
        assert await memory.get_open_position_count_for_strategies(strategy_ids) == (
            await store.get_open_position_count_for_strategies(strategy_ids)
        )
        assert await memory.get_active_risk_dollars_for_strategies(strategy_ids) == pytest.approx(
            await store.get_active_risk_dollars_for_strategies(strategy_ids)
        )
        assert await memory.get_completed_trade_counts_for_strategies(strategy_ids) == (
            await store.get_completed_trade_counts_for_strategies(strategy_ids)
        )
        assert await memory.get_family_aggregate_mnq_eq(strategy_ids) == (
            await store.get_family_aggregate_mnq_eq(strategy_ids)
        )
    finally:
        await _cleanup_postgres_family_fixture(pool, all_strategy_ids)
        await pool.close()


def _instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        root=symbol,
        venue="SMART",
        tick_size=0.25,
        tick_value=0.5,
        multiplier=1.0,
    )


async def _seed_memory_family_fixture(
    repo: InMemoryRepository,
    strategy_ids: list[str],
    other_strategy: str,
) -> None:
    await repo.save_position(_position(strategy_ids[0], "QQQ", 10, risk_dollars=100.0, risk_r=2.0))
    await repo.save_position(_position(strategy_ids[1], "QQQ", -5, risk_dollars=80.0, risk_r=1.5))
    await repo.save_position(_position(other_strategy, "QQQ", 7, risk_dollars=999.0, risk_r=9.0))
    await repo.save_order(_entry_order(f"{strategy_ids[0]}_O1", strategy_ids[0], "NQ", OrderSide.BUY, 2, 40.0, remaining_qty=1))
    await repo.save_order(_entry_order(f"{strategy_ids[1]}_O2", strategy_ids[1], "MNQ", OrderSide.SELL, 3, 60.0))
    now = datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc)
    await repo.save_trade(_trade(f"{strategy_ids[0]}_T1", strategy_ids[0], now - timedelta(minutes=10), exit_ts=now, realized_r=Decimal("0.5")))


async def _seed_postgres_family_fixture(pool, strategy_ids: list[str], other_strategy: str) -> None:
    now = datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc)
    rows = [
        ("QQQ", strategy_ids[0], 10, 100.0, 2.0),
        ("QQQ", strategy_ids[1], -5, 80.0, 1.5),
        ("QQQ", other_strategy, 7, 999.0, 9.0),
    ]
    order_rows = [
        (f"{strategy_ids[0]}_O1", strategy_ids[0], "NQ", OrderSide.BUY.value, 2, OrderStatus.PARTIALLY_FILLED.value, 1, 40.0),
        (f"{strategy_ids[1]}_O2", strategy_ids[1], "MNQ", OrderSide.SELL.value, 3, OrderStatus.ACKED.value, 0, 60.0),
    ]
    async with pool.acquire() as conn:
        for symbol, strategy_id, qty, risk_dollars, risk_r in rows:
            await conn.execute(
                """
                INSERT INTO positions
                    (account_id, instrument_symbol, strategy_id, net_qty, avg_price, open_risk_dollars, open_risk_R)
                VALUES ('DU123', $1, $2, $3, 100.0, $4, $5)
                ON CONFLICT (account_id, instrument_symbol, strategy_id) DO UPDATE SET
                    net_qty = EXCLUDED.net_qty,
                    open_risk_dollars = EXCLUDED.open_risk_dollars,
                    open_risk_R = EXCLUDED.open_risk_R
                """,
                symbol,
                strategy_id,
                qty,
                risk_dollars,
                risk_r,
            )
        for order_id, strategy_id, symbol, side, qty, status, remaining_qty, risk_dollars in order_rows:
            await conn.execute(
                """
                INSERT INTO orders
                    (oms_order_id, client_order_id, strategy_id, account_id, instrument_symbol,
                     side, qty, order_type, role, status, remaining_qty, risk_context)
                VALUES ($1, $2, $3, 'DU123', $4, $5, $6, 'LIMIT', 'ENTRY', $7, $8, $9::jsonb)
                ON CONFLICT (oms_order_id) DO UPDATE SET
                    strategy_id = EXCLUDED.strategy_id,
                    status = EXCLUDED.status,
                    remaining_qty = EXCLUDED.remaining_qty,
                    risk_context = EXCLUDED.risk_context
                """,
                order_id,
                f"C-{order_id}",
                strategy_id,
                symbol,
                side,
                qty,
                status,
                remaining_qty,
                json.dumps({"risk_dollars": risk_dollars}),
            )
        await conn.execute(
            """
            INSERT INTO trades
                (trade_id, strategy_id, account_id, instrument_symbol, direction, quantity,
                 entry_ts, entry_price, exit_ts, exit_price, realized_r)
            VALUES ($1, $2, 'DU123', 'QQQ', 'LONG', 10, $3, 100, $4, 101, 0.5)
            ON CONFLICT (trade_id) DO UPDATE SET
                exit_ts = EXCLUDED.exit_ts,
                realized_r = EXCLUDED.realized_r
            """,
            f"{strategy_ids[0]}_T1",
            strategy_ids[0],
            now - timedelta(minutes=10),
            now,
        )


async def _cleanup_postgres_family_fixture(pool, strategy_ids: list[str]) -> None:
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM order_events WHERE strategy_id = ANY($1::text[])", strategy_ids)
        await conn.execute("DELETE FROM fills WHERE oms_order_id LIKE 'PARITY_%'")
        await conn.execute("DELETE FROM orders WHERE strategy_id = ANY($1::text[])", strategy_ids)
        await conn.execute("DELETE FROM positions WHERE strategy_id = ANY($1::text[])", strategy_ids)
        await conn.execute("DELETE FROM trades WHERE strategy_id = ANY($1::text[])", strategy_ids)
        await conn.execute("DELETE FROM strategy_signals WHERE strategy_id = ANY($1::text[])", strategy_ids)


def _entry_order(
    oms_order_id: str,
    strategy_id: str,
    symbol: str,
    side: OrderSide,
    qty: int,
    risk_dollars: float,
    *,
    remaining_qty: float = 0.0,
) -> OMSOrder:
    return OMSOrder(
        oms_order_id=oms_order_id,
        client_order_id=f"C-{oms_order_id}",
        strategy_id=strategy_id,
        account_id="DU123",
        instrument=_instrument(symbol),
        side=side,
        qty=qty,
        order_type=OrderType.LIMIT,
        role=OrderRole.ENTRY,
        status=OrderStatus.PARTIALLY_FILLED if remaining_qty else OrderStatus.ACKED,
        remaining_qty=remaining_qty,
        risk_context=RiskContext(
            stop_for_risk=90.0,
            planned_entry_price=100.0,
            risk_dollars=risk_dollars,
        ),
    )


def _position(
    strategy_id: str,
    symbol: str,
    net_qty: float,
    *,
    risk_dollars: float,
    risk_r: float,
) -> Position:
    return Position(
        account_id="DU123",
        instrument_symbol=symbol,
        strategy_id=strategy_id,
        net_qty=net_qty,
        avg_price=100.0,
        open_risk_dollars=risk_dollars,
        open_risk_R=risk_r,
    )


def _trade(
    trade_id: str,
    strategy_id: str,
    entry_ts: datetime,
    *,
    exit_ts: datetime | None,
    realized_r: Decimal | None = None,
) -> TradeRow:
    return TradeRow(
        trade_id=trade_id,
        strategy_id=strategy_id,
        account_id="DU123",
        instrument_symbol="QQQ",
        direction="LONG",
        quantity=10,
        entry_ts=entry_ts,
        entry_price=Decimal("100"),
        exit_ts=exit_ts,
        exit_price=Decimal("101") if exit_ts else None,
        realized_r=realized_r,
    )
