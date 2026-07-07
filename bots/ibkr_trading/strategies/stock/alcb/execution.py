"""OMS order construction helpers for ALCB."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry
from libs.oms.models.order import EntryPolicy, OMSOrder, OrderRole, OrderSide, OrderType, RiskContext

from .config import STRATEGY_ID
from .models import CandidateItem, Direction, EntryType, PositionPlan, PositionState


def build_stock_instrument(item: CandidateItem) -> Instrument:
    instrument = Instrument(
        symbol=item.symbol,
        root=item.symbol,
        venue=item.exchange,
        primary_exchange=item.primary_exchange,
        sec_type="STK",
        tick_size=item.tick_size,
        tick_value=item.tick_size,
        multiplier=1.0,
        point_value=item.point_value,
        currency=item.currency,
    )
    InstrumentRegistry.register(instrument)
    return instrument


def _entry_side(direction: Direction) -> OrderSide:
    return OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL


def _exit_side(direction: Direction) -> OrderSide:
    return OrderSide.SELL if direction == Direction.LONG else OrderSide.BUY


def _entry_limit_price(item: CandidateItem, plan: PositionPlan) -> float:
    if plan.entry_type != EntryType.B_SWEEP_RECLAIM:
        return plan.entry_price
    offset = max(item.tick_size, 0.01) * 2.0
    if plan.direction == Direction.LONG:
        return plan.entry_price + offset
    return max(item.tick_size, plan.entry_price - offset)


def build_entry_order(
    item: CandidateItem,
    account_id: str,
    plan: PositionPlan,
    *,
    signal_id: str = "",
    bar_id: str = "",
    exchange_timestamp: datetime | None = None,
    trace_id: str = "",
    lineage_context: dict[str, Any] | None = None,
) -> OMSOrder:
    instrument = build_stock_instrument(item)
    order_type = OrderType.LIMIT
    limit_price = _entry_limit_price(item, plan)
    return OMSOrder(
        client_order_id=f"{item.symbol}-entry-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=_entry_side(plan.direction),
        qty=plan.quantity,
        order_type=order_type,
        limit_price=limit_price,
        role=OrderRole.ENTRY,
        entry_policy=EntryPolicy(ttl_seconds=1800, max_reprices=0),
        risk_context=RiskContext(
            stop_for_risk=plan.stop_price,
            planned_entry_price=plan.entry_price,
            risk_budget_tag="ALCB",
            signal_id=signal_id,
            bar_id=bar_id,
            exchange_timestamp=exchange_timestamp,
            trace_id=trace_id,
            lineage_context=dict(lineage_context or {}),
        ),
    )


def build_stop_order(
    item: CandidateItem,
    account_id: str,
    qty: int,
    stop_price: float,
    direction: Direction,
    *,
    oca_group: str = "",
    oca_type: int = 0,
) -> OMSOrder:
    instrument = build_stock_instrument(item)
    return OMSOrder(
        client_order_id=f"{item.symbol}-stop-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=_exit_side(direction),
        qty=qty,
        order_type=OrderType.STOP,
        stop_price=stop_price,
        tif="GTC",
        role=OrderRole.STOP,
        oca_group=oca_group,
        oca_type=oca_type,
    )


def build_tp_order(
    item: CandidateItem,
    account_id: str,
    qty: int,
    tp_price: float,
    direction: Direction,
    tag: str,
    *,
    oca_group: str = "",
    oca_type: int = 0,
) -> OMSOrder:
    instrument = build_stock_instrument(item)
    return OMSOrder(
        client_order_id=f"{item.symbol}-{tag.lower()}-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=_exit_side(direction),
        qty=qty,
        order_type=OrderType.LIMIT,
        limit_price=tp_price,
        tif="GTC",
        role=OrderRole.TP,
        oca_group=oca_group,
        oca_type=oca_type,
    )


def build_market_exit(
    item: CandidateItem,
    account_id: str,
    qty: int,
    direction: Direction,
    role: OrderRole = OrderRole.EXIT,
    *,
    oca_group: str = "",
    oca_type: int = 0,
) -> OMSOrder:
    instrument = build_stock_instrument(item)
    return OMSOrder(
        client_order_id=f"{item.symbol}-exit-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=_exit_side(direction),
        qty=qty,
        order_type=OrderType.MARKET,
        role=role,
        oca_group=oca_group,
        oca_type=oca_type,
    )


def build_position_from_fill(
    *,
    direction: Direction,
    fill_price: float,
    fill_qty: int,
    stop_price: float,
    tp1_price: float,
    tp2_price: float,
    fill_time,
    setup_tag: str = "UNCLASSIFIED",
) -> PositionState:
    risk_per_share = max(abs(fill_price - stop_price), 0.01)
    return PositionState(
        direction=direction,
        entry_price=fill_price,
        qty_entry=fill_qty,
        qty_open=fill_qty,
        final_stop=stop_price,
        current_stop=stop_price,
        entry_time=fill_time,
        initial_risk_per_share=risk_per_share,
        max_favorable_price=fill_price,
        max_adverse_price=fill_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        setup_tag=setup_tag,
        opened_trade_date=fill_time.date(),
    )
