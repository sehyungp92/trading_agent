"""OMS order construction helpers for IARIC."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry
from libs.oms.models.order import EntryPolicy, OMSOrder, OrderRole, OrderSide, OrderType, RiskContext

from .config import STRATEGY_ID
from .models import PositionState, WatchlistItem


def build_stock_instrument(item: WatchlistItem) -> Instrument:
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


def build_entry_order(
    item: WatchlistItem,
    account_id: str,
    qty: int,
    limit_price: float,
    stop_for_risk: float,
    *,
    signal_id: str = "",
    bar_id: str = "",
    exchange_timestamp: datetime | None = None,
    trace_id: str = "",
    lineage_context: dict[str, Any] | None = None,
) -> OMSOrder:
    instrument = build_stock_instrument(item)
    return OMSOrder(
        client_order_id=f"{item.symbol}-entry-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=OrderSide.BUY,
        qty=qty,
        order_type=OrderType.LIMIT,
        limit_price=limit_price,
        role=OrderRole.ENTRY,
        entry_policy=EntryPolicy(ttl_seconds=30, max_reprices=0),
        risk_context=RiskContext(
            stop_for_risk=stop_for_risk,
            planned_entry_price=limit_price,
            risk_budget_tag="IARIC",
            signal_id=signal_id,
            bar_id=bar_id,
            exchange_timestamp=exchange_timestamp,
            trace_id=trace_id,
            lineage_context=dict(lineage_context or {}),
        ),
    )


def build_stop_order(item: WatchlistItem, account_id: str, qty: int, stop_price: float) -> OMSOrder:
    instrument = build_stock_instrument(item)
    return OMSOrder(
        client_order_id=f"{item.symbol}-stop-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=OrderSide.SELL,
        qty=qty,
        order_type=OrderType.STOP,
        stop_price=stop_price,
        role=OrderRole.STOP,
        tif="GTC",
    )


def build_market_exit(item: WatchlistItem, account_id: str, qty: int, role: OrderRole = OrderRole.EXIT) -> OMSOrder:
    instrument = build_stock_instrument(item)
    return OMSOrder(
        client_order_id=f"{item.symbol}-exit-{uuid.uuid4().hex[:12]}",
        strategy_id=STRATEGY_ID,
        account_id=account_id,
        instrument=instrument,
        side=OrderSide.SELL,
        qty=qty,
        order_type=OrderType.MARKET,
        role=role,
    )


def build_position_from_fill(
    fill_price: float,
    fill_qty: int,
    stop_price: float,
    fill_time,
    setup_tag: str = "UNCLASSIFIED",
) -> PositionState:
    risk_per_share = max(fill_price - stop_price, 0.01)
    return PositionState(
        entry_price=fill_price,
        qty_entry=fill_qty,
        qty_open=fill_qty,
        final_stop=stop_price,
        current_stop=stop_price,
        entry_time=fill_time,
        initial_risk_per_share=risk_per_share,
        max_favorable_price=fill_price,
        max_adverse_price=fill_price,
        setup_tag=setup_tag,
    )
