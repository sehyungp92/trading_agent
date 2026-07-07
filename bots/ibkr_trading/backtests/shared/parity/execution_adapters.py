from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable

from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitExit,
    SubmitMarketExit,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)


class ParityOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ParityOrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass(slots=True, frozen=True)
class ParitySimOrder:
    order_id: str
    symbol: str
    side: ParityOrderSide
    order_type: ParityOrderType
    qty: int
    stop_price: float = 0.0
    limit_price: float = 0.0
    tick_size: float = 0.0
    submit_time: datetime | None = None
    ttl_hours: int = 0
    ttl_minutes: int = 0
    tag: str = ""
    oca_group: str = ""
    invalidation_price: float = 0.0
    triggered_ts: datetime | None = None


def neutral_action_to_sim_order(
    action: (
        SubmitEntry
        | SubmitExit
        | SubmitAddOnEntry
        | SubmitProtectiveStop
        | ReplaceProtectiveStop
        | SubmitProfitTarget
        | SubmitPartialExit
        | SubmitMarketExit
        | FlattenPosition
    ),
    *,
    tick_size: float,
    submit_time: datetime | None = None,
) -> ParitySimOrder:
    if isinstance(action, (SubmitProtectiveStop, ReplaceProtectiveStop)):
        return ParitySimOrder(
            order_id=getattr(action, "target_order_id", "") or getattr(action, "client_order_id", "") or f"SIM-{action.symbol}-STOP",
            symbol=action.symbol,
            side=ParityOrderSide.BUY if action.side == "BUY" else ParityOrderSide.SELL,
            order_type=ParityOrderType.STOP,
            qty=action.qty,
            stop_price=action.stop_price,
            tick_size=tick_size,
            submit_time=submit_time,
            tag=getattr(action, "reason", "") or getattr(action, "role", ""),
            oca_group=getattr(action, "oca_group", ""),
        )

    if isinstance(action, SubmitProfitTarget):
        return ParitySimOrder(
            order_id=action.client_order_id,
            symbol=action.symbol,
            side=ParityOrderSide.BUY if action.side == "BUY" else ParityOrderSide.SELL,
            order_type=ParityOrderType.LIMIT,
            qty=action.qty,
            limit_price=action.limit_price,
            tick_size=tick_size,
            submit_time=submit_time,
            tag=action.role,
            oca_group=action.oca_group,
        )

    if isinstance(action, SubmitMarketExit):
        return ParitySimOrder(
            order_id=action.client_order_id,
            symbol=action.symbol,
            side=ParityOrderSide.BUY if action.side == "BUY" else ParityOrderSide.SELL,
            order_type=ParityOrderType.MARKET,
            qty=action.qty,
            tick_size=tick_size,
            submit_time=submit_time,
            tag=action.role,
            oca_group=action.oca_group,
        )

    if isinstance(action, FlattenPosition):
        side = action.side or "SELL"
        return ParitySimOrder(
            order_id=action.parent_order_id or f"SIM-{action.symbol}-FLATTEN",
            symbol=action.symbol,
            side=ParityOrderSide.BUY if side == "BUY" else ParityOrderSide.SELL,
            order_type=ParityOrderType.MARKET,
            qty=action.qty,
            tick_size=tick_size,
            submit_time=submit_time,
            tag=action.reason,
            oca_group=action.oca_group,
        )

    return ParitySimOrder(
        order_id=action.client_order_id,
        symbol=action.symbol,
        side=ParityOrderSide.BUY if action.side == "BUY" else ParityOrderSide.SELL,
        order_type=_order_type_for(action.order_type),
        qty=action.qty,
        stop_price=action.stop_price or 0.0,
        limit_price=getattr(action, "limit_price", None) or getattr(action, "price", None) or 0.0,
        tick_size=tick_size,
        submit_time=submit_time,
        tag=getattr(action, "role", "") or action.metadata.get("role", ""),
        oca_group=getattr(action, "oca_group", "") or action.metadata.get("oca_group", ""),
    )


def extract_terminal_cancels(
    actions: Iterable[CancelAction | FlattenPosition],
) -> tuple[list[CancelAction], list[FlattenPosition]]:
    cancels: list[CancelAction] = []
    flattens: list[FlattenPosition] = []
    for action in actions:
        if isinstance(action, CancelAction):
            cancels.append(action)
        elif isinstance(action, FlattenPosition):
            flattens.append(action)
    return cancels, flattens


def _order_type_for(order_type: str) -> ParityOrderType:
    mapping = {
        "LIMIT": ParityOrderType.LIMIT,
        "MARKET": ParityOrderType.MARKET,
        "STOP": ParityOrderType.STOP,
        "STOP_LIMIT": ParityOrderType.STOP_LIMIT,
    }
    return mapping[order_type]
