from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone

from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitExit,
)
from strategies.core.events import DecisionEvent
from strategies.core.idle_market import idle_market_details
from strategies.momentum.nqdtc.models import PositionState, WorkingOrder

from .state import (
    NQDTCCoreState,
    NQDTCEntryRequest,
    NQDTCFill,
    NQDTCOrderUpdate,
    NQDTCSimpleRequest,
)

_TERMINAL_STATUSES = {
    "cancelled",
    "expired",
    "rejected",
    "order_cancelled",
    "order_expired",
    "order_rejected",
}
_ACK_STATUSES = {"accepted", "acknowledged", "submitted"}


def on_bar(
    state: NQDTCCoreState,
    *,
    bar_count_5m: int | None = None,
    bar_ts: datetime | None = None,
    entry_request: NQDTCEntryRequest | None = None,
    stop_update: NQDTCSimpleRequest | None = None,
    cancel_order_ids: list[str] | None = None,
    flatten_request: NQDTCSimpleRequest | None = None,
    expire_orders: bool = False,
    idle_market_bars: Sequence[object] | None = None,
    idle_market_symbol: str = "",
    idle_market_timeframe: str = "5m",
) -> tuple[
    NQDTCCoreState,
    list[SubmitEntry | ReplaceProtectiveStop | CancelAction | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | ReplaceProtectiveStop | CancelAction | FlattenPosition] = []
    events: list[DecisionEvent] = []
    event_ts = bar_ts or datetime.now(timezone.utc)

    if bar_count_5m is not None:
        next_state.bar_count_5m = bar_count_5m
    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts

    if entry_request is not None:
        next_state.symbol = entry_request.symbol or next_state.symbol
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=entry_request.symbol,
                side="BUY" if entry_request.direction.value > 0 else "SELL",
                qty=entry_request.qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                price=entry_request.price,
                limit_price=entry_request.limit_price,
                stop_price=entry_request.stop_price,
                metadata={
                    "role": "entry",
                    "subtype": entry_request.subtype.value,
                    "oca_group": entry_request.oca_group,
                    "quality_mult": entry_request.quality_mult,
                    "stop_for_risk": entry_request.stop_for_risk,
                },
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_REQUESTED",
                ts=event_ts,
                symbol=entry_request.symbol,
                timeframe="5m",
                details={
                    "subtype": entry_request.subtype.value,
                    "direction": entry_request.direction.value,
                    "qty": entry_request.qty,
                    "stop_for_risk": entry_request.stop_for_risk,
                },
            )
        )

    if stop_update is not None and next_state.position.open and next_state.position.stop_oms_order_id:
        actions.append(
            ReplaceProtectiveStop(
                symbol=next_state.symbol or next_state.position.symbol,
                target_order_id=next_state.position.stop_oms_order_id,
                side="SELL" if next_state.position.direction.value > 0 else "BUY",
                stop_price=float(stop_update.price or 0.0),
                qty=stop_update.qty or next_state.position.qty_open,
                reason=stop_update.reason,
            )
        )
        events.append(
            DecisionEvent(
                code="STOP_REPLACEMENT_REQUESTED",
                ts=event_ts,
                symbol=next_state.symbol or next_state.position.symbol,
                timeframe="5m",
                details={"stop_price": stop_update.price, "reason": stop_update.reason},
            )
        )

    for order_id in cancel_order_ids or []:
        actions.append(
            CancelAction(
                symbol=next_state.symbol,
                target_order_id=order_id,
                reason="cancel_requested",
            )
        )
        events.append(
            DecisionEvent(
                code="ORDER_CANCEL_REQUESTED",
                ts=event_ts,
                symbol=next_state.symbol,
                timeframe="5m",
                details={"oms_order_id": order_id},
            )
        )

    if flatten_request is not None and next_state.position.open:
        actions.append(
            FlattenPosition(
                symbol=next_state.symbol or next_state.position.symbol,
                reason=flatten_request.reason,
            )
        )
        events.append(
            DecisionEvent(
                code="FLATTEN_REQUESTED",
                ts=event_ts,
                symbol=next_state.symbol or next_state.position.symbol,
                timeframe="5m",
                details={"reason": flatten_request.reason},
            )
        )

    if expire_orders:
        active_orders: list[WorkingOrder] = []
        for order in next_state.working_orders:
            bars_elapsed = next_state.bar_count_5m - order.submitted_bar_idx
            if bars_elapsed >= order.ttl_bars:
                actions.append(
                    CancelAction(
                        symbol=next_state.symbol,
                        target_order_id=order.oms_order_id,
                        reason="ttl_expiry",
                        metadata={"subtype": order.subtype.value},
                    )
                )
                events.append(
                    DecisionEvent(
                        code="ENTRY_EXPIRED",
                        ts=event_ts,
                        symbol=next_state.symbol,
                        timeframe="5m",
                        details={"oms_order_id": order.oms_order_id, "subtype": order.subtype.value},
                    )
                )
            else:
                active_orders.append(order)
        next_state.working_orders = active_orders

    if idle_market_bars is not None and not actions and not events:
        observed_symbol = idle_market_symbol or next_state.symbol
        events.append(
            DecisionEvent(
                code="IDLE_MARKET_OBSERVED",
                ts=event_ts,
                symbol=observed_symbol,
                timeframe=idle_market_timeframe,
                details=idle_market_details(
                    idle_market_bars,
                    symbol=observed_symbol,
                    timeframe=idle_market_timeframe,
                ),
            )
        )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_order_update(
    state: NQDTCCoreState,
    update: NQDTCOrderUpdate,
) -> tuple[NQDTCCoreState, list[SubmitExit], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit] = []
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)
    status = update.status.lower()

    if update.accepted_entry is not None and status in _ACK_STATUSES:
        accepted = update.accepted_entry
        next_state.symbol = accepted.symbol or next_state.symbol
        next_state.working_orders.append(
            WorkingOrder(
                oms_order_id=update.oms_order_id,
                subtype=accepted.subtype,
                direction=accepted.direction,
                price=float(accepted.price or accepted.stop_price or 0.0),
                qty=accepted.qty,
                submitted_bar_idx=accepted.submitted_bar_idx,
                ttl_bars=accepted.ttl_bars,
                oca_group=accepted.oca_group,
                is_limit=accepted.is_limit,
                quality_mult=accepted.quality_mult,
                stop_for_risk=accepted.stop_for_risk,
                expected_fill_price=float(accepted.price or accepted.stop_price or 0.0),
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_SUBMITTED",
                ts=event_ts,
                symbol=accepted.symbol,
                timeframe="5m",
                details={"oms_order_id": update.oms_order_id, "subtype": accepted.subtype.value},
            )
        )
    elif update.order_role == "stop" and next_state.position.open and status in _ACK_STATUSES:
        next_state.position.stop_oms_order_id = update.oms_order_id
        events.append(
            DecisionEvent(
                code="PROTECTIVE_STOP_SUBMITTED",
                ts=event_ts,
                symbol=next_state.position.symbol,
                timeframe="5m",
                details={"stop_oms_order_id": update.oms_order_id},
            )
        )
    elif status in _TERMINAL_STATUSES:
        next_state.working_orders = [
            order for order in next_state.working_orders if order.oms_order_id != update.oms_order_id
        ]
        if next_state.position.open and next_state.position.stop_oms_order_id == update.oms_order_id:
            next_state.position.stop_oms_order_id = ""
            events.append(
                DecisionEvent(
                    code="PROTECTIVE_STOP_CLEARED",
                    ts=event_ts,
                    symbol=next_state.position.symbol,
                    timeframe="5m",
                    details={"status": update.status},
                )
            )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: NQDTCCoreState,
    fill: NQDTCFill,
) -> tuple[NQDTCCoreState, list[SubmitExit], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit] = []
    events: list[DecisionEvent] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)

    matched_order = None
    for order in next_state.working_orders:
        if order.oms_order_id == fill.oms_order_id:
            matched_order = order
            break

    if matched_order is not None and fill.entry_context is not None:
        next_state.working_orders = [
            order for order in next_state.working_orders if order.oms_order_id != fill.oms_order_id
        ]
        qty = fill.fill_qty or matched_order.qty
        next_state.position = PositionState(
            open=True,
            symbol=next_state.symbol or next_state.position.symbol or "NQ",
            direction=matched_order.direction,
            entry_subtype=matched_order.subtype,
            entry_price=fill.fill_price,
            stop_price=matched_order.stop_for_risk,
            initial_stop_price=matched_order.stop_for_risk,
            qty=qty,
            qty_open=qty,
            R_dollars=fill.entry_context.r_dollars,
            quality_mult=matched_order.quality_mult,
            exit_tier=fill.entry_context.exit_tier,
            tp_levels=deepcopy(fill.entry_context.tp_levels),
            mm_level=fill.entry_context.mm_level,
            mm_reached=fill.entry_context.mm_reached,
            highest_since_entry=fill.fill_price,
            lowest_since_entry=fill.fill_price,
            box_high_at_entry=fill.entry_context.box_high_at_entry,
            box_low_at_entry=fill.entry_context.box_low_at_entry,
            box_mid_at_entry=fill.entry_context.box_mid_at_entry,
            entry_session=fill.entry_context.entry_session,
            tp1_only_cap=fill.entry_context.tp1_only_cap,
        )
        actions.append(
            SubmitExit(
                client_order_id=f"{fill.oms_order_id}:protective_stop",
                symbol=next_state.position.symbol,
                side="SELL" if matched_order.direction.value > 0 else "BUY",
                qty=qty,
                order_type="STOP",
                tif="GTC",
                stop_price=matched_order.stop_for_risk,
                metadata={"role": "protective_stop", "entry_oms_order_id": fill.oms_order_id},
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_FILLED",
                ts=event_ts,
                symbol=next_state.position.symbol,
                timeframe="5m",
                details={"fill_price": fill.fill_price, "qty": qty, "subtype": matched_order.subtype.value},
            )
        )
    elif next_state.position.open and (
        fill.oms_order_id == next_state.position.stop_oms_order_id or fill.exit_type
    ):
        symbol = next_state.position.symbol
        next_state.position = PositionState(symbol=symbol)
        events.append(
            DecisionEvent(
                code="EXIT_FILLED",
                ts=event_ts,
                symbol=symbol,
                timeframe="5m",
                details={"fill_price": fill.fill_price, "qty": fill.fill_qty, "exit_type": fill.exit_type},
            )
        )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def _update_last_decision(state: NQDTCCoreState, events: list[DecisionEvent]) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    state.last_bar_ts = latest.ts
