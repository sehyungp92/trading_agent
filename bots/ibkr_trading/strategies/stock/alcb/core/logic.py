from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Iterable

from strategies.core.actions import FlattenPosition, ReplaceProtectiveStop, SubmitEntry, SubmitExit, SubmitPartialExit
from strategies.core.events import DecisionEvent
from strategies.core.idle_market import idle_market_details
from strategies.stock.alcb.models import Direction, EntryType, PositionPlan, T2PositionState

from .state import (
    ALCBCoreState,
    ALCBEntryRequest,
    ALCBFill,
    ALCBFlattenRequest,
    ALCBOrderUpdate,
    ALCBPartialExitRequest,
    ALCBStopUpdateRequest,
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


def build_core_state(engine) -> ALCBCoreState:
    return ALCBCoreState(
        positions=deepcopy(engine._positions),
        or_data=deepcopy(engine._or_data),
        or_built=deepcopy(engine._or_built),
        order_index=deepcopy(engine._order_index),
        pending_entries=deepcopy(engine._pending_entries),
        pending_exits=deepcopy(engine._pending_exits),
        pending_plans=deepcopy(engine._pending_plans),
        entry_meta=deepcopy(engine._entry_meta),
        exit_reasons=deepcopy(engine._exit_reasons),
        last_decision_code=engine._last_decision_code,
        last_decision_details=deepcopy(engine._last_decision_details),
        last_bar_ts=engine._last_bar_ts,
    )


def apply_core_state(engine, state: ALCBCoreState) -> None:
    engine._positions = deepcopy(state.positions)
    engine._or_data = deepcopy(state.or_data)
    engine._or_built = deepcopy(state.or_built)
    engine._order_index = deepcopy(state.order_index)
    engine._pending_entries = deepcopy(state.pending_entries)
    engine._pending_exits = deepcopy(state.pending_exits)
    engine._pending_plans = deepcopy(state.pending_plans)
    engine._entry_meta = deepcopy(state.entry_meta)
    engine._exit_reasons = deepcopy(state.exit_reasons)
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


def apply_carry_roll(state: ALCBCoreState, carried_symbols: Iterable[str]) -> ALCBCoreState:
    next_state = deepcopy(state)
    carried = {symbol.upper() for symbol in carried_symbols}
    for symbol, position in next_state.positions.items():
        if symbol.upper() in carried:
            position.carry_days += 1
    return next_state


def on_bar(
    state: ALCBCoreState,
    *,
    bar_ts: datetime | None = None,
    entry_request: ALCBEntryRequest | None = None,
    stop_update: ALCBStopUpdateRequest | None = None,
    partial_exit_request: ALCBPartialExitRequest | None = None,
    flatten_request: ALCBFlattenRequest | None = None,
    idle_market_bars: Sequence[object] | None = None,
    idle_market_symbol: str = "",
    idle_market_timeframe: str = "5m",
) -> tuple[
    ALCBCoreState,
    list[SubmitEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition] = []
    events: list[DecisionEvent] = []
    event_ts = bar_ts or datetime.now(timezone.utc)

    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts

    if entry_request is not None:
        side = "BUY" if entry_request.plan.direction == Direction.LONG else "SELL"
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=entry_request.symbol,
                side=side,
                qty=entry_request.plan.quantity,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                price=entry_request.plan.entry_price,
                stop_price=entry_request.plan.entry_price,
                metadata={"entry_type": entry_request.plan.entry_type.value, **dict(entry_request.meta)},
                risk_context={
                    "stop_for_risk": entry_request.plan.stop_price,
                    "planned_entry_price": entry_request.plan.entry_price,
                    "risk_dollars": entry_request.plan.risk_dollars,
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
                    "entry_type": entry_request.plan.entry_type.value,
                    "qty": entry_request.plan.quantity,
                    "entry_price": entry_request.plan.entry_price,
                },
            )
        )

    if stop_update is not None:
        position = next_state.positions.get(stop_update.symbol)
        if position is not None and position.stop_order_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=position.stop_order_id,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    stop_price=stop_update.stop_price,
                    qty=stop_update.qty,
                    reason=stop_update.reason,
                    metadata={"trade_id": position.trade_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=stop_update.symbol,
                    timeframe="5m",
                    details={"stop_price": stop_update.stop_price, "reason": stop_update.reason},
                )
            )

    if partial_exit_request is not None:
        position = next_state.positions.get(partial_exit_request.symbol)
        if position is not None:
            actions.append(
                SubmitPartialExit(
                    client_order_id=partial_exit_request.client_order_id,
                    symbol=partial_exit_request.symbol,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    qty=partial_exit_request.qty,
                    order_type=partial_exit_request.order_type,
                    tif=partial_exit_request.tif,
                    metadata={"trade_id": position.trade_id, "reason": partial_exit_request.reason},
                )
            )
            events.append(
                DecisionEvent(
                    code="PARTIAL_EXIT_REQUESTED",
                    ts=event_ts,
                    symbol=partial_exit_request.symbol,
                    timeframe="5m",
                    details={"qty": partial_exit_request.qty, "reason": partial_exit_request.reason},
                )
            )

    if flatten_request is not None:
        position = next_state.positions.get(flatten_request.symbol)
        if position is not None:
            actions.append(
                FlattenPosition(
                    symbol=flatten_request.symbol,
                    reason=flatten_request.reason,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    qty=position.quantity,
                    metadata={"trade_id": position.trade_id},
                )
            )
            events.append(
                DecisionEvent(
                    code="FLATTEN_REQUESTED",
                    ts=event_ts,
                    symbol=flatten_request.symbol,
                    timeframe="5m",
                    details={"reason": flatten_request.reason},
                )
            )

    if idle_market_bars is not None and not actions and not events:
        events.append(
            DecisionEvent(
                code="IDLE_MARKET_OBSERVED",
                ts=event_ts,
                symbol=idle_market_symbol,
                timeframe=idle_market_timeframe,
                details=idle_market_details(
                    idle_market_bars,
                    symbol=idle_market_symbol,
                    timeframe=idle_market_timeframe,
                ),
            )
        )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_order_update(
    state: ALCBCoreState,
    update: ALCBOrderUpdate,
) -> tuple[ALCBCoreState, list[SubmitExit], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit] = []
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)
    status = update.status.lower()

    if update.accepted_entry is not None and status in _ACK_STATUSES:
        accepted = deepcopy(update.accepted_entry)
        next_state.order_index[update.oms_order_id] = (accepted.symbol, "ENTRY")
        next_state.pending_entries[accepted.symbol] = update.oms_order_id
        next_state.pending_plans[update.oms_order_id] = accepted.plan
        next_state.entry_meta[update.oms_order_id] = dict(accepted.meta)
        events.append(
            DecisionEvent(
                code="ENTRY_SUBMITTED",
                ts=event_ts,
                symbol=accepted.symbol,
                timeframe="5m",
                details={"oms_order_id": update.oms_order_id, "entry_type": accepted.plan.entry_type.value},
            )
        )
    elif status in _ACK_STATUSES and update.symbol:
        symbol = update.symbol
        role = update.order_role.upper()
        next_state.order_index[update.oms_order_id] = (symbol, role)
        if role in {"EXIT", "PARTIAL"}:
            next_state.pending_exits[symbol] = update.oms_order_id
            if update.reason:
                next_state.exit_reasons[update.oms_order_id] = update.reason
        elif role == "STOP":
            position = next_state.positions.get(symbol)
            if position is not None:
                position.stop_order_id = update.oms_order_id
            events.append(
                DecisionEvent(
                    code="PROTECTIVE_STOP_SUBMITTED",
                    ts=event_ts,
                    symbol=symbol,
                    timeframe="5m",
                    details={"stop_oms_order_id": update.oms_order_id},
                )
            )
    elif status in _TERMINAL_STATUSES:
        lookup = next_state.order_index.pop(update.oms_order_id, None)
        if lookup is not None:
            symbol, role = lookup
            if role == "ENTRY":
                next_state.pending_entries.pop(symbol, None)
                next_state.pending_plans.pop(update.oms_order_id, None)
                next_state.entry_meta.pop(update.oms_order_id, None)
            elif role in {"EXIT", "PARTIAL"}:
                next_state.pending_exits.pop(symbol, None)
                next_state.exit_reasons.pop(update.oms_order_id, None)
            elif role == "STOP":
                position = next_state.positions.get(symbol)
                if position is not None and position.stop_order_id == update.oms_order_id:
                    position.stop_order_id = ""
            events.append(
                DecisionEvent(
                    code="ORDER_TERMINATED",
                    ts=event_ts,
                    symbol=symbol,
                    timeframe="5m",
                    details={"oms_order_id": update.oms_order_id, "role": role, "status": update.status},
                )
            )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: ALCBCoreState,
    fill: ALCBFill,
) -> tuple[ALCBCoreState, list[SubmitExit | ReplaceProtectiveStop], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[SubmitExit | ReplaceProtectiveStop] = []
    events: list[DecisionEvent] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)

    lookup = next_state.order_index.pop(fill.oms_order_id, None)
    if lookup is None:
        return next_state, actions, events

    symbol, role = lookup
    if role == "ENTRY":
        next_state.pending_entries.pop(symbol, None)
        plan = next_state.pending_plans.pop(fill.oms_order_id, None)
        meta = next_state.entry_meta.pop(fill.oms_order_id, {})
        qty = fill.fill_qty
        if qty <= 0:
            return next_state, actions, events
        if plan is None:
            emergency_stop = (
                fill.entry_context.emergency_stop
                if fill.entry_context is not None and fill.entry_context.emergency_stop is not None
                else round(fill.fill_price * 0.98, 2)
            )
            plan = PositionPlan(
                symbol=symbol,
                direction=Direction.LONG,
                entry_type=EntryType.OR_BREAKOUT,
                entry_price=fill.fill_price,
                stop_price=emergency_stop,
                tp1_price=0.0,
                tp2_price=0.0,
                quantity=qty,
                risk_per_share=max(fill.fill_price - emergency_stop, 0.01),
                risk_dollars=qty * max(fill.fill_price - emergency_stop, 0.01),
                quality_mult=1.0,
                regime_mult=1.0,
                corr_mult=1.0,
            )
        direction = plan.direction
        position = T2PositionState(
            symbol=symbol,
            direction=direction,
            entry_price=fill.fill_price,
            stop_price=plan.stop_price,
            current_stop=plan.stop_price,
            quantity=qty,
            qty_original=qty,
            risk_per_share=max(abs(fill.fill_price - plan.stop_price), 0.01),
            entry_time=event_ts,
            entry_type=meta.get("entry_type", plan.entry_type.value),
            sector=meta.get("sector", ""),
            regime_tier=meta.get("regime_tier", ""),
            momentum_score=meta.get("momentum_score", 0),
            avwap_at_entry=meta.get("avwap", 0.0),
            or_high=meta.get("or_high", 0.0),
            or_low=meta.get("or_low", 0.0),
            max_favorable=fill.fill_price,
            max_adverse=fill.fill_price,
            setup_tag=f"T2_{meta.get('entry_type', plan.entry_type.value)}",
            trade_id=fill.entry_context.trade_id if fill.entry_context is not None else "",
        )
        position.entry_commission = fill.commission
        next_state.positions[symbol] = position
        actions.append(
            SubmitExit(
                client_order_id=f"{fill.oms_order_id}:protective_stop",
                symbol=symbol,
                side="SELL" if direction == Direction.LONG else "BUY",
                qty=qty,
                order_type="STOP",
                tif="GTC",
                stop_price=position.current_stop,
                metadata={"role": "protective_stop", "entry_oms_order_id": fill.oms_order_id},
            )
        )
        events.append(
            DecisionEvent(
                code="ENTRY_FILLED",
                ts=event_ts,
                symbol=symbol,
                timeframe="5m",
                details={"fill_price": fill.fill_price, "qty": qty, "entry_type": position.entry_type},
            )
        )
    else:
        position = next_state.positions.get(symbol)
        if position is None:
            next_state.pending_exits.pop(symbol, None)
            next_state.exit_reasons.pop(fill.oms_order_id, None)
            return next_state, actions, events

        exit_qty = min(fill.fill_qty or position.quantity, position.quantity)
        reason = next_state.exit_reasons.pop(fill.oms_order_id, fill.exit_type or role)
        position.quantity -= exit_qty
        position.exit_commission += fill.commission

        if role == "PARTIAL":
            direction_mult = 1 if position.direction == Direction.LONG else -1
            position.partial_taken = True
            position.partial_qty_exited += exit_qty
            position.realized_partial_pnl += (fill.fill_price - position.entry_price) * exit_qty * direction_mult
            next_state.pending_exits.pop(symbol, None)
            if position.stop_order_id:
                actions.append(
                    ReplaceProtectiveStop(
                        symbol=symbol,
                        target_order_id=position.stop_order_id,
                        side="SELL" if position.direction == Direction.LONG else "BUY",
                        stop_price=position.current_stop,
                        qty=position.quantity,
                        reason="partial_fill_resize",
                        metadata={"trade_id": position.trade_id},
                    )
                )
            events.append(
                DecisionEvent(
                    code="PARTIAL_EXIT_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    timeframe="5m",
                    details={"fill_price": fill.fill_price, "qty": exit_qty, "reason": reason},
                )
            )
        elif position.quantity <= 0:
            next_state.positions.pop(symbol, None)
            next_state.pending_exits.pop(symbol, None)
            events.append(
                DecisionEvent(
                    code="EXIT_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    timeframe="5m",
                    details={"fill_price": fill.fill_price, "qty": exit_qty, "reason": reason},
                )
            )
        else:
            next_state.pending_exits.pop(symbol, None)
            events.append(
                DecisionEvent(
                    code="EXIT_PARTIALLY_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    timeframe="5m",
                    details={"fill_price": fill.fill_price, "qty": exit_qty, "reason": reason},
                )
            )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def _update_last_decision(state: ALCBCoreState, events: list[DecisionEvent]) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    state.last_bar_ts = latest.ts
