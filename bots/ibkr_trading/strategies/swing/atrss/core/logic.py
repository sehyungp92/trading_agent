from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitPartialExit,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.core.idle_market import idle_market_details
from strategies.swing.atrss.models import CandidateType, Direction, LegType, PositionBook, PositionLeg, ReentryState

from .state import (
    ATRSSAddOnARequest,
    ATRSSBarInput,
    ATRSSCoreState,
    ATRSSEntryRequest,
    ATRSSFill,
    ATRSSFlattenRequest,
    ATRSSOrderUpdate,
    ATRSSPartialExitRequest,
    ATRSSStopUpdateRequest,
)

_TERMINAL_STATUSES = {
    "cancelled",
    "expired",
    "rejected",
    "order_cancelled",
    "order_expired",
    "order_rejected",
}

_ACTIVE_ORDER_STATUSES = {
    "accepted",
    "open",
    "pending_submit",
    "submitted",
    "working",
    "order_accepted",
    "order_submitted",
    "order_working",
}


def build_core_state(engine) -> ATRSSCoreState:
    return ATRSSCoreState(
        daily_states=deepcopy(engine.daily_states),
        hourly_states=deepcopy(engine.hourly_states),
        positions=deepcopy(engine.positions),
        reentry_states=deepcopy(engine.reentry_states),
        pending_orders=deepcopy(engine.pending_orders),
        prev_trend_dirs=deepcopy(engine._prev_trend_dirs),
        halt_states=deepcopy(engine.halt_states),
        pending_reverses=deepcopy(engine._pending_reverses),
        pending_flattens=deepcopy(engine._pending_flattens),
        reopen_at=deepcopy(engine._reopen_at),
        breakout_arm_states=deepcopy(engine.breakout_arm_states),
        risk_halted=engine._risk_halted,
        risk_halt_reason=engine._risk_halt_reason,
        last_decision_code=engine._last_decision_code,
        last_decision_details=deepcopy(engine._last_decision_details),
        last_bar_ts=engine._last_bar_ts,
    )


def apply_core_state(engine, state: ATRSSCoreState) -> None:
    engine.daily_states = deepcopy(state.daily_states)
    engine.hourly_states = deepcopy(state.hourly_states)
    engine.positions = deepcopy(state.positions)
    engine.reentry_states = deepcopy(state.reentry_states)
    engine.pending_orders = deepcopy(state.pending_orders)
    engine._prev_trend_dirs = deepcopy(state.prev_trend_dirs)
    engine.halt_states = deepcopy(state.halt_states)
    engine._pending_reverses = deepcopy(state.pending_reverses)
    engine._pending_flattens = deepcopy(state.pending_flattens)
    engine._reopen_at = deepcopy(state.reopen_at)
    engine.breakout_arm_states = deepcopy(state.breakout_arm_states)
    engine._risk_halted = state.risk_halted
    engine._risk_halt_reason = state.risk_halt_reason
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


def on_bar(
    state: ATRSSCoreState,
    payload: ATRSSBarInput | None = None,
    *,
    bar_ts: datetime | None = None,
    entry_request: ATRSSEntryRequest | None = None,
    add_on_a_request: ATRSSAddOnARequest | None = None,
    stop_update: ATRSSStopUpdateRequest | None = None,
    partial_exit_request: ATRSSPartialExitRequest | None = None,
    flatten_request: ATRSSFlattenRequest | None = None,
    idle_market_bars: Sequence[object] | None = None,
    idle_market_symbol: str = "",
    idle_market_timeframe: str = "1h",
) -> tuple[
    ATRSSCoreState,
    list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition] = []
    events: list[DecisionEvent] = []

    if payload is not None and all(
        request is None
        for request in (entry_request, add_on_a_request, stop_update, partial_exit_request, flatten_request)
    ):
        events = _legacy_bar_events(payload)
        if payload.bar_ts is not None:
            next_state.last_bar_ts = payload.bar_ts
        _update_last_decision(next_state, events)
        return next_state, [], events

    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts
    event_ts = bar_ts or datetime.now(timezone.utc)

    if entry_request is not None:
        actions.append(_entry_action(entry_request))
        events.append(
            _event(
                code="ADD_ON_REQUESTED" if entry_request.candidate.type == CandidateType.ADDON_B else "ENTRY_REQUESTED",
                ts=event_ts,
                symbol=entry_request.symbol,
                details={
                    "candidate_type": entry_request.candidate.type.value,
                    "qty": entry_request.candidate.qty,
                    "trigger_price": entry_request.candidate.trigger_price,
                    "initial_stop": entry_request.candidate.initial_stop,
                },
            )
        )

    if add_on_a_request is not None:
        actions.append(
            SubmitAddOnEntry(
                client_order_id=add_on_a_request.client_order_id,
                symbol=add_on_a_request.symbol,
                side="BUY" if add_on_a_request.direction == Direction.LONG else "SELL",
                qty=add_on_a_request.qty,
                order_type=add_on_a_request.order_type,
                tif=add_on_a_request.tif,
                price=add_on_a_request.entry_price,
                risk_context={
                    "stop_for_risk": add_on_a_request.stop_price,
                    "planned_entry_price": add_on_a_request.entry_price,
                },
                metadata={"candidate_type": CandidateType.ADDON_A.value},
            )
        )
        events.append(
            _event(
                code="ADD_ON_REQUESTED",
                ts=event_ts,
                symbol=add_on_a_request.symbol,
                details={
                    "candidate_type": CandidateType.ADDON_A.value,
                    "qty": add_on_a_request.qty,
                    "entry_price": add_on_a_request.entry_price,
                    "initial_stop": add_on_a_request.stop_price,
                },
            )
        )

    if stop_update is not None:
        position = next_state.positions.get(stop_update.symbol)
        if position is not None and position.stop_oms_order_id:
            position.current_stop = stop_update.stop_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=position.stop_oms_order_id,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    stop_price=stop_update.stop_price,
                    qty=min(stop_update.qty, position.total_qty),
                    reason=stop_update.reason,
                )
            )
            events.append(
                _event(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=stop_update.symbol,
                    details={
                        "stop_price": stop_update.stop_price,
                        "qty": min(stop_update.qty, position.total_qty),
                        "reason": stop_update.reason,
                    },
                )
            )

    if partial_exit_request is not None:
        position = next_state.positions.get(partial_exit_request.symbol)
        if position is not None and position.direction != Direction.FLAT:
            actions.append(
                SubmitPartialExit(
                    client_order_id=partial_exit_request.client_order_id,
                    symbol=partial_exit_request.symbol,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    qty=min(partial_exit_request.qty, position.total_qty),
                    order_type=partial_exit_request.order_type,
                    tif=partial_exit_request.tif,
                    metadata={"reason": partial_exit_request.reason},
                )
            )
            events.append(
                _event(
                    code="PARTIAL_EXIT_REQUESTED",
                    ts=event_ts,
                    symbol=partial_exit_request.symbol,
                    details={
                        "qty": min(partial_exit_request.qty, position.total_qty),
                        "reason": partial_exit_request.reason,
                    },
                )
            )

    if flatten_request is not None:
        position = next_state.positions.get(flatten_request.symbol)
        if position is not None and position.direction != Direction.FLAT:
            actions.append(
                FlattenPosition(
                    symbol=flatten_request.symbol,
                    reason=flatten_request.reason,
                    side="SELL" if position.direction == Direction.LONG else "BUY",
                    qty=position.total_qty,
                )
            )
            events.append(
                _event(
                    code="FLATTEN_REQUESTED",
                    ts=event_ts,
                    symbol=flatten_request.symbol,
                    details={"reason": flatten_request.reason, "qty": position.total_qty},
                )
            )

    if idle_market_bars is not None and not actions and not events:
        events.append(
            _event(
                code="IDLE_MARKET_OBSERVED",
                ts=event_ts,
                symbol=idle_market_symbol,
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
    state: ATRSSCoreState,
    update: ATRSSOrderUpdate,
) -> tuple[ATRSSCoreState, list[ReplaceProtectiveStop], list[DecisionEvent]]:
    next_state = deepcopy(state)
    actions: list[ReplaceProtectiveStop] = []
    status = update.status.lower()
    event_ts = update.timestamp or datetime.now(timezone.utc)
    events: list[DecisionEvent] = []

    if status in _ACTIVE_ORDER_STATUSES and update.oms_order_id:
        meta = dict(update.metadata)
        symbol = meta.get("symbol", update.symbol)
        if update.order_role == "stop":
            position = next_state.positions.get(symbol)
            if position is not None:
                position.stop_oms_order_id = update.oms_order_id
                position.stop_pending = False
                events.append(
                    _event(
                        code="STOP_SUBMITTED",
                        ts=event_ts,
                        symbol=symbol,
                        details={
                            "status": status,
                            "qty": meta.get("qty", position.total_qty),
                            "stop_price": meta.get("stop_price", position.current_stop),
                        },
                    )
                )
        elif update.order_role == "flatten":
            next_state.pending_flattens[symbol] = {
                **meta,
                "oms_order_id": update.oms_order_id,
            }
            events.append(
                _event(
                    code="FLATTEN_ORDER_SUBMITTED",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status, "reason": meta.get("reason", ""), "qty": meta.get("qty", 0)},
                )
            )
        elif meta:
            next_state.pending_orders[update.oms_order_id] = meta
            if _candidate_type_value(meta.get("type")) == CandidateType.ADDON_A.value:
                position = next_state.positions.get(symbol)
                if position is not None:
                    position.addon_a_pending_id = update.oms_order_id
            events.append(
                _event(
                    code="ORDER_SUBMITTED",
                    ts=event_ts,
                    symbol=symbol,
                    details={
                        "status": status,
                        "order_role": update.order_role,
                        "order_type": _candidate_type_value(meta.get("type")),
                        "qty": meta.get("qty", 0),
                    },
                )
            )

    if status in _TERMINAL_STATUSES and update.oms_order_id:
        meta = next_state.pending_orders.pop(update.oms_order_id, None)
        if meta is not None:
            symbol = meta.get("symbol", update.symbol)
            if meta.get("type") == CandidateType.ADDON_A:
                position = next_state.positions.get(symbol)
                if position is not None and position.addon_a_pending_id == update.oms_order_id:
                    position.addon_a_pending_id = ""
            events.append(
                _event(
                    code="ORDER_TERMINAL",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status, "order_type": str(meta.get("type", ""))},
                )
            )
        else:
            for position in next_state.positions.values():
                if position.stop_oms_order_id == update.oms_order_id:
                    position.stop_oms_order_id = ""
                    events.append(
                        _event(
                            code="STOP_TERMINAL",
                            ts=event_ts,
                            symbol=position.symbol,
                            details={"status": status},
                        )
                    )
                    break

    if not events and update.decision_code:
        events.append(
            _event(
                code=update.decision_code,
                ts=event_ts,
                symbol=update.symbol,
                details=update.decision_details,
            )
        )

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, actions, events


def on_fill(
    state: ATRSSCoreState,
    fill: ATRSSFill,
) -> tuple[
    ATRSSCoreState,
    list[SubmitProtectiveStop | ReplaceProtectiveStop | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitProtectiveStop | ReplaceProtectiveStop | FlattenPosition] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)
    events: list[DecisionEvent] = []

    if fill.oms_order_id in next_state.pending_orders:
        meta = next_state.pending_orders.pop(fill.oms_order_id)
        symbol = meta["symbol"]
        direction = meta["direction"]
        raw_candidate_type = meta["type"]
        fill_price = fill.fill_price or meta.get("trigger_price", 0.0)
        fill_qty = fill.fill_qty or int(meta.get("qty", 0))

        if str(raw_candidate_type) == "PARTIAL_CLOSE":
            position = next_state.positions.get(symbol)
            if position is not None and position.base_leg is not None:
                partial_qty = int(meta.get("partial_qty", fill_qty))
                position.base_leg.qty = max(1, position.base_leg.qty - partial_qty)
                if position.stop_oms_order_id:
                    actions.append(
                        ReplaceProtectiveStop(
                            symbol=symbol,
                            target_order_id=position.stop_oms_order_id,
                            side="SELL" if position.direction == Direction.LONG else "BUY",
                            stop_price=position.current_stop,
                            qty=position.total_qty,
                            reason="partial_resize",
                        )
                    )
                events.append(
                    _event(
                        code="PARTIAL_EXIT_FILLED",
                        ts=event_ts,
                        symbol=symbol,
                        details={"qty": partial_qty, "price": fill_price, "reason": meta.get("reason", "PARTIAL")},
                    )
                )
            _update_last_decision(next_state, events, preserve_last_bar_ts=True)
            return next_state, actions, events

        candidate_type = (
            raw_candidate_type
            if isinstance(raw_candidate_type, CandidateType)
            else CandidateType(str(raw_candidate_type))
        )

        leg_type = _leg_type_for(candidate_type)
        leg = PositionLeg(
            leg_type=leg_type,
            qty=fill_qty,
            entry_price=fill_price,
            initial_stop=meta["initial_stop"],
            fill_time=event_ts,
            entry_commission=fill.commission,
            oms_order_id=fill.oms_order_id,
        )
        position = next_state.positions.get(symbol)

        if leg_type == LegType.BASE:
            position = PositionBook(
                symbol=symbol,
                direction=direction,
                legs=[leg],
                current_stop=meta["initial_stop"],
                mfe_price=fill_price,
                mae_price=fill_price,
                entry_time=event_ts,
                stop_pending=True,
            )
            next_state.positions[symbol] = position
            actions.append(
                SubmitProtectiveStop(
                    client_order_id=f"{symbol}-stop-{fill.oms_order_id}",
                    symbol=symbol,
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=fill_qty,
                    stop_price=meta["initial_stop"],
                )
            )
            events.append(
                _event(
                    code="ENTRY_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    details={
                        "candidate_type": candidate_type.value,
                        "qty": fill_qty,
                        "price": fill_price,
                        "stop_price": meta["initial_stop"],
                    },
                )
            )
        elif position is None:
            actions.append(
                FlattenPosition(
                    symbol=symbol,
                    reason=f"FLATTEN_ORPHANED_{candidate_type.value}",
                    side="SELL" if direction == Direction.LONG else "BUY",
                    qty=fill_qty,
                )
            )
            events.append(
                _event(
                    code="ORPHANED_ADDON_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    details={"candidate_type": candidate_type.value, "qty": fill_qty, "price": fill_price},
                )
            )
        else:
            position.legs.append(leg)
            if candidate_type == CandidateType.ADDON_A:
                position.addon_a_done = True
                if position.addon_a_pending_id == fill.oms_order_id:
                    position.addon_a_pending_id = ""
            if candidate_type == CandidateType.ADDON_B:
                position.addon_b_done = True
            if position.stop_oms_order_id:
                actions.append(
                    ReplaceProtectiveStop(
                        symbol=symbol,
                        target_order_id=position.stop_oms_order_id,
                        side="SELL" if position.direction == Direction.LONG else "BUY",
                        stop_price=position.current_stop,
                        qty=position.total_qty,
                        reason="add_on_resize",
                    )
                )
            events.append(
                _event(
                    code="ADD_ON_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    details={"candidate_type": candidate_type.value, "qty": fill_qty, "price": fill_price},
                )
            )

        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    flatten_symbol = next(
        (
            symbol
            for symbol, pending in next_state.pending_flattens.items()
            if pending.get("oms_order_id") == fill.oms_order_id
        ),
        None,
    )
    if flatten_symbol is not None:
        position = next_state.positions.pop(flatten_symbol, None)
        next_state.pending_flattens.pop(flatten_symbol, None)
        if position is not None:
            reentry = next_state.reentry_states.get(flatten_symbol, ReentryState())
            reentry.last_exit_time = event_ts
            reentry.last_exit_dir = position.direction
            reentry.last_exit_mfe = position.mfe
            reentry.last_exit_reason = fill.exit_type or "FLATTEN"
            reentry.reset_seen_long = False
            reentry.reset_seen_short = False
            next_state.reentry_states[flatten_symbol] = reentry
            events.append(
                _event(
                    code="EXIT_FILLED",
                    ts=event_ts,
                    symbol=flatten_symbol,
                    details={"reason": fill.exit_type or "FLATTEN", "price": fill.fill_price},
                )
            )
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    for symbol, position in list(next_state.positions.items()):
        if position.stop_oms_order_id != fill.oms_order_id:
            continue
        next_state.positions.pop(symbol, None)
        reentry = next_state.reentry_states.get(symbol, ReentryState())
        reentry.last_exit_time = event_ts
        reentry.last_exit_dir = position.direction
        reentry.last_exit_mfe = position.mfe
        reentry.last_exit_reason = fill.exit_type or "STOP"
        reentry.reset_seen_long = False
        reentry.reset_seen_short = False
        if position.mfe >= 1.0:
            if position.direction == Direction.LONG:
                reentry.voucher_long = True
            else:
                reentry.voucher_short = True
            reentry.voucher_granted_time = event_ts
        next_state.reentry_states[symbol] = reentry
        events.append(
            _event(
                code="STOP_FILLED",
                ts=event_ts,
                symbol=symbol,
                details={"price": fill.fill_price or position.current_stop, "mfe": position.mfe},
            )
        )
        break

    if not events and fill.decision_code:
        events.append(
            _event(
                code=fill.decision_code,
                ts=event_ts,
                symbol=fill.symbol,
                details=fill.decision_details,
            )
        )
    elif not events:
        events.append(
            _event(
                code="UNMATCHED_FILL",
                ts=event_ts,
                symbol=fill.symbol,
                details={
                    "oms_order_id": fill.oms_order_id,
                    "qty": fill.fill_qty,
                    "price": fill.fill_price,
                    "reason": "no_pending_order_or_position",
                },
            )
        )

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, actions, events


def _entry_action(request: ATRSSEntryRequest) -> SubmitEntry | SubmitAddOnEntry:
    action_cls = SubmitAddOnEntry if request.candidate.type == CandidateType.ADDON_B else SubmitEntry
    return action_cls(
        client_order_id=request.client_order_id,
        symbol=request.symbol,
        side="BUY" if request.candidate.direction == Direction.LONG else "SELL",
        qty=request.candidate.qty,
        order_type=request.order_type,
        tif=request.tif,
        stop_price=request.candidate.trigger_price,
        limit_price=request.limit_price,
        risk_context={
            "stop_for_risk": request.candidate.initial_stop,
            "planned_entry_price": request.candidate.trigger_price,
        },
        metadata={"candidate_type": request.candidate.type.value},
    )


def _leg_type_for(candidate_type: Any) -> LegType:
    if candidate_type == CandidateType.ADDON_A:
        return LegType.ADDON_A
    if candidate_type == CandidateType.ADDON_B:
        return LegType.ADDON_B
    return LegType.BASE


def _candidate_type_value(candidate_type: Any) -> str:
    if isinstance(candidate_type, CandidateType):
        return candidate_type.value
    return str(candidate_type or "")


def _legacy_bar_events(payload: ATRSSBarInput) -> list[DecisionEvent]:
    if not payload.decision_code:
        return []
    return [
        _event(
            code=payload.decision_code,
            ts=payload.bar_ts or datetime.now(timezone.utc),
            symbol=payload.symbol,
            details=payload.decision_details,
        )
    ]


def _event(*, code: str, ts: datetime, symbol: str, details: dict[str, Any]) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts, symbol=symbol, timeframe="1h", details=dict(details))


def _update_last_decision(
    state: ATRSSCoreState,
    events: list[DecisionEvent],
    *,
    preserve_last_bar_ts: bool = False,
) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    if latest.ts is not None and not preserve_last_bar_ts:
        state.last_bar_ts = latest.ts
