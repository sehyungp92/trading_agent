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
from strategies.swing.akc_helix.allocator import apply_initial_risk_basis
from strategies.swing.akc_helix.models import Direction, SetupInstance, SetupState

from .state import (
    AKCHelixBarInput,
    AKCHelixCoreState,
    AKCHelixEntryRequest,
    AKCHelixFill,
    AKCHelixFlattenRequest,
    AKCHelixOrderUpdate,
    AKCHelixPartialExitRequest,
    AKCHelixStopUpdateRequest,
)

_TERMINAL_STATUSES = {
    "cancelled",
    "expired",
    "rejected",
    "order_cancelled",
    "order_expired",
    "order_rejected",
}


def build_core_state(engine) -> AKCHelixCoreState:
    return AKCHelixCoreState(
        daily_states=deepcopy(engine.daily_states),
        tf_states=deepcopy(engine.tf_states),
        pivots=deepcopy(engine.pivots),
        regime_4h=deepcopy(engine.regime_4h),
        div_mag_history=deepcopy(engine.div_mag_history),
        active_setups=deepcopy(engine.active_setups),
        pending_setups=deepcopy(engine.pending_setups),
        queued_setups=deepcopy(engine.queued_setups),
        circuit_breakers=deepcopy(engine.circuit_breakers),
        order_to_setup=deepcopy(engine._order_to_setup),
        oca_counter=engine._oca_counter,
        last_b_long_l2_ts=deepcopy(engine._last_b_long_l2_ts),
        last_b_short_h2_ts=deepcopy(engine._last_b_short_h2_ts),
        last_d_long_l2_ts=deepcopy(engine._last_d_long_l2_ts),
        last_d_short_h2_ts=deepcopy(engine._last_d_short_h2_ts),
        regime_streaks=deepcopy(engine._regime_streaks),
        prev_regimes=deepcopy(engine._prev_regimes),
        risk_halted=engine._risk_halted,
        risk_halt_reason=engine._risk_halt_reason,
        last_decision_code=engine._last_decision_code,
        last_decision_details=deepcopy(engine._last_decision_details),
        last_bar_ts=engine._last_bar_ts,
    )


def apply_core_state(engine, state: AKCHelixCoreState) -> None:
    engine.daily_states = deepcopy(state.daily_states)
    engine.tf_states = deepcopy(state.tf_states)
    engine.pivots = deepcopy(state.pivots)
    engine.regime_4h = deepcopy(state.regime_4h)
    engine.div_mag_history = deepcopy(state.div_mag_history)
    engine.active_setups = deepcopy(state.active_setups)
    engine.pending_setups = deepcopy(state.pending_setups)
    engine.queued_setups = deepcopy(state.queued_setups)
    engine.circuit_breakers = deepcopy(state.circuit_breakers)
    engine._order_to_setup = deepcopy(state.order_to_setup)
    engine._oca_counter = state.oca_counter
    engine._last_b_long_l2_ts = deepcopy(state.last_b_long_l2_ts)
    engine._last_b_short_h2_ts = deepcopy(state.last_b_short_h2_ts)
    engine._last_d_long_l2_ts = deepcopy(state.last_d_long_l2_ts)
    engine._last_d_short_h2_ts = deepcopy(state.last_d_short_h2_ts)
    engine._regime_streaks = deepcopy(state.regime_streaks)
    engine._prev_regimes = deepcopy(state.prev_regimes)
    engine._risk_halted = state.risk_halted
    engine._risk_halt_reason = state.risk_halt_reason
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


def on_bar(
    state: AKCHelixCoreState,
    payload: AKCHelixBarInput | None = None,
    *,
    bar_ts: datetime | None = None,
    entry_request: AKCHelixEntryRequest | None = None,
    stop_update: AKCHelixStopUpdateRequest | None = None,
    partial_exit_request: AKCHelixPartialExitRequest | None = None,
    flatten_request: AKCHelixFlattenRequest | None = None,
    idle_market_bars: Sequence[object] | None = None,
    idle_market_symbol: str = "",
    idle_market_timeframe: str = "1h",
) -> tuple[
    AKCHelixCoreState,
    list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | SubmitAddOnEntry | ReplaceProtectiveStop | SubmitPartialExit | FlattenPosition] = []
    events: list[DecisionEvent] = []

    if payload is not None and all(
        request is None
        for request in (entry_request, stop_update, partial_exit_request, flatten_request)
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
        is_add = entry_request.order_role == "add"
        setup = deepcopy(_find_setup(next_state, entry_request.setup.setup_id) or entry_request.setup)
        action_qty = int(entry_request.qty if entry_request.qty is not None else setup.qty_planned)
        if is_add:
            setup.add_done = True
            if setup.setup_id in next_state.active_setups:
                next_state.active_setups[setup.setup_id] = setup
            elif setup.state is SetupState.ACTIVE or setup.qty_open > 0:
                setup.state = SetupState.ACTIVE
                next_state.active_setups[setup.setup_id] = setup
                next_state.pending_setups.pop(setup.setup_id, None)
            else:
                setup.state = SetupState.ARMED
                next_state.pending_setups[setup.setup_id] = setup
        else:
            setup.state = SetupState.ARMED
            if entry_request.order_role == "catchup":
                setup.catchup_order_id = entry_request.client_order_id
            elif entry_request.order_role == "rescue":
                setup.rescue_order_id = entry_request.client_order_id
            else:
                setup.primary_order_id = entry_request.client_order_id
            next_state.pending_setups[setup.setup_id] = setup
        next_state.order_to_setup[entry_request.client_order_id] = setup.setup_id
        action_cls = SubmitAddOnEntry if is_add else SubmitEntry
        actions.append(
            action_cls(
                client_order_id=entry_request.client_order_id,
                symbol=setup.symbol,
                side="BUY" if setup.direction == Direction.LONG else "SELL",
                qty=action_qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                stop_price=setup.bos_level if entry_request.order_type in {"STOP", "STOP_LIMIT"} else None,
                limit_price=entry_request.limit_price,
                oca_group=setup.oca_group,
                role=entry_request.order_role,
                risk_context={
                    "stop_for_risk": setup.stop0,
                    "planned_entry_price": setup.bos_level,
                },
                metadata={"setup_id": setup.setup_id, "setup_class": setup.setup_class.value},
            )
        )
        events.append(
            _event(
                code="ADD_REQUESTED" if action_cls is SubmitAddOnEntry else "ENTRY_REQUESTED",
                ts=event_ts,
                symbol=setup.symbol,
                details={"setup_id": setup.setup_id, "qty": action_qty, "bos_level": setup.bos_level},
            )
        )

    if stop_update is not None:
        setup = _find_setup(next_state, stop_update.setup_id)
        if setup is not None and setup.stop_order_id:
            setup.current_stop = stop_update.stop_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=setup.stop_order_id,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    stop_price=stop_update.stop_price,
                    qty=min(stop_update.qty, max(setup.qty_open, 0)),
                    reason=stop_update.reason,
                )
            )
            events.append(
                _event(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=stop_update.symbol,
                    details={"setup_id": setup.setup_id, "stop_price": stop_update.stop_price},
                )
            )

    if partial_exit_request is not None:
        setup = _find_setup(next_state, partial_exit_request.setup_id)
        if setup is not None and setup.qty_open > 0:
            actions.append(
                SubmitPartialExit(
                    client_order_id=partial_exit_request.client_order_id,
                    symbol=partial_exit_request.symbol,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    qty=min(partial_exit_request.qty, setup.qty_open),
                    order_type=partial_exit_request.order_type,
                    tif=partial_exit_request.tif,
                    metadata={"setup_id": setup.setup_id, "reason": partial_exit_request.reason},
                )
            )
            events.append(
                _event(
                    code="PARTIAL_EXIT_REQUESTED",
                    ts=event_ts,
                    symbol=partial_exit_request.symbol,
                    details={"setup_id": setup.setup_id, "qty": min(partial_exit_request.qty, setup.qty_open)},
                )
            )

    if flatten_request is not None:
        setup = _find_setup(next_state, flatten_request.setup_id)
        if setup is not None and setup.qty_open > 0:
            setup.state = SetupState.CLOSING
            actions.append(
                FlattenPosition(
                    symbol=flatten_request.symbol,
                    reason=flatten_request.reason,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    qty=setup.qty_open,
                )
            )
            events.append(
                _event(
                    code="FLATTEN_REQUESTED",
                    ts=event_ts,
                    symbol=flatten_request.symbol,
                    details={"setup_id": setup.setup_id, "reason": flatten_request.reason},
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
    state: AKCHelixCoreState,
    update: AKCHelixOrderUpdate,
) -> tuple[AKCHelixCoreState, list[ReplaceProtectiveStop], list[DecisionEvent]]:
    next_state = deepcopy(state)
    events: list[DecisionEvent] = []
    event_ts = update.timestamp or datetime.now(timezone.utc)

    if update.status.lower() in _TERMINAL_STATUSES and update.oms_order_id:
        setup_id = next_state.order_to_setup.pop(update.oms_order_id, "")
        setup = _find_setup(next_state, setup_id)
        if setup is not None:
            if setup.stop_order_id == update.oms_order_id:
                setup.stop_order_id = ""
            if setup.primary_order_id == update.oms_order_id:
                setup.primary_order_id = ""
            if setup.catchup_order_id == update.oms_order_id:
                setup.catchup_order_id = ""
            if setup.rescue_order_id == update.oms_order_id:
                setup.rescue_order_id = ""
            events.append(
                _event(
                    code="ORDER_TERMINAL",
                    ts=event_ts,
                    symbol=setup.symbol,
                    details={"setup_id": setup.setup_id, "status": update.status.lower(), "role": update.order_role},
                )
            )

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
    return next_state, [], events


def on_fill(
    state: AKCHelixCoreState,
    fill: AKCHelixFill,
) -> tuple[
    AKCHelixCoreState,
    list[SubmitProtectiveStop | ReplaceProtectiveStop],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    events: list[DecisionEvent] = []
    actions: list[SubmitProtectiveStop | ReplaceProtectiveStop] = []
    event_ts = fill.fill_time or datetime.now(timezone.utc)

    setup_id = next_state.order_to_setup.pop(fill.oms_order_id, "")
    setup = _find_setup(next_state, setup_id)
    if setup is None:
        if fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=fill.symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events, preserve_last_bar_ts=True)
        return next_state, actions, events

    fill_price = fill.fill_price or setup.bos_level
    fill_qty = fill.fill_qty or setup.qty_planned

    if fill.order_role in {"entry", "catchup", "rescue", "add", "unknown"} and fill.exit_type == "":
        if setup.primary_order_id == fill.oms_order_id:
            setup.primary_order_id = ""
        if setup.catchup_order_id == fill.oms_order_id:
            setup.catchup_order_id = ""
        if setup.rescue_order_id == fill.oms_order_id:
            setup.rescue_order_id = ""
        if setup.setup_id in next_state.pending_setups:
            next_state.pending_setups.pop(setup.setup_id, None)
        setup.state = SetupState.ACTIVE
        previous_qty = max(int(setup.qty_open), 0)
        previous_basis = setup.avg_entry_price or setup.fill_price or fill_price
        is_initial_fill = previous_qty <= 0
        if previous_qty <= 0:
            setup.fill_price = fill_price
            setup.avg_entry_price = fill_price
        else:
            setup.avg_entry_price = (
                (previous_basis * previous_qty) + (fill_price * fill_qty)
            ) / (previous_qty + fill_qty)
        setup.fill_qty += fill_qty
        setup.qty_open += fill_qty
        setup.fill_ts = event_ts
        setup.current_stop = setup.current_stop or setup.stop0
        if is_initial_fill:
            actual_r_price = abs(fill_price - setup.stop0)
            if actual_r_price > 0:
                setup.r_price = actual_r_price
            apply_initial_risk_basis(
                setup,
                fill_price,
                fill_qty,
                fill.point_value,
                setup.target_initial_risk_dollars,
            )
        next_state.active_setups[setup.setup_id] = setup
        if setup.fill_qty == fill_qty:
            actions.append(
                SubmitProtectiveStop(
                    client_order_id=f"{setup.symbol}-stop-{setup.setup_id}",
                    symbol=setup.symbol,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    qty=setup.qty_open,
                    stop_price=setup.current_stop,
                    oca_group=setup.oca_group,
                )
            )
            events.append(
                _event(
                    code="ENTRY_FILLED",
                    ts=event_ts,
                    symbol=setup.symbol,
                    details={"setup_id": setup.setup_id, "qty": fill_qty, "price": fill_price},
                )
            )
        elif setup.stop_order_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=setup.symbol,
                    target_order_id=setup.stop_order_id,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    stop_price=setup.current_stop,
                    qty=setup.qty_open,
                    reason="add_resize",
                )
            )
            events.append(
                _event(
                    code="ADD_FILLED",
                    ts=event_ts,
                    symbol=setup.symbol,
                    details={"setup_id": setup.setup_id, "qty": fill_qty, "price": fill_price},
                )
            )
    elif fill.order_role == "partial":
        exit_qty = min(fill_qty, setup.qty_open)
        setup.qty_open = max(0, setup.qty_open - exit_qty)
        if setup.qty_open > 0 and setup.stop_order_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=setup.symbol,
                    target_order_id=setup.stop_order_id,
                    side="SELL" if setup.direction == Direction.LONG else "BUY",
                    stop_price=setup.current_stop,
                    qty=setup.qty_open,
                    reason="partial_resize",
                )
            )
        events.append(
            _event(
                code="PARTIAL_EXIT_FILLED" if setup.qty_open > 0 else "EXIT_FILLED",
                ts=event_ts,
                symbol=setup.symbol,
                details={"setup_id": setup.setup_id, "qty": exit_qty, "price": fill_price},
            )
        )
        if setup.qty_open <= 0:
            setup.state = SetupState.CLOSED
            next_state.active_setups.pop(setup.setup_id, None)
    elif fill.order_role == "stop":
        if setup.stop_order_id == fill.oms_order_id:
            setup.stop_order_id = ""
        setup.qty_open = 0
        setup.state = SetupState.CLOSED
        next_state.active_setups.pop(setup.setup_id, None)
        events.append(
            _event(
                code="STOP_FILLED",
                ts=event_ts,
                symbol=setup.symbol,
                details={"setup_id": setup.setup_id, "price": fill_price},
            )
        )

    _update_last_decision(next_state, events, preserve_last_bar_ts=True)
    return next_state, actions, events


def _legacy_bar_events(payload: AKCHelixBarInput) -> list[DecisionEvent]:
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


def _find_setup(state: AKCHelixCoreState, setup_id: str) -> SetupInstance | None:
    if not setup_id:
        return None
    return state.active_setups.get(setup_id) or state.pending_setups.get(setup_id) or state.queued_setups.get(setup_id)


def _event(*, code: str, ts: datetime, symbol: str, details: dict[str, Any]) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts, symbol=symbol, timeframe="1h", details=dict(details))


def _update_last_decision(
    state: AKCHelixCoreState,
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
