"""Vdubus NQ v4.0 core decision machine -- pure state transitions.

All methods take immutable inputs, deepcopy state, and return
(next_state, actions, events). No async, no OMS, no instrumentation.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from strategies.core.actions import (
    FlattenPosition,
    NeutralAction,
    ReplaceProtectiveStop,
    SubmitExit,
)
from strategies.core.events import DecisionEvent
from strategies.core.idle_market import idle_market_details
from strategies.momentum.vdub.config import STRATEGY_ID
from strategies.momentum.vdub.models import (
    Direction,
    PositionStage,
    PositionState,
)

from .state import (
    VdubCoreState,
    VdubEntrySubmitted,
    VdubFill,
    VdubFlattenRequest,
    VdubOrderUpdate,
    VdubPartialExitDone,
    VdubStopUpdateRequest,
)


# ── build / apply ────────────────────────────────────────────────


def build_core_state(engine) -> VdubCoreState:
    return VdubCoreState(
        regime=deepcopy(engine.regime),
        counters=deepcopy(engine.counters),
        positions=deepcopy(engine.positions),
        working_entries=deepcopy(engine.working_entries),
        event_state=deepcopy(engine.event_state),
        bar_idx=engine._bar_idx,
        last_reset_date=engine._last_reset_date,
        recent_wins=deepcopy(engine._recent_wins),
        last_flatten_oms_id=engine._last_flatten_oms_id,
        last_decision_code=engine._last_decision_code,
        last_decision_details=deepcopy(engine._last_decision_details),
        last_bar_ts=engine._last_bar_ts,
    )


def apply_core_state(engine, state: VdubCoreState) -> None:
    engine.regime = deepcopy(state.regime)
    engine.counters = deepcopy(state.counters)
    engine.positions = deepcopy(state.positions)
    engine.working_entries = deepcopy(state.working_entries)
    engine.event_state = deepcopy(state.event_state)
    engine._bar_idx = state.bar_idx
    engine._last_reset_date = state.last_reset_date
    engine._recent_wins = deepcopy(state.recent_wins)
    engine._last_flatten_oms_id = state.last_flatten_oms_id
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = deepcopy(state.last_decision_details)
    engine._last_bar_ts = state.last_bar_ts


# ── on_bar ───────────────────────────────────────────────────────


def on_bar(
    state: VdubCoreState,
    *,
    bar_ts: datetime | None = None,
    entry_submitted: VdubEntrySubmitted | None = None,
    stop_updates: list[VdubStopUpdateRequest] | None = None,
    flatten_requests: list[VdubFlattenRequest] | None = None,
    partial_exit_done: VdubPartialExitDone | None = None,
    decision_code: str = "",
    decision_details: dict[str, Any] | None = None,
    idle_market_bars: Sequence[object] | None = None,
    idle_market_symbol: str = "",
    idle_market_timeframe: str = "15m",
) -> tuple[VdubCoreState, list[NeutralAction], list[DecisionEvent]]:
    """Process a bar tick: register entries, update stops, flatten positions."""
    next_state = deepcopy(state)
    actions: list[NeutralAction] = []
    events: list[DecisionEvent] = []

    if bar_ts is not None:
        next_state.last_bar_ts = bar_ts

    # Register entry submission in working_entries
    if entry_submitted is not None:
        we = entry_submitted.working_entry
        next_state.working_entries[entry_submitted.oms_order_id] = deepcopy(we)
        events.append(DecisionEvent(
            code="ENTRY_SUBMITTED",
            ts=bar_ts or datetime.now(timezone.utc),
            symbol=STRATEGY_ID,
            timeframe="15m",
            details={
                "entry_type": we.entry_type.value,
                "direction": we.direction.name,
                "qty": we.qty,
                "stop_entry": we.stop_entry,
            },
        ))

    # Process stop updates
    for su in (stop_updates or []):
        for pos in next_state.positions:
            if pos.trade_id == su.pos_id and pos.qty_open > 0:
                old_stop = pos.stop_price
                pos.stop_price = su.new_stop
                side: str = "SELL" if pos.direction == Direction.LONG else "BUY"
                actions.append(ReplaceProtectiveStop(
                    symbol=STRATEGY_ID,
                    target_order_id=pos.stop_oms_order_id,
                    side=side,
                    stop_price=su.new_stop,
                    qty=pos.qty_open,
                    reason=su.reason,
                    metadata={"pos_id": su.pos_id, "old_stop": old_stop},
                ))
                events.append(DecisionEvent(
                    code="STOP_UPDATED",
                    ts=bar_ts or datetime.now(timezone.utc),
                    symbol=STRATEGY_ID,
                    timeframe="15m",
                    details={"pos_id": su.pos_id, "reason": su.reason,
                             "old": old_stop, "new": su.new_stop},
                ))
                break

    # Process flatten requests
    for fr in (flatten_requests or []):
        for pos in next_state.positions:
            if pos.trade_id == fr.pos_id and pos.qty_open > 0:
                actions.append(FlattenPosition(
                    symbol=STRATEGY_ID,
                    reason=fr.reason,
                    side="SELL" if pos.direction == Direction.LONG else "BUY",
                    qty=pos.qty_open,
                    metadata={"pos_id": fr.pos_id},
                ))
                events.append(DecisionEvent(
                    code="FLATTEN_REQUESTED",
                    ts=bar_ts or datetime.now(timezone.utc),
                    symbol=STRATEGY_ID,
                    timeframe="15m",
                    details={"pos_id": fr.pos_id, "reason": fr.reason},
                ))
                break

    # Handle partial exit completion (qty already reduced by engine)
    if partial_exit_done is not None:
        for pos in next_state.positions:
            if pos.trade_id == partial_exit_done.pos_id:
                pos.qty_open = partial_exit_done.new_qty
                events.append(DecisionEvent(
                    code="PARTIAL_EXIT_DONE",
                    ts=bar_ts or datetime.now(timezone.utc),
                    symbol=STRATEGY_ID,
                    timeframe="15m",
                    details={
                        "pos_id": partial_exit_done.pos_id,
                        "qty_closed": partial_exit_done.qty_closed,
                        "new_qty": partial_exit_done.new_qty,
                    },
                ))
                break

    # Record decision
    if decision_code:
        next_state.last_decision_code = decision_code
        next_state.last_decision_details = dict(decision_details or {})
        events.append(DecisionEvent(
            code=decision_code,
            ts=bar_ts or datetime.now(timezone.utc),
            symbol=STRATEGY_ID,
            timeframe="15m",
            details=dict(decision_details or {}),
        ))

    if idle_market_bars is not None and not actions and not events:
        details = idle_market_details(
            idle_market_bars,
            symbol=idle_market_symbol or STRATEGY_ID,
            timeframe=idle_market_timeframe,
        )
        next_state.last_decision_code = "IDLE_MARKET_OBSERVED"
        next_state.last_decision_details = details
        events.append(DecisionEvent(
            code="IDLE_MARKET_OBSERVED",
            ts=bar_ts or datetime.now(timezone.utc),
            symbol=idle_market_symbol or STRATEGY_ID,
            timeframe=idle_market_timeframe,
            details=details,
        ))

    return next_state, actions, events


# ── on_fill ──────────────────────────────────────────────────────


def on_fill(
    state: VdubCoreState,
    fill: VdubFill,
) -> tuple[VdubCoreState, list[NeutralAction], list[DecisionEvent]]:
    """Process a fill: entry, stop, or flatten confirmation."""
    next_state = deepcopy(state)
    actions: list[NeutralAction] = []
    events: list[DecisionEvent] = []

    # Entry fill (engine attaches entry_context when oms_id matches working entry)
    if fill.entry_context is not None:
        _process_entry_fill(next_state, fill, actions, events)
        _update_fill_last_decision(next_state, events, fill.fill_time)
        return next_state, actions, events

    # Flatten fill confirmation
    if (next_state.last_flatten_oms_id
            and fill.oms_order_id == next_state.last_flatten_oms_id):
        _process_flatten_fill(next_state, fill, actions, events)
        _update_fill_last_decision(next_state, events, fill.fill_time)
        return next_state, actions, events

    # Stop fill (match against position stop orders)
    for pos in list(next_state.positions):
        if pos.stop_oms_order_id == fill.oms_order_id:
            _process_stop_fill(next_state, pos, fill, actions, events)
            _update_fill_last_decision(next_state, events, fill.fill_time)
            return next_state, actions, events

    _update_fill_last_decision(next_state, events, fill.fill_time)
    return next_state, actions, events


def _process_entry_fill(
    state: VdubCoreState,
    fill: VdubFill,
    actions: list[NeutralAction],
    events: list[DecisionEvent],
) -> None:
    """Handle entry fill: create position, place protective stop."""
    we = fill.entry_context.working_entry
    now = fill.fill_time or datetime.now(timezone.utc)

    # Create position
    r_points = abs(fill.fill_price - we.initial_stop)
    pos = PositionState(
        trade_id=we.oms_order_id,
        direction=we.direction,
        entry_price=fill.fill_price,
        stop_price=we.initial_stop,
        qty_entry=fill.fill_qty,
        qty_open=fill.fill_qty,
        r_points=r_points,
        stage=PositionStage.ACTIVE_RISK,
        entry_time=now,
        highest_since_entry=fill.fill_price,
        lowest_since_entry=fill.fill_price,
        vwap_used_at_entry=we.vwap_used,
        is_addon=we.is_addon,
        entry_type=we.entry_type,
        entry_session=we.session,
        is_flip_entry=we.is_flip,
        class_mult=we.class_mult,
    )
    state.positions.append(pos)

    # Remove from working entries
    state.working_entries.pop(fill.oms_order_id, None)

    # Update counters
    if we.direction == Direction.LONG:
        state.counters.long_fills += 1
    else:
        state.counters.short_fills += 1

    # Request protective stop placement
    side: str = "SELL" if we.direction == Direction.LONG else "BUY"
    actions.append(SubmitExit(
        client_order_id=f"{pos.trade_id}-stop",
        symbol=STRATEGY_ID,
        side=side,
        qty=fill.fill_qty,
        order_type="STOP",
        stop_price=we.initial_stop,
        metadata={"pos_id": pos.trade_id, "role": "protective_stop",
                  "stop_price": we.initial_stop},
    ))

    events.append(DecisionEvent(
        code="ENTRY_FILLED",
        ts=now,
        symbol=STRATEGY_ID,
        timeframe="15m",
        details={
            "trade_id": pos.trade_id,
            "direction": we.direction.name,
            "entry_type": we.entry_type.value,
            "fill_price": fill.fill_price,
            "qty": fill.fill_qty,
            "initial_stop": we.initial_stop,
            "r_points": round(r_points, 2),
            "is_flip": we.is_flip,
            "is_addon": we.is_addon,
        },
    ))


def _process_stop_fill(
    state: VdubCoreState,
    pos: PositionState,
    fill: VdubFill,
    actions: list[NeutralAction],
    events: list[DecisionEvent],
) -> None:
    """Handle stop fill: compute PnL, close position, update counters."""
    pv = fill.point_value
    now = fill.fill_time or datetime.now(timezone.utc)

    pnl_pts = ((fill.fill_price - pos.entry_price)
               if pos.direction == Direction.LONG
               else (pos.entry_price - fill.fill_price))
    realized_usd = pnl_pts * pv * pos.qty_open
    r = pnl_pts / pos.r_points if pos.r_points > 0 else 0.0

    # Update counters
    state.counters.daily_realized_pnl += realized_usd
    state.recent_wins.append(pnl_pts > 0)

    # Capture details before closing
    trade_id = pos.trade_id
    bars_held = pos.bars_since_entry
    peak_mfe_r = pos.peak_mfe_r
    peak_mae_r = pos.peak_mae_r

    # Remove position
    pos.qty_open = 0
    state.positions = [p for p in state.positions if p.qty_open > 0]

    events.append(DecisionEvent(
        code="STOP_FILLED",
        ts=now,
        symbol=STRATEGY_ID,
        timeframe="15m",
        details={
            "trade_id": trade_id,
            "fill_price": fill.fill_price,
            "pnl_pts": round(pnl_pts, 2),
            "realized_usd": round(realized_usd, 2),
            "r": round(r, 4),
            "direction": pos.direction.name,
            "bars_held": bars_held,
            "peak_mfe_r": peak_mfe_r,
            "peak_mae_r": peak_mae_r,
        },
    ))


def _process_flatten_fill(
    state: VdubCoreState,
    fill: VdubFill,
    actions: list[NeutralAction],
    events: list[DecisionEvent],
) -> None:
    """Handle flatten order fill: clear tracking, close matching position."""
    state.last_flatten_oms_id = None
    now = fill.fill_time or datetime.now(timezone.utc)
    pv = fill.point_value

    for pos in list(state.positions):
        if pos.qty_open > 0:
            pnl_pts = ((fill.fill_price - pos.entry_price)
                       if pos.direction == Direction.LONG
                       else (pos.entry_price - fill.fill_price))
            realized_usd = pnl_pts * pv * pos.qty_open
            r = pnl_pts / pos.r_points if pos.r_points > 0 else 0.0

            state.counters.daily_realized_pnl += realized_usd
            state.recent_wins.append(pnl_pts > 0)

            trade_id = pos.trade_id
            pos.qty_open = 0

            events.append(DecisionEvent(
                code="FLATTEN_FILLED",
                ts=now,
                symbol=STRATEGY_ID,
                timeframe="15m",
                details={
                    "trade_id": trade_id,
                    "fill_price": fill.fill_price,
                    "pnl_pts": round(pnl_pts, 2),
                    "realized_usd": round(realized_usd, 2),
                    "r": round(r, 4),
                },
            ))

    state.positions = [p for p in state.positions if p.qty_open > 0]


def _update_fill_last_decision(
    state: VdubCoreState,
    events: list[DecisionEvent],
    fill_time: datetime | None,
) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    state.last_decision_details = dict(latest.details)
    state.last_bar_ts = latest.ts or fill_time or datetime.now(timezone.utc)


# ── on_order_update ──────────────────────────────────────────────


_TERMINAL = {"cancelled", "expired", "rejected", "cancel_confirmed", "error"}


def on_order_update(
    state: VdubCoreState,
    update: VdubOrderUpdate,
) -> tuple[VdubCoreState, list[NeutralAction], list[DecisionEvent]]:
    """Handle order status changes: cancels/rejects clean up state."""
    next_state = deepcopy(state)
    actions: list[NeutralAction] = []
    events: list[DecisionEvent] = []
    status = update.status.lower()

    if status in _TERMINAL:
        _handle_terminal(next_state, update, actions, events)

    return next_state, actions, events


def _handle_terminal(
    state: VdubCoreState,
    update: VdubOrderUpdate,
    actions: list[NeutralAction],
    events: list[DecisionEvent],
) -> None:
    """Handle terminal order status: clean up working entries or detect lost stops."""
    oms_id = update.oms_order_id
    now = update.timestamp or datetime.now(timezone.utc)

    # Check working entries (pending entry orders)
    if oms_id in state.working_entries:
        state.working_entries.pop(oms_id)
        events.append(DecisionEvent(
            code="ENTRY_CANCELLED",
            ts=now, symbol=STRATEGY_ID, timeframe="15m",
            details={"oms_order_id": oms_id},
        ))
        return

    # Check flatten order failure -- resubmit emergency flatten
    if state.last_flatten_oms_id and oms_id == state.last_flatten_oms_id:
        state.last_flatten_oms_id = None
        if state.positions:
            actions.append(FlattenPosition(
                symbol=STRATEGY_ID,
                reason="FLATTEN_RETRY",
                metadata={"original_oms_id": oms_id},
            ))
            events.append(DecisionEvent(
                code="FLATTEN_RETRY",
                ts=now, symbol=STRATEGY_ID, timeframe="15m",
                details={"oms_order_id": oms_id},
            ))
        return

    # Check stop orders -- position left unprotected
    for pos in state.positions:
        if pos.stop_oms_order_id == oms_id and pos.qty_open > 0:
            pos.stop_oms_order_id = ""
            actions.append(FlattenPosition(
                symbol=STRATEGY_ID,
                reason="STOP_LOST",
                side="SELL" if pos.direction == Direction.LONG else "BUY",
                qty=pos.qty_open,
                metadata={"pos_id": pos.trade_id, "lost_stop_oms_id": oms_id},
            ))
            events.append(DecisionEvent(
                code="STOP_LOST",
                ts=now, symbol=STRATEGY_ID, timeframe="15m",
                details={"pos_id": pos.trade_id, "oms_order_id": oms_id},
            ))
            return
