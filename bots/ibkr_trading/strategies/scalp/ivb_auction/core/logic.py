from __future__ import annotations

from dataclasses import replace

from strategies.core.actions import FlattenPosition, SubmitEntry, SubmitProfitTarget, SubmitProtectiveStop
from strategies.core.events import DecisionEvent
from strategies.scalp._shared.nq_contract import spec_for
from strategies.scalp.ivb_auction.config import IvbModule, TradeDirection
from strategies.scalp.ivb_auction.models import IvbSetup

from .state import IvbAuctionCoreState, IvbBarInput, IvbFill, IvbFlattenRequest


def on_bar(
    state: IvbAuctionCoreState,
    payload: IvbBarInput,
    *,
    flatten_request: IvbFlattenRequest | None = None,
) -> tuple[IvbAuctionCoreState, list, list[DecisionEvent]]:
    if flatten_request is not None:
        setup = state.active_setups.get(flatten_request.setup_id)
        if setup is None or setup.qty_open <= 0:
            return state, [], []
        side = "SELL" if setup.direction is TradeDirection.LONG else "BUY"
        action = FlattenPosition(
            symbol=flatten_request.symbol,
            reason=flatten_request.reason,
            side=side,
            qty=setup.qty_open,
            metadata={"setup_id": setup.setup_id},
        )
        event = _event("IVB_FLATTEN_REQUESTED", payload, setup.symbol)
        return state, [action], [event]

    if payload.qty <= 0 or payload.module is None or payload.trigger is None:
        return state, [], [_event(payload.decision_code or "IVB_NO_SIGNAL", payload, payload.symbol)]
    if any(setup.symbol == payload.symbol and setup.qty_open >= 0 for setup in state.active_setups.values()):
        return state, [], []

    setup_id = f"ivb-{payload.symbol}-{payload.bar_ts:%Y%m%d%H%M%S}"
    setup = IvbSetup(
        setup_id=setup_id,
        symbol=payload.symbol,
        module=payload.module,
        direction=payload.breakout_direction,
        trigger=payload.trigger,
        signal_time=payload.bar_ts,
        score=payload.signal_score,
        entry_price=payload.entry_price,
        stop_price=payload.stop_price,
        tp1_price=payload.tp1_price,
        tp2_price=payload.tp2_price,
        qty=payload.qty,
        size_multiplier=payload.size_multiplier,
        rr_to_tp1=payload.rr_to_tp1,
        metadata=dict(payload.decision_details),
    )
    state.active_setups[setup_id] = setup
    order_id = f"{setup_id}-entry"
    state.order_to_setup[order_id] = setup_id
    state.order_kind[order_id] = "entry"
    side = "BUY" if payload.breakout_direction is TradeDirection.LONG else "SELL"
    action = SubmitEntry(
        client_order_id=order_id,
        symbol=payload.symbol,
        side=side,
        qty=payload.qty,
        order_type="STOP",
        stop_price=payload.entry_price,
        metadata={"setup_id": setup_id, "module": payload.module.value},
        risk_context={"setup_id": setup_id},
    )
    return state, [action], [_event("IVB_SIGNAL_ACCEPTED", payload, payload.symbol)]


def on_fill(
    state: IvbAuctionCoreState,
    fill: IvbFill,
) -> tuple[IvbAuctionCoreState, list, list[DecisionEvent]]:
    setup_id = state.order_to_setup.get(fill.oms_order_id, "")
    setup = state.active_setups.get(setup_id)
    if setup is None:
        return state, [], []
    role = fill.order_role or state.order_kind.get(fill.oms_order_id, "")
    if role == "entry":
        updated = replace(
            setup,
            qty_open=fill.fill_qty,
            avg_entry=fill.fill_price,
            metadata={**setup.metadata, "_remaining_entry_commission": fill.commission},
        )
        state.active_setups[setup_id] = updated
        state.positions[fill.symbol] = updated
        return state, _child_orders(updated), [_fill_event("IVB_ENTRY_FILLED", fill)]

    qty = min(fill.fill_qty, setup.qty_open or fill.fill_qty)
    direction = 1 if setup.direction is TradeDirection.LONG else -1
    gross = (fill.fill_price - setup.avg_entry) * direction * qty * spec_for(fill.symbol).point_value
    entry_commission = float(setup.metadata.get("_remaining_entry_commission", 0.0))
    pnl = gross - fill.commission - entry_commission
    remaining = max(0, (setup.qty_open or qty) - qty)
    updated = replace(setup, qty_open=remaining, metadata={**setup.metadata, "_remaining_entry_commission": 0.0})
    if remaining:
        state.active_setups[setup_id] = updated
        state.positions[fill.symbol] = updated
    else:
        state.active_setups.pop(setup_id, None)
        state.positions.pop(fill.symbol, None)
    state.daily_pnl += pnl
    state.weekly_pnl += pnl
    return state, [], [_fill_event("IVB_EXIT_FILLED", fill, pnl=pnl)]


def _child_orders(setup: IvbSetup) -> list:
    exit_side = "SELL" if setup.direction is TradeDirection.LONG else "BUY"
    stop_id = f"{setup.setup_id}-stop"
    target_id = f"{setup.setup_id}-target"
    return [
        SubmitProtectiveStop(
            client_order_id=stop_id,
            symbol=setup.symbol,
            side=exit_side,
            qty=setup.qty,
            stop_price=setup.stop_price,
            oca_group=setup.setup_id,
            metadata={"setup_id": setup.setup_id},
        ),
        SubmitProfitTarget(
            client_order_id=target_id,
            symbol=setup.symbol,
            side=exit_side,
            qty=setup.qty,
            limit_price=setup.tp1_price,
            oca_group=setup.setup_id,
            metadata={"setup_id": setup.setup_id, "stop_for_risk": setup.stop_price},
        ),
    ]


def _event(code: str, payload: IvbBarInput, symbol: str) -> DecisionEvent:
    return DecisionEvent(code=code, ts=payload.bar_ts, symbol=symbol, timeframe="1m", details=dict(payload.decision_details))


def _fill_event(code: str, fill: IvbFill, *, pnl: float = 0.0) -> DecisionEvent:
    return DecisionEvent(
        code=code,
        ts=fill.fill_time,
        symbol=fill.symbol,
        timeframe="fill",
        details={"order_id": fill.oms_order_id, "pnl": pnl},
    )
