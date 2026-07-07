from __future__ import annotations

from dataclasses import replace

from strategies.core.actions import FlattenPosition, SubmitEntry, SubmitProfitTarget, SubmitProtectiveStop
from strategies.core.events import DecisionEvent
from strategies.scalp._shared.nq_contract import spec_for
from strategies.scalp.po3_reversal.config import EntryType, SetupTier, TradeDirection
from strategies.scalp.po3_reversal.models import Po3Setup

from .state import Po3BarInput, Po3Fill, Po3FlattenRequest, Po3ReversalCoreState


def on_bar(
    state: Po3ReversalCoreState,
    payload: Po3BarInput,
    *,
    flatten_request: Po3FlattenRequest | None = None,
) -> tuple[Po3ReversalCoreState, list, list[DecisionEvent]]:
    if flatten_request is not None:
        setup = state.position
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
        return state, [action], [_event("PO3_FLATTEN_REQUESTED", payload, flatten_request.symbol)]

    if (
        payload.qty <= 0
        or not payload.risk_approved
        or payload.tier is SetupTier.NONE
        or payload.signal_score < payload.signal_threshold
        or payload.direction is TradeDirection.FLAT
    ):
        return state, [], [_event(payload.decision_code or "PO3_NO_SIGNAL", payload, payload.symbol)]
    if state.active_setup is not None:
        return state, [], []
    setup_id = f"po3-{payload.symbol}-{payload.bar_ts:%Y%m%d%H%M%S}"
    setup = Po3Setup(
        setup_id=setup_id,
        symbol=payload.symbol,
        direction=payload.direction,
        tier=payload.tier,
        entry_type=EntryType.STOP_CONFIRMATION,
        signal_time=payload.bar_ts,
        score=payload.signal_score,
        entry_price=payload.entry_price,
        stop_price=payload.stop_price,
        target_price=payload.target_price,
        qty=payload.qty,
        rr=payload.rr,
        metadata=dict(payload.decision_details),
    )
    state.active_setup = setup
    order_id = f"{setup_id}-entry"
    state.order_to_setup[order_id] = setup_id
    state.order_kind[order_id] = "entry"
    side = "BUY" if payload.direction is TradeDirection.LONG else "SELL"
    action = SubmitEntry(
        client_order_id=order_id,
        symbol=payload.symbol,
        side=side,
        qty=payload.qty,
        order_type="STOP",
        stop_price=payload.entry_price,
        metadata={"setup_id": setup_id, "tier": payload.tier.value, "entry_type": setup.entry_type.value},
        risk_context={"setup_id": setup_id},
    )
    return state, [action], [_event("PO3_SIGNAL_ACCEPTED", payload, payload.symbol)]


def on_fill(
    state: Po3ReversalCoreState,
    fill: Po3Fill,
) -> tuple[Po3ReversalCoreState, list, list[DecisionEvent]]:
    setup = state.active_setup
    if setup is None:
        return state, [], []
    role = fill.order_role or state.order_kind.get(fill.oms_order_id, "")
    if role == "entry":
        updated = replace(
            setup,
            qty_open=fill.fill_qty,
            avg_entry=fill.fill_price,
            metadata={**setup.metadata, "_entry_commission": fill.commission},
        )
        state.active_setup = updated
        state.position = updated
        return state, _child_orders(updated), [_fill_event("PO3_ENTRY_FILLED", fill)]

    qty = min(fill.fill_qty, setup.qty_open or fill.fill_qty)
    direction = 1 if setup.direction is TradeDirection.LONG else -1
    gross = (fill.fill_price - setup.avg_entry) * direction * qty * spec_for(fill.symbol).point_value
    pnl = gross - fill.commission - float(setup.metadata.get("_entry_commission", 0.0))
    remaining = max(0, (setup.qty_open or qty) - qty)
    if remaining:
        updated = replace(setup, qty_open=remaining, metadata={**setup.metadata, "_entry_commission": 0.0})
        state.active_setup = updated
        state.position = updated
    else:
        state.active_setup = None
        state.position = None
    state.daily_pnl += pnl
    state.weekly_pnl += pnl
    return state, [], [_fill_event("PO3_EXIT_FILLED", fill, pnl=pnl)]


def _child_orders(setup: Po3Setup) -> list:
    exit_side = "SELL" if setup.direction is TradeDirection.LONG else "BUY"
    return [
        SubmitProtectiveStop(
            client_order_id=f"{setup.setup_id}-stop",
            symbol=setup.symbol,
            side=exit_side,
            qty=setup.qty,
            stop_price=setup.stop_price,
            oca_group=setup.setup_id,
            metadata={"setup_id": setup.setup_id},
        ),
        SubmitProfitTarget(
            client_order_id=f"{setup.setup_id}-target",
            symbol=setup.symbol,
            side=exit_side,
            qty=setup.qty,
            limit_price=setup.target_price,
            oca_group=setup.setup_id,
            metadata={"setup_id": setup.setup_id, "stop_for_risk": setup.stop_price},
        ),
    ]


def _event(code: str, payload: Po3BarInput, symbol: str) -> DecisionEvent:
    return DecisionEvent(code=code, ts=payload.bar_ts, symbol=symbol, timeframe="1m", details=dict(payload.decision_details))


def _fill_event(code: str, fill: Po3Fill, *, pnl: float = 0.0) -> DecisionEvent:
    return DecisionEvent(
        code=code,
        ts=fill.fill_time,
        symbol=fill.symbol,
        timeframe="fill",
        details={"order_id": fill.oms_order_id, "pnl": pnl},
    )
