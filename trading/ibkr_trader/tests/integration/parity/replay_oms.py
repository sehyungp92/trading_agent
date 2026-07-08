from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry
from libs.oms.models.intent import Intent, IntentType, PreapprovedFamilyDecision
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType, RiskContext
from libs.oms.persistence.in_memory import InMemoryRepository
from libs.oms.services.factory import build_multi_strategy_oms, build_oms_service
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
from tests.integration.parity.broker_matching import order_matches as _order_matches
from tests.integration.parity.oms_state import (
    blocked_reasons_from_repo_events as _blocked_reasons_from_events,
    drain_queue as _drain_queue,
    ledger_from_repo as _ledger_from_repo,
    plain_dataclass as _plain_dataclass,
    portfolio_rules_state as _portfolio_rules_state,
)
from tests.integration.parity.family_decisions import validate_family_decision_payload
from tests.integration.parity.fake_ibkr import FakeIBKRExecutionAdapter
from tests.integration.parity.oms_hydration import (
    build_instruments_from_fixture,
    hydrate_repository_from_fixture,
)
from tests.integration.parity.portfolio_rules import portfolio_rules_config_from_fixture
from tests.integration.parity.source_inputs import family_resolver, parse_time, point_value


_APPROVED_FAMILY_DECISION_STATUSES = {"accepted", "reduced"}


@dataclass(slots=True)
class ReplayOmsResult:
    submitted_orders: list[Any]
    events: list[Any]
    trade_ledger: list[dict[str, Any]]
    state: dict[str, Any]
    orders: list[dict[str, Any]]
    positions: list[dict[str, Any]]


async def run_replay_oms_sink(
    fixture: Mapping[str, Any],
    timeline: Sequence[Mapping[str, Any]],
    *,
    strategy_state: Mapping[str, Any],
    family_mode: bool,
) -> ReplayOmsResult:
    InstrumentRegistry.clear()
    instruments = _build_instruments(fixture)
    adapter = FakeIBKRExecutionAdapter(auto_ack=False)
    event_clock = lambda: parse_time(fixture["clock_start"])

    with TemporaryDirectory(prefix="parity-replay-oms-") as instrumentation_dir:
        repo = InMemoryRepository()
        await _hydrate_repository(fixture, repo, instruments)
        if family_mode:
            oms = await _build_family_oms(
                fixture,
                adapter,
                repo,
                instrumentation_dir,
                event_clock=event_clock,
            )
        else:
            oms = await _build_single_oms(
                fixture,
                adapter,
                repo,
                instrumentation_dir,
                event_clock=event_clock,
            )
        repo = oms._handler._repo
        event_queue = oms.stream_all_events()

        try:
            await oms.start()
            for marker in timeline:
                marker_type = str(marker.get("type", ""))
                if marker_type == "action":
                    await _submit_action(
                        oms,
                        fixture,
                        instruments,
                        str(marker.get("strategy_id", "")),
                        marker.get("action"),
                        marker.get("decision", {}),
                    )
                    await _settle_callbacks()
                elif marker_type == "family_reject":
                    await _submit_family_rejection(
                        oms,
                        repo,
                        fixture,
                        instruments,
                        str(marker.get("strategy_id", "")),
                        marker.get("action"),
                        marker.get("decision", {}),
                    )
                    await _settle_callbacks()
                elif marker_type == "broker_event":
                    await _apply_broker_event(fixture, adapter, repo, marker.get("event", {}))
                    await _settle_callbacks()

            events = _drain_queue(event_queue)
            submitted_orders = [
                await repo.get_order(item["oms_order_id"])
                for item in adapter.submitted
            ]
            ledger = await _ledger_from_repo(repo, family_resolver(fixture))
            state = await _state_from_oms(repo, oms, strategy_state)
        finally:
            await oms.stop()
            InstrumentRegistry.clear()

    return ReplayOmsResult(
        submitted_orders=[order for order in submitted_orders if order is not None],
        events=events,
        trade_ledger=ledger,
        state=state,
        orders=state["orders"],
        positions=state["positions"],
    )


async def _build_single_oms(
    fixture: Mapping[str, Any],
    adapter: FakeIBKRExecutionAdapter,
    repository: InMemoryRepository,
    instrumentation_dir: str,
    *,
    event_clock,
) -> Any:
    strategy_cfg = fixture.get("strategy_config", {}) or {}
    account = fixture.get("account_state", {}) or {}
    strategy_id = str(strategy_cfg.get("strategy_id") or fixture.get("surface"))
    return await build_oms_service(
        adapter=adapter,
        strategy_id=strategy_id,
        unit_risk_dollars=float(strategy_cfg.get("unit_risk_dollars", 1_000.0)),
        daily_stop_R=float(strategy_cfg.get("daily_stop_R", 10.0)),
        heat_cap_R=float(strategy_cfg.get("heat_cap_R", 20.0)),
        portfolio_daily_stop_R=float(strategy_cfg.get("portfolio_daily_stop_R", 20.0)),
        portfolio_weekly_stop_R=float(strategy_cfg.get("portfolio_weekly_stop_R", 50.0)),
        db_pool=None,
        repository=repository,
        family_id=str(strategy_cfg.get("family", fixture.get("family", "unknown"))),
        get_current_equity=lambda: float(account.get("equity", 100_000.0)),
        recon_interval_s=3600.0,
        instrumentation_data_dir=instrumentation_dir,
        event_clock=event_clock,
    )


async def _build_family_oms(
    fixture: Mapping[str, Any],
    adapter: FakeIBKRExecutionAdapter,
    repository: InMemoryRepository,
    instrumentation_dir: str,
    *,
    event_clock,
) -> Any:
    family_cfg = fixture.get("family_config", {}) or {}
    account = fixture.get("account_state", {}) or {}
    portfolio_rules = portfolio_rules_config_from_fixture(fixture)
    oms, _coordinator = await build_multi_strategy_oms(
        adapter=adapter,
        strategies=[
            {
                "id": str(item["id"]),
                "unit_risk_dollars": float(item.get("unit_risk_dollars", 1_000.0)),
                "daily_stop_R": float(item.get("daily_stop_R", 10.0)),
                "priority": int(item.get("priority", 99)),
                "max_heat_R": float(item.get("max_heat_R", family_cfg.get("heat_cap_R", 20.0))),
                "max_working_orders": int(item.get("max_working_orders", 4)),
            }
            for item in family_cfg.get("strategies", [])
            if item.get("id")
        ],
        heat_cap_R=float(family_cfg.get("heat_cap_R", 20.0)),
        portfolio_daily_stop_R=float(family_cfg.get("portfolio_daily_stop_R", 20.0)),
        portfolio_weekly_stop_R=float(family_cfg.get("portfolio_weekly_stop_R", 50.0)),
        portfolio_rules_config=portfolio_rules,
        portfolio_unit_risk_dollars=float(
            getattr(portfolio_rules, "reference_unit_risk_dollars", 0.0) or 0.0
        ),
        db_pool=None,
        repository=repository,
        family_id=str(family_cfg.get("family", fixture.get("family", "unknown"))),
        get_current_equity=lambda: float(account.get("equity", 100_000.0)),
        recon_interval_s=3600.0,
        instrumentation_data_dir=instrumentation_dir,
        event_clock=event_clock,
    )
    return oms


async def _submit_family_rejection(
    oms: Any,
    repo: InMemoryRepository,
    fixture: Mapping[str, Any],
    instruments: Mapping[str, Instrument],
    strategy_id: str,
    action: Any,
    decision: Mapping[str, Any],
) -> None:
    order = _order_from_action(fixture, instruments, strategy_id, action)
    if order is None:
        return
    reason = str(decision.get("reason") or "family_replay_rejected")
    order.status = OrderStatus.REJECTED
    order.remaining_qty = order.qty
    order.reject_reason = ""
    await repo.save_order_and_event(order, "RISK_DENIED", {"reason": reason})
    oms.event_bus.emit_risk_denial(order.strategy_id, order.oms_order_id, reason)


async def _submit_action(
    oms: Any,
    fixture: Mapping[str, Any],
    instruments: Mapping[str, Instrument],
    strategy_id: str,
    action: Any,
    decision: Mapping[str, Any] | None = None,
) -> None:
    if action is None:
        return
    if isinstance(action, (CancelAction, ReplaceProtectiveStop, FlattenPosition)):
        await _submit_control_action(oms, strategy_id, action)
        return
    decision_payload = decision or {}
    status = str(decision_payload.get("status", "")).lower()
    if status:
        decision_payload = validate_family_decision_payload(decision_payload)
        status = str(decision_payload.get("status", "")).lower()
    order = _order_from_action(fixture, instruments, strategy_id, action, decision_payload)
    if order is None:
        return
    if status == "rejected":
        raise AssertionError(f"rejected family replay decision reached order submission: {decision_payload}")
    if status in _APPROVED_FAMILY_DECISION_STATUSES:
        await oms.submit_preapproved_family_intent(
            strategy_id=strategy_id,
            order=order,
            decision=_preapproved_family_decision(strategy_id, order, decision_payload),
        )
        return
    await oms.submit_intent(
        Intent(intent_type=IntentType.NEW_ORDER, strategy_id=strategy_id, order=order)
    )


async def _submit_control_action(oms: Any, strategy_id: str, action: Any) -> None:
    if isinstance(action, CancelAction) and action.target_order_id:
        await oms.submit_intent(
            Intent(
                intent_type=IntentType.CANCEL_ORDER,
                strategy_id=strategy_id,
                target_oms_order_id=action.target_order_id,
            )
        )
    elif isinstance(action, ReplaceProtectiveStop) and action.target_order_id:
        await oms.submit_intent(
            Intent(
                intent_type=IntentType.REPLACE_ORDER,
                strategy_id=strategy_id,
                target_oms_order_id=action.target_order_id,
                new_qty=int(action.qty) if getattr(action, "qty", 0) else None,
                new_stop_price=float(action.stop_price),
            )
        )
    elif isinstance(action, FlattenPosition):
        await oms.submit_intent(
            Intent(
                intent_type=IntentType.FLATTEN,
                strategy_id=strategy_id,
                instrument_symbol=action.symbol,
            )
        )


def _order_from_action(
    fixture: Mapping[str, Any],
    instruments: Mapping[str, Instrument],
    strategy_id: str,
    action: Any,
    decision: Mapping[str, Any] | None = None,
) -> OMSOrder | None:
    symbol = str(getattr(action, "symbol", ""))
    inst = instruments.get(symbol)
    if inst is None:
        return None
    account = fixture.get("account_state", {}) or {}
    role = _role_for_action(action)
    order_type = _order_type_for_action(action)
    decision_qty = _decision_approved_qty(action, decision or {})
    order = OMSOrder(
        client_order_id=str(getattr(action, "client_order_id", "")),
        strategy_id=strategy_id,
        account_id=str(account.get("account_id", "ACCT-PARITY")),
        instrument=inst,
        side=OrderSide(str(getattr(action, "side", "BUY")).upper()),
        qty=decision_qty if decision_qty is not None else int(getattr(action, "qty", 0) or 0),
        order_type=order_type,
        limit_price=_limit_price_for_action(action),
        stop_price=_stop_price_for_action(action),
        tif=str(getattr(action, "tif", "DAY") or "DAY"),
        role=role,
        oca_group=str(getattr(action, "oca_group", "") or ""),
    )
    if role is OrderRole.ENTRY:
        order.risk_context = _risk_context_for_action(
            fixture,
            inst,
            strategy_id,
            action,
            qty_override=decision_qty,
        )
    return order


def _decision_approved_qty(action: Any, decision: Mapping[str, Any]) -> int | None:
    status = str(decision.get("status", "")).lower()
    if status not in _APPROVED_FAMILY_DECISION_STATUSES:
        if status:
            validate_family_decision_payload(decision)
        return None
    decision = validate_family_decision_payload(decision)
    approved = int(decision["approved_qty"])
    if approved <= 0:
        return None
    return approved


def _preapproved_family_decision(
    strategy_id: str,
    order: OMSOrder,
    decision: Mapping[str, Any],
) -> PreapprovedFamilyDecision:
    order_match = decision.get("order_match", {}) or {}
    symbol = order.instrument.symbol if order.instrument is not None else str(order_match.get("symbol", ""))
    return PreapprovedFamilyDecision(
        candidate_key=str(decision.get("candidate_key", "")),
        family_surface=str(decision.get("family_surface", "parity_family_replay")),
        strategy_id=str(decision.get("strategy_id", strategy_id)),
        symbol=str(decision.get("symbol", symbol)),
        side=str(decision.get("side", order.side.value)).upper(),
        role=str(decision.get("role", order.role.value)).upper(),
        sequence=int(decision.get("sequence", order_match.get("sequence", 1)) or 1),
        original_qty=int(decision.get("original_qty", order.qty)),
        approved_qty=int(decision.get("approved_qty", order.qty)),
        status=str(decision.get("status", "")),
        reason=str(decision.get("reason", "")),
    )


def _role_for_action(action: Any) -> OrderRole:
    if isinstance(action, (SubmitEntry, SubmitAddOnEntry)):
        return OrderRole.ENTRY
    if isinstance(action, (SubmitExit, SubmitPartialExit, SubmitMarketExit)):
        return OrderRole.EXIT
    if isinstance(action, SubmitProtectiveStop):
        return OrderRole.STOP
    if isinstance(action, SubmitProfitTarget):
        return OrderRole.TP
    return OrderRole.ENTRY


def _order_type_for_action(action: Any) -> OrderType:
    if isinstance(action, SubmitProtectiveStop):
        return OrderType.STOP
    if isinstance(action, SubmitProfitTarget):
        return OrderType.LIMIT
    if isinstance(action, SubmitMarketExit):
        return OrderType.MARKET
    return OrderType(str(getattr(action, "order_type", "MARKET")).upper())


def _limit_price_for_action(action: Any) -> float | None:
    value = getattr(action, "limit_price", None)
    if value is None and _order_type_for_action(action) is OrderType.LIMIT:
        value = getattr(action, "price", None)
    return None if value is None else float(value)


def _stop_price_for_action(action: Any) -> float | None:
    value = getattr(action, "stop_price", None)
    return None if value is None else float(value)


def _risk_context_for_action(
    fixture: Mapping[str, Any],
    inst: Instrument,
    strategy_id: str,
    action: Any,
    *,
    qty_override: int | None = None,
) -> RiskContext:
    raw = getattr(action, "risk_context", {}) or {}
    planned = float(
        raw.get("planned_entry_price")
        or getattr(action, "limit_price", None)
        or getattr(action, "stop_price", None)
        or getattr(action, "price", None)
        or 0.0
    )
    stop = float(raw.get("stop_for_risk") or getattr(action, "stop_price", None) or planned)
    original_qty = int(getattr(action, "qty", 0) or 0)
    qty = qty_override if qty_override is not None else original_qty
    raw_risk = raw.get("risk_dollars")
    if raw_risk is not None and original_qty > 0 and qty != original_qty:
        risk_dollars = float(raw_risk) * (float(qty) / float(original_qty))
    else:
        risk_dollars = float(raw_risk or (qty * abs(planned - stop) * point_value(fixture, inst.symbol)))
    portfolio_size_mult = float(raw.get("portfolio_size_mult", 1.0) or 1.0)
    if original_qty > 0 and qty != original_qty:
        portfolio_size_mult *= float(qty) / float(original_qty)
    return RiskContext(
        stop_for_risk=stop,
        planned_entry_price=planned,
        risk_budget_tag=str(raw.get("risk_budget_tag", strategy_id)),
        risk_dollars=risk_dollars,
        portfolio_size_mult=portfolio_size_mult,
        unit_risk_dollars=float(raw.get("unit_risk_dollars", 0.0) or 0.0),
    )


async def _apply_broker_event(
    fixture: Mapping[str, Any],
    adapter: FakeIBKRExecutionAdapter,
    repo: Any,
    event_spec: Mapping[str, Any],
) -> None:
    submitted = await _submitted_for_match(adapter, repo, event_spec.get("order_match", {}))
    if submitted is None:
        raise AssertionError(f"broker event could not match replay OMS order: {event_spec.get('order_match', {})}")
    broker_order_id = submitted["ref"].broker_order_id
    event_type = str(event_spec.get("event", "fill")).lower()
    if event_type == "fill":
        exec_id = str(event_spec.get("exec_id", f"EXEC-{submitted['oms_order_id']}"))
        adapter.emit_fill(
            broker_order_id,
            exec_id=exec_id,
            price=float(event_spec.get("price", submitted.get("limit_price") or 0.0)),
            qty=float(event_spec.get("qty", submitted.get("qty", 0))),
            commission=float(event_spec.get("commission", 0.0)),
            fill_time=parse_time(event_spec.get("timestamp")),
        )
        await _await(lambda: repo.fill_exists(exec_id))
    elif event_type == "status":
        adapter.emit_status(
            broker_order_id,
            str(event_spec.get("status", "Submitted")),
            remaining=float(event_spec.get("remaining", 0.0)),
        )
    elif event_type == "reject":
        adapter.emit_reject(
            broker_order_id,
            str(event_spec.get("reason", "rejected")),
            int(event_spec.get("error_code", 0)),
            bool(event_spec.get("retryable", False)),
        )


async def _submitted_for_match(
    adapter: FakeIBKRExecutionAdapter,
    repo: Any,
    match: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    sequence = int(match.get("sequence", 1))
    matches = []
    for item in adapter.submitted:
        order = await repo.get_order(item["oms_order_id"])
        if order is not None and _order_matches(order, match):
            matches.append(item)
    if len(matches) < sequence:
        return None
    if len(matches) > sequence and "sequence" not in match:
        raise AssertionError(f"broker event matched multiple replay OMS orders: {match}")
    return matches[sequence - 1]




async def _hydrate_repository(
    fixture: Mapping[str, Any],
    repo: Any,
    instruments: Mapping[str, Instrument],
) -> None:
    await hydrate_repository_from_fixture(fixture, repo, instruments)


async def _state_from_oms(
    repo: InMemoryRepository,
    oms: Any,
    strategy_state: Mapping[str, Any],
) -> dict[str, Any]:
    for sid in getattr(oms, "_strategy_risk_states", {}):
        await oms.get_strategy_risk(str(sid))
    if getattr(oms, "_portfolio_risk_state", None) is not None:
        await oms.get_portfolio_risk()
    orders = [
        {
            "oms_order_id": order.oms_order_id,
            "strategy_id": order.strategy_id,
            "symbol": order.instrument.symbol if order.instrument else "",
            "side": order.side.value,
            "qty": order.qty,
            "order_type": order.order_type.value,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "role": order.role.value,
            "status": order.status.value,
            "filled_qty": order.filled_qty,
            "remaining_qty": order.remaining_qty,
            "avg_fill_price": order.avg_fill_price,
            "client_tag": order.client_order_id,
            "reject_reason": order.reject_reason,
        }
        for order in repo._orders.values()
    ]
    positions = [
        {
            "strategy_id": pos.strategy_id,
            "symbol": pos.instrument_symbol,
            "net_qty": pos.net_qty,
            "avg_price": pos.avg_price,
            "realized_pnl": pos.realized_pnl,
            "open_risk_dollars": pos.open_risk_dollars,
            "open_risk_R": pos.open_risk_R,
        }
        for pos in repo._positions.values()
    ]
    return {
        "orders": orders,
        "positions": positions,
        "strategy_risk": {
            sid: _plain_dataclass(state)
            for sid, state in getattr(oms, "_strategy_risk_states", {}).items()
        },
        "portfolio_risk": [
            _plain_dataclass(getattr(oms, "_portfolio_risk_state"))
        ] if getattr(oms, "_portfolio_risk_state", None) is not None else [],
        "portfolio_rules": _portfolio_rules_state(oms),
        "blocked_reasons": _blocked_reasons_from_events(repo, orders),
        "strategy_state": dict(strategy_state),
    }








def _build_instruments(fixture: Mapping[str, Any]) -> dict[str, Instrument]:
    return build_instruments_from_fixture(fixture)






async def _settle_callbacks() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _await(predicate) -> None:
    for _ in range(100):
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        await asyncio.sleep(0.01)
    result = predicate()
    if asyncio.iscoroutine(result):
        result = await result
    assert result
