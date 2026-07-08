from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from libs.oms.models.instrument import Instrument
from libs.oms.models.instrument_registry import InstrumentRegistry
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderStatus, OrderType, RiskContext
from libs.oms.models.position import Position
from tests.integration.parity.source_inputs import parse_time


class ParityOmsHydrationError(ValueError):
    pass


def build_instruments_from_fixture(fixture: Mapping[str, Any]) -> dict[str, Instrument]:
    instruments: dict[str, Instrument] = {}
    for index, item in enumerate(fixture.get("instruments", [])):
        prefix = f"instruments[{index}]"
        if not isinstance(item, Mapping):
            raise ParityOmsHydrationError(f"{prefix} must be an object")
        symbol_value = item.get("trade_symbol") or item.get("symbol")
        if not symbol_value:
            raise ParityOmsHydrationError(f"{prefix} must define symbol or trade_symbol")
        symbol = str(symbol_value)
        root = str(item.get("root") or item.get("symbol") or symbol)
        tick_size = _finite_float(item.get("tick_size", 0.01), f"{prefix}.tick_size")
        point_value = _finite_float(
            item.get("point_value", item.get("multiplier", 1.0)),
            f"{prefix}.point_value",
        )
        inst = Instrument(
            symbol=symbol,
            root=root,
            venue=str(item.get("venue", "SMART")),
            tick_size=tick_size,
            tick_value=_finite_float(item.get("tick_value", tick_size * point_value), f"{prefix}.tick_value"),
            multiplier=point_value,
            currency=str(item.get("currency", "USD")),
            point_value=point_value,
            contract_expiry=str(item.get("contract_expiry", "")),
            primary_exchange=str(item.get("primary_exchange", "")),
            sec_type=str(item.get("sec_type", "")),
            trading_class=str(item.get("trading_class", root)),
        )
        instruments[symbol] = inst
        instruments.setdefault(str(item.get("symbol", symbol)), inst)
        InstrumentRegistry.register(inst)
    return instruments


async def hydrate_repository_from_fixture(
    fixture: Mapping[str, Any],
    repo: Any,
    instruments: Mapping[str, Instrument],
) -> None:
    initial = fixture.get("initial_repository_state", {}) or {}
    if not initial:
        return
    if not isinstance(initial, Mapping):
        raise ParityOmsHydrationError("initial_repository_state must be an object")
    account_id = str((fixture.get("account_state", {}) or {}).get("account_id", "ACCT-PARITY"))
    for index, item in enumerate(initial.get("orders", []) or []):
        if not isinstance(item, Mapping):
            raise ParityOmsHydrationError(f"initial_repository_state.orders[{index}] must be an object")
        await repo.save_order(_order_from_initial_fixture(item, instruments, account_id, index=index))
    for index, item in enumerate(initial.get("positions", []) or []):
        if not isinstance(item, Mapping):
            raise ParityOmsHydrationError(f"initial_repository_state.positions[{index}] must be an object")
        await repo.save_position(_position_from_initial_fixture(item, instruments, account_id, index=index))


def _order_from_initial_fixture(
    item: Mapping[str, Any],
    instruments: Mapping[str, Instrument],
    account_id: str,
    *,
    index: int,
) -> OMSOrder:
    symbol = str(item.get("symbol") or item.get("instrument_symbol") or "")
    inst = instruments.get(symbol)
    if inst is None:
        raise ParityOmsHydrationError(
            f"initial_repository_state.orders[{index}] references unknown instrument: {symbol!r}"
        )
    order_id = str(item.get("oms_order_id") or item.get("client_order_id") or "")
    if not order_id:
        raise ParityOmsHydrationError(
            f"initial_repository_state.orders[{index}] must define oms_order_id or client_order_id"
        )
    try:
        order = OMSOrder(
            oms_order_id=order_id,
            client_order_id=str(item.get("client_order_id") or item.get("client_tag") or ""),
            strategy_id=str(item.get("strategy_id", "")),
            account_id=str(item.get("account_id") or account_id),
            instrument=inst,
            side=OrderSide(str(item.get("side", "BUY")).upper()),
            qty=_finite_int(item.get("qty", 0), f"initial_repository_state.orders[{index}].qty", positive=True),
            order_type=OrderType(str(item.get("order_type", "LIMIT")).upper()),
            limit_price=_float_or_none(item.get("limit_price"), f"initial_repository_state.orders[{index}].limit_price"),
            stop_price=_float_or_none(item.get("stop_price"), f"initial_repository_state.orders[{index}].stop_price"),
            tif=str(item.get("tif", "DAY")),
            role=OrderRole(str(item.get("role", item.get("order_role", "ENTRY"))).upper()),
            status=OrderStatus(str(item.get("status", "CREATED")).upper()),
            filled_qty=_finite_float(item.get("filled_qty", 0.0), f"initial_repository_state.orders[{index}].filled_qty"),
            remaining_qty=_finite_float(
                item.get("remaining_qty", item.get("qty", 0.0)),
                f"initial_repository_state.orders[{index}].remaining_qty",
            ),
            avg_fill_price=_finite_float(
                item.get("avg_fill_price", 0.0),
                f"initial_repository_state.orders[{index}].avg_fill_price",
            ),
        )
    except ValueError as exc:
        raise ParityOmsHydrationError(
            f"invalid initial_repository_state.orders[{index}] enum value: {exc}"
        ) from exc

    risk = item.get("risk_context")
    if risk not in (None, ""):
        if not isinstance(risk, Mapping):
            raise ParityOmsHydrationError(
                f"initial_repository_state.orders[{index}].risk_context must be an object"
            )
        order.risk_context = _risk_context_from_initial_fixture(risk, order, index=index)
    if (
        order.role is OrderRole.ENTRY
        and order.status in _working_statuses()
        and order.risk_context is None
    ):
        raise ParityOmsHydrationError(
            f"initial_repository_state.orders[{index}] is a working ENTRY without risk_context"
        )
    return order


def _risk_context_from_initial_fixture(
    risk: Mapping[str, Any],
    order: OMSOrder,
    *,
    index: int,
) -> RiskContext:
    prefix = f"initial_repository_state.orders[{index}].risk_context"
    planned = _finite_float(
        risk.get("planned_entry_price", order.limit_price or order.stop_price or 0.0),
        f"{prefix}.planned_entry_price",
    )
    stop = _finite_float(risk.get("stop_for_risk", order.stop_price or planned), f"{prefix}.stop_for_risk")
    return RiskContext(
        stop_for_risk=stop,
        planned_entry_price=planned,
        risk_budget_tag=str(risk.get("risk_budget_tag", order.strategy_id)),
        risk_dollars=_nonnegative_float(risk.get("risk_dollars", 0.0), f"{prefix}.risk_dollars"),
        portfolio_size_mult=_finite_float(
            risk.get("portfolio_size_mult", 1.0),
            f"{prefix}.portfolio_size_mult",
        ),
        unit_risk_dollars=_nonnegative_float(
            risk.get("unit_risk_dollars", 0.0),
            f"{prefix}.unit_risk_dollars",
        ),
    )


def _position_from_initial_fixture(
    item: Mapping[str, Any],
    instruments: Mapping[str, Instrument],
    account_id: str,
    *,
    index: int,
) -> Position:
    prefix = f"initial_repository_state.positions[{index}]"
    symbol = str(item.get("symbol") or item.get("instrument_symbol") or "")
    if symbol not in instruments:
        raise ParityOmsHydrationError(
            f"initial_repository_state.positions[{index}] references unknown instrument: {symbol!r}"
        )
    return Position(
        account_id=str(item.get("account_id") or account_id),
        instrument_symbol=symbol,
        strategy_id=str(item.get("strategy_id", "")),
        net_qty=_finite_float(item.get("net_qty", item.get("qty", 0.0)), f"{prefix}.net_qty"),
        avg_price=_finite_float(item.get("avg_price", item.get("entry_price", 0.0)), f"{prefix}.avg_price"),
        realized_pnl=_finite_float(item.get("realized_pnl", 0.0), f"{prefix}.realized_pnl"),
        unrealized_pnl=_finite_float(item.get("unrealized_pnl", 0.0), f"{prefix}.unrealized_pnl"),
        open_risk_dollars=_finite_float(item.get("open_risk_dollars", 0.0), f"{prefix}.open_risk_dollars"),
        open_risk_R=_finite_float(item.get("open_risk_R", 0.0), f"{prefix}.open_risk_R"),
        last_update_at=parse_time(item["last_update_at"]) if item.get("last_update_at") else None,
    )


def _float_or_none(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    return _finite_float(value, field_name)


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParityOmsHydrationError(f"{field_name} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ParityOmsHydrationError(f"{field_name} must be finite: {value!r}")
    return number


def _nonnegative_float(value: Any, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0:
        raise ParityOmsHydrationError(f"{field_name} must be non-negative: {value!r}")
    return number


def _finite_int(value: Any, field_name: str, *, positive: bool = False) -> int:
    number = _finite_float(value, field_name)
    if not number.is_integer():
        raise ParityOmsHydrationError(f"{field_name} must be an integer: {value!r}")
    integer = int(number)
    if positive and integer <= 0:
        raise ParityOmsHydrationError(f"{field_name} must be positive: {value!r}")
    return integer


def _working_statuses() -> set[OrderStatus]:
    return {
        OrderStatus.RISK_APPROVED,
        OrderStatus.ROUTED,
        OrderStatus.ACKED,
        OrderStatus.WORKING,
        OrderStatus.PARTIALLY_FILLED,
    }
