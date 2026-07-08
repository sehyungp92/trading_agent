from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from tests.integration.parity.live_shadow_contract import normalize_fingerprint_payload
from tests.integration.parity.source_inputs import strategy_ids


def build_family_state(
    fixture: Mapping[str, Any],
    *,
    coordinator_class: str,
    orders: list[Mapping[str, Any]],
    positions: list[Mapping[str, Any]],
    strategy_state: Mapping[str, Any],
    strategy_risk: Mapping[str, Any] | None = None,
    portfolio_risk: list[Mapping[str, Any]] | None = None,
    portfolio_rules: list[Mapping[str, Any]] | None = None,
    overlay_state: Mapping[str, Any] | None = None,
    surface_adapter: str = "",
    blocked_counts: Mapping[str, int] | None = None,
    blocked_reasons: Mapping[str, list[str]] | None = None,
    accepted_quantities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one shared Layer-3 state contract.

    Child contracts compare child-specific order, position, and strategy slices.
    This shared snapshot compares family-level repository exposure, configured
    child coverage, overlay state, and portfolio-surface outcomes once.
    """

    configured_ids = strategy_ids(fixture)
    order_counts: Counter[str] = Counter()
    submitted_roles: dict[str, list[str]] = defaultdict(list)
    filled_entries: Counter[str] = Counter()
    rejected_entries: Counter[str] = Counter()
    for order in orders:
        sid = str(order.get("strategy_id", ""))
        role = str(order.get("role") or order.get("order_role") or "").upper()
        status = str(order.get("status", "")).upper()
        order_counts[f"{sid}:{role}:{status}"] += 1
        if role:
            submitted_roles[sid].append(role)
        if role == "ENTRY" and status == "FILLED":
            filled_entries[sid] += 1
        if role == "ENTRY" and status == "REJECTED":
            rejected_entries[sid] += 1
    accepted_qty = dict(accepted_quantities or _accepted_quantities_from_orders(orders, fixture=fixture))

    position_rows = []
    symbol_exposure: Counter[str] = Counter()
    strategy_exposure: Counter[str] = Counter()
    symbol_risk_dollars: Counter[str] = Counter()
    strategy_risk_dollars: Counter[str] = Counter()
    symbol_risk_R: Counter[str] = Counter()
    strategy_risk_R: Counter[str] = Counter()
    directional_risk_dollars: Counter[str] = Counter()
    directional_risk_R: Counter[str] = Counter()
    for position in positions:
        sid = str(position.get("strategy_id", ""))
        symbol = str(position.get("symbol") or position.get("instrument_symbol") or "")
        net_qty = float(position.get("net_qty", position.get("qty", 0.0)) or 0.0)
        if not sid or not symbol or net_qty == 0:
            continue
        risk_dollars = float(position.get("open_risk_dollars", 0.0) or 0.0)
        risk_R = float(position.get("open_risk_R", 0.0) or 0.0)
        direction = "LONG" if net_qty > 0 else "SHORT"
        signed_qty = int(net_qty) if net_qty.is_integer() else net_qty
        position_rows.append(
            {
                "strategy_id": sid,
                "symbol": symbol,
                "net_qty": signed_qty,
                "open_risk_dollars": _number(risk_dollars),
                "open_risk_R": _number(risk_R),
            }
        )
        symbol_exposure[symbol] += net_qty
        strategy_exposure[sid] += net_qty
        symbol_risk_dollars[symbol] += risk_dollars
        strategy_risk_dollars[sid] += risk_dollars
        symbol_risk_R[symbol] += risk_R
        strategy_risk_R[sid] += risk_R
        directional_risk_dollars[direction] += risk_dollars
        directional_risk_R[direction] += risk_R

    return {
        "coordinator_class": coordinator_class,
        "configured_strategy_ids": configured_ids,
        "repository": {
            "order_counts": dict(sorted(order_counts.items())),
            "filled_entry_counts": dict(sorted(filled_entries.items())),
            "positions": sorted(position_rows, key=lambda row: (row["strategy_id"], row["symbol"])),
            "submitted_roles": {
                sid: sorted(roles)
                for sid, roles in sorted(submitted_roles.items())
            },
        },
        "risk_exposure": {
            "net_qty_by_symbol": {
                symbol: _number(qty)
                for symbol, qty in sorted(symbol_exposure.items())
            },
            "net_qty_by_strategy": {
                sid: _number(qty)
                for sid, qty in sorted(strategy_exposure.items())
            },
            "open_risk_dollars_by_symbol": {
                symbol: _number(value)
                for symbol, value in sorted(symbol_risk_dollars.items())
            },
            "open_risk_dollars_by_strategy": {
                sid: _number(value)
                for sid, value in sorted(strategy_risk_dollars.items())
            },
            "open_risk_R_by_symbol": {
                symbol: _number(value)
                for symbol, value in sorted(symbol_risk_R.items())
            },
            "open_risk_R_by_strategy": {
                sid: _number(value)
                for sid, value in sorted(strategy_risk_R.items())
            },
            "open_risk_dollars_by_direction": {
                direction: _number(value)
                for direction, value in sorted(directional_risk_dollars.items())
            },
            "open_risk_R_by_direction": {
                direction: _number(value)
                for direction, value in sorted(directional_risk_R.items())
            },
        },
        "risk_state": {
            "strategy": _stable_mapping(strategy_risk or {}),
            "portfolio": _stable_sequence(_portfolio_risk_contract(portfolio_risk or [])),
            "portfolio_rules": _stable_sequence(portfolio_rules or []),
        },
        "strategy_state_coverage": {
            sid: sid in strategy_state
            for sid in configured_ids
        },
        "portfolio_surface": {
            "adapter": surface_adapter,
            "accepted_entry_counts": dict(sorted(filled_entries.items())),
            "blocked_entry_counts": dict(sorted((blocked_counts or rejected_entries).items())),
            "blocked_reasons": _stable_mapping(blocked_reasons or {}),
            "accepted_quantities": _stable_mapping(accepted_qty),
        },
        "overlay": dict(overlay_state or {}),
    }


def _number(value: float) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise TypeError(f"unsupported non-finite family state numeric value: {value!r}")
    return int(number) if number.is_integer() else round(number, 6)


def _accepted_quantities_from_orders(
    orders: list[Mapping[str, Any]],
    *,
    fixture: Mapping[str, Any],
) -> dict[str, int | float]:
    totals: Counter[str] = Counter()
    initial_ids = _initial_repository_order_ids(fixture)
    for order in orders:
        if str(order.get("role") or order.get("order_role") or "").upper() != "ENTRY":
            continue
        if str(order.get("status", "")).upper() == "REJECTED":
            continue
        if _order_identity(order) in initial_ids:
            continue
        sid = str(order.get("strategy_id", ""))
        if sid:
            totals[sid] += float(order.get("qty", 0.0) or 0.0)
    return {sid: _number(qty) for sid, qty in sorted(totals.items())}


def _initial_repository_order_ids(fixture: Mapping[str, Any]) -> set[tuple[str, str]]:
    initial = (fixture.get("initial_repository_state", {}) or {}).get("orders", []) or []
    ids: set[tuple[str, str]] = set()
    for row in initial:
        if not isinstance(row, Mapping):
            continue
        identity = _order_identity(row)
        if identity != ("", ""):
            ids.add(identity)
    return ids


def _order_identity(order: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(order.get("oms_order_id") or ""),
        str(order.get("client_tag") or order.get("client_order_id") or ""),
    )


def _portfolio_risk_contract(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    contract_rows: list[Mapping[str, Any]] = []
    for row in rows:
        item = dict(row)
        contract_rows.append(item)
    # Multiple coordinator-owned OMS services may expose identical portfolio
    # rows over the same shared repository. Deduplicate exact duplicates only;
    # distinct open-risk dollars/R values must remain visible so missed
    # hydration in one service fails the family-state comparison.
    return contract_rows


def _stable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _stable_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}


def _stable_sequence(value: list[Mapping[str, Any]]) -> list[Any]:
    deduped: dict[str, Any] = {}
    for item in value:
        stable = _stable_value(item)
        key = _stable_json_key(stable)
        deduped[key] = stable
    return [deduped[key] for key in sorted(deduped)]


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _stable_mapping(value)
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_stable_value(item) for item in value), key=_stable_json_key)
    if isinstance(value, float):
        return _number(value)
    return value


def _stable_json_key(value: Any) -> str:
    return json.dumps(
        normalize_fingerprint_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
