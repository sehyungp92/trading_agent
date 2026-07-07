from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tests.integration.parity.family_decisions import validate_family_decision_payload
from tests.integration.parity.normalizers import normalize_reason
from tests.integration.parity.source_inputs import point_value


def _candidate_risk_pct_by_strategy(
    fixture: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    equity: float,
) -> dict[str, float]:
    risk_pct: dict[str, float] = {}
    for candidate in candidates:
        order = candidate["order"]
        risk = abs(float(candidate["entry_price"]) - float(candidate["stop_price"])) * point_value(fixture, str(order["symbol"])) * max(int(candidate["qty"]), 1)
        if risk > 0:
            risk_pct.setdefault(str(order["strategy_id"]), risk / max(equity, 1.0))
    return risk_pct


def _max_generated_qty(candidates: list[Mapping[str, Any]], strategy_id: str) -> int:
    qtys = [int(candidate["qty"]) for candidate in candidates if str(candidate["order"]["strategy_id"]) == strategy_id]
    return max(qtys) if qtys else 1


def _initial_positions(fixture: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(((fixture.get("initial_repository_state", {}) or {}).get("positions", []) or []))


def _initial_position_stop(fixture: Mapping[str, Any], pos: Mapping[str, Any], entry_price: float) -> float:
    symbol = str(pos.get("symbol") or pos.get("instrument_symbol") or "")
    qty = abs(float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0))
    risk = float(pos.get("open_risk_dollars", 0.0) or 0.0)
    point = point_value(fixture, symbol)
    stop_distance = risk / max(qty * point, 1e-9) if risk > 0 else max(entry_price * 0.01, 0.01)
    direction = 1 if float(pos.get("net_qty", pos.get("qty", 0.0)) or 0.0) > 0 else -1
    return entry_price - stop_distance if direction > 0 else entry_price + stop_distance


def _family_decision(
    candidate: Mapping[str, Any],
    *,
    approved_qty: int,
    status: str,
    reason: str,
) -> dict[str, Any]:
    order = candidate["order"]
    original_qty = int(float(order.get("qty", candidate.get("qty", 0)) or 0))
    approved_qty = max(int(approved_qty), 0)
    return {
        "candidate_key": str(candidate["candidate_key"]),
        "order_match": dict(candidate["order_match"]),
        "strategy_id": str(order["strategy_id"]),
        "symbol": str(order["symbol"]),
        "side": str(order["side"]).upper(),
        "role": str(order.get("role", "ENTRY")).upper(),
        "sequence": int(candidate["order_match"].get("sequence", 1)),
        "original_qty": original_qty,
        "approved_qty": approved_qty,
        "status": status,
        "reason": reason,
    }


def _accepted_status(candidate: Mapping[str, Any], approved_qty: int) -> str:
    original_qty = int(float(candidate["order"].get("qty", candidate.get("qty", 0)) or 0))
    return "reduced" if 0 < approved_qty < original_qty else "accepted"


def _portfolio_reason(reason: str) -> str:
    text = str(reason or "portfolio_rule")
    return text if text.lower().startswith("portfolio rule:") else f"Portfolio rule: {text}"


def _decision_summary(
    decisions: list[Mapping[str, Any]],
    *,
    family_surface: str,
) -> dict[str, Any]:
    normalized_decisions = [
        validate_family_decision_payload(decision, family_surface=family_surface)
        for decision in decisions
    ]
    return {
        "decisions": normalized_decisions,
        "candidate_counts": _decision_counts(normalized_decisions, statuses={"accepted", "reduced", "rejected"}),
        "accepted_counts": _decision_counts(normalized_decisions, statuses={"accepted", "reduced"}),
        "blocked_counts": _decision_counts(normalized_decisions, statuses={"rejected"}),
        "blocked_reasons": _decision_reasons(normalized_decisions),
        "accepted_quantities": _decision_quantities(normalized_decisions),
    }


def _decision_counts(decisions: list[Mapping[str, Any]], *, statuses: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        if str(decision.get("status", "")).lower() not in statuses:
            continue
        sid = str(decision.get("strategy_id", ""))
        if sid:
            counts[sid] = counts.get(sid, 0) + 1
    return dict(sorted(counts.items()))


def _decision_reasons(decisions: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for decision in decisions:
        if str(decision.get("status", "")).lower() != "rejected":
            continue
        sid = str(decision.get("strategy_id", ""))
        reason = normalize_reason(str(decision.get("reason", "")))
        if sid and reason:
            reasons.setdefault(sid, []).append(reason)
    return {sid: sorted(values) for sid, values in sorted(reasons.items())}


def _decision_quantities(decisions: list[Mapping[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, float] = {}
    for decision in decisions:
        if str(decision.get("status", "")).lower() not in {"accepted", "reduced"}:
            continue
        sid = str(decision.get("strategy_id", ""))
        if sid:
            totals[sid] = totals.get(sid, 0.0) + float(decision.get("approved_qty", 0.0) or 0.0)
    return {sid: int(qty) if qty.is_integer() else round(qty, 6) for sid, qty in sorted(totals.items())}


def _run_blocking(fn):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()
