from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any


FAMILY_DECISION_STATUSES = {"accepted", "reduced", "rejected"}


def validate_family_decision_payload(
    decision: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
    family_surface: str = "",
) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise AssertionError(f"family replay decision must be a mapping: {decision!r}")
    payload = dict(decision)
    if family_surface and not payload.get("family_surface"):
        payload["family_surface"] = family_surface

    missing = []
    for field in ("strategy_id", "symbol", "side", "role", "status", "family_surface"):
        if payload.get(field) in (None, ""):
            missing.append(field)
    if "reason" not in payload:
        missing.append("reason")
    if payload.get("candidate_key") in (None, "") and payload.get("sequence") in (None, ""):
        missing.append("candidate_key_or_sequence")
    if missing:
        raise AssertionError(
            f"family replay decision missing required field(s) {missing}: {decision}"
        )

    status = str(payload.get("status", "")).lower()
    if status not in FAMILY_DECISION_STATUSES:
        raise AssertionError(f"unsupported family replay decision status: {decision}")
    original_qty = _positive_int(payload.get("original_qty"), "original_qty", decision)
    approved_qty = _non_negative_int(payload.get("approved_qty"), "approved_qty", decision)
    sequence = _positive_int(payload.get("sequence", 1), "sequence", decision)

    if status == "accepted" and approved_qty != original_qty:
        raise AssertionError(
            f"accepted family replay decision must preserve quantity: {decision}"
        )
    if status == "reduced" and not (0 < approved_qty < original_qty):
        raise AssertionError(
            f"reduced family replay decision must satisfy 0 < approved_qty < original_qty: {decision}"
        )
    if status == "rejected" and approved_qty != 0:
        raise AssertionError(
            f"rejected family replay decision must have approved_qty == 0: {decision}"
        )

    payload["status"] = status
    payload["original_qty"] = original_qty
    payload["approved_qty"] = approved_qty
    payload["sequence"] = sequence
    payload["side"] = str(payload.get("side", "")).upper()
    payload["role"] = str(payload.get("role", "")).upper()

    if candidate is not None:
        _validate_candidate_match(payload, candidate, decision)
    return payload


def _validate_candidate_match(
    payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
    original_decision: Mapping[str, Any],
) -> None:
    order = candidate.get("order", {}) or {}
    expected = {
        "strategy_id": str(order.get("strategy_id", "")),
        "symbol": str(order.get("symbol", "")),
        "side": str(order.get("side", "")).upper(),
        "role": str(order.get("role", "ENTRY")).upper(),
        "candidate_key": str(candidate.get("candidate_key", "")),
    }
    for field, expected_value in expected.items():
        actual = str(payload.get(field, ""))
        if field in {"side", "role"}:
            actual = actual.upper()
        if expected_value and actual and actual != expected_value:
            raise AssertionError(
                f"family replay decision {field} does not match generated candidate: "
                f"decision={original_decision}, candidate={expected}"
            )


def _positive_int(value: Any, field_name: str, decision: Mapping[str, Any]) -> int:
    number = _non_negative_int(value, field_name, decision)
    if number <= 0:
        raise AssertionError(
            f"family replay decision {field_name} must be > 0: {decision}"
        )
    return number


def _non_negative_int(value: Any, field_name: str, decision: Mapping[str, Any]) -> int:
    if isinstance(value, bool):
        raise AssertionError(
            f"family replay decision {field_name} must be an integer: {decision}"
        )
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise AssertionError(
            f"family replay decision {field_name} must be numeric: {decision}"
        ) from exc
    if decimal != decimal.to_integral_value():
        raise AssertionError(
            f"family replay decision {field_name} must be an integer: {decision}"
        )
    number = int(decimal)
    if number < 0:
        raise AssertionError(
            f"family replay decision {field_name} must be >= 0: {decision}"
        )
    return number
