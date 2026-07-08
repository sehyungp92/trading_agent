from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.integration.parity.family_decisions import validate_family_decision_payload
from tests.integration.parity.normalizers import normalize_reason
from tests.integration.parity.replay_candidates import (
    ReplayDecisionTimeline,
    candidate_key as _candidate_key,
    entry_candidate_specs as _entry_candidate_specs,
    order_matches as _order_matches,
)
from tests.integration.parity.source_inputs import action_order_row


def _authoritative_family_timeline(
    fixture: Mapping[str, Any],
    out: ReplayDecisionTimeline,
    family_surface: Mapping[str, Any],
) -> list[dict[str, Any]]:
    decisions = list(family_surface.get("decisions", []) or [])
    candidates = _entry_candidate_specs(fixture, out)
    if not decisions:
        if candidates:
            missing = [str(candidate["candidate_key"]) for candidate in candidates]
            raise AssertionError(
                "family replay surface emitted no decisions for generated candidates: "
                f"{missing}"
            )
        timeline = _non_entry_action_markers(out.timeline)
        timeline.extend(
            {"type": "broker_event", "event": event}
            for event in fixture.get("broker_event_script", []) or []
        )
        return timeline
    candidate_by_key = {str(candidate["candidate_key"]): candidate for candidate in candidates}
    candidate_actions = {
        key: candidate.get("action")
        for key, candidate in candidate_by_key.items()
    }
    timeline = []
    seen: set[str] = set()
    for decision in decisions:
        key = str(decision.get("candidate_key", ""))
        if not key and decision.get("sequence") not in (None, ""):
            key = _candidate_key(decision)
        action = candidate_actions.get(key)
        if action is None:
            raise AssertionError(f"family replay decision has no generated candidate action: {decision}")
        if key in seen:
            raise AssertionError(f"family replay surface emitted duplicate decision for candidate: {key}")
        decision_payload = dict(decision)
        decision_payload.setdefault("candidate_key", key)
        decision_payload.setdefault("family_surface", str(family_surface.get("adapter", "")))
        decision_payload = validate_family_decision_payload(
            decision_payload,
            candidate=candidate_by_key.get(key),
            family_surface=str(family_surface.get("adapter", "")),
        )
        status = str(decision_payload.get("status", "")).lower()
        marker = {
            "type": "family_reject" if status == "rejected" else "action",
            "strategy_id": str(decision_payload.get("strategy_id", "")),
            "action": action,
            "decision": decision_payload,
        }
        timeline.append(marker)
        seen.add(key)
    missing = sorted(set(candidate_actions) - seen)
    if missing:
        raise AssertionError(f"family replay surface omitted generated candidates: {missing}")
    timeline.extend(
        {"type": "broker_event", "event": event}
        for event in fixture.get("broker_event_script", []) or []
    )
    timeline.extend(_non_entry_action_markers(out.timeline))
    return timeline


def _non_entry_action_markers(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers = []
    for marker in timeline:
        if str(marker.get("type", "")) != "action":
            continue
        action = marker.get("action")
        role = action_order_row(action, str(marker.get("strategy_id", ""))).get("role", "")
        if role != "ENTRY":
            markers.append(marker)
    return markers


def _assert_family_surface_matches_sink(family_surface: Mapping[str, Any], sink_orders: list[Mapping[str, Any]]) -> None:
    decisions = list(family_surface.get("decisions", []) or [])
    if not decisions:
        return
    for decision in decisions:
        order = _sink_order_for_decision(sink_orders, decision)
        if order is None:
            raise AssertionError(f"family replay decision did not produce an OMS order: {decision}")
        status = str(order.get("status", "")).upper()
        decision_status = str(decision.get("status", "")).lower()
        expected_qty = int(
            decision.get(
                "original_qty" if decision_status == "rejected" else "approved_qty",
                0,
            )
            or 0
        )
        actual_qty = int(float(order.get("qty", 0) or 0))
        if actual_qty != expected_qty:
            raise AssertionError(
                "family replay decision quantity does not match replay OMS sink: "
                f"decision={decision}, sink_order={order}"
            )
        if decision_status == "rejected":
            if status != "REJECTED":
                raise AssertionError(
                    "family replay rejected decision was not rejected by replay OMS sink: "
                    f"decision={decision}, sink_order={order}"
                )
            expected_reason = normalize_reason(str(decision.get("reason", "")))
            actual_reason = normalize_reason(str(order.get("reject_reason", "")))
            if expected_reason and actual_reason and actual_reason != expected_reason:
                raise AssertionError(
                    "family replay rejection reason does not match replay OMS sink: "
                    f"decision={expected_reason}, sink={actual_reason}"
                )
        elif status == "REJECTED":
            raise AssertionError(
                "family replay accepted decision was rejected by replay OMS sink: "
                f"decision={decision}, sink_order={order}"
            )


def _sink_order_for_decision(
    sink_orders: list[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    match = decision.get("order_match", {}) or {}
    sequence = int(match.get("sequence", decision.get("sequence", 1)) or 1)
    matches = [order for order in sink_orders if _order_matches(order, match)]
    if len(matches) < sequence:
        return None
    if len(matches) > sequence and "sequence" not in match:
        raise AssertionError(f"family replay decision matched multiple OMS sink orders: {decision}")
    return matches[sequence - 1]


def _apply_replay_oms_outcomes_to_strategy_state(state: dict[str, Any]) -> None:
    strategy_state = state.get("strategy_state", {})
    if not isinstance(strategy_state, dict):
        return
    for order in state.get("orders", []) or []:
        if str(order.get("role", "")).upper() != "ENTRY":
            continue
        if str(order.get("status", "")).upper() != "REJECTED":
            continue
        sid = str(order.get("strategy_id", ""))
        if sid == "IARIC_v1" and isinstance(strategy_state.get(sid), dict):
            strategy_state[sid]["last_decision_code"] = "ENTRY_DENIED"


authoritative_family_timeline = _authoritative_family_timeline
assert_family_surface_matches_sink = _assert_family_surface_matches_sink
apply_replay_oms_outcomes_to_strategy_state = _apply_replay_oms_outcomes_to_strategy_state
