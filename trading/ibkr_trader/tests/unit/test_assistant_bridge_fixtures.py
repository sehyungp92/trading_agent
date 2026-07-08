from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from libs.instrumentation.event_contract import REQUIRED_SECTION_6_2_FIELDS
from strategies.stock.instrumentation.src.sidecar import _EVENT_PRIORITY


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "assistant_bridge"
    / "trading_assistant_bridge_golden.jsonl"
)

REQUIRED_EVENT_TYPES = {
    "trade",
    "missed_opportunity",
    "order",
    "inferred_fill",
    "portfolio_rule_check",
    "risk_decision",
    "daily_snapshot",
    "family_daily_snapshot",
    "config_snapshot",
    "deployment",
}

PAYLOAD_IDENTITY_FIELDS = (
    "event_id",
    "assistant_strategy_id",
    *REQUIRED_SECTION_6_2_FIELDS,
)
LINEAGE_FIELDS = ("assistant_strategy_id", *REQUIRED_SECTION_6_2_FIELDS)

JOIN_FIELDS = (
    "payload_key",
    "payload_hash",
    "event_ref",
    "decision_id",
    "decision_ref",
    "intent_id",
    "portfolio_rule_event_id",
    "risk_decision_id",
    "order_id",
    "client_order_id",
    "broker_order_id",
    "oms_order_id",
    "fill_id",
    "trade_id",
    "provisional_order_ref",
    "snapshot_id",
    "artifact_hash",
    "source_artifact_hash",
    "resource_plan_hash",
    "portfolio_policy_hash",
    "source",
    "source_stream",
)


def _load_fixture() -> list[dict]:
    rows: list[dict] = []
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row, dict), line_no
        rows.append(row)
    return rows


def _payload(envelope: Mapping) -> dict:
    raw = envelope["payload"]
    assert isinstance(raw, str), envelope["event_type"]
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_golden_assistant_bridge_fixture_covers_required_trading_events() -> None:
    rows = _load_fixture()
    event_ids = [row["event_id"] for row in rows]

    assert REQUIRED_EVENT_TYPES.issubset({row["event_type"] for row in rows})
    assert len(event_ids) == len(set(event_ids))


def test_golden_assistant_bridge_fixture_duplicates_identity_and_join_keys() -> None:
    for envelope in _load_fixture():
        payload = _payload(envelope)
        lineage = payload.get("lineage")

        assert isinstance(lineage, Mapping), envelope["event_type"]
        assert envelope["exchange_timestamp"]
        assert envelope["event_type"] in _EVENT_PRIORITY
        assert envelope["priority"] == _EVENT_PRIORITY[envelope["event_type"]]

        for field in PAYLOAD_IDENTITY_FIELDS:
            assert envelope.get(field), (envelope["event_type"], field, "envelope")
            assert payload.get(field) == envelope[field], (
                envelope["event_type"],
                field,
                "payload",
            )

        for field in LINEAGE_FIELDS:
            assert lineage.get(field) == payload[field], (envelope["event_type"], field, "lineage")

        for field in JOIN_FIELDS:
            if field in envelope:
                assert payload.get(field) == envelope[field], (
                    envelope["event_type"],
                    field,
                    "payload duplicate",
                )


def test_golden_assistant_bridge_fixture_has_required_scenarios() -> None:
    payloads = [_payload(row) for row in _load_fixture()]
    by_type: dict[str, list[dict]] = {}
    for payload in payloads:
        by_type.setdefault(payload["event_type"], []).append(payload)

    order = next(payload for payload in by_type["order"] if payload["status"] == "accepted")
    assert order["status"] == "accepted"
    assert order["order_id"] == "oms_order_entry_001"
    assert order["risk_decision_id"] == "risk_stock_approve_001"
    assert order["portfolio_rule_event_id"] == "prule_stock_pass_001"

    pass_rule = next(rule for rule in by_type["portfolio_rule_check"] if rule["result"] == "pass")
    assert pass_rule["event_id"] == "prule_stock_pass_001"
    assert pass_rule["approved"] is True

    blocked_rule = next(rule for rule in by_type["portfolio_rule_check"] if rule["result"] == "block")
    assert blocked_rule["event_id"] == "prule_stock_block_001"
    assert blocked_rule["approved"] is False
    assert blocked_rule["details"]["reason"] == "directional_cap: LONG risk too high"
    assert blocked_rule["details"]["blocked_symbol"] == "AAPL"

    risk_decision = by_type["risk_decision"][0]
    assert risk_decision["event_id"] == "risk_stock_approve_001"
    assert risk_decision["decision"] == "approve"

    inferred_fill = by_type["inferred_fill"][0]
    assert inferred_fill["fill_id"] == "fill_entry_001"
    assert inferred_fill["lifecycle_action"] == "inferred_fill"
    assert "status" not in inferred_fill

    missed = by_type["missed_opportunity"][0]
    assert missed["portfolio_rule_event_id"] == "prule_stock_block_001"

    family_snapshot = by_type["family_daily_snapshot"][0]
    assert family_snapshot["source"] == "daily_closeout"
    assert family_snapshot["snapshot_id"] == "family_snap_stock_20260605"
    assert family_snapshot["daily_snapshot"]["snapshot_id"] == "daily_stock_20260605"


def test_golden_assistant_bridge_fixture_keeps_daily_join_refs_closed() -> None:
    rows = _load_fixture()
    by_event_id = {row["event_id"]: row for row in rows}
    daily = next(_payload(row) for row in rows if row["event_type"] == "daily_snapshot")

    for event_id in daily["portfolio_rule_event_ids"]:
        assert by_event_id[event_id]["event_type"] == "portfolio_rule_check"

    for event_id in daily["risk_decision_ids"]:
        assert by_event_id[event_id]["event_type"] == "risk_decision"
