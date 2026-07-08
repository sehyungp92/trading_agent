from __future__ import annotations

from datetime import datetime, timezone

import json

from libs.instrumentation.event_contract import write_decision_event
from libs.instrumentation.lineage import LineageContext
from backtests.shared.parity.decision_capture import normalize_decision_event
from strategies.core.events import DecisionEvent


def test_decision_event_defaults_preserve_old_constructor_shape() -> None:
    event = DecisionEvent(
        code="NO_SIGNAL",
        ts=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        symbol="QQQ",
        timeframe="1h",
    )

    normalized = normalize_decision_event(event)

    assert normalized["schema_version"] == "decision_event_v1"
    assert normalized["event_type"] == "decision_event"
    assert normalized["bot_id"] == ""
    assert normalized["strategy_id"] == ""
    assert normalized["state_ref"] == ""
    assert normalized["emitted_actions"] == []


def test_decision_normalizer_preserves_lineage_state_and_actions() -> None:
    event = DecisionEvent(
        code="ENTRY_INTENT_CREATED",
        ts=datetime(2026, 5, 31, 12, 5, tzinfo=timezone.utc),
        symbol="MNQ",
        timeframe="5m",
        details={"z": 2, "a": {"b": 1}},
        bot_id="momentum_nq_01",
        strategy_id="NQDTC_v2.1",
        family_id="momentum",
        portfolio_id="paper_default",
        strategy_version="NQDTC.2.1",
        config_version="cfg_1",
        portfolio_config_version="pcfg_1",
        risk_config_version="risk_1",
        allocation_version="alloc_1",
        strategy_registry_version="registry_1",
        deployment_id="dep_1",
        parameter_set_id="param_1",
        code_sha="abc123",
        trace_id="trace_1",
        bar_id="MNQ-20260531T120500Z",
        decision_kind="entry_intent",
        sequence=3,
        state_ref="state:after-filter",
        emitted_actions=("SubmitEntry",),
    )

    normalized = normalize_decision_event(event)

    assert normalized["strategy_id"] == "NQDTC_v2.1"
    assert normalized["bot_id"] == "momentum_nq_01"
    assert normalized["family_id"] == "momentum"
    assert normalized["portfolio_config_version"] == "pcfg_1"
    assert normalized["risk_config_version"] == "risk_1"
    assert normalized["allocation_version"] == "alloc_1"
    assert normalized["deployment_id"] == "dep_1"
    assert normalized["parameter_set_id"] == "param_1"
    assert normalized["trace_id"] == "trace_1"
    assert normalized["bar_id"] == "MNQ-20260531T120500Z"
    assert normalized["decision_kind"] == "entry_intent"
    assert normalized["sequence"] == 3
    assert normalized["state_ref"] == "state:after-filter"
    assert normalized["emitted_actions"] == ["SubmitEntry"]
    assert list(normalized["details"]) == ["a", "z"]


def test_write_decision_event_persists_bounded_jsonl_with_lineage(tmp_path) -> None:
    lineage = LineageContext(
        bot_id="momentum_nq_01",
        strategy_id="NQDTC_v2.1",
        family_id="momentum",
        portfolio_id="paper_default",
        strategy_version="NQDTC.2.1",
        config_version="cfg_1",
        deployment_id="dep_1",
        parameter_set_id="param_1",
        code_sha="abc123",
        trace_id="trace_1",
    )
    decision = DecisionEvent(
        code="ENTRY_INTENT_CREATED",
        ts=datetime(2026, 5, 31, 12, 5, tzinfo=timezone.utc),
        symbol="MNQ",
        timeframe="5m",
        details={"decision_ref": "decision_1"},
        emitted_actions=("SubmitEntry",),
    )

    path = write_decision_event(tmp_path, decision, lineage=lineage)
    [payload] = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert path.parent.name == "decisions"
    assert payload["event_type"] == "decision_event"
    assert payload["schema_version"] == "decision_event_v1"
    assert payload["strategy_id"] == "NQDTC_v2.1"
    assert payload["parameter_set_id"] == "param_1"
    assert payload["ts"] == "2026-05-31T12:05:00+00:00"
    assert payload["emitted_actions"] == ["SubmitEntry"]
