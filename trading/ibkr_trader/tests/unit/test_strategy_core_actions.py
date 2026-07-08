from __future__ import annotations

from datetime import datetime, timezone

from backtests.shared.parity.decision_capture import normalize_decision_stream
from backtests.shared.parity.execution_adapters import neutral_action_to_sim_order
from strategies.core.actions import (
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent

UTC = timezone.utc


def test_neutral_action_to_sim_order_maps_entry_and_stop_actions() -> None:
    entry = SubmitEntry(
        client_order_id="entry-1",
        symbol="MNQ",
        side="SELL",
        qty=2,
        order_type="STOP_LIMIT",
        price=18990.0,
        limit_price=18990.0,
        stop_price=18992.0,
        metadata={"role": "entry"},
    )
    stop = ReplaceProtectiveStop(
        symbol="MNQ",
        target_order_id="stop-1",
        side="BUY",
        stop_price=19010.0,
        qty=2,
        reason="trail",
    )

    entry_order = neutral_action_to_sim_order(entry, tick_size=0.25)
    stop_order = neutral_action_to_sim_order(stop, tick_size=0.25)

    assert entry_order.order_id == "entry-1"
    assert entry_order.stop_price == 18992.0
    assert entry_order.limit_price == 18990.0
    assert stop_order.order_id == "stop-1"
    assert stop_order.stop_price == 19010.0


def test_neutral_action_to_sim_order_maps_profit_target_partial_and_flatten_actions() -> None:
    target = SubmitProfitTarget(
        client_order_id="tp-1",
        symbol="NQ",
        side="SELL",
        qty=1,
        limit_price=20025.0,
        oca_group="OCA-1",
    )
    partial = SubmitPartialExit(
        client_order_id="partial-1",
        symbol="NQ",
        side="SELL",
        qty=1,
        order_type="LIMIT",
        limit_price=20010.0,
        role="partial_exit",
        oca_group="OCA-1",
    )
    flatten = FlattenPosition(
        symbol="NQ",
        reason="risk_off",
        side="SELL",
        qty=2,
        parent_order_id="flatten-1",
    )

    target_order = neutral_action_to_sim_order(target, tick_size=0.25)
    partial_order = neutral_action_to_sim_order(partial, tick_size=0.25)
    flatten_order = neutral_action_to_sim_order(flatten, tick_size=0.25)

    assert target_order.order_type.value == "LIMIT"
    assert target_order.limit_price == 20025.0
    assert target_order.oca_group == "OCA-1"
    assert partial_order.order_type.value == "LIMIT"
    assert partial_order.limit_price == 20010.0
    assert flatten_order.order_type.value == "MARKET"
    assert flatten_order.order_id == "flatten-1"


def test_neutral_action_to_sim_order_maps_add_on_entry_and_oca_metadata() -> None:
    add_on = SubmitAddOnEntry(
        client_order_id="add-1",
        symbol="NQ",
        side="BUY",
        qty=1,
        order_type="LIMIT",
        limit_price=20005.0,
        tif="DAY",
        parent_order_id="entry-1",
        oca_group="ADD-LEG-1",
        role="add_on_entry",
        metadata={"ttl_seconds": 30},
    )
    stop = SubmitProtectiveStop(
        client_order_id="stop-1",
        symbol="NQ",
        side="SELL",
        qty=1,
        stop_price=19975.0,
        parent_order_id="entry-1",
        oca_group="ADD-LEG-1",
        role="protective_stop",
    )

    add_on_order = neutral_action_to_sim_order(add_on, tick_size=0.25)
    stop_order = neutral_action_to_sim_order(stop, tick_size=0.25)

    assert add_on_order.order_type.value == "LIMIT"
    assert add_on_order.limit_price == 20005.0
    assert add_on_order.oca_group == "ADD-LEG-1"
    assert add_on_order.tag == "add_on_entry"
    assert stop_order.order_type.value == "STOP"
    assert stop_order.stop_price == 19975.0
    assert stop_order.oca_group == "ADD-LEG-1"
    assert stop_order.tag == "protective_stop"


def test_normalize_decision_stream_preserves_core_fields() -> None:
    events = [
        DecisionEvent(
            code="ENTRY_FILLED",
            ts=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
            symbol="MNQ",
            timeframe="5m",
            details={"qty": 1, "nested": {"b": 2, "a": 1}},
            strategy_id="NQDTC_v2.1",
            family_id="momentum",
            portfolio_id="paper_default",
            config_version="cfg_abc",
            risk_config_version="risk_abc",
            allocation_version="alloc_abc",
            strategy_registry_version="registry_abc",
            deployment_id="dep_2026_04_25_abc",
            parameter_set_id="param_abc",
            code_sha="abc123",
            trace_id="trace_abc",
            bar_id="MNQ-20260425T100000Z",
            decision_kind="fill",
            sequence=7,
            state_ref="state:1",
            emitted_actions=("SubmitProtectiveStop",),
        )
    ]

    normalized = normalize_decision_stream(events)

    assert normalized == [
        {
            "schema_version": "decision_event_v1",
            "event_type": "decision_event",
            "code": "ENTRY_FILLED",
            "ts": "2026-04-25T10:00:00+00:00",
            "symbol": "MNQ",
            "timeframe": "5m",
            "bot_id": "",
            "strategy_id": "NQDTC_v2.1",
            "family_id": "momentum",
            "portfolio_id": "paper_default",
            "strategy_version": "",
            "config_version": "cfg_abc",
            "portfolio_config_version": "",
            "risk_config_version": "risk_abc",
            "allocation_version": "alloc_abc",
            "strategy_registry_version": "registry_abc",
            "deployment_id": "dep_2026_04_25_abc",
            "parameter_set_id": "param_abc",
            "code_sha": "abc123",
            "trace_id": "trace_abc",
            "bar_id": "MNQ-20260425T100000Z",
            "decision_kind": "fill",
            "sequence": 7,
            "state_ref": "state:1",
            "emitted_actions": ["SubmitProtectiveStop"],
            "details": {"nested": {"a": 1, "b": 2}, "qty": 1},
        }
    ]
