from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from libs.oms.persistence.in_memory import InMemoryRepository
from tests.integration.parity.fixtures import (
    ParityFixtureError,
    fixture_source_fingerprint,
    load_parity_fixture,
    normalized_source_payload,
)
from tests.integration.parity.oms_hydration import (
    ParityOmsHydrationError,
    build_instruments_from_fixture,
    hydrate_repository_from_fixture,
)
from tests.integration.parity.source_contract import (
    CONSUMPTION_KINDS,
    EXCLUDED_TOP_LEVEL_KEYS,
    RUNTIME_INPUT_CONTRACT_PATHS,
    SOURCE_FIELD_CONTRACTS,
    SOURCE_TOP_LEVEL_KEYS,
    validate_contract_table,
)
from tests.integration.parity.replay_idle import IDLE_REPLAY_ADAPTERS, run_idle_market_core
from tests.integration.parity.runtime_source import runtime_source_payload
from tests.integration.parity.source_inputs import IDLE_MARKET_INPUT_ARTIFACT_KEYS, idle_market_input

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "parity" / "layer2" / "tpc_entry_fill.json"
MOMENTUM_FAMILY_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "parity"
    / "layer3"
    / "momentum_family_shared_risk.json"
)
STOCK_FAMILY_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "parity"
    / "layer3"
    / "stock_family_collision.json"
)


def test_parity_fixture_fingerprint_is_stable_for_same_source_payload() -> None:
    fixture = load_parity_fixture(FIXTURE)

    assert fixture_source_fingerprint(fixture) == fixture_source_fingerprint(deepcopy(fixture))


def test_parity_fixture_fingerprint_changes_when_market_input_changes() -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["bars"][0]["close"] += 0.01

    assert fixture_source_fingerprint(fixture) != fixture_source_fingerprint(changed)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["artifacts"].update({"new_signal": {"score": 0.75}}),
        lambda payload: payload["initial_strategy_state"]["TPC"].update({"pending_setup": "changed"}),
        lambda payload: payload["initial_family_state"].update({"risk_flag": "changed"}),
        lambda payload: payload["broker_event_script"][0].update({"price": 101.5}),
    ],
)
def test_parity_fixture_fingerprint_changes_for_source_inputs(mutate) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    mutate(changed)

    assert fixture_source_fingerprint(fixture) != fixture_source_fingerprint(changed)


def test_parity_fixture_fingerprint_ignores_expected_outputs_metadata() -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["expected_normalized_outputs"] = {"not": "source"}

    assert normalized_source_payload(fixture) == normalized_source_payload(changed)
    assert fixture_source_fingerprint(fixture) == fixture_source_fingerprint(changed)


@pytest.mark.parametrize(
    "field",
    ["expected_fill_model", "timezone", "market_calendar"],
)
def test_parity_fixture_fingerprint_excludes_unconsumed_metadata(field: str) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    if isinstance(changed[field], dict):
        changed[field]["changed"] = "metadata"
    else:
        changed[field] = "America/New_York"

    assert normalized_source_payload(fixture) == normalized_source_payload(changed)
    assert fixture_source_fingerprint(fixture) == fixture_source_fingerprint(changed)


def test_parity_fixture_fingerprint_ignores_broker_generated_exec_ids() -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["broker_event_script"][0]["exec_id"] = "different-generated-id"

    assert normalized_source_payload(fixture) == normalized_source_payload(changed)
    assert fixture_source_fingerprint(fixture) == fixture_source_fingerprint(changed)


def test_parity_fixture_fingerprint_preserves_broker_script_order() -> None:
    fixture = load_parity_fixture(FIXTURE)
    event_a = deepcopy(fixture["broker_event_script"][0])
    event_b = deepcopy(event_a)
    event_a["price"] = 104.0
    event_a["timestamp"] = "2026-05-20T14:31:00+00:00"
    event_b["price"] = 104.25
    event_b["timestamp"] = "2026-05-20T14:32:00+00:00"
    ordered = deepcopy(fixture)
    reversed_order = deepcopy(fixture)
    ordered["broker_event_script"] = [event_a, event_b]
    reversed_order["broker_event_script"] = [event_b, event_a]

    assert fixture_source_fingerprint(ordered) != fixture_source_fingerprint(reversed_order)


def test_idle_market_input_consumes_bars_in_canonical_time_order() -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    bars = changed["artifacts"]["nqdtc"]["idle_market_input"]["bars"]
    earlier = dict(bars[0])
    earlier["timestamp"] = "2026-05-20T14:25:00+00:00"
    earlier["close"] = float(earlier["close"]) - 1.0
    changed["artifacts"]["nqdtc"]["idle_market_input"]["bars"] = [bars[0], earlier]

    consumed = idle_market_input(changed, "NQDTC_v2.1")["bars"]

    assert consumed == sorted(consumed, key=lambda row: row["timestamp"])


def test_idle_replay_registry_covers_configured_idle_inputs() -> None:
    assert set(IDLE_MARKET_INPUT_ARTIFACT_KEYS) <= set(IDLE_REPLAY_ADAPTERS)


def test_idle_replay_core_rejects_configured_child_without_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)
    monkeypatch.delitem(IDLE_REPLAY_ADAPTERS, "NQDTC_v2.1")

    with pytest.raises(AssertionError, match="missing an idle replay adapter"):
        run_idle_market_core(fixture, "NQDTC_v2.1")


def test_schema_v2_rejects_removed_top_level_orders(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["orders"] = [{"strategy_id": "TPC"}]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="orders"):
        load_parity_fixture(path)


def test_schema_v2_rejects_removed_strategy_inputs(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["strategy_inputs"] = {"entry_actions": []}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="strategy_inputs"):
        load_parity_fixture(path)


def test_schema_v2_rejects_removed_generated_decision_trace(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["generated_decision_trace"] = [{"decision_id": "volatile"}]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="generated_decision_trace"):
        load_parity_fixture(path)


def test_schema_v2_rejects_top_level_entry_actions(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["entry_actions"] = []
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="entry_actions"):
        load_parity_fixture(path)


@pytest.mark.parametrize("payload", [{"parity_entry_signals": []}, {"nested": {"entry_actions": []}}])
def test_schema_v2_rejects_scripted_artifacts(tmp_path: Path, payload: dict) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["artifacts"].update(payload)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="scripted decision"):
        load_parity_fixture(path)


def test_schema_v2_rejects_scripted_decision_fields_anywhere_in_fixture(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["strategy_config"]["nested"] = {"entry_actions": []}
    changed["expected_normalized_outputs"]["entry_actions"] = [{"allowed": "metadata"}]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="strategy_config.nested.entry_actions"):
        load_parity_fixture(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fixture_source_rejects_non_finite_float(value: float) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["bars"][0]["close"] = value

    with pytest.raises(ParityFixtureError, match="non-finite"):
        normalized_source_payload(changed)


def test_schema_v2_rejects_naive_fixture_timestamps(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["clock_start"] = "2026-05-20T14:30:00"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="timezone-aware"):
        load_parity_fixture(path)


def test_schema_v2_rejects_json_non_finite_numbers(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text('{"schema_version": 2, "surface": "TPC", "bad": NaN}', encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="non-finite JSON"):
        load_parity_fixture(path)


def test_schema_v2_allows_expected_fill_model_to_be_omitted(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed.pop("expected_fill_model")
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    loaded = load_parity_fixture(path)

    assert "expected_fill_model" not in normalized_source_payload(loaded)


def test_schema_v2_requires_broker_order_match(tmp_path: Path) -> None:
    fixture = load_parity_fixture(FIXTURE)
    changed = deepcopy(fixture)
    changed["broker_event_script"][0].pop("order_match")
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="order_match"):
        load_parity_fixture(path)


def test_schema_v2_requires_idle_market_input_for_configured_idle_family_child(tmp_path: Path) -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["artifacts"].pop("nqdtc")
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="NQDTC_v2.1"):
        load_parity_fixture(path)


def test_schema_v2_rejects_timestamp_only_no_order_probe(tmp_path: Path) -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["artifacts"]["nqdtc"] = {
        "no_order_probe": {
            "symbol": "NQ",
            "timeframe": "5m",
            "timestamp": "2026-05-20T14:30:00+00:00",
        }
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="timestamp-only"):
        load_parity_fixture(path)


@pytest.mark.parametrize(
    ("fixture_path", "field_name"),
    [
        (MOMENTUM_FAMILY_FIXTURE, "expected_family_decision"),
        (STOCK_FAMILY_FIXTURE, "collision_policy"),
    ],
)
def test_schema_v2_rejects_unconsumed_layer3_initial_family_state(
    tmp_path: Path,
    fixture_path: Path,
    field_name: str,
) -> None:
    fixture = load_parity_fixture(fixture_path)
    changed = deepcopy(fixture)
    changed["initial_family_state"][field_name] = "metadata_not_runtime_state"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="unconsumed Layer-3 source"):
        load_parity_fixture(path)


def test_layer3_source_contract_lists_all_fingerprinted_top_level_fields() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)

    assert set(normalized_source_payload(fixture)) == set(SOURCE_TOP_LEVEL_KEYS)
    assert "family" in SOURCE_TOP_LEVEL_KEYS


def test_layer3_source_contract_declares_nested_consumers() -> None:
    paths = {row.path for row in SOURCE_FIELD_CONTRACTS}

    assert {
        "artifacts.<idle>.idle_market_input",
        "artifacts.iaric",
        "artifacts.nq_regime",
        "artifacts.overlay_rebalance",
        "family_config.portfolio_rules",
        "family_config.strategies",
        "initial_family_state.overlay",
        "initial_repository_state.orders",
        "initial_repository_state.positions",
        "initial_strategy_state.<strategy_id>",
        "strategy_config.config_overrides",
    } <= paths


def test_layer3_source_contract_rejects_unknown_source_field(tmp_path: Path) -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["new_source_field"] = {"score": 1}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="uncontracted top-level source field"):
        load_parity_fixture(path)


def test_normalized_source_payload_rejects_unknown_source_field() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["new_source_field"] = {"score": 1}

    with pytest.raises(ParityFixtureError, match="uncontracted top-level source field"):
        normalized_source_payload(changed)


def test_layer3_source_contract_expected_outputs_are_excluded() -> None:
    assert "expected_normalized_outputs" in EXCLUDED_TOP_LEVEL_KEYS
    assert "expected_trade_count" in EXCLUDED_TOP_LEVEL_KEYS
    assert "expected_normalized_outputs" not in SOURCE_TOP_LEVEL_KEYS


def test_layer3_source_contract_marks_each_field_consumed_excluded_or_rejected() -> None:
    validate_contract_table()
    assert SOURCE_FIELD_CONTRACTS
    assert all(row.consumption_kind in CONSUMPTION_KINDS for row in SOURCE_FIELD_CONTRACTS)
    assert all(
        row.live_consumer or row.replay_consumer or row.post_order_consumer or row.derived_from or row.validation
        for row in SOURCE_FIELD_CONTRACTS
    )
    for row in SOURCE_FIELD_CONTRACTS:
        if row.consumption_kind == "live_replay":
            assert row.live_consumer and row.replay_consumer
        if row.consumption_kind == "derived_runtime":
            assert row.path.startswith("runtime_inputs.")
            assert row.derived_from and row.live_consumer and row.replay_consumer
        if row.consumption_kind in {"excluded_metadata", "rejected"}:
            assert not row.included_in_fingerprint


@pytest.mark.parametrize("field", ["market_calendar", "expected_fill_model", "timezone"])
def test_unconsumed_metadata_source_fields_are_excluded(field: str) -> None:
    contract = next(row for row in SOURCE_FIELD_CONTRACTS if row.path == field)

    assert not contract.included_in_fingerprint
    assert contract.consumption_kind == "excluded_metadata"


def test_top_level_family_is_fingerprinted_and_must_match_configured_family(tmp_path: Path) -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    assert normalized_source_payload(fixture)["family"] == "stock"

    changed = deepcopy(fixture)
    changed["family"] = "momentum"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ParityFixtureError, match="diverges"):
        load_parity_fixture(path)


def test_runtime_inputs_contract_allows_only_consumed_derived_payloads() -> None:
    assert set(RUNTIME_INPUT_CONTRACT_PATHS) == {
        "runtime_inputs.configured_strategy_ids",
        "runtime_inputs.portfolio_rules",
        "runtime_inputs.overlay_rebalance",
    }
    for path in RUNTIME_INPUT_CONTRACT_PATHS:
        contract = next(row for row in SOURCE_FIELD_CONTRACTS if row.path == path)
        assert contract.consumption_kind == "derived_runtime"
        assert contract.derived_from


def test_runtime_source_payload_uses_contracted_runtime_inputs() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    runtime_input_keys = {path.removeprefix("runtime_inputs.") for path in RUNTIME_INPUT_CONTRACT_PATHS}

    assert set(runtime_source_payload(fixture)["runtime_inputs"]) == runtime_input_keys


@pytest.mark.asyncio
async def test_initial_order_hydration_rejects_malformed_order_id() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["initial_repository_state"]["orders"] = [
        {
            "strategy_id": "ALCB_v1",
            "symbol": "MSFT",
            "side": "BUY",
            "qty": 1,
            "order_type": "LIMIT",
            "limit_price": 411.0,
            "role": "ENTRY",
            "status": "WORKING",
            "risk_context": {
                "planned_entry_price": 411.0,
                "stop_for_risk": 401.0,
                "risk_dollars": 10.0,
            },
        }
    ]

    with pytest.raises(ParityOmsHydrationError, match="oms_order_id or client_order_id"):
        await hydrate_repository_from_fixture(
            changed,
            InMemoryRepository(),
            build_instruments_from_fixture(changed),
        )


@pytest.mark.asyncio
async def test_initial_position_hydration_rejects_unknown_instrument() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["initial_repository_state"]["positions"] = [
        {
            "strategy_id": "ALCB_v1",
            "symbol": "UNKNOWN",
            "net_qty": 1,
            "avg_price": 411.0,
        }
    ]

    with pytest.raises(ParityOmsHydrationError, match="unknown instrument"):
        await hydrate_repository_from_fixture(
            changed,
            InMemoryRepository(),
            build_instruments_from_fixture(changed),
        )
