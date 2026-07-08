from __future__ import annotations

import json

import pytest

from strategies.momentum.instrumentation.src.sidecar import Sidecar as MomentumSidecar
from strategies.momentum.instrumentation.src.sidecar import _EVENT_PRIORITY as MOMENTUM_PRIORITY
from strategies.momentum.instrumentation.src.sidecar import _DIR_TO_EVENT_TYPE as MOMENTUM_DIRS
from strategies.stock.instrumentation.src.sidecar import Sidecar as StockSidecar
from strategies.stock.instrumentation.src.sidecar import _EVENT_PRIORITY as STOCK_PRIORITY
from strategies.stock.instrumentation.src.sidecar import _DIR_TO_EVENT_TYPE as STOCK_DIRS
from strategies.swing.instrumentation.src.sidecar import Sidecar as SwingSidecar
from strategies.swing.instrumentation.src.sidecar import _DIR_TO_EVENT_TYPE as SWING_DIRS
from strategies.swing.instrumentation.src.sidecar import _PRIORITY_MAP as SWING_PRIORITY


def _config(tmp_path, bot_id: str) -> dict:
    return {
        "bot_id": bot_id,
        "data_dir": str(tmp_path),
        "sidecar": {"relay_url": "http://relay.local/events", "batch_size": 10},
    }


@pytest.mark.parametrize(
    ("sidecar_cls", "bot_id"),
    [
        (StockSidecar, "stock_trader"),
        (MomentumSidecar, "momentum_nq_01"),
        (SwingSidecar, "swing_multi_01"),
    ],
)
def test_sidecar_envelope_duplicates_payload_lineage(tmp_path, sidecar_cls, bot_id) -> None:
    raw = {
        "timestamp": "2026-05-31T12:00:00+00:00",
        "trade_id": "t1",
        "lineage": {
            "bot_id": bot_id,
            "strategy_id": "IARIC_v1",
            "family_id": "stock",
            "portfolio_id": "paper_default",
            "strategy_version": "IARIC_v1.0.0",
            "config_version": "cfg_1",
            "portfolio_config_version": "pcfg_1",
            "risk_config_version": "risk_1",
            "allocation_version": "alloc_1",
            "strategy_registry_version": "registry_1",
            "deployment_id": "dep_1",
            "parameter_set_id": "param_1",
            "code_sha": "abc123",
            "trace_id": "trace_1",
            "schema_version": "trade_event_v2",
        },
    }

    wrapped = sidecar_cls(_config(tmp_path, bot_id))._wrap_event(raw, "trade")

    assert wrapped["bot_id"] == bot_id
    assert wrapped["strategy_id"] == "IARIC_v1"
    assert wrapped["family_id"] == "stock"
    assert wrapped["deployment_id"] == "dep_1"
    assert wrapped["parameter_set_id"] == "param_1"
    assert wrapped["schema_version"] == "trade_event_v2"
    assert json.loads(wrapped["payload"])["lineage"]["trace_id"] == "trace_1"


@pytest.mark.parametrize(
    ("sidecar_cls", "bot_id", "dir_map"),
    [
        (StockSidecar, "stock_trader", STOCK_DIRS),
        (MomentumSidecar, "momentum_nq_01", MOMENTUM_DIRS),
        (SwingSidecar, "swing_multi_01", SWING_DIRS),
    ],
)
def test_sidecar_watches_new_contract_directories(tmp_path, sidecar_cls, bot_id, dir_map) -> None:
    expected = set()
    for subdir, event_type in dir_map.items():
        path_dir = tmp_path / subdir
        path_dir.mkdir()
        if subdir == "daily":
            path = path_dir / "daily_2026-05-31.json"
            path.write_text(json.dumps({"timestamp": "2026-05-31T12:00:00+00:00"}))
        else:
            path = path_dir / f"{subdir}_2026-05-31.jsonl"
            path.write_text(json.dumps({"timestamp": "2026-05-31T12:00:00+00:00"}) + "\n")
        expected.add((path, event_type))

    sidecar = sidecar_cls(_config(tmp_path, bot_id))

    files = sidecar._get_event_files()
    assert expected.issubset(set(files))


@pytest.mark.parametrize(
    ("sidecar_cls", "bot_id"),
    [
        (StockSidecar, "stock_trader"),
        (MomentumSidecar, "momentum_nq_01"),
        (SwingSidecar, "swing_multi_01"),
    ],
)
def test_sidecar_envelope_extracts_metadata_v2_lineage(tmp_path, sidecar_cls, bot_id) -> None:
    raw = {
        "timestamp": "2026-05-31T12:00:00+00:00",
        "event_metadata": {
            "event_id": "evt_1",
            "bot_id": bot_id,
            "strategy_id": "IARIC_v1",
            "family_id": "stock",
            "portfolio_id": "paper_default",
            "payload_key": "trade_1:exit",
            "schema_version": "event_metadata_v2",
            "trace_id": "trace_metadata",
        },
    }

    wrapped = sidecar_cls(_config(tmp_path, bot_id))._wrap_event(raw, "trade")

    assert wrapped["strategy_id"] == "IARIC_v1"
    assert wrapped["family_id"] == "stock"
    assert wrapped["portfolio_id"] == "paper_default"
    assert wrapped["schema_version"] == "event_metadata_v2"
    assert wrapped["trace_id"] == "trace_metadata"
    assert wrapped["payload_key"] == "trade_1:exit"


@pytest.mark.parametrize("priority_map", [STOCK_PRIORITY, MOMENTUM_PRIORITY, SWING_PRIORITY])
def test_sidecar_priorities_match_contract_table(priority_map) -> None:
    assert priority_map["error"] == 0
    assert priority_map["risk_halt"] == 0
    assert priority_map["deployment"] == 1
    assert priority_map["config_snapshot"] == 1
    assert priority_map["parameter_change"] == 1
    assert priority_map["daily_snapshot"] == 1
    assert priority_map["family_daily_snapshot"] == 1
    assert priority_map["reconciliation_alert"] == 1
    assert priority_map["allocation_freeze"] == 1
    assert priority_map["allocation_unfreeze"] == 1
    assert priority_map["trade_entry"] == 2
    assert priority_map["trade"] == 2
    assert priority_map["missed_opportunity"] == 3
    assert priority_map["order"] == 3
    assert priority_map["filter_decision"] == 3
    assert priority_map["portfolio_rule_check"] == 3
    assert priority_map["risk_denial"] == 3
    assert priority_map["risk_decision"] == 3
    assert priority_map["allocation_drift"] == 3
    assert priority_map["drift_assignment"] == 3
    assert priority_map["admin_correction"] == 3
    assert priority_map["inferred_fill"] == 3
    assert priority_map["coordinator_action"] == 3
    assert priority_map["position_snapshot"] == 4
    assert priority_map["portfolio_snapshot"] == 4
    assert priority_map["allocation_snapshot"] == 4
    assert priority_map["sector_exposure"] == 4
    assert priority_map["correlation_snapshot"] == 4
    assert priority_map["indicator_snapshot"] == 4
    assert priority_map["market_snapshot"] == 4
    assert priority_map["orderbook_context"] == 4
    assert priority_map["post_exit"] == 4
    assert priority_map["decision_event"] == 4
    assert priority_map["heartbeat"] == 5
