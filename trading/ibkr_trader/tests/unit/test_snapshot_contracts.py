from __future__ import annotations

import json

from libs.instrumentation.event_contract import write_startup_events
from libs.instrumentation.lineage import LineageContext
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="swing_multi_01",
        strategy_id="TPC",
        family_id="swing",
        portfolio_id="paper_default",
        account_alias="paper_ibkr_1",
        strategy_version="TPC.1.0.0",
        config_version="cfg_1",
        portfolio_config_version="pcfg_1",
        risk_config_version="risk_1",
        allocation_version="alloc_1",
        strategy_registry_version="registry_1",
        deployment_id="dep_1",
        parameter_set_id="param_1",
        code_sha="abc123",
        trace_id="trace_1",
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_startup_events_write_deployment_config_allocation_portfolio_and_positions(tmp_path) -> None:
    rules = PortfolioRulesConfig(
        family_strategy_ids=("TPC", "ATRSS"),
        same_sector_heat_cap_R=2.75,
    )

    write_startup_events(
        tmp_path,
        _lineage(),
        effective_config={
            "bot_id": "swing_multi_01",
            "api_key": "secret",
            "broker_account": "U1234567",
        },
        allocation_state={"family_targets": {"swing": 0.33}},
        portfolio_state={"net_liquidation": 100_000.0, "account_id": "U1234567"},
        positions=[{"symbol": "QQQ", "qty": 3, "account_id": "U1234567"}],
        portfolio_rules_config=rules,
    )

    expected = {
        "deployments": "deployment",
        "config_snapshots": "config_snapshot",
        "allocations": "allocation_snapshot",
        "portfolio": "portfolio_snapshot",
        "positions": "position_snapshot",
    }
    for subdir, event_type in expected.items():
        files = list((tmp_path / subdir).glob("*.jsonl"))
        assert len(files) == 1
        [event] = _read_jsonl(files[0])
        assert event["event_type"] == event_type
        assert event["bot_id"] == "swing_multi_01"
        assert event["deployment_id"] == "dep_1"
        assert event["portfolio_config_version"] == "pcfg_1"
        assert event["risk_config_version"] == "risk_1"
        assert event["allocation_version"] == "alloc_1"
        assert event["lineage"]["trace_id"] == "trace_1"
        assert "lineage_gaps" not in event

    [config_event] = _read_jsonl(next((tmp_path / "config_snapshots").glob("*.jsonl")))
    [position_event] = _read_jsonl(next((tmp_path / "positions").glob("*.jsonl")))
    [portfolio_event] = _read_jsonl(next((tmp_path / "portfolio").glob("*.jsonl")))
    [allocation_event] = _read_jsonl(next((tmp_path / "allocations").glob("*.jsonl")))
    assert config_event["effective_config"]["api_key"] == "<redacted>"
    assert config_event["effective_config"]["broker_account"] == "<redacted>"
    assert (
        config_event["effective_config"]["effective_portfolio_config"]["portfolio_rules_config"][
            "same_sector_heat_cap_R"
        ]
        == 2.75
    )
    assert (
        config_event["effective_config"]["effective_risk_config"]["portfolio_rules_config"][
            "family_strategy_ids"
        ]
        == ["TPC", "ATRSS"]
    )
    assert position_event["symbol"] == "QQQ"
    assert position_event["qty"] == 3.0
    assert position_event["account_alias"] == "paper_ibkr_1"
    assert "positions" not in position_event
    assert "portfolio_state" not in portfolio_event
    assert "allocation_state" not in allocation_event
    assert "portfolio_heat_R" in portfolio_event
    assert "observed_strategy_weights" in allocation_event
    assert not _contains_key(position_event, "account_id")
    assert not _contains_key(portfolio_event, "account_id")
    assert not _contains_key(allocation_event, "account_id")


def test_startup_events_do_not_emit_placeholder_position_snapshot(tmp_path) -> None:
    write_startup_events(
        tmp_path,
        _lineage(),
        effective_config={"bot_id": "swing_multi_01"},
        allocation_state={},
        portfolio_state={},
        positions=[],
    )

    assert not (tmp_path / "positions").exists()
