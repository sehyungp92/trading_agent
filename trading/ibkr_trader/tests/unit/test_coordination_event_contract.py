from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from libs.instrumentation.lineage import LineageContext, compute_risk_config_version
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from strategies.momentum.coordinator import MomentumFamilyCoordinator
from strategies.stock.coordinator import StockFamilyCoordinator
from strategies.swing.coordinator import SwingFamilyCoordinator


def _lineage(family_id: str) -> LineageContext:
    return LineageContext(
        bot_id=f"{family_id}_bot",
        strategy_id="IARIC_v1",
        family_id=family_id,
        portfolio_id="paper_default",
        account_alias="paper_ibkr_1",
        strategy_version="IARIC_v1.0.0",
        config_version="cfg_runtime",
        portfolio_config_version="pcfg_runtime",
        risk_config_version="risk_startup",
        allocation_version="alloc_runtime",
        strategy_registry_version="registry_runtime",
        deployment_id="dep_runtime",
        parameter_set_id="param_runtime",
        code_sha="abc123",
        trace_id="trace_runtime",
    )


class _FakeInstrumentation:
    def __init__(self, data_dir, family_id: str) -> None:
        self._config = {"data_dir": str(data_dir)}
        self.lineage = _lineage(family_id)
        self.refreshed_rules = None

    def refresh_lineage(self, rules_config) -> None:
        self.refreshed_rules = rules_config
        self.lineage = replace(
            self.lineage,
            risk_config_version=compute_risk_config_version({}, rules_config, {}),
        )


@pytest.mark.parametrize(
    ("coordinator_cls", "family_id"),
    [
        (StockFamilyCoordinator, "stock"),
        (MomentumFamilyCoordinator, "momentum"),
    ],
)
def test_family_coordination_events_are_enriched(tmp_path, coordinator_cls, family_id: str) -> None:
    rules = PortfolioRulesConfig(directional_cap_R=4.0)
    instr = _FakeInstrumentation(tmp_path, family_id)
    coordinator = object.__new__(coordinator_cls)
    coordinator._instrumentations = [instr]
    coordinator._portfolio_checkers = [SimpleNamespace(_cfg=rules)]
    coordinator._regime_adjusted_rules = rules
    base_rules = PortfolioRulesConfig(directional_cap_R=3.0)
    coordinator._base_portfolio_rules = base_rules

    coordinator._emit_regime_event({"family": family_id, "regime": "RISK_OFF"})

    path = next((tmp_path / "coordination_events").glob("coordination_events_*.jsonl"))
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert instr.refreshed_rules is rules
    assert event["event_type"] == "coordinator_action"
    assert event["schema_version"] == "coordinator_action_v1"
    assert event["scope"] == "family"
    assert event["action_type"] == "regime_rules_change"
    assert event["risk_config_version"] == compute_risk_config_version({}, rules, {})
    assert event["risk_config_version_before"] == compute_risk_config_version({}, base_rules, {})
    assert event["risk_config_version_after"] == compute_risk_config_version({}, rules, {})
    assert event["portfolio_rule_config_version_before"] == compute_risk_config_version({}, base_rules, {})
    assert event["portfolio_rule_config_version_after"] == compute_risk_config_version({}, rules, {})
    assert (
        event["effective_config_evidence"]["portfolio_rules_config_before"]["directional_cap_R"]
        == 3.0
    )
    assert (
        event["effective_config_evidence"]["portfolio_rules_config_after"]["directional_cap_R"]
        == 4.0
    )
    assert event["family_id"] == family_id
    assert event["trace_id"] == "trace_runtime"


def test_swing_coordination_events_are_enriched(tmp_path) -> None:
    rules = PortfolioRulesConfig(directional_cap_R=5.0)
    ctx = _FakeInstrumentation(tmp_path, "swing")
    ctx.data_dir = str(tmp_path)
    coordinator = object.__new__(SwingFamilyCoordinator)
    coordinator._instrumentation_ctx = ctx
    coordinator._kits = {}
    coordinator._portfolio_checker = SimpleNamespace(_cfg=rules)
    coordinator._regime_adjusted_rules = rules
    base_rules = PortfolioRulesConfig(directional_cap_R=3.0)
    coordinator._base_portfolio_rules = base_rules

    coordinator._emit_crisis_event({"family": "swing", "alert_level": "RISK_OFF"})

    path = next((tmp_path / "coordination_events").glob("coordination_events_*.jsonl"))
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert ctx.refreshed_rules is rules
    assert event["event_type"] == "coordinator_action"
    assert event["schema_version"] == "coordinator_action_v1"
    assert event["scope"] == "family"
    assert event["action_type"] == "crisis_alert_change"
    assert event["risk_config_version"] == compute_risk_config_version({}, rules, {})
    assert event["risk_config_version_before"] == compute_risk_config_version({}, base_rules, {})
    assert event["risk_config_version_after"] == compute_risk_config_version({}, rules, {})
    assert event["portfolio_rule_config_version_before"] == compute_risk_config_version({}, base_rules, {})
    assert event["portfolio_rule_config_version_after"] == compute_risk_config_version({}, rules, {})
    assert (
        event["effective_config_evidence"]["portfolio_rules_config_before"]["directional_cap_R"]
        == 3.0
    )
    assert (
        event["effective_config_evidence"]["portfolio_rules_config_after"]["directional_cap_R"]
        == 5.0
    )
    assert event["family_id"] == "swing"
    assert event["trace_id"] == "trace_runtime"
