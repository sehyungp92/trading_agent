from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from libs.instrumentation.event_contract import enrich_payload
from libs.instrumentation.lineage import LineageContext, compute_risk_config_version
from libs.oms.instrumentation.portfolio_rule_event import build_portfolio_rule_event
from libs.oms.risk.portfolio_rules import PortfolioRuleChecker, PortfolioRulesConfig
from libs.oms.services.factory import _make_portfolio_rule_logger


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="stock_oms",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_id="paper_default",
        account_alias="paper_ibkr_1",
        strategy_version="IARIC_v1.0.0",
        config_version="cfg_runtime",
        portfolio_config_version="pcfg_runtime",
        risk_config_version="risk_runtime",
        allocation_version="alloc_runtime",
        strategy_registry_version="registry_runtime",
        deployment_id="dep_runtime",
        parameter_set_id="param_runtime",
        code_sha="abc123",
        trace_id="trace_runtime",
    )


def test_portfolio_rule_event_preserves_requested_approved_and_legacy_fields() -> None:
    event = build_portfolio_rule_event(
        {
            "rule": "directional_cap",
            "strategy_id": "IARIC_v1",
            "direction": "LONG",
            "symbol": "AAPL",
            "approved": False,
            "denial_reason": "directional_cap: LONG risk too high",
        },
        portfolio_rules_config=PortfolioRulesConfig(
            directional_cap_R=8.0,
            family_strategy_ids=("IARIC_v1", "ALCB_v1"),
        ),
        request_context={
            "requested_sizing": {"risk_R": 1.0, "qty": 10, "risk_dollars": 100.0},
            "state_before": {"current_equity": 100_000.0, "directional_cap_R": 8.0},
            "current_size_multiplier": 1.0,
        },
        lineage=_lineage(),
    )

    assert event["event_type"] == "portfolio_rule_check"
    assert event["schema_version"] == "portfolio_rule_check_v2"
    assert event["rule_name"] == "directional_cap"
    assert event["result"] == "block"
    assert event["details"]["reason"] == "directional_cap: LONG risk too high"
    assert event["details"]["blocked_symbol"] == "AAPL"
    assert event["requested_sizing"] == {"risk_R": 1.0, "qty": 10, "risk_dollars": 100.0}
    assert event["requested_qty"] == 10
    assert event["approved_qty"] == 0
    assert event["requested_risk_R"] == 1.0
    assert event["approved_risk_R"] == 0.0
    assert event["rule_trace_id"].startswith("rule_trace_")
    assert event["approved_sizing"]["qty"] == 0
    assert event["approved_sizing"]["size_multiplier"] == 0.0
    assert event["thresholds"]["directional_cap_R"] == 8.0
    assert "_VALID_COLLISION_ACTIONS" not in event["thresholds"]
    assert event["state_before"]["current_equity"] == 100_000.0
    assert event["state_after"]["approved"] is False
    assert event["risk_config_version"] == "risk_runtime"
    assert event["portfolio_rule_config_version"].startswith("risk_")
    assert event["allocation_version"] == "alloc_runtime"
    assert event["strategy_registry_version"] == "registry_runtime"
    assert event["param_set_id"] == "param_runtime"
    assert "lineage_gaps" not in event


def test_runtime_lineage_overlays_rule_only_config_version() -> None:
    raw = build_portfolio_rule_event(
        {"rule": "drawdown_tier", "approved": True, "size_multiplier": 0.5},
        portfolio_rules_config=PortfolioRulesConfig(initial_equity=100_000.0),
        request_context={
            "requested_sizing": {"risk_R": 1.0, "qty": 2, "risk_dollars": 200.0},
            "current_size_multiplier": 1.0,
        },
    )

    enriched = enrich_payload(
        raw,
        lineage=_lineage(),
        event_type="portfolio_rule_check",
        scope="portfolio",
    )

    assert raw["portfolio_rule_config_version"].startswith("risk_")
    assert enriched["risk_config_version"] == "risk_runtime"
    assert enriched["portfolio_rule_config_version"] == raw["portfolio_rule_config_version"]


def test_portfolio_rule_event_preserves_scale_to_zero_multiplier() -> None:
    event = build_portfolio_rule_event(
        {"rule": "drawdown_tier", "approved": True, "size_multiplier": 0.0},
        request_context={
            "requested_sizing": {"risk_R": 1.0, "qty": 10, "risk_dollars": 100.0},
            "current_size_multiplier": 1.0,
        },
        lineage=_lineage(),
    )

    assert event["result"] == "scale"
    assert event["approved_qty"] == 0
    assert event["approved_sizing"]["rule_size_multiplier"] == 0.0
    assert event["approved_sizing"]["size_multiplier"] == 0.0
    assert event["size_multiplier_after"] == 0.0


def test_portfolio_rule_logger_uses_current_dynamic_lineage(tmp_path) -> None:
    first_rules = PortfolioRulesConfig(directional_cap_R=3.0)
    second_rules = PortfolioRulesConfig(directional_cap_R=4.0)
    current = {
        "lineage": replace(
            _lineage(),
            risk_config_version=compute_risk_config_version({}, first_rules, {}),
        )
    }
    logger = _make_portfolio_rule_logger(
        data_dir=str(tmp_path),
        family_id="stock",
        lineage=lambda: current["lineage"],
    )

    logger(
        build_portfolio_rule_event(
            {"rule": "directional_cap", "approved": True, "strategy_id": "IARIC_v1"},
            portfolio_rules_config=first_rules,
        )
    )
    current["lineage"] = replace(
        _lineage(),
        risk_config_version=compute_risk_config_version({}, second_rules, {}),
    )
    logger(
        build_portfolio_rule_event(
            {"rule": "directional_cap", "approved": True, "strategy_id": "IARIC_v1"},
            portfolio_rules_config=second_rules,
        )
    )

    path = next((tmp_path / "portfolio_rules").glob("rules_*.jsonl"))
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert events[0]["risk_config_version"] == compute_risk_config_version({}, first_rules, {})
    assert events[1]["risk_config_version"] == compute_risk_config_version({}, second_rules, {})
    assert events[0]["risk_config_version"] != events[1]["risk_config_version"]
    assert events[1]["portfolio_rule_config_version"] == compute_risk_config_version({}, second_rules, {})


@pytest.mark.asyncio
async def test_portfolio_rule_checker_threads_trace_context_into_rule_events() -> None:
    events: list[dict] = []

    async def _signal(_: str):
        return None

    async def _directional_risk(_: str) -> float:
        return 0.0

    checker = PortfolioRuleChecker(
        PortfolioRulesConfig(strategy_size_multipliers=(("IARIC_v1", 0.5),)),
        get_strategy_signal=_signal,
        get_directional_risk_R=_directional_risk,
        get_current_equity=lambda: 100_000.0,
        on_rule_event=events.append,
    )

    result = await checker.check_entry(
        "IARIC_v1",
        "LONG",
        new_risk_R=1.0,
        symbol="AAPL",
        new_qty=10,
        new_risk_dollars=100.0,
        trace_id="trace_signal_1",
        signal_id="sig_1",
        bar_id="bar_1",
        exchange_timestamp=datetime(2026, 5, 31, 14, 30, tzinfo=timezone.utc),
        lineage_context=_lineage(),
    )

    assert result.approved is True
    assert result.requested_qty == 10
    assert result.approved_qty == 5
    assert result.applied_rules == ("strategy_size_multiplier",)
    assert result.lineage_gap is False
    assert events[0]["schema_version"] == "portfolio_rule_check_v2"
    assert events[0]["trace_id"] == "trace_signal_1"
    assert events[0]["signal_id"] == "sig_1"
    assert events[0]["bar_id"] == "bar_1"
    assert events[0]["check_sequence"] == 1
    assert events[0]["approved_qty"] == 5
    assert events[0]["risk_config_version"] == "risk_runtime"
