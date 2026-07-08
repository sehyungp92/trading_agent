from __future__ import annotations

from libs.instrumentation.config_snapshot import (
    build_effective_portfolio_config,
    build_effective_risk_config,
    build_effective_strategy_config,
    default_yaml_watch_paths,
)
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig


def test_effective_strategy_config_includes_manifest_and_redacts_account() -> None:
    snapshot = build_effective_strategy_config(
        "IARIC_v1",
        env={"TRADING_MODE": "paper"},
        runtime_config={"bot_id": "stock_trader"},
    )

    assert snapshot["strategy_manifest"]["strategy_id"] == "IARIC_v1"
    assert snapshot["strategy_manifest"]["family"] == "stock"
    assert snapshot["runtime_config"]["bot_id"] == "stock_trader"
    assert snapshot["connection_group"]["account_id"] == "<redacted>"
    assert snapshot["env_overrides"]["TRADING_MODE"] == "paper"


def test_effective_portfolio_and_risk_configs_include_family_dynamic_rules() -> None:
    rules = PortfolioRulesConfig(
        family_strategy_ids=("IARIC_v1", "ALCB_v1"),
        same_sector_heat_cap_R=3.5,
    )

    portfolio = build_effective_portfolio_config(
        family_id="stock",
        portfolio_rules_config=rules,
    )
    risk = build_effective_risk_config(
        "stock",
        portfolio_rules_config=rules,
    )

    assert "IARIC_v1" in portfolio["enabled_strategy_ids"]
    assert portfolio["portfolio_rules_config"]["same_sector_heat_cap_R"] == 3.5
    assert risk["strategy_risk"]["IARIC_v1"]["unit_risk_dollars"] == 100.0
    assert risk["portfolio_rules_config"]["family_strategy_ids"] == ["IARIC_v1", "ALCB_v1"]
    assert "_VALID_COLLISION_ACTIONS" not in risk["portfolio_rules_config"]


def test_default_yaml_watch_paths_cover_repo_config_inputs() -> None:
    names = {path.name for path in default_yaml_watch_paths()}

    assert {
        "strategies.yaml",
        "portfolio.yaml",
        "sector_map.yaml",
        "routing.yaml",
        "contracts.yaml",
        "event_calendar.yaml",
    }.issubset(names)
