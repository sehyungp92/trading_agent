from __future__ import annotations

import pytest

from strategies.momentum.instrumentation.src.config_watcher import ConfigWatcher as MomentumConfigWatcher
from strategies.stock.instrumentation.src.config_watcher import ConfigWatcher as StockConfigWatcher
from strategies.swing.instrumentation.src.config_watcher import (
    ConfigWatcher as SwingConfigWatcher,
    ParameterChangeEvent,
)


@pytest.mark.parametrize("watcher_cls", [StockConfigWatcher, MomentumConfigWatcher])
def test_config_change_events_use_full_source_versions(tmp_path, watcher_cls) -> None:
    watcher = watcher_cls(
        bot_id="bot",
        config_modules=[],
        data_dir=tmp_path,
        lineage={
            "bot_id": "bot",
            "portfolio_config_version": "pcfg_1",
            "risk_config_version": "risk_1",
            "allocation_version": "alloc_1",
        },
    )

    event = watcher._make_event("strategy.config", "HEAT_CAP_R", 2.0, 3.0)

    assert event["config_version_before"] != event["config_version_after"]
    assert event["portfolio_config_version_before"] == "pcfg_1"
    assert event["portfolio_config_version_after"] == "pcfg_1"
    assert event["risk_config_version_before"].startswith("risk_")
    assert event["risk_config_version_after"].startswith("risk_")
    assert event["risk_config_version_before"] != event["risk_config_version_after"]
    assert event["allocation_version_before"] == "alloc_1"
    assert event["allocation_version_after"] == "alloc_1"
    assert event["is_safety_critical"] is True


@pytest.mark.parametrize("watcher_cls", [StockConfigWatcher, MomentumConfigWatcher])
def test_stock_and_momentum_watch_yaml_config_inputs(tmp_path, watcher_cls) -> None:
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("risk:\n  heat_cap_R: 2.5\n", encoding="utf-8")
    watcher = watcher_cls(
        bot_id="bot",
        config_modules=[],
        data_dir=tmp_path,
        lineage={
            "bot_id": "bot",
            "portfolio_config_version": "pcfg_1",
            "risk_config_version": "risk_1",
            "allocation_version": "alloc_1",
            "proposal_ids": ["proposal-1"],
            "suggestion_ids": ["suggestion-1"],
            "approval_id": "approval-1",
        },
        yaml_paths=[config_path],
    )

    config_path.write_text("risk:\n  heat_cap_R: 3.0\n", encoding="utf-8")
    [event] = watcher.check()

    assert event["event_type"] == "parameter_change"
    assert event["change_source"] == "config_file"
    assert event["config_file"] == str(config_path)
    assert event["param_name"] == "risk.heat_cap_R"
    assert event["old_value"] == 2.5
    assert event["new_value"] == 3.0
    assert event["config_version_before"] != event["config_version_after"]
    assert event["portfolio_config_version_before"] != event["portfolio_config_version_after"]
    assert event["risk_config_version_before"] != event["risk_config_version_after"]
    assert event["allocation_version_before"] != event["allocation_version_after"]
    assert event["approval_id"] == "approval-1"
    assert event["proposal_ids"] == ["proposal-1"]
    assert event["suggestion_ids"] == ["suggestion-1"]


def test_swing_config_watcher_uses_shared_contract_and_preserves_event_shape(tmp_path) -> None:
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("risk:\n  heat_cap_R: 2.5\n", encoding="utf-8")
    watcher = SwingConfigWatcher(
        {
            "bot_id": "bot",
            "data_dir": tmp_path,
            "lineage": {
                "bot_id": "bot",
                "portfolio_config_version": "pcfg_1",
                "risk_config_version": "risk_1",
                "allocation_version": "alloc_1",
            },
        },
        config_modules=[],
        yaml_paths=[config_path],
    )
    watcher.take_baseline()

    config_path.write_text("risk:\n  heat_cap_R: 3.0\n", encoding="utf-8")
    [event] = watcher.check()

    assert isinstance(event, ParameterChangeEvent)
    payload = event.to_dict()
    assert payload["param_name"] == "risk.heat_cap_R"
    assert payload["old_value"] == 2.5
    assert payload["new_value"] == 3.0
    assert payload["risk_config_version_before"] != payload["risk_config_version_after"]
