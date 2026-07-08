from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from libs.instrumentation.lineage import (
    compute_risk_config_version,
    lineage_from_config,
    lineage_from_runtime,
)
from libs.oms.services.factory import (
    _make_fill_lifecycle_writer,
    _make_reconciliation_lifecycle_writer,
)
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from regime.context import RegimeContext
from strategies.momentum.coordinator import MomentumFamilyCoordinator
from strategies.momentum.instrumentation.src.bootstrap import (
    InstrumentationManager as MomentumInstrumentationManager,
)
from strategies.momentum.instrumentation.src.facade import InstrumentationKit as MomentumInstrumentationKit
from strategies.stock.coordinator import StockFamilyCoordinator
from strategies.stock.instrumentation.src.bootstrap import (
    InstrumentationManager as StockInstrumentationManager,
)
from strategies.stock.instrumentation.src.facade import InstrumentationKit as StockInstrumentationKit
from strategies.swing.coordinator import SwingFamilyCoordinator
from strategies.swing.instrumentation.src.context import InstrumentationContext
from tests.unit.test_coordination_event_contract import _FakeInstrumentation


def _regime_ctx(regime: str = "S") -> RegimeContext:
    return RegimeContext(
        regime=regime,
        regime_confidence=0.9,
        stress_level=0.2,
        stress_onset=False,
        shift_velocity=0.0,
        suggested_leverage_mult=1.0,
        regime_allocations={"SPY": 0.3, "TLT": 0.3, "GLD": 0.2, "CASH": 0.2},
    )


class _FakeChecker:
    def __init__(self, cfg: PortfolioRulesConfig):
        self._cfg = cfg

    def update_config(self, cfg: PortfolioRulesConfig) -> None:
        self._cfg = cfg


def _manager_with_facade(tmp_path, *, family: str):
    if family == "stock":
        manager_cls = StockInstrumentationManager
        kit_cls = StockInstrumentationKit
        strategy_id = "IARIC_v1"
        bot_id = "stock_trader"
    else:
        manager_cls = MomentumInstrumentationManager
        kit_cls = MomentumInstrumentationKit
        strategy_id = "NQ_REGIME"
        bot_id = "momentum_nq_01"

    base_rules = PortfolioRulesConfig(directional_cap_R=6.0, initial_equity=50_000.0)
    config = {
        "bot_id": bot_id,
        "strategy_id": strategy_id,
        "strategy_type": "test",
        "family_id": family,
        "data_dir": str(tmp_path),
        "portfolio_id": "paper_default",
        "account_alias": "paper_ibkr_1",
    }
    manager = object.__new__(manager_cls)
    manager._config = dict(config)
    manager._strategy_id = strategy_id
    manager._strategy_type = "test"
    manager._get_applied_config = lambda: base_rules
    manager.lineage = lineage_from_config(
        config,
        family_id=family,
        strategy_id=strategy_id,
        portfolio_rules_config=base_rules,
    )
    manager._config["lineage"] = manager.lineage

    for attr in (
        "error_logger",
        "snapshot_service",
        "trade_logger",
        "missed_logger",
        "order_logger",
        "daily_builder",
    ):
        setattr(manager, attr, SimpleNamespace(_lineage=manager.lineage))
    manager.config_watcher = SimpleNamespace(_lineage=manager.lineage)
    kit = kit_cls(manager, strategy_type="test")
    return manager, kit, base_rules


def _expected_manager_lineage(manager, family: str, rules: PortfolioRulesConfig):
    lineage_config = dict(manager._config)
    lineage_config.pop("lineage", None)
    return lineage_from_config(
        lineage_config,
        family_id=family,
        strategy_id=lineage_config["strategy_id"],
        portfolio_rules_config=rules,
    )


def _read_latest_payload(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


@pytest.mark.parametrize(
    ("coordinator_cls", "family_id"),
    [
        (StockFamilyCoordinator, "stock"),
        (MomentumFamilyCoordinator, "momentum"),
    ],
)
def test_stock_and_momentum_runtime_rule_update_versions(tmp_path, coordinator_cls, family_id: str) -> None:
    base_rules = PortfolioRulesConfig(directional_cap_R=3.0)
    regime_rules = PortfolioRulesConfig(directional_cap_R=4.0)
    crisis_rules = PortfolioRulesConfig(directional_cap_R=2.0)
    instr = _FakeInstrumentation(tmp_path, family_id)
    coordinator = object.__new__(coordinator_cls)
    coordinator._instrumentations = [instr]
    coordinator._portfolio_checkers = [SimpleNamespace(_cfg=regime_rules)]
    coordinator._base_portfolio_rules = base_rules
    coordinator._regime_adjusted_rules = regime_rules

    coordinator._emit_regime_event({"family": family_id, "regime": "RISK_OFF"})
    coordinator._portfolio_checkers = [SimpleNamespace(_cfg=crisis_rules)]
    coordinator._regime_adjusted_rules = crisis_rules
    coordinator._emit_crisis_event({"family": family_id, "alert_level": "CRITICAL"})

    path = next((tmp_path / "coordination_events").glob("coordination_events_*.jsonl"))
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert events[0]["risk_config_version_before"] == compute_risk_config_version({}, base_rules, {})
    assert events[0]["risk_config_version_after"] == compute_risk_config_version({}, regime_rules, {})
    assert events[1]["risk_config_version_before"] == compute_risk_config_version({}, regime_rules, {})
    assert events[1]["risk_config_version_after"] == compute_risk_config_version({}, crisis_rules, {})
    assert instr.lineage.risk_config_version == compute_risk_config_version({}, crisis_rules, {})


def test_swing_runtime_rule_update_versions(tmp_path) -> None:
    base_rules = PortfolioRulesConfig(directional_cap_R=3.0)
    regime_rules = PortfolioRulesConfig(directional_cap_R=5.0)
    ctx = _FakeInstrumentation(tmp_path, "swing")
    ctx.data_dir = str(tmp_path)
    coordinator = object.__new__(SwingFamilyCoordinator)
    coordinator._instrumentation_ctx = ctx
    coordinator._kits = {}
    coordinator._portfolio_checker = SimpleNamespace(_cfg=regime_rules)
    coordinator._base_portfolio_rules = base_rules
    coordinator._regime_adjusted_rules = regime_rules

    coordinator._emit_regime_event({"family": "swing", "regime": "RISK_OFF"})

    path = next((tmp_path / "coordination_events").glob("coordination_events_*.jsonl"))
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["risk_config_version_before"] == compute_risk_config_version({}, base_rules, {})
    assert event["risk_config_version_after"] == compute_risk_config_version({}, regime_rules, {})
    assert ctx.lineage.risk_config_version == compute_risk_config_version({}, regime_rules, {})


@pytest.mark.parametrize(
    ("coordinator_cls", "family_id"),
    [
        (StockFamilyCoordinator, "stock"),
        (MomentumFamilyCoordinator, "momentum"),
    ],
)
def test_real_apply_regime_refreshes_existing_facade_loggers_and_lifecycle_writers(
    tmp_path,
    coordinator_cls,
    family_id: str,
) -> None:
    manager, kit, base_rules = _manager_with_facade(tmp_path, family=family_id)
    checker = _FakeChecker(base_rules)
    coordinator = object.__new__(coordinator_cls)
    coordinator.family_id = family_id
    coordinator._base_portfolio_rules = base_rules
    coordinator._regime_adjusted_rules = None
    coordinator._regime_ctx = None
    coordinator._crisis_ctx = None
    coordinator._portfolio_checkers = [checker]
    coordinator._instrumentations = [manager]
    coordinator._engine_map = {}
    coordinator._base_max_family_contracts = 10

    original_version = manager.lineage.risk_config_version
    original_config_version = manager.lineage.config_version

    coordinator.apply_regime(_regime_ctx("S"))

    expected_lineage = _expected_manager_lineage(manager, family_id, checker._cfg)
    expected_version = expected_lineage.risk_config_version
    assert manager.lineage.risk_config_version == expected_version
    assert manager.lineage.risk_config_version != original_version
    assert manager.lineage.config_version == expected_lineage.config_version
    assert manager.lineage.config_version != original_config_version
    for component in (
        kit._indicator_logger,
        kit._filter_event_logger,
        kit._orderbook_logger,
    ):
        assert component._lineage.risk_config_version == expected_version
        assert component._lineage.config_version == expected_lineage.config_version

    kit.on_indicator_snapshot(
        pair="AAPL" if family_id == "stock" else "NQ",
        indicators={"atr": 2.5},
        signal_name="regime_probe",
        signal_strength=0.8,
        decision="skip",
        strategy_type="test",
        bar_id="bar_1",
    )
    kit.on_filter_decisions(
        [{"filter_name": "risk_probe", "threshold": 1.0, "actual_value": 2.0, "passed": False}],
        pair="AAPL" if family_id == "stock" else "NQ",
        signal_name="regime_probe",
        strategy_type="test",
        bar_id="bar_1",
    )
    kit.on_orderbook_context(
        pair="AAPL" if family_id == "stock" else "NQ",
        best_bid=100.0,
        best_ask=100.1,
        trade_context="signal_eval",
    )

    emitted_paths = [
        next((tmp_path / "indicators").glob("indicators_*.jsonl")),
        next((tmp_path / "filter_decisions").glob("filter_decisions_*.jsonl")),
        next((tmp_path / "orderbook").glob("orderbook_*.jsonl")),
    ]
    for path in emitted_paths:
        payload = _read_latest_payload(path)
        assert payload["risk_config_version"] == expected_version
        assert payload["lineage"]["risk_config_version"] == expected_version
        assert payload["config_version"] == expected_lineage.config_version
        assert payload["lineage"]["config_version"] == expected_lineage.config_version

    fill_writer = _make_fill_lifecycle_writer(str(tmp_path), lambda: manager.lineage)
    position = {
        "portfolio_id": "paper_default",
        "account_alias": "paper_ibkr_1",
        "family_id": family_id,
        "strategy_id": manager._strategy_id,
        "symbol": "AAPL" if family_id == "stock" else "NQ",
        "qty": 1,
        "avg_price": 100.0,
        "mark_price": 101.0,
    }
    fill_writer(
        {
            "position": position,
            "positions": [position],
            "fill": {"exec_id": "fill_1", "price": 101.0},
            "order": {"strategy_id": manager._strategy_id, "symbol": position["symbol"]},
            "portfolio_risk": {"open_risk_R": 0.25},
            "account_state": {"account_alias": "paper_ibkr_1", "equity": 50_000.0},
            "allocation_targets": {},
        }
    )
    position_payload = _read_latest_payload(next((tmp_path / "positions").glob("positions_*.jsonl")))
    assert position_payload["risk_config_version"] == expected_version
    assert position_payload["config_version"] == expected_lineage.config_version

    recon_writer = _make_reconciliation_lifecycle_writer(str(tmp_path), lambda: manager.lineage)
    recon_writer(
        {
            "lifecycle_action": "allocation_freeze",
            "status": "active",
            "details": {"family_id": family_id},
        }
    )
    recon_payload = _read_latest_payload(next((tmp_path / "allocation_drift").glob("allocation_drift_*.jsonl")))
    assert recon_payload["risk_config_version"] == expected_version
    assert recon_payload["config_version"] == expected_lineage.config_version


def test_real_swing_apply_regime_refreshes_existing_context_loggers(tmp_path) -> None:
    base_rules = PortfolioRulesConfig(directional_cap_R=6.0, initial_equity=100_000.0)
    checker = _FakeChecker(base_rules)
    ctx = InstrumentationContext(
        data_dir=str(tmp_path),
        lineage=_FakeInstrumentation(tmp_path, "swing").lineage,
    )
    ctx.indicator_logger = SimpleNamespace(_lineage=ctx.lineage)
    ctx.filter_logger = SimpleNamespace(_lineage=ctx.lineage)
    ctx.orderbook_logger = SimpleNamespace(_lineage=ctx.lineage)
    coordinator = object.__new__(SwingFamilyCoordinator)
    coordinator.family_id = "swing"
    coordinator._base_portfolio_rules = base_rules
    coordinator._regime_adjusted_rules = None
    coordinator._regime_ctx = None
    coordinator._crisis_ctx = None
    coordinator._portfolio_checker = checker
    coordinator._instrumentation_ctx = ctx
    coordinator._kits = {}
    coordinator._overlay_engine = None

    original_lineage = ctx.lineage
    original_version = original_lineage.risk_config_version
    original_config_version = original_lineage.config_version

    coordinator.apply_regime(_regime_ctx("S"))

    expected_lineage = lineage_from_runtime(
        bot_id=original_lineage.bot_id,
        strategy_id=original_lineage.strategy_id,
        family_id=original_lineage.family_id,
        portfolio_id=original_lineage.portfolio_id,
        account_alias=original_lineage.account_alias,
        strategy_version=original_lineage.strategy_version,
        portfolio_rules_config=checker._cfg,
        effective_strategy_config={
            "bot_id": original_lineage.bot_id,
            "strategy_id": original_lineage.strategy_id,
            "family_id": original_lineage.family_id,
            "runtime_refresh": "portfolio_rules",
        },
    )
    expected_version = expected_lineage.risk_config_version
    assert ctx.lineage.risk_config_version == expected_version
    assert ctx.lineage.risk_config_version != original_version
    assert ctx.lineage.config_version == expected_lineage.config_version
    assert ctx.lineage.config_version != original_config_version
    assert ctx.indicator_logger._lineage.risk_config_version == expected_version
    assert ctx.indicator_logger._lineage.config_version == expected_lineage.config_version
    assert ctx.filter_logger._lineage.risk_config_version == expected_version
    assert ctx.filter_logger._lineage.config_version == expected_lineage.config_version
    assert ctx.orderbook_logger._lineage.risk_config_version == expected_version
    assert ctx.orderbook_logger._lineage.config_version == expected_lineage.config_version
    coordination_payload = _read_latest_payload(
        next((tmp_path / "coordination_events").glob("coordination_events_*.jsonl"))
    )
    assert coordination_payload["event_type"] == "coordinator_action"
    assert coordination_payload["risk_config_version"] == expected_version
    assert coordination_payload["config_version"] == expected_lineage.config_version
