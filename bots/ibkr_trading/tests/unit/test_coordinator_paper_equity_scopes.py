from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class _FakeOMS:
    _portfolio_checker = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _FakeEngine:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def subscription_instruments(self):
        return []

    def polling_instruments(self):
        return []

    async def on_quote(self, *args, **kwargs) -> None:
        pass

    async def on_bar(self, *args, **kwargs) -> None:
        pass


class _FakeInstrumentation:
    sidecar = object()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _FakeMarketData:
    def __init__(self, **kwargs) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def ensure_hot_symbols(self, symbols) -> None:
        pass

    async def poll_due_bars(self, instruments, now=None) -> None:
        pass


class _IB:
    async def qualifyContractsAsync(self, contract):
        return [contract]


class _Session:
    def __init__(self) -> None:
        self.ib = _IB()

    async def req_historical_data(self, *args, **kwargs):
        return [SimpleNamespace(close=21000.0)]

    def add_reconnect_callback(self, callback) -> None:
        pass


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        registry=SimpleNamespace(enabled_strategies=lambda live=False: [], strategies={}),
        session=_Session(),
        db_pool=object(),
        account_gate=None,
        portfolio=SimpleNamespace(
            capital=SimpleNamespace(
                paper_initial_equity=100_000.0,
                family_allocations={
                    "swing": 1.0 / 3.0,
                    "momentum": 1.0 / 3.0,
                    "stock": 1.0 / 3.0,
                },
            )
        ),
        trade_recorder=None,
    )


@pytest.mark.asyncio
async def test_momentum_uses_family_paper_equity_scope_and_portfolio_nav_sizing(monkeypatch):
    from strategies.momentum.coordinator import MomentumFamilyCoordinator

    scopes: list[tuple[str, float]] = []
    navs = {"momentum": 12_000.0}

    class FakePaperEquityManager:
        def __init__(self, pool, account_scope: str, initial_equity: float) -> None:
            scopes.append((account_scope, initial_equity))
            self.scope = account_scope

        async def load(self) -> float:
            return navs[self.scope]

    build_oms = AsyncMock(return_value=_FakeOMS())
    monkeypatch.setattr("strategies.momentum.coordinator.get_environment", lambda: "paper")
    monkeypatch.setattr(
        "libs.broker_ibkr.config.loader.IBKRConfig",
        lambda config_dir: SimpleNamespace(
            profile=SimpleNamespace(account_id="DU123"),
            contracts={},
            routes={},
        ),
    )
    monkeypatch.setattr(
        "libs.broker_ibkr.mapping.contract_factory.ContractFactory",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "libs.broker_ibkr.adapters.execution_adapter.IBKRExecutionAdapter",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr("libs.persistence.paper_equity.PaperEquityManager", FakePaperEquityManager)
    monkeypatch.setattr(
        "libs.config.capital_bootstrap.bootstrap_capital",
        lambda equity, config_dir: {
            "NQ_REGIME": SimpleNamespace(allocated_nav=55_000.0, capital_pct=55.0),
            "VdubusNQ_v4": SimpleNamespace(allocated_nav=45_000.0, capital_pct=45.0),
        },
    )
    monkeypatch.setattr("libs.oms.services.factory.build_oms_service", build_oms)
    monkeypatch.setattr(
        "strategies.momentum.instrumentation.src.bootstrap.InstrumentationManager",
        lambda *args, **kwargs: _FakeInstrumentation(),
    )

    ctx = _ctx()
    ctx.portfolio.capital.paper_initial_equity = 30_000.0
    coordinator = MomentumFamilyCoordinator(ctx)
    monkeypatch.setattr(
        coordinator,
        "_build_strategy_descriptors",
        lambda: [
            {
                "strategy_id": "NQ_REGIME",
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "engine_cls": _FakeEngine,
                "build_instruments": lambda: {},
                "instr_type": "nq_regime",
            },
            {
                "strategy_id": "VdubusNQ_v4",
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "engine_cls": _FakeEngine,
                "build_instruments": lambda: {},
                "instr_type": "vdubus",
            },
        ],
    )

    await coordinator.start()
    await coordinator.stop()

    assert scopes == [("momentum", pytest.approx(10_000.0))]
    kwargs_by_sid = {
        call.kwargs["strategy_id"]: call.kwargs for call in build_oms.await_args_list
    }
    for sid in ("NQ_REGIME", "VdubusNQ_v4"):
        assert kwargs_by_sid[sid]["paper_equity_scope"] == "momentum"
        assert kwargs_by_sid[sid]["paper_initial_equity"] == pytest.approx(10_000.0)
        assert kwargs_by_sid[sid]["paper_equity_ref"][0] == pytest.approx(12_000.0)
        assert kwargs_by_sid[sid]["get_current_equity"]() == pytest.approx(12_000.0)
        assert kwargs_by_sid[sid]["live_equity"] is None
        assert kwargs_by_sid[sid]["unit_risk_dollars"] == pytest.approx(120.0)
        assert kwargs_by_sid[sid]["daily_stop_R"] == pytest.approx(1.0)
        assert kwargs_by_sid[sid]["portfolio_unit_risk_dollars"] == pytest.approx(60.0)
        assert kwargs_by_sid[sid]["heat_cap_R"] == pytest.approx(10.0)
        assert kwargs_by_sid[sid]["portfolio_daily_stop_R"] == pytest.approx(2.75)
        assert kwargs_by_sid[sid]["portfolio_weekly_stop_R"] == pytest.approx(9.0)
        rules = kwargs_by_sid[sid]["portfolio_rules_config"]
        assert rules.initial_equity == pytest.approx(10_000.0)
        assert rules.reference_unit_risk_dollars == pytest.approx(60.0)
        assert rules.portfolio_heat_cap_R == pytest.approx(10.0)
        assert rules.max_total_active_positions == 8
        assert dict(rules.max_strategy_active_positions) == {
            "NQ_REGIME": 3,
            "VdubusNQ_v4": 2,
        }
        assert rules.existing_position_mult == pytest.approx(0.85)
        assert rules.fit_to_remaining_heat is True
        assert rules.fit_to_remaining_directional_cap is True
        assert rules.fit_to_remaining_family_cap is True
        strategy_mults = dict(rules.strategy_size_multipliers)
        assert strategy_mults["NQ_REGIME"] == pytest.approx(0.75)
        assert strategy_mults["VdubusNQ_v4"] == pytest.approx(0.95)
        targets = kwargs_by_sid[sid]["allocation_targets"]
        assert targets["families"] == {"momentum": 1.0}
        assert targets["strategies"] == {
            "NQ_REGIME": pytest.approx(0.5),
            "VdubusNQ_v4": pytest.approx(0.5),
        }


@pytest.mark.asyncio
async def test_stock_uses_family_paper_equity_scope_and_portfolio_nav_sizing(monkeypatch):
    from strategies.stock.coordinator import StockFamilyCoordinator

    scopes: list[tuple[str, float]] = []
    navs = {"stock": 12_000.0}

    class FakePaperEquityManager:
        def __init__(self, pool, account_scope: str, initial_equity: float) -> None:
            scopes.append((account_scope, initial_equity))
            self.scope = account_scope

        async def load(self) -> float:
            return navs[self.scope]

    build_oms = AsyncMock(return_value=_FakeOMS())
    monkeypatch.setattr("strategies.stock.coordinator.get_environment", lambda: "paper")
    monkeypatch.setattr(
        "strategies.stock.coordinator.validate_stock_readiness",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr("libs.persistence.paper_equity.PaperEquityManager", FakePaperEquityManager)
    monkeypatch.setattr(
        "libs.config.capital_bootstrap.bootstrap_capital",
        lambda equity, config_dir: {
            "IARIC_v1": SimpleNamespace(allocated_nav=50_000.0, capital_pct=50.0),
            "ALCB_v1": SimpleNamespace(allocated_nav=50_000.0, capital_pct=50.0),
        },
    )
    monkeypatch.setattr("libs.oms.services.factory.build_oms_service", build_oms)
    monkeypatch.setattr(
        "strategies.stock.instrumentation.src.bootstrap.InstrumentationManager",
        lambda *args, **kwargs: _FakeInstrumentation(),
    )

    ctx = _ctx()
    ctx.portfolio.capital.paper_initial_equity = 30_000.0
    coordinator = StockFamilyCoordinator(ctx)
    coordinator._contract_factory = object()
    monkeypatch.setattr(coordinator, "_enabled_stock_strategy_ids", lambda: ("IARIC_v1", "ALCB_v1"))
    monkeypatch.setattr(
        coordinator,
        "_build_strategy_descriptors",
        lambda artifacts, ids: [
            {
                "strategy_id": "IARIC_v1",
                "data_key": "artifact",
                "data_value": object(),
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "heat_cap_R": 1.0,
                "max_concurrent": 9,
                "portfolio_daily_stop_R": 3.0,
                "adapter": lambda session: object(),
                "account_id": "DU123",
                "settings": object(),
                "trade_recorder": object(),
                "diagnostics_factory": lambda: object(),
                "engine_cls": lambda: _FakeEngine,
                "market_data_cls": lambda: _FakeMarketData,
                "instr_type": "iaric",
            },
            {
                "strategy_id": "ALCB_v1",
                "data_key": "artifact",
                "data_value": object(),
                "base_risk_pct": 0.01,
                "daily_stop_R": 2.0,
                "heat_cap_R": 1.0,
                "max_concurrent": 6,
                "portfolio_daily_stop_R": 3.0,
                "adapter": lambda session: object(),
                "account_id": "DU123",
                "settings": object(),
                "trade_recorder": object(),
                "diagnostics_factory": lambda: object(),
                "engine_cls": lambda: _FakeEngine,
                "market_data_cls": lambda: _FakeMarketData,
                "instr_type": "alcb",
            },
        ],
    )

    await coordinator.start()
    await coordinator.stop()

    assert scopes == [("stock", pytest.approx(10_000.0))]
    kwargs_by_sid = {
        call.kwargs["strategy_id"]: call.kwargs for call in build_oms.await_args_list
    }
    for sid in ("IARIC_v1", "ALCB_v1"):
        assert kwargs_by_sid[sid]["paper_equity_scope"] == "stock"
        assert kwargs_by_sid[sid]["paper_initial_equity"] == pytest.approx(10_000.0)
        assert kwargs_by_sid[sid]["paper_equity_ref"][0] == pytest.approx(12_000.0)
        assert kwargs_by_sid[sid]["get_current_equity"]() == pytest.approx(12_000.0)
        assert kwargs_by_sid[sid]["live_equity"] is None
        assert kwargs_by_sid[sid]["unit_risk_dollars"] == pytest.approx(120.0)
        assert kwargs_by_sid[sid]["portfolio_unit_risk_dollars"] == pytest.approx(
            12_000.0 * 0.00648
        )
        assert kwargs_by_sid[sid]["strategy_heat_cap_R"] == pytest.approx(1.0)
        assert kwargs_by_sid[sid]["heat_cap_R"] == pytest.approx(6.5)
        assert kwargs_by_sid[sid]["portfolio_rules_config"].initial_equity == pytest.approx(10_000.0)
        assert kwargs_by_sid[sid]["portfolio_rules_config"].reference_unit_risk_dollars == pytest.approx(
            12_000.0 * 0.00648
        )
        assert kwargs_by_sid[sid]["portfolio_rules_config"].reference_unit_risk_pct == pytest.approx(0.00648)
        assert kwargs_by_sid[sid]["portfolio_rules_config"].max_total_active_positions == 12
        assert kwargs_by_sid[sid]["portfolio_rules_config"].max_symbol_heat_R == pytest.approx(2.2)
        assert kwargs_by_sid[sid]["portfolio_rules_config"].same_sector_heat_cap_R == pytest.approx(3.8)
        assert kwargs_by_sid[sid]["portfolio_rules_config"].max_single_strategy_trade_share == pytest.approx(0.85)
        assert kwargs_by_sid[sid]["portfolio_rules_config"].portfolio_heat_cap_R == pytest.approx(6.5)
        assert dict(kwargs_by_sid[sid]["portfolio_rules_config"].max_strategy_active_positions) == {
            "IARIC_v1": 9,
            "ALCB_v1": 6,
        }
        assert dict(kwargs_by_sid[sid]["portfolio_rules_config"].max_strategy_heat_R) == {
            "IARIC_v1": pytest.approx(1.0),
            "ALCB_v1": pytest.approx(1.0),
        }
        assert kwargs_by_sid[sid]["portfolio_rules_config"].dynamic_allocation_enabled is True
        assert kwargs_by_sid[sid]["portfolio_rules_config"].dynamic_lookback_trades == 60
        assert kwargs_by_sid[sid]["portfolio_rules_config"].dynamic_min_mult == pytest.approx(0.65)
        assert kwargs_by_sid[sid]["portfolio_rules_config"].dynamic_max_mult == pytest.approx(1.22)
        targets = kwargs_by_sid[sid]["allocation_targets"]
        assert targets["families"] == {"stock": 1.0}
        assert targets["strategies"] == {
            "IARIC_v1": pytest.approx(0.5),
            "ALCB_v1": pytest.approx(0.5),
        }


@pytest.mark.asyncio
async def test_swing_uses_family_paper_equity_scope_and_portfolio_nav_sizing(monkeypatch):
    from strategies.swing.coordinator import SwingFamilyCoordinator

    ctx = _ctx()
    ctx.portfolio.capital.paper_initial_equity = 30_000.0
    ctx.registry.strategies = {}
    scopes: list[tuple[str, float]] = []
    engine_kwargs: dict[str, dict] = {}

    class FakePaperEquityManager:
        def __init__(self, pool, account_scope: str, initial_equity: float) -> None:
            scopes.append((account_scope, initial_equity))

        async def load(self) -> float:
            return 12_000.0

    def fake_engine(strategy_id: str):
        def _factory(**kwargs):
            engine_kwargs[strategy_id] = kwargs
            return _FakeEngine(**kwargs)

        return _factory

    fake_coordinator = SimpleNamespace(set_action_logger=lambda callback: None)
    build_oms = AsyncMock(return_value=(_FakeOMS(), fake_coordinator))
    monkeypatch.setattr("libs.oms.persistence.db_config.get_environment", lambda: "paper")
    monkeypatch.setattr(
        "libs.broker_ibkr.config.loader.IBKRConfig",
        lambda config_dir: SimpleNamespace(
            profile=SimpleNamespace(account_id="DU123"),
            contracts={},
            routes={},
        ),
    )
    monkeypatch.setattr(
        "libs.broker_ibkr.mapping.contract_factory.ContractFactory",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "libs.broker_ibkr.adapters.execution_adapter.IBKRExecutionAdapter",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr("libs.persistence.paper_equity.PaperEquityManager", FakePaperEquityManager)
    monkeypatch.setattr(
        "libs.config.capital_bootstrap.bootstrap_capital",
        lambda equity, config_dir: {
            "ATRSS": SimpleNamespace(family="swing", allocated_nav=3_000.0),
            "AKC_HELIX": SimpleNamespace(family="swing", allocated_nav=4_000.0),
            "TPC": SimpleNamespace(family="swing", allocated_nav=3_000.0),
            "IARIC_v1": SimpleNamespace(family="stock", allocated_nav=5_500.0),
            "ALCB_v1": SimpleNamespace(family="stock", allocated_nav=4_500.0),
            "NQ_REGIME": SimpleNamespace(family="momentum", allocated_nav=2_500.0),
        },
    )
    monkeypatch.setattr("libs.oms.services.factory.build_multi_strategy_oms", build_oms)
    monkeypatch.setattr(
        "strategies.swing.atrss.config.build_instruments",
        lambda: {},
    )
    monkeypatch.setattr(
        "strategies.swing.akc_helix.config.build_instruments",
        lambda: {},
    )
    monkeypatch.setattr(
        "strategies.swing.tpc.config.build_instruments",
        lambda: {},
    )
    monkeypatch.setattr("strategies.swing.atrss.engine.ATRSSEngine", fake_engine("ATRSS"))
    monkeypatch.setattr("strategies.swing.akc_helix.engine.HelixEngine", fake_engine("AKC_HELIX"))
    monkeypatch.setattr("strategies.swing.tpc.engine.TPCEngine", fake_engine("TPC"))
    monkeypatch.setattr(
        SwingFamilyCoordinator,
        "_bootstrap_instrumentation_kits",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        SwingFamilyCoordinator,
        "_create_overlay_engine",
        lambda *args, **kwargs: None,
    )

    coordinator = SwingFamilyCoordinator(ctx)
    await coordinator.start()
    await coordinator.stop()

    assert scopes == [("swing", pytest.approx(10_000.0))]
    kwargs = build_oms.await_args.kwargs
    assert kwargs["paper_equity_scope"] == "swing"
    assert kwargs["paper_initial_equity"] == pytest.approx(10_000.0)
    assert kwargs["get_current_equity"]() == pytest.approx(12_000.0)
    assert kwargs["portfolio_rules_config"].initial_equity == pytest.approx(10_000.0)
    assert kwargs["portfolio_rules_config"].reference_unit_risk_dollars == pytest.approx(198.0)
    assert kwargs["portfolio_rules_config"].directional_cap_R == pytest.approx(5.5)
    assert kwargs["portfolio_rules_config"].directional_cap_long_R == pytest.approx(4.0)
    assert kwargs["portfolio_rules_config"].directional_cap_short_R == pytest.approx(4.0)
    assert kwargs["portfolio_rules_config"].dd_tiers == (
        (0.04, 0.90),
        (0.07, 0.70),
        (0.10, 0.50),
        (0.14, 0.25),
        (0.18, 0.00),
    )
    assert kwargs["heat_cap_R"] == pytest.approx(5.5)
    assert kwargs["portfolio_daily_stop_R"] == pytest.approx(3.75)
    assert kwargs["portfolio_weekly_stop_R"] == pytest.approx(9.0)
    assert kwargs["allocation_targets"]["families"] == {"swing": 1.0}
    assert kwargs["allocation_targets"]["strategies"] == {
        "ATRSS": pytest.approx(1.0 / 3.0),
        "AKC_HELIX": pytest.approx(1.0 / 3.0),
        "TPC": pytest.approx(1.0 / 3.0),
    }
    strategies = {item["id"]: item for item in kwargs["strategies"]}
    assert strategies["ATRSS"]["unit_risk_dollars"] == pytest.approx(198.0)
    assert strategies["ATRSS"]["max_heat_R"] == pytest.approx(2.15)
    assert strategies["ATRSS"]["daily_stop_R"] == pytest.approx(2.25)
    assert strategies["AKC_HELIX"]["unit_risk_dollars"] == pytest.approx(156.0)
    assert strategies["AKC_HELIX"]["max_heat_R"] == pytest.approx(2.10)
    assert strategies["AKC_HELIX"]["daily_stop_R"] == pytest.approx(2.5)
    assert strategies["TPC"]["unit_risk_dollars"] == pytest.approx(60.0)
    assert strategies["TPC"]["max_heat_R"] == pytest.approx(4.0)
    assert strategies["TPC"]["daily_stop_R"] == pytest.approx(2.0)
    for sid in ("ATRSS", "AKC_HELIX", "TPC"):
        assert engine_kwargs[sid]["equity"] == pytest.approx(12_000.0)
        assert engine_kwargs[sid]["equity_alloc_pct"] == pytest.approx(1.0)
