from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from libs.oms.config.risk_config import RiskConfig, StrategyRiskConfig
from libs.oms.models.instrument import Instrument
from libs.oms.models.order import OMSOrder, OrderSide, OrderType, RiskContext
from libs.oms.models.risk_state import PortfolioRiskState, StrategyRiskState
from libs.oms.risk.calendar import EventCalendar
from libs.oms.risk.gateway import RiskGateway
from libs.oms.risk.swing_portfolio_adapter import (
    SwingLivePortfolioRiskAdapter,
    SwingPortfolioHeatAdapter,
)


def test_replay_adapter_enforces_shared_swing_heat_rules() -> None:
    slots = [
        SimpleNamespace(
            strategy_id="ATRSS",
            priority=0,
            unit_risk_pct=0.01,
            max_heat_R=1.5,
            daily_stop_R=2.0,
        ),
        SimpleNamespace(
            strategy_id="AKC_HELIX",
            priority=4,
            unit_risk_pct=0.005,
            max_heat_R=1.2,
            daily_stop_R=2.5,
        ),
    ]
    adapter = SwingPortfolioHeatAdapter(
        heat_cap_R=3.0,
        portfolio_daily_stop_R=4.0,
        strategy_slots=slots,
        equity=10_000.0,
    )
    adapter.update_open_risk({"AKC_HELIX": 250.0})

    approved, reason = adapter.can_enter("ATRSS", 60.0)

    assert not approved
    assert reason == "Portfolio heat cap (2.50R + 0.60R > 3.0R)"
    assert adapter.entry_risk_context("ATRSS", 60.0)["portfolio_after_request_R"] == pytest.approx(3.1)


def test_replay_adapter_reserves_same_bar_accepted_risk() -> None:
    slots = [
        SimpleNamespace(
            strategy_id="ATRSS",
            priority=0,
            unit_risk_pct=0.01,
            max_heat_R=3.0,
            daily_stop_R=2.0,
        )
    ]
    adapter = SwingPortfolioHeatAdapter(
        heat_cap_R=1.5,
        portfolio_daily_stop_R=4.0,
        strategy_slots=slots,
        equity=10_000.0,
    )

    approved, reason = adapter.can_enter("ATRSS", 100.0)
    assert approved, reason

    adapter.reserve_entry("ATRSS", 100.0)
    approved, reason = adapter.can_enter("ATRSS", 60.0)

    assert not approved
    assert reason == "Portfolio heat cap (1.00R + 0.60R > 1.5R)"


def test_replay_adapter_can_disable_idle_priority_reservation() -> None:
    slots = [
        SimpleNamespace(
            strategy_id="ATRSS",
            priority=0,
            unit_risk_pct=0.01,
            max_heat_R=3.0,
            daily_stop_R=2.0,
        ),
        SimpleNamespace(
            strategy_id="AKC_HELIX",
            priority=1,
            unit_risk_pct=0.01,
            max_heat_R=3.0,
            daily_stop_R=2.5,
        ),
    ]
    reserved = SwingPortfolioHeatAdapter(
        heat_cap_R=3.0,
        portfolio_daily_stop_R=4.0,
        strategy_slots=slots,
        equity=10_000.0,
    )
    reserved.update_open_risk({"AKC_HELIX": 150.0})

    approved, reason = reserved.can_enter("AKC_HELIX", 100.0)
    assert not approved
    assert reason == "Heat reserved for ATRSS"

    unreserved = SwingPortfolioHeatAdapter(
        heat_cap_R=3.0,
        portfolio_daily_stop_R=4.0,
        strategy_slots=slots,
        equity=10_000.0,
        reserve_idle_higher_priority=False,
    )
    unreserved.update_open_risk({"AKC_HELIX": 150.0})

    approved, reason = unreserved.can_enter("AKC_HELIX", 100.0)
    assert approved, reason


def test_replay_adapter_rebases_heat_units_to_sizing_equity() -> None:
    slots = [
        SimpleNamespace(
            strategy_id="ATRSS",
            priority=0,
            unit_risk_pct=0.01,
            max_heat_R=3.0,
            daily_stop_R=2.0,
        )
    ]
    adapter = SwingPortfolioHeatAdapter(
        heat_cap_R=2.0,
        portfolio_daily_stop_R=4.0,
        strategy_slots=slots,
        equity=10_000.0,
    )
    adapter.update_unit_risk(20_000.0, slots)
    adapter.update_open_risk({"ATRSS": 200.0})

    approved, reason = adapter.can_enter("ATRSS", 200.0)

    assert approved, reason
    context = adapter.entry_risk_context("ATRSS", 200.0)
    assert context["portfolio_open_risk_R"] == pytest.approx(1.0)
    assert context["portfolio_after_request_R"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_live_gateway_routes_swing_heat_through_shared_adapter() -> None:
    today = date(2026, 5, 2)
    risk_config = RiskConfig(
        heat_cap_R=3.0,
        portfolio_daily_stop_R=4.0,
        strategy_configs={
            "ATRSS": StrategyRiskConfig(
                strategy_id="ATRSS",
                unit_risk_dollars=100.0,
                daily_stop_R=2.0,
                priority=0,
                max_heat_R=1.5,
            ),
            "AKC_HELIX": StrategyRiskConfig(
                strategy_id="AKC_HELIX",
                unit_risk_dollars=50.0,
                daily_stop_R=2.5,
                priority=4,
                max_heat_R=1.2,
            ),
        },
        portfolio_urd=100.0,
    )
    strategy_states = {
        "ATRSS": StrategyRiskState(
            strategy_id="ATRSS",
            trade_date=today,
            open_risk_dollars=250.0,
            open_risk_R=2.5,
        ),
        "AKC_HELIX": StrategyRiskState(strategy_id="AKC_HELIX", trade_date=today),
    }

    async def get_strategy_risk(strategy_id: str) -> StrategyRiskState:
        return strategy_states[strategy_id]

    async def get_portfolio_risk() -> PortfolioRiskState:
        return PortfolioRiskState(
            trade_date=today,
            open_risk_dollars=250.0,
            open_risk_R=2.5,
        )

    gateway = RiskGateway(
        config=risk_config,
        calendar=EventCalendar(),
        get_strategy_risk=get_strategy_risk,
        get_portfolio_risk=get_portfolio_risk,
        portfolio_risk_adapter=SwingLivePortfolioRiskAdapter(risk_config),
        family_id="swing",
    )
    order = OMSOrder(
        strategy_id="AKC_HELIX",
        instrument=Instrument(
            symbol="QQQ",
            root="QQQ",
            venue="ARCA",
            tick_size=0.01,
            tick_value=0.01,
            multiplier=10.0,
            sec_type="STK",
        ),
        side=OrderSide.BUY,
        qty=1,
        order_type=OrderType.STOP_LIMIT,
        risk_context=RiskContext(planned_entry_price=100.0, stop_for_risk=94.0),
    )

    denial = await gateway.check_entry(order)

    assert denial == "Portfolio heat cap (2.50R + 0.60R > 3.0R)"


@pytest.mark.asyncio
async def test_live_adapter_preserves_r_based_daily_stop_when_dollars_missing() -> None:
    today = date(2026, 5, 2)
    risk_config = RiskConfig(
        heat_cap_R=3.0,
        portfolio_daily_stop_R=4.0,
        strategy_configs={
            "ATRSS": StrategyRiskConfig(
                strategy_id="ATRSS",
                unit_risk_dollars=100.0,
                daily_stop_R=2.0,
                priority=0,
                max_heat_R=1.5,
            )
        },
        portfolio_urd=100.0,
    )
    strat_state = StrategyRiskState(
        strategy_id="ATRSS",
        trade_date=today,
        daily_realized_R=-2.1,
        daily_realized_pnl=0.0,
    )

    async def get_strategy_risk(_strategy_id: str) -> StrategyRiskState:
        return strat_state

    decision = await SwingLivePortfolioRiskAdapter(risk_config).check_entry(
        strategy_id="ATRSS",
        new_risk_dollars=10.0,
        strat_cfg=risk_config.strategy_configs["ATRSS"],
        strat_risk=strat_state,
        port_risk=PortfolioRiskState(trade_date=today),
        get_strategy_risk=get_strategy_risk,
    )

    assert not decision.approved
    assert decision.reason == "ATRSS daily stop hit"
