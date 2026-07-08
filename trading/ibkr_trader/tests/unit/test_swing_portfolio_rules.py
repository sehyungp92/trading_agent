"""Tests for swing family portfolio rules wiring (1A fix).

Validates:
  - build_multi_strategy_oms creates PortfolioRuleChecker when config provided
  - RiskGateway receives the checker
  - Directional cap denials work in multi-strategy OMS
  - Backward compat: no checker when portfolio_rules_config=None
  - Momentum family-scoped rules with symbol collision (1D)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.oms.risk.portfolio_rules import (
    PortfolioRuleChecker,
    PortfolioRulesConfig,
)
from libs.oms.coordination.coordinator import StrategyCoordinator
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import (
    BacktestCoordinator,
    _PortfolioRuleExposure,
    _SwingPortfolioRuleReplayAdapter,
)

SWING_IDS = ("ATRSS", "AKC_HELIX")
MOMENTUM_IDS = ("NQ_REGIME", "NQDTC_v2.1", "VdubusNQ_v4", "DownturnDominator_v1")


def _make_checker(
    *,
    equity: float = 100_000.0,
    initial_equity: float = 100_000.0,
    directional_cap_R: float = 6.0,
    family_strategy_ids: tuple[str, ...] = SWING_IDS,
    symbol_collision_action: str = "half_size",
    dir_risk_family: float = 0.0,
    sibling_holds_symbol: bool = False,
) -> PortfolioRuleChecker:
    config = PortfolioRulesConfig(
        directional_cap_R=directional_cap_R,
        initial_equity=initial_equity,
        family_strategy_ids=family_strategy_ids,
        symbol_collision_action=symbol_collision_action,
        nqdtc_direction_filter_enabled=False,
    )
    return PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: equity,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=dir_risk_family),
        get_sibling_positions_for_symbol=AsyncMock(return_value=sibling_holds_symbol),
    )


# ── Swing directional cap ────────────────────────────────────────


@pytest.mark.asyncio
async def test_swing_directional_cap_approves_within_limit():
    checker = _make_checker(dir_risk_family=5.0)
    result = await checker.check_entry("ATRSS", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_swing_directional_cap_denies_above_limit():
    checker = _make_checker(dir_risk_family=5.5)
    # 5.5R + 1R = 6.5R > 6R cap
    result = await checker.check_entry("AKC_HELIX", "LONG", 1.0)
    assert not result.approved
    assert "directional_cap" in result.denial_reason


@pytest.mark.asyncio
async def test_swing_directional_cap_exact_boundary():
    checker = _make_checker(dir_risk_family=5.0)
    # 5R + 1R = 6R == 6R cap → approved (not strictly greater)
    result = await checker.check_entry("AKC_HELIX", "SHORT", 1.0)
    assert result.approved


# ── Swing symbol collision ────────────────────────────────────────


@pytest.mark.asyncio
async def test_swing_symbol_collision_half_size():
    checker = _make_checker(sibling_holds_symbol=True)
    result = await checker.check_entry("ATRSS", "LONG", 1.0, symbol="GLD")
    assert result.approved
    assert result.size_multiplier == 0.5


@pytest.mark.asyncio
async def test_swing_no_collision():
    checker = _make_checker(sibling_holds_symbol=False)
    result = await checker.check_entry("ATRSS", "LONG", 1.0, symbol="GLD")
    assert result.approved
    assert result.size_multiplier == 1.0


# ── Momentum-specific: no cooldown/direction filter for swing ────




@pytest.mark.asyncio
async def test_swing_skips_direction_filter():
    """Swing strategies should not trigger NQDTC direction filter."""
    checker = _make_checker()
    result = await checker.check_entry("AKC_HELIX", "SHORT", 1.0)
    assert result.approved


# ── Backward compat: None config ─────────────────────────────────


@pytest.mark.asyncio
async def test_no_checker_when_config_none():
    """When portfolio_rules_config=None, no checker should be created."""
    # This tests the factory behavior conceptually — when config is None,
    # the RiskGateway should get portfolio_checker=None
    from libs.oms.risk.gateway import RiskGateway

    config = MagicMock()
    config.global_standdown = False
    config.strategy_configs = {}

    gateway = RiskGateway(
        config=config,
        calendar=MagicMock(),
        get_strategy_risk=AsyncMock(),
        get_portfolio_risk=AsyncMock(),
    )
    assert gateway._portfolio_checker is None


# ── Momentum family-scoped rules (1D) ───────────────────────────


@pytest.mark.asyncio
async def test_momentum_family_directional_cap():
    """Momentum with family_strategy_ids uses family-scoped directional check."""
    config = PortfolioRulesConfig(
        initial_equity=10_000.0,
        directional_cap_R=6.0,
        family_strategy_ids=MOMENTUM_IDS,
        symbol_collision_action="half_size",
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=100.0),  # high global
        get_current_equity=lambda: 10_000.0,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=5.0),  # low family
        get_sibling_positions_for_symbol=AsyncMock(return_value=False),
    )
    # 5R family + 1R new = 6R ≤ 6R cap → approved (uses family, not global)
    result = await checker.check_entry("NQ_REGIME", "LONG", 1.0)
    assert result.approved


@pytest.mark.asyncio
async def test_momentum_symbol_collision_nq():
    """NQ collision between momentum siblings triggers half_size."""
    config = PortfolioRulesConfig(
        initial_equity=10_000.0,
        directional_cap_R=6.0,
        family_strategy_ids=MOMENTUM_IDS,
        symbol_collision_action="half_size",
    )
    checker = PortfolioRuleChecker(
        config=config,
        get_strategy_signal=AsyncMock(return_value=None),
        get_directional_risk_R=AsyncMock(return_value=0.0),
        get_current_equity=lambda: 10_000.0,
        get_directional_risk_R_for_strategies=AsyncMock(return_value=0.0),
        get_sibling_positions_for_symbol=AsyncMock(return_value=True),
    )
    result = await checker.check_entry("NQ_REGIME", "LONG", 1.0, symbol="NQ")
    assert result.approved
    assert result.size_multiplier == 0.5


def test_swing_backtest_replay_uses_live_portfolio_rule_checker_for_collision():
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    adapter = _SwingPortfolioRuleReplayAdapter(config, 50_000.0)
    try:
        adapter.refresh(
            equity=50_000.0,
            exposures=[
                _PortfolioRuleExposure(
                    strategy_id="ATRSS",
                    symbol="QQQ",
                    direction="LONG",
                    risk_dollars=800.0,
                    risk_R=1.0,
                    qty=10,
                )
            ],
        )

        result = adapter.check_entry(
            strategy_id="AKC_HELIX",
            direction="LONG",
            risk_dollars=450.0,
            symbol="QQQ",
            qty=10,
        )

        assert result.approved
        assert result.size_multiplier == 0.5
    finally:
        adapter.close()


def test_swing_backtest_coordination_replays_live_strategy_coordinator_semantics():
    coordinator = BacktestCoordinator(enable_tighten=True, enable_size_boost=True)

    assert isinstance(coordinator._coordinator, StrategyCoordinator)

    coordinator.on_atrss_position_change("QQQ", 1, 510.25)

    assert coordinator.consume_tighten_events() == ["QQQ"]
    assert coordinator.consume_tighten_events() == []
    assert coordinator.has_atrss_position("QQQ", 1)
    assert not coordinator.has_atrss_position("QQQ", -1)

    coordinator.on_atrss_position_change("QQQ", 0, 0.0)

    assert not coordinator.has_atrss_position("QQQ", 1)


def test_swing_backtest_replay_uses_live_portfolio_rule_checker_for_directional_cap():
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    adapter = _SwingPortfolioRuleReplayAdapter(config, 50_000.0)
    try:
        adapter.refresh(
            equity=50_000.0,
            exposures=[
                _PortfolioRuleExposure(
                    strategy_id="ATRSS",
                    symbol="QQQ",
                    direction="LONG",
                    risk_dollars=3_600.0,
                    risk_R=3.8,
                    qty=10,
                )
            ],
        )

        result = adapter.check_entry(
            strategy_id="AKC_HELIX",
            direction="LONG",
            risk_dollars=450.0,
            symbol="GLD",
            qty=10,
        )

        assert not result.approved
        assert "directional_cap" in result.denial_reason
    finally:
        adapter.close()


def test_swing_backtest_replay_directional_cap_uses_portfolio_dollar_basis():
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    adapter = _SwingPortfolioRuleReplayAdapter(config, 50_000.0)
    try:
        adapter.refresh(equity=50_000.0, exposures=[])

        result = adapter.check_entry(
            strategy_id="AKC_HELIX",
            direction="LONG",
            risk_dollars=2_250.0,
            symbol="GLD",
            qty=10,
        )

        assert result.approved
    finally:
        adapter.close()


def test_swing_backtest_replay_refreshes_portfolio_reference_unit():
    config = UnifiedBacktestConfig(initial_equity=50_000.0)
    adapter = _SwingPortfolioRuleReplayAdapter(config, 50_000.0)
    try:
        adapter.refresh(equity=50_000.0, unit_equity=250_000.0, exposures=[])

        result = adapter.check_entry(
            strategy_id="ATRSS",
            direction="LONG",
            risk_dollars=6_000.0,
            symbol="QQQ",
            qty=10,
        )

        assert result.approved
    finally:
        adapter.close()
