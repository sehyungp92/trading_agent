"""Tests for portfolio manager (rule checker)."""

from datetime import date

import pytest

from crypto_trader.core.models import Side
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.portfolio.manager import PortfolioManager, PortfolioRuleResult
from crypto_trader.portfolio.state import OpenRisk, PortfolioState


def _make_config(**overrides):
    defaults = dict(
        strategies=(
            StrategyAllocation(strategy_id="momentum", priority=1, max_concurrent=2),
            StrategyAllocation(strategy_id="trend", priority=0),
            StrategyAllocation(strategy_id="breakout", priority=2),
        ),
        heat_cap_R=3.0,
        directional_cap_R=2.5,
        portfolio_daily_stop_R=2.0,
        max_total_positions=3,
        symbol_collision="cap",
        symbol_exposure_cap_R=2.0,
    )
    defaults.update(overrides)
    return PortfolioConfig(**defaults)


def _make_manager(config=None, equity=10000.0):
    cfg = config or _make_config()
    state = PortfolioState(equity=equity, peak_equity=equity)
    return PortfolioManager(cfg, state), state


class TestRule1StrategyEnabled:
    def test_unknown_strategy_denied(self):
        mgr, _ = _make_manager()
        result = mgr.check_entry("unknown", "BTC", Side.LONG, 1.0)
        assert not result.approved
        assert "not in portfolio config" in result.denial_reason

    def test_disabled_strategy_denied(self):
        cfg = _make_config(strategies=(
            StrategyAllocation(strategy_id="momentum", enabled=False),
        ))
        mgr, _ = _make_manager(cfg)
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 1.0)
        assert not result.approved
        assert "disabled" in result.denial_reason


class TestRule2MaxTotalPositions:
    def test_at_max_denied(self):
        mgr, state = _make_manager()
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        state.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 1.0))
        state.add_risk(OpenRisk("breakout", "SOL", Side.LONG, 1.0))
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert not result.approved
        assert "max_total_positions" in result.denial_reason

    def test_below_max_approved(self):
        mgr, state = _make_manager()
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        state.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 1.0))
        result = mgr.check_entry("breakout", "SOL", Side.LONG, 0.5)
        assert result.approved


class TestRule3PerStrategyMaxConcurrent:
    def test_at_max_concurrent_denied(self):
        cfg = _make_config(strategies=(
            StrategyAllocation(strategy_id="momentum", max_concurrent=1),
            StrategyAllocation(strategy_id="trend"),
        ))
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("momentum", "ETH", Side.LONG, 0.5)
        assert not result.approved
        assert "max_concurrent" in result.denial_reason

    def test_below_max_concurrent_approved(self):
        mgr, state = _make_manager()
        # momentum has max_concurrent=2
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("momentum", "ETH", Side.LONG, 0.5)
        assert result.approved


class TestRule4HeatCap:
    def test_exceeds_heat_cap_denied(self):
        mgr, state = _make_manager()
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 2.5))
        result = mgr.check_entry("trend", "ETH", Side.LONG, 0.6)
        assert not result.approved
        assert "heat_cap_R" in result.denial_reason

    def test_at_heat_cap_boundary_approved(self):
        mgr, state = _make_manager()
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.5))
        result = mgr.check_entry("trend", "ETH", Side.LONG, 1.0)
        assert result.approved


class TestRule5DirectionalCap:
    def test_exceeds_directional_cap_denied(self):
        mgr, state = _make_manager()
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 2.0))
        result = mgr.check_entry("trend", "ETH", Side.LONG, 0.6)
        assert not result.approved
        assert "directional_cap_R" in result.denial_reason

    def test_opposite_direction_not_affected(self):
        mgr, state = _make_manager()
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 2.0))
        result = mgr.check_entry("trend", "ETH", Side.SHORT, 1.0)
        assert result.approved

    def test_priority_headroom_blocks_low_priority(self):
        cfg = _make_config(
            directional_cap_R=2.5,
            priority_headroom_R=1.0,
            priority_reserve_threshold=1,  # priority >= 1 gets blocked
        )
        mgr, state = _make_manager(cfg)
        # Trend (priority=0) enters first
        state.add_risk(OpenRisk("trend", "BTC", Side.LONG, 2.0))
        # Momentum (priority=1) tries — remaining=0.5 <= headroom=1.0
        result = mgr.check_entry("momentum", "ETH", Side.LONG, 0.5)
        assert not result.approved
        assert "headroom" in result.denial_reason

    def test_default_priority_headroom_does_not_block_priority_zero(self):
        cfg = _make_config(
            directional_cap_R=2.5,
            priority_headroom_R=1.0,
        )
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 2.0))

        result = mgr.check_entry("trend", "ETH", Side.LONG, 0.5)

        assert result.approved


class TestRule6SymbolCollision:
    def test_block_mode_denies_same_symbol(self):
        cfg = _make_config(symbol_collision="block")
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("trend", "BTC", Side.LONG, 0.5)
        assert not result.approved
        assert "symbol_collision=block" in result.denial_reason

    def test_block_mode_allows_different_symbol(self):
        cfg = _make_config(symbol_collision="block")
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("trend", "ETH", Side.LONG, 0.5)
        assert result.approved

    def test_allow_mode_permits_same_symbol(self):
        cfg = _make_config(symbol_collision="allow")
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("trend", "BTC", Side.LONG, 0.5)
        assert result.approved

    def test_cap_mode_within_cap(self):
        cfg = _make_config(symbol_exposure_cap_R=2.0)
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("trend", "BTC", Side.LONG, 0.8)
        assert result.approved

    def test_cap_mode_exceeds_cap(self):
        cfg = _make_config(symbol_exposure_cap_R=1.5)
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("trend", "BTC", Side.LONG, 0.8)
        assert not result.approved
        assert "symbol_exposure_cap_R" in result.denial_reason

    def test_cap_mode_same_strategy_allowed(self):
        """Same strategy stacking doesn't trigger symbol collision."""
        cfg = _make_config(symbol_exposure_cap_R=1.5)
        mgr, state = _make_manager(cfg)
        state.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.4)
        # Same strategy — collision check only looks at OTHER strategies
        assert result.approved


class TestRule7PortfolioDailyStop:
    def test_daily_stop_hit_denied(self):
        mgr, state = _make_manager()
        state.portfolio_daily_pnl_R = -2.0
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert not result.approved
        assert "portfolio_daily_stop_R" in result.denial_reason

    def test_below_daily_stop_approved(self):
        mgr, state = _make_manager()
        state.portfolio_daily_pnl_R = -1.5
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved


class TestRule8StrategyDailyStop:
    def test_strategy_daily_stop_hit_denied(self):
        mgr, state = _make_manager()
        state.daily_pnl_R["momentum"] = -3.0
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert not result.approved
        assert "daily_stop_R" in result.denial_reason

    def test_other_strategy_daily_stop_doesnt_affect(self):
        mgr, state = _make_manager()
        state.daily_pnl_R["trend"] = -3.0
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved


class TestRule9DrawdownTiers:
    def test_no_drawdown_full_size(self):
        mgr, _ = _make_manager()
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved
        assert result.size_multiplier == 1.0

    def test_tier1_reduces_size(self):
        mgr, state = _make_manager()
        state.peak_equity = 10000.0
        state.equity = 8500.0  # 15% DD → tier (0.15, 0.25)
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved
        assert result.size_multiplier == 0.25

    def test_deep_drawdown_blocks(self):
        mgr, state = _make_manager()
        state.peak_equity = 10000.0
        state.equity = 0.0  # 100% DD → tier (1.00, 0.00)
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert not result.approved
        assert "drawdown tier" in result.denial_reason

    def test_moderate_drawdown(self):
        mgr, state = _make_manager()
        state.peak_equity = 10000.0
        state.equity = 8900.0  # 11% DD → tier (0.08, 1.00) still applies
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved
        assert result.size_multiplier == 1.0


class TestRegisterEntryExit:
    def test_register_entry(self):
        mgr, state = _make_manager()
        mgr.register_entry("momentum", "BTC", Side.LONG, 1.0)
        assert state.total_heat_R() == 1.0
        assert state.strategy_position_count("momentum") == 1

    def test_register_exit(self):
        mgr, state = _make_manager()
        mgr.register_entry("momentum", "BTC", Side.LONG, 1.0)
        mgr.register_exit("momentum", "BTC", -0.5)
        assert state.total_heat_R() == 0.0
        assert state.strategy_daily_pnl_R("momentum") == -0.5
        assert state.portfolio_daily_pnl_R == -0.5

    def test_register_exit_no_matching_risk(self):
        mgr, state = _make_manager()
        mgr.register_exit("unknown", "BTC", 0.5)
        # Should not crash, just log warning
        assert state.total_heat_R() == 0.0


class TestCombinedRules:
    def test_all_rules_pass(self):
        mgr, _ = _make_manager()
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved
        assert result.size_multiplier == 1.0

    def test_cascading_multiplier_with_dd(self):
        mgr, state = _make_manager()
        state.peak_equity = 10000.0
        state.equity = 8700.0  # 13% DD → tier (0.12, 0.50)
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved
        assert result.size_multiplier == 0.50

    def test_daily_reset(self):
        mgr, state = _make_manager()
        state.daily_pnl_R["momentum"] = -3.0
        state.portfolio_daily_pnl_R = -3.0
        mgr.maybe_reset_daily(date(2026, 4, 21))
        result = mgr.check_entry("momentum", "BTC", Side.LONG, 0.5)
        assert result.approved
