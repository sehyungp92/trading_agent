"""Tests for portfolio configuration."""

import pytest

from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation


class TestStrategyAllocation:
    def test_defaults(self):
        a = StrategyAllocation(strategy_id="momentum")
        assert a.strategy_id == "momentum"
        assert a.enabled is True
        assert a.base_risk_pct == 0.01
        assert a.max_concurrent == 3
        assert a.daily_stop_R == 3.0
        assert a.priority == 0

    def test_frozen(self):
        a = StrategyAllocation(strategy_id="trend")
        with pytest.raises(AttributeError):
            a.enabled = False  # type: ignore[misc]


class TestPortfolioConfig:
    def _make_config(self, **kwargs):
        strats = kwargs.pop("strategies", (
            StrategyAllocation(strategy_id="momentum", priority=1),
            StrategyAllocation(strategy_id="trend", priority=0),
            StrategyAllocation(strategy_id="breakout", priority=2),
        ))
        return PortfolioConfig(strategies=strats, **kwargs)

    def test_defaults(self):
        cfg = PortfolioConfig()
        assert cfg.heat_cap_R == 6.0
        assert cfg.directional_cap_R == 4.0
        assert cfg.portfolio_daily_stop_R == 5.0
        assert cfg.max_total_positions == 9
        assert cfg.symbol_collision == "cap"
        assert cfg.symbol_exposure_cap_R == 3.0
        assert cfg.terminal_accounting_mode == "terminal_mark"
        assert len(cfg.dd_tiers) == 4

    def test_get_strategy_found(self):
        cfg = self._make_config()
        alloc = cfg.get_strategy("momentum")
        assert alloc is not None
        assert alloc.strategy_id == "momentum"

    def test_get_strategy_not_found(self):
        cfg = self._make_config()
        assert cfg.get_strategy("unknown") is None

    def test_priority_order(self):
        cfg = self._make_config()
        order = cfg.priority_order()
        assert [s.strategy_id for s in order] == ["trend", "momentum", "breakout"]

    def test_frozen(self):
        cfg = PortfolioConfig()
        with pytest.raises(AttributeError):
            cfg.heat_cap_R = 5.0  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        cfg = self._make_config(
            heat_cap_R=4.0,
            symbol_collision="block",
            terminal_accounting_mode="force_close",
        )
        d = cfg.to_dict()
        assert d["heat_cap_R"] == 4.0
        assert d["symbol_collision"] == "block"
        assert d["terminal_accounting_mode"] == "force_close"
        assert len(d["strategies"]) == 3

        cfg2 = PortfolioConfig.from_dict(d)
        assert cfg2.heat_cap_R == 4.0
        assert cfg2.symbol_collision == "block"
        assert cfg2.terminal_accounting_mode == "force_close"
        assert len(cfg2.strategies) == 3
        assert cfg2.strategies[0].strategy_id == "momentum"

    def test_from_dict_defaults(self):
        cfg = PortfolioConfig.from_dict({})
        assert cfg.heat_cap_R == 6.0
        assert cfg.strategies == ()

    def test_dd_tiers_serialization(self):
        cfg = self._make_config()
        d = cfg.to_dict()
        # dd_tiers should be list of lists
        assert isinstance(d["dd_tiers"][0], list)
        cfg2 = PortfolioConfig.from_dict(d)
        assert isinstance(cfg2.dd_tiers[0], tuple)
        assert cfg2.dd_tiers[0] == (0.08, 1.00)
