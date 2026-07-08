"""Tests for portfolio parameter sweep."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import SetupGrade, Side, Trade
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation
from crypto_trader.portfolio.sweep import (
    SweepVariant,
    build_combined_variant,
    find_winners,
    format_sweep_table,
    run_sweep,
)


def _make_trade(
    symbol="BTC",
    direction=Side.LONG,
    pnl=100.0,
    r_multiple=2.0,
    entry_hour=10,
    exit_hour=14,
    day=20,
):
    return Trade(
        trade_id=f"t_{symbol}_{day}_{entry_hour}",
        symbol=symbol,
        direction=direction,
        entry_price=50000.0,
        exit_price=50100.0,
        qty=0.01,
        entry_time=datetime(2026, 4, day, entry_hour, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 4, day, exit_hour, 0, tzinfo=timezone.utc),
        pnl=pnl,
        r_multiple=r_multiple,
        commission=1.0,
        bars_held=16,
        setup_grade=SetupGrade.B,
        exit_reason="protective_stop",
        confluences_used=["ema_zone"],
        confirmation_type="engulfing",
        entry_method="market",
        funding_paid=0.0,
        mae_r=-0.3,
        mfe_r=2.5,
    )


def _make_config():
    return PortfolioConfig(
        initial_equity=10000.0,
        strategies=(
            StrategyAllocation(strategy_id="momentum"),
            StrategyAllocation(strategy_id="trend"),
        ),
        heat_cap_R=3.0,
        max_total_positions=3,
    )


class TestRunSweep:
    def test_baseline_included(self):
        cfg = _make_config()
        trades = {
            "momentum": [_make_trade(pnl=100, r_multiple=2.0, day=20)],
            "trend": [_make_trade(pnl=50, r_multiple=1.0, day=21)],
        }
        results = run_sweep(cfg, trades, [])
        assert len(results) == 1
        assert results[0].variant_name == "__baseline__"
        assert results[0].n_approved > 0

    def test_variant_computes_deltas(self):
        cfg = _make_config()
        trades = {
            "momentum": [_make_trade(pnl=100, r_multiple=2.0, day=20)],
            "trend": [_make_trade(pnl=50, r_multiple=1.0, day=21)],
        }
        variant = SweepVariant(
            name="tight_heat",
            description="Tighter heat cap",
            config_factory=lambda c: replace(c, heat_cap_R=1.0),
        )
        results = run_sweep(cfg, trades, [variant])
        assert len(results) == 2
        tight = results[1]
        assert tight.variant_name == "tight_heat"
        assert "total_R" in tight.deltas

    def test_blocking_variant_reduces_approvals(self):
        cfg = _make_config()
        trades = {
            "momentum": [_make_trade(pnl=100, r_multiple=2.0, day=20)],
            "trend": [_make_trade(pnl=50, r_multiple=1.0, day=20, entry_hour=10)],
        }
        variant = SweepVariant(
            name="max_1_pos",
            description="Max 1 position",
            config_factory=lambda c: replace(c, max_total_positions=1),
        )
        results = run_sweep(cfg, trades, [variant])
        baseline = results[0]
        restricted = results[1]
        assert restricted.n_approved <= baseline.n_approved


class TestFindWinners:
    def test_finds_improving_variants(self):
        cfg = _make_config()
        trades = {
            "momentum": [
                _make_trade(pnl=100, r_multiple=2.0, day=20),
                _make_trade(pnl=50, r_multiple=1.0, day=21),
            ],
        }
        variant = SweepVariant(
            name="wider_heat",
            description="Wider heat cap",
            config_factory=lambda c: replace(c, heat_cap_R=5.0),
        )
        results = run_sweep(cfg, trades, [variant])
        winners = find_winners(results)
        # wider heat should at least not be worse
        assert isinstance(winners, list)

    def test_empty_results(self):
        assert find_winners([]) == []


class TestBuildCombinedVariant:
    def test_combines_winners(self):
        from crypto_trader.portfolio.sweep import SweepResult

        variant1 = SweepVariant(
            "v1", "test1",
            config_factory=lambda c: replace(c, heat_cap_R=4.0),
        )
        variant2 = SweepVariant(
            "v2", "test2",
            config_factory=lambda c: replace(c, directional_cap_R=3.0),
        )

        winners = [
            SweepResult("v1", 10, 0, 5.0, 0.05, 500.0, 10500.0),
            SweepResult("v2", 10, 0, 4.0, 0.04, 400.0, 10400.0),
        ]

        combined = build_combined_variant(winners, [variant1, variant2])
        assert combined is not None
        assert "v1" in combined.name
        assert "v2" in combined.name

        # Apply combined factory
        cfg = _make_config()
        new_cfg = combined.config_factory(cfg)
        assert new_cfg.heat_cap_R == 4.0
        assert new_cfg.directional_cap_R == 3.0

    def test_no_winners(self):
        assert build_combined_variant([], []) is None


class TestFormatSweepTable:
    def test_formats_results(self):
        from crypto_trader.portfolio.sweep import SweepResult

        results = [
            SweepResult("__baseline__", 10, 2, 5.0, 0.06, 500.0, 10500.0),
            SweepResult("tight_heat", 8, 4, 3.0, 0.04, 300.0, 10300.0,
                         deltas={"total_R": -2.0}),
        ]
        table = format_sweep_table(results)
        assert "__baseline__" in table
        assert "tight_heat" in table
        assert "TotalR" in table

    def test_empty_results(self):
        assert format_sweep_table([]) == "No results."
