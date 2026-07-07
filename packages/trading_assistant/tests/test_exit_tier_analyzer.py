"""Tests for ExitTierAnalyzer (Phase 5 — Exit Tier Analysis)."""
from __future__ import annotations

import statistics
from datetime import datetime, timezone

import pytest

from trading_assistant.schemas.strategy_profile import (
    ExitProfile,
    ExitTier,
    StrategyProfile,
    StrategyRegistry,
)
from trading_assistant.skills.exit_tier_analyzer import ExitTierAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    pnl: float = 100.0,
    mfe_r: float | None = None,
    mae_r: float | None = None,
    mfe_pct: float | None = None,
    pnl_pct: float | None = None,
    entry_price: float = 50000.0,
    mfe_price: float | None = None,
    atr_at_entry: float = 500.0,
    side: str = "LONG",
    trade_id: str = "t1",
) -> dict:
    """Return a dict that can be passed to TradeEvent(**d)."""
    return {
        "trade_id": trade_id,
        "bot_id": "test_bot",
        "pair": "BTC/USDT",
        "side": side,
        "entry_time": datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        "exit_time": datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        "entry_price": entry_price,
        "exit_price": entry_price + pnl,
        "position_size": 1.0,
        "pnl": pnl,
        "pnl_pct": pnl_pct if pnl_pct is not None else pnl / entry_price * 100,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "mfe_pct": mfe_pct,
        "mfe_price": mfe_price,
        "atr_at_entry": atr_at_entry,
    }


def _trades_from_dicts(dicts: list[dict]):
    from trading_assistant.schemas.events import TradeEvent
    return [TradeEvent(**d) for d in dicts]


def _make_registry(
    bot_id: str = "test_bot",
    tiers: list[ExitTier] | None = None,
    has_chandelier: bool = False,
    sub_engines: list[str] | None = None,
) -> StrategyRegistry:
    """Build a StrategyRegistry with one strategy that has an exit_profile."""
    if tiers is None:
        tiers = [
            ExitTier(tier_name="TP1", tier_type="take_profit", r_target=1.0, partial_pct=0.5),
            ExitTier(tier_name="TP2", tier_type="take_profit", r_target=2.0, partial_pct=0.5),
        ]
    exit_profile = ExitProfile(
        tiers=tiers,
        has_chandelier=has_chandelier,
    )
    profile = StrategyProfile(
        display_name="TestStrategy",
        bot_id=bot_id,
        sub_engines=sub_engines or ["MOMENTUM"],
        exit_profile=exit_profile,
    )
    return StrategyRegistry(strategies={"test_strat": profile})


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestExitTierSchemas:
    def test_exit_tier_instantiation(self):
        tier = ExitTier(
            tier_name="TP1",
            tier_type="take_profit",
            r_target=1.5,
            partial_pct=0.5,
        )
        assert tier.tier_name == "TP1"
        assert tier.tier_type == "take_profit"
        assert tier.r_target == 1.5
        assert tier.partial_pct == 0.5

    def test_exit_tier_defaults(self):
        tier = ExitTier()
        assert tier.tier_name == ""
        assert tier.tier_type == ""
        assert tier.r_target == 0.0
        assert tier.partial_pct == 1.0

    def test_exit_profile_instantiation(self):
        profile = ExitProfile(
            tiers=[ExitTier(tier_name="TP1", r_target=1.0)],
            has_chandelier=True,
            regime_multipliers={"trending": 1.2, "volatile": 0.8},
        )
        assert len(profile.tiers) == 1
        assert profile.has_chandelier is True
        assert profile.regime_multipliers["trending"] == 1.2

    def test_exit_profile_defaults(self):
        profile = ExitProfile()
        assert profile.tiers == []
        assert profile.has_chandelier is False
        assert profile.regime_multipliers == {}


# ---------------------------------------------------------------------------
# MFE R-multiple computation
# ---------------------------------------------------------------------------

class TestComputeMfeR:
    def test_direct_mfe_r(self):
        """When mfe_r is available directly, use it."""
        trades = _trades_from_dicts([_make_trade(mfe_r=2.5)])
        analyzer = ExitTierAnalyzer()
        result = analyzer._compute_mfe_r(trades[0])
        assert result == 2.5

    def test_mfe_r_none_when_no_data(self):
        """When neither mfe_r nor fallback data available, return None."""
        trades = _trades_from_dicts([_make_trade(mfe_r=None, mfe_pct=None)])
        analyzer = ExitTierAnalyzer()
        result = analyzer._compute_mfe_r(trades[0])
        assert result is None

    def test_mfe_r_zero(self):
        """mfe_r of 0.0 is valid (not None)."""
        trades = _trades_from_dicts([_make_trade(mfe_r=0.0)])
        analyzer = ExitTierAnalyzer()
        result = analyzer._compute_mfe_r(trades[0])
        assert result == 0.0


# ---------------------------------------------------------------------------
# MAE R-multiple computation
# ---------------------------------------------------------------------------

class TestComputeMaeR:
    def test_direct_mae_r(self):
        trades = _trades_from_dicts([_make_trade(mae_r=-0.5)])
        analyzer = ExitTierAnalyzer()
        result = analyzer._compute_mae_r(trades[0])
        assert result == -0.5

    def test_mae_r_none_when_no_data(self):
        trades = _trades_from_dicts([_make_trade(mae_r=None)])
        analyzer = ExitTierAnalyzer()
        result = analyzer._compute_mae_r(trades[0])
        assert result is None


# ---------------------------------------------------------------------------
# Tier hit rates
# ---------------------------------------------------------------------------

class TestTierHitRates:
    def test_correct_hit_rate_calculation(self):
        """Verify hit rate = % trades with MFE >= tier R-target."""
        # 3 trades with MFE R of 1.5, 2.5, 0.5
        trade_dicts = [
            _make_trade(mfe_r=1.5, pnl=100, trade_id="t1"),
            _make_trade(mfe_r=2.5, pnl=200, trade_id="t2"),
            _make_trade(mfe_r=0.5, pnl=-50, trade_id="t3"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [
            {"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.5},
            {"tier_name": "TP2", "r_target": 2.0, "partial_pct": 0.5},
        ]
        analyzer = ExitTierAnalyzer()
        result = analyzer._tier_hit_rates(trades, tiers)

        assert len(result) == 2
        # TP1 (r_target=1.0): 2 of 3 trades have MFE >= 1.0
        assert result[0]["tier_name"] == "TP1"
        assert result[0]["hit_count"] == 2
        assert result[0]["total_trades"] == 3
        assert result[0]["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

        # TP2 (r_target=2.0): 1 of 3 trades have MFE >= 2.0
        assert result[1]["tier_name"] == "TP2"
        assert result[1]["hit_count"] == 1
        assert result[1]["hit_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_all_trades_hit(self):
        """100% hit rate when all trades exceed target."""
        trade_dicts = [
            _make_trade(mfe_r=3.0, pnl=100, trade_id=f"t{i}")
            for i in range(5)
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [{"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.5}]
        analyzer = ExitTierAnalyzer()
        result = analyzer._tier_hit_rates(trades, tiers)
        assert result[0]["hit_rate"] == pytest.approx(1.0)
        assert result[0]["hit_count"] == 5

    def test_no_trades_hit(self):
        """0% hit rate when no trades exceed target."""
        trade_dicts = [
            _make_trade(mfe_r=0.3, pnl=-50, trade_id=f"t{i}")
            for i in range(5)
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [{"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.5}]
        analyzer = ExitTierAnalyzer()
        result = analyzer._tier_hit_rates(trades, tiers)
        assert result[0]["hit_rate"] == pytest.approx(0.0)
        assert result[0]["hit_count"] == 0

    def test_no_mfe_data_empty_result(self):
        """Trades without MFE data -> empty tier stats."""
        trade_dicts = [
            _make_trade(mfe_r=None, pnl=100, trade_id=f"t{i}")
            for i in range(5)
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [{"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.5}]
        analyzer = ExitTierAnalyzer()
        result = analyzer._tier_hit_rates(trades, tiers)
        assert result == []

    def test_multiple_tiers_different_rates(self):
        """Different hit rates for different tier targets."""
        trade_dicts = [
            _make_trade(mfe_r=0.8, pnl=50, trade_id="t1"),
            _make_trade(mfe_r=1.2, pnl=100, trade_id="t2"),
            _make_trade(mfe_r=1.8, pnl=150, trade_id="t3"),
            _make_trade(mfe_r=2.5, pnl=200, trade_id="t4"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [
            {"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.33},
            {"tier_name": "TP2", "r_target": 1.5, "partial_pct": 0.33},
            {"tier_name": "TP3", "r_target": 2.0, "partial_pct": 0.34},
        ]
        analyzer = ExitTierAnalyzer()
        result = analyzer._tier_hit_rates(trades, tiers)
        assert len(result) == 3
        # TP1: 3/4, TP2: 2/4, TP3: 1/4
        assert result[0]["hit_count"] == 3
        assert result[1]["hit_count"] == 2
        assert result[2]["hit_count"] == 1

    def test_avg_mfe_when_hit(self):
        """Average MFE for trades that hit should be computed correctly."""
        trade_dicts = [
            _make_trade(mfe_r=1.5, pnl=100, trade_id="t1"),
            _make_trade(mfe_r=2.5, pnl=200, trade_id="t2"),
            _make_trade(mfe_r=0.3, pnl=-50, trade_id="t3"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [{"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.5}]
        analyzer = ExitTierAnalyzer()
        result = analyzer._tier_hit_rates(trades, tiers)
        # Only trades with MFE >= 1.0 are hits: 1.5 and 2.5
        assert result[0]["avg_mfe_when_hit"] == pytest.approx(
            statistics.mean([1.5, 2.5]), abs=0.01,
        )


# ---------------------------------------------------------------------------
# Optimal tier targets
# ---------------------------------------------------------------------------

class TestOptimalTierTargets:
    def test_grid_search_finds_value(self):
        """Grid search should find an optimal R-target."""
        trade_dicts = [
            _make_trade(mfe_r=1.5, pnl=100, trade_id="t1"),
            _make_trade(mfe_r=2.0, pnl=200, trade_id="t2"),
            _make_trade(mfe_r=0.8, pnl=-50, trade_id="t3"),
            _make_trade(mfe_r=1.2, pnl=80, trade_id="t4"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [{"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.5}]
        analyzer = ExitTierAnalyzer()
        result = analyzer._optimal_tier_targets(trades, tiers)
        assert len(result) == 1
        assert "optimal_target" in result[0]
        assert "current_target" in result[0]
        assert "improvement_pct" in result[0]
        assert result[0]["current_target"] == pytest.approx(1.0)

    def test_no_mfe_data_empty(self):
        """No MFE data -> empty optimization results."""
        trade_dicts = [
            _make_trade(mfe_r=None, pnl=100, trade_id=f"t{i}")
            for i in range(5)
        ]
        trades = _trades_from_dicts(trade_dicts)
        tiers = [{"tier_name": "TP1", "r_target": 1.0, "partial_pct": 0.5}]
        analyzer = ExitTierAnalyzer()
        result = analyzer._optimal_tier_targets(trades, tiers)
        assert result == []


# ---------------------------------------------------------------------------
# Stop placement analysis
# ---------------------------------------------------------------------------

class TestStopPlacementAnalysis:
    def test_mae_distribution(self):
        """Winners with MAE data should produce distribution analysis."""
        trade_dicts = [
            _make_trade(pnl=100, mae_r=-0.3, trade_id="t1"),
            _make_trade(pnl=200, mae_r=-0.6, trade_id="t2"),
            _make_trade(pnl=50, mae_r=-1.2, trade_id="t3"),
            _make_trade(pnl=-100, mae_r=-2.0, trade_id="t4"),  # loser excluded
        ]
        trades = _trades_from_dicts(trade_dicts)
        analyzer = ExitTierAnalyzer()
        result = analyzer._stop_placement_analysis(trades)

        assert result != {}
        assert result["total_winners_with_mae"] == 3  # only winners
        assert "avg_winner_mae_r" in result
        assert "median_winner_mae_r" in result
        assert "levels" in result
        assert len(result["levels"]) == 6  # thresholds: 0.25, 0.5, 0.75, 1.0, 1.5, 2.0

        # At threshold 0.25: all 3 winners had MAE >= 0.25
        level_025 = result["levels"][0]
        assert level_025["mae_threshold_r"] == 0.25
        assert level_025["pct_winners_touched"] == pytest.approx(1.0)

        # At threshold 1.0: only 1 winner (mae=-1.2) had MAE >= 1.0
        level_10 = result["levels"][3]
        assert level_10["mae_threshold_r"] == 1.0
        assert level_10["pct_winners_touched"] == pytest.approx(1 / 3, abs=0.01)

    def test_no_winners_empty_result(self):
        """Only losers -> empty result."""
        trade_dicts = [
            _make_trade(pnl=-100, mae_r=-1.0, trade_id="t1"),
            _make_trade(pnl=-50, mae_r=-0.5, trade_id="t2"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        analyzer = ExitTierAnalyzer()
        result = analyzer._stop_placement_analysis(trades)
        assert result == {}

    def test_winners_without_mae_empty_result(self):
        """Winners but no MAE data -> empty result."""
        trade_dicts = [
            _make_trade(pnl=100, mae_r=None, trade_id="t1"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        analyzer = ExitTierAnalyzer()
        result = analyzer._stop_placement_analysis(trades)
        assert result == {}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_trades(self):
        """Empty trade list -> None."""
        registry = _make_registry()
        analyzer = ExitTierAnalyzer(strategy_registry=registry)
        result = analyzer.analyze([], "test_bot")
        assert result is None

    def test_no_exit_profile(self):
        """Bot without exit_profile -> None."""
        profile = StrategyProfile(display_name="NoExit", bot_id="test_bot")
        registry = StrategyRegistry(strategies={"s1": profile})
        analyzer = ExitTierAnalyzer(strategy_registry=registry)
        trades = _trades_from_dicts([_make_trade(mfe_r=1.5)])
        result = analyzer.analyze(trades, "test_bot")
        assert result is None

    def test_no_registry(self):
        """No strategy_registry -> None."""
        analyzer = ExitTierAnalyzer(strategy_registry=None)
        trades = _trades_from_dicts([_make_trade(mfe_r=1.5)])
        result = analyzer.analyze(trades, "test_bot")
        assert result is None

    def test_no_mfe_data_in_trades(self):
        """Trades without any MFE data -> None."""
        registry = _make_registry()
        analyzer = ExitTierAnalyzer(strategy_registry=registry)
        trade_dicts = [
            _make_trade(mfe_r=None, mfe_pct=None, trade_id=f"t{i}")
            for i in range(10)
        ]
        trades = _trades_from_dicts(trade_dicts)
        result = analyzer.analyze(trades, "test_bot")
        assert result is None

    def test_empty_tiers_in_profile(self):
        """Exit profile with empty tiers list -> None."""
        registry = _make_registry(tiers=[])
        analyzer = ExitTierAnalyzer(strategy_registry=registry)
        trades = _trades_from_dicts([_make_trade(mfe_r=1.5)])
        result = analyzer.analyze(trades, "test_bot")
        assert result is None


# ---------------------------------------------------------------------------
# Integration: full analyze()
# ---------------------------------------------------------------------------

class TestAnalyzeIntegration:
    def test_full_analysis_structure(self):
        """Full analysis produces dict with expected keys."""
        registry = _make_registry(
            tiers=[
                ExitTier(tier_name="TP1", tier_type="take_profit", r_target=1.0, partial_pct=0.5),
                ExitTier(tier_name="TP2", tier_type="take_profit", r_target=2.0, partial_pct=0.5),
            ],
            has_chandelier=True,
        )
        trade_dicts = [
            _make_trade(mfe_r=1.5, pnl=100, mae_r=-0.3, trade_id="t1"),
            _make_trade(mfe_r=2.5, pnl=200, mae_r=-0.5, trade_id="t2"),
            _make_trade(mfe_r=0.5, pnl=-50, mae_r=-1.0, trade_id="t3"),
            _make_trade(mfe_r=3.0, pnl=300, mae_r=-0.2, trade_id="t4"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        analyzer = ExitTierAnalyzer(strategy_registry=registry)
        result = analyzer.analyze(trades, "test_bot", period="2026-W10")

        assert result is not None
        assert result["bot_id"] == "test_bot"
        assert result["period"] == "2026-W10"
        assert result["has_chandelier"] is True

        # Tiers
        assert "tiers" in result
        assert len(result["tiers"]) == 2
        assert result["tiers"][0]["tier_name"] == "TP1"
        assert result["tiers"][1]["tier_name"] == "TP2"

        # Optimal targets
        assert "optimal_targets" in result
        assert len(result["optimal_targets"]) == 2

        # Stop placement
        assert "stop_placement" in result
        assert "levels" in result["stop_placement"]

    def test_analysis_with_mixed_mfe_availability(self):
        """Some trades have MFE, some don't — analysis should work on available data."""
        registry = _make_registry()
        trade_dicts = [
            _make_trade(mfe_r=1.5, pnl=100, trade_id="t1"),
            _make_trade(mfe_r=None, pnl=50, trade_id="t2"),  # no MFE
            _make_trade(mfe_r=2.0, pnl=200, trade_id="t3"),
        ]
        trades = _trades_from_dicts(trade_dicts)
        analyzer = ExitTierAnalyzer(strategy_registry=registry)
        result = analyzer.analyze(trades, "test_bot")
        assert result is not None
        # Total trades in tier stats should be 2 (those with MFE data)
        for tier in result["tiers"]:
            assert tier["total_trades"] == 2

    def test_bot_not_in_registry(self):
        """Bot not found in registry -> no exit_profile -> None."""
        registry = _make_registry(bot_id="other_bot")
        analyzer = ExitTierAnalyzer(strategy_registry=registry)
        trades = _trades_from_dicts([_make_trade(mfe_r=1.5)])
        result = analyzer.analyze(trades, "test_bot")
        assert result is None
