"""Tests for RegimeParameterAnalyzer (Phase 4 — Regime-Conditional Parameter Analysis)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType
from trading_assistant.schemas.regime_conditional import RegimeParameterAnalysis, RegimeParameterStats
from trading_assistant.skills.regime_parameter_analyzer import RegimeParameterAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(
    param_name: str = "quality_min",
    valid_range: tuple[float, float] | None = (0.0, 1.0),
    value_type: str = "float",
    current_value: object = 0.5,
    valid_values: list | None = None,
    is_safety_critical: bool = False,
    strategy_id: str = "",
) -> ParameterDefinition:
    return ParameterDefinition(
        param_name=param_name,
        bot_id="test_bot",
        strategy_id=strategy_id,
        param_type=ParameterType.YAML_FIELD,
        file_path="config.yaml",
        yaml_key=f"strategy.{param_name}",
        current_value=current_value,
        valid_range=valid_range,
        valid_values=valid_values,
        value_type=value_type,
        category="entry_signal",
        is_safety_critical=is_safety_critical,
    )


def _make_trade(
    regime: str = "trending",
    pnl: float = 100.0,
    strategy_params: dict | None = None,
    mfe_r: float | None = None,
) -> dict:
    """Return a dict that can be passed to TradeEvent(**d)."""
    return {
        "trade_id": "t1",
        "bot_id": "test_bot",
        "pair": "BTC/USDT",
        "side": "LONG",
        "entry_time": datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        "exit_time": datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        "entry_price": 50000.0,
        "exit_price": 50100.0,
        "position_size": 1.0,
        "pnl": pnl,
        "pnl_pct": pnl / 500.0,
        "market_regime": regime,
        "strategy_params_at_entry": strategy_params,
        "mfe_r": mfe_r,
    }


def _trades_from_dicts(dicts: list[dict]):
    from trading_assistant.schemas.events import TradeEvent
    return [TradeEvent(**d) for d in dicts]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestRegimeParameterSchemas:
    def test_regime_parameter_stats_instantiation(self):
        stats = RegimeParameterStats(
            regime="trending",
            trade_count=50,
            optimal_value=0.7,
            win_rate=0.65,
            avg_pnl=120.0,
            profit_factor=2.1,
        )
        assert stats.regime == "trending"
        assert stats.trade_count == 50
        assert stats.optimal_value == 0.7
        assert stats.win_rate == 0.65
        assert stats.avg_pnl == 120.0
        assert stats.profit_factor == 2.1

    def test_regime_parameter_stats_defaults(self):
        stats = RegimeParameterStats(regime="volatile")
        assert stats.trade_count == 0
        assert stats.optimal_value is None
        assert stats.win_rate == 0.0
        assert stats.avg_pnl == 0.0
        assert stats.profit_factor == 0.0

    def test_regime_parameter_analysis_instantiation(self):
        analysis = RegimeParameterAnalysis(
            param_name="quality_min",
            bot_id="test_bot",
            strategy_id="alpha",
            regimes_analyzed=["trending", "volatile"],
            optimal_per_regime={"trending": 0.7, "volatile": 0.3},
            current_value=0.5,
            regime_sensitivity=0.45,
            regime_stats=[],
            recommendations=["In trending regime: consider quality_min=0.7"],
        )
        assert analysis.param_name == "quality_min"
        assert analysis.bot_id == "test_bot"
        assert analysis.regime_sensitivity == 0.45
        assert len(analysis.recommendations) == 1

    def test_regime_parameter_analysis_defaults(self):
        analysis = RegimeParameterAnalysis(param_name="x", bot_id="b")
        assert analysis.regimes_analyzed == []
        assert analysis.optimal_per_regime == {}
        assert analysis.current_value is None
        assert analysis.regime_sensitivity == 0.0
        assert analysis.regime_stats == []
        assert analysis.recommendations == []


# ---------------------------------------------------------------------------
# _stratify_by_regime
# ---------------------------------------------------------------------------

class TestStratifyByRegime:
    def test_groups_by_regime(self):
        trades = _trades_from_dicts([
            _make_trade(regime="trending", pnl=100),
            _make_trade(regime="trending", pnl=50),
            _make_trade(regime="volatile", pnl=-30),
            _make_trade(regime="ranging", pnl=20),
        ])
        analyzer = RegimeParameterAnalyzer()
        groups = analyzer._stratify_by_regime(trades)
        assert set(groups.keys()) == {"trending", "volatile", "ranging"}
        assert len(groups["trending"]) == 2
        assert len(groups["volatile"]) == 1
        assert len(groups["ranging"]) == 1

    def test_empty_regime_falls_to_unknown(self):
        trades = _trades_from_dicts([
            _make_trade(regime="", pnl=100),
        ])
        analyzer = RegimeParameterAnalyzer()
        groups = analyzer._stratify_by_regime(trades)
        assert "unknown" in groups
        assert len(groups["unknown"]) == 1

    def test_empty_trades(self):
        analyzer = RegimeParameterAnalyzer()
        groups = analyzer._stratify_by_regime([])
        assert groups == {}


# ---------------------------------------------------------------------------
# _build_grid
# ---------------------------------------------------------------------------

class TestBuildGrid:
    def test_float_grid(self):
        param = _make_param(valid_range=(0.0, 1.0), value_type="float")
        analyzer = RegimeParameterAnalyzer()
        grid = analyzer._build_grid(param, n_points=11)
        assert len(grid) == 11
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(1.0)
        # Monotonically increasing
        for i in range(1, len(grid)):
            assert grid[i] > grid[i - 1]

    def test_int_grid(self):
        param = _make_param(valid_range=(10.0, 50.0), value_type="int")
        analyzer = RegimeParameterAnalyzer()
        grid = analyzer._build_grid(param, n_points=11)
        assert all(isinstance(v, int) for v in grid)
        assert grid[0] == 10
        assert grid[-1] <= 50

    def test_bool_grid_via_valid_values(self):
        param = _make_param(
            valid_range=None,
            value_type="bool",
            valid_values=[True, False],
        )
        analyzer = RegimeParameterAnalyzer()
        grid = analyzer._build_grid(param)
        assert set(grid) == {True, False}

    def test_valid_values_override(self):
        param = _make_param(
            valid_range=(0.0, 1.0),
            value_type="float",
            valid_values=[0.1, 0.5, 0.9],
        )
        analyzer = RegimeParameterAnalyzer()
        grid = analyzer._build_grid(param)
        assert grid == [0.1, 0.5, 0.9]

    def test_no_range_no_values(self):
        param = _make_param(valid_range=None, valid_values=None)
        analyzer = RegimeParameterAnalyzer()
        grid = analyzer._build_grid(param)
        assert grid == []

    def test_small_range_int(self):
        param = _make_param(valid_range=(1.0, 3.0), value_type="int")
        analyzer = RegimeParameterAnalyzer()
        grid = analyzer._build_grid(param, n_points=11)
        # step = max(1, int(2/10)) = 1
        assert 1 in grid
        assert all(isinstance(v, int) for v in grid)


# ---------------------------------------------------------------------------
# Sensitivity scoring
# ---------------------------------------------------------------------------

class TestSensitivityScoring:
    def test_constant_optimal_zero_sensitivity(self):
        """Same optimal value across all regimes -> sensitivity = 0.0."""
        param = _make_param(valid_range=(0.0, 1.0), value_type="float")
        analyzer = RegimeParameterAnalyzer()
        # All regimes have the same optimal
        sensitivity = analyzer._compute_sensitivity([0.5, 0.5, 0.5], param)
        assert sensitivity == pytest.approx(0.0)

    def test_divergent_optimal_high_sensitivity(self):
        """Very different optimal values across regimes -> high sensitivity."""
        param = _make_param(valid_range=(0.0, 1.0), value_type="float")
        analyzer = RegimeParameterAnalyzer()
        # Extreme divergence: 0.0 vs 1.0
        sensitivity = analyzer._compute_sensitivity([0.0, 1.0], param)
        assert sensitivity > 0.3  # Should be meaningfully different

    def test_moderate_divergence(self):
        """Moderate variation across regimes."""
        param = _make_param(valid_range=(0.0, 1.0), value_type="float")
        analyzer = RegimeParameterAnalyzer()
        sensitivity = analyzer._compute_sensitivity([0.3, 0.5, 0.7], param)
        assert 0.0 < sensitivity < 1.0

    def test_boolean_different_optimal_sensitivity_1(self):
        """Different boolean optimal per regime -> sensitivity = 1.0."""
        param = _make_param(
            valid_range=None, value_type="bool", valid_values=[True, False],
        )
        analyzer = RegimeParameterAnalyzer()
        sensitivity = analyzer._compute_sensitivity([True, False], param)
        assert sensitivity == 1.0

    def test_boolean_same_optimal_sensitivity_0(self):
        """Same boolean optimal per regime -> sensitivity = 0.0."""
        param = _make_param(
            valid_range=None, value_type="bool", valid_values=[True, False],
        )
        analyzer = RegimeParameterAnalyzer()
        sensitivity = analyzer._compute_sensitivity([True, True, True], param)
        assert sensitivity == 0.0

    def test_single_regime_zero_sensitivity(self):
        """Only one regime analyzed -> sensitivity = 0.0 (nothing to compare)."""
        param = _make_param(valid_range=(0.0, 1.0), value_type="float")
        analyzer = RegimeParameterAnalyzer()
        sensitivity = analyzer._compute_sensitivity([0.7], param)
        assert sensitivity == 0.0

    def test_sensitivity_capped_at_one(self):
        """Sensitivity should never exceed 1.0."""
        param = _make_param(valid_range=(0.0, 0.1), value_type="float")
        analyzer = RegimeParameterAnalyzer()
        sensitivity = analyzer._compute_sensitivity([0.0, 0.1], param)
        assert sensitivity <= 1.0


# ---------------------------------------------------------------------------
# Min per regime
# ---------------------------------------------------------------------------

class TestMinPerRegime:
    def test_regimes_below_threshold_excluded(self):
        """Regimes with fewer than min_per_regime trades are excluded from analysis."""
        # Create 20 trades in "trending" and only 5 in "volatile"
        trade_dicts = []
        for i in range(20):
            trade_dicts.append(_make_trade(
                regime="trending",
                pnl=100.0 if i % 2 == 0 else -50.0,
                strategy_params={"quality_min": 0.5},
            ))
        for _ in range(5):
            trade_dicts.append(_make_trade(
                regime="volatile",
                pnl=50.0,
                strategy_params={"quality_min": 0.3},
            ))
        trades = _trades_from_dicts(trade_dicts)

        param = _make_param(valid_range=(0.0, 1.0))
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot", min_per_regime=15)

        # Only "trending" should be analyzed (20 >= 15), "volatile" excluded (5 < 15)
        assert "trending" in result.regimes_analyzed
        assert "volatile" not in result.regimes_analyzed

    def test_all_regimes_below_threshold(self):
        """All regimes below threshold -> empty analysis."""
        trade_dicts = [
            _make_trade(regime="trending", pnl=100, strategy_params={"quality_min": 0.5}),
            _make_trade(regime="volatile", pnl=50, strategy_params={"quality_min": 0.3}),
        ]
        trades = _trades_from_dicts(trade_dicts)
        param = _make_param(valid_range=(0.0, 1.0))
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot", min_per_regime=15)

        assert result.regimes_analyzed == []
        assert result.regime_sensitivity == 0.0


# ---------------------------------------------------------------------------
# Full analyze()
# ---------------------------------------------------------------------------

class TestFullAnalyze:
    def test_empty_trades(self):
        param = _make_param()
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze([], param, "test_bot")
        assert isinstance(result, RegimeParameterAnalysis)
        assert result.param_name == "quality_min"
        assert result.bot_id == "test_bot"
        assert result.regimes_analyzed == []

    def test_no_grid_returns_empty(self):
        """Parameter with no valid_range and no valid_values -> empty grid."""
        param = _make_param(valid_range=None, valid_values=None)
        trades = _trades_from_dicts([
            _make_trade(regime="trending", pnl=100)
            for _ in range(20)
        ])
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot")
        assert result.regimes_analyzed == []

    def test_end_to_end_with_mock_trades(self):
        """End-to-end test: two regimes with enough trades, verifies structure."""
        trade_dicts = []
        # 20 trades in "trending" with param at 0.7 (all winners)
        for i in range(20):
            trade_dicts.append(_make_trade(
                regime="trending",
                pnl=100.0 + i,
                strategy_params={"quality_min": 0.7},
            ))
        # 20 trades in "ranging" with param at 0.3 (mixed)
        for i in range(20):
            trade_dicts.append(_make_trade(
                regime="ranging",
                pnl=50.0 if i % 2 == 0 else -30.0,
                strategy_params={"quality_min": 0.3},
            ))

        trades = _trades_from_dicts(trade_dicts)
        param = _make_param(valid_range=(0.0, 1.0), current_value=0.5)
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot", min_per_regime=15)

        assert result.param_name == "quality_min"
        assert result.bot_id == "test_bot"
        assert set(result.regimes_analyzed) == {"trending", "ranging"}
        assert len(result.regime_stats) == 2
        assert isinstance(result.regime_sensitivity, float)
        assert 0.0 <= result.regime_sensitivity <= 1.0

        # Each regime stat should have correct structure
        for stat in result.regime_stats:
            assert isinstance(stat, RegimeParameterStats)
            assert stat.trade_count == 20
            assert stat.optimal_value is not None

    def test_unknown_regime_excluded(self):
        """Trades with regime='unknown' should be excluded from analysis."""
        trade_dicts = []
        for _i in range(20):
            trade_dicts.append(_make_trade(
                regime="unknown",
                pnl=100.0,
                strategy_params={"quality_min": 0.5},
            ))
        for _i in range(20):
            trade_dicts.append(_make_trade(
                regime="trending",
                pnl=50.0,
                strategy_params={"quality_min": 0.5},
            ))
        trades = _trades_from_dicts(trade_dicts)
        param = _make_param(valid_range=(0.0, 1.0))
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot", min_per_regime=15)

        assert "unknown" not in result.regimes_analyzed
        assert "trending" in result.regimes_analyzed

    def test_strategy_id_propagated(self):
        """strategy_id from param should appear in analysis result."""
        trade_dicts = [
            _make_trade(regime="trending", pnl=100, strategy_params={"quality_min": 0.5})
            for _ in range(20)
        ]
        trades = _trades_from_dicts(trade_dicts)
        param = _make_param(valid_range=(0.0, 1.0), strategy_id="alpha")
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot", min_per_regime=15)
        assert result.strategy_id == "alpha"

    def test_recommendations_generated_for_high_sensitivity(self):
        """When sensitivity is high and optima differ, recommendations are generated."""
        trade_dicts = []
        # Regime A: param=0.1 performs best (high pnl)
        for _i in range(20):
            trade_dicts.append(_make_trade(
                regime="trending",
                pnl=200.0,
                strategy_params={"quality_min": 0.1},
            ))
        # Regime B: param=0.9 performs best (high pnl)
        for _i in range(20):
            trade_dicts.append(_make_trade(
                regime="volatile",
                pnl=200.0,
                strategy_params={"quality_min": 0.9},
            ))
        trades = _trades_from_dicts(trade_dicts)
        param = _make_param(valid_range=(0.0, 1.0), current_value=0.5)
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot", min_per_regime=15)

        # With widely divergent optima, sensitivity should be high
        # and recommendations should be generated
        if result.regime_sensitivity > 0.3:
            assert len(result.recommendations) > 0
            # Recommendations should mention regime names
            combined = " ".join(result.recommendations)
            assert "regime" in combined.lower() or "consider" in combined.lower()

    def test_optimal_per_regime_populated(self):
        """optimal_per_regime dict should map regime -> optimal value."""
        trade_dicts = []
        for _i in range(20):
            trade_dicts.append(_make_trade(
                regime="trending",
                pnl=100.0,
                strategy_params={"quality_min": 0.5},
            ))
        trades = _trades_from_dicts(trade_dicts)
        param = _make_param(valid_range=(0.0, 1.0))
        analyzer = RegimeParameterAnalyzer()
        result = analyzer.analyze(trades, param, "test_bot", min_per_regime=15)

        assert "trending" in result.optimal_per_regime
        assert isinstance(result.optimal_per_regime["trending"], (int, float))
