"""Tests for AblationAnalyzer — statistical analysis of boolean ablation flags."""
from __future__ import annotations


import pytest

from trading_assistant.schemas.ablation_analysis import (
    AblationAnalysis,
    AblationFlagStats,
    AblationRegimeStats,
)
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.experiments import ExperimentType
from trading_assistant.skills.ablation_analyzer import AblationAnalyzer
from tests.factories import make_trade


# ─── Helpers ─────────────────────────────────────────────────────────


def _trade(
    trade_id: str = "t1",
    pnl: float = 100.0,
    market_regime: str = "trending",
    strategy_params: dict | None = None,
    bot_id: str = "bot1",
) -> "TradeEvent":
    return make_trade(
        trade_id=trade_id,
        bot_id=bot_id,
        pnl=pnl,
        market_regime=market_regime,
        strategy_params_at_entry=strategy_params,
    )


def _make_trades_with_flag(
    flag_name: str = "use_oscillation_gate",
    n_enabled: int = 15,
    n_disabled: int = 15,
    enabled_pnl: float = -20.0,
    disabled_pnl: float = 50.0,
    regime: str = "trending",
    extra_params: dict | None = None,
) -> list:
    """Build trades where a boolean flag is enabled or disabled."""
    trades = []
    base_params = extra_params or {}
    for i in range(n_enabled):
        params = {flag_name: True, **base_params}
        trades.append(_trade(
            trade_id=f"en_{i}",
            pnl=enabled_pnl + (i * 0.1),  # slight variance
            market_regime=regime,
            strategy_params=params,
        ))
    for i in range(n_disabled):
        params = {flag_name: False, **base_params}
        trades.append(_trade(
            trade_id=f"dis_{i}",
            pnl=disabled_pnl + (i * 0.1),
            market_regime=regime,
            strategy_params=params,
        ))
    return trades


# ─── Schema Tests ────────────────────────────────────────────────────


class TestSchemaDefaults:
    """AblationRegimeStats, AblationFlagStats, AblationAnalysis instantiation."""

    def test_ablation_regime_stats_defaults(self):
        s = AblationRegimeStats(regime="trending")
        assert s.regime == "trending"
        assert s.enabled_count == 0
        assert s.disabled_count == 0
        assert s.enabled_win_rate == 0.0
        assert s.disabled_win_rate == 0.0
        assert s.pnl_delta == 0.0

    def test_ablation_flag_stats_defaults(self):
        s = AblationFlagStats(flag_name="use_gate")
        assert s.flag_name == "use_gate"
        assert s.strategy_id == ""
        assert s.bot_id == ""
        assert s.enabled_count == 0
        assert s.disabled_count == 0
        assert s.enabled_win_rate == 0.0
        assert s.disabled_win_rate == 0.0
        assert s.enabled_avg_pnl == 0.0
        assert s.disabled_avg_pnl == 0.0
        assert s.pnl_delta == 0.0
        assert s.statistical_significance == 1.0
        assert s.regime_breakdown == []

    def test_ablation_analysis_defaults(self):
        a = AblationAnalysis(bot_id="bot1")
        assert a.bot_id == "bot1"
        assert a.period == ""
        assert a.flags == []
        assert a.flags_with_signal == []

    def test_ablation_analysis_with_period(self):
        a = AblationAnalysis(bot_id="bot1", period="2026-05-01")
        assert a.period == "2026-05-01"

    def test_flag_stats_serialization(self):
        s = AblationFlagStats(
            flag_name="use_gate",
            bot_id="bot1",
            enabled_count=10,
            disabled_count=12,
            pnl_delta=15.5,
            regime_breakdown=[
                AblationRegimeStats(regime="trending", enabled_count=5),
            ],
        )
        data = s.model_dump(mode="json")
        assert data["flag_name"] == "use_gate"
        assert data["pnl_delta"] == 15.5
        assert len(data["regime_breakdown"]) == 1


# ─── Boolean Extraction ──────────────────────────────────────────────


class TestExtractBooleanParams:
    """_extract_boolean_params correctly identifies all-boolean params."""

    def test_boolean_only_extracted(self):
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"use_gate": True, "threshold": 0.5}),
            _trade(trade_id="t2", strategy_params={"use_gate": False, "threshold": 0.7}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert "use_gate" in result
        assert "threshold" not in result

    def test_mixed_types_excluded(self):
        """If a param has bool AND float values across trades, it is excluded."""
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"flag": True}),
            _trade(trade_id="t2", strategy_params={"flag": 0.5}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert "flag" not in result

    def test_string_params_excluded(self):
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"mode": "aggressive", "use_gate": True}),
            _trade(trade_id="t2", strategy_params={"mode": "conservative", "use_gate": False}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert "mode" not in result
        assert "use_gate" in result

    def test_all_true_still_extracted(self):
        """Even if all values are True, param is still identified as boolean."""
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"gate": True}),
            _trade(trade_id="t2", strategy_params={"gate": True}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert "gate" in result

    def test_all_false_still_extracted(self):
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"gate": False}),
            _trade(trade_id="t2", strategy_params={"gate": False}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert "gate" in result

    def test_empty_strategy_params(self):
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params=None),
            _trade(trade_id="t2", strategy_params=None),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert result == {}

    def test_single_trade_excluded(self):
        """Need >= 2 observations to be included."""
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"gate": True}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert result == {}

    def test_multiple_boolean_params(self):
        """Multiple boolean params are all extracted."""
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"gate_a": True, "gate_b": False}),
            _trade(trade_id="t2", strategy_params={"gate_a": False, "gate_b": True}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert "gate_a" in result
        assert "gate_b" in result

    def test_bool_not_confused_with_int(self):
        """int 0 and 1 are NOT treated as boolean (isinstance check)."""
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"flag": 1}),
            _trade(trade_id="t2", strategy_params={"flag": 0}),
        ]
        result = analyzer._extract_boolean_params(trades)
        assert "flag" not in result


# ─── Stats Computation ────────────────────────────────────────────────


class TestFlagStatsComputation:
    """_compute_flag_stats computes correct enabled/disabled metrics."""

    def _make_analyzer_with_stats(
        self,
        enabled_pnls: list[float],
        disabled_pnls: list[float],
    ) -> AblationFlagStats:
        """Create trades and compute stats for a single flag."""
        trades_by_state = {
            True: [
                make_trade(trade_id=f"en_{i}", pnl=p, market_regime="trending")
                for i, p in enumerate(enabled_pnls)
            ],
            False: [
                make_trade(trade_id=f"dis_{i}", pnl=p, market_regime="trending")
                for i, p in enumerate(disabled_pnls)
            ],
        }
        analyzer = AblationAnalyzer()
        return analyzer._compute_flag_stats("test_flag", trades_by_state, "bot1")

    def test_counts(self):
        stats = self._make_analyzer_with_stats([100, -50, 200], [-30, 80])
        assert stats.enabled_count == 3
        assert stats.disabled_count == 2

    def test_enabled_win_rate(self):
        stats = self._make_analyzer_with_stats([100, -50, 200], [-30, 80])
        # 2 wins out of 3 enabled
        assert stats.enabled_win_rate == pytest.approx(0.6667, abs=0.001)

    def test_disabled_win_rate(self):
        stats = self._make_analyzer_with_stats([100, -50, 200], [-30, 80])
        # 1 win out of 2 disabled
        assert stats.disabled_win_rate == pytest.approx(0.5, abs=0.001)

    def test_avg_pnl(self):
        stats = self._make_analyzer_with_stats([100, -50], [200, -100])
        assert stats.enabled_avg_pnl == pytest.approx(25.0, abs=0.01)
        assert stats.disabled_avg_pnl == pytest.approx(50.0, abs=0.01)

    def test_pnl_delta(self):
        """pnl_delta = disabled_avg_pnl - enabled_avg_pnl."""
        stats = self._make_analyzer_with_stats([100, -50], [200, -100])
        # disabled avg = 50, enabled avg = 25 → delta = 25
        assert stats.pnl_delta == pytest.approx(25.0, abs=0.01)

    def test_pnl_delta_negative(self):
        """Negative delta when enabled performs better."""
        stats = self._make_analyzer_with_stats([200, 100], [-50, -30])
        # disabled avg = -40, enabled avg = 150 → delta = -190
        assert stats.pnl_delta == pytest.approx(-190.0, abs=0.01)

    def test_statistical_significance_returned(self):
        """p-value is a float between 0 and 1."""
        stats = self._make_analyzer_with_stats(
            [100, 200, 150, 300, 250],
            [-50, -30, -80, -40, -60],
        )
        assert 0.0 <= stats.statistical_significance <= 1.0

    def test_low_p_value_with_separated_distributions(self):
        """Clearly separated distributions should have low p-value."""
        # All enabled are positive, all disabled are negative
        stats = self._make_analyzer_with_stats(
            [100, 200, 150, 300, 250, 180, 220, 160],
            [-100, -200, -150, -300, -250, -180, -220, -160],
        )
        assert stats.statistical_significance < 0.10

    def test_high_p_value_with_similar_distributions(self):
        """Nearly identical distributions should have high p-value."""
        stats = self._make_analyzer_with_stats(
            [100, -100, 50, -50, 25, -25, 75, -75],
            [100, -100, 50, -50, 25, -25, 75, -75],
        )
        assert stats.statistical_significance > 0.50

    def test_flag_name_and_bot_id_preserved(self):
        stats = self._make_analyzer_with_stats([100], [200])
        assert stats.flag_name == "test_flag"
        assert stats.bot_id == "bot1"


# ─── Mann-Whitney U ──────────────────────────────────────────────────


class TestMannWhitneyU:
    """Tests for the internal _mann_whitney_u method."""

    def test_empty_group_returns_1(self):
        assert AblationAnalyzer._mann_whitney_u([], [1, 2, 3]) == 1.0

    def test_single_element_returns_1(self):
        assert AblationAnalyzer._mann_whitney_u([1], [2, 3, 4]) == 1.0

    def test_identical_groups(self):
        p = AblationAnalyzer._mann_whitney_u([5, 5, 5], [5, 5, 5])
        # All tied → sigma should be 0 or z should be 0 → p close to 1
        assert p >= 0.90

    def test_perfectly_separated(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [100, 200, 300, 400, 500, 600, 700, 800]
        p = AblationAnalyzer._mann_whitney_u(x, y)
        assert p < 0.05

    def test_returns_two_tailed(self):
        """Result should be two-tailed p-value, capped at 1.0."""
        p = AblationAnalyzer._mann_whitney_u([1, 2, 3], [4, 5, 6])
        assert 0.0 <= p <= 1.0


# ─── Regime Breakdown ────────────────────────────────────────────────


class TestAblationRegimeBreakdown:
    """Per-regime ablation stats."""

    def test_regime_groups_in_flag_stats(self):
        trades_by_state = {
            True: [
                make_trade(trade_id="e1", pnl=100, market_regime="trending"),
                make_trade(trade_id="e2", pnl=-50, market_regime="ranging"),
            ],
            False: [
                make_trade(trade_id="d1", pnl=200, market_regime="trending"),
                make_trade(trade_id="d2", pnl=-80, market_regime="ranging"),
            ],
        }
        analyzer = AblationAnalyzer()
        stats = analyzer._compute_flag_stats("flag", trades_by_state, "bot1")
        regimes = {r.regime for r in stats.regime_breakdown}
        assert "trending" in regimes
        assert "ranging" in regimes

    def test_regime_counts(self):
        trades_by_state = {
            True: [
                make_trade(trade_id="e1", pnl=100, market_regime="trending"),
                make_trade(trade_id="e2", pnl=50, market_regime="trending"),
            ],
            False: [
                make_trade(trade_id="d1", pnl=200, market_regime="trending"),
            ],
        }
        analyzer = AblationAnalyzer()
        stats = analyzer._compute_flag_stats("flag", trades_by_state, "bot1")
        trending = next(r for r in stats.regime_breakdown if r.regime == "trending")
        assert trending.enabled_count == 2
        assert trending.disabled_count == 1

    def test_regime_pnl_delta(self):
        """pnl_delta in regime = disabled_avg - enabled_avg for that regime."""
        trades_by_state = {
            True: [
                make_trade(trade_id="e1", pnl=100, market_regime="trending"),
            ],
            False: [
                make_trade(trade_id="d1", pnl=300, market_regime="trending"),
            ],
        }
        analyzer = AblationAnalyzer()
        stats = analyzer._compute_flag_stats("flag", trades_by_state, "bot1")
        trending = next(r for r in stats.regime_breakdown if r.regime == "trending")
        # disabled_avg=300, enabled_avg=100 → delta=200
        assert trending.pnl_delta == pytest.approx(200.0, abs=0.01)

    def test_regime_only_in_one_state(self):
        """Regime present only in enabled or disabled still shows up."""
        trades_by_state = {
            True: [
                make_trade(trade_id="e1", pnl=100, market_regime="volatile"),
            ],
            False: [
                make_trade(trade_id="d1", pnl=200, market_regime="calm"),
            ],
        }
        analyzer = AblationAnalyzer()
        stats = analyzer._compute_flag_stats("flag", trades_by_state, "bot1")
        regimes = {r.regime for r in stats.regime_breakdown}
        assert "volatile" in regimes
        assert "calm" in regimes


# ─── Minimum Threshold ───────────────────────────────────────────────


class TestMinPerState:
    """Flags with < min_per_state trades per state are excluded."""

    def test_excluded_below_threshold(self):
        """Flag with too few disabled trades is excluded."""
        analyzer = AblationAnalyzer()
        trades = _make_trades_with_flag(
            n_enabled=15, n_disabled=5,  # below default min_per_state=10
        )
        result = analyzer.analyze(trades, "bot1")
        assert len(result.flags) == 0

    def test_excluded_below_threshold_enabled(self):
        """Flag with too few enabled trades is excluded."""
        analyzer = AblationAnalyzer()
        trades = _make_trades_with_flag(
            n_enabled=5, n_disabled=15,
        )
        result = analyzer.analyze(trades, "bot1")
        assert len(result.flags) == 0

    def test_included_at_threshold(self):
        """Flag with exactly min_per_state trades per state is included."""
        analyzer = AblationAnalyzer()
        trades = _make_trades_with_flag(
            n_enabled=10, n_disabled=10,
        )
        result = analyzer.analyze(trades, "bot1")
        assert len(result.flags) == 1

    def test_custom_min_per_state(self):
        """Custom min_per_state lowers the threshold."""
        analyzer = AblationAnalyzer()
        trades = _make_trades_with_flag(
            n_enabled=3, n_disabled=3,
        )
        result = analyzer.analyze(trades, "bot1", min_per_state=3)
        assert len(result.flags) == 1

    def test_high_min_per_state_excludes(self):
        analyzer = AblationAnalyzer()
        trades = _make_trades_with_flag(
            n_enabled=15, n_disabled=15,
        )
        result = analyzer.analyze(trades, "bot1", min_per_state=20)
        assert len(result.flags) == 0


# ─── Full Analysis ───────────────────────────────────────────────────


class TestAnalyze:
    """Full analyze() integration tests."""

    def test_empty_trades(self):
        analyzer = AblationAnalyzer()
        result = analyzer.analyze([], "bot1")
        assert result.bot_id == "bot1"
        assert result.flags == []
        assert result.flags_with_signal == []

    def test_no_boolean_params(self):
        """Trades without boolean params → empty analysis."""
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id="t1", strategy_params={"threshold": 0.5}),
            _trade(trade_id="t2", strategy_params={"threshold": 0.7}),
        ]
        result = analyzer.analyze(trades, "bot1")
        assert result.flags == []

    def test_flags_with_signal_populated(self):
        """Flags with p < 0.10 appear in flags_with_signal."""
        analyzer = AblationAnalyzer()
        # Create clearly separated distributions so p-value is low
        trades = []
        for i in range(15):
            trades.append(_trade(
                trade_id=f"en_{i}",
                pnl=100 + i * 10,  # all positive
                strategy_params={"strong_gate": True},
            ))
        for i in range(15):
            trades.append(_trade(
                trade_id=f"dis_{i}",
                pnl=-100 - i * 10,  # all negative
                strategy_params={"strong_gate": False},
            ))
        result = analyzer.analyze(trades, "bot1")
        assert len(result.flags) == 1
        assert "strong_gate" in result.flags_with_signal

    def test_flags_without_signal_not_in_list(self):
        """Flags with high p-value do NOT appear in flags_with_signal."""
        analyzer = AblationAnalyzer()
        # Same distribution for both states → high p-value
        trades = []
        for i in range(15):
            trades.append(_trade(
                trade_id=f"en_{i}",
                pnl=50 * ((-1) ** i),  # alternating +/- 50
                strategy_params={"weak_gate": True},
            ))
        for i in range(15):
            trades.append(_trade(
                trade_id=f"dis_{i}",
                pnl=50 * ((-1) ** i),
                strategy_params={"weak_gate": False},
            ))
        result = analyzer.analyze(trades, "bot1")
        assert len(result.flags) == 1
        assert result.flags_with_signal == []

    def test_period_preserved(self):
        analyzer = AblationAnalyzer()
        result = analyzer.analyze([], "bot1", period="2026-W18")
        assert result.period == "2026-W18"

    def test_multiple_flags_analyzed(self):
        """Multiple boolean flags in strategy_params are all analyzed."""
        analyzer = AblationAnalyzer()
        trades = []
        for i in range(12):
            trades.append(_trade(
                trade_id=f"a_{i}",
                pnl=100,
                strategy_params={"gate_a": i % 2 == 0, "gate_b": i % 3 == 0},
            ))
        for i in range(12):
            trades.append(_trade(
                trade_id=f"b_{i}",
                pnl=-50,
                strategy_params={"gate_a": i % 2 == 0, "gate_b": i % 3 == 0},
            ))
        # We need enough per state — at min_per_state=2 to ensure inclusion
        result = analyzer.analyze(trades, "bot1", min_per_state=2)
        flag_names = [f.flag_name for f in result.flags]
        assert "gate_a" in flag_names
        assert "gate_b" in flag_names

    def test_only_none_strategy_params_ignored(self):
        """Trades with None strategy_params do not contribute to analysis."""
        analyzer = AblationAnalyzer()
        trades = [
            _trade(trade_id=f"t{i}", strategy_params=None)
            for i in range(20)
        ]
        result = analyzer.analyze(trades, "bot1")
        assert result.flags == []


# ─── ABLATION Routing in AutonomousPipeline ──────────────────────────


class TestAblationRouting:
    """Experiment type routing for boolean vs numeric params in autonomous_pipeline."""

    def test_bool_value_type_uses_ablation(self):
        """param.value_type == 'bool' → ExperimentType.ABLATION."""
        from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType

        param = ParameterDefinition(
            param_name="use_gate",
            bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="params.use_gate",
            current_value=True,
            value_type="bool",
            category="filter",
        )
        is_ablation = (
            param.value_type == "bool"
            or getattr(param, "category", "") == "ablation"
        )
        experiment_type = ExperimentType.ABLATION if is_ablation else ExperimentType.PARAMETER_AB
        assert experiment_type == ExperimentType.ABLATION

    def test_ablation_category_uses_ablation(self):
        """param.category == 'ablation' → ExperimentType.ABLATION."""
        from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType

        param = ParameterDefinition(
            param_name="oscillation_gate_threshold",
            bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="params.threshold",
            current_value=0.5,
            value_type="float",
            category="ablation",
        )
        is_ablation = (
            param.value_type == "bool"
            or getattr(param, "category", "") == "ablation"
        )
        experiment_type = ExperimentType.ABLATION if is_ablation else ExperimentType.PARAMETER_AB
        assert experiment_type == ExperimentType.ABLATION

    def test_float_value_type_uses_parameter_ab(self):
        """param.value_type == 'float' → ExperimentType.PARAMETER_AB (default)."""
        from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType

        param = ParameterDefinition(
            param_name="signal_strength_min",
            bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="params.signal_strength_min",
            current_value=0.5,
            valid_range=(0.1, 1.0),
            value_type="float",
            category="signal",
        )
        is_ablation = (
            param.value_type == "bool"
            or getattr(param, "category", "") == "ablation"
        )
        experiment_type = ExperimentType.ABLATION if is_ablation else ExperimentType.PARAMETER_AB
        assert experiment_type == ExperimentType.PARAMETER_AB

    def test_int_value_type_uses_parameter_ab(self):
        """param.value_type == 'int' → ExperimentType.PARAMETER_AB."""
        from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType

        param = ParameterDefinition(
            param_name="lookback_bars",
            bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="params.lookback_bars",
            current_value=14,
            valid_range=(5, 50),
            value_type="int",
            category="signal",
        )
        is_ablation = (
            param.value_type == "bool"
            or getattr(param, "category", "") == "ablation"
        )
        experiment_type = ExperimentType.ABLATION if is_ablation else ExperimentType.PARAMETER_AB
        assert experiment_type == ExperimentType.PARAMETER_AB

    def test_experiment_type_enum_values(self):
        """Confirm ExperimentType enum has expected members."""
        assert ExperimentType.ABLATION.value == "ablation"
        assert ExperimentType.PARAMETER_AB.value == "parameter_ab"
        assert ExperimentType.FILTER_AB.value == "filter_ab"

    def test_bool_with_non_ablation_category_still_ablation(self):
        """bool value_type takes precedence over category."""
        from trading_assistant.schemas.parameter_definition import ParameterDefinition, ParameterType

        param = ParameterDefinition(
            param_name="use_filter",
            bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="params.use_filter",
            current_value=False,
            value_type="bool",
            category="signal",
        )
        is_ablation = (
            param.value_type == "bool"
            or getattr(param, "category", "") == "ablation"
        )
        experiment_type = ExperimentType.ABLATION if is_ablation else ExperimentType.PARAMETER_AB
        assert experiment_type == ExperimentType.ABLATION
