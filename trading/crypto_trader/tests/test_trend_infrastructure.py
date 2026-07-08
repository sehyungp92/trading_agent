"""Tests for infrastructure changes supporting multi-strategy."""

import pytest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from crypto_trader.strategy.trend.config import TrendConfig
from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations


class TestConfigMutatorGeneric:
    def test_apply_mutations_trend_config(self):
        """config_mutator works with TrendConfig (not just MomentumConfig)."""
        cfg = TrendConfig()
        mutated = apply_mutations(cfg, {"regime.a_min_adx": 25.0})
        assert mutated.regime.a_min_adx == 25.0
        assert cfg.regime.a_min_adx == 12.0  # Original unchanged (baked from round 1)

    def test_apply_multiple_mutations(self):
        cfg = TrendConfig()
        mutated = apply_mutations(cfg, {
            "regime.a_min_adx": 22.0,
            "stops.atr_mult": 1.5,
            "exits.tp1_r": 1.2,
        })
        assert mutated.regime.a_min_adx == 22.0
        assert mutated.stops.atr_mult == 1.5
        assert mutated.exits.tp1_r == 1.2

    def test_invalid_section_raises(self):
        cfg = TrendConfig()
        with pytest.raises(ValueError, match="Unknown config section"):
            apply_mutations(cfg, {"nonexistent.field": 1})

    def test_invalid_field_raises(self):
        cfg = TrendConfig()
        with pytest.raises(ValueError, match="Unknown field"):
            apply_mutations(cfg, {"regime.nonexistent_field": 1})

    def test_empty_mutations_returns_copy(self):
        cfg = TrendConfig()
        mutated = apply_mutations(cfg, {})
        assert mutated is not cfg
        assert mutated.regime.a_min_adx == cfg.regime.a_min_adx


class TestRunnerStrategyType:
    def test_create_strategy_momentum(self):
        """Default strategy_type='momentum' creates MomentumStrategy."""
        from crypto_trader.backtest.runner import _create_strategy
        from crypto_trader.strategy.momentum.config import MomentumConfig
        from crypto_trader.core.models import TimeFrame

        cfg = MomentumConfig()
        strategy, tfs, primary = _create_strategy("momentum", cfg)
        assert strategy.name == "momentum_pullback"
        assert primary == TimeFrame.M15
        assert TimeFrame.M15 in tfs

    def test_create_strategy_trend(self):
        """strategy_type='trend' creates TrendStrategy."""
        from crypto_trader.backtest.runner import _create_strategy
        from crypto_trader.core.models import TimeFrame

        cfg = TrendConfig()
        strategy, tfs, primary = _create_strategy("trend", cfg)
        assert strategy.name == "trend_anchor"
        assert primary == TimeFrame.M15
        assert TimeFrame.D1 in tfs
        assert TimeFrame.H1 in tfs
        assert TimeFrame.M15 in tfs


class TestTrendConfigSerialization:
    def test_to_dict_from_dict_all_sections(self):
        """All 12 sections round-trip correctly."""
        cfg = TrendConfig()
        d = cfg.to_dict()
        cfg2 = TrendConfig.from_dict(d)

        assert cfg2.h1_indicators.ema_fast == cfg.h1_indicators.ema_fast
        assert cfg2.d1_indicators.ema_fast == cfg.d1_indicators.ema_fast
        assert cfg2.regime.a_min_adx == cfg.regime.a_min_adx
        assert cfg2.setup.impulse_min_atr_move == cfg.setup.impulse_min_atr_move
        assert cfg2.confirmation.enable_engulfing == cfg.confirmation.enable_engulfing
        assert cfg2.entry.entry_on_close == cfg.entry.entry_on_close
        assert cfg2.stops.atr_mult == cfg.stops.atr_mult
        assert cfg2.exits.tp1_r == cfg.exits.tp1_r
        assert cfg2.trail.trail_r_ceiling == cfg.trail.trail_r_ceiling
        assert cfg2.risk.risk_pct_a == cfg.risk.risk_pct_a
        assert cfg2.limits.max_concurrent_positions == cfg.limits.max_concurrent_positions
        assert cfg2.filters.funding_filter_enabled == cfg.filters.funding_filter_enabled
        assert cfg2.reentry.enabled == cfg.reentry.enabled
        assert cfg2.symbol_filter.btc_direction == cfg.symbol_filter.btc_direction

    def test_from_dict_with_strategy_wrapper(self):
        """CLI loads configs with 'strategy' wrapper key."""
        cfg = TrendConfig()
        wrapped = {"strategy": cfg.to_dict()}
        cfg2 = TrendConfig.from_dict(wrapped.get("strategy", {}))
        assert cfg2.regime.a_min_adx == cfg.regime.a_min_adx


class TestTrendPluginDiagnostics:
    """Tests for trend plugin diagnostic enhancements."""

    def test_phase_scoring_emphasis_all_phases(self):
        from crypto_trader.optimize.trend_plugin import PHASE_SCORING_EMPHASIS, SCORING_WEIGHTS
        for phase in range(1, 7):
            weights = PHASE_SCORING_EMPHASIS.get(phase, SCORING_WEIGHTS)
            assert abs(sum(weights.values()) - 1.0) < 0.01, f"Phase {phase} weights don't sum to 1.0"
            assert set(weights.keys()) == set(SCORING_WEIGHTS.keys()), f"Phase {phase} key mismatch"

    def test_phase_diagnostic_modules_coverage(self):
        from crypto_trader.optimize.trend_plugin import PHASE_DIAGNOSTIC_MODULES
        for phase in range(1, 7):
            modules = PHASE_DIAGNOSTIC_MODULES[phase]
            assert isinstance(modules, list)
            assert len(modules) >= 1
            assert "D6" in modules  # Overview always included
        # Phase 6 should have all modules
        assert len(PHASE_DIAGNOSTIC_MODULES[6]) == 6

    def test_phase_gate_criteria_structure(self):
        from crypto_trader.optimize.trend_plugin import PHASE_GATE_CRITERIA
        from crypto_trader.optimize.types import GateCriterion
        for phase, criteria in PHASE_GATE_CRITERIA.items():
            assert isinstance(criteria, list)
            for gc in criteria:
                assert isinstance(gc, GateCriterion)
                assert gc.metric in (
                    "total_trades",
                    "max_drawdown_pct",
                    "profit_factor",
                    "exit_efficiency",
                )

    def test_build_gate_criteria_phase_specific(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin, PHASE_GATE_CRITERIA
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        # Phase 1 has specific criteria aligned with the stricter round objective.
        criteria = plugin._build_gate_criteria(1)
        thresholds = {gc.metric: gc.threshold for gc in criteria}
        assert thresholds["total_trades"] == 30

        # Phase 2 also has specific criteria
        criteria2 = plugin._build_gate_criteria(2)
        thresholds2 = {gc.metric: gc.threshold for gc in criteria2}
        assert thresholds2["total_trades"] == 30  # From PHASE_GATE_CRITERIA

    def test_get_phase_spec_has_correct_fields(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin, PHASE_SCORING_EMPHASIS
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        spec = plugin.get_phase_spec(1, None)
        assert spec.phase_num == 1
        assert spec.name == "Signal & Setup"
        assert spec.scoring_weights == PHASE_SCORING_EMPHASIS[1]
        assert len(spec.gate_criteria) >= 1
        assert spec.gate_criteria_fn is not None
        assert spec.analysis_policy.diagnostic_gap_fn is not None
        assert spec.analysis_policy.suggest_experiments_fn is not None
        assert spec.analysis_policy.build_extra_analysis_fn is not None
        assert spec.analysis_policy.format_extra_analysis_fn is not None

    def test_suggest_experiments_fn_signature(self):
        """suggest_experiments_fn must accept 4 args (phase, metrics, weaknesses, strengths)."""
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        # Should not raise with 4 args
        result = plugin._suggest_experiments_fn(1, {"total_trades": 5}, ["low trades"], [])
        assert isinstance(result, list)

    def test_diagnostic_gap_fn_basic(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        gaps = plugin._diagnostic_gap_fn(1, {
            "total_trades": 5, "max_drawdown_pct": 40, "profit_factor": 0.8,
        })
        assert len(gaps) >= 2  # low trades + high DD + low PF

    def test_diagnostic_gap_fn_uses_ratio_capture_thresholds(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date

        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        plugin._last_result = MagicMock(trades=[MagicMock()] * 5)

        mock_insights = MagicMock()
        mock_insights.mfe_capture = {"avg_capture_pct": 0.35, "avg_giveback_pct": 0.65}
        mock_insights.per_confirmation = {}
        mock_insights.concentration = {}
        mock_insights.direction = {}
        mock_insights.duration = {}
        mock_insights.exit_attribution = {}

        with patch(
            "crypto_trader.backtest.diagnostics.extract_diagnostic_insights",
            return_value=mock_insights,
        ):
            gaps = plugin._diagnostic_gap_fn(3, {"total_trades": 5})

        assert any("35%" in gap for gap in gaps)
        assert any("65%" in gap for gap in gaps)

    def test_build_extra_analysis_fn_returns_dict(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        # Without _last_result, returns empty dict
        result = plugin._build_extra_analysis_fn(1, {}, [], [])
        assert result == {}

    def test_format_extra_analysis_fn(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        text = plugin._format_extra_analysis_fn({"avg_mfe_r": 1.5, "capture_pct": 45.0})
        assert "avg_mfe_r" in text
        assert "1.5000" in text

    def test_format_extra_analysis_fn_nested(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        text = plugin._format_extra_analysis_fn({
            "exit_reasons": {"protective_stop": 5, "tp1": 3},
        })
        assert "exit_reasons:" in text
        assert "protective_stop" in text

    def test_format_extra_analysis_fn_empty(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        assert plugin._format_extra_analysis_fn({}) == ""

    def test_build_verdict_actionable(self):
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        from crypto_trader.backtest.config import BacktestConfig
        from crypto_trader.backtest.metrics import PerformanceMetrics
        from datetime import date
        plugin = TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        pm = MagicMock(spec=PerformanceMetrics)
        pm.total_trades = 5
        pm.win_rate = 40.0
        pm.profit_factor = 0.8
        pm.sharpe_ratio = 1.0
        pm.max_drawdown_pct = 35.0
        pm.net_return_pct = -2.0
        verdict = plugin._build_verdict(pm)
        assert "Loosen regime" in verdict
        assert "Tighten trail" in verdict
        assert "signal quality" in verdict


class TestRound6TrendPlugin:
    def test_round6_phase_spec_uses_round6_weights(self):
        from crypto_trader.backtest.config import BacktestConfig
        from crypto_trader.optimize.trend_round6_plugin import (
            ROUND6_PHASE_NAMES,
            ROUND6_PHASE_SCORING_EMPHASIS,
            ROUND6_SCORING_WEIGHTS,
            Round6TrendPlugin,
        )

        plugin = Round6TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        spec = plugin.get_phase_spec(4, None)

        assert spec.name == ROUND6_PHASE_NAMES[4]
        assert spec.scoring_weights == ROUND6_PHASE_SCORING_EMPHASIS[4]
        assert abs(sum(ROUND6_SCORING_WEIGHTS.values()) - 1.0) < 1e-9


class TestRound7TrendPlugin:
    def test_round7_phase_spec_uses_round7_weights(self):
        from crypto_trader.backtest.config import BacktestConfig
        from crypto_trader.optimize.trend_round7_plugin import (
            ROUND7_PHASE_NAMES,
            ROUND7_PHASE_SCORING_EMPHASIS,
            ROUND7_SCORING_WEIGHTS,
            Round7TrendPlugin,
        )

        plugin = Round7TrendPlugin(
            BacktestConfig(date(2026, 3, 1), date(2026, 4, 1), ["BTC"]),
            TrendConfig(),
        )
        spec = plugin.get_phase_spec(1, None)

        assert spec.name == ROUND7_PHASE_NAMES[1]
        assert spec.scoring_weights == ROUND7_PHASE_SCORING_EMPHASIS[1]
        assert abs(sum(ROUND7_SCORING_WEIGHTS.values()) - 1.0) < 1e-9

    def test_confirmation_disable_map_keys(self):
        from crypto_trader.optimize.trend_plugin import CONFIRMATION_DISABLE_MAP
        # All values should be valid config paths
        for ctype, path in CONFIRMATION_DISABLE_MAP.items():
            assert path.startswith("confirmation.enable_")


class TestParallelStrategyRouting:
    def test_deserialize_config_momentum(self):
        from crypto_trader.optimize.parallel import _deserialize_config
        from crypto_trader.strategy.momentum.config import MomentumConfig

        cfg = MomentumConfig()
        d = cfg.to_dict()
        result = _deserialize_config(d, "momentum")
        assert isinstance(result, MomentumConfig)

    def test_deserialize_config_trend(self):
        from crypto_trader.optimize.parallel import _deserialize_config

        cfg = TrendConfig()
        d = cfg.to_dict()
        result = _deserialize_config(d, "trend")
        assert isinstance(result, TrendConfig)
