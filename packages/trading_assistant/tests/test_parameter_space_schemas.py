"""Tests for shared parameter-space and cost-model schemas."""

from trading_assistant.schemas.cost_model import CostModelConfig, SlippageModel
from trading_assistant.schemas.parameter_space import (
    OptimizationConfig,
    OptimizationObjective,
    ParameterDef,
    ParameterSpace,
    RobustnessConfig,
)


class TestCostModelConfig:
    def test_defaults(self):
        config = CostModelConfig()
        assert config.fees_per_trade_bps == 7.0
        assert config.slippage_model == SlippageModel.FIXED
        assert config.cost_multipliers == [1.0, 1.5, 2.0]

    def test_empirical_slippage_source(self):
        config = CostModelConfig(
            slippage_model=SlippageModel.EMPIRICAL,
            slippage_source="slippage.json",
        )
        assert config.slippage_model == SlippageModel.EMPIRICAL
        assert config.slippage_source == "slippage.json"


class TestParameterSpace:
    def test_grid_values_include_max_value(self):
        param = ParameterDef(
            name="signal_strength_min",
            min_value=0.1,
            max_value=0.3,
            step=0.1,
            current_value=0.2,
        )

        assert param.grid_values == [0.1, 0.2, 0.3]

    def test_total_combinations_and_current_params(self):
        space = ParameterSpace(
            bot_id="bot_alpha",
            parameters=[
                ParameterDef(name="a", min_value=1, max_value=3, step=1, current_value=2),
                ParameterDef(name="b", min_value=0.1, max_value=0.2, step=0.1, current_value=0.1),
            ],
        )

        assert space.total_combinations == 6
        assert space.current_params == {"a": 2, "b": 0.1}

    def test_empty_space_has_no_combinations(self):
        space = ParameterSpace(bot_id="bot_alpha")
        assert space.total_combinations == 0
        assert space.current_params == {}


class TestOptimizationAndRobustnessConfig:
    def test_optimization_defaults_to_calmar(self):
        config = OptimizationConfig()
        assert config.objective == OptimizationObjective.CALMAR
        assert config.max_drawdown_constraint == 0.15

    def test_robustness_defaults(self):
        config = RobustnessConfig()
        assert config.neighborhood_test is True
        assert config.regime_stability is True
        assert config.min_profitable_regimes == 3
