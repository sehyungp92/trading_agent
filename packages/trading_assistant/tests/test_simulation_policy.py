# tests/test_simulation_policy.py
"""Tests for simulation policy schemas."""
from trading_assistant.schemas.simulation_policy import (
    FillModel,
    SimulationPolicy,
    TPSLConfig,
    TPSLMethod,
)


class TestSimulationPolicySchemas:
    def test_fill_model_enum_values(self):
        assert FillModel.MID_PRICE.value == "mid_price"
        assert FillModel.HYPOTHETICAL.value == "hypothetical"
        assert FillModel.ASK_FOR_LONG.value == "ask_for_long"
        assert FillModel.WORST_CASE.value == "worst_case"

    def test_tpsl_method_enum_values(self):
        assert TPSLMethod.FIXED_PCT.value == "fixed_pct"
        assert TPSLMethod.ATR_MULTIPLE.value == "atr_multiple"
        assert TPSLMethod.NONE.value == "none"

    def test_policy_defaults(self):
        policy = SimulationPolicy(bot_id="bot1")
        assert policy.fill_model == FillModel.HYPOTHETICAL
        assert policy.slippage_bps == 5.0
        assert policy.fees_bps == 7.0
        assert policy.tpsl.method == TPSLMethod.NONE

    def test_policy_custom_values(self):
        policy = SimulationPolicy(
            bot_id="bot1",
            strategy_id="mean_revert",
            fill_model=FillModel.ASK_FOR_LONG,
            slippage_bps=10.0,
            fees_bps=5.0,
            tpsl=TPSLConfig(method=TPSLMethod.FIXED_PCT, tp_pct=2.0, sl_pct=1.0),
        )
        assert policy.fill_model == FillModel.ASK_FOR_LONG
        assert policy.tpsl.tp_pct == 2.0
