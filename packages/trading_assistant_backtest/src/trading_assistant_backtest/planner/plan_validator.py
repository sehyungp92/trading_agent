"""Plan validation wrapper."""

from __future__ import annotations

from trading_assistant_backtest.contract_models import OptimizerExperimentPlan


def validate_plan(payload: dict) -> OptimizerExperimentPlan:
    return OptimizerExperimentPlan.model_validate(payload)
