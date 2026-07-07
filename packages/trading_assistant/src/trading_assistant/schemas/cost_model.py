"""Shared transaction cost model configuration schemas."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SlippageModel(str, Enum):
    FIXED = "fixed"
    SPREAD_PROPORTIONAL = "spread_proportional"
    EMPIRICAL = "empirical"


class CostModelConfig(BaseModel):
    fees_per_trade_bps: float = 7.0
    slippage_model: SlippageModel = SlippageModel.FIXED
    fixed_slippage_bps: float = 5.0
    slippage_source: str = ""
    spread_impact: bool = True
    reject_if_only_profitable_at_zero_cost: bool = True
    cost_sensitivity_test: bool = True
    cost_multipliers: list[float] = Field(default_factory=lambda: [1.0, 1.5, 2.0])


__all__ = ["CostModelConfig", "SlippageModel"]
