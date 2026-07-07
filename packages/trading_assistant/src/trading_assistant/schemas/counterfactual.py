"""Counterfactual simulation schemas — what-if trade replay."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field


class ScenarioType(str, Enum):
    REMOVE_FILTER = "remove_filter"
    ADD_REGIME_GATE = "add_regime_gate"
    EXCLUDE_TRADES = "exclude_trades"


class CounterfactualScenario(BaseModel):
    scenario_type: ScenarioType
    description: str
    parameters: dict = {}


class CounterfactualResult(BaseModel):
    scenario: CounterfactualScenario
    baseline_pnl: float = 0.0
    modified_pnl: float = 0.0
    baseline_trade_count: int = 0
    modified_trade_count: int = 0
    baseline_win_rate: float = 0.0
    modified_win_rate: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delta_pnl(self) -> float:
        return self.modified_pnl - self.baseline_pnl

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delta_win_rate(self) -> float:
        return self.modified_win_rate - self.baseline_win_rate
