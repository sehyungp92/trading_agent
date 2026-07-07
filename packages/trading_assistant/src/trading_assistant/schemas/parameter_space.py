"""Shared parameter search-space and robustness configuration schemas."""
from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class OptimizationObjective(str, Enum):
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    PROFIT_FACTOR = "profit_factor"


class OptimizationConfig(BaseModel):
    objective: OptimizationObjective = OptimizationObjective.CALMAR
    secondary: str = "max_drawdown"
    max_drawdown_constraint: float = 0.15


class RobustnessConfig(BaseModel):
    neighborhood_test: bool = True
    regime_stability: bool = True
    min_trades_per_fold: int = 30
    neighborhood_pct: float = 0.1
    min_profitable_regimes: int = 3
    total_regime_types: int = 4


class ParameterDef(BaseModel):
    """A single tunable parameter with its search range."""

    name: str
    min_value: float
    max_value: float
    step: float
    current_value: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grid_values(self) -> list[float]:
        if self.step <= 0:
            return [self.min_value]
        values: list[float] = []
        v = self.min_value
        while v <= self.max_value + self.step * 1e-9:
            values.append(round(v, 10))
            v += self.step
        return values


class ParameterSpace(BaseModel):
    """The full parameter search space for a bot."""

    bot_id: str
    parameters: list[ParameterDef] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_combinations(self) -> int:
        if not self.parameters:
            return 0
        return math.prod(len(p.grid_values) for p in self.parameters)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_params(self) -> dict[str, float]:
        return {p.name: p.current_value for p in self.parameters}


__all__ = [
    "OptimizationConfig",
    "OptimizationObjective",
    "ParameterDef",
    "ParameterSpace",
    "RobustnessConfig",
]
