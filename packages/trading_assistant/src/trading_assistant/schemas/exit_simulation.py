"""Exit strategy simulation schemas."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field


class ExitStrategyType(str, Enum):
    FIXED_STOP = "fixed_stop"
    TRAILING_STOP = "trailing_stop"
    ATR_STOP = "atr_stop"
    TIME_BASED = "time_based"


class ExitStrategyConfig(BaseModel):
    strategy_type: ExitStrategyType
    params: dict = {}


class TradeExitComparison(BaseModel):
    trade_id: str
    actual_pnl: float
    simulated_pnl: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def improvement(self) -> float:
        return self.simulated_pnl - self.actual_pnl


class ExitSimulationResult(BaseModel):
    strategy: ExitStrategyConfig
    total_trades: int = 0
    trades_with_data: int = 0
    baseline_pnl: float = 0.0
    simulated_pnl: float = 0.0
    comparisons: list[TradeExitComparison] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def improvement(self) -> float:
        return self.simulated_pnl - self.baseline_pnl


class ExitSweepResult(BaseModel):
    bot_id: str
    configs_tested: int = 0
    baseline_pnl: float = 0.0
    results: list[ExitSimulationResult] = []
    best_strategy: ExitStrategyConfig | None = None
    best_improvement: float = 0.0
