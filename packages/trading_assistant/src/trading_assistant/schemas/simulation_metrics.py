"""Shared simulation and replay performance metrics."""
from __future__ import annotations

from pydantic import BaseModel, computed_field


class SimulationMetrics(BaseModel):
    """Performance metrics from a simulation or replay run."""

    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades_by_regime: dict[str, int] = {}
    pnl_by_regime: dict[str, float] = {}
    daily_pnl: dict[str, float] = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expectancy(self) -> float:
        """Return win-rate weighted average win/loss expectancy."""
        if self.avg_loss == 0 or self.total_trades == 0:
            return 0.0
        return self.win_rate * (self.avg_win / self.avg_loss)


__all__ = ["SimulationMetrics"]
