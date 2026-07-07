"""Daily metrics schemas — Pydantic models for reduced/curated daily data.

These define the output format of the data reduction pipeline (skills/build_daily_metrics.py).
Claude sees these pre-scored, pre-classified structures — not raw trade data.
"""
from __future__ import annotations

from pydantic import BaseModel, computed_field


class PerStrategySummary(BaseModel):
    """Per-strategy daily performance breakdown."""

    strategy_id: str
    trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_entry_slippage_bps: float | None = None
    avg_mfe_pct: float | None = None
    avg_mae_pct: float | None = None
    avg_exit_efficiency: float | None = None
    symbols_traded: list[str] = []


class BotDailySummary(BaseModel):
    """Aggregated daily stats for a single bot."""

    date: str  # YYYY-MM-DD
    bot_id: str
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_rolling_30d: float = 0.0
    sortino_rolling_30d: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    exposure_pct: float = 0.0
    missed_count: int = 0
    missed_would_have_won: int = 0
    error_count: int = 0
    uptime_pct: float = 100.0
    avg_process_quality: float = 100.0
    calmar_rolling_30d: float = 0.0
    per_strategy_summary: dict[str, PerStrategySummary] = {}
    lineage_summary: dict | None = None
    lineage_gap: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades

    @computed_field  # type: ignore[prop-decorator]
    @property
    def profit_factor(self) -> float:
        total_wins = self.avg_win * self.win_count
        total_losses = abs(self.avg_loss) * self.loss_count
        if total_losses == 0:
            return float("inf") if total_wins > 0 else 0.0
        return total_wins / total_losses


class WinnerLoserRecord(BaseModel):
    """A single notable winning or losing trade with full context."""

    trade_id: str
    bot_id: str
    pair: str
    side: str
    pnl: float
    pnl_pct: float = 0.0
    entry_signal: str = ""
    exit_reason: str = ""
    market_regime: str = ""
    process_quality_score: int = 100
    root_causes: list[str] = []
    entry_price: float = 0.0
    exit_price: float = 0.0
    atr_at_entry: float = 0.0


class ProcessFailureRecord(BaseModel):
    """A trade with process_quality_score < 60 — flagged for review."""

    trade_id: str
    bot_id: str
    pair: str
    process_quality_score: int
    root_causes: list[str]
    pnl: float
    entry_signal: str = ""
    market_regime: str = ""


class NotableMissedRecord(BaseModel):
    """A missed opportunity where outcome > 2× avg win."""

    bot_id: str
    pair: str
    signal: str
    blocked_by: str = ""
    hypothetical_entry: float = 0.0
    outcome_24h: float = 0.0
    confidence: float = 0.0
    assumption_tags: list[str] = []


class RegimeAnalysis(BaseModel):
    """PnL breakdown by market regime for one bot on one day."""

    bot_id: str
    date: str
    regime_pnl: dict[str, float] = {}
    regime_trade_count: dict[str, int] = {}
    regime_win_rate: dict[str, float] = {}


class FilterAnalysis(BaseModel):
    """Impact analysis for each filter: how many trades blocked, net PnL effect."""

    bot_id: str
    date: str
    filter_block_counts: dict[str, int] = {}
    filter_saved_pnl: dict[str, float] = {}
    filter_missed_pnl: dict[str, float] = {}


class AnomalyRecord(BaseModel):
    """A statistically unusual event detected in today's data."""

    bot_id: str
    date: str
    anomaly_type: str
    description: str
    severity: str = "medium"  # low | medium | high
    related_trades: list[str] = []


class RootCauseSummary(BaseModel):
    """Distribution of root causes across all trades for one bot on one day."""

    bot_id: str
    date: str
    distribution: dict[str, int] = {}
    total_trades: int = 0


class MacroRegimeAnalysis(BaseModel):
    """Daily performance breakdown by macro regime (G/R/S/D)."""

    bot_id: str
    date: str
    macro_regime: str = ""  # active macro regime for the day
    regime_confidence: float = 0.0
    stress_level: float = 0.0
    applied_config: dict = {}  # directional_cap_R, unit_risk_mult, etc.
    regime_pnl_30d: float = 0.0  # trailing 30d PnL while in this regime
    regime_trade_count_30d: int = 0
    regime_win_rate_30d: float = 0.0
