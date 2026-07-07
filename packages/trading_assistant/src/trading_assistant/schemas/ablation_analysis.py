"""Ablation analysis schemas — statistical comparison of boolean flag on/off states.

Enables analysis like "disabling fade_oscillation_gate improves win rate from 28%
to 41% (n=47, p=0.03)" instead of diluting booleans into 3-bucket numeric splits.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AblationRegimeStats(BaseModel):
    """Per-regime stats for an ablation flag."""

    regime: str
    enabled_count: int = 0
    disabled_count: int = 0
    enabled_win_rate: float = 0.0
    disabled_win_rate: float = 0.0
    pnl_delta: float = 0.0  # disabled_avg_pnl - enabled_avg_pnl


class AblationFlagStats(BaseModel):
    """Statistics for a single boolean ablation flag."""

    flag_name: str
    strategy_id: str = ""
    bot_id: str = ""
    enabled_count: int = 0
    disabled_count: int = 0
    enabled_win_rate: float = 0.0
    disabled_win_rate: float = 0.0
    enabled_avg_pnl: float = 0.0
    disabled_avg_pnl: float = 0.0
    pnl_delta: float = 0.0  # disabled_avg_pnl - enabled_avg_pnl
    statistical_significance: float = 1.0  # p-value from Mann-Whitney U
    regime_breakdown: list[AblationRegimeStats] = Field(default_factory=list)


class AblationAnalysis(BaseModel):
    """Full ablation analysis for a bot on a given period."""

    bot_id: str
    period: str = ""
    flags: list[AblationFlagStats] = Field(default_factory=list)
    flags_with_signal: list[str] = Field(default_factory=list)  # flags where p < 0.10 and both states have >= min trades
