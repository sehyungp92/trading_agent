# schemas/synergy_analysis.py
"""Synergy analysis schemas — cross-strategy redundancy and complementarity scoring."""
from __future__ import annotations

from pydantic import BaseModel


class StrategyPairAnalysis(BaseModel):
    """Pairwise analysis of two strategies across bots."""

    strategy_a: str  # "bot_id:strategy_id"
    strategy_b: str
    correlation_30d: float = 0.0
    correlation_60d: float | None = None
    correlation_90d: float | None = None
    classification: str = "neutral"  # redundant | complementary | neutral | cannibalistic
    diversification_benefit: float = 0.0  # marginal Sharpe contribution
    signal_overlap_pct: float = 0.0  # % of entries within same day
    same_instrument: bool = False
    recommendation: str = ""


class StrategyMarginalContribution(BaseModel):
    """Marginal contribution of a strategy to portfolio Sharpe."""

    strategy_key: str  # "bot_id:strategy_id"
    bot_id: str
    strategy_id: str
    sharpe_with: float = 0.0  # portfolio Sharpe including this strategy
    sharpe_without: float = 0.0  # portfolio Sharpe excluding this strategy
    marginal_sharpe: float = 0.0  # with - without
    pnl_contribution_pct: float = 0.0


class SynergyReport(BaseModel):
    """Full synergy analysis report."""

    week_start: str
    week_end: str
    strategy_pairs: list[StrategyPairAnalysis] = []
    marginal_contributions: list[StrategyMarginalContribution] = []
    redundant_pairs: list[str] = []  # "stratA vs stratB" summaries
    complementary_pairs: list[str] = []
    total_strategies: int = 0
