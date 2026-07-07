"""Interaction analysis schemas — coordinator events and cross-strategy effects.

Used by skills/interaction_analyzer.py to assess coordinator benefit on swing_multi_01.
"""
from __future__ import annotations

from pydantic import BaseModel


class CoordinatorAction(BaseModel):
    """A single coordinator action event (parsed from coordination JSONL)."""

    timestamp: str = ""
    action: str  # "tighten_stop_be" | "size_boost" | "overlay_signal_change"
    trigger_strategy: str = ""
    target_strategy: str = ""
    symbol: str = ""
    rule: str = ""
    details: dict = {}
    outcome: str = ""  # "applied" | "skipped_already_tighter" | "emitted"


class InteractionEffect(BaseModel):
    """Aggregated effect of a coordinator rule on trades."""

    rule: str  # "rule_1" | "rule_2" | "ema_crossover"
    action_type: str
    trigger_strategy: str
    target_strategy: str
    action_count: int = 0
    affected_trades: int = 0
    estimated_pnl_impact: float = 0.0  # positive = net benefit
    estimated_pnl_without: float = 0.0  # counterfactual
    net_benefit: float = 0.0
    confidence: float = 0.0


class InteractionReport(BaseModel):
    """Complete coordinator interaction analysis for a week."""

    week_start: str
    week_end: str
    bot_id: str = "swing_multi_01"  # currently only swing_multi_01 has coordinator
    total_coordination_events: int = 0
    effects: list[InteractionEffect] = []
    overlay_regime_summary: dict = {}  # regime → {trades, pnl, win_rate}
    net_coordinator_benefit: float = 0.0
    recommendation: str = ""
