# schemas/simulation_policy.py
"""Simulation policy schemas — per-bot configuration for missed opportunity outcome computation."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class FillModel(str, Enum):
    """How the hypothetical entry price is determined."""
    MID_PRICE = "mid_price"
    ASK_FOR_LONG = "ask_for_long"
    WORST_CASE = "worst_case"
    HYPOTHETICAL = "hypothetical"


class TPSLMethod(str, Enum):
    """Take-profit / stop-loss computation method."""
    FIXED_PCT = "fixed_pct"
    ATR_MULTIPLE = "atr_multiple"
    RISK_REWARD = "risk_reward"
    NONE = "none"


class TPSLConfig(BaseModel):
    """Take-profit / stop-loss configuration."""
    method: TPSLMethod = TPSLMethod.NONE
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    atr_tp_multiple: float = 0.0
    atr_sl_multiple: float = 0.0
    risk_reward_ratio: float = 0.0


class SimulationPolicy(BaseModel):
    """Per-bot configuration for how missed opportunity outcomes are computed."""
    bot_id: str
    strategy_id: str = ""
    fill_model: FillModel = FillModel.HYPOTHETICAL
    slippage_bps: float = 5.0
    fees_bps: float = 7.0
    default_position_size: float = 1.0
    tpsl: TPSLConfig = TPSLConfig()
    outcome_horizons: list[str] = ["1h", "4h", "24h"]
    confidence_threshold: float = 0.0
