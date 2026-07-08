"""Strategy ID aliases shared by bot telemetry and assistant ingestion."""

from __future__ import annotations


ASSISTANT_STRATEGY_IDS = {
    "momentum": "MomentumPullback_M15",
    "trend": "InstitutionalAnchor_H1",
    "breakout": "VolumeProfileBreakout_M30",
}


def assistant_strategy_id(strategy_id: str) -> str:
    """Return the assistant/profile ID for a bot-internal strategy ID."""
    return ASSISTANT_STRATEGY_IDS.get(strategy_id, strategy_id)
