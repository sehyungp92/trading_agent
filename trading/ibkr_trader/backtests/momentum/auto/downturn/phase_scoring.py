"""Compatibility shim for downturn phase scoring."""
from backtests.momentum.auto.downturn.plugin import PHASE_HARD_REJECTS, PHASE_WEIGHTS, score_phase_metrics

score_phase = score_phase_metrics

__all__ = ["PHASE_HARD_REJECTS", "PHASE_WEIGHTS", "score_phase"]
