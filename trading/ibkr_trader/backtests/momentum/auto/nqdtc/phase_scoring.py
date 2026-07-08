"""NQDTC phase scoring -- re-exports from plugin for backwards compatibility."""
from backtests.momentum.auto.nqdtc.plugin import PHASE_HARD_REJECTS, PHASE_WEIGHTS, score_phase_metrics

score_phase = score_phase_metrics

__all__ = ["PHASE_HARD_REJECTS", "PHASE_WEIGHTS", "score_phase"]
