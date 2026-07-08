"""Regime transition events for instrumentation."""
from dataclasses import dataclass


@dataclass
class RegimeTransitionEvent:
    event_type: str = "regime_transition"
    from_regime: str = ""
    to_regime: str = ""
    regime_confidence: float = 0.0
    stress_level: float = 0.0
    stress_onset: bool = False
    shift_velocity: float = 0.0
    timestamp: str = ""
