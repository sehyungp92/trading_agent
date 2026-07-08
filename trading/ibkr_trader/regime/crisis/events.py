"""Crisis transition events for instrumentation."""
from dataclasses import dataclass


@dataclass
class CrisisTransitionEvent:
    event_type: str = "crisis_transition"
    from_level: str = ""
    to_level: str = ""
    from_level_int: int = 0
    to_level_int: int = 0
    risk_multiplier: float = 1.0
    dd_tier_multiplier: float = 1.0
    primary_warning_count: int = 0
    primary_crisis_count: int = 0
    dominant_channel: str = ""
    timestamp: str = ""
