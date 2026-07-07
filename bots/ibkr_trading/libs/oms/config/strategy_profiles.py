"""Strategy execution profiles."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyProfile:
    """Execution discipline profile per strategy per role."""

    strategy_id: str
    default_ttl_bars: int = 2
    default_ttl_seconds: Optional[int] = None
    max_reprices: int = 0
    teleport_ticks: Optional[int] = None
    cancel_on_teleport: bool = True
    # Market hours rules
    no_entry_first_minutes: int = 0
    no_entry_last_minutes: int = 0
    # Flatten rules
    mandatory_flatten_time: Optional[str] = None  # "15:50" ET string
