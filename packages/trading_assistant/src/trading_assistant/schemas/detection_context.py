# schemas/detection_context.py
"""Detection context schemas — record threshold context on strategy suggestions.

Records which threshold values were used at detection time, enabling correlation
of threshold settings with suggestion outcomes for adaptive threshold learning.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, computed_field


class DetectionContext(BaseModel):
    """Records the threshold context at the time a strategy suggestion was produced."""

    detector_name: str  # e.g. "alpha_decay", "filter_cost"
    bot_id: str = ""
    threshold_name: str = ""  # e.g. "decay_threshold", "filter_cost_threshold"
    threshold_value: float = 0.0  # the threshold used at detection time
    observed_value: float = 0.0  # the actual observed value that triggered detection
    sample_size: int = 0  # trade/observation count backing the detection (0 = unknown)
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def margin(self) -> float:
        """How far the observed value was from the threshold."""
        return abs(self.observed_value - self.threshold_value)


class ThresholdRecord(BaseModel):
    """A single learned threshold for a detector/bot combination."""

    detector_name: str
    bot_id: str
    threshold_name: str
    default_value: float  # the hardcoded default
    learned_value: float | None = None  # computed from outcomes
    sample_count: int = 0
    confidence: float = 0.0  # 0-1, higher = more data
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def effective_value(self) -> float:
        """Return learned value if available and confident, else default."""
        if self.learned_value is not None and self.confidence > 0:
            return self.learned_value
        return self.default_value


class ThresholdProfile(BaseModel):
    """Collection of learned thresholds for a bot."""

    bot_id: str
    thresholds: dict[str, ThresholdRecord] = {}  # keyed by "{detector_name}:{threshold_name}"
    total_outcomes_used: int = 0
