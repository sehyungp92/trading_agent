"""Sticky de-escalation state machine.

Immediate escalation, but requires consecutive days below threshold
to de-escalate. This prevents whipsawing during volatile but not
genuinely crisis periods.

R3 additions:
  - Accelerated de-escalation: when raw is NORMAL for N+ consecutive days,
    jump directly to NORMAL instead of stepping down one level at a time.
  - Recovery ramp: after de-escalating from WARNING/CRISIS, gradually ramp
    risk multiplier from RECOVERY_RAMP_FLOOR to 1.0 over RECOVERY_RAMP_DAYS.
"""
from __future__ import annotations

import logging

from regime.crisis import config as C

logger = logging.getLogger(__name__)


class HysteresisTracker:
    """Tracks consecutive days below threshold for sticky de-escalation.

    State:
        current_level: int (0-3)
        days_below: int (consecutive days the raw signal has been below current_level)
        days_all_normal: int (consecutive days raw==0, for accelerated de-escalation)
        _last_deesc_from: int (level de-escalated FROM, for recovery ramp)
        _deesc_day: int (counter since last de-escalation to NORMAL)
    """

    def __init__(self, initial_level: int = 0) -> None:
        self.current_level: int = initial_level
        self.days_below: int = 0
        self.days_at_level: int = 0   # consecutive days at current_level
        self.days_all_normal: int = 0           # consecutive days raw==0
        self.hard_credit_impulse_warning_days: int = 0
        self._last_deesc_from: int = 0          # level de-escalated FROM (for ramp)
        self._deesc_day: int = 0                # counter since last de-escalation

    def apply_hard_credit_impulse_bridge(
        self,
        raw_level: int,
        candidate_active: bool,
    ) -> int:
        """Promote raw WARNING after persistent credit-impulse confirmation."""
        persist_days = getattr(C, "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS", 0)
        if persist_days <= 0:
            self.hard_credit_impulse_warning_days = 0
            return raw_level

        if candidate_active:
            self.hard_credit_impulse_warning_days += 1
        else:
            self.hard_credit_impulse_warning_days = 0

        if self.hard_credit_impulse_warning_days >= persist_days:
            return max(raw_level, 2)
        return raw_level

    def update(self, raw_level: int) -> int:
        """Apply hysteresis to a raw alert level.

        Args:
            raw_level: The conjunction-computed alert level (0-3)

        Returns:
            The hysteresis-adjusted alert level (0-3)
        """
        prev_level = self.current_level

        # Track consecutive all-normal days (for accelerated de-escalation)
        if raw_level == 0:
            self.days_all_normal += 1
        else:
            self.days_all_normal = 0

        # Increment recovery ramp counter when at NORMAL after de-escalation
        if self.current_level == 0 and self._last_deesc_from > 1:
            self._deesc_day += 1
            ramp_days = getattr(C, "RECOVERY_RAMP_DAYS", 0)
            if ramp_days > 0 and self._deesc_day >= ramp_days:
                self._last_deesc_from = 0  # ramp complete

        # Immediate escalation: always allow going up
        if raw_level > self.current_level:
            self.current_level = raw_level
            self.days_below = 0
            # Reset recovery state on re-escalation
            self._last_deesc_from = 0
            self._deesc_day = 0
            logger.info(
                "Crisis alert ESCALATED to level %d (%s)",
                raw_level, C.ALERT_LEVELS[raw_level],
            )
        elif raw_level == self.current_level:
            # Same level: reset de-escalation counter
            self.days_below = 0
        else:
            # Raw is below current: increment days_below
            self.days_below += 1

            accel_days = getattr(C, "ACCEL_DEESCALATE_NORMAL_DAYS", 0)

            # Accelerated de-escalation: raw NORMAL for N+ days -> jump to NORMAL
            if (accel_days > 0 and raw_level == 0
                    and self.days_all_normal >= accel_days
                    and self.current_level > 0):
                old_level = self.current_level
                self._last_deesc_from = old_level
                self._deesc_day = 0
                self.current_level = 0
                self.days_below = 0
                logger.info(
                    "Crisis alert ACCEL DE-ESCALATED from %d (%s) to NORMAL "
                    "after %d consecutive normal days",
                    old_level, C.ALERT_LEVELS[old_level], accel_days,
                )
            else:
                # Standard step-down de-escalation
                required_days = self._required_days_below()
                if self.days_below >= required_days:
                    old_level = self.current_level
                    self.current_level = max(self.current_level - 1, raw_level)
                    self.days_below = 0
                    # Track de-escalation for recovery ramp (only from WARNING/CRISIS)
                    if self.current_level == 0 and old_level >= 2:
                        self._last_deesc_from = old_level
                        self._deesc_day = 0
                    logger.info(
                        "Crisis alert DE-ESCALATED from %d (%s) to %d (%s) "
                        "after %d consecutive days below",
                        old_level, C.ALERT_LEVELS[old_level],
                        self.current_level, C.ALERT_LEVELS[self.current_level],
                        required_days,
                    )

        # Track consecutive days at the (possibly new) level
        if self.current_level == prev_level:
            self.days_at_level += 1
        else:
            self.days_at_level = 1  # first day at new level

        return self.current_level

    def _required_days_below(self) -> int:
        """Days required before de-escalating from current level."""
        if self.current_level == 3:  # CRISIS -> WARNING
            return C.DEESCALATE_CRISIS_DAYS
        if self.current_level == 2:  # WARNING -> WATCH
            return C.DEESCALATE_WARNING_DAYS
        if self.current_level == 1:  # WATCH -> NORMAL
            return C.DEESCALATE_WATCH_DAYS
        return 0

    @property
    def recovery_ramp_mult(self) -> float:
        """Risk multiplier during recovery ramp (0.75..1.0). Pure read, no side effects."""
        ramp_days = getattr(C, "RECOVERY_RAMP_DAYS", 0)
        if ramp_days <= 0 or self._last_deesc_from <= 1:
            return 1.0  # disabled or was only WATCH
        if self.current_level > 0:
            return 1.0  # still elevated, ramp not active
        floor = getattr(C, "RECOVERY_RAMP_FLOOR", 0.75)
        progress = self._deesc_day / ramp_days
        return floor + (1.0 - floor) * min(progress, 1.0)

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        return {
            "current_level": self.current_level,
            "days_below": self.days_below,
            "days_at_level": self.days_at_level,
            "days_all_normal": self.days_all_normal,
            "hard_credit_impulse_warning_days": self.hard_credit_impulse_warning_days,
            "_last_deesc_from": self._last_deesc_from,
            "_deesc_day": self._deesc_day,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HysteresisTracker:
        """Restore from persisted state."""
        tracker = cls(initial_level=data.get("current_level", 0))
        tracker.days_below = data.get("days_below", 0)
        tracker.days_at_level = data.get("days_at_level", 0)
        tracker.days_all_normal = data.get("days_all_normal", 0)
        tracker.hard_credit_impulse_warning_days = data.get(
            "hard_credit_impulse_warning_days", 0,
        )
        tracker._last_deesc_from = data.get("_last_deesc_from", 0)
        tracker._deesc_day = data.get("_deesc_day", 0)
        return tracker
