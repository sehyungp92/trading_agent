"""Conjunction-gated alert level computation.

The core logic: no single indicator can trigger WARNING or CRISIS.
This is the primary defense against the 40% false positive rate that
plagued the stress HMM.
"""
from __future__ import annotations

import logging

from regime.crisis import config as C
from regime.crisis.indicators import CrisisIndicators

logger = logging.getLogger(__name__)


def is_hard_credit_impulse_warning_candidate(indicators: CrisisIndicators) -> bool:
    """Return whether credit impulse can arm the hard WARNING bridge."""
    persist_days = getattr(C, "HARD_CREDIT_IMPULSE_WARNING_PERSIST_DAYS", 0)
    if persist_days <= 0:
        return False
    mode = (indicators.stress_formation_mode or "").lower()
    if "credit_impulse" not in mode:
        return False
    return (
        indicators.warning_count
        >= getattr(C, "HARD_CREDIT_IMPULSE_WARNING_MIN_PRIMARY", 1)
    )


def compute_alert_level(indicators: CrisisIndicators) -> tuple[str, int]:
    """Compute alert level from indicator readings using conjunction logic.

    Returns:
        (alert_level_str, alert_level_int) e.g. ("WARNING", 2)
    """
    watch_count = indicators.watch_count
    warning_count = indicators.warning_count
    crisis_count = indicators.crisis_count

    # LEVEL 3 (CRISIS): 2+ primary at Crisis, OR 3+ at Warning
    if crisis_count >= C.CRISIS_MIN_PRIMARY:
        return C.ALERT_CRISIS, 3
    if warning_count >= C.CRISIS_ALT_WARNING:
        return C.ALERT_CRISIS, 3

    # LEVEL 2 (WARNING): 2+ primary at Warning
    if warning_count >= C.WARNING_MIN_PRIMARY:
        return C.ALERT_WARNING, 2

    # LEVEL 2 (WARNING) -- Hybrid: when any channel is at CRISIS,
    # allow WARNING with fewer channels at WARNING+.
    if (crisis_count >= C.HYBRID_WARNING_MIN_CRISIS
            and warning_count >= C.HYBRID_WARNING_MIN_PRIMARY):
        return C.ALERT_WARNING, 2

    # LEVEL 1 (WATCH): 1+ primary at Watch
    if watch_count >= C.WATCH_MIN_PRIMARY:
        return C.ALERT_WATCH, 1

    # LEVEL 0 (NORMAL)
    return C.ALERT_NORMAL, 0


def compute_advisory_level(
    indicators: CrisisIndicators,
    action_level_int: int = 0,
) -> tuple[str, int, str]:
    """Compute stricter user-facing advisory level.

    Internal WATCH is intentionally easy to trigger because it is a hysteresis
    buffer. External WATCH should mean "not actionable yet, but worth seeing".
    WARNING/CRISIS mirror the action level so dashboards and risk events agree.
    """
    if action_level_int >= 2:
        level_int = min(action_level_int, 3)
        return (
            C.ALERT_LEVELS[level_int],
            level_int,
            f"portfolio action active: {C.ALERT_LEVELS[level_int]}",
        )

    if indicators.crisis_count >= C.ADVISORY_WATCH_MIN_CRISIS:
        return (
            C.ALERT_WATCH,
            1,
            f"{indicators.crisis_count} primary channel(s) at CRISIS",
        )
    if indicators.warning_count >= C.ADVISORY_WATCH_MIN_WARNING:
        return (
            C.ALERT_WATCH,
            1,
            f"{indicators.warning_count} primary channel(s) at WARNING+",
        )
    if indicators.stress_formation_score >= C.STRESS_FORMATION_MIN_SCORE:
        return (
            C.ALERT_WATCH,
            1,
            (
                f"stress formation {indicators.stress_formation_mode}: "
                f"{indicators.stress_formation_reason}"
            ),
        )
    if indicators.watch_count >= C.ADVISORY_WATCH_MIN_PRIMARY:
        return (
            C.ALERT_WATCH,
            1,
            f"{indicators.watch_count} primary channels at WATCH+",
        )

    return C.ALERT_NORMAL, 0, "internal watch only"
