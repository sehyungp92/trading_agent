from __future__ import annotations

from datetime import datetime
from enum import Enum

from strategies.momentum.nq_regime import config
from strategies.scalp._shared.time_utils import to_et


class SessionPhase(str, Enum):
    PRE_MARKET = "pre_market"
    OPENING_OBSERVATION = "opening_observation"
    EARLY_SWEEP_WATCH = "early_sweep_watch"
    PRIMARY_DECISION = "primary_decision"
    LUNCH_COMPRESSION = "lunch_compression"
    PM_CONTINUATION = "pm_continuation"
    LATE_PM_RESTRICTED = "late_pm_restricted"
    MANAGE_ONLY = "manage_only"
    HARD_FLATTEN = "hard_flatten"
    CLOSED = "closed"


def get_session_phase(ts: datetime) -> SessionPhase:
    et_time = to_et(ts).time()
    if et_time < config.RTH_OPEN_ET:
        return SessionPhase.PRE_MARKET
    if et_time < config.EARLY_SWEEP_START_ET:
        return SessionPhase.OPENING_OBSERVATION
    if et_time < config.IB_END_ET:
        return SessionPhase.EARLY_SWEEP_WATCH
    if et_time < config.PRIMARY_WINDOW_END_ET:
        return SessionPhase.PRIMARY_DECISION
    if et_time < config.LUNCH_END_ET:
        return SessionPhase.LUNCH_COMPRESSION
    if et_time < config.PM_WINDOW_START_ET:
        return SessionPhase.LUNCH_COMPRESSION
    if et_time < config.LATE_PM_START_ET:
        return SessionPhase.PM_CONTINUATION
    if et_time < config.NO_NEW_ENTRIES_AFTER_ET:
        return SessionPhase.LATE_PM_RESTRICTED
    if et_time < config.HARD_FLATTEN_ET:
        return SessionPhase.MANAGE_ONLY
    if et_time < config.EMERGENCY_FLATTEN_ET:
        return SessionPhase.HARD_FLATTEN
    return SessionPhase.CLOSED


def entries_allowed(phase: SessionPhase, module: str) -> bool:
    if phase in {SessionPhase.MANAGE_ONLY, SessionPhase.HARD_FLATTEN, SessionPhase.CLOSED}:
        return False
    if module == config.ModuleId.STRUCTURAL_EXPANSION.value:
        return phase in {SessionPhase.PRIMARY_DECISION, SessionPhase.PM_CONTINUATION, SessionPhase.LATE_PM_RESTRICTED}
    if module == config.ModuleId.LIQUIDITY_REVERSION.value:
        allowed = {SessionPhase.EARLY_SWEEP_WATCH, SessionPhase.PRIMARY_DECISION, SessionPhase.PM_CONTINUATION}
        if config.ALLOW_LATE_PM_REVERSION:
            allowed.add(SessionPhase.LATE_PM_RESTRICTED)
        return phase in allowed
    if module == config.ModuleId.SECOND_WIND.value:
        return phase in {SessionPhase.PM_CONTINUATION, SessionPhase.LATE_PM_RESTRICTED}
    return False


def should_hard_flatten(ts: datetime) -> bool:
    return get_session_phase(ts) in {SessionPhase.HARD_FLATTEN, SessionPhase.CLOSED}
