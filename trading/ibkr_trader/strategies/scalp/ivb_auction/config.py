from __future__ import annotations

from enum import Enum, IntEnum


class TradeDirection(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class IvbModule(str, Enum):
    NONE = "none"
    A1_CONTINUATION = "a1_continuation"
    A2_RECLAIM = "a2_reclaim"


class EntryTrigger(str, Enum):
    BREAKOUT = "breakout"
    PROFILE_RELOAD = "profile_reload"
    RECLAIM_RETEST = "reclaim_retest"


A1_MIN_SCORE = 70.0
A2_MIN_SCORE = 65.0
MIN_HOLD_SECONDS = 60.0
MIN_BUFFER_PTS = 0.0
MIN_IVB_RANGE_POINTS = 20.0
MAX_IVB_RANGE_POINTS = 200.0
MAX_CHASE_EXTENSION_RANGE_FRACTION = 0.35
BASE_RISK_PCT = 0.005
STOP_CAP_IVB_FRACTION = 0.50
MIN_R_TO_TP1 = 1.0
A2_MIN_R_TO_TP1 = 0.75
TP1_QUANTILE = 0.60
TP2_QUANTILE = 0.90
