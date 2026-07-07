from __future__ import annotations

from enum import Enum, IntEnum


class TradeDirection(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class SetupTier(str, Enum):
    NONE = "none"
    A = "A"
    B = "B"


class EntryType(str, Enum):
    STOP_CONFIRMATION = "stop_confirmation"
    LIMIT_RETEST = "limit_retest"


DAILY_LOOKBACK = 20
SCORE_THRESHOLD_A = 5.0
SCORE_THRESHOLD_B = 4.0
MIN_NQ_SWEEP_TICKS = 4
MIN_SMT_STRENGTH = 0.01
MIN_BODY_PERCENT = 0.50
ENTRY_OFFSET_TICKS = 1
A_RISK_PCT = 0.005
B_RISK_PCT = 0.0025
STOP_MIN_BUFFER_TICKS = 4
