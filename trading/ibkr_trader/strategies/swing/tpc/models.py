from __future__ import annotations

from enum import Enum

from strategies.swing._shared.models import Direction


class PullbackType(str, Enum):
    TYPE_A = "classic_38_62"
    TYPE_B = "shallow_23_38"
    TYPE_C = "second_entry"


class TPCState(str, Enum):
    SCANNING = "scanning"
    PULLBACK_DETECTED = "pullback_detected"
    CONFIRMATION_PENDING = "confirmation_pending"
    ENTRY_READY = "entry_ready"
    IN_POSITION = "in_position"
    PARTIAL_T1 = "partial_t1"
    PARTIAL_T2 = "partial_t2"
    TRAILING_RUNNER = "trailing_runner"


class RegimeGrade(str, Enum):
    A_PLUS = "a_plus"
    VALID = "valid"
    INVALID = "invalid"


__all__ = ["Direction", "PullbackType", "TPCState", "RegimeGrade"]

