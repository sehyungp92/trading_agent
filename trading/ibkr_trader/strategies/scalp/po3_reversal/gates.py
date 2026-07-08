from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from strategies.scalp._shared.time_utils import to_et

from .config import TradeDirection


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reason: str = ""


def session_gate(value: datetime) -> GateResult:
    local = to_et(value).time()
    return GateResult(time(9, 30) <= local <= time(15, 15), "session")


def direction_gate(
    daily_bias: TradeDirection,
    h4_bias: TradeDirection,
    direction: TradeDirection,
) -> GateResult:
    if direction is TradeDirection.FLAT:
        return GateResult(False, "flat")
    if daily_bias not in {TradeDirection.FLAT, direction}:
        return GateResult(False, "daily_bias")
    if h4_bias not in {TradeDirection.FLAT, direction}:
        return GateResult(False, "h4_bias")
    return GateResult(True)
