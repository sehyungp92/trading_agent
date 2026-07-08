from __future__ import annotations

from dataclasses import dataclass

from strategies.scalp._shared.levels import IVBLevels

from . import config

MIN_HOLD_SECONDS = config.MIN_HOLD_SECONDS
MIN_BUFFER_PTS = config.MIN_BUFFER_PTS
MIN_IVB_RANGE_POINTS = config.MIN_IVB_RANGE_POINTS
MAX_IVB_RANGE_POINTS = config.MAX_IVB_RANGE_POINTS


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = ()

    def __iter__(self):
        yield self.passed
        yield list(self.reasons)


def ivb_range_gate(ivb: IVBLevels) -> GateResult:
    passed = MIN_IVB_RANGE_POINTS <= ivb.range_pts <= MAX_IVB_RANGE_POINTS
    return GateResult(passed, () if passed else ("ivb_range",))


def breakout_acceptance(
    *,
    direction: int,
    close: float,
    high: float,
    low: float,
    ivb: IVBLevels,
    held_seconds: float,
    breakout_volume: float,
    rolling_volume_median: float,
    delta_60s: float | None,
    rolling_delta_median: float | None,
) -> GateResult:
    del high, low, breakout_volume, rolling_volume_median, delta_60s, rolling_delta_median
    if held_seconds < MIN_HOLD_SECONDS:
        return GateResult(False, ("hold_time",))
    if direction > 0 and close < ivb.high + MIN_BUFFER_PTS:
        return GateResult(False, ("breakout_buffer",))
    if direction < 0 and close > ivb.low - MIN_BUFFER_PTS:
        return GateResult(False, ("breakout_buffer",))
    return GateResult(True)
