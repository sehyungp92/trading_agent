from __future__ import annotations

from dataclasses import dataclass

from strategies.momentum.nq_regime.config import IBType, classify_ib_range


@dataclass(frozen=True, slots=True)
class KeyLevels:
    pdh: float = 0.0
    pdl: float = 0.0
    pdm: float = 0.0
    onh: float = 0.0
    onl: float = 0.0
    onm: float = 0.0
    pmh: float = 0.0
    pml: float = 0.0
    pmm: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    orh: float = 0.0
    orl: float = 0.0
    weekly_high: float = 0.0
    weekly_low: float = 0.0

    def major_resistance_levels(self) -> tuple[float, ...]:
        return tuple(level for level in (self.pdh, self.onh, self.pmh, self.vah, self.orh, self.weekly_high) if level > 0)

    def major_support_levels(self) -> tuple[float, ...]:
        return tuple(level for level in (self.pdl, self.onl, self.pml, self.val, self.orl, self.weekly_low) if level > 0)


@dataclass(frozen=True, slots=True)
class IBLevels:
    high: float = 0.0
    low: float = 0.0
    mid: float = 0.0
    range_pts: float = 0.0
    ib_type: IBType = IBType.UNCLASSIFIED


def build_ib_levels(high: float, low: float) -> IBLevels:
    if high <= 0 or low <= 0 or high <= low:
        return IBLevels()
    range_pts = high - low
    return IBLevels(
        high=high,
        low=low,
        mid=(high + low) / 2.0,
        range_pts=range_pts,
        ib_type=classify_ib_range(range_pts),
    )


def nearest_resistance(price: float, levels: KeyLevels | None) -> float | None:
    if levels is None:
        return None
    candidates = [level for level in levels.major_resistance_levels() if level > price]
    return min(candidates) if candidates else None


def nearest_support(price: float, levels: KeyLevels | None) -> float | None:
    if levels is None:
        return None
    candidates = [level for level in levels.major_support_levels() if 0 < level < price]
    return max(candidates) if candidates else None

