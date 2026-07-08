from __future__ import annotations

from .models import PriceBar


def atr(bars: list[PriceBar], period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    ranges: list[float] = []
    sample = bars[-period:]
    prev_close = sample[0].close
    for bar in sample[1:]:
        ranges.append(max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close)))
        prev_close = bar.close
    return sum(ranges) / len(ranges) if ranges else 0.0
