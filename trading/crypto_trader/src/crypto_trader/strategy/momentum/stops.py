"""Stop price calculation — structure-first with ATR buffer."""

from __future__ import annotations

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.config import StopParams


class StopPlacer:
    def __init__(self, params: StopParams) -> None:
        self._p = params

    def compute(self, m15_bars: list[Bar], direction: Side, atr: float, symbol: str = "BTC") -> float:
        structural = self._find_structural_level(m15_bars, direction)
        entry_price = m15_bars[-1].close

        # Choose buffer mult based on symbol
        if symbol in self._p.major_symbols:
            buff_mult = self._p.atr_buffer_mult
        else:
            buff_mult = self._p.alt_atr_buffer_mult

        # Clamp buffer
        buff_mult = max(self._p.atr_buffer_min, min(self._p.atr_buffer_max, buff_mult))
        buffer = atr * buff_mult

        if direction == Side.LONG:
            stop_level = structural - buffer
        else:
            stop_level = structural + buffer

        # Enforce minimum stop distance
        if self._p.min_stop_atr_mult > 0:
            min_distance = atr * self._p.min_stop_atr_mult
            stop_distance = abs(entry_price - stop_level)
            if stop_distance < min_distance:
                if direction == Side.LONG:
                    stop_level = entry_price - min_distance
                else:
                    stop_level = entry_price + min_distance

        return stop_level

    def _find_structural_level(self, bars: list[Bar], direction: Side) -> float:
        lookback = min(self._p.swing_lookback, len(bars))
        recent = bars[-lookback:]

        if direction == Side.LONG:
            # Find the lowest swing low
            swing_low = min(b.low for b in recent)
            # Refine: look for a proper swing point
            for i in range(1, len(recent) - 1):
                if recent[i].low < recent[i - 1].low and recent[i].low < recent[i + 1].low:
                    swing_low = min(swing_low, recent[i].low)
            return swing_low
        else:
            # Find the highest swing high
            swing_high = max(b.high for b in recent)
            for i in range(1, len(recent) - 1):
                if recent[i].high > recent[i - 1].high and recent[i].high > recent[i + 1].high:
                    swing_high = max(swing_high, recent[i].high)
            return swing_high
