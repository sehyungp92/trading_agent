from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import TradeDirection
from .models import PriceBar


class FvgState(str, Enum):
    ACTIVE = "active"
    INVERTED = "inverted"


@dataclass(slots=True)
class Fvg:
    direction: TradeDirection
    lower: float
    upper: float
    state: FvgState = FvgState.ACTIVE


class FvgStateMachine:
    def __init__(self, *, symbol: str = "NQ", max_age_bars: int = 3) -> None:
        self.symbol = symbol
        self.max_age_bars = max_age_bars
        self.gaps: list[Fvg] = []
        self._bars: list[PriceBar] = []

    def update(self, bar: PriceBar) -> None:
        self._bars.append(bar)
        if len(self._bars) >= 3:
            first = self._bars[-3]
            current = self._bars[-1]
            if current.low > first.high:
                self.gaps.append(Fvg(TradeDirection.LONG, first.high, current.low))
            elif current.high < first.low:
                self.gaps.append(Fvg(TradeDirection.SHORT, current.high, first.low))
        for gap in self.gaps:
            if gap.state is FvgState.ACTIVE:
                if gap.direction is TradeDirection.LONG and bar.close < gap.lower:
                    gap.direction = TradeDirection.SHORT
                    gap.state = FvgState.INVERTED
                elif gap.direction is TradeDirection.SHORT and bar.close > gap.upper:
                    gap.direction = TradeDirection.LONG
                    gap.state = FvgState.INVERTED

    def get_ifvg_entry(self, direction: TradeDirection) -> Fvg | None:
        for gap in reversed(self.gaps):
            if gap.direction is direction:
                return gap
        return None
