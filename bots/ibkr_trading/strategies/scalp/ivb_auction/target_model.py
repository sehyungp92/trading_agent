from __future__ import annotations

from dataclasses import dataclass

from strategies.scalp._shared.levels import IVBLevels

from .config import TradeDirection


@dataclass(frozen=True, slots=True)
class Targets:
    tp1: float
    tp2: float


def reclaim_targets(*, entry_price: float, direction: TradeDirection, ivb: IVBLevels) -> Targets:
    del entry_price
    if direction is TradeDirection.LONG:
        return Targets(tp1=ivb.poc, tp2=ivb.high)
    if direction is TradeDirection.SHORT:
        return Targets(tp1=ivb.poc, tp2=ivb.low)
    return Targets(tp1=ivb.poc, tp2=ivb.poc)


def fallback_targets(*, entry_price: float, direction: TradeDirection, ivb: IVBLevels) -> Targets:
    extension = max(ivb.range_pts, 1.0)
    if direction is TradeDirection.LONG:
        return Targets(tp1=entry_price + extension * 0.5, tp2=entry_price + extension)
    if direction is TradeDirection.SHORT:
        return Targets(tp1=entry_price - extension * 0.5, tp2=entry_price - extension)
    return Targets(tp1=entry_price, tp2=entry_price)
