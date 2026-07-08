from __future__ import annotations

from dataclasses import dataclass

from .config import TradeDirection
from .models import PriceBar


@dataclass(frozen=True, slots=True)
class SmtResult:
    present: bool
    direction: TradeDirection = TradeDirection.FLAT
    strength: float = 0.0
    nq_extreme: float = 0.0
    es_extreme: float = 0.0


def detect_smt_divergence(
    nq_bars: list[PriceBar],
    es_bars: list[PriceBar],
    direction: TradeDirection,
    *,
    min_strength: float = 0.01,
) -> SmtResult:
    if len(nq_bars) < 2 or len(es_bars) < 2 or direction is TradeDirection.FLAT:
        return SmtResult(False)
    if direction is TradeDirection.LONG:
        nq_prev = min(bar.low for bar in nq_bars[:-1])
        es_prev = min(bar.low for bar in es_bars[:-1])
        nq_last = nq_bars[-1].low
        es_last = es_bars[-1].low
        strength = max(0.0, (nq_last - nq_prev) + (es_prev - es_last))
        return SmtResult(strength >= min_strength, direction, strength, nq_last, es_last)
    nq_prev = max(bar.high for bar in nq_bars[:-1])
    es_prev = max(bar.high for bar in es_bars[:-1])
    nq_last = nq_bars[-1].high
    es_last = es_bars[-1].high
    strength = max(0.0, (nq_prev - nq_last) + (es_last - es_prev))
    return SmtResult(strength >= min_strength, direction, strength, nq_last, es_last)
