from __future__ import annotations

from .config import SetupTier, TradeDirection
from .models import Po3Context, PriceBar


def build_context(daily: list[PriceBar], h4: list[PriceBar], h1: list[PriceBar]) -> Po3Context:
    daily_bias = _bias(daily)
    h4_bias = _bias(h4)
    target = h1[-1].high if h1 else 0.0
    if daily_bias is TradeDirection.SHORT:
        target = h1[-1].low if h1 else 0.0
    return Po3Context(daily_bias=daily_bias, h4_bias=h4_bias, h1_target=target)


def determine_trade_tier(
    daily_bias: TradeDirection,
    h4_bias: TradeDirection,
    conditions: dict[str, bool],
) -> SetupTier:
    score = sum(1 for value in conditions.values() if value)
    aligned = daily_bias in {TradeDirection.FLAT, h4_bias}
    if aligned and score >= 4:
        return SetupTier.A
    if score >= 3:
        return SetupTier.B
    return SetupTier.NONE


def _bias(bars: list[PriceBar]) -> TradeDirection:
    if not bars:
        return TradeDirection.FLAT
    bar = bars[-1]
    if bar.close > bar.open:
        return TradeDirection.LONG
    if bar.close < bar.open:
        return TradeDirection.SHORT
    return TradeDirection.FLAT
