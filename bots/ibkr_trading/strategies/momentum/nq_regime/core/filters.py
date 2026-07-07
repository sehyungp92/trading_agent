from __future__ import annotations

from datetime import datetime, timedelta

from strategies.momentum.nq_regime.config import TradeSide
from strategies.momentum.nq_regime.core.indicators import IndicatorSnapshot
from strategies.momentum.nq_regime.core.levels import KeyLevels, nearest_resistance, nearest_support
from strategies.momentum.nq_regime.modules.base import NewsEvent


def active_news_event(ts: datetime, scheduled_news: list[NewsEvent] | None, *, buffer_minutes: int = 30) -> NewsEvent | None:
    if not scheduled_news:
        return None
    for event in scheduled_news:
        if event.tier > 1:
            continue
        start = event.start_ts - timedelta(minutes=buffer_minutes)
        end = event.end_ts + timedelta(minutes=buffer_minutes)
        if start <= ts <= end:
            return event
    return None


def trend_day_veto(side: TradeSide, indicators: IndicatorSnapshot) -> bool:
    if side is TradeSide.LONG:
        return indicators.trend_direction < 0 and indicators.vwap_slope < -2.0
    if side is TradeSide.SHORT:
        return indicators.trend_direction > 0 and indicators.vwap_slope > 2.0
    return False


def room_to_next_level_r(
    *,
    side: TradeSide,
    entry: float,
    stop: float,
    levels: KeyLevels | None,
    fallback_target: float,
) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if side is TradeSide.LONG:
        level = nearest_resistance(entry, levels) or fallback_target
        room = max(0.0, level - entry)
    elif side is TradeSide.SHORT:
        level = nearest_support(entry, levels) or fallback_target
        room = max(0.0, entry - level)
    else:
        room = 0.0
    return room / risk if risk > 0 else 0.0

