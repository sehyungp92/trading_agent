from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from strategies.swing._shared.etf_core import ETFBarInput, SetupSnapshot
from strategies.swing._shared.models import Direction
from strategies.swing._shared.session import is_in_session_window
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.models import RegimeGrade


def session_filter(timestamp: datetime, cfg: TPCSymbolConfig) -> bool:
    if not is_in_session_window(timestamp, cfg.primary_windows_et):
        return False
    if cfg.avoid_windows_et and is_in_session_window(timestamp, cfg.avoid_windows_et):
        return False
    local = timestamp.astimezone(ZoneInfo("America/New_York"))
    local_weekday = local.weekday()
    local_hour = local.hour
    if local_weekday == 4 and local_hour >= 14:
        return False
    return True


def news_filter(timestamp: datetime, cfg: TPCSymbolConfig) -> bool:
    del timestamp, cfg
    return True


def regime_direction(bar_input: ETFBarInput, cfg: TPCSymbolConfig) -> tuple[Direction, RegimeGrade, str]:
    bars_4h = bar_input.bars_4h
    bars_1h = bar_input.bars_1h
    if bars_4h is None or bars_1h is None or len(bars_4h) < cfg.ma_100_period + 2:
        return Direction.FLAT, RegimeGrade.INVALID, "insufficient_4h"
    ind = bar_input.indicators
    close = float(bars_4h.closes[-1])
    ma50 = ind.get("ma50_4h", np.nan)
    ma100 = ind.get("ma100_4h", np.nan)
    rsi = ind.get("rsi_4h", np.nan)
    atr4 = ind.get("atr_4h", np.nan)
    adx4 = ind.get("adx_4h", np.nan)
    plus_di = ind.get("plus_di_4h", np.nan)
    minus_di = ind.get("minus_di_4h", np.nan)
    ema20_1h = ind.get("ema20_1h", np.nan)
    required = [ma50, ma100, rsi, atr4, ema20_1h]
    if cfg.min_adx_4h > 0 or cfg.max_adx_4h > 0 or cfg.require_di_alignment:
        required.extend([adx4, plus_di, minus_di])
    if any(np.isnan(x) for x in required):
        return Direction.FLAT, RegimeGrade.INVALID, "nan_regime"
    ma50_slope = ma50 - float(np.nanmean(bars_4h.closes[-6:-1]))
    ma100_slope = ma100 - float(np.nanmean(bars_4h.closes[-12:-2]))
    extended = abs(bar_input.bar_15m.close - ema20_1h) > cfg.max_extension_atr_mult * atr4
    if extended:
        return Direction.FLAT, RegimeGrade.INVALID, "extended_from_value"
    if cfg.min_adx_4h > 0 and adx4 < cfg.min_adx_4h:
        return Direction.FLAT, RegimeGrade.INVALID, "trend_strength_low"
    if cfg.max_adx_4h > 0 and adx4 > cfg.max_adx_4h:
        return Direction.FLAT, RegimeGrade.INVALID, "trend_strength_overextended"
    if close > ma50 and ma50_slope > 0 and rsi >= cfg.rsi_long_band[0]:
        if not _trend_quality_ok(Direction.LONG, ma50_slope, ma100_slope, atr4, plus_di, minus_di, cfg):
            return Direction.FLAT, RegimeGrade.INVALID, "long_trend_quality"
        grade = RegimeGrade.A_PLUS if close > ma100 and ma50 > ma100 and ma100_slope > 0 and rsi <= cfg.rsi_a_plus_long_max else RegimeGrade.VALID
        return Direction.LONG, grade, "long_regime"
    if close < ma50 and ma50_slope < 0 and rsi <= cfg.rsi_short_band[1]:
        if not _trend_quality_ok(Direction.SHORT, ma50_slope, ma100_slope, atr4, plus_di, minus_di, cfg):
            return Direction.FLAT, RegimeGrade.INVALID, "short_trend_quality"
        grade = RegimeGrade.A_PLUS if close < ma100 and ma50 < ma100 and ma100_slope < 0 and rsi >= cfg.rsi_a_plus_short_min else RegimeGrade.VALID
        return Direction.SHORT, grade, "short_regime"
    return Direction.FLAT, RegimeGrade.INVALID, "ma_rsi_conflict"


def _trend_quality_ok(
    direction: Direction,
    ma50_slope: float,
    ma100_slope: float,
    atr4: float,
    plus_di: float,
    minus_di: float,
    cfg: TPCSymbolConfig,
) -> bool:
    atr = max(float(atr4), 1e-9)
    if cfg.require_di_alignment:
        if direction == Direction.LONG and not (plus_di > minus_di):
            return False
        if direction == Direction.SHORT and not (minus_di > plus_di):
            return False
    if cfg.min_ma50_slope_atr_4h > 0:
        slope_r = ma50_slope / atr if direction == Direction.LONG else -ma50_slope / atr
        if slope_r < cfg.min_ma50_slope_atr_4h:
            return False
    if cfg.min_ma100_slope_atr_4h > 0:
        slope_r = ma100_slope / atr if direction == Direction.LONG else -ma100_slope / atr
        if slope_r < cfg.min_ma100_slope_atr_4h:
            return False
    return True


def pullback_valid(setup: SetupSnapshot, cfg: TPCSymbolConfig) -> tuple[bool, str]:
    if setup.risk_per_share <= 0:
        return False, "invalid_risk"
    if setup.meta.get("depth", 0.0) > cfg.fib_a_high + 0.05:
        return False, "too_deep"
    if setup.meta.get("rr", 0.0) < 2.0:
        return False, "rr_below_2"
    return True, ""


def daily_room_filter(
    entry_price: float,
    stop_price: float,
    direction: Direction,
    daily_levels: list[float],
    min_room_r: float = 2.0,
) -> bool:
    risk = abs(entry_price - stop_price)
    if risk <= 0 or not daily_levels:
        return True
    required_room = max(float(min_room_r), 0.0) * risk
    if direction == Direction.LONG:
        levels = [level for level in daily_levels if level > entry_price]
        return not levels or min(levels) - entry_price >= required_room
    levels = [level for level in daily_levels if level < entry_price]
    return not levels or entry_price - max(levels) >= required_room
