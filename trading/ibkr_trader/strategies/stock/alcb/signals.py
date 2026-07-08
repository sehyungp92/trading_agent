"""Signal helpers for ALCB."""

from __future__ import annotations

import bisect
from datetime import datetime, time
from math import sqrt
from statistics import fmean

from .config import ET, StrategySettings
from .data import StrategyDataStore
from .models import (
    Bar,
    Box,
    BreakoutQualification,
    Campaign,
    CompressionTier,
    Direction,
    EntryType,
    Regime,
    ResearchDailyBar,
)


def _close_values(bars: list[ResearchDailyBar | Bar]) -> list[float]:
    return [float(bar.close) for bar in bars]


def _high_values(bars: list[ResearchDailyBar | Bar]) -> list[float]:
    return [float(bar.high) for bar in bars]


def _low_values(bars: list[ResearchDailyBar | Bar]) -> list[float]:
    return [float(bar.low) for bar in bars]


def _volume_values(bars: list[ResearchDailyBar | Bar]) -> list[float]:
    return [float(bar.volume) for bar in bars]


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    result = [values[0]]
    current = values[0]
    for value in values[1:]:
        current = (alpha * value) + ((1.0 - alpha) * current)
        result.append(current)
    return result


def pct_change(values: list[float], periods: int) -> float:
    if len(values) <= periods or values[-periods - 1] <= 0:
        return 0.0
    return (values[-1] - values[-periods - 1]) / values[-periods - 1]


def rolling_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = max(0.0, min(quantile, 1.0)) * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] + ((ordered[hi] - ordered[lo]) * frac)


def _true_ranges(bars: list[ResearchDailyBar | Bar]) -> list[float]:
    if len(bars) < 2:
        return []
    values: list[float] = []
    prev_close = float(bars[0].close)
    for bar in bars[1:]:
        high = float(bar.high)
        low = float(bar.low)
        values.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        prev_close = float(bar.close)
    return values


def atr_from_bars(bars: list[ResearchDailyBar | Bar], period: int) -> float:
    if period > 0 and len(bars) > period + 1:
        bars = bars[-(period + 1):]
    true_ranges = _true_ranges(bars)
    if len(true_ranges) < period:
        if not true_ranges:
            return 0.0
        return fmean(true_ranges)
    return fmean(true_ranges[-period:])


def daily_rvol(bars: list[ResearchDailyBar], lookback: int = 20) -> float:
    if len(bars) < 2:
        return 0.0
    sample = [float(bar.volume) for bar in bars[-lookback - 1 : -1]]
    baseline = fmean(sample) if sample else 0.0
    if baseline <= 0:
        return 0.0
    return float(bars[-1].volume) / baseline


def intraday_rvol_30m(bars: list[Bar], lookback: int = 20) -> float:
    if len(bars) < 2:
        return 0.0
    sample = [float(bar.volume) for bar in bars[-lookback - 1 : -1]]
    baseline = fmean(sample) if sample else 0.0
    if baseline <= 0:
        return 0.0
    return float(bars[-1].volume) / baseline


def adx_from_bars(bars: list[Bar], period: int = 14) -> float:
    if len(bars) < period + 2:
        return 0.0
    bars = bars[-(period + 1):]
    true_ranges: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for prev, bar in zip(bars[:-1], bars[1:]):
        up_move = bar.high - prev.high
        down_move = prev.low - bar.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        true_ranges.append(max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close)))
    if len(true_ranges) < period:
        return 0.0
    tr = sum(true_ranges[-period:])
    if tr <= 0:
        return 0.0
    plus_di = 100.0 * (sum(plus_dm[-period:]) / tr)
    minus_di = 100.0 * (sum(minus_dm[-period:]) / tr)
    denom = plus_di + minus_di
    if denom <= 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / denom


def rs_percentile(symbol: str, universe: dict[str, list[ResearchDailyBar]]) -> float:
    return rs_percentiles(universe).get(symbol, 0.0)


def rs_percentiles(universe: dict[str, list[ResearchDailyBar]]) -> dict[str, float]:
    """Compute RS percentiles for the full universe in one pass."""
    scores: dict[str, float] = {}
    spy = universe.get("SPY") or []
    spy_close = _close_values(spy)
    spy20 = pct_change(spy_close, 20) if spy_close else 0.0
    spy60 = pct_change(spy_close, 60) if spy_close else 0.0
    for name, bars in universe.items():
        closes = _close_values(bars)
        if len(closes) < 61:
            scores[name] = 0.0
            continue
        scores[name] = (pct_change(closes, 20) - spy20) + (pct_change(closes, 60) - spy60)
    ordered = sorted(scores.values())
    if not ordered:
        return {}
    total = len(ordered)
    return {
        name: bisect.bisect_right(ordered, score) / total
        for name, score in scores.items()
    }


def accumulation_score(bars: list[ResearchDailyBar], settings: StrategySettings | None = None) -> float:
    if len(bars) < 10:
        return 0.0
    recent = bars[-20:]
    up_volume = [bar.volume for bar in recent if bar.close >= bar.open]
    down_volume = [bar.volume for bar in recent if bar.close < bar.open]
    up_closes = sum(1 for bar in recent if bar.cpr >= 0.65 and bar.close >= bar.open)
    if not up_volume:
        return 0.0
    up_avg = fmean(up_volume)
    down_avg = fmean(down_volume) if down_volume else 0.0
    score = 0.5
    if up_avg > max(down_avg, 1.0):
        score += 0.25
    if up_closes >= max(2, len(recent) // 6):
        score += 0.25
    # Opt 5: Volume trend acceleration
    mid = len(recent) // 2
    first_half_up = [b.volume for b in recent[:mid] if b.close >= b.open]
    second_half_up = [b.volume for b in recent[mid:] if b.close >= b.open]
    first_avg = fmean(first_half_up) if first_half_up else 0.0
    second_avg = fmean(second_half_up) if second_half_up else 0.0
    bonus = settings.accum_vol_trend_bonus if settings else 0.15
    if first_avg > 0 and second_avg > first_avg:
        score += bonus
    # Opt 5: Recency bonus
    last_5 = recent[-5:]
    recency_up = sum(1 for bar in last_5 if bar.cpr >= 0.65 and bar.close >= bar.open)
    if recency_up >= 3:
        score += 0.10
    return max(0.0, min(score, 1.0))


def distribution_score(bars: list[ResearchDailyBar], settings: StrategySettings | None = None) -> float:
    if len(bars) < 10:
        return 0.0
    recent = bars[-20:]
    down_volume = [bar.volume for bar in recent if bar.close <= bar.open]
    up_volume = [bar.volume for bar in recent if bar.close > bar.open]
    weak_closes = sum(1 for bar in recent if bar.cpr <= 0.35 and bar.close <= bar.open)
    if not down_volume:
        return 0.0
    down_avg = fmean(down_volume)
    up_avg = fmean(up_volume) if up_volume else 0.0
    score = 0.5
    if down_avg > max(up_avg, 1.0):
        score += 0.25
    if weak_closes >= max(2, len(recent) // 6):
        score += 0.25
    # Opt 5: Volume trend acceleration (distribution mirror)
    mid = len(recent) // 2
    first_half_down = [b.volume for b in recent[:mid] if b.close <= b.open]
    second_half_down = [b.volume for b in recent[mid:] if b.close <= b.open]
    first_avg = fmean(first_half_down) if first_half_down else 0.0
    second_avg = fmean(second_half_down) if second_half_down else 0.0
    bonus = settings.accum_vol_trend_bonus if settings else 0.15
    if first_avg > 0 and second_avg > first_avg:
        score += bonus
    # Opt 5: Recency bonus (distribution mirror)
    last_5 = recent[-5:]
    recency_down = sum(1 for bar in last_5 if bar.cpr <= 0.35 and bar.close <= bar.open)
    if recency_down >= 3:
        score += 0.10
    return max(0.0, min(score, 1.0))


def choose_box_length(daily_bars: list[ResearchDailyBar], settings: StrategySettings) -> int:
    if len(daily_bars) < 55:
        return settings.box_length_mid
    proposals: list[int] = []
    for offset in range(settings.hysteresis_days):
        subset = daily_bars[: len(daily_bars) - offset]
        atr14 = atr_from_bars(subset, 14)
        atr50 = atr_from_bars(subset, 50)
        ratio = atr14 / atr50 if atr50 > 0 else 1.0
        if ratio < settings.atr_ratio_low:
            proposals.append(settings.box_length_low)
        elif ratio <= settings.atr_ratio_high:
            proposals.append(settings.box_length_mid)
        else:
            proposals.append(settings.box_length_high)
    if proposals.count(proposals[0]) == len(proposals):
        return proposals[0]
    return max(set(proposals), key=proposals.count)


def squeeze_history(daily_bars: list[ResearchDailyBar], L: int, lookback: int) -> list[float]:
    if len(daily_bars) < L + 2:
        return []
    values: list[float] = []
    for end in range(L, len(daily_bars)):
        window = daily_bars[max(0, end - L) : end]
        atr50 = atr_from_bars(daily_bars[:end], 50)
        if not window or atr50 <= 0:
            continue
        values.append((max(bar.high for bar in window) - min(bar.low for bar in window)) / atr50)
    return values[-lookback:]


def detect_compression_box(daily_bars: list[ResearchDailyBar], settings: StrategySettings) -> Box | None:
    L = choose_box_length(daily_bars, settings)
    if len(daily_bars) < L:
        return None
    window = daily_bars[-L:]
    range_high = max(bar.high for bar in window)
    range_low = min(bar.low for bar in window)
    box_height = range_high - range_low
    box_mid = (range_high + range_low) / 2.0
    atr50 = atr_from_bars(daily_bars, 50)
    squeeze_metric = box_height / atr50 if atr50 > 0 else 999.0
    containment = sum(1 for bar in window if range_low <= bar.close <= range_high) / len(window)
    if containment < settings.min_containment or squeeze_metric > settings.max_squeeze_metric:
        return None
    history = squeeze_history(daily_bars[:-1], L, settings.squeeze_lookback)
    q30 = rolling_quantile(history, settings.sq_good_quantile) if history else squeeze_metric
    q65 = rolling_quantile(history, settings.sq_loose_quantile) if history else squeeze_metric
    if squeeze_metric <= q30:
        tier = CompressionTier.GOOD
    elif squeeze_metric >= q65:
        tier = CompressionTier.LOOSE
    else:
        tier = CompressionTier.NEUTRAL
    return Box(
        start_date=str(window[0].trade_date),
        end_date=str(window[-1].trade_date),
        L_used=L,
        high=float(range_high),
        low=float(range_low),
        mid=float(box_mid),
        height=float(box_height),
        containment=float(containment),
        squeeze_metric=float(squeeze_metric),
        tier=tier,
    )


def _anchor_datetime(anchor_ts: str | None, bars: list[Bar]) -> datetime | None:
    if anchor_ts:
        try:
            return datetime.fromisoformat(anchor_ts)
        except ValueError:
            pass
    if not bars:
        return None
    return bars[0].start_time


def bars_since_anchor(bars: list[Bar], anchor_ts: str | None) -> list[Bar]:
    anchor = _anchor_datetime(anchor_ts, bars)
    if anchor is None:
        return bars[:]
    return [bar for bar in bars if bar.start_time >= anchor or bar.end_time >= anchor]


def compute_campaign_avwap_series(bars: list[Bar], anchor_ts: str | None) -> list[tuple[datetime, float]]:
    active = bars_since_anchor(bars, anchor_ts)
    result: list[tuple[datetime, float]] = []
    cum_pv = 0.0
    cum_vol = 0.0
    for bar in active:
        cum_pv += bar.typical_price * bar.volume
        cum_vol += bar.volume
        if cum_vol > 0:
            result.append((bar.end_time, cum_pv / cum_vol))
    return result


def compute_campaign_avwap(bars: list[Bar], anchor_ts: str | None) -> float:
    series = compute_campaign_avwap_series(bars, anchor_ts)
    if not series:
        return 0.0
    return float(series[-1][1])


def compute_weekly_vwap(bars: list[Bar]) -> float:
    if not bars:
        return 0.0
    latest = bars[-1].end_time.astimezone(ET)
    monday = latest.date().toordinal() - latest.weekday()
    active = [bar for bar in bars if bar.end_time.astimezone(ET).date().toordinal() >= monday]
    cum_pv = sum(bar.typical_price * bar.volume for bar in active)
    cum_vol = sum(bar.volume for bar in active)
    return (cum_pv / cum_vol) if cum_vol > 0 else 0.0


def classify_4h_regime(bars_4h: list[Bar]) -> Regime:
    closes = _close_values(bars_4h)
    if len(closes) < 10:
        return Regime.TRANSITIONAL
    ema50 = ema(closes, min(50, len(closes)))
    close = closes[-1]
    slope = ema50[-1] - ema50[max(0, len(ema50) - 4)]
    adx = adx_from_bars(bars_4h[-30:], 14)
    if close > ema50[-1] and slope > 0 and adx >= 20:
        return Regime.BULL
    if close < ema50[-1] and slope < 0 and adx >= 20:
        return Regime.BEAR
    if adx < 18:
        return Regime.CHOP
    return Regime.TRANSITIONAL


def market_regime_from_proxies(
    store: StrategyDataStore,
    proxy_symbols: tuple[str, ...] = ("SPY", "QQQ"),
) -> tuple[Regime, dict[str, str]]:
    proxy_regimes: dict[str, str] = {}
    bulls = 0
    bears = 0
    chops = 0
    for symbol in proxy_symbols:
        bars = store.bars_4h(symbol)
        if not bars:
            continue
        regime = classify_4h_regime(bars)
        proxy_regimes[symbol] = regime.value
        bulls += int(regime == Regime.BULL)
        bears += int(regime == Regime.BEAR)
        chops += int(regime == Regime.CHOP)
    if not proxy_regimes:
        return Regime.TRANSITIONAL, proxy_regimes
    if bulls and not bears:
        return Regime.BULL, proxy_regimes
    if bears and not bulls:
        return Regime.BEAR, proxy_regimes
    if chops == len(proxy_regimes):
        return Regime.CHOP, proxy_regimes
    return Regime.TRANSITIONAL, proxy_regimes


def daily_trend_sign(daily_bars: list[ResearchDailyBar]) -> int:
    closes = _close_values(daily_bars)
    if len(closes) < 50:
        return 0
    ema20_vals = ema(closes, 20)
    ema50_vals = ema(closes, 50)
    if len(closes) < 150:
        if closes[-1] > ema20_vals[-1] > ema50_vals[-1]:
            return 1
        if closes[-1] < ema20_vals[-1] < ema50_vals[-1]:
            return -1
        return 0
    ema150_vals = ema(closes, 150)
    ema50_rising = ema50_vals[-1] > ema50_vals[-4]
    ema50_falling = ema50_vals[-1] < ema50_vals[-4]
    if closes[-1] > ema50_vals[-1] > ema150_vals[-1] and ema50_rising:
        return 1
    if closes[-1] < ema50_vals[-1] < ema150_vals[-1] and ema50_falling:
        return -1
    return 0


def directional_regime_pass(direction: Direction, stock_regime: Regime, market_regime: Regime, trend_sign: int) -> tuple[bool, dict[str, str | int]]:
    hard_block = False
    if direction == Direction.LONG:
        hard_block = stock_regime == Regime.BEAR and market_regime == Regime.BEAR and trend_sign < 0
    else:
        hard_block = stock_regime == Regime.BULL and market_regime == Regime.BULL and trend_sign > 0
    return (not hard_block), {
        "stock_regime": stock_regime.value,
        "market_regime": market_regime.value,
        "daily_trend_sign": trend_sign,
    }


def qualifies_structural_breakout(daily_bars: list[ResearchDailyBar], box: Box) -> Direction | None:
    close = daily_bars[-1].close
    if close > box.high:
        return Direction.LONG
    if close < box.low:
        return Direction.SHORT
    return None


def displacement_pass(daily_bars: list[ResearchDailyBar], bars_30m: list[Bar], direction: Direction, anchor_ts: str | None, settings: StrategySettings) -> tuple[bool, float, float]:
    close_d = float(daily_bars[-1].close)
    atr14 = atr_from_bars(daily_bars, 14)
    atr50 = atr_from_bars(daily_bars, 50)
    avwap_d = compute_campaign_avwap(bars_30m, anchor_ts)
    disp = abs(close_d - avwap_d) / atr14 if atr14 > 0 else 0.0
    history_values: list[float] = []
    for idx in range(55, len(daily_bars)):
        hist_daily = daily_bars[:idx]
        hist_30m = [bar for bar in bars_30m if bar.end_time.date() <= hist_daily[-1].trade_date]
        hist_atr14 = atr_from_bars(hist_daily, 14)
        hist_box = detect_compression_box(hist_daily, settings)
        if hist_atr14 <= 0 or hist_box is None:
            continue
        hist_anchor = f"{hist_box.start_date}T09:30:00-05:00"
        hist_avwap = compute_campaign_avwap(hist_30m, hist_anchor)
        if hist_avwap <= 0:
            continue
        history_values.append(abs(hist_daily[-1].close - hist_avwap) / hist_atr14)
    quantile = settings.base_q_disp + (settings.atr_expansion_q_disp_adj if atr14 > atr50 and atr50 > 0 else 0.0)
    disp_threshold = rolling_quantile(history_values, quantile) if history_values else 0.75
    return disp >= disp_threshold, float(disp), float(disp_threshold)


def breakout_reject(daily_bars: list[ResearchDailyBar], direction: Direction, settings: StrategySettings) -> bool:
    row = daily_bars[-1]
    atr14 = atr_from_bars(daily_bars, 14)
    rvol_d = daily_rvol(daily_bars)
    bar_range = row.high - row.low
    if bar_range <= 0:
        return False
    body = abs(row.close - row.open)
    body_ratio = body / bar_range
    if direction == Direction.LONG:
        adverse_wick = row.high - max(row.open, row.close)
    else:
        adverse_wick = min(row.open, row.close) - row.low
    wick_ratio = adverse_wick / bar_range
    return (
        bar_range > settings.breakout_reject_range_atr * atr14
        and (body_ratio < settings.breakout_reject_min_body_ratio or wick_ratio > settings.breakout_reject_max_wick_ratio)
        and rvol_d > settings.breakout_reject_min_rvol_d
    )


def qualify_breakout(daily_bars: list[ResearchDailyBar], bars_30m: list[Bar], campaign: Campaign, settings: StrategySettings) -> BreakoutQualification | None:
    if campaign.box is None:
        return None
    direction = qualifies_structural_breakout(daily_bars, campaign.box)
    if direction is None:
        return None
    disp_ok, disp_val, disp_th = displacement_pass(daily_bars, bars_30m, direction, campaign.avwap_anchor_ts, settings)
    if not disp_ok or breakout_reject(daily_bars, direction, settings):
        return None
    # Opt 4: Breakout volume confirmation — soft gate
    rvol_d = daily_rvol(daily_bars)
    if rvol_d < settings.breakout_min_rvol_d:
        adjusted_threshold = disp_th * (1.0 + settings.breakout_low_vol_disp_premium)
        if disp_val < adjusted_threshold:
            return None
    if (campaign.reentry_block_opposite_enhanced
            and campaign.breakout is not None
            and direction != campaign.breakout.direction
            and disp_val < 1.10 * disp_th):
        return None
    return BreakoutQualification(
        direction=direction,
        breakout_date=str(daily_bars[-1].trade_date),
        structural_pass=True,
        displacement_pass=True,
        disp_value=disp_val,
        disp_threshold=disp_th,
        breakout_rejected=False,
        rvol_d=rvol_d,
        score_components={
            "sq_good": 1.0 if campaign.box.tier == CompressionTier.GOOD else 0.0,
            "sq_loose": 1.0 if campaign.box.tier == CompressionTier.LOOSE else 0.0,
            "vol_confirmed": 1.0 if rvol_d >= settings.breakout_min_rvol_d else 0.0,
        },
    )


def ttm_squeeze_direction_bonus(closes: list[float], direction: Direction) -> int:
    if len(closes) < 10:
        return 0
    ema20 = ema(closes, min(20, len(closes)))
    momentum = closes[-1] - closes[-5]
    trend_up = closes[-1] > ema20[-1] and ema20[-1] >= ema20[-2]
    trend_down = closes[-1] < ema20[-1] and ema20[-1] <= ema20[-2]
    if direction == Direction.LONG:
        if trend_up and momentum > 0:
            return 1
        if trend_down and momentum < 0:
            return -1
    else:
        if trend_down and momentum < 0:
            return 1
        if trend_up and momentum > 0:
            return -1
    return 0


def determine_intraday_mode(direction: Direction, stock_regime: Regime, market_regime: Regime) -> str:
    if market_regime == Regime.CHOP:
        return "DEGRADED"
    if direction == Direction.LONG and market_regime == Regime.BEAR:
        return "DEGRADED"
    if direction == Direction.SHORT and market_regime == Regime.BULL:
        return "DEGRADED"
    if stock_regime == Regime.CHOP:
        return "DEGRADED"
    return "NORMAL"


def intraday_evidence_score(symbol: str, campaign: Campaign, store: StrategyDataStore, settings: StrategySettings) -> tuple[int, dict[str, int]]:
    bars = store.bars_30m(symbol)
    if len(bars) < 2 or campaign.box is None or campaign.breakout is None:
        return 0, {}
    latest = bars[-1]
    prev = bars[-2]
    avwap = compute_campaign_avwap(bars, campaign.avwap_anchor_ts)
    rvol = intraday_rvol_30m(bars)
    stock_regime = classify_4h_regime(store.bars_4h(symbol))
    market_regime, _ = market_regime_from_proxies(store)
    breakout_level = campaign.box.high if campaign.breakout.direction == Direction.LONG else campaign.box.low
    score = 0
    detail: dict[str, int] = {}
    if campaign.breakout.direction == Direction.LONG and latest.close > avwap:
        score += 1
        detail["above_avwap"] = 1
    elif campaign.breakout.direction == Direction.SHORT and latest.close < avwap:
        score += 1
        detail["below_avwap"] = 1
    clv = close_location_value(latest)
    if campaign.breakout.direction == Direction.LONG and clv < 0.30:
        score -= 1
        detail["weak_reclaim"] = -1
    elif campaign.breakout.direction == Direction.SHORT and clv > 0.70:
        score -= 1
        detail["weak_reclaim"] = -1
    if rvol >= settings.intraday_rvol_strong:
        score += 2
        detail["rvol_30m"] = 2
    elif rvol > settings.intraday_rvol_min:
        score += 1
        detail["rvol_30m"] = 1
    if campaign.breakout.direction == Direction.LONG:
        if prev.close > breakout_level and latest.close > breakout_level:
            score += 1
            detail["two_closes"] = 1
    else:
        if prev.close < breakout_level and latest.close < breakout_level:
            score += 1
            detail["two_closes"] = 1
    if campaign.box.tier == CompressionTier.GOOD:
        score += 1
        detail["sq_good"] = 1
    elif campaign.box.tier == CompressionTier.LOOSE:
        score -= 1
        detail["sq_loose"] = -1
    aligned_4h = (
        campaign.breakout.direction == Direction.LONG
        and stock_regime == Regime.BULL
        and market_regime != Regime.BEAR
    ) or (
        campaign.breakout.direction == Direction.SHORT
        and stock_regime == Regime.BEAR
        and market_regime != Regime.BULL
    )
    if aligned_4h:
        score += 1
        detail["regime_alignment"] = 1
    bonus = ttm_squeeze_direction_bonus(_close_values(bars), campaign.breakout.direction)
    score += bonus
    if bonus:
        detail["ttm_bonus"] = bonus
    # Opt 2: Displacement bonus
    if campaign.breakout.disp_threshold > 0:
        disp_ratio = campaign.breakout.disp_value / campaign.breakout.disp_threshold
        if disp_ratio >= settings.evidence_disp_bonus_threshold:
            score += 1
            detail["strong_displacement"] = 1
    # Opt 6: Time-of-day adjustment
    bar_time = latest.end_time.astimezone(ET).time()
    if bar_time <= settings.early_entry_end and rvol >= settings.early_entry_rvol_bonus_min:
        score += 1
        detail["early_session_strength"] = 1
    elif bar_time >= settings.late_entry_start and rvol < settings.intraday_rvol_min:
        score -= 1
        detail["late_session_penalty"] = -1
    return score, detail


def _market_price_accepts(direction: Direction, latest: Bar, ref: float) -> bool:
    if direction == Direction.LONG:
        return latest.close > ref
    return latest.close < ref


def entry_a_trigger(symbol: str, campaign: Campaign, store: StrategyDataStore) -> float | None:
    bars = store.bars_30m(symbol)
    if not bars or campaign.box is None or campaign.breakout is None:
        return None
    latest = bars[-1]
    avwap = compute_campaign_avwap(bars, campaign.avwap_anchor_ts)
    breakout_level = campaign.box.high if campaign.breakout.direction == Direction.LONG else campaign.box.low
    ref = max(avwap, breakout_level) if campaign.breakout.direction == Direction.LONG else min(avwap, breakout_level)
    if campaign.breakout.direction == Direction.LONG:
        if latest.low <= ref and latest.close > ref:
            return float(ref)
    else:
        if latest.high >= ref and latest.close < ref:
            return float(ref)
    return None


def entry_b_trigger(symbol: str, campaign: Campaign, store: StrategyDataStore, settings: StrategySettings) -> float | None:
    bars = store.bars_30m(symbol)
    if not bars or campaign.breakout is None:
        return None
    latest = bars[-1]
    avwap = compute_campaign_avwap(bars, campaign.avwap_anchor_ts)
    atr_d = atr_from_bars(store.daily_bars(symbol), 14)
    if campaign.breakout.direction == Direction.LONG:
        if latest.low < avwap - (0.25 * atr_d) and latest.close > avwap:
            return float(latest.close)
    else:
        if latest.high > avwap + (0.25 * atr_d) and latest.close < avwap:
            return float(latest.close)
    return None


def entry_c_trigger(symbol: str, campaign: Campaign, store: StrategyDataStore) -> float | None:
    bars = store.bars_30m(symbol)
    if len(bars) < 2 or campaign.box is None or campaign.breakout is None:
        return None
    latest = bars[-1]
    prev = bars[-2]
    atr_d = atr_from_bars(store.daily_bars(symbol), 14)
    breakout_level = campaign.box.high if campaign.breakout.direction == Direction.LONG else campaign.box.low
    if campaign.breakout.direction == Direction.LONG:
        if prev.close > breakout_level and latest.close > breakout_level and latest.low >= breakout_level - (0.25 * atr_d):
            return float(latest.close)
    else:
        if prev.close < breakout_level and latest.close < breakout_level and latest.high <= breakout_level + (0.25 * atr_d):
            return float(latest.close)
    return None


def in_entry_window(now: datetime, settings: StrategySettings) -> bool:
    et_now = now.astimezone(ET).time()
    return settings.first_30m_close <= et_now < settings.entry_end


def is_30m_boundary(now: datetime) -> bool:
    et_now = now.astimezone(ET)
    minute = et_now.minute
    return et_now.second == 0 and minute in {0, 30}


def close_location_value(bar: Bar) -> float:
    width = max(bar.high - bar.low, 1e-9)
    return (bar.close - bar.low) / width


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# Momentum continuation (T1) signals
# ---------------------------------------------------------------------------

def compute_opening_range(bars_5m: list, n_bars: int) -> tuple[float, float, float]:
    """Compute opening range high/low/volume from first N bars of the day.

    Returns (or_high, or_low, or_volume) — returns (0, 0, 0) if insufficient bars.
    """
    if len(bars_5m) < n_bars:
        return 0.0, 0.0, 0.0
    window = bars_5m[:n_bars]
    or_high = max(b.high for b in window)
    or_low = min(b.low for b in window)
    or_volume = sum(b.volume for b in window)
    return or_high, or_low, or_volume


def compute_session_avwap(bars_5m: list, up_to_idx: int) -> float:
    """Cumulative VWAP from session open up to (including) bar index."""
    cum_pv = 0.0
    cum_vol = 0.0
    for bar in bars_5m[: up_to_idx + 1]:
        tp = (bar.high + bar.low + bar.close) / 3.0
        cum_pv += tp * bar.volume
        cum_vol += bar.volume
    return (cum_pv / cum_vol) if cum_vol > 0 else 0.0


def is_momentum_breakout(
    bar,
    prior_day_high: float,
    or_high: float,
    rvol: float,
    cpr: float,
    settings: StrategySettings,
    *,
    use_rvol_filter: bool = True,
    use_cpr_filter: bool = True,
    use_pdh_breakout: bool = True,
) -> tuple[bool, EntryType | None]:
    """Check if bar triggers a momentum breakout.

    Returns (triggered, entry_type) or (False, None).
    """
    if use_rvol_filter and rvol < settings.rvol_threshold:
        return False, None
    if use_cpr_filter and cpr < settings.cpr_threshold:
        return False, None

    above_or = bar.close > or_high
    above_pdh = bar.close > prior_day_high if use_pdh_breakout else False

    if above_or and above_pdh:
        return True, EntryType.COMBINED_BREAKOUT
    if above_or:
        return True, EntryType.OR_BREAKOUT
    if above_pdh:
        return True, EntryType.PDH_BREAKOUT
    return False, None


def compute_momentum_score(
    bar,
    bars_today: list,
    prior_day_high: float,
    prior_day_close: float,
    or_high: float,
    avwap: float,
    adx_val: float,
    sector_flow: float,
    settings: StrategySettings,
    *,
    use_avwap_filter: bool = True,
) -> tuple[int, dict[str, int]]:
    """8-factor momentum scoring. Each factor contributes 0 or 1."""
    score = 0
    detail: dict[str, int] = {}

    # 1. Price > prior day high
    if bar.close > prior_day_high:
        score += 1
        detail["above_pdh"] = 1

    # 2. Price > opening range high
    if bar.close > or_high:
        score += 1
        detail["above_or"] = 1

    # 3. Bar volume surge vs session average
    if len(bars_today) >= 2:
        avg_vol = fmean([b.volume for b in bars_today[:-1]])
        if avg_vol > 0 and bar.volume / avg_vol >= 1.3:
            score += 1
            detail["bar_vol_surge"] = 1

    # 4. CPR ≥ threshold on breakout bar
    bar_range = max(bar.high - bar.low, 1e-9)
    bar_cpr = (bar.close - bar.low) / bar_range
    if bar_cpr >= settings.cpr_threshold:
        score += 1
        detail["strong_cpr"] = 1

    # 5. Price > session AVWAP
    if use_avwap_filter and avwap > 0 and bar.close > avwap:
        score += 1
        detail["above_avwap"] = 1
    elif not use_avwap_filter:
        score += 1
        detail["above_avwap"] = 1

    # 6. ADX > threshold (trending)
    if adx_val >= settings.adx_threshold:
        score += 1
        detail["adx_trending"] = 1

    # 7. Sector flow positive
    if sector_flow > 0:
        score += 1
        detail["sector_flow_pos"] = 1

    # 8. Gap up from prior close
    if bars_today and bars_today[0].open > prior_day_close:
        score += 1
        detail["gap_up"] = 1

    return score, detail


def compute_bar_rvol(bar_volume: float, expected_5m_volume: float) -> float:
    """Relative volume vs expected 5m baseline from ResearchSymbol."""
    if expected_5m_volume <= 0:
        return 0.0
    return bar_volume / expected_5m_volume
