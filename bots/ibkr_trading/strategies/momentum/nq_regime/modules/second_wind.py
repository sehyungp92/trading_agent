from __future__ import annotations

from statistics import fmean
from typing import Any

from strategies.momentum.nq_regime import config
from strategies.momentum.nq_regime.config import Grade, ModuleId, TradeSide
from strategies.momentum.nq_regime.core.filters import room_to_next_level_r
from strategies.momentum.nq_regime.core.indicators import IndicatorSnapshot
from strategies.momentum.nq_regime.core.state import BarData, BarEvent, RegimeCoreState
from strategies.momentum.nq_regime.modules.base import SetupCandidate
from strategies.scalp._shared.nq_contract import round_to_tick
from strategies.scalp._shared.time_utils import to_et


def evaluate(state: RegimeCoreState, event: BarEvent, indicators: IndicatorSnapshot) -> SetupCandidate | None:
    if not _inside_second_wind_window(event.ts):
        return None
    bias, bias_score = evaluate_bias(state, indicators)
    if bias is TradeSide.FLAT:
        return None

    candidates: list[SetupCandidate] = []
    classic = _classic_squeeze_fire(state, event, indicators, bias, bias_score)
    if classic is not None:
        candidates.append(classic)
    if config.SECOND_WIND_VWAP_RECLAIM_ENABLED:
        item = _vwap_reclaim_candidate(state, event, indicators, bias, bias_score)
        if item is not None:
            candidates.append(item)
    if config.SECOND_WIND_MICRO_COMPRESSION_ENABLED:
        item = _micro_compression_candidate(state, event, indicators, bias, bias_score)
        if item is not None:
            candidates.append(item)
    if config.SECOND_WIND_RANGE_ACCEPTANCE_ENABLED:
        item = _range_acceptance_candidate(state, event, indicators, bias, bias_score)
        if item is not None:
            candidates.append(item)
    if config.SECOND_WIND_SECOND_LEG_ENABLED:
        item = _second_leg_candidate(state, event, indicators, bias, bias_score)
        if item is not None:
            candidates.append(item)

    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (candidate.valid, candidate.grade.value == "A+", candidate.score, candidate.target_room_r),
        reverse=True,
    )[0]


def evaluate_bias(state: RegimeCoreState, indicators: IndicatorSnapshot) -> tuple[TradeSide, int]:
    if len(state.bars_15m) < 8:
        return TradeSide.FLAT, 0
    recent = state.bars_15m[-8:]
    closes = [bar.close for bar in recent]
    highs = [bar.high for bar in recent]
    lows = [bar.low for bar in recent]
    above_vwap = closes[-1] > indicators.vwap
    below_vwap = closes[-1] < indicators.vwap
    am_control = indicators.am_vwap_control
    hh_hl = highs[-1] >= max(highs[-6:-1]) and lows[-1] >= min(lows[-6:-1])
    ll_lh = lows[-1] <= min(lows[-6:-1]) and highs[-1] <= max(highs[-6:-1])
    ema_bull = indicators.ema20_15m >= indicators.ema50_15m
    ema_bear = indicators.ema20_15m <= indicators.ema50_15m
    fast_bull = indicators.ema9_15m >= indicators.ema20_15m
    fast_bear = indicators.ema9_15m <= indicators.ema20_15m
    pullback_bull = abs(closes[-1] - indicators.ema9_15m) <= max(indicators.atr_15m * 0.5, 1.0)
    pullback_bear = pullback_bull
    compression_near_ema = abs(closes[-1] - indicators.ema20_15m) <= max(indicators.atr_15m, 1.0)

    bull_score = sum(
        [
            above_vwap,
            am_control > 0,
            hh_hl,
            ema_bull,
            fast_bull,
            pullback_bull or compression_near_ema,
            _had_impulse(state, TradeSide.LONG),
        ]
    )
    bear_score = sum(
        [
            below_vwap,
            am_control < 0,
            ll_lh,
            ema_bear,
            fast_bear,
            pullback_bear or compression_near_ema,
            _had_impulse(state, TradeSide.SHORT),
        ]
    )
    if bull_score > bear_score and bull_score >= 4:
        return TradeSide.LONG, int(bull_score)
    if bear_score > bull_score and bear_score >= 4:
        return TradeSide.SHORT, int(bear_score)
    return TradeSide.FLAT, int(max(bull_score, bear_score))


def _classic_squeeze_fire(
    state: RegimeCoreState,
    event: BarEvent,
    indicators: IndicatorSnapshot,
    bias: TradeSide,
    bias_score: int,
) -> SetupCandidate | None:
    if not event.is_new_15m or event.bar_15m_closed is None or len(state.bars_15m) < 20:
        return None
    bar = event.bar_15m_closed
    squeeze_duration = max(indicators.squeeze_duration, state.second_wind_state.squeeze_duration)
    if squeeze_duration < config.SECOND_WIND_MIN_SQUEEZE_BARS:
        return None
    recent_squeeze = state.bars_15m[-(squeeze_duration + 1):-1] if squeeze_duration > 0 else state.bars_15m[-6:-1]
    if len(recent_squeeze) < 3:
        return None
    squeeze_high = max(sample.high for sample in recent_squeeze)
    squeeze_low = min(sample.low for sample in recent_squeeze)
    if bias is TradeSide.LONG and not (bar.high > squeeze_high and bar.close > squeeze_high):
        return None
    if bias is TradeSide.SHORT and not (bar.low < squeeze_low and bar.close < squeeze_low):
        return None

    close_loc = bar.close_location_long if bias is TradeSide.LONG else bar.close_location_short
    score = bias_score
    score += 2 if 5 <= squeeze_duration <= 8 else 1 if 3 <= squeeze_duration <= 10 else 0
    score += 1 if close_loc >= config.SECOND_WIND_TRIGGER_CLOSE_MIN else 0
    score += 1 if indicators.volume_multiple_15m >= 1.2 else 0
    score += 1 if bar.range_pts >= 0.75 * max(indicators.atr_15m, 1.0) else 0
    return _build_candidate(
        state,
        bar,
        indicators,
        bias,
        setup_type="pm_squeeze_fire",
        level=squeeze_high if bias is TradeSide.LONG else squeeze_low,
        score=score,
        details={
            "squeeze_duration": squeeze_duration,
            "squeeze_high": squeeze_high,
            "squeeze_low": squeeze_low,
            "close_location": close_loc,
            "volume_multiple": indicators.volume_multiple_15m,
            "bias_score": bias_score,
        },
        structural_low=squeeze_low,
        structural_high=squeeze_high,
    )


def _vwap_reclaim_candidate(
    state: RegimeCoreState,
    event: BarEvent,
    indicators: IndicatorSnapshot,
    bias: TradeSide,
    bias_score: int,
) -> SetupCandidate | None:
    if len(state.bars_5m) < 8 or indicators.vwap <= 0:
        return None
    bar = event.bar_5m
    prev = state.bars_5m[-2]
    if bias is TradeSide.LONG and not (prev.close <= indicators.vwap and bar.close > indicators.vwap):
        return None
    if bias is TradeSide.SHORT and not (prev.close >= indicators.vwap and bar.close < indicators.vwap):
        return None
    close_loc = bar.close_location_long if bias is TradeSide.LONG else bar.close_location_short
    reclaim_distance = abs(bar.close - indicators.vwap)
    ema_aligned = _ema_aligned(indicators, bias)
    score = bias_score + 1
    score += 1 if close_loc >= 0.65 else 0
    score += 1 if indicators.volume_multiple_5m >= 1.0 else 0
    score += 1 if bias.sign * indicators.vwap_slope >= -max(indicators.atr_5m * 0.05, 0.25) else 0
    score += 1 if ema_aligned else 0
    return _build_candidate(
        state,
        bar,
        indicators,
        bias,
        setup_type="pm_vwap_reclaim",
        level=indicators.vwap,
        score=score,
        details={
            "close_location": close_loc,
            "volume_multiple": indicators.volume_multiple_5m,
            "bias_score": bias_score,
            "vwap": indicators.vwap,
            "reclaim_distance": reclaim_distance,
            "ema_aligned": ema_aligned,
        },
    )


def _micro_compression_candidate(
    state: RegimeCoreState,
    event: BarEvent,
    indicators: IndicatorSnapshot,
    bias: TradeSide,
    bias_score: int,
) -> SetupCandidate | None:
    if len(state.bars_5m) < 10:
        return None
    bar = event.bar_5m
    sample = state.bars_5m[-7:-1]
    if len(sample) < 5:
        return None
    avg_range = fmean([sample_bar.range_pts for sample_bar in sample])
    if avg_range > max(indicators.atr_5m * config.SECOND_WIND_MICRO_RANGE_ATR_MULT, 1.0):
        return None
    high = max(sample_bar.high for sample_bar in sample)
    low = min(sample_bar.low for sample_bar in sample)
    if bias is TradeSide.LONG and not (bar.high > high and bar.close > high):
        return None
    if bias is TradeSide.SHORT and not (bar.low < low and bar.close < low):
        return None
    close_loc = bar.close_location_long if bias is TradeSide.LONG else bar.close_location_short
    score = bias_score + 2
    score += 1 if close_loc >= 0.65 else 0
    score += 1 if indicators.volume_multiple_5m >= 0.9 else 0
    score += 1 if _ema_aligned(indicators, bias) else 0
    return _build_candidate(
        state,
        bar,
        indicators,
        bias,
        setup_type="pm_micro_compression_break",
        level=high if bias is TradeSide.LONG else low,
        score=score,
        details={
            "compression_range": avg_range,
            "compression_high": high,
            "compression_low": low,
            "close_location": close_loc,
            "volume_multiple": indicators.volume_multiple_5m,
            "bias_score": bias_score,
        },
        structural_low=low,
        structural_high=high,
    )


def _range_acceptance_candidate(
    state: RegimeCoreState,
    event: BarEvent,
    indicators: IndicatorSnapshot,
    bias: TradeSide,
    bias_score: int,
) -> SetupCandidate | None:
    if not state.ib_locked or len(state.bars_5m) < 8:
        return None
    bar = event.bar_5m
    prev = state.bars_5m[-2]
    level = state.ib_levels.high if bias is TradeSide.LONG else state.ib_levels.low
    if bias is TradeSide.LONG and not (prev.close > level and bar.close > level and bar.close > indicators.vwap):
        return None
    if bias is TradeSide.SHORT and not (prev.close < level and bar.close < level and bar.close < indicators.vwap):
        return None
    close_loc = bar.close_location_long if bias is TradeSide.LONG else bar.close_location_short
    score = bias_score + 2
    score += 1 if close_loc >= 0.60 else 0
    score += 1 if indicators.volume_multiple_5m >= 0.85 else 0
    score += 1 if _ema_aligned(indicators, bias) else 0
    return _build_candidate(
        state,
        bar,
        indicators,
        bias,
        setup_type="pm_range_acceptance",
        level=level,
        score=score,
        details={
            "ib_high": state.ib_levels.high,
            "ib_low": state.ib_levels.low,
            "close_location": close_loc,
            "volume_multiple": indicators.volume_multiple_5m,
            "bias_score": bias_score,
        },
    )


def _second_leg_candidate(
    state: RegimeCoreState,
    event: BarEvent,
    indicators: IndicatorSnapshot,
    bias: TradeSide,
    bias_score: int,
) -> SetupCandidate | None:
    if len(state.bars_5m) < 12:
        return None
    bar = event.bar_5m
    lookback = max(4, config.SECOND_WIND_PULLBACK_LOOKBACK_BARS)
    sample = state.bars_5m[-(lookback + 2):-1]
    if len(sample) < 5:
        return None
    ema = indicators.ema9_15m or indicators.ema20_15m or bar.close
    pullback_buffer = max(
        indicators.atr_5m * config.SECOND_WIND_SECOND_LEG_PULLBACK_BUFFER_ATR_MULT,
        config.SECOND_WIND_SECOND_LEG_PULLBACK_BUFFER_MIN_PTS,
    )
    if bias is TradeSide.LONG:
        had_pullback = any(sample_bar.low <= ema + pullback_buffer for sample_bar in sample)
        level = max(sample_bar.high for sample_bar in sample[-3:])
        if not (had_pullback and bar.high > level and bar.close > level):
            return None
        breakout_distance = bar.close - level
    else:
        had_pullback = any(sample_bar.high >= ema - pullback_buffer for sample_bar in sample)
        level = min(sample_bar.low for sample_bar in sample[-3:])
        if not (had_pullback and bar.low < level and bar.close < level):
            return None
        breakout_distance = level - bar.close
    close_loc = bar.close_location_long if bias is TradeSide.LONG else bar.close_location_short
    had_impulse = _had_impulse(state, bias)
    score = bias_score + 2
    score += 1 if close_loc >= 0.65 else 0
    score += 1 if indicators.volume_multiple_5m >= 0.85 else 0
    score += 1 if had_impulse else 0
    return _build_candidate(
        state,
        bar,
        indicators,
        bias,
        setup_type="pm_second_leg",
        level=level,
        score=score,
        details={
            "pullback_ema": ema,
            "pullback_buffer": pullback_buffer,
            "close_location": close_loc,
            "volume_multiple": indicators.volume_multiple_5m,
            "bias_score": bias_score,
            "breakout_distance": max(0.0, breakout_distance),
            "had_impulse": had_impulse,
        },
    )


def _build_candidate(
    state: RegimeCoreState,
    bar: BarData,
    indicators: IndicatorSnapshot,
    side: TradeSide,
    *,
    setup_type: str,
    level: float,
    score: int,
    details: dict[str, Any],
    structural_low: float | None = None,
    structural_high: float | None = None,
) -> SetupCandidate | None:
    entry, stop, entry_model = _entry_and_stop(state, bar, side, level, indicators, structural_low, structural_high)
    risk = abs(entry - stop)
    if risk <= 0:
        return None

    vetoes: list[str] = []
    pm_score = _pm_score(state, indicators)
    setup_min_score = max(config.SECOND_WIND_MIN_SCORE, _setup_min_score(setup_type))
    setup_min_volume = max(config.SECOND_WIND_MIN_VOLUME_MULTIPLE, _setup_min_volume(setup_type))
    setup_min_close_location = _setup_min_close_location(setup_type)
    pm_floor = (
        config.SECOND_WIND_PM_TRANSITION_MIN_SCORE
        if getattr(state.regime, "name", "") == "TRANSITION" or _setup_requires_pm_transition(setup_type)
        else config.SECOND_WIND_MIN_PM_SCORE
    )
    pm_floor = max(pm_floor, _setup_min_pm_score(setup_type))
    if pm_score < pm_floor:
        vetoes.append("weak_pm_continuation_score")
    if score < setup_min_score:
        vetoes.append("weak_setup_score")
    if details.get("close_location", 1.0) < config.SECOND_WIND_TRIGGER_CLOSE_MIN and setup_type == "pm_squeeze_fire":
        vetoes.append("weak_trigger_close")
    if setup_min_close_location > 0 and details.get("close_location", 1.0) < setup_min_close_location:
        vetoes.append("weak_setup_close_location")
    if details.get("volume_multiple", 1.0) < setup_min_volume:
        vetoes.append("weak_volume")
    vetoes.extend(_setup_specific_vetoes(setup_type, details, indicators))
    stop_cap = min(config.SECOND_WIND_STOP_CAP, config.SECOND_WIND_MAX_STOP_PTS)
    if risk > stop_cap:
        vetoes.append("stop_exceeds_cap")

    measured = max(
        (structural_high - structural_low) if structural_low is not None and structural_high is not None else 0.0,
        risk * 2.0,
    )
    fallback = entry + side.sign * measured
    room_r = room_to_next_level_r(
        side=side,
        entry=entry,
        stop=stop,
        levels=state.levels,
        fallback_target=fallback,
    )
    if room_r < config.TARGET_ROOM_MIN_R:
        vetoes.append("insufficient_room")

    grade = _grade(score, vetoes)
    targets = (
        round_to_tick(entry + side.sign * risk, config.TICK_SIZE),
        round_to_tick(entry + side.sign * max(2.0 * risk, measured), config.TICK_SIZE),
        round_to_tick(entry + side.sign * max(2.5 * risk, measured * 1.5), config.TICK_SIZE),
    )
    invalidation = stop if config.SECOND_WIND_ENTRY_STOP_INVALIDATION_ENABLED else (
        structural_low if side is TradeSide.LONG else structural_high
    )
    if invalidation is None:
        invalidation = stop
    enriched = {
        **details,
        "pm_score": pm_score,
        "pm_score_floor": pm_floor,
        "setup_score_floor": setup_min_score,
        "volume_multiple_floor": setup_min_volume,
        "close_location_floor": setup_min_close_location,
        "entry_source_bar": bar.ts.isoformat(),
    }
    return SetupCandidate(
        candidate_id=f"nqreg-sw-{setup_type}-{bar.ts.strftime('%Y%m%d%H%M')}-{side.value}",
        module=ModuleId.SECOND_WIND,
        side=side,
        setup_type=setup_type,
        timestamp=bar.ts,
        level=level,
        score=score,
        grade=grade,
        entry_price=entry,
        stop_price=stop,
        targets=targets,
        entry_model=entry_model,
        risk_pct=0.0,
        invalidation_price=invalidation,
        target_room_r=room_r,
        vetoes=tuple(vetoes),
        details=enriched,
    )


def _entry_and_stop(
    state: RegimeCoreState,
    bar: BarData,
    side: TradeSide,
    level: float,
    indicators: IndicatorSnapshot,
    structural_low: float | None = None,
    structural_high: float | None = None,
) -> tuple[float, float, str]:
    atr_stop = config.SECOND_WIND_ATR_STOP_MULT * max(indicators.atr_15m, 4.0)
    recent = state.bars_5m[-6:] if state.bars_5m else []
    if side is TradeSide.LONG:
        structural_stop = structural_low if structural_low is not None else min((sample.low for sample in recent), default=bar.low)
    else:
        structural_stop = structural_high if structural_high is not None else max((sample.high for sample in recent), default=bar.high)

    mode = config.SECOND_WIND_ENTRY_MODEL
    if mode == "trigger_close":
        entry = bar.close
        entry_model = _close_entry_model_for_level(level, bar, side)
    elif mode == "breakout_stop":
        entry = bar.high + config.TICK_SIZE if side is TradeSide.LONG else bar.low - config.TICK_SIZE
        entry_model = "momentum_breakout"
    elif mode == "trigger_midpoint":
        midpoint = (bar.open + bar.close) / 2.0
        entry = max(level, min(bar.close, midpoint)) if side is TradeSide.LONG else min(level, max(bar.close, midpoint))
        entry_model = "pullback_after_fire"
    elif mode == "ema_pullback":
        ema = indicators.ema9_15m or indicators.ema20_15m or bar.close
        entry = max(level, min(bar.close, ema)) if side is TradeSide.LONG else min(level, max(bar.close, ema))
        entry_model = "ema_pullback_after_fire"
    else:
        entry = level
        entry_model = "breakout_close_retest"

    entry = round_to_tick(entry, config.TICK_SIZE, "up" if side is TradeSide.LONG and mode == "breakout_stop" else "down" if side is TradeSide.SHORT and mode == "breakout_stop" else "nearest")
    if side is TradeSide.LONG:
        recent_stop = min((sample.low for sample in recent), default=structural_stop) - 2 * config.TICK_SIZE
        stop = max(recent_stop, entry - atr_stop)
        stop = min(stop, entry - config.TICK_SIZE)
        if structural_low is not None:
            stop = max(stop, structural_low - config.TICK_SIZE)
        return entry, round_to_tick(stop, config.TICK_SIZE, "down"), entry_model

    recent_stop = max((sample.high for sample in recent), default=structural_stop) + 2 * config.TICK_SIZE
    stop = min(recent_stop, entry + atr_stop)
    stop = max(stop, entry + config.TICK_SIZE)
    if structural_high is not None:
        stop = min(stop, structural_high + config.TICK_SIZE)
    return entry, round_to_tick(stop, config.TICK_SIZE, "up"), entry_model


def _close_entry_model_for_level(level: float, bar: BarData, side: TradeSide) -> str:
    del level
    if side is TradeSide.LONG and bar.open <= bar.close:
        return "sw_reclaim_close"
    if side is TradeSide.SHORT and bar.open >= bar.close:
        return "sw_reclaim_close"
    return "breakout_close"


def _setup_min_score(setup_type: str) -> int:
    if setup_type == "pm_vwap_reclaim":
        return config.SECOND_WIND_VWAP_RECLAIM_MIN_SCORE
    if setup_type == "pm_second_leg":
        return config.SECOND_WIND_SECOND_LEG_MIN_SCORE
    return 0


def _setup_min_pm_score(setup_type: str) -> float:
    if setup_type == "pm_vwap_reclaim":
        return config.SECOND_WIND_VWAP_RECLAIM_MIN_PM_SCORE
    if setup_type == "pm_second_leg":
        return config.SECOND_WIND_SECOND_LEG_MIN_PM_SCORE
    return 0.0


def _setup_min_volume(setup_type: str) -> float:
    if setup_type == "pm_vwap_reclaim":
        return config.SECOND_WIND_VWAP_RECLAIM_MIN_VOLUME_MULTIPLE
    if setup_type == "pm_second_leg":
        return config.SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE
    return 0.0


def _setup_min_close_location(setup_type: str) -> float:
    if setup_type == "pm_vwap_reclaim":
        return config.SECOND_WIND_VWAP_RECLAIM_MIN_CLOSE_LOCATION
    if setup_type == "pm_second_leg":
        return config.SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION
    return 0.0


def _setup_requires_pm_transition(setup_type: str) -> bool:
    if setup_type == "pm_vwap_reclaim":
        return config.SECOND_WIND_VWAP_RECLAIM_REQUIRE_PM_TRANSITION
    if setup_type == "pm_second_leg":
        return config.SECOND_WIND_SECOND_LEG_REQUIRE_PM_TRANSITION
    return False


def _setup_specific_vetoes(
    setup_type: str,
    details: dict[str, Any],
    indicators: IndicatorSnapshot,
) -> list[str]:
    vetoes: list[str] = []
    if setup_type == "pm_vwap_reclaim":
        min_reclaim = max(
            config.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_PTS,
            config.SECOND_WIND_VWAP_RECLAIM_MIN_RECLAIM_ATR * max(indicators.atr_5m, 1.0),
        )
        max_reclaim = config.SECOND_WIND_VWAP_RECLAIM_MAX_RECLAIM_ATR * max(indicators.atr_5m, 1.0)
        reclaim_distance = float(details.get("reclaim_distance", 0.0) or 0.0)
        if min_reclaim > 0 and reclaim_distance < min_reclaim:
            vetoes.append("weak_vwap_reclaim_distance")
        if max_reclaim < 999.0 * max(indicators.atr_5m, 1.0) and reclaim_distance > max_reclaim:
            vetoes.append("chasing_vwap_reclaim")
        if config.SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT and not bool(details.get("ema_aligned")):
            vetoes.append("vwap_reclaim_ema_misaligned")
    elif setup_type == "pm_second_leg":
        min_breakout = max(
            config.SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_PTS,
            config.SECOND_WIND_SECOND_LEG_MIN_BREAKOUT_ATR * max(indicators.atr_5m, 1.0),
        )
        breakout_distance = float(details.get("breakout_distance", 0.0) or 0.0)
        if min_breakout > 0 and breakout_distance < min_breakout:
            vetoes.append("weak_second_leg_breakout")
        if config.SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE and not bool(details.get("had_impulse")):
            vetoes.append("second_leg_missing_impulse")
    return vetoes


def _inside_second_wind_window(ts) -> bool:
    et = to_et(ts)
    minute = et.hour * 60 + et.minute
    return config.SECOND_WIND_MIN_ENTRY_MINUTE_ET <= minute <= config.SECOND_WIND_MAX_ENTRY_MINUTE_ET


def _pm_score(state: RegimeCoreState, indicators: IndicatorSnapshot) -> float:
    if len(state.bars_15m) < 10:
        return 0.0
    trend = abs(indicators.trend_direction)
    squeeze = min(1.0, max(indicators.squeeze_duration, state.second_wind_state.squeeze_duration) / 5.0)
    control = min(1.0, abs(indicators.am_vwap_control) * 2.0)
    return max(0.0, min(1.0, 0.45 * trend + 0.35 * squeeze + 0.20 * control))


def _ema_aligned(indicators: IndicatorSnapshot, side: TradeSide) -> bool:
    if side is TradeSide.LONG:
        return indicators.ema9_15m >= indicators.ema20_15m >= indicators.ema50_15m
    return indicators.ema9_15m <= indicators.ema20_15m <= indicators.ema50_15m


def _had_impulse(state: RegimeCoreState, side: TradeSide) -> bool:
    if len(state.bars_15m) < 8:
        return False
    sample = state.bars_15m[-8:-4]
    ranges = [bar.range_pts for bar in sample if bar.range_pts > 0]
    avg_range = fmean(ranges) if ranges else 0.0
    if avg_range <= 0:
        return False
    if side is TradeSide.LONG:
        return any(bar.close > bar.open and bar.range_pts >= 1.2 * avg_range for bar in sample)
    return any(bar.close < bar.open and bar.range_pts >= 1.2 * avg_range for bar in sample)


def _grade(score: int, vetoes: list[str]) -> Grade:
    if vetoes:
        return Grade.INVALID
    if score >= config.SECOND_WIND_A_PLUS_SCORE:
        return Grade.A_PLUS
    if score >= config.SECOND_WIND_A_SCORE:
        return Grade.A
    if score >= config.SECOND_WIND_MIN_SCORE:
        return Grade.B
    return Grade.INVALID
