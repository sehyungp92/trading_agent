from __future__ import annotations

from statistics import fmean

from strategies.momentum.nq_regime import config
from strategies.momentum.nq_regime.config import Grade, IBType, ModuleId, TradeSide
from strategies.momentum.nq_regime.core.filters import room_to_next_level_r
from strategies.momentum.nq_regime.core.indicators import IndicatorSnapshot
from strategies.momentum.nq_regime.core.session import SessionPhase
from strategies.momentum.nq_regime.core.state import BarData, BarEvent, RegimeCoreState
from strategies.momentum.nq_regime.modules.base import SetupCandidate
from strategies.scalp._shared.nq_contract import round_to_tick
from strategies.scalp._shared.time_utils import to_et


def evaluate(state: RegimeCoreState, event: BarEvent, indicators: IndicatorSnapshot) -> SetupCandidate | None:
    if not state.ib_locked:
        return None
    if not event.is_new_15m or event.bar_15m_closed is None:
        if config.STRUCTURAL_PULLBACK_RECLAIM_ENABLED:
            candidate = _pullback_reclaim_candidate(state, event, indicators)
            if candidate is not None:
                return candidate
        if config.STRUCTURAL_CONTINUATION_ENABLED:
            return _continuation_candidate(state, event, indicators)
        return None
    bar = event.bar_15m_closed
    if bar.close > state.ib_levels.high:
        side = TradeSide.LONG
        level = state.ib_levels.high
    elif bar.close < state.ib_levels.low:
        side = TradeSide.SHORT
        level = state.ib_levels.low
    else:
        return None
    if (
        config.STRUCTURAL_BLOCK_OPPOSITE_PM_BREAKOUT
        and state.phase in {SessionPhase.PM_CONTINUATION, SessionPhase.LATE_PM_RESTRICTED}
        and state.expansion_state.active_break_side is not side
    ):
        return None
    score, vetoes, details = score_breakout(state, bar, side, indicators)
    _apply_structural_quality_vetoes(state, bar, side, score, vetoes, indicators)
    context_tracking_enabled = config.STRUCTURAL_PULLBACK_RECLAIM_ENABLED or config.STRUCTURAL_CONTINUATION_ENABLED
    if context_tracking_enabled and _breakout_context_valid(score, vetoes) and state.phase is SessionPhase.PRIMARY_DECISION:
        _record_active_break(state, bar, side, level, score)
    entry, entry_model, entry_vetoes = _initial_entry(state, bar, side, level, indicators, score)
    vetoes.extend(entry_vetoes)
    stop = _structural_stop(state, bar, side, level, indicators, entry)
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    if entry_model in {"fvg_retest", "midpoint_retest", "adaptive_retest"}:
        entry_distance_r = abs(bar.close - entry) / risk
        details["entry_distance_r"] = entry_distance_r
        if entry_distance_r > config.STRUCTURAL_RETEST_ENTRY_MAX_DISTANCE_R:
            vetoes.append("entry_too_far_from_signal")
    if risk < config.STRUCTURAL_MIN_STOP_PTS:
        vetoes.append("structural_stop_too_tight")
    if risk > config.STRUCTURAL_MAX_STOP_PTS:
        vetoes.append("structural_stop_too_wide")
    targets = _targets(state.ib_type, entry, risk, side)
    room_r = room_to_next_level_r(
        side=side,
        entry=entry,
        stop=stop,
        levels=state.levels,
        fallback_target=targets[-1],
    )
    if room_r < config.TARGET_ROOM_MIN_R:
        vetoes.append("insufficient_room")
    if room_r > config.STRUCTURAL_TARGET_ROOM_MAX_R:
        vetoes.append("room_too_open")
    grade = _grade(score, vetoes)
    if (
        not context_tracking_enabled
        and grade not in {Grade.C, Grade.INVALID}
        and state.phase is SessionPhase.PRIMARY_DECISION
    ):
        _record_active_break(state, bar, side, level, score)
    if not config.STRUCTURAL_ACCEPTANCE_ENTRY_ENABLED:
        return None
    return SetupCandidate(
        candidate_id=f"nqreg-exp-{bar.ts.strftime('%Y%m%d%H%M')}-{side.value}",
        module=ModuleId.STRUCTURAL_EXPANSION,
        side=side,
        setup_type="ib_acceptance_breakout",
        timestamp=bar.ts,
        level=level,
        score=score,
        grade=grade,
        entry_price=round_to_tick(entry, config.TICK_SIZE),
        stop_price=stop,
        targets=targets,
        entry_model=entry_model,
        risk_pct=0.0,
        invalidation_price=state.ib_levels.mid,
        target_room_r=room_r,
        vetoes=tuple(vetoes),
        details=details,
    ) 


def _breakout_context_valid(score: int, vetoes: list[str]) -> bool:
    context_vetoes = {
        "wick_only_or_body_not_beyond_ib",
        "weak_breakout_candle",
        "vwap_misaligned",
        "overextended_from_ib",
        "structural_time_block",
        "second_acceptance_missing",
        "long_score_below_floor",
        "short_score_below_floor",
        "live_breadth_contradiction",
    }
    return score >= config.STRUCTURAL_MIN_SCORE and not any(veto in context_vetoes for veto in vetoes)


def _record_active_break(
    state: RegimeCoreState,
    bar: BarData,
    side: TradeSide,
    level: float,
    score: int,
) -> None:
    fvg = _fvg_zone(state.bars_15m[-3:], side)
    state.expansion_state.active_break_side = side
    state.expansion_state.active_break_ts = bar.ts
    state.expansion_state.active_break_bar_index = state.bar_index
    state.expansion_state.active_break_level = level
    state.expansion_state.active_break_high = bar.high
    state.expansion_state.active_break_low = bar.low
    state.expansion_state.active_break_midpoint = _breakout_midpoint(bar)
    state.expansion_state.active_break_fvg_low = fvg[0] if fvg else 0.0
    state.expansion_state.active_break_fvg_high = fvg[1] if fvg else 0.0
    state.expansion_state.active_break_score = score
    state.expansion_state.retest_expiry_bar = state.bar_index + 6
    state.expansion_state.pullback_side = TradeSide.FLAT
    state.expansion_state.pullback_ts = None
    state.expansion_state.pullback_bar_index = -1
    state.expansion_state.pullback_reference = 0.0
    state.expansion_state.pullback_extreme = 0.0
    state.expansion_state.pullback_trigger = 0.0


def _initial_entry(
    state: RegimeCoreState,
    bar: BarData,
    side: TradeSide,
    level: float,
    indicators: IndicatorSnapshot,
    score: int,
) -> tuple[float, str, list[str]]:
    mode = config.STRUCTURAL_ENTRY_MODE
    if mode == "hybrid_close_adaptive":
        close_entry = round_to_tick(bar.close, config.TICK_SIZE)
        close_stop = _structural_stop(state, bar, side, level, indicators, close_entry)
        close_risk = abs(close_entry - close_stop)
        if (
            score >= config.STRUCTURAL_HYBRID_CLOSE_MIN_SCORE
            and close_risk > 0
            and close_risk <= config.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS
        ):
            return close_entry, "breakout_close", []
        fvg_entry = _fvg_retest_entry(state, bar, side)
        if fvg_entry is not None:
            return fvg_entry, "adaptive_retest", []
        return _ib_retest_entry(side, level), "adaptive_retest", []
    if mode == "breakout_close":
        return round_to_tick(bar.close, config.TICK_SIZE), "breakout_close", []
    if mode == "acceptance_stop":
        stop_offset = side.sign * config.STRUCTURAL_STOP_ENTRY_OFFSET_TICKS * config.TICK_SIZE
        raw = (bar.high + stop_offset) if side is TradeSide.LONG else (bar.low + stop_offset)
        return round_to_tick(raw, config.TICK_SIZE), "momentum_breakout", []
    if mode == "fvg_retest" or (mode == "adaptive_retest" and config.STRUCTURAL_ADAPTIVE_RETEST_PREFERS_FVG):
        fvg_entry = _fvg_retest_entry(state, bar, side)
        if fvg_entry is not None:
            return fvg_entry, "fvg_retest" if mode == "fvg_retest" else "adaptive_retest", []
        if mode == "fvg_retest":
            return _ib_retest_entry(side, level), "fvg_retest", ["fvg_retest_missing"]
    if mode == "midpoint_retest" or config.STRUCTURAL_MIDPOINT_RETEST_ENABLED:
        return _midpoint_retest_entry(bar, side, level), "midpoint_retest" if mode == "midpoint_retest" else "adaptive_retest", []
    if mode == "retest":
        return _ib_retest_entry(side, level), "retest", []
    if score >= config.STRUCTURAL_A_PLUS_SCORE and state.ib_type is not IBType.WIDE:
        return round_to_tick(bar.close, config.TICK_SIZE), "breakout_close", []
    if config.STRUCTURAL_FVG_RETEST_ENABLED:
        fvg_entry = _fvg_retest_entry(state, bar, side)
        if fvg_entry is not None:
            return fvg_entry, "fvg_retest", []
    return _ib_retest_entry(side, level), "retest", []


def _ib_retest_entry(side: TradeSide, level: float) -> float:
    retest_offset = side.sign * config.STRUCTURAL_RETEST_OFFSET_TICKS * config.TICK_SIZE
    return round_to_tick(level + retest_offset, config.TICK_SIZE)


def _midpoint_retest_entry(bar: BarData, side: TradeSide, level: float) -> float:
    midpoint = _breakout_midpoint(bar)
    level_ref = level + side.sign * config.STRUCTURAL_RETEST_OFFSET_TICKS * config.TICK_SIZE
    if side is TradeSide.LONG:
        raw = max(level_ref, min(bar.close - config.TICK_SIZE, midpoint))
        return round_to_tick(raw, config.TICK_SIZE, "down")
    raw = min(level_ref, max(bar.close + config.TICK_SIZE, midpoint))
    return round_to_tick(raw, config.TICK_SIZE, "up")


def _fvg_retest_entry(state: RegimeCoreState, bar: BarData, side: TradeSide) -> float | None:
    zone = _fvg_zone(state.bars_15m[-3:], side)
    if zone is None:
        return None
    low, high = zone
    if high - low > config.STRUCTURAL_FVG_RETEST_MAX_GAP_PTS:
        return None
    offset = side.sign * config.STRUCTURAL_RETEST_OFFSET_TICKS * config.TICK_SIZE
    if side is TradeSide.LONG:
        raw = min(bar.close - config.TICK_SIZE, high + offset)
        return round_to_tick(raw, config.TICK_SIZE, "down")
    raw = max(bar.close + config.TICK_SIZE, low + offset)
    return round_to_tick(raw, config.TICK_SIZE, "up")


def _breakout_midpoint(bar: BarData) -> float:
    return (bar.high + bar.low) / 2.0


def _pullback_reclaim_candidate(
    state: RegimeCoreState,
    event: BarEvent,
    indicators: IndicatorSnapshot,
) -> SetupCandidate | None:
    if state.phase not in {SessionPhase.PRIMARY_DECISION, SessionPhase.PM_CONTINUATION, SessionPhase.LATE_PM_RESTRICTED}:
        return None
    bar = event.bar_5m
    if not _entry_time_allowed(bar):
        return None
    side = state.expansion_state.active_break_side
    if side is TradeSide.FLAT or state.expansion_state.active_break_ts is None:
        return None
    age_minutes = (bar.ts - state.expansion_state.active_break_ts).total_seconds() / 60.0
    if age_minutes < 0 or age_minutes > config.STRUCTURAL_PULLBACK_RECLAIM_MAX_AGE_MINUTES:
        return None
    if _accepted_side_from_recent_15m(state, indicators) is not side:
        return None

    level = state.ib_levels.high if side is TradeSide.LONG else state.ib_levels.low
    reference = _pullback_reference(bar, side, level, indicators)
    if reference <= 0:
        return None
    band = _pullback_band(indicators)
    close_loc = bar.close_location_long if side is TradeSide.LONG else bar.close_location_short
    if side is TradeSide.LONG:
        touched = bar.low <= reference + band and bar.close > state.ib_levels.mid
        reclaimed = bar.close > reference + config.TICK_SIZE and bar.close > state.ib_levels.high
        extreme = bar.low
    else:
        touched = bar.high >= reference - band and bar.close < state.ib_levels.mid
        reclaimed = bar.close < reference - config.TICK_SIZE and bar.close < state.ib_levels.low
        extreme = bar.high

    if touched:
        state.expansion_state.pullback_side = side
        state.expansion_state.pullback_ts = bar.ts
        state.expansion_state.pullback_bar_index = state.bar_index
        state.expansion_state.pullback_reference = reference
        state.expansion_state.pullback_extreme = extreme

    prior_pullback = state.expansion_state.pullback_side is side and state.expansion_state.pullback_bar_index >= 0
    pullback_age_bars = state.bar_index - state.expansion_state.pullback_bar_index if prior_pullback else 9999
    if not touched and (not prior_pullback or pullback_age_bars > config.STRUCTURAL_PULLBACK_RECLAIM_MAX_PULLBACK_BARS):
        return None
    if not reclaimed:
        return None

    score = 0
    vetoes: list[str] = []
    score += 2
    score += 2 if touched else 1
    score += 2
    if (side is TradeSide.LONG and bar.close > indicators.vwap) or (side is TradeSide.SHORT and bar.close < indicators.vwap):
        score += 1
    else:
        vetoes.append("vwap_misaligned")
    if indicators.trend_direction == side.sign:
        score += 1
    elif config.STRUCTURAL_PULLBACK_RECLAIM_REQUIRE_TREND:
        vetoes.append("trend_not_confirmed")
    if max(indicators.volume_multiple_5m, indicators.volume_multiple_15m) >= config.STRUCTURAL_PULLBACK_RECLAIM_MIN_VOLUME_MULTIPLE:
        score += 1
    else:
        vetoes.append("pullback_volume_below_floor")
    if close_loc >= config.STRUCTURAL_PULLBACK_RECLAIM_MIN_CLOSE_LOCATION:
        score += 1
    else:
        vetoes.append("weak_reclaim_close")
    if _overextended_from_ib(state, bar.close, side):
        vetoes.append("overextended_from_ib")
    if state.ib_type is IBType.NARROW and config.STRUCTURAL_BLOCK_NARROW_IB:
        vetoes.append("narrow_ib_blocked")
    if state.ib_type is IBType.NORMAL and config.STRUCTURAL_BLOCK_NORMAL_IB:
        vetoes.append("normal_ib_blocked")
    if state.ib_type is IBType.WIDE and config.STRUCTURAL_BLOCK_WIDE_IB:
        vetoes.append("wide_ib_blocked")
    if side is TradeSide.LONG and score < config.STRUCTURAL_LONG_MIN_SCORE:
        vetoes.append("long_score_below_floor")
    if side is TradeSide.SHORT and score < config.STRUCTURAL_SHORT_MIN_SCORE:
        vetoes.append("short_score_below_floor")

    entry, entry_model = _pullback_reclaim_entry(bar, side, reference)
    stop = _structural_stop(state, bar, side, level, indicators, entry)
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    if risk < config.STRUCTURAL_MIN_STOP_PTS:
        vetoes.append("structural_stop_too_tight")
    if risk > config.STRUCTURAL_MAX_STOP_PTS:
        vetoes.append("structural_stop_too_wide")
    targets = _targets(state.ib_type, entry, risk, side)
    room_r = room_to_next_level_r(
        side=side,
        entry=entry,
        stop=stop,
        levels=state.levels,
        fallback_target=targets[-1],
    )
    if room_r < config.TARGET_ROOM_MIN_R:
        vetoes.append("insufficient_room")
    if room_r < config.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R:
        vetoes.append("pullback_room_below_floor")
    if room_r > config.STRUCTURAL_PULLBACK_RECLAIM_MAX_ROOM_R:
        vetoes.append("pullback_room_too_open")
    if room_r > config.STRUCTURAL_TARGET_ROOM_MAX_R:
        vetoes.append("room_too_open")
    grade = _pullback_reclaim_grade(score, vetoes)
    return SetupCandidate(
        candidate_id=f"nqreg-exp-pbr-{bar.ts.strftime('%Y%m%d%H%M')}-{side.value}",
        module=ModuleId.STRUCTURAL_EXPANSION,
        side=side,
        setup_type="ib_pullback_reclaim",
        timestamp=bar.ts,
        level=level,
        score=score,
        grade=grade,
        entry_price=entry,
        stop_price=stop,
        targets=targets,
        entry_model=entry_model,
        risk_pct=0.0,
        invalidation_price=state.ib_levels.mid,
        target_room_r=room_r,
        vetoes=tuple(vetoes),
        details={
            "body_pct": bar.body_pts / max(bar.range_pts, 0.25),
            "close_location": close_loc,
            "volume_multiple": max(indicators.volume_multiple_5m, indicators.volume_multiple_15m),
            "ib_type": state.ib_type.value,
            "pullback_reclaim": True,
            "pullback_reference": reference,
            "pullback_age_bars": pullback_age_bars if prior_pullback else 0,
            "active_break_age_minutes": age_minutes,
        },
    )


def _continuation_candidate(
    state: RegimeCoreState,
    event: BarEvent,
    indicators: IndicatorSnapshot,
) -> SetupCandidate | None:
    if state.phase not in {SessionPhase.PRIMARY_DECISION, SessionPhase.PM_CONTINUATION, SessionPhase.LATE_PM_RESTRICTED}:
        return None
    bar = event.bar_5m
    if not _entry_time_allowed(bar):
        return None
    side = state.expansion_state.active_break_side
    if side is TradeSide.FLAT:
        if config.STRUCTURAL_CONTINUATION_REQUIRE_ACTIVE_BREAK:
            return None
        side = _accepted_side_from_recent_15m(state, indicators)
    if side is TradeSide.FLAT:
        return None
    if (
        state.expansion_state.active_break_ts is not None
        and config.STRUCTURAL_CONTINUATION_MAX_AGE_MINUTES > 0
        and (bar.ts - state.expansion_state.active_break_ts).total_seconds() / 60.0 > config.STRUCTURAL_CONTINUATION_MAX_AGE_MINUTES
    ):
        return None
    if config.STRUCTURAL_CONTINUATION_REQUIRE_15M_ACCEPTANCE and _accepted_side_from_recent_15m(state, indicators) is not side:
        return None

    level = state.ib_levels.high if side is TradeSide.LONG else state.ib_levels.low
    close_loc = bar.close_location_long if side is TradeSide.LONG else bar.close_location_short
    score = 0
    vetoes: list[str] = []

    if side is TradeSide.LONG:
        outside_ib = bar.close > state.ib_levels.high
        vwap_aligned = bar.close > indicators.vwap
        ema_ref = max(state.ib_levels.high, indicators.ema9_15m or state.ib_levels.high)
        pullback_held = bar.low <= ema_ref + 2 * config.TICK_SIZE and bar.close > ema_ref
        overextended = bar.close > state.ib_levels.high + max(state.ib_levels.range_pts * 1.25, indicators.atr_15m * 2.0)
    else:
        outside_ib = bar.close < state.ib_levels.low
        vwap_aligned = bar.close < indicators.vwap
        ema_ref = min(state.ib_levels.low, indicators.ema9_15m or state.ib_levels.low)
        pullback_held = bar.high >= ema_ref - 2 * config.TICK_SIZE and bar.close < ema_ref
        overextended = bar.close < state.ib_levels.low - max(state.ib_levels.range_pts * 1.25, indicators.atr_15m * 2.0)

    if not outside_ib:
        return None
    score += 2 if state.expansion_state.active_break_side is side else 1
    if vwap_aligned:
        score += 2
    else:
        vetoes.append("vwap_misaligned")
    if pullback_held:
        score += 2
    elif close_loc >= 0.65:
        score += 1
    else:
        vetoes.append("no_pullback_or_strong_close")
    if close_loc >= config.STRUCTURAL_CONTINUATION_MIN_CLOSE_LOCATION:
        score += 1
    else:
        vetoes.append("weak_continuation_close")
    if indicators.trend_direction == side.sign:
        score += 1
    elif config.STRUCTURAL_CONTINUATION_REQUIRE_TREND:
        vetoes.append("trend_not_confirmed")
    if indicators.volume_multiple_5m >= 0.90 or indicators.volume_multiple_15m >= 1.0:
        score += 1
    if max(indicators.volume_multiple_5m, indicators.volume_multiple_15m) < config.STRUCTURAL_CONTINUATION_MIN_VOLUME_MULTIPLE:
        vetoes.append("continuation_volume_below_floor")
    if state.phase is SessionPhase.PM_CONTINUATION:
        score += 1
    if overextended:
        vetoes.append("overextended_from_ib")
    if side is TradeSide.LONG and score < config.STRUCTURAL_LONG_MIN_SCORE:
        vetoes.append("long_score_below_floor")
    if side is TradeSide.SHORT and score < config.STRUCTURAL_SHORT_MIN_SCORE:
        vetoes.append("short_score_below_floor")

    entry, entry_model = _continuation_entry(bar, side, level, indicators)
    stop = _structural_stop(state, bar, side, level, indicators, entry)
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    if risk < config.STRUCTURAL_MIN_STOP_PTS:
        vetoes.append("structural_stop_too_tight")
    if risk > config.STRUCTURAL_MAX_STOP_PTS:
        vetoes.append("structural_stop_too_wide")
    targets = _targets(state.ib_type, entry, risk, side)
    room_r = room_to_next_level_r(
        side=side,
        entry=entry,
        stop=stop,
        levels=state.levels,
        fallback_target=targets[-1],
    )
    if room_r < config.TARGET_ROOM_MIN_R:
        vetoes.append("insufficient_room")
    if room_r < config.STRUCTURAL_CONTINUATION_MIN_ROOM_R:
        vetoes.append("continuation_room_below_floor")
    if room_r > config.STRUCTURAL_CONTINUATION_MAX_ROOM_R:
        vetoes.append("continuation_room_too_open")
    if room_r > config.STRUCTURAL_TARGET_ROOM_MAX_R:
        vetoes.append("room_too_open")
    grade = _continuation_grade(score, vetoes)
    return SetupCandidate(
        candidate_id=f"nqreg-exp-cont-{bar.ts.strftime('%Y%m%d%H%M')}-{side.value}",
        module=ModuleId.STRUCTURAL_EXPANSION,
        side=side,
        setup_type="ib_continuation_reentry",
        timestamp=bar.ts,
        level=level,
        score=score,
        grade=grade,
        entry_price=entry,
        stop_price=stop,
        targets=targets,
        entry_model=entry_model,
        risk_pct=0.0,
        invalidation_price=state.ib_levels.mid,
        target_room_r=room_r,
        vetoes=tuple(vetoes),
        details={
            "body_pct": bar.body_pts / max(bar.range_pts, 0.25),
            "close_location": close_loc,
            "volume_multiple": indicators.volume_multiple_5m,
            "ib_type": state.ib_type.value,
            "continuation": True,
            "active_break_side": state.expansion_state.active_break_side.value,
        },
    )


def score_breakout(
    state: RegimeCoreState,
    bar: BarData,
    side: TradeSide,
    indicators: IndicatorSnapshot,
) -> tuple[int, list[str], dict]:
    score = 0
    vetoes: list[str] = []
    details: dict = {}
    body = bar.body_pts
    candle_range = max(bar.range_pts, 0.25)
    close_loc = bar.close_location_long if side is TradeSide.LONG else bar.close_location_short
    body_beyond = (min(bar.open, bar.close) > state.ib_levels.high) if side is TradeSide.LONG else (max(bar.open, bar.close) < state.ib_levels.low)
    if body_beyond:
        score += 2
    else:
        vetoes.append("wick_only_or_body_not_beyond_ib")
    if body / candle_range >= config.STRUCTURAL_MIN_BODY_PCT and close_loc >= config.STRUCTURAL_MIN_CLOSE_LOCATION:
        score += 2
    else:
        vetoes.append("weak_breakout_candle")
    if (side is TradeSide.LONG and bar.close > indicators.vwap) or (side is TradeSide.SHORT and bar.close < indicators.vwap):
        score += 1
    else:
        vetoes.append("vwap_misaligned")
    prior_bodies = [sample.body_pts for sample in state.bars_15m[-7:-1] if sample.body_pts > 0]
    avg_body = fmean(prior_bodies) if prior_bodies else body
    if avg_body > 0 and body >= 1.5 * avg_body:
        score += 1
    if indicators.volume_multiple_15m >= 1.3:
        score += 1
    if _has_fvg(state.bars_15m[-3:], side):
        score += 1
    if not _overextended_from_ib(state, bar.close, side):
        score += 1
    else:
        vetoes.append("overextended_from_ib")
    if config.STRUCTURAL_BLOCK_NARROW_IB and state.ib_type is IBType.NARROW:
        vetoes.append("narrow_ib_blocked")
    if config.STRUCTURAL_BLOCK_NORMAL_IB and state.ib_type is IBType.NORMAL:
        vetoes.append("normal_ib_blocked")
    if config.STRUCTURAL_BLOCK_WIDE_IB and state.ib_type is IBType.WIDE:
        vetoes.append("wide_ib_blocked")
    score += 1
    if bool((bar.vwap or indicators.vwap) and ((side is TradeSide.LONG and indicators.vwap <= bar.close) or (side is TradeSide.SHORT and indicators.vwap >= bar.close))):
        score += 1
    live = state.last_decision_details.get("live_context", {})
    if live.get("breadth_alignment") in {side.value, "aligned", True}:
        score += 1
    elif live.get("breadth_alignment") in {"opposed", "opposite"}:
        vetoes.append("live_breadth_contradiction")
    details.update(
        {
            "body_pct": body / candle_range,
            "close_location": close_loc,
            "volume_multiple": indicators.volume_multiple_15m,
            "avg_body": avg_body,
            "ib_type": state.ib_type.value,
        }
    )
    return score, vetoes, details


def _apply_structural_quality_vetoes(
    state: RegimeCoreState,
    bar: BarData,
    side: TradeSide,
    score: int,
    vetoes: list[str],
    indicators: IndicatorSnapshot,
) -> None:
    if not _entry_time_allowed(bar):
        vetoes.append("structural_time_block")
    if config.STRUCTURAL_REQUIRE_SECOND_ACCEPTANCE and not _second_acceptance_confirmed(state, side, indicators):
        vetoes.append("second_acceptance_missing")
    if side is TradeSide.LONG and score < config.STRUCTURAL_LONG_MIN_SCORE:
        vetoes.append("long_score_below_floor")
    if side is TradeSide.SHORT and score < config.STRUCTURAL_SHORT_MIN_SCORE:
        vetoes.append("short_score_below_floor")


def _entry_time_allowed(bar: BarData) -> bool:
    et = to_et(bar.ts)
    minute = et.hour * 60 + et.minute
    return config.STRUCTURAL_MIN_ENTRY_MINUTE_ET <= minute <= config.STRUCTURAL_MAX_ENTRY_MINUTE_ET


def _second_acceptance_confirmed(
    state: RegimeCoreState,
    side: TradeSide,
    indicators: IndicatorSnapshot,
) -> bool:
    if len(state.bars_15m) < 2:
        return False
    prior = state.bars_15m[-2]
    if side is TradeSide.LONG:
        return prior.close > state.ib_levels.high and prior.close > indicators.vwap
    if side is TradeSide.SHORT:
        return prior.close < state.ib_levels.low and prior.close < indicators.vwap
    return False


def _grade(score: int, vetoes: list[str]) -> Grade:
    if vetoes:
        return Grade.INVALID
    if score >= config.STRUCTURAL_A_PLUS_SCORE:
        return Grade.A_PLUS
    if score >= config.STRUCTURAL_MIN_SCORE:
        return Grade.A
    if score >= 5:
        return Grade.B
    return Grade.INVALID


def _continuation_grade(score: int, vetoes: list[str]) -> Grade:
    if vetoes:
        return Grade.INVALID
    if score >= max(config.STRUCTURAL_A_PLUS_SCORE, config.STRUCTURAL_CONTINUATION_MIN_SCORE + 2):
        return Grade.A_PLUS
    if score >= config.STRUCTURAL_CONTINUATION_MIN_SCORE:
        return Grade.A
    if score >= max(5, config.STRUCTURAL_CONTINUATION_MIN_SCORE - 1):
        return Grade.B
    return Grade.INVALID


def _pullback_reclaim_grade(score: int, vetoes: list[str]) -> Grade:
    if vetoes:
        return Grade.INVALID
    if score >= max(config.STRUCTURAL_A_PLUS_SCORE, config.STRUCTURAL_PULLBACK_RECLAIM_MIN_SCORE + 2):
        return Grade.A_PLUS
    if score >= config.STRUCTURAL_PULLBACK_RECLAIM_MIN_SCORE:
        return Grade.A
    if score >= max(5, config.STRUCTURAL_PULLBACK_RECLAIM_MIN_SCORE - 1):
        return Grade.B
    return Grade.INVALID


def _targets(ib_type: IBType, entry: float, risk: float, side: TradeSide) -> tuple[float, float, float]:
    if ib_type is IBType.NARROW:
        multiples = (
            config.STRUCTURAL_NARROW_TARGET1_R,
            config.STRUCTURAL_NARROW_TARGET2_R,
            config.STRUCTURAL_NARROW_TARGET3_R,
        )
    elif ib_type is IBType.WIDE:
        multiples = (
            config.STRUCTURAL_WIDE_TARGET1_R,
            config.STRUCTURAL_WIDE_TARGET2_R,
            config.STRUCTURAL_WIDE_TARGET3_R,
        )
    else:
        multiples = (
            config.STRUCTURAL_NORMAL_TARGET1_R,
            config.STRUCTURAL_NORMAL_TARGET2_R,
            config.STRUCTURAL_NORMAL_TARGET3_R,
        )
    sign = side.sign
    return tuple(round_to_tick(entry + sign * risk * mult, config.TICK_SIZE) for mult in multiples)


def _continuation_entry(
    bar: BarData,
    side: TradeSide,
    level: float,
    indicators: IndicatorSnapshot,
) -> tuple[float, str]:
    mode = config.STRUCTURAL_CONTINUATION_ENTRY_MODE
    offset = side.sign * config.STRUCTURAL_CONTINUATION_ENTRY_OFFSET_TICKS * config.TICK_SIZE
    if mode == "breakout_stop":
        raw = (bar.high + offset) if side is TradeSide.LONG else (bar.low + offset)
        return round_to_tick(raw, config.TICK_SIZE), "momentum_breakout"
    if mode == "pullback":
        ema = indicators.ema9_15m or level
        if side is TradeSide.LONG:
            raw = min(bar.close - config.TICK_SIZE, max(level, ema) + offset)
            return round_to_tick(raw, config.TICK_SIZE, "down"), "continuation_pullback"
        raw = max(bar.close + config.TICK_SIZE, min(level, ema) + offset)
        return round_to_tick(raw, config.TICK_SIZE, "up"), "continuation_pullback"
    return round_to_tick(bar.close, config.TICK_SIZE), "continuation_reentry"


def _pullback_reference(
    bar: BarData,
    side: TradeSide,
    level: float,
    indicators: IndicatorSnapshot,
) -> float:
    if side is TradeSide.LONG:
        candidates = [level]
        if 0 < indicators.vwap < bar.close:
            candidates.append(indicators.vwap)
        if 0 < indicators.ema9_15m < bar.close:
            candidates.append(indicators.ema9_15m)
        return max(candidates)
    if side is TradeSide.SHORT:
        candidates = [level]
        if indicators.vwap > bar.close:
            candidates.append(indicators.vwap)
        if indicators.ema9_15m > bar.close:
            candidates.append(indicators.ema9_15m)
        return min(candidates)
    return 0.0


def _pullback_band(indicators: IndicatorSnapshot) -> float:
    atr_band = max(indicators.atr_5m, 1.0) * config.STRUCTURAL_PULLBACK_RECLAIM_BAND_ATR_MULT
    return max(2 * config.TICK_SIZE, min(config.STRUCTURAL_PULLBACK_RECLAIM_MAX_BAND_PTS, atr_band))


def _pullback_reclaim_entry(bar: BarData, side: TradeSide, reference: float) -> tuple[float, str]:
    mode = config.STRUCTURAL_PULLBACK_RECLAIM_ENTRY_MODE
    offset = side.sign * config.STRUCTURAL_PULLBACK_RECLAIM_ENTRY_OFFSET_TICKS * config.TICK_SIZE
    if mode == "pullback":
        if side is TradeSide.LONG:
            raw = min(bar.close - config.TICK_SIZE, reference + offset)
            return round_to_tick(raw, config.TICK_SIZE, "down"), "pullback_reclaim_limit"
        raw = max(bar.close + config.TICK_SIZE, reference + offset)
        return round_to_tick(raw, config.TICK_SIZE, "up"), "pullback_reclaim_limit"
    if mode == "breakout_stop":
        raw = (bar.high + offset) if side is TradeSide.LONG else (bar.low + offset)
        return round_to_tick(raw, config.TICK_SIZE), "momentum_breakout"
    return round_to_tick(bar.close, config.TICK_SIZE), "reclaim_close"


def _stop_buffer(atr_15m: float) -> float:
    if atr_15m <= 20:
        return 5.0
    if atr_15m >= 60:
        return 20.0
    return 10.0


def _structural_stop(
    state: RegimeCoreState,
    bar: BarData,
    side: TradeSide,
    level: float,
    indicators: IndicatorSnapshot,
    entry: float,
) -> float:
    buffer_pts = _stop_buffer(indicators.atr_15m)
    if config.STRUCTURAL_STOP_MODEL == "recent_5m":
        return _recent_5m_stop(state, bar, side, level, indicators, entry, buffer_pts)
    stop = (
        min(bar.low, level) - buffer_pts
        if side is TradeSide.LONG
        else max(bar.high, level) + buffer_pts
    )
    return round_to_tick(stop, config.TICK_SIZE, "down" if side is TradeSide.LONG else "up")


def _recent_5m_stop(
    state: RegimeCoreState,
    bar: BarData,
    side: TradeSide,
    level: float,
    indicators: IndicatorSnapshot,
    entry: float,
    max_buffer: float,
) -> float:
    recent = state.bars_5m[-6:] if state.bars_5m else []
    local_buffer = max(2 * config.TICK_SIZE, min(4.0, max(indicators.atr_5m, 1.0) * 0.25))
    min_risk = max(4 * config.TICK_SIZE, min(max_buffer, max(indicators.atr_5m, 2.0)))
    if side is TradeSide.LONG:
        recent_low = min((sample.low for sample in recent), default=min(bar.low, level))
        stop = max(recent_low - local_buffer, level - max_buffer)
        if stop >= entry - config.TICK_SIZE:
            stop = entry - min_risk
        return round_to_tick(stop, config.TICK_SIZE, "down")
    recent_high = max((sample.high for sample in recent), default=max(bar.high, level))
    stop = min(recent_high + local_buffer, level + max_buffer)
    if stop <= entry + config.TICK_SIZE:
        stop = entry + min_risk
    return round_to_tick(stop, config.TICK_SIZE, "up")


def _accepted_side_from_recent_15m(state: RegimeCoreState, indicators: IndicatorSnapshot) -> TradeSide:
    if not state.bars_15m:
        return TradeSide.FLAT
    bar = state.bars_15m[-1]
    if bar.close > state.ib_levels.high and bar.close > indicators.vwap:
        return TradeSide.LONG
    if bar.close < state.ib_levels.low and bar.close < indicators.vwap:
        return TradeSide.SHORT
    return TradeSide.FLAT


def _has_fvg(recent: list[BarData], side: TradeSide) -> bool:
    return _fvg_zone(recent, side) is not None


def _fvg_zone(recent: list[BarData], side: TradeSide) -> tuple[float, float] | None:
    if len(recent) < 3:
        return None
    a, _, c = recent[-3], recent[-2], recent[-1]
    if side is TradeSide.LONG:
        if c.low > a.high:
            return (a.high, c.low)
        return None
    if side is TradeSide.SHORT:
        if c.high < a.low:
            return (c.high, a.low)
        return None
    return None


def _overextended_from_ib(state: RegimeCoreState, price: float, side: TradeSide) -> bool:
    if state.ib_levels.range_pts <= 0:
        return False
    if side is TradeSide.LONG:
        return price > state.ib_levels.high + state.ib_levels.range_pts
    if side is TradeSide.SHORT:
        return price < state.ib_levels.low - state.ib_levels.range_pts
    return False
