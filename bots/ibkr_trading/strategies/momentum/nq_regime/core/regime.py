from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from strategies.momentum.nq_regime import config
from strategies.momentum.nq_regime.config import ModuleId, TradeSide
from strategies.momentum.nq_regime.core.indicators import IndicatorSnapshot
from strategies.momentum.nq_regime.core.state import BarData, RegimeCoreState
from strategies.momentum.nq_regime.modules.base import BlockedCandidate, SetupCandidate


class Regime(IntEnum):
    UNCLASSIFIED = 0
    STRUCTURAL_EXPANSION = 1
    LIQUIDITY_REVERSION = 2
    PM_CONTINUATION = 3
    NEWS_DISTORTED = 4
    DEAD_CHOP = 5
    TRANSITION = 6


@dataclass(frozen=True, slots=True)
class RegimeScores:
    expansion: float = 0.0
    reversion: float = 0.0
    pm_continuation: float = 0.0
    news_distorted: float = 0.0
    dead_chop: float = 0.0


@dataclass(frozen=True, slots=True)
class RegimeResult:
    regime: Regime
    scores: RegimeScores
    confidence: float
    margin: float
    trigger: str = "checkpoint"


def classify_regime(
    state: RegimeCoreState,
    indicators: IndicatorSnapshot,
    bar: BarData,
    *,
    news_active: bool = False,
    trigger: str = "checkpoint",
) -> RegimeResult:
    if news_active:
        scores = RegimeScores(news_distorted=1.0)
        return RegimeResult(Regime.NEWS_DISTORTED, scores, 1.0, 1.0, trigger)
    expansion = _expansion_score(state, indicators, bar)
    reversion = _reversion_score(state, indicators, bar)
    pm = _pm_score(state, indicators, bar)
    dead = _dead_chop_score(state, indicators, bar)
    scores = RegimeScores(expansion=expansion, reversion=reversion, pm_continuation=pm, dead_chop=dead)
    pairs = [
        (Regime.STRUCTURAL_EXPANSION, scores.expansion),
        (Regime.LIQUIDITY_REVERSION, scores.reversion),
        (Regime.PM_CONTINUATION, scores.pm_continuation),
        (Regime.DEAD_CHOP, scores.dead_chop),
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    winning_regime, confidence = pairs[0]
    margin = confidence - pairs[1][1]
    if confidence < config.REGIME_MIN_CONFIDENCE or margin < config.REGIME_MIN_MARGIN:
        winning_regime = Regime.TRANSITION
    return RegimeResult(winning_regime, scores, confidence, margin, trigger)


def route_candidates(
    regime: Regime,
    candidates: list[SetupCandidate],
) -> tuple[SetupCandidate | None, tuple[BlockedCandidate, ...], str]:
    selected: SetupCandidate | None = None
    blocked: list[BlockedCandidate] = []
    allowed = _allowed_modules(regime)
    valid_candidates = sorted(
        [candidate for candidate in candidates if candidate.valid],
        key=lambda candidate: (candidate.grade.value == "A+", candidate.score, candidate.target_room_r),
        reverse=True,
    )
    for candidate in valid_candidates:
        fallback_allowed = (
            config.ROUTE_ALLOW_A_PLUS_FALLBACK
            and candidate.grade.value == "A+"
            and candidate.score >= config.ROUTE_FALLBACK_MIN_SCORE
        )
        candidate_led_allowed = _candidate_led_allowed(candidate)
        if candidate.module not in allowed and not fallback_allowed and not candidate_led_allowed:
            blocked.append(BlockedCandidate(candidate, "regime_mismatch"))
            continue
        if selected is None:
            selected = candidate
        else:
            blocked.append(BlockedCandidate(candidate, "one_module_one_trade"))
    for candidate in candidates:
        if candidate in valid_candidates:
            continue
        blocked.append(BlockedCandidate(candidate, "veto_active" if candidate.vetoes else "below_grade"))
    if selected is None:
        return None, tuple(blocked), "no_valid_candidate"
    conflict = _conflict_reason(selected, valid_candidates)
    if conflict:
        return None, tuple([*blocked, BlockedCandidate(selected, conflict)]), conflict
    return selected, tuple(blocked), f"selected_{selected.module.value}"


def _candidate_led_allowed(candidate: SetupCandidate) -> bool:
    if candidate.module is ModuleId.SECOND_WIND and config.SECOND_WIND_CANDIDATE_LED_ENABLED:
        pm_score = float(candidate.details.get("pm_score", 0.0) or 0.0)
        return (
            candidate.score >= config.SECOND_WIND_CANDIDATE_LED_MIN_SCORE
            and candidate.target_room_r >= config.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R
            and pm_score >= config.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE
        )
    if not config.ROUTE_CANDIDATE_LED_ENABLED:
        return False
    return (
        candidate.score >= config.ROUTE_CANDIDATE_LED_MIN_SCORE
        and candidate.target_room_r >= config.ROUTE_CANDIDATE_LED_MIN_ROOM_R
    )


def module_for_regime(regime: Regime) -> ModuleId:
    if regime is Regime.STRUCTURAL_EXPANSION:
        return ModuleId.STRUCTURAL_EXPANSION
    if regime is Regime.LIQUIDITY_REVERSION:
        return ModuleId.LIQUIDITY_REVERSION
    if regime is Regime.PM_CONTINUATION:
        return ModuleId.SECOND_WIND
    return ModuleId.NONE


def _allowed_modules(regime: Regime) -> set[ModuleId]:
    if regime is Regime.STRUCTURAL_EXPANSION:
        return {ModuleId.STRUCTURAL_EXPANSION}
    if regime is Regime.LIQUIDITY_REVERSION:
        return {ModuleId.LIQUIDITY_REVERSION}
    if regime is Regime.PM_CONTINUATION:
        return {ModuleId.SECOND_WIND, ModuleId.STRUCTURAL_EXPANSION}
    return set()


def _expansion_score(state: RegimeCoreState, indicators: IndicatorSnapshot, bar: BarData) -> float:
    if not state.ib_locked:
        return 0.0
    outside = 0.0
    if bar.close > state.ib_levels.high and bar.close > indicators.vwap:
        outside = min(1.0, (bar.close - state.ib_levels.high) / max(state.ib_levels.range_pts * 0.25, 1.0))
    elif bar.close < state.ib_levels.low and bar.close < indicators.vwap:
        outside = min(1.0, (state.ib_levels.low - bar.close) / max(state.ib_levels.range_pts * 0.25, 1.0))
    candle_quality = min(1.0, bar.body_pts / max(bar.range_pts * 0.50, 1.0))
    volume = min(1.0, indicators.volume_multiple_15m / 1.3)
    return max(0.0, min(1.0, 0.45 * outside + 0.30 * candle_quality + 0.15 * volume + 0.10 * abs(indicators.trend_direction)))


def _reversion_score(state: RegimeCoreState, indicators: IndicatorSnapshot, bar: BarData) -> float:
    if not state.ib_locked:
        return 0.0
    inside_ib = state.ib_levels.low <= bar.close <= state.ib_levels.high
    vwap_flat = abs(indicators.vwap_slope) <= max(indicators.atr_5m * 0.20, 2.0)
    two_way = 1.0 if inside_ib else 0.4
    wick_share = 1.0 - min(1.0, bar.body_pts / max(bar.range_pts, 0.25))
    sweep_context = 0.5 if (bar.high > state.ib_levels.high or bar.low < state.ib_levels.low) else 0.0
    return max(0.0, min(1.0, 0.35 * two_way + 0.25 * float(vwap_flat) + 0.25 * wick_share + 0.15 * sweep_context))


def _pm_score(state: RegimeCoreState, indicators: IndicatorSnapshot, bar: BarData) -> float:
    del bar
    if len(state.bars_15m) < 10:
        return 0.0
    trend = abs(indicators.trend_direction)
    squeeze = min(1.0, max(indicators.squeeze_duration, state.second_wind_state.squeeze_duration) / 5.0)
    control = min(1.0, abs(indicators.am_vwap_control) * 2.0)
    return max(0.0, min(1.0, 0.45 * trend + 0.35 * squeeze + 0.20 * control))


def _dead_chop_score(state: RegimeCoreState, indicators: IndicatorSnapshot, bar: BarData) -> float:
    if not state.ib_locked:
        return 0.0
    near_vwap = abs(bar.close - indicators.vwap) <= max(indicators.atr_5m * 0.5, 2.0)
    weak_volume = indicators.volume_multiple_5m < 0.75
    compressed = state.ib_levels.range_pts < config.IB_NARROW_MAX and indicators.atr_15m < max(state.ib_levels.range_pts * 0.4, 1.0)
    return min(1.0, 0.35 * float(near_vwap) + 0.35 * float(weak_volume) + 0.30 * float(compressed))


def _conflict_reason(selected: SetupCandidate, candidates: list[SetupCandidate]) -> str:
    if selected.module is ModuleId.LIQUIDITY_REVERSION:
        for candidate in candidates:
            if candidate.module is ModuleId.SECOND_WIND and candidate.side is not TradeSide.FLAT:
                return "second_wind_priority_over_pm_fade"
    return ""
