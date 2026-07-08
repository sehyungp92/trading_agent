from __future__ import annotations

from strategies.momentum.nq_regime import config
from strategies.momentum.nq_regime.config import Grade, ModuleId, TradeSide
from strategies.momentum.nq_regime.core.filters import room_to_next_level_r, trend_day_veto
from strategies.momentum.nq_regime.core.indicators import IndicatorSnapshot
from strategies.momentum.nq_regime.core.session import SessionPhase
from strategies.momentum.nq_regime.core.state import BarEvent, RegimeCoreState
from strategies.momentum.nq_regime.modules.base import SetupCandidate
from strategies.scalp._shared.nq_contract import round_to_tick


def evaluate(state: RegimeCoreState, event: BarEvent, indicators: IndicatorSnapshot) -> SetupCandidate | None:
    early_sweep_allowed = (
        config.REVERSION_ALLOW_EARLY_SWEEP
        and state.phase is SessionPhase.EARLY_SWEEP_WATCH
        and state.levels is not None
    )
    if not state.ib_locked and not early_sweep_allowed:
        return None
    bar = event.bar_5m
    sweep = _detect_sweep(state, bar)
    if sweep is None:
        return None
    side, swept_level, penetration = sweep
    score = 0
    vetoes: list[str] = []
    score += 2 if _major_level(state, swept_level, side) else 1
    if 4.0 <= penetration <= 12.0:
        score += 1
    if penetration < config.REVERSION_MIN_PENETRATION_PTS:
        vetoes.append("penetration_too_shallow")
    if penetration > config.REVERSION_MAX_PENETRATION_PTS:
        vetoes.append("penetration_too_deep")
    if _reclaimed(bar, swept_level, side):
        score += 2
    else:
        vetoes.append("sweep_not_reclaimed")
    if _has_rejection_wick(bar, side):
        score += 1
    if indicators.volume_multiple_5m >= 1.5:
        score += 1
    value_factors = _value_trap_factors(state, bar.close, side, swept_level, indicators)
    score += min(4, value_factors * 2)
    if value_factors >= 2:
        score += 1
    if value_factors < config.REVERSION_MIN_VALUE_FACTORS:
        vetoes.append("insufficient_value_factors")
    if not trend_day_veto(side, indicators):
        score += 2
    else:
        vetoes.append("trend_day_veto")
    stop = (bar.low - 2 * config.TICK_SIZE) if side is TradeSide.LONG else (bar.high + 2 * config.TICK_SIZE)
    stop = round_to_tick(stop, config.TICK_SIZE, "down" if side is TradeSide.LONG else "up")
    if config.REVERSION_ENTRY_MODEL == "reclaim_close":
        entry = round_to_tick(bar.close, config.TICK_SIZE)
        entry_model = "reclaim_close"
    elif config.REVERSION_ENTRY_MODEL == "structure_shift":
        entry = round_to_tick(
            bar.high + config.TICK_SIZE if side is TradeSide.LONG else bar.low - config.TICK_SIZE,
            config.TICK_SIZE,
        )
        entry_model = "structure_shift"
    else:
        offset = side.sign * config.REVERSION_RETEST_OFFSET_TICKS * config.TICK_SIZE
        entry = round_to_tick(swept_level + offset, config.TICK_SIZE)
        entry_model = "swept_level_retest"
    risk = abs(entry - stop)
    cap = config.REVERSION_A_PLUS_STOP_CAP if score >= config.REVERSION_A_PLUS_SCORE else config.REVERSION_STANDARD_STOP_CAP
    if risk > cap:
        vetoes.append("stop_exceeds_cap")

    room_r, vwap_room_r, vwap_target, fallback = _room_targets(state, indicators, side, entry, stop)
    if (
        config.REVERSION_ENTRY_MODEL == "adaptive_reclaim_retest"
        and score >= config.REVERSION_ADAPTIVE_MARKET_MIN_SCORE
        and penetration <= config.REVERSION_ADAPTIVE_MARKET_MAX_PENETRATION_PTS
        and max(room_r, vwap_room_r) >= config.REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R
    ):
        market_entry = round_to_tick(bar.close, config.TICK_SIZE)
        market_risk = abs(market_entry - stop)
        market_cap = config.REVERSION_A_PLUS_STOP_CAP if score >= config.REVERSION_A_PLUS_SCORE else config.REVERSION_STANDARD_STOP_CAP
        if market_risk <= market_cap:
            entry = market_entry
            entry_model = "adaptive_reclaim_close"
            risk = market_risk
            room_r, vwap_room_r, vwap_target, fallback = _room_targets(state, indicators, side, entry, stop)

    if max(room_r, vwap_room_r) < config.TARGET_ROOM_MIN_R:
        vetoes.append("vwap_target_less_than_1p5r")
    score += 2 if vwap_room_r >= config.TARGET_ROOM_MIN_R else 0
    score += 1 if vwap_room_r >= 2.5 else 0
    grade = _grade(score, vetoes)
    targets = (
        round_to_tick(entry + side.sign * min(risk, abs(vwap_target - entry)), config.TICK_SIZE),
        round_to_tick(vwap_target, config.TICK_SIZE),
        round_to_tick(fallback, config.TICK_SIZE),
    )
    return SetupCandidate(
        candidate_id=f"nqreg-rev-{bar.ts.strftime('%Y%m%d%H%M')}-{side.value}",
        module=ModuleId.LIQUIDITY_REVERSION,
        side=side,
        setup_type="failed_liquidity_sweep",
        timestamp=bar.ts,
        level=swept_level,
        score=score,
        grade=grade,
        entry_price=entry,
        stop_price=stop,
        targets=targets,
        entry_model=entry_model,
        risk_pct=0.0,
        invalidation_price=stop,
        target_room_r=max(room_r, vwap_room_r),
        vetoes=tuple(vetoes),
        details={
            "penetration": penetration,
            "value_factors": value_factors,
            "vwap_room_r": vwap_room_r,
            "volume_multiple": indicators.volume_multiple_5m,
        },
    )


def _room_targets(
    state: RegimeCoreState,
    indicators: IndicatorSnapshot,
    side: TradeSide,
    entry: float,
    stop: float,
) -> tuple[float, float, float, float]:
    risk = abs(entry - stop)
    vwap_target = indicators.vwap if indicators.vwap > 0 else state.ib_levels.mid
    if side is TradeSide.LONG and vwap_target <= entry:
        vwap_target = state.ib_levels.mid or (entry + 1.5 * risk)
    if side is TradeSide.SHORT and vwap_target >= entry:
        vwap_target = state.ib_levels.mid or (entry - 1.5 * risk)
    fallback = entry + side.sign * max(2.0 * risk, abs(vwap_target - entry))
    room_r = room_to_next_level_r(side=side, entry=entry, stop=stop, levels=state.levels, fallback_target=fallback)
    vwap_room_r = abs(vwap_target - entry) / risk if risk > 0 else 0.0
    return room_r, vwap_room_r, vwap_target, fallback


def _detect_sweep(state: RegimeCoreState, bar) -> tuple[TradeSide, float, float] | None:
    downside = list(state.levels.major_support_levels() if state.levels else ())
    upside = list(state.levels.major_resistance_levels() if state.levels else ())
    if config.REVERSION_ENABLE_SWING_LEVELS:
        swing_support, swing_resistance = _swing_levels(state, bar.close)
        downside.extend(swing_support)
        upside.extend(swing_resistance)
    if state.ib_levels.low > 0:
        downside.append(state.ib_levels.low)
    if state.ib_levels.high > 0:
        upside.append(state.ib_levels.high)
    prior = _detect_delayed_reclaim(state, bar)
    if prior is not None:
        return prior
    for level in sorted({round(x, 2) for x in downside if x > 0}, reverse=True):
        if bar.low < level and bar.close >= level:
            _record_sweep(state, TradeSide.LONG, level, bar.ts, state.bar_index, bar.low)
            state.reversion_state.last_sweep_side = TradeSide.FLAT
            return TradeSide.LONG, level, level - bar.low
        if bar.low < level:
            _record_sweep(state, TradeSide.LONG, level, bar.ts, state.bar_index, bar.low)
            return None
    for level in sorted({round(x, 2) for x in upside if x > 0}):
        if bar.high > level and bar.close <= level:
            _record_sweep(state, TradeSide.SHORT, level, bar.ts, state.bar_index, bar.high)
            state.reversion_state.last_sweep_side = TradeSide.FLAT
            return TradeSide.SHORT, level, bar.high - level
        if bar.high > level:
            _record_sweep(state, TradeSide.SHORT, level, bar.ts, state.bar_index, bar.high)
            return None
    return None


def _swing_levels(state: RegimeCoreState, price: float) -> tuple[list[float], list[float]]:
    lookback = max(8, config.REVERSION_SWING_LOOKBACK_BARS)
    radius = max(1, config.REVERSION_SWING_RADIUS)
    bars = state.bars_5m[-(lookback + 1):-1]
    if len(bars) < radius * 2 + 1:
        return [], []
    support: list[float] = []
    resistance: list[float] = []
    for idx in range(radius, len(bars) - radius):
        pivot = bars[idx]
        left = bars[idx - radius:idx]
        right = bars[idx + 1:idx + radius + 1]
        if pivot.low < min(bar.low for bar in left) and pivot.low <= min(bar.low for bar in right):
            support.append(pivot.low)
        if pivot.high > max(bar.high for bar in left) and pivot.high >= max(bar.high for bar in right):
            resistance.append(pivot.high)
    support = _nearest_unique_levels([level for level in support if level < price], price, reverse=True)
    resistance = _nearest_unique_levels([level for level in resistance if level > price], price, reverse=False)
    return support, resistance


def _nearest_unique_levels(levels: list[float], price: float, *, reverse: bool) -> list[float]:
    unique: list[float] = []
    for level in sorted({round(item, 2) for item in levels}, key=lambda item: abs(item - price)):
        if all(abs(level - existing) >= 2.0 for existing in unique):
            unique.append(level)
        if len(unique) >= config.REVERSION_SWING_MAX_LEVELS_PER_SIDE:
            break
    return sorted(unique, reverse=reverse)


def _detect_delayed_reclaim(state: RegimeCoreState, bar) -> tuple[TradeSide, float, float] | None:
    side = state.reversion_state.last_sweep_side
    level = state.reversion_state.last_sweep_level
    if side is TradeSide.FLAT or level <= 0:
        return None
    bars_since = state.bar_index - state.reversion_state.last_sweep_bar_index
    if bars_since < 1:
        return None
    if bars_since > 3:
        state.reversion_state.last_sweep_side = TradeSide.FLAT
        return None
    if side is TradeSide.LONG and bar.close >= level:
        penetration = max(0.0, level - state.reversion_state.last_sweep_extreme)
        state.reversion_state.last_sweep_side = TradeSide.FLAT
        return side, level, penetration
    if side is TradeSide.SHORT and bar.close <= level:
        penetration = max(0.0, state.reversion_state.last_sweep_extreme - level)
        state.reversion_state.last_sweep_side = TradeSide.FLAT
        return side, level, penetration
    return None


def _record_sweep(
    state: RegimeCoreState,
    side: TradeSide,
    level: float,
    ts,
    bar_index: int,
    extreme: float,
) -> None:
    state.reversion_state.last_sweep_side = side
    state.reversion_state.last_sweep_level = level
    state.reversion_state.last_sweep_ts = ts
    state.reversion_state.last_sweep_bar_index = bar_index
    state.reversion_state.last_sweep_extreme = extreme
    state.reversion_state.sweeps_seen += 1


def _reclaimed(bar, level: float, side: TradeSide) -> bool:
    return bar.close >= level if side is TradeSide.LONG else bar.close <= level


def _has_rejection_wick(bar, side: TradeSide) -> bool:
    candle_range = max(bar.range_pts, 0.25)
    if side is TradeSide.LONG:
        wick = min(bar.open, bar.close) - bar.low
    else:
        wick = bar.high - max(bar.open, bar.close)
    return wick / candle_range >= 0.30


def _major_level(state: RegimeCoreState, level: float, side: TradeSide) -> bool:
    if state.levels is None:
        return level in {state.ib_levels.high, state.ib_levels.low}
    levels = state.levels.major_support_levels() if side is TradeSide.LONG else state.levels.major_resistance_levels()
    return any(abs(level - item) <= 0.5 for item in levels) or level in {state.ib_levels.high, state.ib_levels.low}


def _value_trap_factors(state: RegimeCoreState, close: float, side: TradeSide, swept_level: float, indicators: IndicatorSnapshot) -> int:
    factors = 0
    sd = max(indicators.vwap_sd, 1.0)
    if side is TradeSide.LONG:
        factors += int(close <= indicators.vwap - 2.0 * sd)
        factors += int(indicators.rsi14_15m < 25.0)
        factors += int(indicators.macd_15m > indicators.macd_signal_15m)
        factors += int(swept_level in {getattr(state.levels, "val", 0.0), getattr(state.levels, "pdl", 0.0), getattr(state.levels, "pml", 0.0), getattr(state.levels, "onl", 0.0)})
        if state.bars_15m:
            factors += int(state.bars_15m[-1].close >= swept_level)
    else:
        factors += int(close >= indicators.vwap + 2.0 * sd)
        factors += int(indicators.rsi14_15m > 75.0)
        factors += int(indicators.macd_15m < indicators.macd_signal_15m)
        factors += int(swept_level in {getattr(state.levels, "vah", 0.0), getattr(state.levels, "pdh", 0.0), getattr(state.levels, "pmh", 0.0), getattr(state.levels, "onh", 0.0)})
        if state.bars_15m:
            factors += int(state.bars_15m[-1].close <= swept_level)
    return factors


def _grade(score: int, vetoes: list[str]) -> Grade:
    if vetoes:
        return Grade.INVALID
    if score >= config.REVERSION_A_PLUS_SCORE:
        return Grade.A_PLUS
    if score >= config.REVERSION_A_SCORE:
        return Grade.A
    if score >= config.REVERSION_MIN_SCORE:
        return Grade.B
    return Grade.INVALID
