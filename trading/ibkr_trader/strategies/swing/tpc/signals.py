from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategies.swing._shared import indicators as ind
from strategies.swing._shared.etf_core import ETFBarInput
from strategies.swing._shared.models import Direction
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.models import PullbackType, RegimeGrade


@dataclass(frozen=True, slots=True)
class PullbackCandidate:
    pullback_type: PullbackType
    low: float
    high: float
    depth: float
    orderly: bool
    value_hits: int


def detect_pullback(
    bar_input: ETFBarInput,
    direction: Direction,
    regime_grade: RegimeGrade,
    cfg: TPCSymbolConfig,
) -> PullbackCandidate | None:
    return _detect_pullback_on_timeframe(bar_input, direction, regime_grade, cfg, timeframe="1h")


def detect_pullback_30m(
    bar_input: ETFBarInput,
    direction: Direction,
    regime_grade: RegimeGrade,
    cfg: TPCSymbolConfig,
) -> PullbackCandidate | None:
    return _detect_pullback_on_timeframe(bar_input, direction, regime_grade, cfg, timeframe="30m")


def _detect_pullback_on_timeframe(
    bar_input: ETFBarInput,
    direction: Direction,
    regime_grade: RegimeGrade,
    cfg: TPCSymbolConfig,
    *,
    timeframe: str,
) -> PullbackCandidate | None:
    bars = bar_input.bars_30m if timeframe == "30m" else bar_input.bars_1h
    if bars is None or len(bars) < cfg.pullback_max_bars_1h + 8:
        return None
    lookback = min(cfg.pullback_max_bars_1h + 6, len(bars))
    highs = bars.highs[-lookback:]
    lows = bars.lows[-lookback:]
    closes = bars.closes[-lookback:]
    if direction == Direction.LONG:
        impulse_low = float(np.nanmin(lows[: max(2, lookback // 2)]))
        impulse_high = float(np.nanmax(highs))
        current_low = float(np.nanmin(lows[-cfg.pullback_max_bars_1h:]))
        depth = (impulse_high - current_low) / max(impulse_high - impulse_low, 1e-9)
        value_levels = _finite_values(
            bar_input.indicators.get(f"ema20_{timeframe}", np.nan),
            bar_input.indicators.get(f"ema50_{timeframe}", np.nan),
            bar_input.indicators.get(f"vwap_{timeframe}", np.nan),
        )
        value_hits = sum(current_low <= level for level in value_levels)
    else:
        impulse_high = float(np.nanmax(highs[: max(2, lookback // 2)]))
        impulse_low = float(np.nanmin(lows))
        current_high = float(np.nanmax(highs[-cfg.pullback_max_bars_1h:]))
        depth = (current_high - impulse_low) / max(impulse_high - impulse_low, 1e-9)
        value_levels = _finite_values(
            bar_input.indicators.get(f"ema20_{timeframe}", np.nan),
            bar_input.indicators.get(f"ema50_{timeframe}", np.nan),
            bar_input.indicators.get(f"vwap_{timeframe}", np.nan),
        )
        value_hits = sum(current_high >= level for level in value_levels)
    if value_hits <= 0 or depth <= 0:
        return None
    recent_ranges = highs[-cfg.pullback_max_bars_1h:] - lows[-cfg.pullback_max_bars_1h:]
    impulse_ranges = highs[: cfg.pullback_min_bars_1h] - lows[: cfg.pullback_min_bars_1h]
    orderly = float(np.nanmean(recent_ranges)) <= float(np.nanmean(impulse_ranges)) * 1.25
    if cfg.pullback_orderly_required and not orderly:
        return None
    if cfg.pullback_volume_contract_max > 0 and len(bars.volumes) >= lookback:
        recent_vol = float(np.nanmean(bars.volumes[-cfg.pullback_max_bars_1h:]))
        impulse_vol = float(np.nanmean(bars.volumes[: cfg.pullback_min_bars_1h]))
        if impulse_vol > 0 and recent_vol > impulse_vol * cfg.pullback_volume_contract_max:
            return None
    if cfg.fib_a_low <= depth <= cfg.fib_a_high and value_hits >= cfg.type_a_value_hits_min:
        ptype = PullbackType.TYPE_A
    elif (
        cfg.type_b_enabled
        and cfg.fib_b_low <= depth <= cfg.fib_b_high
        and (not cfg.type_b_requires_a_plus or regime_grade == RegimeGrade.A_PLUS)
        and value_hits >= cfg.type_b_value_hits_min
    ):
        ptype = PullbackType.TYPE_B
    elif (
        cfg.type_c_enabled
        and cfg.type_c_mode in {"shallow", "shallow_or_reentry"}
        and cfg.fib_c_low <= depth <= cfg.fib_c_high
        and (not cfg.type_c_requires_a_plus or regime_grade == RegimeGrade.A_PLUS)
        and value_hits >= cfg.type_c_value_hits_min
    ):
        ptype = PullbackType.TYPE_C
    else:
        return None
    if timeframe == "30m":
        if not _pb30_ema20_context_allowed(bar_input, direction, cfg, highs, lows, closes):
            return None
        if not _pb30_ma_transition_allowed(bar_input, direction, cfg):
            return None
    low = float(np.nanmin(lows[-cfg.pullback_max_bars_1h:]))
    high = float(np.nanmax(highs[-cfg.pullback_max_bars_1h:]))
    return PullbackCandidate(ptype, low, high, float(depth), bool(orderly), int(value_hits))


def _pb30_ema20_context_allowed(
    bar_input: ETFBarInput,
    direction: Direction,
    cfg: TPCSymbolConfig,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> bool:
    if not cfg.pb30_ema20_context_enabled:
        return True
    ema20 = float(bar_input.indicators.get("ema20_30m", np.nan))
    if not np.isfinite(ema20) or len(closes) < 2:
        return False
    lookback = int(cfg.pb30_ema20_context_lookback_bars_30m or cfg.pullback_max_bars_1h)
    lookback = max(2, min(lookback, len(closes)))
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]
    atr = float(bar_input.indicators.get("atr_30m", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        atr = float(bar_input.indicators.get("atr_15m", np.nan))
    tolerance = max(float(cfg.pb30_ema20_context_distance_atr), 0.0) * max(atr, 0.0)
    mode = (cfg.pb30_ema20_context_mode or "touch").lower()
    min_low = float(np.nanmin(recent_lows))
    max_high = float(np.nanmax(recent_highs))
    prev_close = float(recent_closes[-2])
    current_close = float(recent_closes[-1])
    close_15m = float(getattr(bar_input.bar_15m, "close", np.nan)) if bar_input.bar_15m is not None else np.nan
    if direction == Direction.LONG:
        touched = min_low <= ema20 <= max_high
        near = min_low <= ema20 + tolerance and max_high >= ema20 - tolerance
        pierced = min_low <= ema20 - tolerance and max_high >= ema20
        held = near and current_close >= ema20
        reclaimed = near and prev_close <= ema20 and current_close >= ema20
        confirmed = near and np.isfinite(close_15m) and close_15m >= ema20
    else:
        touched = min_low <= ema20 <= max_high
        near = max_high >= ema20 - tolerance and min_low <= ema20 + tolerance
        pierced = max_high >= ema20 + tolerance and min_low <= ema20
        held = near and current_close <= ema20
        reclaimed = near and prev_close >= ema20 and current_close <= ema20
        confirmed = near and np.isfinite(close_15m) and close_15m <= ema20
    if mode == "touch":
        return touched
    if mode == "near":
        return near
    if mode == "pierce":
        return pierced
    if mode == "hold":
        return held
    if mode == "reclaim":
        return reclaimed
    if mode == "confirm":
        return confirmed
    return touched


def _pb30_ma_transition_allowed(
    bar_input: ETFBarInput,
    direction: Direction,
    cfg: TPCSymbolConfig,
) -> bool:
    if not cfg.pb30_ma_transition_enabled:
        return True
    bars = bar_input.bars_30m
    if bars is None or len(bars.closes) < max(cfg.ema_50_period, cfg.ema_20_period) + 2:
        return False
    closes = np.asarray(bars.closes, dtype=float)
    lookback = max(1, int(cfg.pb30_ma_transition_lookback_bars_30m or 1))
    transition_window = max(lookback + 1, int(cfg.pb30_ma_transition_window_bars_30m or lookback + 1))
    fast_period = max(2, int(cfg.ema_20_period))
    slow_period = max(fast_period + 1, int(cfg.ema_50_period))
    fast_now = _ma_value(closes, fast_period, offset=0)
    fast_prev = _ma_value(closes, fast_period, offset=lookback)
    slow_now = _ma_value(closes, slow_period, offset=0)
    slow_prev = _ma_value(closes, slow_period, offset=lookback)
    fast_then = _ma_value(closes, fast_period, offset=transition_window)
    slow_then = _ma_value(closes, slow_period, offset=transition_window)
    values = (fast_now, fast_prev, slow_now, slow_prev)
    if any(not np.isfinite(value) for value in values):
        return False
    atr = float(bar_input.indicators.get("atr_30m", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        atr_values = ind.atr(bars.highs, bars.lows, bars.closes, cfg.atr_period)
        atr = float(atr_values[-1]) if len(atr_values) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return False
    min_slope = max(0.0, float(cfg.pb30_ma_transition_min_slope_atr))
    if direction == Direction.LONG:
        fast_slope = (fast_now - fast_prev) / max(atr * lookback, 1e-9)
        slow_slope = (slow_now - slow_prev) / max(atr * lookback, 1e-9)
        stack_ok = fast_now >= slow_now
        transitioned = (
            np.isfinite(fast_then)
            and np.isfinite(slow_then)
            and fast_then <= slow_then
            and stack_ok
        )
    else:
        fast_slope = (fast_prev - fast_now) / max(atr * lookback, 1e-9)
        slow_slope = (slow_prev - slow_now) / max(atr * lookback, 1e-9)
        stack_ok = fast_now <= slow_now
        transitioned = (
            np.isfinite(fast_then)
            and np.isfinite(slow_then)
            and fast_then >= slow_then
            and stack_ok
        )
    fast_ok = fast_slope >= min_slope
    slow_ok = slow_slope >= min_slope
    mode = (cfg.pb30_ma_transition_mode or "fast_slope").lower()
    if mode == "fast_slope":
        return fast_ok
    if mode == "stack":
        return stack_ok
    if mode == "slope_and_stack":
        return fast_ok and stack_ok
    if mode == "fast_slow_slope":
        return fast_ok and slow_ok
    if mode == "transition":
        return stack_ok and (transitioned or fast_ok)
    if mode == "transition_only":
        return bool(transitioned)
    return fast_ok


def _ma_value(values: np.ndarray, period: int, *, offset: int = 0) -> float:
    end = len(values) - max(int(offset), 0)
    start = end - max(int(period), 1)
    if start < 0 or end > len(values) or start >= end:
        return float("nan")
    return float(np.nanmean(values[start:end]))


def check_confirmation(
    bar_input: ETFBarInput,
    direction: Direction,
    cfg: TPCSymbolConfig,
) -> tuple[bool, list[str]]:
    ok, triggers = _check_confirmation_on_timeframe(bar_input, direction, cfg, "15m")
    if ok:
        return True, triggers
    ok30, triggers30 = _check_confirmation_on_timeframe(bar_input, direction, cfg, "30m")
    if ok30:
        return True, triggers30
    return False, triggers or triggers30


def _check_confirmation_on_timeframe(
    bar_input: ETFBarInput,
    direction: Direction,
    cfg: TPCSymbolConfig,
    timeframe: str,
) -> tuple[bool, list[str]]:
    bars = bar_input.bars_30m if timeframe == "30m" else bar_input.bars_15m
    if bars is None or len(bars) < 6:
        return False, []
    O, H, L, C = bars.opens[-1], bars.highs[-1], bars.lows[-1], bars.closes[-1]
    prev_closes = bars.closes[-6:-1]
    prev_highs = bars.highs[-6:-1]
    prev_lows = bars.lows[-6:-1]
    rng = max(H - L, 1e-9)
    triggers: list[str] = []
    vwap = bar_input.indicators.get(f"vwap_{timeframe}", np.nan)
    ema20 = bar_input.indicators.get(f"ema20_{timeframe}", np.nan)
    vol_sma = bar_input.indicators.get(f"volume_sma_{timeframe}", np.nan)
    prefix = "" if timeframe == "15m" else f"{timeframe}_"
    if direction == Direction.LONG:
        if C > O and (min(O, C) - L) / rng >= 0.35:
            triggers.append(f"{prefix}bullish_reversal")
        if not np.isnan(vwap) and C > vwap:
            triggers.append(f"{prefix}vwap_reclaim")
        if L > float(np.nanmin(prev_lows[-3:])):
            triggers.append(f"{prefix}higher_low")
        if C > float(np.nanmax(prev_highs[-3:])):
            triggers.append(f"{prefix}micro_break")
        if C > ema20:
            triggers.append(f"{prefix}trendline_break")
        if (C - L) / rng >= 2.0 / 3.0:
            triggers.append(f"{prefix}upper_third_close")
    else:
        if C < O and (H - max(O, C)) / rng >= 0.35:
            triggers.append(f"{prefix}bearish_reversal")
        if not np.isnan(vwap) and C < vwap:
            triggers.append(f"{prefix}vwap_loss")
        if H < float(np.nanmax(prev_highs[-3:])):
            triggers.append(f"{prefix}lower_high")
        if C < float(np.nanmin(prev_lows[-3:])):
            triggers.append(f"{prefix}micro_break")
        if C < ema20:
            triggers.append(f"{prefix}trendline_break")
        if (C - L) / rng <= 1.0 / 3.0:
            triggers.append(f"{prefix}lower_third_close")
    if not np.isnan(vol_sma) and bars.volumes[-1] >= cfg.volume_expansion_mult * vol_sma:
        triggers.append(f"{prefix}volume_expansion")
    return len(set(triggers)) >= cfg.confirmation_required, triggers


def _finite_values(*values: float) -> list[float]:
    return [float(value) for value in values if np.isfinite(value)]
