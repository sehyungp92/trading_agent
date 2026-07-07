"""Trailing stop management — structure-based and EMA-based."""

from __future__ import annotations

import numpy as np

from crypto_trader.core.models import Bar, Position, Side
from crypto_trader.strategy.momentum.config import TrailParams
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot, _ema


def _confirmation_family(confirmation_type: str | None) -> str:
    pattern = (confirmation_type or "").lower()
    if pattern in {"inside_bar_break", "base_break"}:
        return "continuation"
    if pattern in {"micro_structure_shift", "hammer", "engulfing"}:
        return "inflection"
    return "other"


def _scope_matches(scope: str, family: str) -> bool:
    normalized = (scope or "all").lower()
    if normalized in {"all", "any"}:
        return True
    if normalized in {"none", "off"}:
        return False
    return normalized == family


class TrailManager:
    def __init__(self, params: TrailParams) -> None:
        self._p = params
        self._current_stops: dict[str, float] = {}

    def update(
        self,
        position: Position,
        m15_bars: list[Bar],
        indicators: IndicatorSnapshot | None,
        current_stop: float | None = None,
        bars_since_entry: int = 0,
        current_r: float = 0.0,
        mfe_r: float = 0.0,
        confirmation_type: str | None = None,
    ) -> float | None:
        if indicators is None or len(m15_bars) < 5:
            return None

        # Activation gate: trail only after bars OR R threshold is met
        if (bars_since_entry < self._p.trail_activation_bars
                and current_r < self._p.trail_activation_r):
            return None

        sym = position.symbol
        atr = indicators.atr
        family = _confirmation_family(confirmation_type)
        runner_active = (
            self._p.runner_trail_enabled
            and _scope_matches(self._p.runner_trail_scope, family)
            and mfe_r >= self._p.runner_trigger_r
        )
        basis = (self._p.trail_r_basis or "current").lower()
        buffer_wide = self._p.trail_buffer_wide
        buffer_tight = self._p.trail_buffer_tight
        r_ceiling = self._p.trail_r_ceiling
        if runner_active:
            basis = (self._p.runner_trail_r_basis or basis).lower()
            buffer_wide = self._p.runner_trail_buffer_wide
            buffer_tight = self._p.runner_trail_buffer_tight
            r_ceiling = self._p.runner_trail_r_ceiling

        if self._p.trail_r_adaptive:
            # R-adaptive: loose when flat, tight when deep in profit
            adaptive_r = mfe_r if basis == "mfe" else current_r
            r_clamped = max(0.0, min(adaptive_r, r_ceiling))
            r_frac = r_clamped / r_ceiling if r_ceiling > 0 else 0.0
            buffer_mult = buffer_wide * (1.0 - r_frac) + buffer_tight * r_frac
            buffer = atr * buffer_mult
        else:
            # Legacy: fixed buffer + warmup decay
            buffer = atr * self._p.trail_atr_buffer
            bars_active = bars_since_entry - self._p.trail_activation_bars
            if bars_active >= 0 and self._p.trail_warmup_bars > 0:
                warmup_remaining = max(0.0, 1.0 - bars_active / self._p.trail_warmup_bars)
                buffer += atr * self._p.trail_warmup_buffer_mult * warmup_remaining

        # MFE-floor: prevent buffer from getting too tight on proven trades
        if self._p.trail_mfe_floor_enabled and mfe_r >= self._p.trail_mfe_floor_threshold:
            min_buffer = atr * self._p.trail_mfe_floor_buffer
            buffer = max(buffer, min_buffer)

        if current_stop is not None:
            self._current_stops[sym] = current_stop

        candidates: list[float] = []
        mode = (self._p.trail_mode or "components").lower()

        if mode == "components":
            # Structure trailing: behind swing points
            if self._p.trail_behind_structure:
                struct_stop = self._structure_trail(m15_bars, position.direction, buffer)
                if struct_stop is not None:
                    candidates.append(struct_stop)

            # EMA trailing: behind EMA
            if self._p.trail_behind_ema:
                ema_stop = self._ema_trail(m15_bars, position.direction, buffer)
                if ema_stop is not None:
                    candidates.append(ema_stop)
        elif mode == "ema":
            ema_stop = self._ema_trail(m15_bars, position.direction, buffer)
            if ema_stop is not None:
                candidates.append(ema_stop)
        elif mode == "chandelier":
            chandelier_stop = self._chandelier_trail(m15_bars, position.direction, atr)
            if chandelier_stop is not None:
                candidates.append(chandelier_stop)
        elif mode == "hybrid":
            ema_stop = self._ema_trail(m15_bars, position.direction, buffer)
            if ema_stop is not None:
                candidates.append(ema_stop)
            chandelier_stop = self._chandelier_trail(m15_bars, position.direction, atr)
            if chandelier_stop is not None:
                candidates.append(chandelier_stop)

        if not candidates:
            return None

        # Pick trail stop: tightest (max for long) or generous (min for long)
        if self._p.trail_use_tightest:
            new_stop = max(candidates) if position.direction == Side.LONG else min(candidates)
        else:
            new_stop = min(candidates) if position.direction == Side.LONG else max(candidates)

        # Only move stop favorably
        old_stop = self._current_stops.get(sym)
        if old_stop is not None:
            if position.direction == Side.LONG and new_stop <= old_stop:
                return None
            if position.direction == Side.SHORT and new_stop >= old_stop:
                return None

        self._current_stops[sym] = new_stop
        return new_stop

    def remove(self, symbol: str) -> None:
        self._current_stops.pop(symbol, None)

    def _structure_trail(
        self, bars: list[Bar], direction: Side, buffer: float
    ) -> float | None:
        # Find swing lows (long) or swing highs (short) from recent bars
        recent = bars[-20:] if len(bars) >= 20 else bars
        if len(recent) < 3:
            return None

        if direction == Side.LONG:
            swing_lows: list[float] = []
            for i in range(1, len(recent) - 1):
                if recent[i].low < recent[i - 1].low and recent[i].low < recent[i + 1].low:
                    swing_lows.append(recent[i].low)
            if swing_lows:
                return swing_lows[-1] - buffer
        else:
            swing_highs: list[float] = []
            for i in range(1, len(recent) - 1):
                if recent[i].high > recent[i - 1].high and recent[i].high > recent[i + 1].high:
                    swing_highs.append(recent[i].high)
            if swing_highs:
                return swing_highs[-1] + buffer

        return None

    def _ema_trail(
        self, bars: list[Bar], direction: Side, buffer: float
    ) -> float | None:
        closes = np.array([b.close for b in bars])
        if len(closes) < self._p.trail_ema_period:
            return None

        ema_arr = _ema(closes, self._p.trail_ema_period)
        ema_val = float(ema_arr[-1])

        if direction == Side.LONG:
            return ema_val - buffer
        else:
            return ema_val + buffer

    def _chandelier_trail(
        self, bars: list[Bar], direction: Side, atr: float
    ) -> float | None:
        if atr <= 0:
            return None

        lookback = max(2, self._p.trail_chandelier_lookback)
        recent = bars[-lookback:] if len(bars) >= lookback else bars
        if not recent:
            return None

        offset = atr * self._p.trail_chandelier_atr_mult
        if direction == Side.LONG:
            highest_high = max(bar.high for bar in recent)
            return highest_high - offset

        lowest_low = min(bar.low for bar in recent)
        return lowest_low + offset
