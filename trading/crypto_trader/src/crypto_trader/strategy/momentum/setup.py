"""Pullback zone detection and grading."""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_trader.core.models import Bar, SetupGrade, Side
from crypto_trader.strategy.momentum.bias import BiasResult, _find_swing_points
from crypto_trader.strategy.momentum.config import SetupParams
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot


@dataclass(frozen=True)
class SetupResult:
    grade: SetupGrade
    zone_price: float
    confluences: tuple[str, ...]
    room_r: float
    projected_r: float
    stop_level: float
    fib_levels: dict[float, float]  # created fresh per instance


class SetupDetector:
    def __init__(self, params: SetupParams) -> None:
        self._p = params

    def detect(
        self,
        m15_bars: list[Bar],
        h1_bars: list[Bar],
        m15_ind: IndicatorSnapshot,
        bias: BiasResult,
        min_confluences_override: int | None = None,
    ) -> SetupResult | None:
        if bias.direction is None or len(m15_bars) < 10 or len(h1_bars) < 5:
            return None

        direction = bias.direction
        price = m15_bars[-1].close
        atr = m15_ind.atr

        if atr <= 0:
            return None

        # Check for invalid pullback conditions
        if self._is_invalid_pullback(m15_bars, direction, atr):
            return None

        # Extended reaction: reject if prior impulse overextended beyond EMA
        if self._p.reject_extended_reaction:
            lookback = min(20, len(m15_bars))
            recent = m15_bars[-lookback:]
            if direction == Side.LONG:
                extension = (max(b.high for b in recent) - m15_ind.ema_fast) / atr
            else:
                extension = (m15_ind.ema_fast - min(b.low for b in recent)) / atr
            if extension > 4.0:
                return None

        # Determine pullback zone price (where price is pulling back to)
        zone = self._find_zone(m15_bars, direction)

        # Check confluences
        confluences = self._check_confluences(
            m15_bars, h1_bars, m15_ind, direction, zone,
        )

        # Compute fib levels from last H1 impulse
        fib_levels = self._compute_fib_levels_from_impulse(h1_bars, direction)

        # Check if price is in fib zone
        if fib_levels:
            fib_low_price = fib_levels.get(self._p.fib_low, 0)
            fib_high_price = fib_levels.get(self._p.fib_high, 0)
            if fib_low_price and fib_high_price:
                if direction == Side.LONG and fib_high_price <= price <= fib_low_price:
                    confluences.append("fib_zone")
                elif direction == Side.SHORT and fib_low_price <= price <= fib_high_price:
                    confluences.append("fib_zone")

        # RSI pullback quality check
        if self._p.use_rsi_pullback_filter:
            rsi = m15_ind.rsi
            if direction == Side.LONG and rsi > self._p.rsi_pullback_threshold:
                return None  # RSI too high — not a genuine pullback
            if direction == Side.SHORT and rsi < (100 - self._p.rsi_pullback_threshold):
                return None  # RSI too low — not a genuine pullback

        # Compute stop level (structural)
        stop_level = self._compute_stop(m15_bars, direction, atr)

        # Room measurement
        stop_dist = abs(price - stop_level)
        if stop_dist <= 0:
            return None

        obstacle = self._find_nearest_obstacle(h1_bars, direction, price)
        room_dist = abs(obstacle - price) if obstacle else stop_dist * 3.0
        room_r = room_dist / stop_dist
        projected_r = room_r  # simplified projection

        # Grading
        n_confluences = len(confluences)
        min_conf_b = min_confluences_override if min_confluences_override is not None else self._p.min_confluences_b
        if (
            n_confluences >= self._p.min_confluences_a
            and room_r >= self._p.min_room_a
            and m15_ind.adx > 20
            and m15_ind.adx_rising
        ):
            grade = SetupGrade.A
        elif n_confluences >= min_conf_b and room_r >= self._p.min_room_b:
            grade = SetupGrade.B
        else:
            return None  # C-grade → reject

        return SetupResult(
            grade=grade,
            zone_price=zone,
            confluences=tuple(confluences),
            room_r=room_r,
            projected_r=projected_r,
            stop_level=stop_level,
            fib_levels=fib_levels,
        )

    def _find_zone(self, bars: list[Bar], direction: Side) -> float:
        """Find pullback zone — the area price is retracing to."""
        if direction == Side.LONG:
            # Zone near recent lows in pullback
            recent = bars[-5:]
            return min(b.low for b in recent)
        else:
            recent = bars[-5:]
            return max(b.high for b in recent)

    def _check_confluences(
        self,
        m15_bars: list[Bar],
        h1_bars: list[Bar],
        m15_ind: IndicatorSnapshot,
        direction: Side,
        zone: float,
    ) -> list[str]:
        confluences: list[str] = []
        price = m15_bars[-1].close
        atr = m15_ind.atr
        tolerance = atr * 0.5  # price within 0.5 ATR of level

        # M15 EMA20
        if abs(price - m15_ind.ema_fast) < tolerance:
            confluences.append("m15_ema20")

        # M15 EMA50
        if abs(price - m15_ind.ema_mid) < tolerance:
            confluences.append("m15_ema50")

        # H1 EMA levels (approximate from last H1 bar)
        if h1_bars:
            h1_closes = [b.close for b in h1_bars]
            if len(h1_closes) >= 20:
                h1_ema20 = self._simple_ema(h1_closes, 20)
                if abs(price - h1_ema20) < tolerance:
                    confluences.append("h1_ema20")
            if len(h1_closes) >= 50:
                h1_ema50 = self._simple_ema(h1_closes, 50)
                if abs(price - h1_ema50) < tolerance:
                    confluences.append("h1_ema50")

        # Prior high/low flip
        swings = _find_swing_points(m15_bars[-20:] if len(m15_bars) >= 20 else m15_bars)
        for _, swing_price, swing_type in swings[:-2]:  # exclude very recent
            if abs(price - swing_price) < tolerance:
                confluences.append("prior_hl_flip")
                break

        return confluences

    @staticmethod
    def _simple_ema(closes: list[float], period: int) -> float:
        alpha = 2.0 / (period + 1)
        val = closes[0]
        for c in closes[1:]:
            val = alpha * c + (1 - alpha) * val
        return val

    def _compute_fib_levels_from_impulse(
        self, h1_bars: list[Bar], direction: Side
    ) -> dict[float, float]:
        lookback = min(self._p.impulse_lookback, len(h1_bars))
        recent = h1_bars[-lookback:]
        if len(recent) < 3:
            return {}

        if direction == Side.LONG:
            swing_low = min(b.low for b in recent)
            swing_high = max(b.high for b in recent)
        else:
            swing_low = min(b.low for b in recent)
            swing_high = max(b.high for b in recent)

        diff = swing_high - swing_low
        if diff <= 0:
            return {}

        levels = {}
        for fib in (0.236, 0.382, 0.5, 0.618, 0.786):
            if direction == Side.LONG:
                levels[fib] = swing_high - fib * diff
            else:
                levels[fib] = swing_low + fib * diff
        return levels

    def _compute_stop(self, bars: list[Bar], direction: Side, atr: float) -> float:
        """Approximate structural stop for room measurement (pre-entry estimate)."""
        lookback = min(self._p.swing_lookback, len(bars))
        recent = bars[-lookback:]
        # Use a conservative ATR buffer for the estimate
        buffer = atr * 0.3
        if direction == Side.LONG:
            return min(b.low for b in recent) - buffer
        else:
            return max(b.high for b in recent) + buffer

    def _find_nearest_obstacle(
        self, h1_bars: list[Bar], direction: Side, price: float
    ) -> float | None:
        if len(h1_bars) < 3:
            return None
        swings = _find_swing_points(h1_bars[-30:] if len(h1_bars) >= 30 else h1_bars)
        if not swings:
            return None

        if direction == Side.LONG:
            # Nearest swing high above current price
            above = [s[1] for s in swings if s[2] == "high" and s[1] > price]
            return min(above) if above else None
        else:
            # Nearest swing low below current price
            below = [s[1] for s in swings if s[2] == "low" and s[1] < price]
            return max(below) if below else None

    def _is_invalid_pullback(self, bars: list[Bar], direction: Side, atr: float) -> bool:
        if len(bars) < 5:
            return False

        recent = bars[-5:]

        # Mid-nowhere: no clear confluence zone
        if self._p.reject_mid_nowhere:
            ranges = [b.high - b.low for b in recent]
            avg_range = sum(ranges) / len(ranges)
            if avg_range < atr * 0.2:
                return True  # Dead, meaningless range

        # Parabolic extension: multiple large bars in one direction
        if self._p.reject_parabolic_extension:
            bodies = [abs(b.close - b.open) for b in recent]
            large_bars = sum(1 for body in bodies if body > atr * 1.5)
            if large_bars >= 3:
                return True

        # Impulsive breakdown: sharp move against direction
        if self._p.reject_impulsive_breakdown:
            last_bar = recent[-1]
            body = last_bar.close - last_bar.open
            if direction == Side.LONG and body < -atr * 2.0:
                return True
            if direction == Side.SHORT and body > atr * 2.0:
                return True

        return False
