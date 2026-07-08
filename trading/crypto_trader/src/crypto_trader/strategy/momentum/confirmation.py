"""Candlestick pattern confirmation detection."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.config import ConfirmationParams


@dataclass(frozen=True)
class ConfirmationResult:
    pattern_type: str
    trigger_price: float
    bar_index: int
    volume_confirmed: bool


class ConfirmationDetector:
    def __init__(self, params: ConfirmationParams) -> None:
        self._p = params

    def check(
        self,
        m15_bars: list[Bar],
        zone_price: float,
        direction: Side,
        volume_ma: float,
        atr: float,
    ) -> ConfirmationResult | None:
        if len(m15_bars) < 3:
            return None

        # Check patterns in priority order on the most recent bars
        checkers = []
        if self._p.enable_engulfing:
            checkers.append(self._check_engulfing)
        if self._p.enable_hammer:
            checkers.append(self._check_hammer)
        if self._p.enable_inside_bar:
            checkers.append(self._check_inside_bar)
        if self._p.enable_micro_shift:
            checkers.append(self._check_micro_shift)
        if self._p.enable_base_break:
            checkers.append(self._check_base_break)

        for checker in checkers:
            result = checker(m15_bars, zone_price, direction, volume_ma)
            if result is not None:
                if not self._zone_proximity_ok(m15_bars, result, zone_price, atr):
                    continue
                return result
        return None

    def _volume_ok(self, bar: Bar, volume_ma: float) -> bool:
        if not self._p.require_volume_confirm:
            return True
        if volume_ma <= 0:
            return True
        return bar.volume >= self._p.volume_threshold_mult * volume_ma

    def _zone_proximity_ok(
        self,
        bars: list[Bar],
        result: ConfirmationResult,
        zone_price: float,
        atr: float,
    ) -> bool:
        if not self._p.require_zone_proximity:
            return True
        if atr <= 0:
            return True
        bar_idx = max(0, min(result.bar_index, len(bars) - 1))
        reference_price = bars[bar_idx].close
        return abs(reference_price - zone_price) <= atr * self._p.zone_proximity_atr

    def _check_engulfing(
        self, bars: list[Bar], zone: float, direction: Side, vol_ma: float
    ) -> ConfirmationResult | None:
        cur = bars[-1]
        prev = bars[-2]
        cur_body = cur.close - cur.open
        prev_body = prev.close - prev.open

        if direction == Side.LONG:
            # Bullish engulfing: current bullish body fully covers prior bearish body
            if cur_body > 0 and prev_body < 0:
                if cur.close > prev.open and cur.open < prev.close:
                    return ConfirmationResult(
                        pattern_type="bullish_engulfing",
                        trigger_price=cur.close,
                        bar_index=len(bars) - 1,
                        volume_confirmed=self._volume_ok(cur, vol_ma),
                    )
        else:
            # Bearish engulfing
            if cur_body < 0 and prev_body > 0:
                if cur.close < prev.open and cur.open > prev.close:
                    return ConfirmationResult(
                        pattern_type="bearish_engulfing",
                        trigger_price=cur.close,
                        bar_index=len(bars) - 1,
                        volume_confirmed=self._volume_ok(cur, vol_ma),
                    )
        return None

    def _check_hammer(
        self, bars: list[Bar], zone: float, direction: Side, vol_ma: float
    ) -> ConfirmationResult | None:
        cur = bars[-1]
        body = abs(cur.close - cur.open)
        if body < 1e-10:
            return None

        if direction == Side.LONG:
            # Hammer: lower wick >= ratio * body, small upper wick
            lower_wick = min(cur.open, cur.close) - cur.low
            upper_wick = cur.high - max(cur.open, cur.close)
            if lower_wick >= self._p.hammer_wick_ratio * body and upper_wick < body:
                return ConfirmationResult(
                    pattern_type="hammer",
                    trigger_price=cur.close,
                    bar_index=len(bars) - 1,
                    volume_confirmed=self._volume_ok(cur, vol_ma),
                )
        else:
            # Shooting star: upper wick >= ratio * body, small lower wick
            upper_wick = cur.high - max(cur.open, cur.close)
            lower_wick = min(cur.open, cur.close) - cur.low
            if upper_wick >= self._p.hammer_wick_ratio * body and lower_wick < body:
                return ConfirmationResult(
                    pattern_type="shooting_star",
                    trigger_price=cur.close,
                    bar_index=len(bars) - 1,
                    volume_confirmed=self._volume_ok(cur, vol_ma),
                )
        return None

    def _check_inside_bar(
        self, bars: list[Bar], zone: float, direction: Side, vol_ma: float
    ) -> ConfirmationResult | None:
        if len(bars) < 3:
            return None
        mother = bars[-2]
        inside = bars[-1]

        # Inside bar: current bar's range within mother bar's range
        if inside.high <= mother.high and inside.low >= mother.low:
            if direction == Side.LONG:
                trigger = mother.high  # break above mother bar
            else:
                trigger = mother.low  # break below mother bar
            return ConfirmationResult(
                pattern_type="inside_bar_break",
                trigger_price=trigger,
                bar_index=len(bars) - 1,
                volume_confirmed=self._volume_ok(inside, vol_ma),
            )
        return None

    def _check_micro_shift(
        self, bars: list[Bar], zone: float, direction: Side, vol_ma: float
    ) -> ConfirmationResult | None:
        min_bars = self._p.micro_shift_min_bars
        if len(bars) < min_bars + 2:
            return None

        recent = bars[-(min_bars + 2):]

        if direction == Side.LONG:
            # Look for HL -> HH pattern (minimum 3-bar micro structure)
            lows = [b.low for b in recent]
            highs = [b.high for b in recent]
            # Find a higher low followed by a higher high
            for i in range(1, len(recent) - 1):
                if lows[i] > lows[i - 1] and highs[i + 1] > highs[i]:
                    return ConfirmationResult(
                        pattern_type="micro_structure_shift",
                        trigger_price=recent[-1].close,
                        bar_index=len(bars) - 1,
                        volume_confirmed=self._volume_ok(bars[-1], vol_ma),
                    )
        else:
            # Look for LH -> LL pattern
            lows = [b.low for b in recent]
            highs = [b.high for b in recent]
            for i in range(1, len(recent) - 1):
                if highs[i] < highs[i - 1] and lows[i + 1] < lows[i]:
                    return ConfirmationResult(
                        pattern_type="micro_structure_shift",
                        trigger_price=recent[-1].close,
                        bar_index=len(bars) - 1,
                        volume_confirmed=self._volume_ok(bars[-1], vol_ma),
                    )
        return None

    def _check_base_break(
        self, bars: list[Bar], zone: float, direction: Side, vol_ma: float
    ) -> ConfirmationResult | None:
        min_b = self._p.base_break_min_bars
        max_b = self._p.base_break_max_bars

        if len(bars) < max_b + 1:
            return None

        # Check for contracting-range base followed by a break
        for length in range(max_b, min_b - 1, -1):
            base_bars = bars[-(length + 1):-1]
            break_bar = bars[-1]

            # Check contracting ranges
            ranges = [b.high - b.low for b in base_bars]
            contracting = all(ranges[i] <= ranges[i - 1] * 1.1 for i in range(1, len(ranges)))
            if not contracting:
                continue

            base_high = max(b.high for b in base_bars)
            base_low = min(b.low for b in base_bars)

            if direction == Side.LONG and break_bar.close > base_high:
                return ConfirmationResult(
                    pattern_type="base_break",
                    trigger_price=break_bar.close,
                    bar_index=len(bars) - 1,
                    volume_confirmed=self._volume_ok(break_bar, vol_ma),
                )
            elif direction == Side.SHORT and break_bar.close < base_low:
                return ConfirmationResult(
                    pattern_type="base_break",
                    trigger_price=break_bar.close,
                    bar_index=len(bars) - 1,
                    volume_confirmed=self._volume_ok(break_bar, vol_ma),
                )
        return None
