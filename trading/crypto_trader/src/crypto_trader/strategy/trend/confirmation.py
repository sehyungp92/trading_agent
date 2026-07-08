"""Trend confirmation — trigger pattern detection on H1 bars."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .config import TrendConfirmationParams


@dataclass(frozen=True)
class TriggerResult:
    pattern: str           # "engulfing", "hammer", "ema_reclaim", "structure_break"
    trigger_price: float   # For conservative entry (break-of-high/low)
    bar_index: int
    volume_confirmed: bool


class TriggerDetector:
    """Detect confirmation trigger patterns on H1 bars."""

    def __init__(self, cfg: TrendConfirmationParams) -> None:
        self._cfg = cfg

    def check(
        self,
        h1_bars: list[Bar],
        direction: Side,
        h1_ind: IndicatorSnapshot,
    ) -> TriggerResult | None:
        if len(h1_bars) < 2:
            return None

        cfg = self._cfg
        current = h1_bars[-1]
        prev = h1_bars[-2]
        vol_ok = self._volume_confirmed(h1_bars, h1_ind)

        # 1. Engulfing
        if cfg.enable_engulfing:
            result = self._check_engulfing(current, prev, direction, vol_ok)
            if self._accept_result(result):
                return result

        # 2. Hammer / rejection candle
        if cfg.enable_hammer:
            result = self._check_hammer(current, direction, vol_ok)
            if self._accept_result(result):
                return result

        # 3. EMA reclaim
        if cfg.enable_ema_reclaim:
            result = self._check_ema_reclaim(current, prev, direction, h1_ind, vol_ok)
            if self._accept_result(result):
                return result

        # 4. Structure break
        if cfg.enable_structure_break and len(h1_bars) >= 5:
            result = self._check_structure_break(h1_bars, direction, vol_ok)
            if self._accept_result(result):
                return result

        return None

    def _accept_result(self, result: TriggerResult | None) -> bool:
        if result is None:
            return False
        if self._cfg.enforce_volume_on_trigger and not result.volume_confirmed:
            return False
        return True

    def _check_engulfing(
        self,
        current: Bar,
        prev: Bar,
        direction: Side,
        vol_ok: bool,
    ) -> TriggerResult | None:
        curr_body = current.close - current.open
        prev_body = prev.close - prev.open

        if direction == Side.LONG:
            # Bullish engulfing: current body positive and engulfs prior negative body
            if (curr_body > 0 and prev_body < 0
                    and current.close > prev.open
                    and current.open < prev.close):
                return TriggerResult("engulfing", current.high, 0, vol_ok)
        else:
            # Bearish engulfing
            if (curr_body < 0 and prev_body > 0
                    and current.close < prev.open
                    and current.open > prev.close):
                return TriggerResult("engulfing", current.low, 0, vol_ok)

        return None

    def _check_hammer(
        self,
        bar: Bar,
        direction: Side,
        vol_ok: bool,
    ) -> TriggerResult | None:
        body = abs(bar.close - bar.open)
        if body == 0:
            return None

        if direction == Side.LONG:
            # Lower wick should be long (hammer)
            lower_wick = min(bar.open, bar.close) - bar.low
            if lower_wick >= self._cfg.hammer_wick_ratio * body:
                return TriggerResult("hammer", bar.high, 0, vol_ok)
        else:
            # Upper wick should be long (shooting star / inverted hammer)
            upper_wick = bar.high - max(bar.open, bar.close)
            if upper_wick >= self._cfg.hammer_wick_ratio * body:
                return TriggerResult("hammer", bar.low, 0, vol_ok)

        return None

    def _check_ema_reclaim(
        self,
        current: Bar,
        prev: Bar,
        direction: Side,
        h1_ind: IndicatorSnapshot,
        vol_ok: bool,
    ) -> TriggerResult | None:
        ema = h1_ind.ema_fast

        if direction == Side.LONG:
            # Prior bar below EMA, current bar closes above
            if prev.close < ema and current.close > ema:
                return TriggerResult("ema_reclaim", current.high, 0, vol_ok)
        else:
            # Prior bar above EMA, current bar closes below
            if prev.close > ema and current.close < ema:
                return TriggerResult("ema_reclaim", current.low, 0, vol_ok)

        return None

    def _check_structure_break(
        self,
        bars: list[Bar],
        direction: Side,
        vol_ok: bool,
    ) -> TriggerResult | None:
        """Break of pullback structure (lower-high for long, higher-low for short)."""
        current = bars[-1]
        lookback = bars[-5:-1]  # Exclude current bar

        if direction == Side.LONG:
            # Find the pullback's lower-high in last 4 bars
            recent_highs = [b.high for b in lookback]
            if len(recent_highs) >= 2:
                lower_high = min(recent_highs[-2:])
                if current.close > lower_high:
                    return TriggerResult("structure_break", current.high, 0, vol_ok)
        else:
            recent_lows = [b.low for b in lookback]
            if len(recent_lows) >= 2:
                higher_low = max(recent_lows[-2:])
                if current.close < higher_low:
                    return TriggerResult("structure_break", current.low, 0, vol_ok)

        return None

    def _volume_confirmed(
        self,
        bars: list[Bar],
        h1_ind: IndicatorSnapshot,
    ) -> bool:
        """Check if current bar has above-average volume."""
        if not self._cfg.require_volume_confirm:
            return True
        if len(bars) < 2:
            return False

        current = bars[-1]
        vol_ma = h1_ind.volume_ma if h1_ind.volume_ma and h1_ind.volume_ma > 0 else None
        if vol_ma is None:
            # Fallback: average of last 20 bars
            lookback = bars[-min(20, len(bars)):]
            vol_ma = sum(b.volume for b in lookback) / len(lookback)

        return current.volume >= vol_ma * self._cfg.volume_threshold_mult
