"""Trend stop placement — swing-based and ATR-based stops."""

from __future__ import annotations

from crypto_trader.core.models import Bar, Side

from .config import TrendStopParams


class StopPlacer:
    """Compute protective stop level using swing and/or ATR methods."""

    def __init__(self, cfg: TrendStopParams) -> None:
        self._cfg = cfg

    def compute(
        self,
        h1_bars: list[Bar],
        direction: Side,
        atr: float,
        entry_price: float,
    ) -> float:
        cfg = self._cfg

        # Swing-based stop
        swing_stop = None
        if cfg.use_swing:
            lookback = min(cfg.swing_lookback, len(h1_bars))
            recent = h1_bars[-lookback:]
            if direction == Side.LONG:
                swing_low = min(b.low for b in recent)
                swing_stop = swing_low
            else:
                swing_high = max(b.high for b in recent)
                swing_stop = swing_high

        # ATR-based stop
        if direction == Side.LONG:
            atr_stop = entry_price - cfg.atr_mult * atr
        else:
            atr_stop = entry_price + cfg.atr_mult * atr

        # Select stop
        if swing_stop is not None and cfg.use_farther:
            # Take the FARTHER (more generous) stop
            if direction == Side.LONG:
                stop = min(swing_stop, atr_stop)
            else:
                stop = max(swing_stop, atr_stop)
        elif swing_stop is not None:
            stop = swing_stop
        else:
            stop = atr_stop

        # Apply buffer
        if direction == Side.LONG:
            stop *= (1 - cfg.buffer_pct)
        else:
            stop *= (1 + cfg.buffer_pct)

        # Enforce minimum stop distance
        stop_distance = abs(entry_price - stop)
        min_distance = cfg.min_stop_atr * atr
        if stop_distance < min_distance:
            if direction == Side.LONG:
                stop = entry_price - min_distance
            else:
                stop = entry_price + min_distance

        return stop
