"""Trend trailing stop — R-adaptive with optional structure trail."""

from __future__ import annotations

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .config import TrendTrailParams


class TrailManager:
    """R-adaptive trailing stop with optional structure-based trail."""

    def __init__(self, cfg: TrendTrailParams) -> None:
        self._cfg = cfg

    def update(
        self,
        sym: str,
        direction: Side,
        h1_bars: list[Bar],
        h1_ind: IndicatorSnapshot,
        current_stop: float | None,
        bars_since_entry: int,
        current_r: float,
        mfe_r: float,
    ) -> float | None:
        """Return new stop price, or None if no update needed."""
        cfg = self._cfg

        # Activation gate (OR logic)
        if not (bars_since_entry >= cfg.trail_activation_bars
                or current_r >= cfg.trail_activation_r):
            return None

        if not h1_bars or h1_ind.atr <= 0:
            return None

        bar = h1_bars[-1]
        atr = h1_ind.atr

        # R-adaptive buffer calculation
        if cfg.trail_r_adaptive:
            adaptive_r = mfe_r if cfg.trail_use_mfe_for_adaptive else current_r
            r_clamped = max(0.0, min(adaptive_r, cfg.trail_r_ceiling))
            r_frac = r_clamped / cfg.trail_r_ceiling if cfg.trail_r_ceiling > 0 else 0
            buffer_mult = cfg.trail_buffer_wide * (1 - r_frac) + cfg.trail_buffer_tight * r_frac
        else:
            buffer_mult = cfg.trail_buffer_wide

        if direction == Side.LONG:
            r_adaptive_stop = bar.close - atr * buffer_mult
        else:
            r_adaptive_stop = bar.close + atr * buffer_mult

        new_stop = r_adaptive_stop

        # Structure trail (when enabled)
        if cfg.structure_trail_enabled:
            struct_stop = self._structure_stop(h1_bars, direction, atr)
            if struct_stop is not None:
                # Pick the tighter (more aggressive) stop
                if direction == Side.LONG:
                    new_stop = max(new_stop, struct_stop)
                else:
                    new_stop = min(new_stop, struct_stop)

        # Only advance stop — never retreat
        if current_stop is not None:
            if direction == Side.LONG and new_stop <= current_stop:
                return None
            if direction == Side.SHORT and new_stop >= current_stop:
                return None

        return new_stop

    def remove(self, sym: str) -> None:
        pass  # No per-symbol state to clean up

    def _structure_stop(
        self,
        bars: list[Bar],
        direction: Side,
        atr: float,
    ) -> float | None:
        """Find swing-based trailing stop."""
        cfg = self._cfg
        lookback = min(cfg.structure_swing_lookback, len(bars) - 2)
        if lookback < 3:
            return None

        window = bars[-lookback - 2:]  # Extra for fractal check

        # Find most recent swing low (long) or swing high (short)
        for i in range(len(window) - 3, 1, -1):
            b = window[i]
            if direction == Side.LONG:
                if (b.low < window[i - 1].low and b.low < window[i + 1].low):
                    return b.low - atr * 0.3
            else:
                if (b.high > window[i - 1].high and b.high > window[i + 1].high):
                    return b.high + atr * 0.3

        return None
