"""Breakout trailing stop — R-adaptive with optional structure trail."""

from __future__ import annotations

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .config import BreakoutTrailParams


class TrailManager:
    """R-adaptive trailing stop with optional structure-based trail."""

    def __init__(self, cfg: BreakoutTrailParams) -> None:
        self._p = cfg
        self._last_stops: dict[str, float] = {}

    def update(
        self,
        sym: str,
        direction: Side,
        bars: list[Bar],
        m30_ind: IndicatorSnapshot | None,
        current_stop: float | None,
        bars_since_entry: int,
        current_r: float,
        mfe_r: float,
    ) -> float | None:
        """Return a new trailing stop price, or ``None`` if no update needed."""
        cfg = self._p

        # Activation gate (OR logic) — uses mfe_r so intra-bar peaks count
        if not (bars_since_entry >= cfg.trail_activation_bars
                or mfe_r >= cfg.trail_activation_r):
            return None

        if not cfg.trail_r_adaptive:
            return None

        atr = m30_ind.atr if m30_ind is not None else 0.0
        if atr <= 0:
            return None

        if not bars:
            return None
        bar = bars[-1]

        # R-adaptive buffer calculation
        r_clamped = max(0.0, min(mfe_r, cfg.trail_r_ceiling))
        r_frac = r_clamped / cfg.trail_r_ceiling if cfg.trail_r_ceiling > 0 else 0.0
        buffer_mult = cfg.trail_buffer_wide * (1 - r_frac) + cfg.trail_buffer_tight * r_frac

        if direction == Side.LONG:
            new_stop = bar.close - atr * buffer_mult
        else:
            new_stop = bar.close + atr * buffer_mult

        # Structure trail (optional)
        if cfg.structure_trail_enabled and len(bars) >= cfg.structure_swing_lookback:
            lookback = bars[-cfg.structure_swing_lookback:]
            if direction == Side.LONG:
                struct_stop = min(b.low for b in lookback)
                new_stop = max(new_stop, struct_stop)  # Tighter of two
            else:
                struct_stop = max(b.high for b in lookback)
                new_stop = min(new_stop, struct_stop)

        # Only move stop in favorable direction
        if current_stop is not None:
            if direction == Side.LONG and new_stop <= current_stop:
                return None
            if direction == Side.SHORT and new_stop >= current_stop:
                return None

        self._last_stops[sym] = new_stop
        return new_stop

    def remove(self, sym: str) -> None:
        """Clean up per-symbol trailing state."""
        self._last_stops.pop(sym, None)
