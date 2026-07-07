"""Breakout stop placement — balance edge, retest extreme, and ATR stops."""

from __future__ import annotations

from crypto_trader.core.models import Bar, Side

from .config import BreakoutStopParams
from .setup import BreakoutSetupResult


class StopPlacer:
    """Compute protective stop level using three sources with use_farther logic."""

    def __init__(self, cfg: BreakoutStopParams) -> None:
        self._p = cfg

    def compute(
        self,
        setup: BreakoutSetupResult,
        retest_bar: Bar | None,
        entry_price: float,
        atr: float,
        direction: Side,
    ) -> float:
        """Return a stop-loss price for the breakout trade.

        *retest_bar*: the retest candle for Model 2 (``None`` for Model 1).
        """
        cfg = self._p
        candidates: list[float] = []

        # 1. Balance edge stop
        if cfg.use_balance_edge:
            zone = setup.balance_zone
            if direction == Side.LONG:
                candidates.append(zone.lower)
            else:
                candidates.append(zone.upper)

        # 2. Retest extreme stop (Model 2 only)
        if cfg.use_retest_extreme and retest_bar is not None:
            if direction == Side.LONG:
                candidates.append(retest_bar.low)
            else:
                candidates.append(retest_bar.high)

        # 3. ATR stop
        if cfg.use_atr_stop:
            if direction == Side.LONG:
                candidates.append(entry_price - cfg.atr_mult * atr)
            else:
                candidates.append(entry_price + cfg.atr_mult * atr)

        # Select stop
        if not candidates:
            # Fallback: pure ATR
            if direction == Side.LONG:
                stop = entry_price - cfg.atr_mult * atr
            else:
                stop = entry_price + cfg.atr_mult * atr
        elif cfg.use_farther:
            # Most generous (farther from entry)
            if direction == Side.LONG:
                stop = min(candidates)
            else:
                stop = max(candidates)
        else:
            # Tightest stop
            if direction == Side.LONG:
                stop = max(candidates)
            else:
                stop = min(candidates)

        # Apply buffer
        if direction == Side.LONG:
            stop *= 1 - cfg.buffer_pct
        else:
            stop *= 1 + cfg.buffer_pct

        # Enforce minimum stop distance
        min_dist = cfg.min_stop_atr * atr
        dist = abs(entry_price - stop)
        if dist < min_dist:
            if direction == Side.LONG:
                stop = entry_price - min_dist
            else:
                stop = entry_price + min_dist

        return stop
