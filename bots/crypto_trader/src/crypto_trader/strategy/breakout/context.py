"""H4 directional bias analyzer — simplified regime classification."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .config import ContextParams


@dataclass(frozen=True)
class ContextBias:
    """Directional bias derived from H4 indicators."""

    direction: Side | None  # LONG, SHORT, or None (neutral)
    strength: str  # "strong", "moderate", "none"
    reasons: tuple[str, ...]


class ContextAnalyzer:
    """Evaluate H4 directional bias from EMA arrangement and ADX strength."""

    def __init__(self, cfg: ContextParams) -> None:
        self._p = cfg

    def evaluate(self, h4_ind: IndicatorSnapshot | None) -> ContextBias:
        """Return directional bias from H4 indicator snapshot.

        Returns neutral bias when *h4_ind* is ``None`` (warmup period).
        """
        if h4_ind is None:
            return ContextBias(direction=None, strength="none", reasons=("warmup",))

        ema_f = h4_ind.ema_fast
        ema_m = h4_ind.ema_mid
        ema_s = h4_ind.ema_slow
        adx = h4_ind.adx

        # --- Strong bias: full EMA ordering + ADX above strong threshold ---
        long_ordered = ema_f > ema_m > ema_s
        short_ordered = ema_f < ema_m < ema_s

        if long_ordered and adx >= self._p.strong_min_adx:
            return ContextBias(
                direction=Side.LONG,
                strength="strong",
                reasons=("ema_ordered_long", "adx_strong"),
            )
        if short_ordered and adx >= self._p.strong_min_adx:
            return ContextBias(
                direction=Side.SHORT,
                strength="strong",
                reasons=("ema_ordered_short", "adx_strong"),
            )

        # --- Moderate bias: fast vs mid EMA + ADX above threshold ---
        if ema_f > ema_m and adx >= self._p.h4_adx_threshold:
            return ContextBias(
                direction=Side.LONG,
                strength="moderate",
                reasons=("ema_fast_above_mid",),
            )
        if ema_f < ema_m and adx >= self._p.h4_adx_threshold:
            return ContextBias(
                direction=Side.SHORT,
                strength="moderate",
                reasons=("ema_fast_below_mid",),
            )

        # --- No directional edge ---
        reasons: list[str] = []
        if adx < self._p.h4_adx_threshold:
            reasons.append("adx_below_threshold")
        if not (ema_f > ema_m) and not (ema_f < ema_m):
            reasons.append("ema_flat")

        return ContextBias(
            direction=None,
            strength="none",
            reasons=tuple(reasons) if reasons else ("no_directional_edge",),
        )
