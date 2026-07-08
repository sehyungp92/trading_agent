"""Model 1 (breakout-close) and Model 2 (retest) confirmation."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .config import BreakoutConfirmParams
from .setup import BreakoutSetupResult


@dataclass(frozen=True)
class BreakoutConfirmation:
    """Confirmed entry signal with model attribution."""

    model: str  # "model1_close" or "model2_retest"
    trigger_price: float
    bar_index: int
    volume_confirmed: bool


@dataclass
class _PendingRetest:
    """Internal state for a pending Model 2 retest."""

    setup: BreakoutSetupResult
    breakout_bar_idx: int


class ConfirmationDetector:
    """Detect Model 1 (breakout-close) and Model 2 (retest) confirmations."""

    def __init__(self, cfg: BreakoutConfirmParams) -> None:
        self._p = cfg
        self._pending: dict[str, _PendingRetest] = {}

    def check_breakout_close(
        self,
        bar: Bar,
        setup: BreakoutSetupResult,
        m30_ind: IndicatorSnapshot | None,
    ) -> BreakoutConfirmation | None:
        """Model 1: immediate entry on breakout close with quality gates."""
        # Volume gate
        volume_ok = True
        if m30_ind is not None and m30_ind.volume_ma is not None and m30_ind.volume_ma > 0:
            volume_mult = bar.volume / m30_ind.volume_ma
            if self._p.model1_require_volume and volume_mult < self._p.model1_min_volume_mult:
                return None
            volume_ok = volume_mult >= 1.0
        elif self._p.model1_require_volume:
            return None

        # Directional close gate
        if self._p.model1_require_direction_close:
            if setup.direction == Side.LONG and bar.close < bar.open:
                return None  # Bearish close on long breakout
            if setup.direction == Side.SHORT and bar.close > bar.open:
                return None  # Bullish close on short breakout

        return BreakoutConfirmation(
            model="model1_close",
            trigger_price=bar.close,
            bar_index=0,
            volume_confirmed=volume_ok,
        )

    def register_breakout(
        self,
        sym: str,
        setup: BreakoutSetupResult,
        bar_idx: int,
    ) -> None:
        """Register a breakout for Model 2 retest monitoring.

        Keeps an existing same-zone pending retest so repeated closes outside a
        zone do not reset the deterministic retest expiry window.
        """
        existing = self._pending.get(sym)
        if (
            existing is not None
            and existing.setup.balance_zone == setup.balance_zone
            and existing.setup.direction == setup.direction
        ):
            return
        self._pending[sym] = _PendingRetest(
            setup=setup,
            breakout_bar_idx=bar_idx,
        )

    def check_retest(
        self,
        sym: str,
        bar: Bar,
        bars: list[Bar],
        atr: float,
        bar_index: int,
    ) -> BreakoutConfirmation | None:
        """Check whether price retested the balance edge within the window.

        Returns a confirmation if the retest is valid, else ``None``.
        Expired pendings are cleaned up automatically.
        """
        pending = self._pending.get(sym)
        if pending is None:
            return None

        bars_since = bar_index - pending.breakout_bar_idx
        if bars_since > self._p.retest_max_bars:
            del self._pending[sym]
            return None

        zone = pending.setup.balance_zone
        direction = pending.setup.direction
        retest_depth = self._p.retest_zone_atr * atr

        # For LONG: price pulled back near zone.upper
        # For SHORT: price pulled back near zone.lower
        if direction == Side.LONG:
            near_edge = (
                abs(bar.low - zone.upper) <= retest_depth or bar.low <= zone.upper
            )
            closing_in_direction = bar.close > bar.open  # Bullish close
            rejection_ok = bar.close >= zone.upper
        else:
            near_edge = (
                abs(bar.high - zone.lower) <= retest_depth or bar.high >= zone.lower
            )
            closing_in_direction = bar.close < bar.open  # Bearish close
            rejection_ok = bar.close <= zone.lower

        if not near_edge or not closing_in_direction:
            return None

        if self._p.retest_require_rejection and not rejection_ok:
            return None

        # Optional volume decline check
        volume_ok = True
        if self._p.retest_require_volume_decline and len(bars) >= 2:
            volume_ok = bar.volume < bars[-2].volume * self._p.volume_decline_threshold
            if not volume_ok:
                return None

        del self._pending[sym]
        return BreakoutConfirmation(
            model="model2_retest",
            trigger_price=bar.close,
            bar_index=bar_index,
            volume_confirmed=volume_ok,
        )

    def has_pending(self, sym: str) -> bool:
        """Return ``True`` if *sym* has a pending retest."""
        return sym in self._pending

    def clear_pending(self, sym: str) -> None:
        """Remove any pending retest for *sym*."""
        self._pending.pop(sym, None)

    def get_pending_setup(self, sym: str) -> BreakoutSetupResult | None:
        """Return the setup associated with a pending retest, or ``None``."""
        p = self._pending.get(sym)
        return p.setup if p else None
