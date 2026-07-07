"""D1 regime classification for trend strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .config import RegimeParams


@dataclass(frozen=True)
class RegimeResult:
    tier: str               # "A", "B", "none"
    direction: Side | None  # LONG, SHORT, or None
    adx: float
    ema_fast: float
    ema_mid: float
    reasons: tuple[str, ...]


@dataclass
class StructureState:
    """Tracks D1 swing sequence for HH/HL or LL/LH classification."""
    last_swing_highs: list[float] = field(default_factory=list)
    last_swing_lows: list[float] = field(default_factory=list)
    pattern: str = "mixed"  # "bullish", "bearish", "mixed"


class StructureTracker:
    """Detects swing highs/lows from D1 bars using N-bar fractal."""

    def __init__(self, lookback: int = 10) -> None:
        self._lookback = lookback
        self._bars: list[Bar] = []
        self._state = StructureState()

    @property
    def state(self) -> StructureState:
        return self._state

    def update(self, bar: Bar) -> None:
        self._bars.append(bar)
        if len(self._bars) < 5:
            return

        # Check if bar at -3 is a swing (need 2 bars on each side)
        idx = len(self._bars) - 3
        candidate = self._bars[idx]
        left1, left2 = self._bars[idx - 2], self._bars[idx - 1]
        right1, right2 = self._bars[idx + 1], self._bars[idx + 2]

        # Swing high: candidate high > neighbors
        if (candidate.high > left1.high and candidate.high > left2.high
                and candidate.high > right1.high and candidate.high > right2.high):
            self._state.last_swing_highs.append(candidate.high)
            if len(self._state.last_swing_highs) > 4:
                self._state.last_swing_highs = self._state.last_swing_highs[-4:]

        # Swing low: candidate low < neighbors
        if (candidate.low < left1.low and candidate.low < left2.low
                and candidate.low < right1.low and candidate.low < right2.low):
            self._state.last_swing_lows.append(candidate.low)
            if len(self._state.last_swing_lows) > 4:
                self._state.last_swing_lows = self._state.last_swing_lows[-4:]

        self._classify()

    def _classify(self) -> None:
        highs = self._state.last_swing_highs
        lows = self._state.last_swing_lows

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1] > highs[-2]
            hl = lows[-1] > lows[-2]
            ll = lows[-1] < lows[-2]
            lh = highs[-1] < highs[-2]

            if hh and hl:
                self._state.pattern = "bullish"
            elif ll and lh:
                self._state.pattern = "bearish"
            else:
                self._state.pattern = "mixed"


class RegimeClassifier:
    """Classify current D1 regime into A-tier, B-tier, or no-trade."""

    def __init__(self, cfg: RegimeParams) -> None:
        self._cfg = cfg

    def evaluate_h1(self, close: float, h1_ind: IndicatorSnapshot) -> RegimeResult | None:
        """H1-level regime supplement — returns B-tier only."""
        if not self._cfg.h1_regime_enabled:
            return None

        adx = h1_ind.adx
        ema_f = h1_ind.ema_fast
        ema_m = h1_ind.ema_mid

        if adx < self._cfg.h1_min_adx:
            return None

        # Check LONG
        if close > ema_f and close > ema_m and ema_f > ema_m:
            return RegimeResult(
                "B", Side.LONG, adx, ema_f, ema_m,
                ("h1_regime", f"h1_adx_{adx:.1f}>={self._cfg.h1_min_adx}"),
            )

        # Check SHORT
        if close < ema_f and close < ema_m and ema_f < ema_m:
            return RegimeResult(
                "B", Side.SHORT, adx, ema_f, ema_m,
                ("h1_regime", f"h1_adx_{adx:.1f}>={self._cfg.h1_min_adx}"),
            )

        return None

    def evaluate(
        self,
        d1_bar: Bar,
        d1_ind: IndicatorSnapshot,
        structure: StructureState,
    ) -> RegimeResult:
        cfg = self._cfg
        adx = d1_ind.adx
        ema_f = d1_ind.ema_fast
        ema_m = d1_ind.ema_mid
        close = d1_bar.close

        # No-trade zone
        if adx < cfg.no_trade_max_adx:
            return RegimeResult("none", None, adx, ema_f, ema_m,
                                ("adx_below_no_trade",))

        # Determine direction candidates
        long_result = self._check_direction(
            close, ema_f, ema_m, adx, d1_ind, structure, Side.LONG
        )
        short_result = self._check_direction(
            close, ema_f, ema_m, adx, d1_ind, structure, Side.SHORT
        )

        # Prefer A-tier, then B-tier
        if long_result and long_result.tier == "A":
            return long_result
        if short_result and short_result.tier == "A":
            return short_result
        if long_result and long_result.tier == "B":
            return long_result
        if short_result and short_result.tier == "B":
            return short_result

        return RegimeResult("none", None, adx, ema_f, ema_m,
                            ("no_directional_alignment",))

    def _check_direction(
        self,
        close: float,
        ema_f: float,
        ema_m: float,
        adx: float,
        d1_ind: IndicatorSnapshot,
        structure: StructureState,
        direction: Side,
    ) -> RegimeResult | None:
        cfg = self._cfg

        if direction == Side.LONG:
            price_above_fast = close > ema_f
            price_above_mid = close > ema_m
            ema_ordered = ema_f > ema_m
            struct_ok = structure.pattern == "bullish"
        else:
            price_above_fast = close < ema_f
            price_above_mid = close < ema_m
            ema_ordered = ema_f < ema_m
            struct_ok = structure.pattern == "bearish"

        # A-tier: full alignment
        a_reasons: list[str] = []
        a_ok = True

        if price_above_fast and price_above_mid:
            a_reasons.append("price_beyond_emas")
        else:
            a_ok = False

        if adx >= cfg.a_min_adx:
            a_reasons.append(f"adx_{adx:.1f}>={cfg.a_min_adx}")
        else:
            a_ok = False

        if cfg.require_ema_cross:
            if ema_ordered:
                a_reasons.append("ema_cross_confirmed")
            else:
                a_ok = False

        if cfg.require_structure:
            if struct_ok:
                a_reasons.append(f"structure_{structure.pattern}")
            else:
                a_ok = False

        if a_ok:
            return RegimeResult("A", direction, adx, ema_f, ema_m,
                                tuple(a_reasons))

        # B-tier: relaxed
        b_reasons: list[str] = []
        b_ok = True

        if price_above_fast:
            b_reasons.append("price_beyond_fast_ema")
        else:
            b_ok = False

        if adx >= cfg.b_min_adx:
            b_reasons.append(f"adx_{adx:.1f}>={cfg.b_min_adx}")
        else:
            b_ok = False

        if cfg.b_adx_rising_required:
            if d1_ind.adx_rising:
                b_reasons.append("adx_rising")
            else:
                b_ok = False

        if b_ok:
            return RegimeResult("B", direction, adx, ema_f, ema_m,
                                tuple(b_reasons))

        return None
