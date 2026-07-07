"""Multi-timeframe directional bias detection."""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_trader.core.models import Bar, Side
from crypto_trader.strategy.momentum.config import BiasParams
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot


@dataclass(frozen=True)
class BiasResult:
    direction: Side | None
    h4_score: int  # 0-3
    h1_score: int  # 0-3
    confidence: float  # 0.0-1.0
    reasons: tuple[str, ...]


class BiasDetector:
    def __init__(self, params: BiasParams) -> None:
        self._p = params

    def compute(
        self,
        h4_bars: list[Bar],
        h1_bars: list[Bar],
        h4_ind: IndicatorSnapshot | None,
        h1_ind: IndicatorSnapshot | None,
    ) -> BiasResult:
        if h4_ind is None or h1_ind is None or len(h4_bars) < 5 or len(h1_bars) < 5:
            return BiasResult(direction=None, h4_score=0, h1_score=0, confidence=0.0, reasons=("insufficient data",))

        h4_long, h4_short, h4_reasons = self._score_h4(h4_bars, h4_ind)
        h1_long, h1_short, h1_reasons = self._score_h1(h1_bars, h1_ind)

        h4_dir = self._direction_from_scores(h4_long, h4_short, self._p.min_4h_conditions)
        h1_dir = self._direction_from_scores(h1_long, h1_short, self._p.min_1h_conditions)

        reasons = h4_reasons + h1_reasons

        # Disagreement or neutrality
        if h4_dir is None or h1_dir is None:
            return BiasResult(
                direction=None,
                h4_score=max(h4_long, h4_short),
                h1_score=max(h1_long, h1_short),
                confidence=0.0,
                reasons=tuple(reasons + ["neutral or insufficient conditions"]),
            )
        if h4_dir != h1_dir:
            return BiasResult(
                direction=None,
                h4_score=max(h4_long, h4_short),
                h1_score=max(h1_long, h1_short),
                confidence=0.0,
                reasons=tuple(reasons + ["H4/H1 disagree"]),
            )

        h4_score = h4_long if h4_dir == Side.LONG else h4_short
        h1_score = h1_long if h1_dir == Side.LONG else h1_short
        confidence = (h4_score + h1_score) / 6.0

        return BiasResult(
            direction=h4_dir,
            h4_score=h4_score,
            h1_score=h1_score,
            confidence=confidence,
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------

    def _score_h4(self, bars: list[Bar], ind: IndicatorSnapshot) -> tuple[int, int, list[str]]:
        long_count = 0
        short_count = 0
        reasons: list[str] = []

        # 1) Price vs EMA200
        price = bars[-1].close
        if price > ind.ema_slow:
            long_count += 1
            reasons.append("H4: price > EMA200")
        elif price < ind.ema_slow:
            short_count += 1
            reasons.append("H4: price < EMA200")

        # 2) EMA200 slope
        slope_arr = ind.ema_slow_arr
        if len(slope_arr) >= self._p.h4_ema_slope_lookback:
            slope = slope_arr[-1] - slope_arr[-self._p.h4_ema_slope_lookback]
            if slope > 0:
                long_count += 1
                reasons.append("H4: EMA200 rising")
            elif slope < 0:
                short_count += 1
                reasons.append("H4: EMA200 falling")

        # 3) HH/HL structure
        structure = _detect_structure(bars, self._p.h4_structure_lookback)
        if structure == "bullish":
            long_count += 1
            reasons.append("H4: bullish structure (HH/HL)")
        elif structure == "bearish":
            short_count += 1
            reasons.append("H4: bearish structure (LH/LL)")

        return long_count, short_count, reasons

    def _score_h1(self, bars: list[Bar], ind: IndicatorSnapshot) -> tuple[int, int, list[str]]:
        long_count = 0
        short_count = 0
        reasons: list[str] = []

        # 1) EMA20 vs EMA50
        if ind.ema_fast > ind.ema_mid:
            long_count += 1
            reasons.append("H1: EMA20 > EMA50")
        elif ind.ema_fast < ind.ema_mid:
            short_count += 1
            reasons.append("H1: EMA20 < EMA50")

        # 2) Structure
        structure = _detect_structure(bars, 20)
        if structure == "bullish":
            long_count += 1
            reasons.append("H1: bullish structure")
        elif structure == "bearish":
            short_count += 1
            reasons.append("H1: bearish structure")

        # 3) ADX confirmation
        if ind.adx > self._p.h1_adx_threshold:
            if ind.di_plus > ind.di_minus:
                long_count += 1
                reasons.append(f"H1: ADX {ind.adx:.1f} with DI+ dominant")
            elif ind.di_minus > ind.di_plus:
                short_count += 1
                reasons.append(f"H1: ADX {ind.adx:.1f} with DI- dominant")

        return long_count, short_count, reasons

    @staticmethod
    def _direction_from_scores(long_score: int, short_score: int, min_conditions: int) -> Side | None:
        if long_score >= min_conditions and long_score > short_score:
            return Side.LONG
        if short_score >= min_conditions and short_score > long_score:
            return Side.SHORT
        return None


def _detect_structure(bars: list[Bar], lookback: int) -> str:
    """Detect swing structure: 'bullish', 'bearish', or 'neutral'."""
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    if len(recent) < 5:
        return "neutral"

    swings = _find_swing_points(recent)
    if len(swings) < 4:
        return "neutral"

    # Check last 4 swings for HH/HL (bullish) or LH/LL (bearish)
    last_four = swings[-4:]
    highs = [s[1] for s in last_four if s[2] == "high"]
    lows = [s[1] for s in last_four if s[2] == "low"]

    hh = len(highs) >= 2 and highs[-1] > highs[-2]
    hl = len(lows) >= 2 and lows[-1] > lows[-2]
    lh = len(highs) >= 2 and highs[-1] < highs[-2]
    ll = len(lows) >= 2 and lows[-1] < lows[-2]

    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "neutral"


def _find_swing_points(bars: list[Bar]) -> list[tuple[int, float, str]]:
    """Find swing highs and lows. Returns list of (index, price, 'high'|'low')."""
    swings: list[tuple[int, float, str]] = []
    if len(bars) < 3:
        return swings
    for i in range(1, len(bars) - 1):
        if bars[i].high > bars[i - 1].high and bars[i].high > bars[i + 1].high:
            swings.append((i, bars[i].high, "high"))
        if bars[i].low < bars[i - 1].low and bars[i].low < bars[i + 1].low:
            swings.append((i, bars[i].low, "low"))
    return sorted(swings, key=lambda s: s[0])
