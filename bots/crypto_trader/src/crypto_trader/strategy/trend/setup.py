"""Trend setup detection: impulse plus pullback into an anchor zone."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Bar, SetupGrade, Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .config import TrendSetupParams
from .regime import RegimeResult


@dataclass(frozen=True)
class TrendSetupResult:
    grade: SetupGrade
    direction: Side
    impulse_start: float
    impulse_end: float
    impulse_atr_move: float
    pullback_depth: float
    confluences: tuple[str, ...]
    zone_price: float
    room_r: float
    stop_level: float
    setup_score: float = 0.0


@dataclass(frozen=True)
class ImpulseLeg:
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    atr_move: float


class SetupDetector:
    """Detect trend continuation setups: impulse -> pullback -> anchor zone."""

    _CONFLUENCE_WEIGHTS: dict[str, float] = {
        "h1_ema_zone": 0.85,
        "breakout_retest": 1.00,
        "rsi_pullback": 0.70,
        "d1_ema_support": 0.95,
        "weekly_level": 0.85,
        "volume_contraction": 0.55,
    }

    def __init__(self, cfg: TrendSetupParams) -> None:
        self._cfg = cfg

    def detect(
        self,
        h1_bars: list[Bar],
        h1_ind: IndicatorSnapshot,
        d1_ind: IndicatorSnapshot | None,
        regime: RegimeResult,
        weekly_high: float | None,
        weekly_low: float | None,
        min_confluences_override: int | None = None,
    ) -> TrendSetupResult | None:
        if not h1_bars or regime.tier == "none" or regime.direction is None:
            return None

        cfg = self._cfg
        direction = regime.direction
        atr = h1_ind.atr
        if atr <= 0:
            return None

        current_bar = h1_bars[-1]
        current_price = current_bar.close

        impulse = self._find_impulse(h1_bars, direction, atr)
        if impulse is None:
            return None

        imp_range = abs(impulse.end_price - impulse.start_price)
        if imp_range == 0:
            return None

        if direction == Side.LONG:
            pullback_depth = (impulse.end_price - current_price) / imp_range
        else:
            pullback_depth = (current_price - impulse.end_price) / imp_range

        if pullback_depth < 0 or pullback_depth > cfg.pullback_max_retrace:
            return None

        in_ema_zone = self._in_ema_zone(current_price, h1_ind.ema_fast, h1_ind.ema_mid, direction)

        if cfg.require_orderly_pullback and not self._is_orderly_basic(h1_ind):
            return None

        pullback_bars = h1_bars[impulse.end_idx + 1:]
        if len(pullback_bars) > cfg.pullback_max_bars:
            return None

        impulse_bars = h1_bars[impulse.start_idx:impulse.end_idx + 1]
        if cfg.strict_orderly_pullback and not self._is_orderly_strict(
            pullback_bars=pullback_bars,
            impulse_bars=impulse_bars,
            direction=direction,
        ):
            return None

        confluences: list[str] = []
        if in_ema_zone:
            confluences.append("h1_ema_zone")

        if abs(current_price - impulse.start_price) < 0.5 * atr:
            confluences.append("breakout_retest")

        rsi = h1_ind.rsi
        if rsi is not None and cfg.pullback_rsi_low <= rsi <= cfg.pullback_rsi_high:
            confluences.append("rsi_pullback")

        if d1_ind is not None:
            d1_atr = d1_ind.atr if d1_ind.atr > 0 else atr * 4
            near_d1_ema = (
                abs(current_price - d1_ind.ema_fast) < 0.5 * d1_atr
                or abs(current_price - d1_ind.ema_mid) < 0.5 * d1_atr
            )
            if near_d1_ema:
                confluences.append("d1_ema_support")

        if weekly_high is not None and weekly_low is not None:
            if (abs(current_price - weekly_high) < 0.5 * atr
                    or abs(current_price - weekly_low) < 0.5 * atr):
                confluences.append("weekly_level")

        if len(h1_bars) >= 10:
            recent_vol = sum(b.volume for b in h1_bars[-5:]) / 5
            earlier_vol = sum(b.volume for b in h1_bars[-10:-5]) / 5
            if earlier_vol > 0 and recent_vol < 0.8 * earlier_vol:
                confluences.append("volume_contraction")

        min_conf = (
            min_confluences_override
            if min_confluences_override is not None
            else cfg.min_confluences
        )
        if len(confluences) < min_conf:
            return None

        stop_level = self._estimate_stop(h1_bars, direction, atr)
        stop_distance = abs(current_price - stop_level)
        if stop_distance <= 0:
            return None

        if direction == Side.LONG:
            room_r = (impulse.end_price - current_price) / stop_distance
        else:
            room_r = (current_price - impulse.end_price) / stop_distance

        if cfg.weekly_room_filter_enabled:
            weekly_room_r = self._forward_weekly_room_r(
                current_price=current_price,
                stop_distance=stop_distance,
                direction=direction,
                weekly_high=weekly_high,
                weekly_low=weekly_low,
            )
            if weekly_room_r is not None and weekly_room_r < cfg.min_weekly_room_r:
                return None

        setup_score = self._score_confluences(confluences)
        b_quality_ok = True
        a_quality_ok = len(confluences) >= 3
        if cfg.use_weighted_confluence:
            b_quality_ok = setup_score >= cfg.min_setup_score_b
            a_quality_ok = setup_score >= cfg.min_setup_score_a

        h1_adx = h1_ind.adx
        if (regime.tier == "A"
                and room_r >= cfg.min_room_r_a
                and h1_adx > 20
                and a_quality_ok):
            grade = SetupGrade.A
        elif room_r >= cfg.min_room_r and b_quality_ok:
            grade = SetupGrade.B
        else:
            return None

        return TrendSetupResult(
            grade=grade,
            direction=direction,
            impulse_start=impulse.start_price,
            impulse_end=impulse.end_price,
            impulse_atr_move=impulse.atr_move,
            pullback_depth=pullback_depth,
            confluences=tuple(confluences),
            zone_price=current_price,
            room_r=room_r,
            stop_level=stop_level,
            setup_score=setup_score,
        )

    def _find_impulse(
        self,
        bars: list[Bar],
        direction: Side,
        atr: float,
    ) -> ImpulseLeg | None:
        cfg = self._cfg
        lookback = min(cfg.impulse_lookback, len(bars) - 1)
        if lookback < cfg.impulse_min_bars:
            return None

        window = bars[-lookback:]
        offset = len(bars) - len(window)

        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []

        for i in range(2, len(window) - 2):
            bar = window[i]
            if (bar.high > window[i - 1].high and bar.high > window[i - 2].high
                    and bar.high > window[i + 1].high and bar.high > window[i + 2].high):
                swing_highs.append((i + offset, bar.high))
            if (bar.low < window[i - 1].low and bar.low < window[i - 2].low
                    and bar.low < window[i + 1].low and bar.low < window[i + 2].low):
                swing_lows.append((i + offset, bar.low))

        if direction == Side.LONG:
            for sh_idx, sh_price in reversed(swing_highs):
                for sl_idx, sl_price in reversed(swing_lows):
                    if sl_idx < sh_idx:
                        move = (sh_price - sl_price) / atr
                        bars_count = sh_idx - sl_idx
                        if move >= cfg.impulse_min_atr_move and bars_count >= cfg.impulse_min_bars:
                            if not cfg.require_completed_impulse or sh_idx <= len(bars) - 3:
                                return ImpulseLeg(sl_idx, sh_idx, sl_price, sh_price, move)
                        break
        else:
            for sl_idx, sl_price in reversed(swing_lows):
                for sh_idx, sh_price in reversed(swing_highs):
                    if sh_idx < sl_idx:
                        move = (sh_price - sl_price) / atr
                        bars_count = sl_idx - sh_idx
                        if move >= cfg.impulse_min_atr_move and bars_count >= cfg.impulse_min_bars:
                            if not cfg.require_completed_impulse or sl_idx <= len(bars) - 3:
                                return ImpulseLeg(sh_idx, sl_idx, sh_price, sl_price, move)
                        break

        return None

    def _in_ema_zone(
        self,
        price: float,
        ema_fast: float,
        ema_mid: float,
        direction: Side,
    ) -> bool:
        upper = max(ema_fast, ema_mid)
        lower = min(ema_fast, ema_mid)

        if direction == Side.LONG:
            return lower <= price <= upper * 1.01
        return lower * 0.99 <= price <= upper

    def _is_orderly_basic(self, h1_ind: IndicatorSnapshot) -> bool:
        cfg = self._cfg
        rsi = h1_ind.rsi
        return rsi is None or (cfg.pullback_rsi_low <= rsi <= cfg.pullback_rsi_high)

    def _is_orderly_strict(
        self,
        pullback_bars: list[Bar],
        impulse_bars: list[Bar],
        direction: Side,
    ) -> bool:
        cfg = self._cfg
        if len(pullback_bars) < cfg.orderly_min_countertrend_bars:
            return False
        if not impulse_bars:
            return False

        countertrend_bars = 0
        for bar in pullback_bars:
            if direction == Side.LONG and bar.close <= bar.open:
                countertrend_bars += 1
            elif direction == Side.SHORT and bar.close >= bar.open:
                countertrend_bars += 1

        if countertrend_bars < cfg.orderly_min_countertrend_bars:
            return False

        impulse_body = sum(abs(bar.close - bar.open) for bar in impulse_bars) / len(impulse_bars)
        pullback_body = sum(abs(bar.close - bar.open) for bar in pullback_bars) / len(pullback_bars)
        if impulse_body > 0 and pullback_body > impulse_body * cfg.orderly_max_body_frac:
            return False

        impulse_volume = sum(bar.volume for bar in impulse_bars) / len(impulse_bars)
        pullback_volume = sum(bar.volume for bar in pullback_bars) / len(pullback_bars)
        if impulse_volume > 0 and pullback_volume > impulse_volume * cfg.orderly_max_countertrend_volume_ratio:
            return False

        return True

    def _score_confluences(self, confluences: list[str]) -> float:
        return sum(self._CONFLUENCE_WEIGHTS.get(name, 0.5) for name in confluences)

    def _forward_weekly_room_r(
        self,
        *,
        current_price: float,
        stop_distance: float,
        direction: Side,
        weekly_high: float | None,
        weekly_low: float | None,
    ) -> float | None:
        if stop_distance <= 0:
            return None

        if direction == Side.LONG and weekly_high is not None and weekly_high > current_price:
            return (weekly_high - current_price) / stop_distance
        if direction == Side.SHORT and weekly_low is not None and weekly_low < current_price:
            return (current_price - weekly_low) / stop_distance
        return None

    def _estimate_stop(
        self,
        bars: list[Bar],
        direction: Side,
        atr: float,
    ) -> float:
        lookback = min(10, len(bars))
        recent = bars[-lookback:]

        if direction == Side.LONG:
            swing_low = min(bar.low for bar in recent)
            return swing_low - 0.3 * atr

        swing_high = max(bar.high for bar in recent)
        return swing_high + 0.3 * atr
