"""Breakout detection + LVN runway + 8-confluence grading."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Bar, SetupGrade, Side
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

from .balance import BalanceZone
from .config import BreakoutSetupParams, BreakoutSymbolFilterParams
from .context import ContextBias
from .profile import VolumeProfiler, VolumeProfileResult


@dataclass(frozen=True)
class BreakoutSetupResult:
    """A validated breakout setup ready for confirmation / entry."""

    grade: SetupGrade  # A or B (A+ stored via is_a_plus)
    is_a_plus: bool
    direction: Side
    balance_zone: BalanceZone
    breakout_price: float  # bar.close
    lvn_runway_atr: float
    confluences: tuple[str, ...]
    room_r: float  # projected reward / stop distance
    volume_mult: float  # breakout volume / avg volume
    body_ratio: float  # |close-open| / (high-low)
    signal_variant: str = "core"
    risk_scale: float = 1.0


@dataclass(frozen=True)
class BlockedRelaxedBodySignal:
    """A relaxed-body setup that qualified except for the direction rule."""

    setup: BreakoutSetupResult
    blocked_rule: str


class BreakoutDetector:
    """Detect breakouts from balance zones and grade them."""

    def __init__(
        self,
        setup_cfg: BreakoutSetupParams,
        symbol_filter_cfg: BreakoutSymbolFilterParams | None = None,
    ) -> None:
        self._p = setup_cfg
        self._symbol_filter = symbol_filter_cfg or BreakoutSymbolFilterParams()
        self._blocked_relaxed_body_signals: list[BlockedRelaxedBodySignal] = []

    def detect(
        self,
        bar: Bar,
        zones: list[BalanceZone],
        profile: VolumeProfileResult | None,
        profiler: VolumeProfiler,
        context: ContextBias,
        m30_ind: IndicatorSnapshot | None,
        atr: float,
        sym: str | None = None,
    ) -> BreakoutSetupResult | None:
        """Scan active zones for a valid breakout on *bar*.

        Returns the best setup (most confluences) or ``None``.
        """
        if not zones or atr <= 0:
            return None

        self._blocked_relaxed_body_signals.clear()

        best: BreakoutSetupResult | None = None
        best_conf_count = -1

        for zone in zones:
            result = self._evaluate_zone(
                bar=bar,
                zone=zone,
                profile=profile,
                profiler=profiler,
                context=context,
                m30_ind=m30_ind,
                atr=atr,
                sym=sym or bar.symbol,
            )
            if result is not None and len(result.confluences) > best_conf_count:
                best = result
                best_conf_count = len(result.confluences)

        return best

    def _evaluate_zone(
        self,
        bar: Bar,
        zone: BalanceZone,
        profile: VolumeProfileResult | None,
        profiler: VolumeProfiler,
        context: ContextBias,
        m30_ind: IndicatorSnapshot | None,
        atr: float,
        sym: str,
    ) -> BreakoutSetupResult | None:
        p = self._p

        if bar.close > zone.upper:
            direction = Side.LONG
            breakout_dist = bar.close - zone.upper
        elif bar.close < zone.lower:
            direction = Side.SHORT
            breakout_dist = zone.lower - bar.close
        else:
            return None

        dist_atr = breakout_dist / atr
        if dist_atr < p.min_breakout_atr or dist_atr > p.max_breakout_atr:
            return None

        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return None
        body_ratio = abs(bar.close - bar.open) / bar_range

        volume_mult = 0.0
        if m30_ind is not None and m30_ind.volume_ma is not None and m30_ind.volume_ma > 0:
            volume_mult = bar.volume / m30_ind.volume_ma

        if p.require_volume_surge and volume_mult < p.volume_surge_mult:
            return None

        is_relaxed_body = False
        blocked_relaxed_rule: str | None = None
        if body_ratio < p.body_ratio_min:
            if not p.relaxed_body_enabled or body_ratio < p.relaxed_body_min:
                return None
            blocked_relaxed_rule = self._relaxed_body_block_reason(sym, direction)
            if blocked_relaxed_rule is None and not self._relaxed_body_allowed(sym, direction, body_ratio):
                return None
            is_relaxed_body = True

        lvn_runway_atr = 0.0
        if profile is not None:
            lvn_runway_atr = profiler.find_lvn_runway(
                profile,
                bar.close,
                direction,
                atr,
            )

        confluences: list[str] = []
        if context.direction == direction:
            confluences.append("h4_alignment")

        if volume_mult >= p.volume_surge_mult:
            confluences.append("volume_surge")

        if lvn_runway_atr >= p.min_lvn_runway_atr:
            confluences.append("lvn_runway")

        if zone.bars_in_zone >= 16:
            confluences.append("balance_duration")

        if zone.volume_contracting:
            confluences.append("volume_contraction")

        if m30_ind is not None:
            if direction == Side.LONG and bar.close > m30_ind.ema_fast:
                confluences.append("ema_support")
            elif direction == Side.SHORT and bar.close < m30_ind.ema_fast:
                confluences.append("ema_support")

        if profile is not None and zone.lower <= profile.poc <= zone.upper:
            confluences.append("poc_alignment")

        if profile is not None:
            hvn_in_zone = sum(
                1 for level in profile.hvn_levels if zone.lower <= level <= zone.upper
            )
            if hvn_in_zone >= 2:
                confluences.append("multi_hvn")

        conf_count = len(confluences)
        is_countertrend = context.direction is not None and context.direction != direction

        signal_variant = "core"
        risk_scale = 1.0
        if is_relaxed_body:
            if conf_count < p.relaxed_body_min_confluences:
                return None
            if p.relaxed_body_require_volume_surge and volume_mult < p.volume_surge_mult:
                return None
            grade = SetupGrade.B
            is_a_plus = False
            signal_variant = "relaxed_body"
            risk_scale = p.relaxed_body_risk_scale
        else:
            if conf_count >= p.min_confluences_a_plus:
                grade = SetupGrade.A
                is_a_plus = True
            elif conf_count >= p.min_confluences_a:
                grade = SetupGrade.A
                is_a_plus = False
            elif conf_count >= p.min_confluences_b:
                grade = SetupGrade.B
                is_a_plus = False
            else:
                return None

        if is_countertrend:
            grade = SetupGrade.B
            is_a_plus = False

        zone_width = zone.upper - zone.lower
        buffer = atr * 0.3
        estimated_stop = zone_width + buffer
        if estimated_stop <= 0:
            return None
        room_r = lvn_runway_atr * atr / estimated_stop

        if is_relaxed_body:
            min_room = p.relaxed_body_min_room_r
        else:
            min_room = p.min_room_r_a if grade == SetupGrade.A else p.min_room_r_b
        if room_r < min_room:
            return None

        setup = BreakoutSetupResult(
            grade=grade,
            is_a_plus=is_a_plus,
            direction=direction,
            balance_zone=zone,
            breakout_price=bar.close,
            lvn_runway_atr=lvn_runway_atr,
            confluences=tuple(confluences),
            room_r=room_r,
            volume_mult=volume_mult,
            body_ratio=body_ratio,
            signal_variant=signal_variant,
            risk_scale=risk_scale,
        )

        if blocked_relaxed_rule is not None:
            self._blocked_relaxed_body_signals.append(
                BlockedRelaxedBodySignal(
                    setup=setup,
                    blocked_rule=blocked_relaxed_rule,
                )
            )
            return None

        return setup

    def _relaxed_body_allowed(
        self,
        sym: str,
        direction: Side,
        body_ratio: float,
    ) -> bool:
        p = self._p
        if not p.relaxed_body_enabled:
            return False
        if body_ratio < p.relaxed_body_min:
            return False

        rule = getattr(self._symbol_filter, f"{sym.lower()}_relaxed_body_direction", "disabled")
        if rule == "disabled":
            return False
        if rule == "both":
            return True
        if rule == "long_only":
            return direction == Side.LONG
        if rule == "short_only":
            return direction == Side.SHORT
        return False

    def _relaxed_body_block_reason(self, sym: str, direction: Side) -> str | None:
        """Return the direction rule that blocked an otherwise valid relaxed-body setup."""
        rule = getattr(self._symbol_filter, f"{sym.lower()}_relaxed_body_direction", "disabled")
        if rule == "disabled":
            return f"{sym.lower()}_relaxed_body_direction=disabled"
        if rule == "long_only" and direction == Side.SHORT:
            return f"{sym.lower()}_relaxed_body_direction=long_only"
        if rule == "short_only" and direction == Side.LONG:
            return f"{sym.lower()}_relaxed_body_direction=short_only"
        return None

    def consume_blocked_relaxed_body_signals(self) -> list[BlockedRelaxedBodySignal]:
        """Return and clear the blocked relaxed-body setups seen on the last detect call."""
        blocked = list(self._blocked_relaxed_body_signals)
        self._blocked_relaxed_body_signals.clear()
        return blocked
