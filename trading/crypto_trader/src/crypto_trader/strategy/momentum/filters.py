"""Environment, session, and funding filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from crypto_trader.strategy.momentum.config import FilterParams, SessionParams
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot


@dataclass(frozen=True)
class FilterResult:
    allowed: bool
    confidence_modifier: float  # 0.0-1.0
    reasons: list[str]
    require_a_grade: bool


class EnvironmentFilter:
    def __init__(self, filter_params: FilterParams, session_params: SessionParams) -> None:
        self._fp = filter_params
        self._sp = session_params

    def check(
        self,
        indicators: IndicatorSnapshot,
        funding_rate: float,
        current_time: datetime,
    ) -> FilterResult:
        reasons: list[str] = []
        confidence = 1.0
        require_a = False

        # Hard no-trade conditions
        if indicators.atr_avg > 0:
            atr_ratio = indicators.atr / indicators.atr_avg
            if atr_ratio > self._fp.atr_expansion_mult:
                return FilterResult(
                    allowed=False, confidence_modifier=0.0,
                    reasons=[f"ATR expansion {atr_ratio:.1f}x — panic"], require_a_grade=False,
                )
            if atr_ratio < self._fp.atr_compression_mult:
                return FilterResult(
                    allowed=False, confidence_modifier=0.0,
                    reasons=[f"ATR compression {atr_ratio:.2f}x — dead market"], require_a_grade=False,
                )

        if indicators.adx < self._fp.adx_chop_threshold:
            return FilterResult(
                allowed=False, confidence_modifier=0.0,
                reasons=[f"ADX {indicators.adx:.1f} < {self._fp.adx_chop_threshold} — chop"],
                require_a_grade=False,
            )

        if abs(funding_rate) > self._fp.funding_extreme_threshold:
            return FilterResult(
                allowed=False, confidence_modifier=0.0,
                reasons=[f"Funding extreme {funding_rate:.4f}"],
                require_a_grade=False,
            )

        # Session windows
        hour = current_time.hour
        in_preferred = self._in_preferred_session(hour)
        if not in_preferred:
            confidence *= 0.7
            reasons.append(f"Outside preferred session (hour={hour})")
            if self._sp.reduced_window_require_a:
                require_a = True

        # Funding adjustment
        if abs(funding_rate) > self._fp.funding_moderate_threshold:
            confidence *= 0.8
            reasons.append(f"Moderate funding {funding_rate:.4f}")

        return FilterResult(
            allowed=True,
            confidence_modifier=confidence,
            reasons=reasons,
            require_a_grade=require_a,
        )

    def _in_preferred_session(self, hour: int) -> bool:
        # London 08-16 UTC or NY 13-21 UTC
        london = self._sp.london_start <= hour < self._sp.london_end
        ny = self._sp.ny_start <= hour < self._sp.ny_end
        return london or ny
