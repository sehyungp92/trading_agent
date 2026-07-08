"""Breakout position sizing with three-tier risk and leverage constraints."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Position, SetupGrade, Side

from .config import BreakoutLimitParams, BreakoutRiskParams


@dataclass(frozen=True)
class SizingResult:
    """Computed position size with risk metadata."""

    qty: float
    leverage: float
    liquidation_price: float
    risk_pct_actual: float
    notional: float
    was_reduced: bool
    reduction_reason: str | None


class PositionSizer:
    """Compute position size with A+/A/B risk tiers and leverage clamping."""

    def __init__(
        self, risk_cfg: BreakoutRiskParams, limit_cfg: BreakoutLimitParams,
    ) -> None:
        self._r = risk_cfg
        self._l = limit_cfg

    def compute(
        self,
        equity: float,
        entry_price: float,
        stop_distance: float,
        grade: SetupGrade,
        is_a_plus: bool,
        symbol: str,
        open_positions: list[Position],
        direction: Side,
        risk_scale: float = 1.0,
    ) -> tuple[SizingResult | None, str]:
        """Return a sizing result and reason (empty on success)."""
        # Reject if max concurrent reached
        if len(open_positions) >= self._l.max_concurrent_positions:
            return None, "max_concurrent"

        if equity <= 0 or entry_price <= 0 or stop_distance <= 0:
            return None, "invalid_inputs"

        # Select risk tier
        if is_a_plus:
            risk_pct = self._r.risk_pct_a_plus
        elif grade == SetupGrade.A:
            risk_pct = self._r.risk_pct_a
        else:
            risk_pct = self._r.risk_pct_b

        if risk_scale <= 0:
            return None, "zero_risk_scale"
        risk_pct *= risk_scale
        risk_pct = min(risk_pct, self._r.max_risk_pct)

        # Compute raw quantity
        risk_dollars = equity * risk_pct
        qty = risk_dollars / stop_distance

        # Compute leverage
        notional = qty * entry_price
        leverage = notional / equity

        # Clamp leverage
        is_major = symbol in self._r.major_symbols
        max_lev = self._r.max_leverage_major if is_major else self._r.max_leverage_alt
        was_reduced = False
        reduction_reason: str | None = None

        if leverage > max_lev:
            leverage = max_lev
            notional = equity * leverage
            qty = notional / entry_price
            was_reduced = True
            reduction_reason = f"leverage_clamped_{max_lev}"

        if leverage < self._r.min_leverage:
            return None, "below_min_leverage"

        # Liquidation price
        if direction == Side.LONG:
            liq = entry_price * (1 - 1 / leverage) if leverage > 0 else 0.0
        else:
            liq = entry_price * (1 + 1 / leverage) if leverage > 0 else float("inf")

        # Recalculate actual risk
        risk_pct_actual = (qty * stop_distance) / equity if equity > 0 else 0.0

        return SizingResult(
            qty=qty,
            leverage=leverage,
            liquidation_price=liq,
            risk_pct_actual=risk_pct_actual,
            notional=notional,
            was_reduced=was_reduced,
            reduction_reason=reduction_reason,
        ), ""
