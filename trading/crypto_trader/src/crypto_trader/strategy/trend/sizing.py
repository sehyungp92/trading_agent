"""Trend position sizing with leverage constraints."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Position, SetupGrade, Side

from .config import TrendLimitParams, TrendRiskParams


@dataclass(frozen=True)
class SizingResult:
    qty: float
    leverage: float
    liquidation_price: float
    risk_pct_actual: float
    notional: float
    was_reduced: bool
    reduction_reason: str | None


class PositionSizer:
    """Compute position size with risk and leverage constraints."""

    def __init__(self, cfg: TrendRiskParams, limits: TrendLimitParams) -> None:
        self._cfg = cfg
        self._limits = limits

    def compute(
        self,
        equity: float,
        entry_price: float,
        stop_distance: float,
        grade: SetupGrade,
        symbol: str,
        open_positions: list[Position],
        direction: Side,
        risk_scale: float = 1.0,
    ) -> tuple[SizingResult | None, str]:
        if len(open_positions) >= self._limits.max_concurrent_positions:
            return None, "max_concurrent"

        if equity <= 0 or entry_price <= 0 or stop_distance <= 0:
            return None, "invalid_inputs"

        cfg = self._cfg

        # Select risk percentage
        if grade == SetupGrade.A:
            risk_pct = cfg.risk_pct_a
        else:
            risk_pct = cfg.risk_pct_b
        risk_pct *= max(risk_scale, 0.0)
        risk_pct = min(risk_pct, cfg.max_risk_pct)
        if risk_pct <= 0:
            return None, "zero_risk_pct"

        # Compute raw quantity
        risk_amount = equity * risk_pct
        qty = risk_amount / stop_distance

        # Compute leverage
        notional = qty * entry_price
        leverage = notional / equity

        # Clamp leverage
        is_major = symbol in cfg.major_symbols
        max_lev = cfg.max_leverage_major if is_major else cfg.max_leverage_alt
        was_reduced = False
        reduction_reason = None

        if leverage > max_lev:
            leverage = max_lev
            qty = equity * leverage / entry_price
            notional = qty * entry_price
            was_reduced = True
            reduction_reason = f"leverage_clamped_{max_lev}"

        if leverage < cfg.min_leverage:
            return None, "below_min_leverage"

        # Compute liquidation price
        if direction == Side.LONG:
            liquidation_price = entry_price * (1 - 1 / leverage)
        else:
            liquidation_price = entry_price * (1 + 1 / leverage)

        # Recalculate actual risk
        risk_pct_actual = (qty * stop_distance) / equity if equity > 0 else 0

        return SizingResult(
            qty=qty,
            leverage=leverage,
            liquidation_price=liquidation_price,
            risk_pct_actual=risk_pct_actual,
            notional=notional,
            was_reduced=was_reduced,
            reduction_reason=reduction_reason,
        ), ""
