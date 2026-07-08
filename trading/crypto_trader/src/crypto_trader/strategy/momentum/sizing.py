"""Position sizing, leverage calculation, and risk checks."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_trader.core.models import Position, SetupGrade, Side
from crypto_trader.strategy.momentum.config import RiskParams


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
    def __init__(self, params: RiskParams) -> None:
        self._p = params

    def compute(
        self,
        equity: float,
        entry_price: float,
        stop_distance: float,
        setup_grade: SetupGrade,
        symbol: str,
        open_positions: list[Position],
        direction: Side,
    ) -> tuple[SizingResult | None, str]:
        if setup_grade == SetupGrade.C:
            return None, "c_grade"

        # Max concurrent positions
        if len(open_positions) >= self._p.max_concurrent_positions:
            return None, "max_concurrent"

        # 1. Risk amount
        risk_pct = self._p.risk_pct_a if setup_grade == SetupGrade.A else self._p.risk_pct_b
        risk_amount = equity * risk_pct

        # 2. Size
        if stop_distance <= 0:
            return None, "zero_stop_distance"
        qty = risk_amount / stop_distance

        # 3. Notional
        notional = qty * entry_price

        # 4. Leverage
        leverage = notional / equity if equity > 0 else 0

        # 5. Leverage bounds
        was_reduced = False
        reduction_reason = None
        max_lev = self._p.max_leverage_major if symbol in self._p.major_symbols else self._p.max_leverage_alt
        if leverage > max_lev:
            leverage = max_lev
            notional = equity * leverage
            qty = notional / entry_price
            was_reduced = True
            reduction_reason = f"leverage clamped to {max_lev}x"
        if leverage < self._p.min_leverage:
            # Position too small for min leverage — that's fine, just use natural leverage
            pass

        risk_pct_actual = (qty * stop_distance) / equity if equity > 0 else 0

        # 6. Liquidation price check
        # Liq price ~= entry ± entry/leverage (simplified)
        if direction == Side.LONG:
            liq_price = entry_price * (1 - 1.0 / leverage) if leverage > 1 else 0.0
            stop_price = entry_price - stop_distance
        else:
            liq_price = entry_price * (1 + 1.0 / leverage) if leverage > 1 else entry_price * 10
            stop_price = entry_price + stop_distance

        # Ensure liquidation is far enough from stop
        atr_approx = stop_distance / 0.3  # rough ATR estimate from stop dist
        liq_to_stop = abs(liq_price - stop_price)
        min_liq_buffer = self._p.min_liquidation_buffer_atr * atr_approx
        if liq_to_stop < min_liq_buffer and leverage > self._p.min_leverage:
            # Reduce leverage to push liquidation further
            old_leverage = leverage
            leverage = max(self._p.min_leverage, leverage * 0.8)
            notional = equity * leverage
            qty = notional / entry_price
            was_reduced = True
            reduction_reason = f"liq buffer: leverage {old_leverage:.1f}x → {leverage:.1f}x"
            # Recalculate liq
            if direction == Side.LONG:
                liq_price = entry_price * (1 - 1.0 / leverage) if leverage > 1 else 0.0
            else:
                liq_price = entry_price * (1 + 1.0 / leverage) if leverage > 1 else entry_price * 10

        # 7. Correlation check
        correlated_risk = sum(
            self._position_risk(p, equity) for p in open_positions
            if self._is_correlated(p.symbol, symbol)
        )
        if correlated_risk + risk_pct_actual > self._p.max_correlated_risk:
            return None, "correlated_risk_cap"

        # Gross risk check
        gross_risk = sum(self._position_risk(p, equity) for p in open_positions)
        if gross_risk + risk_pct_actual > self._p.max_gross_risk:
            return None, "gross_risk_cap"

        return SizingResult(
            qty=qty,
            leverage=leverage,
            liquidation_price=liq_price,
            risk_pct_actual=risk_pct_actual,
            notional=notional,
            was_reduced=was_reduced,
            reduction_reason=reduction_reason,
        ), ""

    def _position_risk(self, pos: Position, equity: float) -> float:
        if equity <= 0:
            return 0.0
        # Approximate per-position risk as the average risk allocation
        # (exact stop distance isn't tracked on Position, so use config midpoint)
        return (self._p.risk_pct_a + self._p.risk_pct_b) / 2

    @staticmethod
    def _is_correlated(sym_a: str, sym_b: str) -> bool:
        # All crypto assets considered correlated for now
        return True
