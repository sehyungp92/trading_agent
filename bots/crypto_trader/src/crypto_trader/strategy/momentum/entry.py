"""Entry signal generation — produces a single entry Order."""

from __future__ import annotations

from crypto_trader.core.models import Order, OrderType, Side
from crypto_trader.strategy.momentum.config import EntryParams
from crypto_trader.strategy.momentum.confirmation import ConfirmationResult
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.momentum.setup import SetupResult
from crypto_trader.strategy.momentum.sizing import SizingResult


class EntrySignal:
    def __init__(self, params: EntryParams) -> None:
        self._p = params

    def _resolve_mode(self, confirmation: ConfirmationResult) -> str:
        mode = (self._p.mode or "legacy").lower()
        if mode == "close":
            return "close"
        if mode == "break":
            return "break"
        if mode == "confirmation_specific":
            return "break" if confirmation.pattern_type == "inside_bar_break" else "close"
        if self._p.entry_on_close:
            return "close"
        if self._p.entry_on_break:
            return "break"
        return "close"

    def resolve_mode(self, confirmation: ConfirmationResult) -> str:
        return self._resolve_mode(confirmation)

    def estimate_entry_price(
        self,
        confirmation: ConfirmationResult,
        close_price: float,
    ) -> float:
        entry_method = self._resolve_mode(confirmation)
        if entry_method == "break":
            return confirmation.trigger_price
        return close_price

    def generate(
        self,
        setup: SetupResult,
        confirmation: ConfirmationResult,
        indicators: IndicatorSnapshot,
        sizing: SizingResult,
        direction: Side,
        symbol: str,
        bars_since_confirmation: int = 0,
    ) -> Order | None:
        # Chase rule
        if bars_since_confirmation > self._p.max_bars_after_confirmation:
            return None

        entry_method = self.resolve_mode(confirmation)

        if entry_method == "close":
            order_type = OrderType.MARKET
            limit_price = None
            stop_price = None
            ttl_bars = None
        else:
            order_type = OrderType.STOP
            stop_price = confirmation.trigger_price
            limit_price = None
            ttl_bars = self._p.max_bars_after_confirmation

        # TP levels — R-multiples measured from entry to stop
        entry_est = confirmation.trigger_price
        stop_dist = abs(entry_est - setup.stop_level)
        if stop_dist <= 0:
            stop_dist = indicators.atr  # fallback
        if direction == Side.LONG:
            tp1 = entry_est + stop_dist * 1.0
            tp2 = entry_est + stop_dist * 2.0
        else:
            tp1 = entry_est - stop_dist * 1.0
            tp2 = entry_est - stop_dist * 2.0

        return Order(
            order_id="",
            symbol=symbol,
            side=direction,
            order_type=order_type,
            qty=sizing.qty,
            limit_price=limit_price,
            stop_price=stop_price,
            ttl_bars=ttl_bars,
            tag="entry",
            metadata={
                "setup_grade": setup.grade.value,
                "entry_method": entry_method,
                "confluences": setup.confluences,
                "confirmation_type": confirmation.pattern_type,
                "stop_level": setup.stop_level,
                "tp1_level": tp1,
                "tp2_level": tp2,
                "leverage": sizing.leverage,
                "liquidation_price": sizing.liquidation_price,
                "risk_pct": sizing.risk_pct_actual,
            },
        )
