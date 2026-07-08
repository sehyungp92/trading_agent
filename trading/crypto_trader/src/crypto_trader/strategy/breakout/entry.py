"""Breakout entry order generation."""

from __future__ import annotations

from crypto_trader.core.models import Bar, Order, OrderType, Side

from .config import BreakoutEntryParams
from .confirmation import BreakoutConfirmation
from .setup import BreakoutSetupResult
from .sizing import SizingResult


class EntryGenerator:
    """Generate entry orders for breakout setups."""

    def __init__(self, cfg: BreakoutEntryParams) -> None:
        self._p = cfg

    def generate(
        self,
        bar: Bar,
        direction: Side,
        qty: float,
        sizing_result: SizingResult,
        setup: BreakoutSetupResult,
        confirmation: BreakoutConfirmation,
        symbol: str,
        order_id: str,
    ) -> Order | None:
        """Build an entry order based on the confirmation model.

        Model 1 + model1_entry_on_close  -> MARKET at bar.close
        Model 2 + model2_entry_on_close  -> MARKET at bar.close
        Model 2 + model2_entry_on_break  -> STOP at breakout level
        """
        if confirmation.model == "model1_close":
            if not self._p.model1_entry_on_close:
                return None
            order_type = OrderType.MARKET
            stop_price = None

        elif confirmation.model == "model2_retest":
            if self._p.model2_entry_on_close:
                order_type = OrderType.MARKET
                stop_price = None
            elif self._p.model2_entry_on_break:
                order_type = OrderType.STOP
                if direction == Side.LONG:
                    stop_price = setup.balance_zone.upper
                else:
                    stop_price = setup.balance_zone.lower
            else:
                return None
        else:
            return None

        metadata = {
            "setup_grade": setup.grade.value,
            "is_a_plus": setup.is_a_plus,
            "entry_method": confirmation.model,
            "signal_variant": setup.signal_variant,
            "confluences": list(setup.confluences),
            "stop_level": None,  # Filled by strategy after stop computation
            "leverage": sizing_result.leverage,
            "liquidation_price": sizing_result.liquidation_price,
            "risk_pct": sizing_result.risk_pct_actual,
            "room_r": setup.room_r,
        }

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=direction,
            order_type=order_type,
            qty=qty,
            stop_price=stop_price,
            tag="entry",
            ttl_bars=self._p.max_bars_after_signal if order_type == OrderType.STOP else None,
            metadata=metadata,
        )
