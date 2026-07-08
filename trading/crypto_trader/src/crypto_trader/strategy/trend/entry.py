"""Trend entry order generation."""

from __future__ import annotations

from crypto_trader.core.models import Bar, Order, OrderType, Side

from .config import TrendEntryParams
from .confirmation import TriggerResult
from .setup import TrendSetupResult
from .sizing import SizingResult


class EntryGenerator:
    """Generate entry orders based on setup and trigger."""

    def __init__(self, cfg: TrendEntryParams) -> None:
        self._cfg = cfg

    def generate(
        self,
        bar: Bar,
        direction: Side,
        qty: float,
        sizing_result: SizingResult,
        setup: TrendSetupResult,
        trigger: TriggerResult | None,
        symbol: str,
        order_id: str,
        *,
        is_reentry: bool = False,
    ) -> Order | None:
        cfg = self._cfg

        mode = (cfg.mode or "legacy").lower()
        use_market = False
        use_break = False

        if mode == "legacy":
            use_market = cfg.entry_on_close
            use_break = cfg.entry_on_break
        elif mode == "close":
            use_market = True
        elif mode == "break":
            use_break = True
        elif mode == "hybrid_grade":
            use_market = setup.grade.value == "A"
            use_break = not use_market
        elif mode == "confirm_preferred":
            use_break = trigger is not None
            use_market = trigger is None
        elif mode == "reentry_break":
            use_market = not is_reentry
            use_break = is_reentry
        elif mode == "reentry_confirm_preferred":
            use_market = not (is_reentry and trigger is not None)
            use_break = is_reentry and trigger is not None
        else:
            use_market = cfg.entry_on_close
            use_break = cfg.entry_on_break

        metadata = {
            "setup_grade": setup.grade.value,
            "confluences": list(setup.confluences),
            "stop_level": setup.stop_level,
            "leverage": sizing_result.leverage,
            "liquidation_price": sizing_result.liquidation_price,
            "risk_pct": sizing_result.risk_pct_actual,
            "confirmation": trigger.pattern if trigger else "none",
            "room_r": setup.room_r,
            "setup_score": getattr(setup, "setup_score", 0.0),
            "is_reentry": is_reentry,
        }

        if use_market:
            return Order(
                order_id=order_id,
                symbol=symbol,
                side=direction,
                order_type=OrderType.MARKET,
                qty=qty,
                tag="entry",
                metadata={**metadata, "entry_method": "aggressive"},
            )
        if use_break and trigger is not None:
            stop_price = trigger.trigger_price
            return Order(
                order_id=order_id,
                symbol=symbol,
                side=direction,
                order_type=OrderType.STOP,
                qty=qty,
                stop_price=stop_price,
                tag="entry",
                ttl_bars=cfg.max_bars_after_confirmation,
                metadata={**metadata, "entry_method": "conservative"},
            )

        return None
