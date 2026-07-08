from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .config import StrategySettings
from .core.state import IARICEntryRequest
from .models import MarketSnapshot, PBSymbolState, PortfolioState, WatchlistItem
from .risk import adjust_qty_for_portfolio_constraints, compute_order_quantity, weekday_sizing_multiplier


@dataclass(frozen=True, slots=True)
class IARICEntryRequestBuild:
    entry_request: IARICEntryRequest | None
    reason: str
    entry_price: float
    sizing_mult: float
    gap_up_mult: float


def build_ready_entry_request(
    *,
    symbol: str,
    state: PBSymbolState,
    item: WatchlistItem,
    market: MarketSnapshot,
    portfolio: PortfolioState,
    symbol_to_sector: Mapping[str, str],
    settings: StrategySettings,
    now: datetime,
    route: str,
) -> IARICEntryRequestBuild:
    if market.last_price is None or state.stop_level <= 0:
        return IARICEntryRequestBuild(None, "missing_market_or_stop", 0.0, float(state.sizing_mult), 1.0)

    entry_price = float(market.ask if market.ask > 0 else market.last_price + item.tick_size)
    sizing_mult = float(state.sizing_mult)
    gap_up_mult = max(0.0, float(settings.pb_gap_up_size_mult)) if item.entry_gap_pct > 0 else 1.0
    risk_unit = sizing_mult * weekday_sizing_multiplier(now, settings) * gap_up_mult
    qty = compute_order_quantity(
        account_equity=portfolio.account_equity,
        base_risk_fraction=portfolio.base_risk_fraction,
        final_risk_unit=risk_unit,
        entry_price=entry_price,
        stop_level=state.stop_level,
    )
    qty, reason = adjust_qty_for_portfolio_constraints(
        portfolio=portfolio,
        item=item,
        intended_qty=qty,
        entry_price=entry_price,
        stop_level=state.stop_level,
        symbol_to_sector=dict(symbol_to_sector),
        settings=settings,
    )
    if qty <= 0:
        return IARICEntryRequestBuild(None, reason, entry_price, sizing_mult, gap_up_mult)

    return IARICEntryRequestBuild(
        IARICEntryRequest(
            client_order_id=f"{symbol}-entry-{int(now.timestamp())}",
            symbol=symbol,
            route=route,
            qty=qty,
            limit_price=entry_price,
            stop_price=state.stop_level,
            metadata={
                "daily_signal_score": state.daily_signal_score,
                "route": route,
                "sizing_mult": round(sizing_mult, 6),
                "entry_gap_pct": round(item.entry_gap_pct, 6),
                "gap_up_size_mult": round(gap_up_mult, 6),
            },
        ),
        "ok",
        entry_price,
        sizing_mult,
        gap_up_mult,
    )
