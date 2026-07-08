"""Risk helpers for IARIC."""

from __future__ import annotations

from datetime import datetime

from .config import ET, StrategySettings
from .models import PortfolioState, WatchlistItem


def timing_gate_allows_entry(now: datetime, settings: StrategySettings) -> bool:
    et = now.astimezone(ET).time()
    if et < settings.open_block_end:
        return False
    if et >= settings.entry_end:
        return False
    if settings.close_block_start <= et <= settings.forced_flatten:
        return False
    return True


def timing_multiplier(now: datetime, settings: StrategySettings) -> float:
    et = now.astimezone(ET).time()
    for start, end, multiplier in settings.timing_sizing:
        if start <= et < end:
            return multiplier
    return 0.0


def weekday_sizing_multiplier(now: datetime, settings: StrategySettings) -> float:
    """Day-of-week risk budget multiplier (research parity)."""
    wd = now.astimezone(ET).weekday()  # Mon=0 ... Fri=4
    if wd == 1:
        return float(settings.pb_tuesday_mult)
    if wd == 2:
        return float(settings.pb_wednesday_mult)
    if wd == 3:
        return float(settings.pb_thursday_mult)
    if wd == 4:
        return float(settings.pb_friday_mult)
    return 1.0


def compute_final_risk_unit(item: WatchlistItem) -> float:
    """Compute sizing risk unit.

    For pullback V2 (sizing_mult > 0): uses pullback tier-based sizing_mult.
    For legacy T1: uses conviction_multiplier.
    """
    if item.sizing_mult > 0 and item.daily_signal_score > 0:
        return item.sizing_mult * item.regime_risk_multiplier
    return item.conviction_multiplier * item.regime_risk_multiplier


def compute_order_quantity(
    account_equity: float,
    base_risk_fraction: float,
    final_risk_unit: float,
    entry_price: float,
    stop_level: float,
) -> int:
    risk_dollars = account_equity * base_risk_fraction * final_risk_unit
    per_share_risk = max(entry_price - stop_level, 0.01)
    shares = int(risk_dollars // per_share_risk)
    return max(shares, 0)


def max_positions_for_regime(tier: str, settings: StrategySettings) -> int:
    if tier == "A":
        return settings.max_positions_tier_a
    if tier == "B":
        return settings.max_positions_tier_b
    return 0


def adjust_qty_for_portfolio_constraints(
    portfolio: PortfolioState,
    item: WatchlistItem,
    intended_qty: int,
    entry_price: float,
    stop_level: float,
    symbol_to_sector: dict[str, str],
    settings: StrategySettings,
) -> tuple[int, str]:
    if intended_qty <= 0:
        return 0, "qty_zero"
    if portfolio.regime_allows_no_new_entries:
        return 0, "regime_block"
    if len(portfolio.open_positions) >= max_positions_for_regime(item.regime_tier, settings):
        return 0, "max_positions"
    if portfolio.sector_position_count(symbol_to_sector, item.sector) >= settings.max_positions_per_sector:
        return 0, "sector_position_cap"

    intended_risk = intended_qty * max(entry_price - stop_level, 0.01)
    current_total_risk = portfolio.open_risk_dollars() + sum(portfolio.pending_entry_risk.values())
    regime_risk_cap = portfolio.account_equity * portfolio.base_risk_fraction * max_positions_for_regime(item.regime_tier, settings)
    if current_total_risk + intended_risk > regime_risk_cap:
        remaining_risk = max(0.0, regime_risk_cap - current_total_risk)
        adjusted_qty = int(remaining_risk // max(entry_price - stop_level, 0.01))
        if adjusted_qty < int(intended_qty * settings.minimum_remaining_size_pct):
            return 0, "risk_budget_cap"
        return adjusted_qty, "risk_budget_reduced"

    sector_risk = portfolio.sector_open_risk(symbol_to_sector, item.sector)
    max_sector_risk = max(current_total_risk + intended_risk, 1e-9) * settings.sector_risk_cap_pct
    if sector_risk + intended_risk > max_sector_risk:
        remaining_risk = max(0.0, max_sector_risk - sector_risk)
        adjusted_qty = int(remaining_risk // max(entry_price - stop_level, 0.01))
        if adjusted_qty < int(intended_qty * settings.minimum_remaining_size_pct):
            return 0, "sector_risk_cap"
        return adjusted_qty, "sector_risk_reduced"

    return intended_qty, "ok"


def pretrade_risk_check(
    portfolio: PortfolioState,
    item: WatchlistItem,
    qty: int,
    entry_price: float,
    stop_level: float,
    symbol_to_sector: dict[str, str],
    settings: StrategySettings,
) -> bool:
    adjusted_qty, _ = adjust_qty_for_portfolio_constraints(
        portfolio=portfolio,
        item=item,
        intended_qty=qty,
        entry_price=entry_price,
        stop_level=stop_level,
        symbol_to_sector=symbol_to_sector,
        settings=settings,
    )
    return adjusted_qty > 0
