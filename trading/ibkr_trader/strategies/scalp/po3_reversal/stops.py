from __future__ import annotations

from strategies.scalp._shared.nq_contract import compute_contracts, round_to_tick

from . import config
from .config import SetupTier, TradeDirection
from .models import PriceBar


def compute_entry_price(bar: PriceBar, direction: TradeDirection) -> float:
    offset = config.ENTRY_OFFSET_TICKS * 0.25
    if direction is TradeDirection.LONG:
        return round_to_tick(bar.high + offset, 0.25, "up")
    if direction is TradeDirection.SHORT:
        return round_to_tick(bar.low - offset, 0.25, "down")
    return round_to_tick(bar.close, 0.25)


def compute_stop_price(*, direction: TradeDirection, smt_extreme: float, atr_1m: float) -> float:
    buffer = max(config.STOP_MIN_BUFFER_TICKS * 0.25, atr_1m * 0.25)
    if direction is TradeDirection.LONG:
        return round_to_tick(smt_extreme - buffer, 0.25, "down")
    if direction is TradeDirection.SHORT:
        return round_to_tick(smt_extreme + buffer, 0.25, "up")
    return smt_extreme


def reward_to_risk(entry: float, stop: float, target: float, direction: TradeDirection) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return max(0.0, (target - entry) * int(direction) / risk)


def target_passes_rr(rr: float, tier: SetupTier) -> bool:
    return rr >= (1.5 if tier is SetupTier.A else 1.0)


def compute_position_size(
    *,
    equity: float,
    tier: SetupTier,
    entry: float,
    stop: float,
    symbol: str,
) -> int:
    risk_pct = config.A_RISK_PCT if tier is SetupTier.A else config.B_RISK_PCT
    return compute_contracts(equity, risk_pct, entry, stop, symbol)
