from __future__ import annotations

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp._shared.nq_contract import compute_contracts, round_to_tick

from . import config
from .config import IvbModule, TradeDirection


def continuation_stop(*, direction: TradeDirection, ivb: IVBLevels) -> float:
    return ivb.low if direction is TradeDirection.LONG else ivb.high


def reclaim_stop(
    *,
    direction: TradeDirection,
    ivb: IVBLevels,
    failed_break_extreme: float,
) -> float:
    if direction is TradeDirection.LONG:
        return round_to_tick(min(ivb.low, failed_break_extreme) - 0.25, 0.25, "down")
    return round_to_tick(max(ivb.high, failed_break_extreme) + 0.25, 0.25, "up")


def reward_to_risk(entry: float, stop: float, target: float, direction: TradeDirection) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = (target - entry) * int(direction)
    return max(0.0, reward / risk)


def stop_within_cap(entry: float, stop: float, ivb: IVBLevels) -> bool:
    cap = max(ivb.range_pts * config.STOP_CAP_IVB_FRACTION, 0.25)
    return abs(entry - stop) <= cap


def compute_position_size(
    *,
    equity: float,
    module: IvbModule,
    size_multiplier: float,
    entry: float,
    stop: float,
    symbol: str,
) -> int:
    module_mult = 0.75 if module is IvbModule.A2_RECLAIM else 1.0
    risk_pct = config.BASE_RISK_PCT * module_mult * max(0.1, size_multiplier)
    return compute_contracts(equity, risk_pct, entry, stop, symbol)
