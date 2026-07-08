from __future__ import annotations

from dataclasses import dataclass

from .config import TradeDirection


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    side: str
    price: float
    touches: int


@dataclass(frozen=True, slots=True)
class SweepResult:
    swept: bool
    side: str = ""
    level: float = 0.0
    distance_ticks: float = 0.0
    direction: TradeDirection = TradeDirection.FLAT


def detect_liquidity_pools(
    highs: list[float],
    lows: list[float],
    *,
    min_touches: int = 2,
    tolerance_ticks: int = 2,
    tick_size: float = 0.25,
    symbol: str = "NQ",
) -> list[LiquidityPool]:
    del symbol
    tolerance = tolerance_ticks * tick_size
    pools: list[LiquidityPool] = []
    pools.extend(_cluster_levels(highs, "buy_side", min_touches, tolerance))
    pools.extend(_cluster_levels(lows, "sell_side", min_touches, tolerance))
    return pools


def detect_sweep(
    price: float,
    pools: list[LiquidityPool],
    *,
    min_ticks: int = 4,
    tick_size: float = 0.25,
    side: str | None = None,
) -> SweepResult:
    candidates = [pool for pool in pools if side is None or pool.side == side]
    for pool in candidates:
        if pool.side == "sell_side" and price <= pool.price - min_ticks * tick_size:
            return SweepResult(True, pool.side, pool.price, (pool.price - price) / tick_size, TradeDirection.LONG)
        if pool.side == "buy_side" and price >= pool.price + min_ticks * tick_size:
            return SweepResult(True, pool.side, pool.price, (price - pool.price) / tick_size, TradeDirection.SHORT)
    return SweepResult(False)


def _cluster_levels(values: list[float], side: str, min_touches: int, tolerance: float) -> list[LiquidityPool]:
    pools: list[LiquidityPool] = []
    used: set[int] = set()
    for idx, value in enumerate(values):
        if idx in used:
            continue
        cluster = [(idx, other) for idx, other in enumerate(values) if abs(other - value) <= tolerance]
        if len(cluster) >= min_touches:
            used.update(item[0] for item in cluster)
            pools.append(LiquidityPool(side=side, price=sum(item[1] for item in cluster) / len(cluster), touches=len(cluster)))
    return pools
