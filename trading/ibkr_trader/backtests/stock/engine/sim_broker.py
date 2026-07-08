"""Simulated broker for stock order fill simulation.

Adapted from momentum SimBroker with stock-specific changes:
- Per-share commission (vs per-contract)
- Decimal price handling (no tick_size rounding needed for most stocks)
- Spread-based slippage model (bps of price)
- point_value = 1.0 for stocks
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

from backtests.stock.config import SlippageConfig
from backtests.stock.models import Direction

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    STOP_LIMIT = auto()
    MARKET = auto()
    STOP = auto()
    LIMIT = auto()


class FillStatus(Enum):
    FILLED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    CANCELLED = auto()


@dataclass
class SimOrder:
    """A pending order in the simulated broker."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: int
    stop_price: float = 0.0
    limit_price: float = 0.0
    submit_time: datetime | None = None
    ttl_hours: int = 6
    ttl_minutes: int = 0
    tag: str = ""
    oca_group: str = ""
    invalidation_price: float = 0.0
    triggered_ts: datetime | None = None

    @property
    def direction(self) -> int:
        return Direction.LONG if self.side == OrderSide.BUY else Direction.SHORT


@dataclass
class FillResult:
    """Result of processing an order against a bar."""

    order: SimOrder
    status: FillStatus
    fill_price: float = 0.0
    fill_time: datetime | None = None
    slippage_dollars: float = 0.0
    commission: float = 0.0


@dataclass
class SimBroker:
    """Simulated broker for stock orders against OHLC bars.

    Handles:
    - Stop-limit, market, stop, and limit order fills
    - Spread-based slippage (bps of price)
    - Per-share commission
    - Order expiry (TTL)
    - OCA group cancellation
    - Halt/limit simulation (zero-range bar detection)
    """

    slippage_config: SlippageConfig = field(default_factory=SlippageConfig)
    pending_orders: list[SimOrder] = field(default_factory=list)
    _next_id: int = field(default=0, repr=False)
    _consecutive_zero_range: dict[str, int] = field(default_factory=dict, repr=False)
    _halted: dict[str, bool] = field(default_factory=dict, repr=False)
    _was_halted: dict[str, bool] = field(default_factory=dict, repr=False)

    def next_order_id(self) -> str:
        self._next_id += 1
        return f"SIM-{self._next_id}"

    def submit_order(self, order: SimOrder) -> None:
        self.pending_orders.append(order)

    def cancel_orders(self, symbol: str, tag: str | None = None) -> list[SimOrder]:
        cancelled = []
        remaining = []
        for o in self.pending_orders:
            if o.symbol == symbol and (tag is None or o.tag == tag):
                cancelled.append(o)
            else:
                remaining.append(o)
        self.pending_orders = remaining
        return cancelled

    def cancel_all(self, symbol: str) -> list[SimOrder]:
        return self.cancel_orders(symbol)

    def cancel_oca_group(self, oca_group: str) -> list[SimOrder]:
        if not oca_group:
            return []
        cancelled = []
        remaining = []
        for o in self.pending_orders:
            if o.oca_group == oca_group:
                cancelled.append(o)
            else:
                remaining.append(o)
        self.pending_orders = remaining
        return cancelled

    def is_halted(self, symbol: str) -> bool:
        return self._halted.get(symbol, False)

    def process_bar(
        self,
        symbol: str,
        bar_time: datetime,
        O: float,
        H: float,
        L: float,
        C: float,
    ) -> list[FillResult]:
        """Check all pending orders against the current bar."""
        sc = self.slippage_config

        # Halt detection: zero-range bar
        is_zero_range = (H == L) or (abs(H - L) < 0.005)
        if is_zero_range:
            self._consecutive_zero_range[symbol] = self._consecutive_zero_range.get(symbol, 0) + 1
        else:
            self._consecutive_zero_range[symbol] = 0

        prev_halted = self._halted.get(symbol, False)
        if self._consecutive_zero_range.get(symbol, 0) >= sc.halt_zero_range_bars:
            self._halted[symbol] = True
        else:
            self._halted[symbol] = False

        is_halted = self._halted[symbol]
        just_reopened = prev_halted and not is_halted
        self._was_halted[symbol] = just_reopened

        results: list[FillResult] = []
        still_pending: list[SimOrder] = []
        filled_oca_groups: set[str] = set()

        # Process stops before limits (conservative: adverse fills first)
        _priority = {OrderType.STOP: 0, OrderType.STOP_LIMIT: 1, OrderType.MARKET: 2, OrderType.LIMIT: 3}
        sorted_orders = sorted(self.pending_orders, key=lambda o: _priority.get(o.order_type, 9))

        for order in sorted_orders:
            if order.symbol != symbol:
                still_pending.append(order)
                continue

            # Check OCA: skip if a sibling already filled this bar
            if order.oca_group and order.oca_group in filled_oca_groups:
                results.append(FillResult(
                    order=order, status=FillStatus.CANCELLED, fill_time=bar_time,
                ))
                continue

            # Check expiry
            if order.submit_time:
                if order.ttl_minutes > 0:
                    expiry = order.submit_time + timedelta(minutes=order.ttl_minutes)
                elif order.ttl_hours > 0:
                    expiry = order.submit_time + timedelta(hours=order.ttl_hours)
                else:
                    expiry = None
                if expiry is not None and bar_time >= expiry:
                    results.append(FillResult(
                        order=order, status=FillStatus.EXPIRED, fill_time=bar_time,
                    ))
                    continue

            # Check invalidation price
            if order.invalidation_price > 0.0:
                invalidated = False
                if order.side == OrderSide.BUY and L <= order.invalidation_price:
                    invalidated = True
                elif order.side == OrderSide.SELL and H >= order.invalidation_price:
                    invalidated = True
                if invalidated:
                    results.append(FillResult(
                        order=order, status=FillStatus.CANCELLED, fill_time=bar_time,
                    ))
                    continue

            # During halt: suppress all fills
            if is_halted:
                still_pending.append(order)
                continue

            result = self._try_fill(order, bar_time, O, H, L, C, extra_slip=just_reopened)
            if result is not None:
                if (result.status == FillStatus.REJECTED
                        and order.order_type == OrderType.STOP_LIMIT
                        and order.triggered_ts is None):
                    order.triggered_ts = bar_time
                    still_pending.append(order)
                    continue
                results.append(result)
                if result.status == FillStatus.FILLED and order.oca_group:
                    filled_oca_groups.add(order.oca_group)
            else:
                still_pending.append(order)

        # OCA cancellation
        if filled_oca_groups:
            final_pending: list[SimOrder] = []
            for order in still_pending:
                if order.oca_group and order.oca_group in filled_oca_groups:
                    results.append(FillResult(
                        order=order, status=FillStatus.CANCELLED, fill_time=bar_time,
                    ))
                else:
                    final_pending.append(order)
            self.pending_orders = final_pending
        else:
            self.pending_orders = still_pending
        return results

    def _get_slippage_bps(self, price: float, extra_slip: bool = False) -> float:
        """Get slippage in dollar terms based on bps model."""
        sc = self.slippage_config
        bps = sc.slip_bps_normal
        if extra_slip:
            bps += sc.halt_extra_slip_bps
        return price * bps / 10_000

    def _try_fill(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float, C: float,
        extra_slip: bool = False,
    ) -> FillResult | None:
        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar_time, O, extra_slip)
        elif order.order_type == OrderType.STOP:
            return self._fill_stop(order, bar_time, O, H, L, extra_slip)
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._fill_stop_limit(order, bar_time, O, H, L, extra_slip)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar_time, O, H, L)
        return None

    def _fill_market(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float,
        extra_slip: bool = False,
    ) -> FillResult:
        slip = self._get_slippage_bps(O, extra_slip)
        if order.side == OrderSide.BUY:
            fill = O + slip
        else:
            fill = O - slip
        fill = round(fill, 2)

        commission = self.slippage_config.commission_per_share * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_dollars=slip * order.qty,
            commission=commission,
        )

    def _fill_stop(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float,
        extra_slip: bool = False,
    ) -> FillResult | None:
        triggered = False
        gap_open = False

        if order.side == OrderSide.SELL:
            if O <= order.stop_price:
                triggered = True
                gap_open = True
            elif L <= order.stop_price:
                triggered = True
        else:
            if O >= order.stop_price:
                triggered = True
                gap_open = True
            elif H >= order.stop_price:
                triggered = True

        if not triggered:
            return None

        slip = self._get_slippage_bps(order.stop_price, extra_slip)
        if order.side == OrderSide.SELL:
            base = min(O, order.stop_price) if gap_open else order.stop_price
            fill = round(base - slip, 2)
        else:
            base = max(O, order.stop_price) if gap_open else order.stop_price
            fill = round(base + slip, 2)

        commission = self.slippage_config.commission_per_share * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_dollars=slip * order.qty,
            commission=commission,
        )

    def _fill_stop_limit(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float,
        extra_slip: bool = False,
    ) -> FillResult | None:
        triggered = False
        gap_open = False

        if order.side == OrderSide.BUY:
            if O >= order.stop_price:
                triggered = True
                gap_open = True
            elif H >= order.stop_price:
                triggered = True
        else:
            if O <= order.stop_price:
                triggered = True
                gap_open = True
            elif L <= order.stop_price:
                triggered = True

        if not triggered:
            return None

        slip = self._get_slippage_bps(order.stop_price, extra_slip)

        if order.side == OrderSide.BUY:
            base = max(O, order.stop_price) if gap_open else order.stop_price
            fill = round(base + slip, 2)
            if fill > order.limit_price:
                return FillResult(
                    order=order, status=FillStatus.REJECTED,
                    fill_price=fill, fill_time=bar_time,
                )
        else:
            base = min(O, order.stop_price) if gap_open else order.stop_price
            fill = round(base - slip, 2)
            if fill < order.limit_price:
                return FillResult(
                    order=order, status=FillStatus.REJECTED,
                    fill_price=fill, fill_time=bar_time,
                )

        commission = self.slippage_config.commission_per_share * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_dollars=slip * order.qty,
            commission=commission,
        )

    def _fill_limit(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float,
    ) -> FillResult | None:
        """Fill a passive limit order (conservative trade-through model).

        Buy LIMIT at P: fills when L <= P - 0.01 (trade-through by 1 cent).
        No slippage on limit fills (maker assumption).
        """
        tick = 0.01
        if order.side == OrderSide.BUY:
            if O <= order.limit_price - tick:
                fill = O
            elif L <= order.limit_price - tick:
                fill = order.limit_price
            else:
                return None
        else:
            if O >= order.limit_price + tick:
                fill = O
            elif H >= order.limit_price + tick:
                fill = order.limit_price
            else:
                return None

        commission = self.slippage_config.commission_per_share * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_dollars=0.0,
            commission=commission,
        )
