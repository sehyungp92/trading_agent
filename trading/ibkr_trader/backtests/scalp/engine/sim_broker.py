from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

from strategies.scalp._shared.nq_contract import round_to_tick


class OrderSide(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class FillStatus(Enum):
    FILLED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    CANCELLED = auto()


@dataclass
class ScalpSlippageConfig:
    entry_stop_ticks_base: int = 1
    entry_stop_ticks_stress: int = 3
    stop_loss_ticks_base: int = 2
    stop_loss_ticks_stress: int = 6
    limit_target_ticks: int = 0
    market_exit_ticks: int = 2
    commission_per_contract: float = 2.25


@dataclass
class SimOrder:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: int
    stop_price: float = 0.0
    limit_price: float = 0.0
    tick_size: float = 0.25
    submit_time: datetime | None = None
    earliest_fill_time: datetime | None = None
    ttl_minutes: int = 60
    tag: str = ""
    oca_group: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FillResult:
    order: SimOrder
    status: FillStatus
    fill_price: float = 0.0
    fill_time: datetime | None = None
    slippage_ticks: int = 0
    commission: float = 0.0


@dataclass
class SimBroker:
    slippage: ScalpSlippageConfig = field(default_factory=ScalpSlippageConfig)
    pending_orders: list[SimOrder] = field(default_factory=list)
    _next_id: int = 0

    def next_order_id(self) -> str:
        self._next_id += 1
        return f"SCALP-SIM-{self._next_id}"

    def submit_order(self, order: SimOrder) -> None:
        self.pending_orders.append(order)

    def cancel_orders(self, symbol: str, *, tag: str | None = None, oca_group: str | None = None) -> list[SimOrder]:
        cancelled: list[SimOrder] = []
        kept: list[SimOrder] = []
        for order in self.pending_orders:
            matches = order.symbol == symbol
            if tag is not None:
                matches = matches and order.tag == tag
            if oca_group is not None:
                matches = matches and order.oca_group == oca_group
            if matches:
                cancelled.append(order)
            else:
                kept.append(order)
        self.pending_orders = kept
        return cancelled

    def cancel_order_id(self, order_id: str) -> SimOrder | None:
        cancelled: SimOrder | None = None
        kept: list[SimOrder] = []
        for order in self.pending_orders:
            if order.order_id == order_id and cancelled is None:
                cancelled = order
            else:
                kept.append(order)
        self.pending_orders = kept
        return cancelled

    def process_tick(
        self,
        symbol: str,
        tick_time: datetime,
        price: float,
        bid: float | None,
        ask: float | None,
        tick_size: float = 0.25,
    ) -> list[FillResult]:
        buy_price = ask if ask and ask > 0 else price
        sell_price = bid if bid and bid > 0 else price
        return self._process_price_probe(
            symbol,
            tick_time,
            open_price=price,
            high=price,
            low=price,
            buy_price=buy_price,
            sell_price=sell_price,
            tick_size=tick_size,
        )

    def process_bar(
        self,
        symbol: str,
        bar_time: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        tick_size: float = 0.25,
    ) -> list[FillResult]:
        del close
        return self._process_price_probe(
            symbol,
            bar_time,
            open_price=open_price,
            high=high,
            low=low,
            buy_price=open_price,
            sell_price=open_price,
            tick_size=tick_size,
        )

    def _process_price_probe(
        self,
        symbol: str,
        event_time: datetime,
        *,
        open_price: float,
        high: float,
        low: float,
        buy_price: float,
        sell_price: float,
        tick_size: float,
    ) -> list[FillResult]:
        triggered: list[FillResult] = []
        still_pending: list[SimOrder] = []
        for order in self.pending_orders:
            if order.symbol != symbol:
                still_pending.append(order)
                continue
            if order.earliest_fill_time is not None and event_time <= order.earliest_fill_time:
                still_pending.append(order)
                continue
            if order.submit_time and order.ttl_minutes > 0 and event_time >= order.submit_time + timedelta(minutes=order.ttl_minutes):
                triggered.append(FillResult(order, FillStatus.EXPIRED, fill_time=event_time))
                continue
            result = self._try_fill(order, event_time, open_price, high, low, buy_price, sell_price, tick_size)
            if result is None:
                still_pending.append(order)
            else:
                triggered.append(result)

        triggered = self._resolve_oca(triggered)
        filled_oca = {result.order.oca_group for result in triggered if result.status is FillStatus.FILLED and result.order.oca_group}
        if filled_oca:
            still_pending = [order for order in still_pending if order.oca_group not in filled_oca]
        self.pending_orders = still_pending
        return triggered

    def _try_fill(
        self,
        order: SimOrder,
        event_time: datetime,
        open_price: float,
        high: float,
        low: float,
        buy_price: float,
        sell_price: float,
        tick_size: float,
    ) -> FillResult | None:
        if order.order_type is OrderType.MARKET:
            price = buy_price if order.side is OrderSide.BUY else sell_price
            slip = self.slippage.market_exit_ticks
            fill = self._slipped(order, price, tick_size, slip)
            return self._filled(order, event_time, fill, slip)
        if order.order_type is OrderType.LIMIT:
            return self._fill_limit(order, event_time, open_price, high, low)
        if order.order_type is OrderType.STOP:
            return self._fill_stop(order, event_time, open_price, high, low, tick_size)
        if order.order_type is OrderType.STOP_LIMIT:
            result = self._fill_stop(order, event_time, open_price, high, low, tick_size)
            if result is None:
                return None
            if order.side is OrderSide.BUY and result.fill_price > order.limit_price:
                return FillResult(order, FillStatus.REJECTED, result.fill_price, event_time)
            if order.side is OrderSide.SELL and result.fill_price < order.limit_price:
                return FillResult(order, FillStatus.REJECTED, result.fill_price, event_time)
            return result
        return None

    def _fill_limit(self, order: SimOrder, event_time: datetime, open_price: float, high: float, low: float) -> FillResult | None:
        if order.side is OrderSide.BUY:
            if open_price <= order.limit_price:
                fill = open_price
            elif low <= order.limit_price:
                fill = order.limit_price
            else:
                return None
        else:
            if open_price >= order.limit_price:
                fill = open_price
            elif high >= order.limit_price:
                fill = order.limit_price
            else:
                return None
        slip = self.slippage.limit_target_ticks if order.tag in {"target", "tp1", "tp2"} else 0
        return self._filled(order, event_time, self._slipped(order, fill, order.tick_size, slip), slip)

    def _fill_stop(self, order: SimOrder, event_time: datetime, open_price: float, high: float, low: float, tick_size: float) -> FillResult | None:
        if order.side is OrderSide.BUY:
            if open_price >= order.stop_price:
                base = max(open_price, order.stop_price)
            elif high >= order.stop_price:
                base = order.stop_price
            else:
                return None
        else:
            if open_price <= order.stop_price:
                base = min(open_price, order.stop_price)
            elif low <= order.stop_price:
                base = order.stop_price
            else:
                return None
        slip = self.slippage.stop_loss_ticks_base if order.tag in {"stop", "protective_stop"} else self.slippage.entry_stop_ticks_base
        return self._filled(order, event_time, self._slipped(order, base, tick_size, slip), slip)

    def _slipped(self, order: SimOrder, price: float, tick_size: float, slip_ticks: int) -> float:
        if order.side is OrderSide.BUY:
            return round_to_tick(price + slip_ticks * tick_size, tick_size, "up")
        return round_to_tick(price - slip_ticks * tick_size, tick_size, "down")

    def _filled(self, order: SimOrder, event_time: datetime, fill_price: float, slip_ticks: int) -> FillResult:
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill_price,
            fill_time=event_time,
            slippage_ticks=slip_ticks,
            commission=self.slippage.commission_per_contract * order.qty,
        )

    def _resolve_oca(self, results: list[FillResult]) -> list[FillResult]:
        by_oca: dict[str, list[FillResult]] = {}
        output: list[FillResult] = []
        for result in results:
            if result.status is not FillStatus.FILLED or not result.order.oca_group:
                output.append(result)
                continue
            by_oca.setdefault(result.order.oca_group, []).append(result)
        for group_results in by_oca.values():
            # If stop and target are both touched in the same bar, assume stop first.
            group_results.sort(key=lambda item: 0 if item.order.tag in {"stop", "protective_stop"} else 1)
            output.append(group_results[0])
        return output
