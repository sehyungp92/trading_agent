"""Simulated broker for stop-limit order fill simulation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

from backtests.momentum.config import SlippageConfig, round_to_tick
from backtests.momentum.models import Direction

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    STOP_LIMIT = auto()
    MARKET = auto()
    STOP = auto()
    LIMIT = auto()  # Passive limit order (for ETF entry A/C)


class FillStatus(Enum):
    FILLED = auto()
    REJECTED = auto()  # Limit exceeded
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
    stop_price: float = 0.0   # Trigger price for stop/stop-limit
    limit_price: float = 0.0  # Max fill price for stop-limit
    tick_size: float = 0.01
    submit_time: datetime | None = None
    ttl_hours: int = 6
    ttl_minutes: int = 0       # Minute-level TTL (used instead of ttl_hours when > 0)
    tag: str = ""  # e.g. "entry", "protective_stop", "addon_a"
    oca_group: str = ""        # OCA group ID; fill cancels siblings
    invalidation_price: float = 0.0  # Cancel if price breaches this level
    triggered_ts: datetime | None = None  # When stop-limit stop triggered but limit rejected

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
    slippage_ticks: int = 0
    commission: float = 0.0
    filled_at_open: bool = False


@dataclass
class SimBroker:
    """Simulated broker that processes stop-limit orders against OHLC bars.

    Handles:
    - Stop-limit entry fills (gap-open, intra-bar trigger, limit rejection)
    - Market order fills at open
    - Protective stop fills
    - Order expiry (TTL)
    - Slippage and commissions
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
        """Add an order to the pending queue."""
        self.pending_orders.append(order)

    def cancel_orders(
        self,
        symbol: str,
        tag: str | None = None,
    ) -> list[SimOrder]:
        """Cancel pending orders for a symbol, optionally filtered by tag."""
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
        """Cancel all pending orders for a symbol."""
        return self.cancel_orders(symbol)

    def cancel_oca_group(self, oca_group: str) -> list[SimOrder]:
        """Cancel all pending orders in an OCA group."""
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

    def fill_marketable_ioc_limit(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float,
        H: float,
        L: float,
        C: float,
        tick_size: float,
    ) -> FillResult:
        """Evaluate an immediate marketable IOC limit at the decision reference.

        The order is never queued. It is modeled as a taker order sent after
        the completed signal bar, so the only causal price reference available
        to this deterministic backtest is the completed-bar close. High/low are
        accepted for call-site symmetry with other bar fill helpers but are not
        used to decide a post-close IOC fill.
        """
        del O, H, L
        slip_ticks = self._get_slippage_ticks(bar_time, 0, C, tick_size)
        if order.side == OrderSide.BUY:
            fill = round_to_tick(C + slip_ticks * tick_size, tick_size, "up")
            if fill > order.limit_price:
                return FillResult(
                    order=order,
                    status=FillStatus.REJECTED,
                    fill_price=order.limit_price,
                    fill_time=bar_time,
                )
        else:
            fill = round_to_tick(C - slip_ticks * tick_size, tick_size, "down")
            if fill < order.limit_price:
                return FillResult(
                    order=order,
                    status=FillStatus.REJECTED,
                    fill_price=order.limit_price,
                    fill_time=bar_time,
                )

        commission = self.slippage_config.commission_per_contract * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=slip_ticks,
            commission=commission,
            filled_at_open=False,
        )

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
        tick_size: float,
    ) -> list[FillResult]:
        """Check all pending orders against the current bar.

        Detects halt conditions (zero-range bars) and suppresses stop fills
        during halts. On reopen, fills with additional slippage.
        """
        sc = self.slippage_config
        # Halt detection: zero-range bar
        is_zero_range = (H == L) or (abs(H - L) < tick_size * 0.5)
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
            if order.oca_group and order.oca_group in filled_oca_groups:
                results.append(FillResult(
                    order=order, status=FillStatus.CANCELLED, fill_time=bar_time,
                ))
                continue

            # Check expiry (minute-level takes precedence over hour-level)
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

            # During halt: suppress all fills except cancel expired
            if is_halted:
                still_pending.append(order)
                continue

            result = self._try_fill(order, bar_time, O, H, L, C, tick_size,
                                     extra_slip=just_reopened)
            if result is not None:
                # Track triggered-but-not-filled for stop-limits
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

        # OCA cancellation: cancel siblings of filled orders
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

    def _try_fill(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float, C: float,
        tick_size: float,
        extra_slip: bool = False,
    ) -> FillResult | None:
        """Attempt to fill a single order. Returns None if not triggered."""
        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar_time, O, tick_size, extra_slip)
        elif order.order_type == OrderType.STOP:
            return self._fill_stop(order, bar_time, O, H, L, tick_size, extra_slip)
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._fill_stop_limit(order, bar_time, O, H, L, tick_size, extra_slip)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar_time, O, H, L, tick_size)
        return None

    def _get_slippage_ticks(
        self,
        bar_time: datetime,
        extra_slip: bool = False,
        ref_price: float = 0.0,
        tick_size: float = 0.01,
    ) -> int:
        """Get slippage ticks based on time of day, halt reopen, and spread.

        When spread_bps > 0 and ref_price is provided, adds half-spread
        slippage (in ticks) on top of the base tick slippage.
        """
        sc = self.slippage_config
        base = sc.slip_ticks_illiquid if bar_time.hour in sc.illiquid_hours else sc.slip_ticks_normal
        if extra_slip:
            base += sc.halt_extra_slip_ticks
        if sc.spread_bps > 0 and ref_price > 0 and tick_size > 0:
            half_spread_dollars = ref_price * sc.spread_bps / 10_000 / 2
            base += max(1, int(half_spread_dollars / tick_size))
        return base

    def _fill_market(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float,
        tick_size: float,
        extra_slip: bool = False,
    ) -> FillResult:
        """Fill market order at open + slippage."""
        slip_ticks = self._get_slippage_ticks(bar_time, extra_slip, O, tick_size)
        if order.side == OrderSide.BUY:
            fill = round_to_tick(O + slip_ticks * tick_size, tick_size, "up")
        else:
            fill = round_to_tick(O - slip_ticks * tick_size, tick_size, "down")

        commission = self.slippage_config.commission_per_contract * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=slip_ticks,
            commission=commission,
            filled_at_open=True,
        )

    def _fill_stop(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float,
        tick_size: float,
        extra_slip: bool = False,
    ) -> FillResult | None:
        """Fill protective stop order."""
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

        slip_ticks = self._get_slippage_ticks(bar_time, extra_slip, order.stop_price, tick_size)
        if order.side == OrderSide.SELL:
            if gap_open:
                base = min(O, order.stop_price)
            else:
                base = order.stop_price
            fill = round_to_tick(base - slip_ticks * tick_size, tick_size, "down")
        else:
            if gap_open:
                base = max(O, order.stop_price)
            else:
                base = order.stop_price
            fill = round_to_tick(base + slip_ticks * tick_size, tick_size, "up")

        commission = self.slippage_config.commission_per_contract * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=slip_ticks,
            commission=commission,
            filled_at_open=gap_open,
        )

    def _fill_stop_limit(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float,
        tick_size: float,
        extra_slip: bool = False,
    ) -> FillResult | None:
        """Fill stop-limit entry order."""
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

        slip_ticks = self._get_slippage_ticks(bar_time, extra_slip, order.stop_price, tick_size)

        if order.side == OrderSide.BUY:
            if gap_open:
                base = max(O, order.stop_price)
            else:
                base = order.stop_price
            fill = round_to_tick(base + slip_ticks * tick_size, tick_size, "up")
            # Limit check
            if fill > order.limit_price:
                return FillResult(
                    order=order,
                    status=FillStatus.REJECTED,
                    fill_price=fill,
                    fill_time=bar_time,
                )
        else:
            if gap_open:
                base = min(O, order.stop_price)
            else:
                base = order.stop_price
            fill = round_to_tick(base - slip_ticks * tick_size, tick_size, "down")
            # Limit check (for short, limit is the minimum acceptable price)
            if fill < order.limit_price:
                return FillResult(
                    order=order,
                    status=FillStatus.REJECTED,
                    fill_price=fill,
                    fill_time=bar_time,
                )

        commission = self.slippage_config.commission_per_contract * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=slip_ticks,
            commission=commission,
            filled_at_open=gap_open,
        )

    def _fill_limit(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float,
        tick_size: float,
    ) -> FillResult | None:
        """Fill a passive limit order (conservative trade-through model).

        Buy LIMIT at P: fills only when price trades through by >= 1 tick
        (L <= P - tick_size).  Sell LIMIT at P: H >= P + tick_size.
        No slippage on limit fills (maker assumption).
        Gap-through fills at open price when open is >= 1 tick through.
        """
        if order.side == OrderSide.BUY:
            if O <= order.limit_price - tick_size:
                # Gap through limit by >= 1 tick — fill at open (favorable)
                fill = O
                filled_at_open = True
            elif L <= order.limit_price - tick_size:
                # Trade-through by >= 1 tick — fill at limit
                fill = order.limit_price
                filled_at_open = False
            else:
                return None  # Touch only or not reached
        else:  # SELL
            if O >= order.limit_price + tick_size:
                fill = O
                filled_at_open = True
            elif H >= order.limit_price + tick_size:
                fill = order.limit_price
                filled_at_open = False
            else:
                return None

        commission = self.slippage_config.commission_per_contract * order.qty
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=0,
            commission=commission,
            filled_at_open=filled_at_open,
        )
