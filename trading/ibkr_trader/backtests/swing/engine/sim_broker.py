"""Simulated broker for stop-limit order fill simulation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum, auto
from zoneinfo import ZoneInfo

from libs.broker_ibkr.risk_support.tick_rules import round_to_tick

from backtests.swing.config import SlippageConfig
from backtests.swing.models import Direction

logger = logging.getLogger(__name__)
_NEW_YORK_TZ = ZoneInfo("America/New_York")


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
    tag: str = ""  # e.g. "entry", "protective_stop", "addon_a"
    fill_window_start_et: str | None = None
    fill_window_end_et: str | None = None

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

    def _compute_commission(self, qty: int) -> float:
        """Compute commission with IBKR minimum ($0.35/order for equities)."""
        sc = self.slippage_config
        raw = sc.commission_per_contract * qty
        # Apply minimum only for ETF-level rates (< $0.10/share)
        if sc.commission_per_contract < 0.10:
            return max(0.35, raw)
        return raw

    def submit_order(self, order: SimOrder) -> None:
        """Add an order to the pending queue."""
        self.pending_orders.append(order)

    @staticmethod
    def _parse_window_time(value: str | None) -> time | None:
        if not value:
            return None
        hour, minute = value.split(":")
        return time(int(hour), int(minute))

    def _fill_window_open(self, order: SimOrder, bar_time: datetime) -> bool:
        start = self._parse_window_time(order.fill_window_start_et)
        end = self._parse_window_time(order.fill_window_end_et)
        if start is None or end is None:
            return True
        if bar_time.tzinfo is not None:
            current = bar_time.astimezone(_NEW_YORK_TZ).time()
        else:
            current = bar_time.time()
        return start <= current <= end

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

        for order in self.pending_orders:
            if order.symbol != symbol:
                still_pending.append(order)
                continue

            # Check expiry
            if order.submit_time and order.ttl_hours > 0:
                expiry = order.submit_time + timedelta(hours=order.ttl_hours)
                if bar_time >= expiry:
                    results.append(FillResult(
                        order=order, status=FillStatus.EXPIRED, fill_time=bar_time,
                    ))
                    continue

            # During halt: suppress all fills except cancel expired
            if is_halted:
                still_pending.append(order)
                continue

            if not self._fill_window_open(order, bar_time):
                still_pending.append(order)
                continue

            result = self._try_fill(order, bar_time, O, H, L, C, tick_size,
                                     extra_slip=just_reopened)
            if result is not None:
                results.append(result)
            else:
                still_pending.append(order)

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

    def fill_market_order(
        self,
        order: SimOrder,
        bar_time: datetime,
        price: float,
        tick_size: float,
        extra_slip: bool = False,
    ) -> FillResult:
        """Fill an immediate market order through the broker friction model."""
        if order.order_type != OrderType.MARKET:
            raise ValueError("fill_market_order only supports market orders")
        return self._fill_market(order, bar_time, price, tick_size, extra_slip)

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

        commission = self._compute_commission(order.qty)
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=slip_ticks,
            commission=commission,
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

        commission = self._compute_commission(order.qty)
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=slip_ticks,
            commission=commission,
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

        commission = self._compute_commission(order.qty)
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=slip_ticks,
            commission=commission,
        )

    def _fill_limit(
        self,
        order: SimOrder,
        bar_time: datetime,
        O: float, H: float, L: float,
        tick_size: float,
    ) -> FillResult | None:
        """Fill a passive limit order.

        Buy LIMIT: fills when bar low <= limit_price.
        Sell LIMIT: fills when bar high >= limit_price.
        No slippage on limit fills (maker assumption for ETFs).
        Favorable gap-through fills at open price.
        """
        if order.side == OrderSide.BUY:
            if O <= order.limit_price:
                # Gap through limit — fill at open (favorable)
                fill = O
            elif L <= order.limit_price:
                # Intra-bar touch — fill at limit
                fill = order.limit_price
            else:
                return None  # Not triggered
        else:  # SELL
            if O >= order.limit_price:
                fill = O
            elif H >= order.limit_price:
                fill = order.limit_price
            else:
                return None

        commission = self._compute_commission(order.qty)
        return FillResult(
            order=order,
            status=FillStatus.FILLED,
            fill_price=fill,
            fill_time=bar_time,
            slippage_ticks=0,
            commission=commission,
        )
