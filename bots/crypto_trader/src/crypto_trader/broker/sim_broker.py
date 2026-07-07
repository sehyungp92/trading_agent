"""Simulated broker for backtesting crypto perpetual futures.

Implements BrokerAdapter protocol with crypto-specific features:
- BPS-based fees and slippage (not tick-based)
- Fractional quantities (0.001 BTC)
- Hourly funding accrual
- Tiered margin liquidation
- OCA order groups
- TTL-based order expiry
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import structlog

from crypto_trader.core.models import (
    Bar,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TerminalMark,
    Trade,
)
from crypto_trader.core.order_semantics import EXIT_OCA_POLICY, STOP_LOSS_TRIGGER_TAGS, is_exit_order
from crypto_trader.exchange.funding import FundingHelper
from crypto_trader.exchange.meta import AssetMeta

log = structlog.get_logger()
EXIT_ONLY_STOP_TAGS = STOP_LOSS_TRIGGER_TAGS - {"stop"}


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


class SimBroker:
    """Simulated broker with crypto perpetual futures semantics.

    Fill ordering per bar (strict priority):
      1. Expire TTL orders
      2. Stop orders (protective stops — adverse fills first)
      3. Stop-limit orders (entry triggers)
      4. Market orders (fill at open ± spread/slippage)
      5. Limit orders (conservative trade-through)
      6. OCA cancellations
      7. Funding accrual (hourly)
      8. Mark-to-market equity update
      9. Liquidation check
    """

    _STATE_SNAPSHOT_FIELDS = (
        "_initial_equity",
        "_equity",
        "_cash",
        "_pending_orders",
        "_positions",
        "_closed_trades",
        "_terminal_marks",
        "_fills",
        "_equity_history",
        "_liquidation_equity_history",
        "_funding_log",
        "_last_funding_hour",
        "_next_order_id",
        "_trade_id",
        "_last_bar",
        "_last_prices",
        "_bar_count_per_position",
        "_deferring",
        "_deferred_orders",
        "_cancelled_oca_orders",
    )

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        taker_fee_bps: float = 3.5,
        maker_fee_bps: float = 1.0,
        slippage_bps: float = 2.0,
        spread_bps: float = 2.0,
        asset_meta: AssetMeta | None = None,
        funding_helper: FundingHelper | None = None,
        default_leverage: float = 10.0,
        funding_helpers: dict[str, FundingHelper] | None = None,
    ) -> None:
        self._initial_equity = initial_equity
        self._equity = initial_equity
        self._cash = initial_equity
        self.taker_fee_bps = taker_fee_bps
        self.maker_fee_bps = maker_fee_bps
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.asset_meta = asset_meta
        self.funding_helper = funding_helper
        self.default_leverage = default_leverage
        self._funding_helpers: dict[str, FundingHelper] = funding_helpers or {}

        # Internal state
        self._pending_orders: list[Order] = []
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[Trade] = []
        self._terminal_marks: list[TerminalMark] = []
        self._fills: list[Fill] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._liquidation_equity_history: list[tuple[datetime, float]] = []
        self._funding_log: list[dict] = []
        self._last_funding_hour: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._next_order_id: int = 1
        self._trade_id: int = 1
        self._last_bar: Bar | None = None
        self._last_prices: dict[str, float] = {}  # last known price per symbol
        self._bar_count_per_position: dict[str, int] = {}  # bars since position opened

        # Order deferral — prevents higher-TF timing leak (Finding 1)
        self._deferring: bool = False
        self._deferred_orders: list[Order] = []
        self._cancelled_oca_orders: list[Order] = []

    def snapshot_state(self) -> dict[str, Any]:
        """Return an in-memory checkpoint of all mutable broker state."""
        return {
            field: deepcopy(getattr(self, field))
            for field in self._STATE_SNAPSHOT_FIELDS
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a checkpoint produced by :meth:`snapshot_state`."""
        for field in self._STATE_SNAPSHOT_FIELDS:
            if field in snapshot:
                setattr(self, field, deepcopy(snapshot[field]))

    # -------------------------------------------------------------------
    # BrokerAdapter interface
    # -------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit an order. Assigns order_id and queues for processing."""
        # Validate order
        if order.qty <= 0:
            order.status = OrderStatus.REJECTED
            log.debug("order.rejected", reason="qty <= 0", qty=order.qty)
            return ""
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            order.status = OrderStatus.REJECTED
            log.warning("order.rejected", reason="limit order missing limit_price")
            return ""
        if order.order_type == OrderType.STOP and order.stop_price is None:
            order.status = OrderStatus.REJECTED
            log.warning("order.rejected", reason="stop order missing stop_price")
            return ""
        if order.order_type == OrderType.STOP_LIMIT and (order.stop_price is None or order.limit_price is None):
            order.status = OrderStatus.REJECTED
            log.warning("order.rejected", reason="stop_limit order missing prices")
            return ""

        oid = str(self._next_order_id)
        self._next_order_id += 1
        order.order_id = oid
        order.status = OrderStatus.PENDING
        if self._last_bar is not None:
            order.submit_time = self._last_bar.timestamp

        if self._deferring:
            self._deferred_orders.append(order)
        else:
            self._pending_orders.append(order)

        log.debug("order.submitted", order_id=oid, symbol=order.symbol, side=order.side.value, type=order.order_type.value,
                   deferred=self._deferring)
        return oid

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific pending or deferred order."""
        for order in self._pending_orders:
            if order.order_id == order_id and order.status in (OrderStatus.PENDING, OrderStatus.WORKING):
                order.status = OrderStatus.CANCELLED
                return True
        for order in self._deferred_orders:
            if order.order_id == order_id and order.status in (OrderStatus.PENDING, OrderStatus.WORKING):
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def cancel_all(self, symbol: str = "") -> int:
        """Cancel all open orders (pending + deferred), optionally filtered by symbol."""
        count = 0
        for order in (*self._pending_orders, *self._deferred_orders):
            if order.status not in (OrderStatus.PENDING, OrderStatus.WORKING):
                continue
            if symbol and order.symbol != symbol:
                continue
            order.status = OrderStatus.CANCELLED
            count += 1
        return count

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        active = (OrderStatus.PENDING, OrderStatus.WORKING)
        return [
            o for o in self._pending_orders
            if o.status in active and (not symbol or o.symbol == symbol)
        ]

    def get_equity(self) -> float:
        return self._equity

    def get_fills_since(self, since: datetime) -> list[Fill]:
        return [f for f in self._fills if f.timestamp >= since]

    def drain_cancelled_oca_orders(self) -> list[Order]:
        cancelled = list(self._cancelled_oca_orders)
        self._cancelled_oca_orders.clear()
        return cancelled

    def get_funding_rate(self, symbol: str, timestamp_ms: int) -> float:
        """Return the most recent funding rate for a symbol at a given time."""
        helper = self._funding_helpers.get(symbol) or self.funding_helper
        if helper is None:
            return 0.0
        return float(helper.get_rate_at(timestamp_ms))

    # -------------------------------------------------------------------
    # Order deferral control (Finding 1: timing leak prevention)
    # -------------------------------------------------------------------

    def start_deferring(self) -> None:
        """Begin deferring: new orders go to _deferred_orders, not _pending_orders."""
        self._deferring = True

    def stop_deferring(self) -> None:
        """Stop deferring: subsequent orders go directly to _pending_orders."""
        self._deferring = False

    def activate_deferred(self) -> None:
        """Promote deferred orders into _pending_orders for the next process_bar."""
        if self._deferred_orders:
            self._pending_orders.extend(self._deferred_orders)
            self._deferred_orders.clear()

    def check_entry_bar_stops(self, bar: Bar) -> list[Fill]:
        """Recheck protective stops against the bar that triggered entry (Finding 2).

        After a market entry fills at bar.open and on_fill submits a protective
        stop, that stop is not checked until the NEXT process_bar — missing
        intra-bar adverse moves. This method rechecks only newly-submitted
        stops against the current bar.
        """
        new_fills = self._process_stop_orders(bar)
        if new_fills:
            self._process_oca_cancels(new_fills)
        return new_fills

    # -------------------------------------------------------------------
    # Read-only accessors
    # -------------------------------------------------------------------

    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    @property
    def closed_trades(self) -> list[Trade]:
        return list(self._closed_trades)

    @property
    def terminal_marks(self) -> list[TerminalMark]:
        return list(self._terminal_marks)

    @property
    def equity_history(self) -> list[tuple[datetime, float]]:
        return list(self._equity_history)

    @property
    def liquidation_equity_history(self) -> list[tuple[datetime, float]]:
        return list(self._liquidation_equity_history)

    # -------------------------------------------------------------------
    # Bar processing pipeline
    # -------------------------------------------------------------------

    def process_bar(self, bar: Bar) -> list[Fill]:
        """Process all pending orders against this bar. Returns list of fills."""
        fills: list[Fill] = []

        # Track last known price per symbol
        self._last_prices[bar.symbol] = bar.close

        # Track bars held for open positions
        if bar.symbol in self._positions:
            self._bar_count_per_position[bar.symbol] = (
                self._bar_count_per_position.get(bar.symbol, 0) + 1
            )

        # 1. Expire TTL orders
        self._expire_orders(bar)

        # 2. Stops (protective — adverse fills first)
        fills.extend(self._process_stop_orders(bar))

        # 3. Stop-limits (entry triggers)
        fills.extend(self._process_stop_limit_orders(bar))

        # 4. Market orders
        fills.extend(self._process_market_orders(bar))

        # 5. Limit orders
        fills.extend(self._process_limit_orders(bar))

        # 6. OCA cancellations
        self._process_oca_cancels(fills)

        # 7. Funding accrual
        self._process_funding(bar)

        # 8. Equity mark-to-market
        self._update_equity(bar)

        # 9. Liquidation check
        liq_fills = self._check_liquidations(bar)
        fills.extend(liq_fills)
        if liq_fills:
            self.refresh_current_bar_equity(bar.timestamp)

        # Clean up cancelled/filled orders
        self._pending_orders = [
            o for o in self._pending_orders
            if o.status in (OrderStatus.PENDING, OrderStatus.WORKING)
        ]

        self._last_bar = bar
        return fills

    # -------------------------------------------------------------------
    # Order expiry
    # -------------------------------------------------------------------

    def _expire_orders(self, bar: Bar) -> None:
        """Increment bar counters and expire orders past their TTL."""
        for order in self._pending_orders:
            if order.status not in (OrderStatus.PENDING, OrderStatus.WORKING):
                continue
            if order.symbol != bar.symbol:
                continue
            order._bars_alive += 1
            if order.ttl_bars is not None and order._bars_alive > order.ttl_bars:
                order.status = OrderStatus.EXPIRED
                log.debug("order.expired", order_id=order.order_id, bars_alive=order._bars_alive)

    # -------------------------------------------------------------------
    # Stop orders
    # -------------------------------------------------------------------

    def _process_stop_orders(self, bar: Bar) -> list[Fill]:
        """Process protective stop orders with gap detection."""
        fills: list[Fill] = []
        for order in self._pending_orders:
            if order.status not in (OrderStatus.PENDING, OrderStatus.WORKING):
                continue
            if order.symbol != bar.symbol:
                continue
            if order.order_type != OrderType.STOP:
                continue
            if order.stop_price is None:
                continue
            if order.tag in EXIT_ONLY_STOP_TAGS:
                pos = self._positions.get(order.symbol)
                # Protective stops are exit-only orders. If the position is already
                # gone, or the order side no longer opposes the live position,
                # cancel the orphan instead of letting it create a reverse trade.
                if pos is None or pos.direction == order.side:
                    order.status = OrderStatus.CANCELLED
                    log.debug(
                        "exit_stop.cancelled",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        reason="orphaned",
                    )
                    continue
                # Keep protective stops exit-only even if their qty becomes stale
                # after partial fills or same-bar liquidation paths.
                if order.qty > pos.qty:
                    log.debug(
                        "exit_stop.qty_clamped",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        order_qty=order.qty,
                        position_qty=pos.qty,
                    )
                    order.qty = pos.qty

            fill = self._try_fill_stop(order, bar)
            if fill is not None:
                fills.append(fill)
        return fills

    def _try_fill_stop(self, order: Order, bar: Bar) -> Fill | None:
        """Try to fill a stop order against this bar.

        Gap detection: if bar opens past the stop, fill at open (worse price).
        """
        stop = order.stop_price
        slip_mult = self.slippage_bps / 10_000

        if order.side == Side.SHORT:
            # Sell stop: triggers when price falls to stop level
            gap = bar.open <= stop
            triggered = bar.low <= stop

            if not triggered and not gap:
                return None

            if gap:
                fill_price = bar.open * (1 - slip_mult)
            else:
                fill_price = stop * (1 - slip_mult)

        else:
            # Buy stop: triggers when price rises to stop level
            gap = bar.open >= stop
            triggered = bar.high >= stop

            if not triggered and not gap:
                return None

            if gap:
                fill_price = bar.open * (1 + slip_mult)
            else:
                fill_price = stop * (1 + slip_mult)

        return self._execute_fill(order, fill_price, bar.timestamp, is_taker=True)

    # -------------------------------------------------------------------
    # Stop-limit orders
    # -------------------------------------------------------------------

    def _process_stop_limit_orders(self, bar: Bar) -> list[Fill]:
        """Process stop-limit orders: trigger at stop, then check limit."""
        fills: list[Fill] = []
        for order in self._pending_orders:
            if order.status not in (OrderStatus.PENDING, OrderStatus.WORKING):
                continue
            if order.symbol != bar.symbol:
                continue
            if order.order_type != OrderType.STOP_LIMIT:
                continue
            if order.stop_price is None or order.limit_price is None:
                continue

            fill = self._try_fill_stop_limit(order, bar)
            if fill is not None:
                fills.append(fill)
        return fills

    def _try_fill_stop_limit(self, order: Order, bar: Bar) -> Fill | None:
        """Try to fill a stop-limit order.

        First check if stop is triggered. Then check if the limit price
        is achievable within the bar. If triggered but limit exceeded, REJECT.
        Applies slippage since these are taker fills.
        """
        stop = order.stop_price
        limit = order.limit_price
        slip_mult = self.slippage_bps / 10_000

        if order.side == Side.LONG:
            # Buy stop-limit: stop triggers above stop_price
            gap = bar.open >= stop
            triggered = bar.high >= stop

            if not triggered and not gap:
                return None

            # Triggered — compute fill price with slippage
            base_price = bar.open if gap else stop
            fill_price = base_price * (1 + slip_mult)
            if fill_price > limit:
                order.status = OrderStatus.REJECTED
                log.debug("stop_limit.rejected", order_id=order.order_id, fill_price=fill_price, limit=limit)
                return None

            return self._execute_fill(order, fill_price, bar.timestamp, is_taker=True)

        else:
            # Sell stop-limit: stop triggers below stop_price
            gap = bar.open <= stop
            triggered = bar.low <= stop

            if not triggered and not gap:
                return None

            base_price = bar.open if gap else stop
            fill_price = base_price * (1 - slip_mult)
            if fill_price < limit:
                order.status = OrderStatus.REJECTED
                log.debug("stop_limit.rejected", order_id=order.order_id, fill_price=fill_price, limit=limit)
                return None

            return self._execute_fill(order, fill_price, bar.timestamp, is_taker=True)

    # -------------------------------------------------------------------
    # Market orders
    # -------------------------------------------------------------------

    def _process_market_orders(self, bar: Bar) -> list[Fill]:
        """Process market orders: fill at open ± (spread/2 + slippage)."""
        fills: list[Fill] = []
        for order in self._pending_orders:
            if order.status not in (OrderStatus.PENDING, OrderStatus.WORKING):
                continue
            if order.symbol != bar.symbol:
                continue
            if order.order_type != OrderType.MARKET:
                continue

            cost_bps = (self.spread_bps / 2 + self.slippage_bps) / 10_000

            if order.side == Side.LONG:
                fill_price = bar.open * (1 + cost_bps)
            else:
                fill_price = bar.open * (1 - cost_bps)

            fill = self._execute_fill(order, fill_price, bar.timestamp, is_taker=True)
            if fill is not None:
                fills.append(fill)
        return fills

    # -------------------------------------------------------------------
    # Limit orders
    # -------------------------------------------------------------------

    def _process_limit_orders(self, bar: Bar) -> list[Fill]:
        """Process limit orders with conservative trade-through model.

        Buy limit fills only if bar.low < limit_price (strict <, not <=).
        Sell limit fills only if bar.high > limit_price (strict >).
        Maker fee, no slippage.
        """
        fills: list[Fill] = []
        for order in self._pending_orders:
            if order.status not in (OrderStatus.PENDING, OrderStatus.WORKING):
                continue
            if order.symbol != bar.symbol:
                continue
            if order.order_type != OrderType.LIMIT:
                continue
            if order.limit_price is None:
                continue

            limit = order.limit_price

            if order.side == Side.LONG:
                if bar.low < limit:
                    fill = self._execute_fill(order, limit, bar.timestamp, is_taker=False)
                    if fill:
                        fills.append(fill)
            else:
                if bar.high > limit:
                    fill = self._execute_fill(order, limit, bar.timestamp, is_taker=False)
                    if fill:
                        fills.append(fill)

            # Transition to WORKING after first bar exposure
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.WORKING

        return fills

    # -------------------------------------------------------------------
    # OCA (One-Cancels-All) groups
    # -------------------------------------------------------------------

    def _process_oca_cancels(self, fills: list[Fill]) -> None:
        """Cancel sibling orders in OCA groups when a member fills."""
        filled_groups: set[str] = set()
        for fill in fills:
            # Find the order to get its OCA group
            for order in self._pending_orders:
                if order.order_id == fill.order_id and order.oca_group:
                    if not self._should_cancel_oca_siblings_for_fill(fill, order):
                        continue
                    filled_groups.add(order.oca_group)

        if not filled_groups:
            return

        for order in self._pending_orders:
            if order.oca_group in filled_groups:
                # Don't cancel the one that just filled
                if not any(f.order_id == order.order_id for f in fills):
                    if order.status not in (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.CANCELLED):
                        continue
                    order.status = OrderStatus.CANCELLED
                    order.metadata["cancel_reason"] = "oca_sibling_filled"
                    if not any(cancelled.order_id == order.order_id for cancelled in self._cancelled_oca_orders):
                        self._cancelled_oca_orders.append(deepcopy(order))
                    log.debug("oca.cancelled", order_id=order.order_id, group=order.oca_group)

    def _should_cancel_oca_siblings_for_fill(self, fill: Fill, order: Order) -> bool:
        if not self._uses_terminal_close_oca_policy(order):
            return True
        position = self._positions.get(fill.symbol)
        return position is None or abs(float(position.qty or 0.0)) <= 1e-12

    @staticmethod
    def _uses_terminal_close_oca_policy(order: Order) -> bool:
        metadata = dict(order.metadata or {})
        policy = str(metadata.get("oca_policy") or "")
        if policy == EXIT_OCA_POLICY:
            return True
        if _boolish(metadata.get("reduce_only")) or _boolish(metadata.get("exit_only")):
            return True
        return is_exit_order(order)

    # -------------------------------------------------------------------
    # Funding accrual
    # -------------------------------------------------------------------

    def _process_funding(self, bar: Bar) -> None:
        """Accrue funding at hourly boundaries."""
        if not self._positions:
            return

        bar_hour = bar.timestamp.replace(minute=0, second=0, microsecond=0)

        if bar_hour <= self._last_funding_hour:
            return

        self._last_funding_hour = bar_hour

        for symbol, pos in self._positions.items():
            # Per-symbol funding helper with fallback to global (Finding 6)
            helper = self._funding_helpers.get(symbol) or self.funding_helper
            if helper is not None:
                ts_ms = int(bar_hour.timestamp() * 1000)
                rate = helper.get_rate_at(ts_ms)
            else:
                rate = 0.0

            if rate == 0.0:
                continue

            notional = pos.qty * pos.avg_entry
            sign = 1.0 if pos.direction == Side.LONG else -1.0
            cost = notional * rate * sign

            self._cash -= cost
            pos.realized_pnl -= cost

            self._funding_log.append({
                "timestamp": bar_hour,
                "symbol": symbol,
                "rate": rate,
                "notional": notional,
                "cost": cost,
            })
            log.debug("funding.accrued", symbol=symbol, rate=rate, cost=cost)

    # -------------------------------------------------------------------
    # Equity mark-to-market
    # -------------------------------------------------------------------

    def _record_history_snapshot(
        self,
        history: list[tuple[datetime, float]],
        timestamp: datetime,
        equity: float,
        *,
        replace: bool = False,
    ) -> None:
        if replace and history and history[-1][0] == timestamp:
            history[-1] = (timestamp, equity)
            return
        history.append((timestamp, equity))

    def _record_equity_snapshot(self, timestamp: datetime, equity: float, *, replace: bool = False) -> None:
        self._record_history_snapshot(self._equity_history, timestamp, equity, replace=replace)

    def _record_liquidation_equity_snapshot(
        self,
        timestamp: datetime,
        equity: float,
        *,
        replace: bool = False,
    ) -> None:
        self._record_history_snapshot(
            self._liquidation_equity_history,
            timestamp,
            equity,
            replace=replace,
        )

    def _mark_to_market_equity(self) -> float:
        """Compute current equity from cash and raw mark-to-market prices."""
        position_value = 0.0
        for symbol, pos in self._positions.items():
            price = self._last_prices.get(symbol, pos.avg_entry)
            if pos.direction == Side.LONG:
                pos.unrealized_pnl = (price - pos.avg_entry) * pos.qty
            else:
                pos.unrealized_pnl = (pos.avg_entry - price) * pos.qty
            leverage = pos.leverage or self.default_leverage
            margin = pos.qty * pos.avg_entry / leverage
            position_value += margin + pos.unrealized_pnl
        return self._cash + position_value

    def _mark_to_market_liquidation_equity(self) -> float:
        """Compute equity assuming open positions are liquidated at the current net mark."""
        liquidation_equity = self._cash
        for symbol, pos in self._positions.items():
            raw_price = self._last_prices.get(symbol, pos.avg_entry)
            net_price = self._terminal_liquidation_price(pos, raw_price)
            leverage = pos.leverage or self.default_leverage
            margin = pos.qty * pos.avg_entry / leverage
            if pos.direction == Side.LONG:
                unrealized_pnl = (net_price - pos.avg_entry) * pos.qty
            else:
                unrealized_pnl = (pos.avg_entry - net_price) * pos.qty
            exit_commission = pos.qty * net_price * self.taker_fee_bps / 10_000
            liquidation_equity += margin + unrealized_pnl - exit_commission
        return liquidation_equity

    def _update_equity(self, bar: Bar) -> None:
        """Update equity to reflect margin + unrealized PnL at current prices."""
        self._equity = self._mark_to_market_equity()
        liquidation_equity = self._mark_to_market_liquidation_equity()
        self._record_equity_snapshot(bar.timestamp, self._equity)
        self._record_liquidation_equity_snapshot(bar.timestamp, liquidation_equity)

    def refresh_current_bar_equity(self, timestamp: datetime) -> None:
        """Rewrite the current bar snapshot after same-bar closes or liquidations."""
        self._equity = self._mark_to_market_equity()
        liquidation_equity = self._mark_to_market_liquidation_equity()
        self._record_equity_snapshot(timestamp, self._equity, replace=True)
        self._record_liquidation_equity_snapshot(timestamp, liquidation_equity, replace=True)

    # -------------------------------------------------------------------
    # Liquidation
    # -------------------------------------------------------------------

    def _check_liquidations(self, bar: Bar) -> list[Fill]:
        """Force-close positions that breach maintenance margin.

        Uses bar.low for longs and bar.high for shorts (worst-case intra-bar
        adverse price) instead of bar.close via _last_prices (Finding 3b).
        """
        fills: list[Fill] = []
        symbols_to_close = []

        for symbol, pos in self._positions.items():
            if bar.symbol != symbol:
                continue  # Only check positions matching this bar's symbol
            # Worst-case intra-bar price for liquidation check
            if pos.direction == Side.LONG:
                adverse_price = bar.low
            else:
                adverse_price = bar.high
            notional = pos.qty * adverse_price
            leverage = pos.leverage or self.default_leverage
            margin_used = notional / leverage

            # Get maintenance margin from tiers if available
            maintenance = self._get_maintenance_margin(symbol, notional)

            # Unrealized PnL at adverse price
            if pos.direction == Side.LONG:
                adverse_upnl = (adverse_price - pos.avg_entry) * pos.qty
            else:
                adverse_upnl = (pos.avg_entry - adverse_price) * pos.qty

            # Liquidation: margin_used + unrealized_pnl <= maintenance
            if margin_used + adverse_upnl <= maintenance:
                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            pos = self._positions[symbol]
            # Liquidation fill at the adverse extreme
            if pos.direction == Side.LONG:
                price = bar.low
            else:
                price = bar.high
            log.info("liquidation", symbol=symbol, qty=pos.qty, entry=pos.avg_entry, price=price)

            # Force close at current price with taker fee
            close_side = Side.SHORT if pos.direction == Side.LONG else Side.LONG
            liq_order = Order(
                order_id=str(self._next_order_id),
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                qty=pos.qty,
                tag="liquidation",
            )
            self._next_order_id += 1
            fill = self._execute_fill(liq_order, price, bar.timestamp, is_taker=True)
            if fill:
                fills.append(fill)

        return fills

    def _get_maintenance_margin(self, symbol: str, notional: float) -> float:
        """Get maintenance margin from asset meta tiers, or default 0.5%."""
        if self.asset_meta and symbol in self.asset_meta.margin_tiers:
            tiers = self.asset_meta.margin_tiers[symbol]
            for tier in tiers:
                if notional <= tier.max_notional:
                    return notional * tier.maintenance_margin
            # Past all tiers — use the last tier's rate
            if tiers:
                return notional * tiers[-1].maintenance_margin
        # Default: 0.5% maintenance margin
        return notional * 0.005

    # -------------------------------------------------------------------
    # Fill execution + position lifecycle
    # -------------------------------------------------------------------

    def _execute_fill(
        self,
        order: Order,
        fill_price: float,
        timestamp: datetime,
        is_taker: bool,
    ) -> Fill | None:
        """Execute a fill: create Fill, update position, compute PnL if closing."""
        fee_bps = self.taker_fee_bps if is_taker else self.maker_fee_bps
        commission = order.qty * fill_price * fee_bps / 10_000

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            fill_price=fill_price,
            commission=commission,
            timestamp=timestamp,
            tag=order.tag,
        )

        order.status = OrderStatus.FILLED
        self._fills.append(fill)
        self._cash -= commission

        # Update position
        self._apply_fill_to_position(fill)

        log.debug(
            "fill.executed",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            price=fill_price,
            commission=commission,
        )

        return fill

    def _resolve_leverage(self, fill: Fill) -> float:
        """Extract leverage from the order metadata, or fall back to default."""
        for order in self._pending_orders:
            if order.order_id == fill.order_id and order.metadata:
                return order.metadata.get("leverage", self.default_leverage)
        # Also check deferred orders
        for order in self._deferred_orders:
            if order.order_id == fill.order_id and order.metadata:
                return order.metadata.get("leverage", self.default_leverage)
        return self.default_leverage

    def _apply_fill_to_position(self, fill: Fill) -> None:
        """Update or create position based on fill. Creates Trade on close."""
        symbol = fill.symbol
        fill_direction = Side.LONG if fill.side == Side.LONG else Side.SHORT
        existing = self._positions.get(symbol)

        if existing is None:
            # New position — use leverage from order metadata (Finding 3)
            leverage = self._resolve_leverage(fill)
            self._positions[symbol] = Position(
                symbol=symbol,
                direction=fill_direction,
                qty=fill.qty,
                avg_entry=fill.fill_price,
                open_time=fill.timestamp,
                leverage=leverage,
            )
            # Deduct margin (notional / leverage) from cash
            self._cash -= fill.qty * fill.fill_price / leverage
            return

        if existing.direction == fill_direction:
            # Adding to position — weighted average entry
            leverage = existing.leverage or self.default_leverage
            old_notional = existing.qty * existing.avg_entry
            new_notional = fill.qty * fill.fill_price
            existing.avg_entry = (old_notional + new_notional) / (existing.qty + fill.qty)
            existing.qty += fill.qty
            # Deduct additional margin
            self._cash -= fill.qty * fill.fill_price / leverage
        else:
            # Reducing or closing — opposite direction fill
            leverage = existing.leverage or self.default_leverage
            close_qty = min(fill.qty, existing.qty)

            # Compute PnL on the closed portion
            if existing.direction == Side.LONG:
                pnl = close_qty * (fill.fill_price - existing.avg_entry)
            else:
                pnl = close_qty * (existing.avg_entry - fill.fill_price)

            # Return margin + PnL to cash
            self._cash += close_qty * existing.avg_entry / leverage + pnl

            remaining = existing.qty - close_qty

            if remaining <= 1e-12:
                # Position fully closed — create Trade
                # Pass raw closing PnL; _create_trade adds accumulated funding from realized_pnl
                self._create_trade(existing, fill, close_qty, pnl)
                del self._positions[symbol]

                # Cancel remaining orders for this symbol to prevent orphaned
                # fills (e.g. TP1 MARKET filling after trail stop already closed
                # the position, creating an unintended reverse position).
                for o in self._pending_orders:
                    if (o.symbol == symbol
                            and o.status in (OrderStatus.PENDING, OrderStatus.WORKING)):
                        o.status = OrderStatus.CANCELLED

                # Handle overshoot: if fill.qty > existing.qty, open reverse position
                overshoot = fill.qty - close_qty
                if overshoot > 1e-12:
                    overshoot_leverage = self._resolve_leverage(fill)
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        direction=fill_direction,
                        qty=overshoot,
                        avg_entry=fill.fill_price,
                        open_time=fill.timestamp,
                        leverage=overshoot_leverage,
                    )
                    self._cash -= overshoot * fill.fill_price / overshoot_leverage
            else:
                existing.partial_exit_pnl += pnl
                existing.partial_exit_commission += fill.commission
                existing.partial_exit_qty += close_qty
                existing.qty = remaining

    def _create_trade(self, pos: Position, exit_fill: Fill, qty: float, pnl: float) -> None:
        """Create a completed Trade record from a closed position."""
        # Compute total commission: sum of all fills for this symbol during this position
        entry_commission = self._compute_entry_commission(pos)
        total_commission = entry_commission + exit_fill.commission

        # Bars held is tracked per-position
        bars_held = self._bar_count_per_position.pop(pos.symbol, 0)

        # Funding paid: realized_pnl accumulates funding costs via -= cost
        # So negative realized_pnl means funding was paid out, positive means received
        funding_paid = -pos.realized_pnl
        total_closed_qty = pos.partial_exit_qty + qty
        price_pnl_gross = pnl + pos.partial_exit_pnl
        avg_exit_price = exit_fill.fill_price
        if total_closed_qty > 0:
            if pos.direction == Side.LONG:
                avg_exit_price = pos.avg_entry + price_pnl_gross / total_closed_qty
            else:
                avg_exit_price = pos.avg_entry - price_pnl_gross / total_closed_qty

        trade = Trade(
            trade_id=str(self._trade_id),
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.avg_entry,
            exit_price=avg_exit_price,
            qty=total_closed_qty,
            entry_time=pos.open_time or exit_fill.timestamp,
            exit_time=exit_fill.timestamp,
            pnl=price_pnl_gross + pos.realized_pnl,
            r_multiple=None,
            commission=total_commission + pos.partial_exit_commission,
            bars_held=bars_held,
            setup_grade=None,
            exit_reason=exit_fill.tag or "close",
            confluences_used=None,
            confirmation_type=None,
            entry_method=None,
            funding_paid=funding_paid,
            mae_r=None,
            mfe_r=None,
        )
        self._trade_id += 1
        self._closed_trades.append(trade)
        log.debug("trade.closed", trade_id=trade.trade_id, symbol=trade.symbol, pnl=trade.pnl)

    def _terminal_liquidation_price(self, pos: Position, raw_price: float) -> float:
        """Apply the same spread/slippage assumptions as a market exit."""
        cost_bps = (self.spread_bps / 2 + self.slippage_bps) / 10_000
        if pos.direction == Side.LONG:
            return raw_price * (1 - cost_bps)
        return raw_price * (1 + cost_bps)

    def _build_terminal_mark(self, symbol: str, pos: Position, timestamp: datetime) -> tuple[TerminalMark, float]:
        raw_price = self._last_prices.get(symbol, pos.avg_entry)
        net_price = self._terminal_liquidation_price(pos, raw_price)
        leverage = pos.leverage or self.default_leverage
        remaining_margin = pos.qty * pos.avg_entry / leverage
        if pos.direction == Side.LONG:
            remaining_pnl = pos.qty * (net_price - pos.avg_entry)
        else:
            remaining_pnl = pos.qty * (pos.avg_entry - net_price)
        exit_commission = pos.qty * net_price * self.taker_fee_bps / 10_000
        entry_commission = self._compute_entry_commission(pos)

        marked_pnl_net = (
            pos.partial_exit_pnl
            + pos.realized_pnl
            + remaining_pnl
            - entry_commission
            - pos.partial_exit_commission
            - exit_commission
        )
        equity_contribution = remaining_margin + remaining_pnl - exit_commission

        mark = TerminalMark(
            symbol=symbol,
            direction=pos.direction,
            qty=pos.qty,
            timestamp=timestamp,
            entry_price=pos.avg_entry,
            mark_price_raw=raw_price,
            mark_price_net_liquidation=net_price,
            unrealized_pnl_net=marked_pnl_net,
            unrealized_r_at_mark=None,
            leverage=pos.leverage,
            liquidation_price=pos.liquidation_price,
        )
        return mark, equity_contribution

    def mark_open_positions(self) -> list[TerminalMark]:
        """Create explicit terminal marks and append a final net-liquidation snapshot."""
        self._terminal_marks = []
        if not self._equity_history and not self._positions:
            return []

        timestamp = self._last_bar.timestamp if self._last_bar else datetime.now(timezone.utc)
        final_liquidation_equity = self._cash
        for symbol, pos in self._positions.items():
            mark, contribution = self._build_terminal_mark(symbol, pos, timestamp)
            self._terminal_marks.append(mark)
            final_liquidation_equity += contribution

        self._record_liquidation_equity_snapshot(
            timestamp,
            final_liquidation_equity,
            replace=bool(
                self._liquidation_equity_history
                and self._liquidation_equity_history[-1][0] == timestamp
            ),
        )
        return list(self._terminal_marks)

    def close_open_positions(self) -> list[Fill]:
        """Force-close all open positions at last known prices (backtest end).

        Applies spread/slippage cost to match normal market order fills (Finding 4a).
        """
        fills = []
        for symbol in list(self._positions.keys()):
            pos = self._positions[symbol]
            raw_price = self._last_prices.get(symbol, pos.avg_entry)
            close_side = Side.SHORT if pos.direction == Side.LONG else Side.LONG
            price = self._terminal_liquidation_price(pos, raw_price)
            order = Order(
                order_id=str(self._next_order_id),
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                qty=pos.qty,
                tag="backtest_end",
            )
            self._next_order_id += 1
            ts = self._last_bar.timestamp if self._last_bar else datetime.now(timezone.utc)
            fill = self._execute_fill(order, price, ts, is_taker=True)
            if fill:
                fills.append(fill)
        return fills

    def _compute_entry_commission(self, pos: Position) -> float:
        """Sum commissions from entry fills for this position."""
        if pos.open_time is None:
            return 0.0
        total = 0.0
        entry_side = pos.direction
        for fill in self._fills:
            if fill.symbol == pos.symbol and fill.side == entry_side and fill.timestamp >= pos.open_time:
                total += fill.commission
        return total
