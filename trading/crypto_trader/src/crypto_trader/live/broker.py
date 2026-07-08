"""Hyperliquid broker adapter implementing BrokerAdapter protocol."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import structlog

from crypto_trader.core.models import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from crypto_trader.core.order_semantics import STOP_LOSS_TRIGGER_TAGS
from crypto_trader.exchange.precision import round_price, round_size

log = structlog.get_logger()

# Default lot/tick sizes (fetched from exchange at init in production)
_DEFAULT_LOT_SIZE = {
    "BTC": 0.001,
    "ETH": 0.01,
    "SOL": 0.1,
}

_DEFAULT_TICK_SIZE = {
    "BTC": 0.1,
    "ETH": 0.01,
    "SOL": 0.001,
}


class HyperliquidBroker:
    """BrokerAdapter implementation for Hyperliquid perpetual futures.

    Uses the hyperliquid-python-sdk for REST-based order management.
    Supports both testnet and mainnet.
    """

    def __init__(
        self,
        wallet_address: str,
        private_key: str | None = None,
        is_testnet: bool = True,
        max_slippage_pct: float = 0.005,
        lot_sizes: dict[str, float] | None = None,
        tick_sizes: dict[str, float] | None = None,
        rate_limit_per_sec: float = 5.0,
    ) -> None:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        self._address = wallet_address
        self._private_key = private_key
        self._is_testnet = is_testnet
        self._max_slippage_pct = max_slippage_pct
        self._lot_sizes = lot_sizes or dict(_DEFAULT_LOT_SIZE)
        self._tick_sizes = tick_sizes or dict(_DEFAULT_TICK_SIZE)

        base_url = constants.TESTNET_API_URL if is_testnet else constants.MAINNET_API_URL
        self._info = Info(base_url, skip_ws=True)

        self._exchange = None
        if private_key is not None:
            from hyperliquid.exchange import Exchange
            from eth_account import Account
            wallet = Account.from_key(private_key)
            self._exchange = Exchange(wallet, base_url)

        # Order tracking
        self._orders: dict[str, Order] = {}  # local_id -> Order
        self._oid_map: dict[str, str] = {}   # exchange_oid -> local_id
        self._local_to_oid: dict[str, str] = {}  # local_id -> exchange_oid
        self._next_local_order_seq = 1

        # Rate limiting
        self._last_request_time = 0.0
        self._rate_limit_interval = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0.2

        log.info(
            "broker.init",
            address=wallet_address[:8] + "...",
            testnet=is_testnet,
        )

    def submit_order(self, order: Order) -> str:
        """Submit an order to Hyperliquid. Returns local order_id."""
        self._ensure_local_order_id(order)
        if order.oca_group:
            order.metadata["oca_group"] = order.oca_group

        if self._exchange is None:
            log.warning("broker.read_only", msg="Cannot submit orders without private key")
            order.status = OrderStatus.REJECTED
            return order.order_id

        self._rate_limit()

        is_buy = order.side == Side.LONG
        sz = round_size(order.qty, self._lot_sizes.get(order.symbol, 0.001))

        if sz <= 0:
            log.warning("broker.zero_qty", symbol=order.symbol, original_qty=order.qty)
            order.status = OrderStatus.REJECTED
            return order.order_id

        try:
            if order.order_type == OrderType.MARKET:
                result = self._submit_market(order.symbol, is_buy, sz, order)
            elif order.order_type == OrderType.LIMIT:
                result = self._submit_limit(order.symbol, is_buy, sz, order)
            elif order.order_type == OrderType.STOP:
                result = self._submit_stop(order.symbol, is_buy, sz, order)
            else:
                log.warning("broker.unsupported_order_type", order_type=order.order_type.value)
                order.status = OrderStatus.REJECTED
                return order.order_id

            self._process_submit_result(result, order)

        except Exception:
            log.exception("broker.submit_failed", symbol=order.symbol)
            order.status = OrderStatus.REJECTED

        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by local order_id."""
        if self._exchange is None:
            return False

        exchange_oid = self._local_to_oid.get(order_id)
        if exchange_oid is None:
            log.warning("broker.cancel_unknown", order_id=order_id)
            return False

        self._rate_limit()

        try:
            order = self._orders.get(order_id)
            if order is None:
                return False

            result = self._exchange.cancel(order.symbol, int(exchange_oid))

            if _is_success(result):
                order.status = OrderStatus.CANCELLED
                log.debug("broker.cancelled", order_id=order_id, oid=exchange_oid)
                return True

            log.warning("broker.cancel_failed", order_id=order_id, result=result)
            return False

        except Exception:
            log.exception("broker.cancel_failed", order_id=order_id)
            return False

    def cancel_all(self, symbol: str = "") -> int:
        """Cancel all open orders, optionally filtered by symbol."""
        if self._exchange is None:
            return 0

        open_orders = self.get_open_orders(symbol)
        cancelled = 0
        for order in open_orders:
            if self.cancel_order(order.order_id):
                cancelled += 1
        return cancelled

    def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol from exchange."""
        self._rate_limit()

        try:
            state = self._info.user_state(self._address)
            for asset_pos in state.get("assetPositions", []):
                pos = _parse_position(asset_pos)
                if pos is not None and pos.symbol == symbol:
                    return pos
            return None
        except Exception:
            log.exception("broker.get_position_failed", symbol=symbol)
            return None

    def get_positions(self) -> list[Position]:
        """Get all open positions from exchange."""
        self._rate_limit()

        positions = []
        try:
            state = self._info.user_state(self._address)
            for asset_pos in state.get("assetPositions", []):
                pos = _parse_position(asset_pos)
                if pos is not None:
                    positions.append(pos)
        except Exception:
            log.exception("broker.get_positions_failed")

        return positions

    def get_open_orders(self, symbol: str = "") -> list[Order]:
        """Get all open orders from exchange."""
        self._rate_limit()

        orders = []
        try:
            raw_orders = self._info.open_orders(self._address)
            for raw in raw_orders:
                coin = raw.get("coin", "")

                if symbol and coin != symbol:
                    continue

                oid = str(raw.get("oid", ""))
                local_id = self._oid_map.get(oid, oid)
                tracked_order = self._orders.get(local_id)

                side = Side.LONG if raw.get("side", "") == "B" else Side.SHORT
                metadata = (
                    dict(tracked_order.metadata)
                    if tracked_order is not None
                    else {}
                )
                bars_alive = tracked_order._bars_alive if tracked_order is not None else 0
                if tracked_order is not None and tracked_order.ttl_bars is not None:
                    metadata.setdefault("ttl_bars_alive", bars_alive)
                reduce_only = _boolish(raw.get("reduceOnly", raw.get("reduce_only")))
                if reduce_only:
                    metadata["reduce_only"] = True
                oca_group = str(
                    metadata.get("oca_group")
                    or raw.get("ocaGroup")
                    or raw.get("oca_group")
                    or raw.get("cloidGroup")
                    or ""
                )
                if oca_group:
                    metadata["oca_group"] = oca_group

                order = Order(
                    order_id=local_id,
                    symbol=coin,
                    side=side,
                    order_type=tracked_order.order_type if tracked_order is not None else OrderType.LIMIT,
                    qty=float(raw.get("sz", "0")),
                    limit_price=(
                        tracked_order.limit_price
                        if tracked_order is not None
                        else float(raw.get("limitPx", "0"))
                    ),
                    stop_price=tracked_order.stop_price if tracked_order is not None else None,
                    status=OrderStatus.WORKING,
                    tag=tracked_order.tag if tracked_order is not None else "",
                    oca_group=oca_group or (tracked_order.oca_group if tracked_order is not None else None),
                    time_in_force=tracked_order.time_in_force if tracked_order is not None else "GTC",
                    ttl_bars=tracked_order.ttl_bars if tracked_order is not None else None,
                    metadata=metadata,
                    _bars_alive=bars_alive,
                )
                orders.append(order)
                if oid:
                    self._orders.setdefault(local_id, order)
                    self._oid_map.setdefault(oid, local_id)
                    self._local_to_oid.setdefault(local_id, oid)

        except Exception:
            log.exception("broker.get_open_orders_failed")

        return orders

    def get_equity(self) -> float:
        """Get current account equity from exchange."""
        self._rate_limit()

        try:
            state = self._info.user_state(self._address)
            margin = state.get("marginSummary", {})
            return float(margin.get("accountValue", "0"))
        except Exception:
            log.exception("broker.get_equity_failed")
            return 0.0

    def get_fills_since(self, since: datetime) -> list[Fill]:
        """Get fills since a given timestamp."""
        self._rate_limit()

        fills = []
        try:
            start_ms = int(since.timestamp() * 1000)
            raw_fills = self._info.user_fills_by_time(self._address, start_ms)

            for raw in raw_fills:
                coin = raw.get("coin", "")
                side = Side.LONG if raw.get("side", "") == "B" else Side.SHORT
                fill_ts = datetime.fromtimestamp(
                    raw.get("time", 0) / 1000, tz=timezone.utc
                )
                oid = str(raw.get("oid", ""))
                local_id = self._oid_map.get(oid, oid)
                exchange_fill_id = str(
                    raw.get("hash")
                    or raw.get("tid")
                    or raw.get("fillId")
                    or raw.get("id")
                    or ""
                )
                tag = ""
                if local_id in self._orders:
                    tag = self._orders[local_id].tag

                fills.append(Fill(
                    order_id=local_id,
                    symbol=coin,
                    side=side,
                    qty=float(raw.get("sz", "0")),
                    fill_price=float(raw.get("px", "0")),
                    commission=float(raw.get("fee", "0")),
                    timestamp=fill_ts,
                    tag=tag,
                    exchange_order_id=oid,
                    exchange_fill_id=exchange_fill_id,
                    raw=dict(raw),
                ))

        except Exception:
            log.exception("broker.get_fills_failed")

        return fills

    def get_order_owner(self, order_id: str) -> str | None:
        """Get the strategy_id that submitted an order."""
        local_id = self._oid_map.get(str(order_id), order_id)
        order = self._orders.get(local_id)
        if order:
            return order.metadata.get("strategy_id")
        return None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _ensure_local_order_id(self, order: Order) -> None:
        """Guarantee a non-empty local/client order id before any live action."""
        if order.order_id:
            local_id = order.order_id
        else:
            strategy_id = str(order.metadata.get("strategy_id") or "unknown")
            local_id = f"hl_{strategy_id}_{order.symbol}_{self._next_local_order_seq:06d}"
            self._next_local_order_seq += 1
            order.order_id = local_id

        order.metadata["client_order_id"] = str(order.metadata.get("client_order_id") or local_id)

    def _submit_market(self, symbol: str, is_buy: bool, sz: float, order: Order) -> dict:
        """Submit a market order (IOC limit at slippage-adjusted price)."""
        mid_price = self._get_mid_price(symbol)
        if mid_price is None:
            return {"status": "err", "response": "could not get mid price"}

        slippage = 1.0 + self._max_slippage_pct if is_buy else 1.0 - self._max_slippage_pct
        limit_px = round_price(
            mid_price * slippage,
            self._tick_sizes.get(order.symbol, 0.01),
        )

        return self._exchange.order(
            symbol, is_buy, sz, limit_px,
            {"limit": {"tif": "Ioc"}},
            reduce_only=bool(order.metadata.get("reduce_only", False)),
        )

    def _submit_limit(self, symbol: str, is_buy: bool, sz: float, order: Order) -> dict:
        """Submit a GTC limit order."""
        limit_px = round_price(
            order.limit_price or 0,
            self._tick_sizes.get(order.symbol, 0.01),
        )
        return self._exchange.order(
            symbol, is_buy, sz, limit_px,
            {"limit": {"tif": "Gtc"}},
            reduce_only=bool(order.metadata.get("reduce_only", False)),
        )

    def _submit_stop(self, symbol: str, is_buy: bool, sz: float, order: Order) -> dict:
        """Submit a stop-market order (trigger order)."""
        trigger_px = round_price(
            order.stop_price or 0,
            self._tick_sizes.get(order.symbol, 0.01),
        )
        # tpsl: "sl" for stop-loss, "tp" for take-profit
        tpsl = "sl" if order.tag in STOP_LOSS_TRIGGER_TAGS else "tp"
        return self._exchange.order(
            symbol, is_buy, sz, trigger_px,
            {"trigger": {"triggerPx": str(trigger_px), "tpsl": tpsl, "isMarket": True}},
            reduce_only=bool(order.metadata.get("reduce_only", False)),
        )

    def _process_submit_result(self, result: dict, order: Order) -> None:
        """Process the exchange response for order submission."""
        if _is_success(result):
            response = result.get("response", {})
            data = response.get("data", {})
            statuses = data.get("statuses", [])

            if statuses and "resting" in statuses[0]:
                oid = str(statuses[0]["resting"]["oid"])
                order.status = OrderStatus.WORKING
            elif statuses and "filled" in statuses[0]:
                oid = str(statuses[0]["filled"]["oid"])
                order.status = OrderStatus.FILLED
            else:
                oid = order.order_id
                order.status = OrderStatus.WORKING

            self._orders[order.order_id] = order
            self._oid_map[oid] = order.order_id
            self._local_to_oid[order.order_id] = oid

            log.info(
                "broker.order_submitted",
                local_id=order.order_id,
                oid=oid,
                symbol=order.symbol,
                side=order.side.value,
                qty=order.qty,
                type=order.order_type.value,
            )
        else:
            order.status = OrderStatus.REJECTED
            log.warning(
                "broker.order_rejected",
                order_id=order.order_id,
                result=result,
            )

    def _get_mid_price(self, symbol: str) -> float | None:
        """Get mid-market price for a symbol."""
        try:
            all_mids = self._info.all_mids()
            return float(all_mids.get(symbol, 0))
        except Exception:
            log.exception("broker.get_mid_failed", symbol=symbol)
            return None

    def _rate_limit(self) -> None:
        """Simple rate limiter — sleep if needed."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_interval:
            time.sleep(self._rate_limit_interval - elapsed)
        self._last_request_time = time.monotonic()


def _parse_position(asset_pos: dict) -> Position | None:
    """Parse a single position from Hyperliquid user_state response."""
    pos = asset_pos.get("position", {})
    coin = pos.get("coin", "")
    szi = float(pos.get("szi", "0"))
    if szi == 0:
        return None

    direction = Side.LONG if szi > 0 else Side.SHORT
    return Position(
        symbol=coin,
        direction=direction,
        qty=abs(szi),
        avg_entry=float(pos.get("entryPx", "0")),
        unrealized_pnl=float(pos.get("unrealizedPnl", "0")),
        leverage=float(pos.get("leverage", {}).get("value", "1")),
        liquidation_price=_safe_float(pos.get("liquidationPx")),
    )


def _is_success(result: dict) -> bool:
    """Check if an exchange API call was successful."""
    return result.get("status", "") == "ok"


def _safe_float(value: Any) -> float | None:
    """Convert to float, returning None for empty/null values."""
    if value is None or value == "" or value == "null":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False
