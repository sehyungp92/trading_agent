"""IBKRExecutionAdapter - sole public interface for OMS.

Merged adapter supporting both calling conventions:
  - swing/momentum style: submit_order(oms_order_id, contract_symbol, contract_expiry, ...)
  - stock style: submit_order(oms_order_id, instrument=inst, ...)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from ib_async import Trade

from ..error_map import classify_error
from ..session import UnifiedIBSession
from ..throttler import PacingChannel
from ..logging.audit import log_broker_command, log_broker_response
from ..logging.trace_ids import generate_trace_id
from ..mapping.contract_factory import ContractFactory
from ..mapping.order_mapper import OrderMapper
from ..models.types import (
    BrokerOrderRef,
    ExecutionReport,
    IBContractSpec,
    OrderStatusEvent,
    PositionSnapshot,
)
from ..reconciler.snapshots import SnapshotFetcher
from ..risk_support.tick_rules import round_to_tick
from ..state.cache import IBCache

if TYPE_CHECKING:
    from libs.oms.models.instrument import Instrument

logger = logging.getLogger(__name__)


class OrderNotFoundError(Exception):
    pass


class IBKRExecutionAdapter:
    """Narrow interface between OMS and IBKR.

    OMS should only import this class + typed models.

    Supports two submit_order signatures:
    - Legacy (swing/momentum): contract_symbol + contract_expiry strings
    - Rich (stock): instrument: Instrument object
    Both resolve through ContractFactory.resolve().
    """

    def __init__(
        self, session: UnifiedIBSession, contract_factory: ContractFactory, account: str
    ):
        self._session = session
        self._factory = contract_factory
        self._account = account
        self._cache = IBCache()
        self._snapshots = SnapshotFetcher(session.ib)

        # Callbacks — OMS registers these
        self.on_ack: Callable[[str, BrokerOrderRef], None] = lambda *a: None
        self.on_reject: Callable[[str, str, int, bool], None] = lambda *a: None
        self.on_fill: Callable[
            [str, str, float, float, datetime, float], None
        ] = lambda *a: None
        self.on_status: Callable[[str, str, float], None] = lambda *a: None
        self.on_positions_snapshot: Callable[
            [list[PositionSnapshot]], None
        ] = lambda *a: None

        # Wire IB events
        self._session.ib.orderStatusEvent += self._handle_order_status
        self._session.ib.execDetailsEvent += self._handle_exec_details
        self._session.ib.errorEvent += self._handle_error

    async def submit_order(
        self,
        oms_order_id: str,
        contract_symbol: str = "",
        contract_expiry: str = "",
        action: str = "",
        order_type: str = "",
        qty: int = 0,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: str = "DAY",
        oca_group: str = "",
        oca_type: int = 0,
        client_order_id: str | None = None,
        instrument: Instrument | None = None,
    ) -> BrokerOrderRef:
        """Submit an order to IBKR. Returns BrokerOrderRef on successful transmission.

        Two calling conventions:
        - Legacy: submit_order(id, contract_symbol="MNQ", contract_expiry="202506", action=..., ...)
        - Rich:   submit_order(id, instrument=inst, action=..., ...)
        """
        trace_id = generate_trace_id(oms_order_id)
        await self._session.throttled(PacingChannel.ORDERS)

        # Resolve contract: instrument path vs symbol/expiry path
        if instrument is not None:
            symbol = instrument.root or instrument.symbol
            expiry = instrument.contract_expiry
            contract, spec = await self._factory.resolve(
                symbol=symbol,
                expiry=expiry,
                instrument=instrument,
            )
            log_symbol = instrument.symbol
            log_extra = {"sec_type": instrument.sec_type}
        else:
            contract, spec = await self._factory.resolve(contract_symbol, contract_expiry)
            log_symbol = contract_symbol
            log_extra = {}

        # Round prices to tick
        if limit_price is not None:
            limit_price = round_to_tick(limit_price, spec.tick_size)
        if stop_price is not None:
            stop_price = round_to_tick(stop_price, spec.tick_size)

        ib_order = OrderMapper.to_ib_order(
            action=action,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            account=self._account,
            oca_group=oca_group,
            oca_type=oca_type,
            order_ref=client_order_id or oms_order_id,
        )

        log_broker_command(
            trace_id,
            oms_order_id,
            "submit",
            {
                "symbol": log_symbol,
                **log_extra,
                "action": action,
                "type": order_type,
                "qty": qty,
                "limit": limit_price,
                "stop": stop_price,
            },
        )

        trade = self._session.ib.placeOrder(contract, ib_order)
        ref = BrokerOrderRef(
            broker_order_id=trade.order.orderId,
            perm_id=trade.order.permId,
            con_id=contract.conId,
        )
        self._cache.register_order(oms_order_id, ref.broker_order_id, ref.perm_id)
        self._cache.contracts[spec.con_id] = spec
        return ref

    async def cancel_order(self, broker_order_id: int, perm_id: int = 0) -> None:
        """Cancel an order by broker order ID."""
        await self._session.throttled(PacingChannel.ORDERS)
        oms_id = self._cache.lookup_oms_id(broker_order_id)
        trace_id = generate_trace_id(oms_id or "")
        log_broker_command(
            trace_id,
            oms_id or "",
            "cancel",
            {"broker_order_id": broker_order_id},
        )
        for trade in self._session.ib.openTrades():
            if trade.order.orderId == broker_order_id:
                self._session.ib.cancelOrder(trade.order)
                return
        logger.warning(
            f"Order {broker_order_id} not found in open trades for cancel"
        )

    async def replace_order(
        self,
        broker_order_id: int,
        new_qty: int | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> BrokerOrderRef:
        """Modify an order (IB cancel/replace)."""
        await self._session.throttled(PacingChannel.ORDERS)
        for trade in self._session.ib.openTrades():
            if trade.order.orderId == broker_order_id:
                if new_qty is not None:
                    trade.order.totalQuantity = new_qty
                if new_limit_price is not None:
                    trade.order.lmtPrice = new_limit_price
                if new_stop_price is not None:
                    trade.order.auxPrice = new_stop_price
                self._session.ib.placeOrder(trade.contract, trade.order)
                return BrokerOrderRef(broker_order_id, trade.order.permId, 0)
        raise OrderNotFoundError(f"Order {broker_order_id} not found")

    async def request_open_orders(self) -> list[OrderStatusEvent]:
        await self._session.throttled(PacingChannel.ORDERS)
        return await self._snapshots.fetch_open_orders()

    async def request_positions(self) -> list[PositionSnapshot]:
        await self._session.throttled(PacingChannel.ORDERS)
        positions = await self._snapshots.fetch_positions()
        for position in positions:
            self._cache.contracts.setdefault(
                position.con_id,
                IBContractSpec(
                    con_id=position.con_id,
                    symbol=position.symbol,
                    sec_type="",
                    exchange="",
                    currency="USD",
                    multiplier=0.0,
                    tick_size=0.0,
                    trading_class="",
                    last_trade_date="",
                ),
            )
        return positions

    async def request_executions(
        self, since_ts: datetime | None = None
    ) -> list[ExecutionReport]:
        await self._session.throttled(PacingChannel.ORDERS)
        return await self._snapshots.fetch_executions(since_ts)

    async def rebuild_cache(
        self,
        oms_order_id_resolver,
        fill_exists_check=None,
        fill_importer=None,
    ) -> None:
        """Restore broker/OMS order mappings after a restart.

        OMS-3: passes fill_exists_check + fill_importer through so the cache
        can import any broker executions missing from OMS state before marking
        their exec_ids seen. Without this, a fill that occurred while the
        runtime was down would be permanently dropped on the next reconnect.
        """
        await self._cache.rebuild_from_broker(
            self._snapshots,
            oms_order_id_resolver=oms_order_id_resolver,
            fill_exists_check=fill_exists_check,
            fill_importer=fill_importer,
        )

    @property
    def is_ready(self) -> bool:
        return self._session.is_ready

    @property
    def is_congested(self) -> bool:
        return self._session.is_congested

    @property
    def cache(self) -> IBCache:
        return self._cache

    @property
    def account_id(self) -> str:
        return self._account

    @property
    def client_id(self) -> int | None:
        first_group = next(iter(getattr(self._session, "groups", {}).values()), None)
        config = getattr(first_group, "config", None)
        value = getattr(config, "client_id", None)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _handle_order_status(self, trade: Trade) -> None:
        """Route IB orderStatus events to OMS callbacks."""
        oms_id = self._cache.lookup_oms_id(trade.order.orderId)
        if not oms_id:
            return

        status = trade.orderStatus.status
        trace_id = generate_trace_id(oms_id)
        log_broker_response(
            trace_id,
            oms_id,
            "status",
            {
                "status": status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
            },
        )

        if status == "PreSubmitted":
            ref = BrokerOrderRef(trade.order.orderId, trade.order.permId, 0)
            self.on_ack(oms_id, ref)
            self._cache.mark_acked(oms_id)
        elif status == "Submitted":
            # Only emit on_ack for Submitted if we haven't already from PreSubmitted
            if not self._cache.is_acked(oms_id):
                ref = BrokerOrderRef(trade.order.orderId, trade.order.permId, 0)
                self.on_ack(oms_id, ref)
            self.on_status(oms_id, status, trade.orderStatus.remaining)
        elif status == "Cancelled":
            self.on_status(oms_id, "Cancelled", trade.orderStatus.remaining)
        elif status == "Inactive":
            self.on_reject(oms_id, "Order inactive", 0, False)
        elif status == "Filled":
            self.on_status(oms_id, status, trade.orderStatus.remaining)
        else:
            self.on_status(oms_id, status, trade.orderStatus.remaining)

    def _handle_exec_details(self, trade: Trade, fill) -> None:
        """Route IB execution/fill events to OMS callbacks."""
        exec_id = fill.execution.execId
        if self._cache.is_fill_seen(exec_id):
            return

        oms_id = self._cache.lookup_oms_id(trade.order.orderId)
        if not oms_id:
            logger.warning(
                "Received execution %s for unmapped broker_order_id=%s; leaving unmarked for replay",
                exec_id,
                trade.order.orderId,
            )
            return

        trace_id = generate_trace_id(oms_id)
        commission = (
            fill.commissionReport.commission if fill.commissionReport else 0.0
        )
        log_broker_response(
            trace_id,
            oms_id,
            "fill",
            {
                "exec_id": exec_id,
                "price": fill.execution.price,
                "qty": fill.execution.shares,
                "commission": commission,
            },
        )

        self.on_fill(
            oms_id,
            exec_id,
            fill.execution.price,
            fill.execution.shares,
            fill.execution.time,
            commission,
        )
        self._cache.mark_fill_seen(exec_id)

    def _handle_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Route IB errors to OMS callbacks."""
        oms_id = self._cache.lookup_oms_id(reqId)
        if oms_id:
            category, retryable = classify_error(errorCode, errorString)
            self.on_reject(oms_id, errorString, errorCode, retryable)
