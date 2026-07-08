"""Broker state snapshot fetching for reconciliation."""
from datetime import datetime
from ib_async import IB, ExecutionFilter
from ..models.types import (
    BrokerOrderStatus,
    ExecutionReport,
    OrderStatusEvent,
    PositionSnapshot,
)


class SnapshotFetcher:
    """Fetches broker state snapshots for reconciliation."""

    def __init__(self, ib: IB):
        self._ib = ib

    async def fetch_open_orders(self) -> list[OrderStatusEvent]:
        """Request all open orders and return normalized list."""
        trades = await self._ib.reqAllOpenOrdersAsync()
        return [self._normalize_order(t) for t in trades if t.order]

    async def fetch_positions(self) -> list[PositionSnapshot]:
        """Request all positions."""
        positions = await self._ib.reqPositionsAsync()
        return [
            PositionSnapshot(
                account=p.account,
                con_id=p.contract.conId,
                symbol=p.contract.symbol,
                qty=p.position,
                avg_cost=p.avgCost,
            )
            for p in positions
        ]

    async def fetch_executions(
        self, since_ts: datetime | None = None
    ) -> list[ExecutionReport]:
        """Request executions, optionally filtered by time."""
        filt = ExecutionFilter()
        if since_ts:
            filt.time = since_ts.strftime("%Y%m%d %H:%M:%S")
        fills = await self._ib.reqExecutionsAsync(filt)
        return [self._normalize_execution(f) for f in fills]

    def _normalize_order(self, trade) -> OrderStatusEvent:
        status_map = {
            "PendingSubmit": BrokerOrderStatus.PENDING_SUBMIT,
            "PendingCancel": BrokerOrderStatus.PENDING_CANCEL,
            "PreSubmitted": BrokerOrderStatus.PRE_SUBMITTED,
            "Submitted": BrokerOrderStatus.SUBMITTED,
            "Cancelled": BrokerOrderStatus.CANCELLED,
            "Filled": BrokerOrderStatus.FILLED,
            "Inactive": BrokerOrderStatus.INACTIVE,
        }
        os = trade.orderStatus
        return OrderStatusEvent(
            broker_order_id=trade.order.orderId,
            perm_id=trade.order.permId,
            status=status_map.get(os.status, BrokerOrderStatus.INACTIVE),
            filled_qty=os.filled,
            remaining_qty=os.remaining,
            avg_fill_price=os.avgFillPrice,
            last_fill_price=os.lastFillPrice,
            order_ref=str(getattr(trade.order, "orderRef", "") or ""),
            account=str(getattr(trade.order, "account", "") or ""),
            client_id=getattr(trade.order, "clientId", None),
        )

    def _normalize_execution(self, fill) -> ExecutionReport:
        ex = fill.execution
        return ExecutionReport(
            exec_id=ex.execId,
            broker_order_id=ex.orderId,
            perm_id=ex.permId,
            symbol=fill.contract.symbol,
            side=ex.side,
            qty=ex.shares,
            price=ex.price,
            timestamp=ex.time,
            commission=fill.commissionReport.commission if fill.commissionReport else 0.0,
            exchange=ex.exchange,
        )
