"""In-memory repository for development and testing."""
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, AsyncIterator
from typing import Optional

from ..models.fill import Fill
from ..models.instrument import Instrument
from ..models.instrument_registry import InstrumentRegistry
from ..models.order import (
    EntryPolicy,
    OMSOrder,
    OrderRole,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskContext,
)
from ..models.position import Position
from .schema import (
    AdapterStateRow,
    RiskDailyPortfolioRow,
    RiskDailyStrategyRow,
    StrategyStateRow,
    TradeMarksRow,
    TradeRow,
)

logger = logging.getLogger(__name__)

QUEUE_CLAIM_TTL_SECONDS = 30
QUEUE_SUBMIT_INFLIGHT_TTL_SECONDS = 300


class InMemoryRepository:
    """In-memory implementation of OMSRepository for development.
    Same interface as the asyncpg-backed repository.
    """

    def __init__(self):
        self._orders: dict[str, OMSOrder] = {}
        self._events: list[dict] = []
        self._fills: dict[str, Fill] = {}
        self._positions: dict[tuple[str, str, str], Position] = {}  # (strategy, account, symbol)
        self._risk_daily_strategy: dict[tuple[date, str], RiskDailyStrategyRow] = {}
        self._risk_daily_portfolio: dict[tuple[date, str], RiskDailyPortfolioRow] = {}
        self._trades: dict[str, TradeRow] = {}
        self._trade_marks: dict[str, TradeMarksRow] = {}
        self._strategy_state: dict[str, StrategyStateRow] = {}
        self._adapter_state: dict[str, AdapterStateRow] = {}
        self._strategy_signals: dict[str, dict[str, Any]] = {}

    @asynccontextmanager
    async def transaction(self, conn=None) -> AsyncIterator[None]:
        yield None

    async def save_order(self, order: OMSOrder, conn=None) -> None:
        """Upsert current order state."""
        self._orders[order.oms_order_id] = order

    async def save_event(self, oms_order_id: str, event_type: str, payload: dict, conn=None) -> None:
        """Append to order events."""
        self._events.append({
            "oms_order_id": oms_order_id,
            "event_type": event_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc),
        })

    async def save_fill(self, fill: Fill, conn=None) -> bool:
        if fill.broker_fill_id in self._fills:
            return False
        self._fills[fill.broker_fill_id] = fill
        return True

    async def save_order_and_event(
        self,
        order: OMSOrder,
        event_type: str,
        payload: dict,
        conn=None,
    ) -> None:
        await self.save_order(order)
        await self.save_event(order.oms_order_id, event_type, payload)

    async def save_order_fill_and_event(
        self,
        order: OMSOrder,
        fill: Fill,
        event_type: str,
        payload: dict,
    ) -> bool:
        inserted = await self.save_fill(fill)
        if not inserted:
            return False
        await self.save_order(order)
        await self.save_event(order.oms_order_id, event_type, payload)
        return True

    async def fill_exists(self, broker_fill_id: str) -> bool:
        return broker_fill_id in self._fills

    async def get_order(self, oms_order_id: str) -> Optional[OMSOrder]:
        """Load order by ID. Returns None if not found."""
        return self._orders.get(oms_order_id)

    async def get_order_id_by_client_order_id(
        self, strategy_id: str, client_order_id: str
    ) -> Optional[str]:
        """Look up oms_order_id by client_order_id for idempotency."""
        for order in self._orders.values():
            if order.strategy_id == strategy_id and order.client_order_id == client_order_id:
                return order.oms_order_id
        return None

    async def get_order_id_by_broker_order_id(
        self, broker_order_id: int
    ) -> Optional[str]:
        """Resolve an OMS order ID from a persisted broker order ID."""
        for order in self._orders.values():
            if str(order.broker_order_id or "") == str(broker_order_id):
                return order.oms_order_id
        return None

    async def get_pending_entry_risk_R(self, unit_risk_dollars: float) -> float:
        """Sum risk_R of working ENTRY orders."""
        working_statuses = {
            OrderStatus.RISK_APPROVED,
            OrderStatus.QUEUED,
            OrderStatus.ROUTED,
            OrderStatus.ACKED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
        }
        total_risk = 0.0
        for order in self._orders.values():
            if order.role == OrderRole.ENTRY and order.status in working_statuses:
                if order.risk_context:
                    risk = order.risk_context.risk_dollars or 0.0
                    # Scale by remaining qty for partially filled orders
                    if order.status == OrderStatus.PARTIALLY_FILLED:
                        qty = order.qty or 1
                        remaining = order.remaining_qty or 0
                        risk = risk * (remaining / qty) if qty > 0 else 0.0
                    total_risk += risk
        return total_risk / unit_risk_dollars if unit_risk_dollars > 0 else 0.0

    async def get_working_orders(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[OMSOrder]:
        """Get all non-terminal orders for a strategy."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.DONE,
        }
        result = []
        for order in self._orders.values():
            if order.strategy_id == strategy_id and order.status not in terminal:
                if instrument_symbol is None or (
                    order.instrument and order.instrument.symbol == instrument_symbol
                ):
                    result.append(order)
        return result

    async def count_working_orders(self, strategy_id: str) -> int:
        """Count non-terminal orders for a strategy."""
        orders = await self.get_working_orders(strategy_id)
        return len(orders)

    async def get_positions(
        self, strategy_id: str, instrument_symbol: str = None
    ) -> list[Position]:
        result = []
        for key, pos in self._positions.items():
            strat, _, symbol = key
            if strat == strategy_id:
                if instrument_symbol is None or symbol == instrument_symbol:
                    result.append(pos)
        return result

    async def save_position(self, position: Position) -> None:
        """Upsert position."""
        key = (position.strategy_id, position.account_id, position.instrument_symbol)
        self._positions[key] = position

    async def get_all_working_orders(self) -> list[OMSOrder]:
        """Get all non-terminal orders across all strategies."""
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.DONE,
        }
        return [o for o in self._orders.values() if o.status not in terminal]

    async def get_all_positions(self) -> list[Position]:
        """Get all positions across all strategies."""
        return list(self._positions.values())

    async def get_positions_for_strategies(
        self, strategy_ids: list[str],
    ) -> list[Position]:
        """Get positions for specific strategies only (family-scoped)."""
        ids = set(strategy_ids or [])
        if not ids:
            return []
        return [pos for pos in self._positions.values() if pos.strategy_id in ids]

    async def get_pending_entry_risk_R_for_strategies(
        self,
        strategy_ids: list[str],
        unit_risk_dollars: float,
    ) -> float:
        """Sum working ENTRY risk-R for a strategy family."""
        if not strategy_ids or unit_risk_dollars <= 0:
            return 0.0
        risk = self._pending_entry_risk_dollars(strategy_ids=set(strategy_ids))
        return risk / unit_risk_dollars

    async def get_directional_risk_R(self, direction: str) -> float:
        """Sum open risk R for all positions in a given direction."""
        return self._open_position_risk_R(direction=direction)

    async def get_directional_risk_R_for_strategies(
        self,
        direction: str,
        strategy_ids: list[str],
    ) -> float:
        """Sum open risk R for a direction, filtered to strategy IDs."""
        if not strategy_ids:
            return 0.0
        return self._open_position_risk_R(direction=direction, strategy_ids=set(strategy_ids))

    async def get_directional_risk_dollars_for_strategies(
        self,
        direction: str,
        strategy_ids: list[str],
    ) -> float:
        """Sum active risk dollars in a direction, including pending entries."""
        if not strategy_ids:
            return 0.0
        ids = set(strategy_ids)
        return (
            self._open_position_risk_dollars(direction=direction, strategy_ids=ids)
            + self._pending_entry_risk_dollars(direction=direction, strategy_ids=ids)
        )

    async def get_sibling_positions_for_symbol(
        self,
        strategy_ids: list[str],
        symbol: str,
    ) -> bool:
        """Check if any sibling strategy holds an open position in the given symbol."""
        ids = set(strategy_ids or [])
        return any(
            pos.strategy_id in ids
            and pos.instrument_symbol == symbol
            and pos.net_qty != 0
            for pos in self._positions.values()
        )

    async def get_open_position_count_for_strategies(
        self,
        strategy_ids: list[str],
    ) -> int:
        """Count family open positions plus pending entry orders."""
        if not strategy_ids:
            return 0
        ids = set(strategy_ids)
        open_positions = sum(
            1
            for pos in self._positions.values()
            if pos.strategy_id in ids and pos.net_qty != 0
        )
        pending_entries = sum(
            1
            for order in self._orders.values()
            if order.strategy_id in ids
            and order.role == OrderRole.ENTRY
            and order.status in self._working_statuses()
        )
        return open_positions + pending_entries

    async def get_symbol_open_risk_dollars_for_strategies(
        self,
        strategy_ids: list[str],
        symbol: str,
    ) -> float:
        """Sum open risk dollars for one symbol within a strategy family."""
        if not strategy_ids or not symbol:
            return 0.0
        ids = set(strategy_ids)
        return sum(
            float(pos.open_risk_dollars or 0.0)
            for pos in self._positions.values()
            if pos.strategy_id in ids
            and pos.instrument_symbol == symbol
            and pos.net_qty != 0
        )

    async def get_symbols_open_risk_dollars_for_strategies(
        self,
        strategy_ids: list[str],
        symbols: list[str],
    ) -> float:
        """Sum open risk dollars for a set of symbols within a strategy family."""
        if not strategy_ids or not symbols:
            return 0.0
        ids = set(strategy_ids)
        symbol_set = set(symbols)
        return sum(
            float(pos.open_risk_dollars or 0.0)
            for pos in self._positions.values()
            if pos.strategy_id in ids
            and pos.instrument_symbol in symbol_set
            and pos.net_qty != 0
        )

    async def get_active_risk_dollars_for_strategies(
        self,
        strategy_ids: list[str],
    ) -> float:
        """Sum open position risk plus pending entry risk for a family."""
        if not strategy_ids:
            return 0.0
        ids = set(strategy_ids)
        return (
            self._open_position_risk_dollars(strategy_ids=ids)
            + self._pending_entry_risk_dollars(strategy_ids=ids)
        )

    async def get_completed_trade_counts_for_strategies(
        self,
        strategy_ids: list[str],
    ) -> dict[str, int]:
        """Count completed trades per strategy for family balance rules."""
        ids = set(strategy_ids or [])
        counts: dict[str, int] = {}
        for trade in self._trades.values():
            if trade.strategy_id in ids and trade.exit_ts is not None:
                counts[trade.strategy_id] = counts.get(trade.strategy_id, 0) + 1
        return counts

    async def get_recent_strategy_r_multiples(
        self,
        strategy_id: str,
        limit: int,
    ) -> list[float]:
        """Most recent completed trade R values for live dynamic allocation."""
        if not strategy_id or limit <= 0:
            return []
        rows = [
            trade
            for trade in self._trades.values()
            if trade.strategy_id == strategy_id
            and trade.exit_ts is not None
            and trade.realized_r is not None
        ]
        rows.sort(key=lambda trade: trade.exit_ts or datetime.min, reverse=True)
        return [float(trade.realized_r) for trade in rows[:limit]]

    async def get_family_aggregate_mnq_eq(self, strategy_ids: list[str]) -> int:
        """Sum open and pending contracts, converting NQ to 10x MNQ-eq."""
        if not strategy_ids:
            return 0
        ids = set(strategy_ids)
        total = 0
        for pos in self._positions.values():
            if pos.strategy_id in ids and pos.net_qty != 0:
                qty = int(abs(pos.net_qty))
                total += qty * 10 if pos.instrument_symbol == "NQ" else qty
        for order in self._orders.values():
            if (
                order.strategy_id in ids
                and order.role == OrderRole.ENTRY
                and order.status in self._working_statuses()
            ):
                qty = int(order.remaining_qty if order.remaining_qty > 0 else order.qty)
                symbol = self._order_symbol(order)
                total += qty * 10 if symbol == "NQ" else qty
        return total

    async def upsert_strategy_signal(
        self,
        strategy_id: str,
        direction: str,
        entry_ts: datetime,
    ) -> None:
        """Record a strategy's latest entry direction and time."""
        today = entry_ts.date()
        existing = self._strategy_signals.get(strategy_id)
        same_day_count = (
            int(existing.get("daily_entry_count", 0))
            if existing and existing.get("signal_date") == today
            else 0
        )
        self._strategy_signals[strategy_id] = {
            "strategy_id": strategy_id,
            "last_entry_ts": entry_ts,
            "last_direction": direction,
            "daily_entry_count": same_day_count + 1,
            "signal_date": today,
            "chop_score": int(existing.get("chop_score", 0)) if existing else 0,
        }

    async def update_chop_score(self, strategy_id: str, chop_score: int) -> None:
        """Update NQDTC chop score for cross-strategy throttling."""
        existing = self._strategy_signals.get(strategy_id)
        if existing is not None:
            existing["chop_score"] = int(chop_score)

    async def get_strategy_signal(self, strategy_id: str) -> Optional[dict]:
        """Get a strategy's latest signal."""
        signal = self._strategy_signals.get(strategy_id)
        return dict(signal) if signal is not None else None

    async def get_all_strategy_signals(self) -> list[dict]:
        """Get all strategy signals for cross-strategy checks."""
        return [dict(signal) for signal in self._strategy_signals.values()]

    async def upsert_risk_daily_strategy(self, row: RiskDailyStrategyRow) -> None:
        self._risk_daily_strategy[(row.trade_date, row.strategy_id)] = row

    async def get_risk_daily_strategy(
        self,
        strategy_id: str,
        trade_date: date,
    ) -> Optional[RiskDailyStrategyRow]:
        return self._risk_daily_strategy.get((trade_date, strategy_id))

    async def get_risk_daily_strategies_for_date(
        self,
        trade_date: date,
        strategy_ids: list[str] | None = None,
    ) -> list[RiskDailyStrategyRow]:
        ids = set(strategy_ids or [])
        rows = [
            row
            for (row_date, strategy_id), row in self._risk_daily_strategy.items()
            if row_date == trade_date and (not ids or strategy_id in ids)
        ]
        return sorted(rows, key=lambda row: row.strategy_id)

    async def get_risk_daily_strategy_totals(
        self,
        start_date: date,
        end_date: date,
        strategy_ids: list[str] | None = None,
    ) -> dict[str, Decimal]:
        ids = set(strategy_ids or [])
        rows = [
            row
            for (row_date, strategy_id), row in self._risk_daily_strategy.items()
            if start_date <= row_date <= end_date and (not ids or strategy_id in ids)
        ]
        return {
            "total_r": sum((row.daily_realized_r for row in rows), Decimal("0")),
            "total_usd": sum(
                ((row.daily_realized_usd or Decimal("0")) for row in rows),
                Decimal("0"),
            ),
        }

    async def halt_strategy(
        self,
        strategy_id: str,
        reason: str,
        trade_date: date,
    ) -> None:
        key = (trade_date, strategy_id)
        row = self._risk_daily_strategy.get(key)
        if row is not None:
            self._risk_daily_strategy[key] = replace(
                row,
                halted=True,
                halt_reason=reason,
                last_update_at=datetime.now(timezone.utc),
            )

    async def upsert_risk_daily_portfolio(self, row: RiskDailyPortfolioRow) -> None:
        self._risk_daily_portfolio[(row.trade_date, row.family_id)] = row

    async def get_risk_daily_portfolio(
        self,
        trade_date: date,
        family_id: str = "unknown",
    ) -> Optional[RiskDailyPortfolioRow]:
        return self._risk_daily_portfolio.get((trade_date, family_id))

    async def halt_portfolio(
        self,
        reason: str,
        trade_date: date,
        family_id: str = "unknown",
    ) -> None:
        key = (trade_date, family_id)
        row = self._risk_daily_portfolio.get(key)
        if row is not None:
            self._risk_daily_portfolio[key] = replace(
                row,
                halted=True,
                halt_reason=reason,
                last_update_at=datetime.now(timezone.utc),
            )

    async def save_trade(self, row: TradeRow) -> None:
        existing = self._trades.get(row.trade_id)
        if existing is None:
            self._trades[row.trade_id] = row
            return
        self._trades[row.trade_id] = replace(
            existing,
            exit_ts=row.exit_ts,
            exit_price=row.exit_price,
            realized_r=row.realized_r,
            realized_usd=row.realized_usd,
            exit_reason=row.exit_reason,
            notes=row.notes,
            meta_json=row.meta_json,
        )

    async def get_trades_since(self, since: datetime) -> list[TradeRow]:
        rows = [trade for trade in self._trades.values() if trade.entry_ts >= since]
        return sorted(rows, key=lambda trade: trade.entry_ts)

    async def get_open_trades(self) -> list[TradeRow]:
        rows = [trade for trade in self._trades.values() if trade.exit_ts is None]
        return sorted(rows, key=lambda trade: trade.entry_ts)

    async def save_trade_marks(self, row: TradeMarksRow) -> None:
        self._trade_marks[row.trade_id] = row

    async def upsert_strategy_state(self, row: StrategyStateRow) -> None:
        existing = self._strategy_state.get(row.strategy_id)
        if existing is None:
            self._strategy_state[row.strategy_id] = row
            return
        decision_code = row.last_decision_code or existing.last_decision_code
        details_json = row.last_decision_details_json
        if not row.last_decision_code and details_json == "{}":
            details_json = existing.last_decision_details_json
        self._strategy_state[row.strategy_id] = replace(
            existing,
            instance_id=row.instance_id,
            last_heartbeat_ts=row.last_heartbeat_ts,
            mode=row.mode,
            stand_down_reason=row.stand_down_reason,
            last_decision_code=decision_code,
            last_decision_details_json=details_json,
            last_error_ts=row.last_error_ts,
            last_error=row.last_error,
            last_seen_bar_ts=row.last_seen_bar_ts,
            heat_r=row.heat_r,
            daily_pnl_r=row.daily_pnl_r,
        )

    async def record_strategy_decision(
        self,
        strategy_id: str,
        decision_code: str,
        details: Optional[dict] = None,
        last_seen_bar_ts: Optional[datetime] = None,
    ) -> None:
        if not decision_code:
            return
        existing = self._strategy_state.get(strategy_id)
        row = StrategyStateRow(
            strategy_id=strategy_id,
            instance_id=existing.instance_id if existing else "primary",
            last_heartbeat_ts=existing.last_heartbeat_ts if existing else datetime.now(timezone.utc),
            mode=existing.mode if existing else "RUNNING",
            stand_down_reason=existing.stand_down_reason if existing else None,
            last_decision_code=decision_code,
            last_decision_details_json=json.dumps(details or {}, default=str),
            last_error_ts=existing.last_error_ts if existing else None,
            last_error=existing.last_error if existing else None,
            last_seen_bar_ts=last_seen_bar_ts or (existing.last_seen_bar_ts if existing else None),
            heat_r=existing.heat_r if existing else Decimal("0"),
            daily_pnl_r=existing.daily_pnl_r if existing else Decimal("0"),
        )
        self._strategy_state[strategy_id] = row

    async def get_strategy_states(self) -> list[StrategyStateRow]:
        return list(self._strategy_state.values())

    async def mark_order_queued(
        self,
        order_id: str,
        priority: int,
        reason: str,
        queued_at: datetime,
        expires_at: datetime | None,
        conn=None,
    ) -> Optional[OMSOrder]:
        order = self._orders.get(order_id)
        if order is None or order.status not in {OrderStatus.RISK_APPROVED, OrderStatus.QUEUED}:
            return None
        order.status = OrderStatus.QUEUED
        order.queued_at = order.queued_at or queued_at
        order.queue_priority = int(priority)
        order.queue_reason = reason
        order.queue_expires_at = expires_at
        order.queue_claimed_by = ""
        order.queue_claimed_at = None
        order.queue_claim_expires_at = None
        order.last_update_at = queued_at
        await self.save_event(
            order_id,
            "ORDER_QUEUED",
            {
                "priority": priority,
                "reason": reason,
                "queued_at": queued_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )
        return order

    async def claim_queued_orders(
        self,
        limit: int,
        claimant_id: str,
        now: datetime,
        conn=None,
    ) -> list[OMSOrder]:
        if limit <= 0:
            return []
        ready = [
            order
            for order in self._orders.values()
            if order.status == OrderStatus.QUEUED
            and (
                order.submitted_at is None
                or order.queue_claimed_by == ""
            )
            and (
                order.queue_claim_expires_at is None
                or order.queue_claim_expires_at < now
            )
            and (
                order.submitted_at is not None
                or order.queue_expires_at is None
                or order.queue_expires_at > now
            )
        ]
        ready.sort(
            key=lambda order: (
                order.queue_priority if order.queue_priority is not None else 9999,
                order.queued_at or datetime.min.replace(tzinfo=timezone.utc),
            )
        )
        claimed = ready[:limit]
        claim_expires_at = now + timedelta(seconds=QUEUE_CLAIM_TTL_SECONDS)
        for order in claimed:
            order.queue_claimed_by = claimant_id
            order.queue_claimed_at = now
            order.queue_claim_expires_at = claim_expires_at
            order.queue_attempt = int(order.queue_attempt or 0) + 1
            order.last_update_at = now
        return claimed

    async def release_queued_order(
        self,
        order_id: str,
        claimant_id: str,
        conn=None,
    ) -> None:
        order = self._orders.get(order_id)
        if (
            order is not None
            and order.status == OrderStatus.QUEUED
            and order.queue_claimed_by == claimant_id
            and order.submitted_at is None
        ):
            order.queue_claimed_by = ""
            order.queue_claimed_at = None
            order.queue_claim_expires_at = None
            order.last_update_at = datetime.now(timezone.utc)

    async def release_queued_claims(
        self,
        claimant_id: str,
        conn=None,
    ) -> int:
        released = 0
        now = datetime.now(timezone.utc)
        for order in self._orders.values():
            if (
                order.status == OrderStatus.QUEUED
                and order.queue_claimed_by == claimant_id
                and order.submitted_at is None
            ):
                order.queue_claimed_by = ""
                order.queue_claimed_at = None
                order.queue_claim_expires_at = None
                order.last_update_at = now
                released += 1
        return released

    async def clear_queue_claim(self, order_id: str, conn=None) -> None:
        order = self._orders.get(order_id)
        if order is not None:
            order.queue_claimed_by = ""
            order.queue_claimed_at = None
            order.queue_claim_expires_at = None
            order.last_update_at = datetime.now(timezone.utc)

    async def mark_queued_order_expired(
        self,
        order_id: str,
        reason: str,
        conn=None,
    ) -> Optional[OMSOrder]:
        order = self._orders.get(order_id)
        if (
            order is None
            or order.status != OrderStatus.QUEUED
            or order.submitted_at is not None
        ):
            return None
        now = datetime.now(timezone.utc)
        order.status = OrderStatus.EXPIRED
        order.queue_denial_reason = reason
        order.queue_claimed_by = ""
        order.queue_claimed_at = None
        order.queue_claim_expires_at = None
        order.last_update_at = now
        await self.save_event(
            order_id,
            "QUEUED_ORDER_EXPIRED",
            {"reason": reason, "expired_at": now.isoformat()},
        )
        return order

    async def mark_queued_order_dequeued(
        self,
        order_id: str,
        claimant_id: str,
        dequeued_at: datetime,
        conn=None,
    ) -> Optional[OMSOrder]:
        order = self._orders.get(order_id)
        if (
            order is None
            or order.status != OrderStatus.QUEUED
            or order.queue_claimed_by != claimant_id
        ):
            return None
        order.status = OrderStatus.RISK_APPROVED
        order.dequeued_at = dequeued_at
        order.queue_claimed_by = ""
        order.queue_claimed_at = None
        order.queue_claim_expires_at = None
        order.last_update_at = dequeued_at
        await self.save_event(
            order_id,
            "QUEUED_ORDER_DEQUEUED",
            {"claimant_id": claimant_id, "dequeued_at": dequeued_at.isoformat()},
        )
        return order

    async def mark_queued_order_submit_started(
        self,
        order_id: str,
        claimant_id: str,
        started_at: datetime,
        conn=None,
    ) -> Optional[OMSOrder]:
        order = self._orders.get(order_id)
        if (
            order is None
            or order.status != OrderStatus.QUEUED
            or order.queue_claimed_by != claimant_id
        ):
            return None
        order.dequeued_at = order.dequeued_at or started_at
        order.submitted_at = order.submitted_at or started_at
        order.queue_claim_expires_at = started_at + timedelta(
            seconds=QUEUE_SUBMIT_INFLIGHT_TTL_SECONDS
        )
        order.last_update_at = started_at
        await self.save_event(
            order_id,
            "QUEUED_ORDER_SUBMIT_STARTED",
            {"claimant_id": claimant_id, "started_at": started_at.isoformat()},
        )
        return order

    async def mark_queued_order_submitted(
        self,
        order_id: str,
        claimant_id: str,
        broker_order_id: int | str | None,
        perm_id: int | str | None,
        submitted_at: datetime,
        dequeued_at: datetime,
        conn=None,
    ) -> Optional[OMSOrder]:
        if broker_order_id in (None, ""):
            return None
        order = self._orders.get(order_id)
        if (
            order is None
            or order.status != OrderStatus.QUEUED
            or order.queue_claimed_by != claimant_id
            or order.broker_order_id is not None
        ):
            return None
        order.status = OrderStatus.ROUTED
        order.broker_order_id = int(broker_order_id)
        order.perm_id = int(perm_id) if perm_id not in (None, "") else None
        order.dequeued_at = order.dequeued_at or dequeued_at
        order.submitted_at = submitted_at
        order.queue_claimed_by = ""
        order.queue_claimed_at = None
        order.queue_claim_expires_at = None
        order.last_update_at = submitted_at
        await self.save_event(
            order_id,
            "QUEUED_ORDER_SUBMITTED",
            {
                "claimant_id": claimant_id,
                "submitted_at": submitted_at.isoformat(),
                "broker_order_id": str(broker_order_id),
                "perm_id": int(perm_id) if perm_id not in (None, "") else None,
            },
        )
        return order

    async def mark_queued_order_denied(
        self,
        order_id: str,
        claimant_id: str,
        reason: str,
        conn=None,
    ) -> Optional[OMSOrder]:
        order = self._orders.get(order_id)
        if (
            order is None
            or order.status != OrderStatus.QUEUED
            or order.queue_claimed_by != claimant_id
        ):
            return None
        now = datetime.now(timezone.utc)
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        order.queue_denial_reason = reason
        order.queue_claimed_by = ""
        order.queue_claimed_at = None
        order.queue_claim_expires_at = None
        order.last_update_at = now
        await self.save_event(
            order_id,
            "QUEUED_ORDER_DENIED",
            {"claimant_id": claimant_id, "reason": reason, "denied_at": now.isoformat()},
        )
        return order

    async def expire_due_queued_orders(
        self,
        now: datetime | None = None,
    ) -> list[OMSOrder]:
        now = now or datetime.now(timezone.utc)
        expired: list[OMSOrder] = []
        for order in list(self._orders.values()):
            if (
                order.status == OrderStatus.QUEUED
                and order.queue_expires_at is not None
                and order.queue_expires_at <= now
                and order.submitted_at is None
            ):
                marked = await self.mark_queued_order_expired(
                    order.oms_order_id,
                    "queue TTL expired",
                )
                if marked is not None:
                    expired.append(marked)
        return expired

    async def recover_inflight_queued_orders(
        self,
        now: datetime,
        reason: str,
        conn=None,
    ) -> list[OMSOrder]:
        recovered: list[OMSOrder] = []
        for order in list(self._orders.values()):
            if (
                order.queued_at is None
                or order.broker_order_id is not None
            ):
                continue
            if order.status == OrderStatus.QUEUED:
                if (
                    order.submitted_at is None
                    or not order.queue_claimed_by
                    or (
                        order.queue_claim_expires_at is not None
                        and order.queue_claim_expires_at > now
                    )
                ):
                    continue
                order.queue_claimed_by = ""
                order.queue_claimed_at = None
                order.queue_claim_expires_at = None
                order.queue_reason = order.queue_reason or reason
                order.last_update_at = now
                await self.save_event(
                    order.oms_order_id,
                    "QUEUED_ORDER_RECOVERED",
                    {
                        "reason": reason,
                        "recovered_at": now.isoformat(),
                        "submit_inflight": True,
                    },
                )
                recovered.append(order)
                continue
            if order.status not in {OrderStatus.RISK_APPROVED, OrderStatus.ROUTED}:
                continue
            order.queue_claimed_by = ""
            order.queue_claimed_at = None
            order.queue_claim_expires_at = None
            order.last_update_at = now
            if (
                order.submitted_at is None
                and order.queue_expires_at is not None
                and order.queue_expires_at <= now
            ):
                order.status = OrderStatus.EXPIRED
                order.queue_denial_reason = "queue TTL expired during in-flight recovery"
                await self.save_event(
                    order.oms_order_id,
                    "QUEUED_ORDER_EXPIRED",
                    {
                        "reason": "queue TTL expired during in-flight recovery",
                        "expired_at": now.isoformat(),
                    },
                )
            else:
                order.status = OrderStatus.QUEUED
                order.queue_reason = order.queue_reason or reason
                await self.save_event(
                    order.oms_order_id,
                    "QUEUED_ORDER_RECOVERED",
                    {"reason": reason, "recovered_at": now.isoformat()},
                )
            recovered.append(order)
        return recovered

    async def get_queued_order_summary(
        self,
        account_id: str | None = None,
        family_id: str | None = None,
    ) -> dict[str, Any]:
        del family_id
        queued = [
            order
            for order in self._orders.values()
            if order.status == OrderStatus.QUEUED
            and (account_id is None or order.account_id == account_id)
        ]
        oldest = min(
            (order.queued_at for order in queued if order.queued_at is not None),
            default=None,
        )
        now = datetime.now(timezone.utc)
        return {
            "queued_count": len(queued),
            "oldest_queued_at": oldest,
            "oldest_queued_age_seconds": (
                (now - oldest).total_seconds() if oldest else None
            ),
        }

    async def upsert_adapter_state(self, row: AdapterStateRow) -> None:
        self._adapter_state[row.adapter_id] = row

    async def record_adapter_disconnect(
        self,
        adapter_id: str,
        error_code: str = None,
        error_msg: str = None,
    ) -> None:
        row = self._adapter_state.get(adapter_id)
        if row is not None:
            self._adapter_state[adapter_id] = replace(
                row,
                connected=False,
                last_disconnect_ts=datetime.now(timezone.utc),
                disconnect_count_24h=row.disconnect_count_24h + 1,
                last_error_code=error_code or row.last_error_code,
                last_error_message=error_msg or row.last_error_message,
            )

    async def record_adapter_connect(self, adapter_id: str) -> None:
        row = self._adapter_state.get(adapter_id)
        if row is not None:
            self._adapter_state[adapter_id] = replace(
                row,
                connected=True,
                last_heartbeat_ts=datetime.now(timezone.utc),
            )

    @staticmethod
    def _working_statuses() -> set[OrderStatus]:
        return {
            OrderStatus.RISK_APPROVED,
            OrderStatus.QUEUED,
            OrderStatus.ROUTED,
            OrderStatus.ACKED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
        }

    @staticmethod
    def _direction_matches_qty(net_qty: float, direction: str | None) -> bool:
        if direction is None:
            return True
        direction_upper = direction.upper()
        if direction_upper == "LONG":
            return net_qty > 0
        return net_qty < 0

    @staticmethod
    def _direction_matches_side(side: OrderSide, direction: str | None) -> bool:
        if direction is None:
            return True
        return side == (OrderSide.BUY if direction.upper() == "LONG" else OrderSide.SELL)

    @staticmethod
    def _order_symbol(order: OMSOrder) -> str:
        return order.instrument.symbol if order.instrument else ""

    @staticmethod
    def _risk_context_value(risk_context: RiskContext | dict | None, key: str) -> float:
        if risk_context is None:
            return 0.0
        if isinstance(risk_context, dict):
            return float(risk_context.get(key, 0.0) or 0.0)
        return float(getattr(risk_context, key, 0.0) or 0.0)

    def _open_position_risk_R(
        self,
        *,
        direction: str | None = None,
        strategy_ids: set[str] | None = None,
    ) -> float:
        return sum(
            float(pos.open_risk_R or 0.0)
            for pos in self._positions.values()
            if pos.net_qty != 0
            and (strategy_ids is None or pos.strategy_id in strategy_ids)
            and self._direction_matches_qty(pos.net_qty, direction)
        )

    def _open_position_risk_dollars(
        self,
        *,
        direction: str | None = None,
        strategy_ids: set[str] | None = None,
    ) -> float:
        return sum(
            float(pos.open_risk_dollars or 0.0)
            for pos in self._positions.values()
            if pos.net_qty != 0
            and (strategy_ids is None or pos.strategy_id in strategy_ids)
            and self._direction_matches_qty(pos.net_qty, direction)
        )

    def _pending_entry_risk_dollars(
        self,
        *,
        direction: str | None = None,
        strategy_ids: set[str] | None = None,
    ) -> float:
        total = 0.0
        for order in self._orders.values():
            if order.role != OrderRole.ENTRY or order.status not in self._working_statuses():
                continue
            if strategy_ids is not None and order.strategy_id not in strategy_ids:
                continue
            if not self._direction_matches_side(order.side, direction):
                continue
            risk = self._risk_context_value(order.risk_context, "risk_dollars")
            if order.qty > 0 and order.remaining_qty > 0:
                risk *= order.remaining_qty / order.qty
            total += risk
        return total
